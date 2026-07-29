#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""容量模式(igpu_moe_capacity)的端到端正确性自测 —— 不需要 35B 模型。

验证核心等式在**真硬件、真跨进程**下成立::

    full(E 专家, 全在 dGPU)  ==  hot(dGPU 上 [0, E-K))  +  cold(iGPU sidecar 上 [E-K, E))

同时顺带压到:sidecar 的 subprocess 拉起 + ZE_AFFINITY_MASK 生效、共享内存协议、
分块 fp8 量化注册、logits 置换、prefill / decode 两种 M。

跑法::

    cd /llm/zhuyong/vllm
    PYTHONPATH=/llm/zhuyong/vllm python3 igpu_moe_offload/test_capacity_split.py
    # 只测 fp8 冷权重:  --quant fp8   (默认两种都测)
    # 换形状:           --experts 32 --hidden 256 --inter 64 --top-k 4 --cold-k 12
"""

from __future__ import annotations

import argparse
import sys

import torch

from vllm.model_executor.layers.fused_moe import igpu_moe_capacity as cap


def build_reference(ipex, w13, w2, E, start_id=0, scales=None):
    if scales is None:
        return ipex.llm.modules.GatedMLPMOE(
            w13, w2, use_prepack=True, experts_start_id=start_id)
    s13, s2 = scales
    return ipex.llm.modules.GatedMLPMOE(
        w13, w2, w1_scale_inv=s13, w2_scale_inv=s2,
        a1_scale_inv=None, a2_scale_inv=None,
        use_prepack=True, experts_start_id=start_id)


def call(mod, x, logits, top_k, renormalize):
    return mod(x, False, top_k, logits, renormalize, None, None, None,
               "softmax", "silu", None)


def run_case(args, quant: str) -> bool:
    import intel_extension_for_pytorch as ipex

    from vllm import _custom_ops as ops

    E, K = args.experts, args.cold_k
    H, I = args.hidden, args.inter
    w13_up = 2 * I
    hot_n = E - K
    dev = "xpu:0"          # 本进程钉在 dGPU(默认 device_count()==1 就是独显)
    dt = torch.float16

    torch.manual_seed(0)
    w13 = (torch.randn(E, w13_up, H, dtype=dt) * 0.05)
    w2 = (torch.randn(E, H, I, dtype=dt) * 0.05)

    print(f"\n=== case: cold_quant={quant}  E={E} K={K} hot={hot_n} "
          f"H={H} I={I} top_k={args.top_k} ===")

    # ---- 参照:全部 E 个专家都在 dGPU 上 ----------------------------------- #
    if quant == "fp8":
        # 与 XPUFp8MoEMethod 一致:每个专家一个 scale。热/冷两侧拿到的是同一个
        # 专家的同一份权重,所以 scale 也相同 —— 差异只来自两块硬件的 kernel 数值。
        qdt = torch.float8_e5m2 if torch.xpu.is_available() else torch.float8_e4m3fn
        from vllm.platforms import current_platform
        qdt = current_platform.fp8_dtype()
        w13_d = torch.empty((E, w13_up, H), dtype=qdt, device=dev)
        w2_d = torch.empty((E, H, I), dtype=qdt, device=dev)
        s13 = torch.empty((E,), dtype=torch.float32, device=dev)
        s2 = torch.empty((E,), dtype=torch.float32, device=dev)
        w13x, w2x = w13.to(dev), w2.to(dev)
        for e in range(E):
            w13_d[e], s13[e] = ops.scaled_fp8_quant(w13x[e])
            w2_d[e], s2[e] = ops.scaled_fp8_quant(w2x[e])
        del w13x, w2x
        full = build_reference(ipex, w13_d, w2_d, E, 0, (s13, s2))
        hot = build_reference(ipex, w13_d[:hot_n].clone(), w2_d[:hot_n].clone(),
                              hot_n, 0, (s13[:hot_n].clone(), s2[:hot_n].clone()))
    else:
        w13_d, w2_d = w13.to(dev), w2.to(dev)
        full = build_reference(ipex, w13_d, w2_d, E, 0)
        hot = build_reference(ipex, w13_d[:hot_n].clone().contiguous(),
                              w2_d[:hot_n].clone().contiguous(), hot_n, 0)

    # ---- 冷侧:真 sidecar ------------------------------------------------- #
    cfg = cap.ColdCfg(
        global_experts=E, cold_k=K, hidden=H, inter=I, w13_up=w13_up,
        num_layers=1, max_tokens=max(args.prefill_tokens, 8),
        top_k=args.top_k, renormalize=True, use_grouped_topk=False,
        topk_group=None, num_expert_group=None,
        scoring_func="softmax", activation="silu",
        param_dtype="float16", act_dtype="float16", logits_dtype="float16",
        cold_quant=quant, reg_chunk=4, use_perm=cap.capacity_use_perm(),
        mask=cap.capacity_mask(), debug=True,
    )
    sc = cap.ColdExpertSidecar(cfg)
    sc.spawn()
    ok = True
    try:
        # 模拟 weight_loader:把冷专家 [E-K, E) 逐个塞进暂存,再注册
        for gid in range(hot_n, E):
            cid = sc.cold_local_id(gid)
            sc.staging_w13(0, cid).copy_(w13[gid])
            sc.staging_w2(0, cid).copy_(w2[gid])
            done = sc.note_staged(0, n_w13=w13_up * H, n_w2=H * I)
        assert done, "staging accounting did not complete"
        sc.register_layer(0)
        sc.finish_registration()

        for M in (args.prefill_tokens, 1):
            x = (torch.randn(M, H, dtype=dt) * 0.5).to(dev)
            logits = (torch.randn(M, E, dtype=dt)).to(dev)

            ref = call(full, x, logits, args.top_k, True)
            seq = sc.send(0, x, logits)
            h = call(hot, x, logits, args.top_k, True)
            c = sc.wait(seq, h.device, h.dtype)
            got = h + c

            torch.xpu.synchronize()
            # 防假阳性:冷侧必须真的贡献了东西,否则 hot==full 也会"通过"
            cold_mag = c.float().abs().max().item()
            hot_only = (ref.float() - h.float()).abs().max().item()
            if cold_mag == 0.0 or hot_only == 0.0:
                print(f"  DEGENERATE: cold_max={cold_mag:.3e} "
                      f"|ref-hot|max={hot_only:.3e} — the cold half contributed "
                      f"nothing, this case proves nothing.")
                ok = False
            diff = (ref.float() - got.float()).abs()
            denom = ref.float().abs().max().clamp_min(1e-6)
            rel = (diff.max() / denom).item()
            # 少数 token 整行错(= 选错专家)和所有行都有小噪声(= 数值精度),
            # 是完全不同的两种病,必须区分开
            row_bad = (diff.max(dim=-1).values > 0.05 * denom)
            n_bad = int(row_bad.sum().item())
            tag = "PREFILL" if M > 1 else "DECODE "
            # 两半分别落在 B60 与核显上,kernel 数值不会逐位一致;fp8 更松一点
            tol = 5e-2 if quant == "fp8" else 2e-2
            good = rel <= tol
            ok &= good
            print(f"  {tag} M={M:<5d} max|diff|={diff.max().item():.3e}  "
                  f"rel={rel:.3e}  tol={tol:.0e}  bad_rows={n_bad}/{M}  "
                  f"{'OK' if good else 'FAIL'}")
        if args.bench:
            import time as _t

            for M in (args.prefill_tokens, 1):
                x = (torch.randn(M, H, dtype=dt) * 0.5).to(dev)
                logits = torch.randn(M, E, dtype=dt).to(dev)
                for _ in range(3):                       # warmup
                    s = sc.send(0, x, logits)
                    sc.wait(s, dev, dt)
                n = 20 if M == 1 else 5
                # 冷侧单独计时(iGPU 分支的关键路径)
                torch.xpu.synchronize()
                t0 = _t.perf_counter()
                for _ in range(n):
                    s = sc.send(0, x, logits)
                    sc.wait(s, dev, dt)
                cold_us = (_t.perf_counter() - t0) / n * 1e6
                # overlap 后的每层墙钟
                torch.xpu.synchronize()
                t0 = _t.perf_counter()
                for _ in range(n):
                    s = sc.send(0, x, logits)
                    h = call(hot, x, logits, args.top_k, True)
                    _ = h + sc.wait(s, h.device, h.dtype)
                torch.xpu.synchronize()
                rt_us = (_t.perf_counter() - t0) / n * 1e6
                # dGPU 全量基线(等价于不卸载)
                torch.xpu.synchronize()
                t0 = _t.perf_counter()
                for _ in range(n):
                    _ = call(full, x, logits, args.top_k, True)
                torch.xpu.synchronize()
                base_us = (_t.perf_counter() - t0) / n * 1e6
                tag = "PREFILL" if M > 1 else "DECODE "
                print(f"  {tag} M={M:<5d} cold_branch={cold_us:8.1f}us  "
                      f"layer_RT={rt_us:8.1f}us  dGPU_full={base_us:8.1f}us  "
                      f"→ {rt_us / base_us:.2f}x of no-offload")
    finally:
        sc.shutdown()
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--cold-k", type=int, default=12)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--inter", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--prefill-tokens", type=int, default=256)
    p.add_argument("--quant", choices=["fp8", "none", "both"], default="both")
    p.add_argument("--bench", action="store_true")
    args = p.parse_args()

    if not torch.xpu.is_available():
        print("no XPU visible; nothing to test", file=sys.stderr)
        return 2
    print(f"dGPU side: {torch.xpu.get_device_name(0)} "
          f"(device_count={torch.xpu.device_count()})")
    print(f"sidecar mask: ZE_AFFINITY_MASK={cap.capacity_mask()}")

    quants = ["none", "fp8"] if args.quant == "both" else [args.quant]
    ok = True
    for q in quants:
        ok &= run_case(args, q)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
