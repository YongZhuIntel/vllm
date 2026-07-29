#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""容量模式的 **vLLM 集成路径** 自测 —— 建真的 ``FusedMoE`` 层,不用 35B 权重。

test_capacity_split.py 只测了 sidecar 本身;这个脚本测的是改在 vLLM 里的那部分:

1. ``FusedMoE.__init__`` 在容量模式下把 ``local_num_experts`` 改成 E-K、
   ``_expert_map`` 把 [E-K, E) 置 -1  → **独显上只分配 E-K 份权重**(直接查
   ``w13_weight.shape[0]``)。
2. ``FusedMoE.weight_loader`` 把 -1 的那些分流进 sidecar 暂存,一层齐了自动注册。
3. ``quant_method.apply`` 走 ``apply_with_cold``,prefill / decode 都 = hot + cold。

参照值:同一份权重全量喂给一个普通(容量模式关掉的)FusedMoE。

跑法::

    cd /llm/zhuyong/vllm
    PYTHONPATH=/llm/zhuyong/vllm python3 igpu_moe_offload/test_capacity_layer.py
    PYTHONPATH=/llm/zhuyong/vllm python3 igpu_moe_offload/test_capacity_layer.py --experts 256 \
        --hidden 2048 --inter 512 --top-k 8 --cold-frac 0.45 --tokens 512
"""

from __future__ import annotations

import argparse
import os
import sys


def build_layer(E, H, I, top_k, dtype, prefix, quant_config=None):
    """建一个 FusedMoE 并把权重按 vLLM 的 weight_loader 协议喂进去。"""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    layer = FusedMoE(
        num_experts=E,
        top_k=top_k,
        hidden_size=H,
        intermediate_size=I,
        params_dtype=dtype,
        reduce_results=False,
        renormalize=True,
        quant_config=quant_config,
        prefix=prefix,
    )
    return layer


def load_weights(layer, w13, w2, E):
    """按 (w1, w3, w2) 三个 shard 逐专家喂,和真 checkpoint 的调用方式一致。

    weight_name 必须是真实的参数名 —— ``FusedMoE.weight_loader`` 最后是靠
    ``"weight" in weight_name`` 才走到 ``_load_model_weight_or_group_weight_scale``
    的,随便传个 "w1" 会静默 return False 什么都不写。
    """
    I = w2.shape[2]
    n_ok = 0
    # param 和 loader 都只解析一次,和真实加载路径一致:
    # ``Qwen3_5Model.load_fused_expert_weights`` 是 ``param = params_dict[name]``
    # + ``param.weight_loader`` 取一次,然后对 E 个专家复用同一个可调用对象。
    # 每个 shard 重新 getattr 会得到不同的东西 —— fp8 streaming 路径在
    # materialize 时用**原始** loader 重新 register_parameter 了 w13_weight,
    # 于是第二个 shard 起就绕过 patched_weight_loader,整个流式量化 +
    # 冷热交错的逻辑一次都测不到(2026-07-29 的 e2e 崩溃就是这么漏掉的)。
    bound = {}
    for name in ("w13_weight", "w2_weight"):
        param = getattr(layer, name)
        bound[name] = (param,
                       getattr(param, "weight_loader", layer.weight_loader))
    for e in range(E):
        for src, shard, name in (
            (w13[e, :I], "w1", "w13_weight"),
            (w13[e, I:], "w3", "w13_weight"),
            (w2[e], "w2", "w2_weight"),
        ):
            param, loader = bound[name]
            ok = loader(param=param, loaded_weight=src, weight_name=name,
                        shard_id=shard, expert_id=e, return_success=True)
            n_ok += bool(ok)
    expected = 3 * E
    assert n_ok == expected, (
        f"weight_loader accepted {n_ok}/{expected} shards — the harness is "
        "feeding it wrong, the test would be meaningless")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--inter", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--cold-frac", type=float, default=0.45)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--fp8", action="store_true",
                   help="走 XPUFp8MoEMethod(在线量化)路径,即真实部署用的那条")
    args = p.parse_args()

    E, H, I, top_k = args.experts, args.hidden, args.inter, args.top_k

    # 必须在 import vllm 之前设好:FusedMoE.__init__ 读的是这些
    os.environ["VLLM_XPU_IGPU_MOE_COLD_FRAC"] = str(args.cold_frac)
    os.environ["VLLM_XPU_IGPU_MOE_COLD_QUANT"] = "fp8" if args.fp8 else "none"
    # 与部署命令一致:走 meta-device streaming 量化(带 CopyNumelCounter 的那条)。
    # 必须硬设 —— =1 那条 legacy 路径把权重建在 CPU 上,ipex 的 fp8 量化算子只有
    # XPU 实现,会直接报 "Could not run ... with arguments from the 'CPU' backend"。
    # (这是 fork 里已有的问题,与容量模式无关。)
    os.environ["VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT"] = "0"
    os.environ["VLLM_XPU_IGPU_MOE_DEBUG"] = "1"
    os.environ.setdefault("VLLM_XPU_IGPU_MOE_MASK", "1")

    import torch

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if not torch.xpu.is_available():
        print("no XPU visible; nothing to test", file=sys.stderr)
        return 2

    torch.set_default_device("xpu")
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29591", backend="gloo")
    initialize_model_parallel(1, 1)

    dt = torch.float16
    torch.manual_seed(0)
    w13 = (torch.randn(E, 2 * I, H, dtype=dt, device="cpu") * 0.05)
    w2 = (torch.randn(E, H, I, dtype=dt, device="cpu") * 0.05)

    vc = VllmConfig()
    vc.scheduler_config.max_num_batched_tokens = max(args.tokens, 64)

    qcfg = None
    if args.fp8:
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config
        qcfg = Fp8Config(is_checkpoint_fp8_serialized=False,
                         activation_scheme="dynamic")
    tol = 6e-2 if args.fp8 else 2e-2

    # ---------- 参照:容量模式关闭,E 个专家全在独显 ---------- #
    os.environ["VLLM_XPU_IGPU_MOE_CAPACITY"] = "0"
    with set_current_vllm_config(vc):
        ref_layer = build_layer(E, H, I, top_k, dt, "ref.experts", qcfg)
    load_weights(ref_layer, w13, w2, E)
    ref_layer.quant_method.process_weights_after_loading(ref_layer)
    print(f"[ref]  quant={type(ref_layer.quant_method).__name__} "
          f"local_num_experts={ref_layer.local_num_experts} "
          f"w13_weight={tuple(ref_layer.w13_weight.shape)}")

    # ---------- 被测:容量模式打开 ---------- #
    os.environ["VLLM_XPU_IGPU_MOE_CAPACITY"] = "1"
    from vllm.model_executor.layers.fused_moe import igpu_moe_capacity as cap

    cap.reset_layer_ids()
    with set_current_vllm_config(vc):
        cap_layer = build_layer(E, H, I, top_k, dt, "cap.experts", qcfg)
    K = cap_layer._igpu_cold_k
    hot_n = E - K
    print(f"[cap]  cold_k={K} local_num_experts={cap_layer.local_num_experts} "
          f"w13_weight={tuple(cap_layer.w13_weight.shape)} "
          f"lid={cap_layer._igpu_cold_lid}")

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'}  {msg}")
        ok &= bool(cond)

    check(K > 0, f"cold_k resolved to {K} (>0)")
    check(cap_layer.local_num_experts == hot_n,
          f"local_num_experts == E-K ({cap_layer.local_num_experts} == {hot_n})")
    check(cap_layer.w13_weight.shape[0] == hot_n,
          f"dGPU w13_weight has only E-K experts "
          f"({cap_layer.w13_weight.shape[0]} == {hot_n})")
    check(cap_layer.w2_weight.shape[0] == hot_n,
          f"dGPU w2_weight has only E-K experts "
          f"({cap_layer.w2_weight.shape[0]} == {hot_n})")
    vram_saved = (E - hot_n) / E
    check(True, f"→ {vram_saved:.0%} of this layer's expert VRAM never allocated")

    try:
        load_weights(cap_layer, w13, w2, E)
        sc = cap.ColdExpertSidecar.instance()
        check(sc is not None and sc.is_registered(cap_layer._igpu_cold_lid),
              "cold experts auto-registered when the layer finished loading")
        cap_layer.quant_method.process_weights_after_loading(cap_layer)

        for M in (args.tokens, 1):
            x = (torch.randn(M, H, dtype=dt) * 0.5).to("xpu")
            logits = torch.randn(M, E, dtype=dt).to("xpu")
            ref = ref_layer.quant_method.apply(
                ref_layer, ref_layer.router, x, logits)
            got = cap_layer.quant_method.apply(
                cap_layer, cap_layer.router, x, logits)
            torch.xpu.synchronize()
            d = (ref.float() - got.float()).abs()
            denom = ref.float().abs().max().clamp_min(1e-6)
            rel = (d.max() / denom).item()
            bad = int((d.max(dim=-1).values > 0.05 * denom).sum().item())
            tag = "PREFILL" if M > 1 else "DECODE "
            check(rel <= tol and bad == 0,
                  f"{tag} M={M:<5d} rel={rel:.3e} bad_rows={bad}/{M}")
    finally:
        sc = cap.ColdExpertSidecar.instance()
        if sc is not None:
            sc.shutdown()

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
