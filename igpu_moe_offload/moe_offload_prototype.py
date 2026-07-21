#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
纯 vLLM(纯 Python / torch-xpu / IPEX)冷专家 iGPU 卸载原型 —— prefill overlap 验证。

背景结论(见 STATUS.md §7):
  - decode(batch=1)是 LPDDR5 带宽 bound,卸载 no-go(batching 也救不了)。
  - prefill 计算 bound(iGPU XMX ~17 TFLOP/s)→ 卸载有价值。

本原型验证的核心问题(prefill):
  把一部分冷专家发 iGPU 计算,与 dGPU 自身工作(attention + 热专家 + 其余冷专家)
  **真正并行**,每层总时间能否 < dGPU 全干(baseline)?最优卸载专家数 K* 是多少?

显存模型:冷专家权重 dGPU+iGPU **双份**。prefill 卸载一部分冷专家到 iGPU(提速);
decode 冷专家仍在 dGPU(不卸载)。纯 prefill 提速,不省显存。

架构:main 编排;dGPU/iGPU 各 spawn 一个子进程,import torch 前设 ZE_AFFINITY_MASK。
  控制块 ctrl(SharedMemory, int64[8]):
    [0] req_seq       dGPU 递增发起一次请求
    [1] resp_seq      iGPU 算完置为 req_seq
    [2] n_rows        本次请求每专家处理的行数 M(= tokens_per_expert)
    [3] igpu_ns       iGPU 上报的全程 wall(host<->dev copy + compute),ns
    [4] stop          1 = 退出
    [5] n_exp         本次请求卸载到 iGPU 的冷专家数 K(每请求可变 → 单进程跑完整扫描)
    [6] igpu_cmp_ns   iGPU 纯计算(xpu.Event 设备时间线),ns
  数据块:in_buf(dGPU 写 / iGPU 读)、out_buf(iGPU 写 / dGPU 读),fp16,M×H。

测量方法学(正确性是重点,修掉旧版 busy-loop+synchronize 的队列 drain artifact):
  - dGPU 侧:提交**固定 kernel 数量**的工作,用 xpu.Event 在设备时间线计时,
    只在最后 synchronize 一次。时序:先发 REQ + 提交 dGPU 工作(async),再自旋等
    RESP → 两设备硬件真并行;RT = max(dGPU own, iGPU 往返)。
  - baseline = 扫描里 K=0 那点(dGPU 全干,iGPU 空闲,无搬运),同一次运行 apples-to-apples。

注意:torch 只在 worker 内部 import(spawn 会重导入本模块)。
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

CTRL_SLOTS = 8  # int64
IDX_REQ, IDX_RESP, IDX_NROW, IDX_IGPU_NS, IDX_STOP = 0, 1, 2, 3, 4
IDX_NEXP, IDX_IGPU_CMP_NS = 5, 6  # 每请求卸载专家数 K;iGPU 纯计算 ns

HEAD_DIM = 128  # attention proxy 的 head_dim


@dataclass
class Cfg:
    hidden: int          # H
    inter: int           # I (moe_intermediate_size)
    cold_experts: int    # iGPU 侧常驻的冷专家权重份数(池大小)
    hot_experts: int     # 每层恒在 dGPU 的热专家数
    cold_active: int     # 每层 active 的冷专家数(其中 K 个卸载到 iGPU,余下留 dGPU)
    top_k: int           # MoE 路由 top-k(决定 tokens_per_expert)
    attn_scale: float    # dGPU attention proxy 重复次数(灵敏度旋钮)
    prefill_tokens: int  # T
    decode_tokens: int
    mode: str            # "compute" | "echo"(echo 只回传,隔离纯搬运)
    batched: bool        # True = bmm 批处理专家(少 launch);False = 逐专家循环
    sweep: tuple         # 要扫描的卸载专家数 K 列表
    ctrl_name: str
    in_name: str
    out_name: str
    max_rows: int        # in/out buffer 的行数上限


def _spin_open(name, retries=2000):
    last = None
    for _ in range(retries):
        try:
            return shared_memory.SharedMemory(name=name)
        except FileNotFoundError as e:
            last = e
            time.sleep(0.005)
    raise last


def _tokens_per_expert(T, top_k, n_active_experts):
    """prefill 时 token 按 top_k 路由分摊到各 active 专家的平均行数。"""
    return max(1, math.ceil(T * top_k / max(1, n_active_experts)))


# --------------------------------------------------------------------------- #
# iGPU worker: recv M 行激活 -> xpu -> K 个冷专家 FFN(over M 行)-> host -> 回传
# --------------------------------------------------------------------------- #
def igpu_worker(cfg: Cfg):
    os.environ["ZE_AFFINITY_MASK"] = "1"  # 必须在 import torch 之前
    import torch
    import intel_extension_for_pytorch as ipex  # noqa: F401

    dev = "xpu:0"  # 本进程里 xpu:0 == iGPU
    torch.manual_seed(0)
    H, I, E = cfg.hidden, cfg.inter, cfg.cold_experts
    # 每个冷专家一份权重,常驻 iGPU(系统内存)。w13 融合 gate+up。
    w13 = [torch.randn(H, 2 * I, device=dev, dtype=torch.float16) * 0.02 for _ in range(E)]
    w2 = [torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02 for _ in range(E)]
    if cfg.batched:
        # 批处理:堆成 [E,H,2I] / [E,I,H],运行时切前 K 份用 bmm 一次算完。
        W13 = torch.stack(w13).contiguous()
        W2 = torch.stack(w2).contiguous()
    torch.xpu.synchronize()

    ctrl = _spin_open(cfg.ctrl_name)
    in_shm = _spin_open(cfg.in_name)
    out_shm = _spin_open(cfg.out_name)
    c = np.ndarray((CTRL_SLOTS,), dtype=np.int64, buffer=ctrl.buf)
    in_np = np.ndarray((cfg.max_rows, H), dtype=np.float16, buffer=in_shm.buf)
    out_np = np.ndarray((cfg.max_rows, H), dtype=np.float16, buffer=out_shm.buf)

    silu = torch.nn.functional.silu
    cmp0 = torch.xpu.Event(enable_timing=True)
    cmp1 = torch.xpu.Event(enable_timing=True)

    last = 0
    while True:
        while c[IDX_REQ] == last and c[IDX_STOP] == 0:
            pass
        if c[IDX_STOP] != 0:
            break
        seq = int(c[IDX_REQ])
        M = int(c[IDX_NROW])
        K = int(c[IDX_NEXP])

        t0 = time.perf_counter_ns()
        # host -> iGPU(iGPU 用系统内存,这一跳很便宜)
        x = torch.from_numpy(in_np[:M]).to(dev, non_blocking=True)
        cmp0.record()
        if cfg.mode == "compute" and K > 0:
            if cfg.batched:
                # 同一 M 行喂给 K 个专家(uniform 近似;FLOP 与真实一致)。
                xb = x.unsqueeze(0).expand(K, M, H)          # [K,M,H]
                h = torch.bmm(xb, W13[:K])                    # [K,M,2I]
                g, u = h.chunk(2, dim=-1)
                out = torch.bmm(silu(g) * u, W2[:K]).sum(0)   # [M,H]
            else:
                out = torch.zeros(M, H, device=dev, dtype=torch.float16)
                for e in range(K):
                    h = x @ w13[e]
                    g, u = h.chunk(2, dim=-1)
                    out += (silu(g) * u) @ w2[e]
        else:
            out = x
        cmp1.record()
        res = out.to("cpu")  # iGPU -> host
        torch.xpu.synchronize()

        out_np[:M] = res.numpy()
        c[IDX_IGPU_NS] = time.perf_counter_ns() - t0
        c[IDX_IGPU_CMP_NS] = int(cmp0.elapsed_time(cmp1) * 1e6)  # ms -> ns
        c[IDX_RESP] = seq  # 发布结果
        last = seq

    for s in (ctrl, in_shm, out_shm):
        s.close()


# --------------------------------------------------------------------------- #
# dGPU worker: 发 K 个冷专家的激活 -> 并行做自己的活(attn+hot+余下冷)-> 等结果
# --------------------------------------------------------------------------- #
def dgpu_worker(cfg: Cfg):
    os.environ["ZE_AFFINITY_MASK"] = "0"
    import torch
    import intel_extension_for_pytorch as ipex  # noqa: F401

    dev = "xpu:0"  # 本进程里 xpu:0 == dGPU
    torch.manual_seed(1)
    H, I = cfg.hidden, cfg.inter
    silu = torch.nn.functional.silu

    # --- dGPU 常驻权重(双份显存模型)---
    # attention proxy 权重(全程 T 行,不路由)
    Wqkv = torch.randn(H, 3 * H, device=dev, dtype=torch.float16) * 0.02
    Wo = torch.randn(H, H, device=dev, dtype=torch.float16) * 0.02
    n_heads = max(1, H // HEAD_DIM)
    # 热专家 + cold_active 份冷专家(只按需分配,不是全部 96)
    n_dgpu_ffn = cfg.hot_experts + cfg.cold_active
    dw13 = [torch.randn(H, 2 * I, device=dev, dtype=torch.float16) * 0.02 for _ in range(n_dgpu_ffn)]
    dw2 = [torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02 for _ in range(n_dgpu_ffn)]
    torch.xpu.synchronize()

    ctrl = _spin_open(cfg.ctrl_name)
    in_shm = _spin_open(cfg.in_name)
    out_shm = _spin_open(cfg.out_name)
    c = np.ndarray((CTRL_SLOTS,), dtype=np.int64, buffer=ctrl.buf)
    in_np = np.ndarray((cfg.max_rows, H), dtype=np.float16, buffer=in_shm.buf)
    out_np = np.ndarray((cfg.max_rows, H), dtype=np.float16, buffer=out_shm.buf)

    def ffn(x, w13, w2):
        h = x @ w13
        g, u = h.chunk(2, dim=-1)
        return (silu(g) * u) @ w2

    def attention_proxy(x_attn):
        # 真实 prefill attention 形状:qkv 投影 + 逐头 scores/softmax/ctx + 输出投影。
        T = x_attn.shape[0]
        qkv = x_attn @ Wqkv
        q, k, v = qkv.split(H, dim=-1)
        q = q.view(T, n_heads, HEAD_DIM).transpose(0, 1)  # [nh,T,hd]
        k = k.view(T, n_heads, HEAD_DIM).transpose(0, 1)
        v = v.view(T, n_heads, HEAD_DIM).transpose(0, 1)
        scores = torch.bmm(q, k.transpose(-1, -2)) * (HEAD_DIM ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, v).transpose(0, 1).reshape(T, H)
        return ctx @ Wo

    def dgpu_own_work(K, M, T, x_attn, x_ffn):
        """提交固定量的 dGPU 工作(async):attention + hot FFN + 余下冷 FFN。"""
        n_reps = max(0, round(cfg.attn_scale))
        acc = None
        for _ in range(n_reps):
            acc = attention_proxy(x_attn)
        # 每专家处理 M 行(uniform 分摊)
        n_local_ffn = cfg.hot_experts + (cfg.cold_active - K)
        out = torch.zeros(M, H, device=dev, dtype=torch.float16)
        for e in range(n_local_ffn):
            out = out + ffn(x_ffn, dw13[e], dw2[e])
        return out, acc

    ev0 = torch.xpu.Event(enable_timing=True)
    ev1 = torch.xpu.Event(enable_timing=True)

    def one_round(K, M, T, seq):
        # 送给 iGPU 的激活(M 行);dGPU 自己的 attention 用 T 行,本地生成不走 shm。
        host_x = torch.randn(M, H, dtype=torch.float16).numpy()
        in_np[:M] = host_x
        x_ffn = torch.from_numpy(host_x).to(dev, non_blocking=True)
        x_attn = torch.randn(T, H, device=dev, dtype=torch.float16)

        c[IDX_NROW] = M
        c[IDX_NEXP] = K
        t0 = time.perf_counter_ns()
        if K > 0:
            c[IDX_REQ] = seq  # 先踢 iGPU,使其与 dGPU 硬件并行
        ev0.record()
        _out, _acc = dgpu_own_work(K, M, T, x_attn, x_ffn)  # async 提交固定量工作
        ev1.record()
        if K > 0:
            while c[IDX_RESP] != seq and c[IDX_STOP] == 0:  # CPU 自旋等 iGPU
                pass
        torch.xpu.synchronize()  # 保证 dGPU own-work 与结果都完成
        rt_ns = time.perf_counter_ns() - t0
        dgpu_own_ns = int(ev0.elapsed_time(ev1) * 1e6)  # ms -> ns
        igpu_ns = int(c[IDX_IGPU_NS]) if K > 0 else 0
        igpu_cmp_ns = int(c[IDX_IGPU_CMP_NS]) if K > 0 else 0
        if K > 0:
            _ = out_np[:M]  # 结果已就绪(真实里 .to(dev) 合并)
        return rt_ns, dgpu_own_ns, igpu_ns, igpu_cmp_ns

    def bench_k(K, M, T, warmup=5, n=50):
        seq = int(c[IDX_REQ])
        for _ in range(warmup):
            seq += 1
            one_round(K, M, T, seq)
        rt, dg, ig, igc = [], [], [], []
        for _ in range(n):
            seq += 1
            r, d, i, ic = one_round(K, M, T, seq)
            rt.append(r); dg.append(d); ig.append(i); igc.append(ic)
        p50 = lambda a: sorted(a)[len(a) // 2] / 1e3  # ns -> us
        return p50(rt), p50(dg), p50(ig), p50(igc)

    print(f"=== dGPU worker up (dev={torch.xpu.get_device_name(0)}) ===", flush=True)

    if cfg.mode == "echo":
        # 纯搬运基线(K=cold_active,M=prefill 分摊行数)
        T = cfg.prefill_tokens
        M = _tokens_per_expert(T, cfg.top_k, cfg.hot_experts + cfg.cold_active)
        rt, dg, ig, igc = bench_k(cfg.cold_active, M, T)
        print(f"[ECHO] T={T} M/expert={M} K={cfg.cold_active}", flush=True)
        print(f"  往返 RT(p50)={rt:8.1f}us  搬运={ig - igc:6.1f}us", flush=True)
    else:
        T = cfg.prefill_tokens
        M = _tokens_per_expert(T, cfg.top_k, cfg.hot_experts + cfg.cold_active)
        print(f"\n[SWEEP] PREFILL T={T} tokens_per_expert={M} "
              f"(top_k={cfg.top_k}, hot={cfg.hot_experts}, cold_active={cfg.cold_active})",
              flush=True)
        print(f"  {'K':>3} {'dGPU_own':>10} {'iGPU_cmp':>10} {'iGPU_side':>10} "
              f"{'transport':>10} {'overlap_RT':>11}", flush=True)
        rows = {}
        for K in cfg.sweep:
            if K > cfg.cold_active:
                continue
            rt, dg, ig, igc = bench_k(K, M, T)
            transport = (ig - igc) if K > 0 else 0.0
            rows[K] = rt
            tag = "  <- baseline" if K == 0 else ""
            print(f"  {K:>3} {dg:9.1f}u {igc:9.1f}u {ig:9.1f}u "
                  f"{transport:9.1f}u {rt:10.1f}u{tag}", flush=True)

        if 0 in rows:
            base = rows[0]
            kstar = min((k for k in rows if k > 0), key=lambda k: rows[k], default=None)
            if kstar is not None:
                sp = base / rows[kstar]
                print(f"\n[RESULT] baseline={base:.1f}us  optimal K*={kstar} "
                      f"(offload frac {kstar}/{cfg.cold_active}={kstar / cfg.cold_active:.2f})",
                      flush=True)
                print(f"         overlap={rows[kstar]:.1f}us  "
                      f"speedup={base:.1f}/{rows[kstar]:.1f} = {sp:.2f}x ({(sp - 1) * 100:+.0f}%)",
                      flush=True)
        print("  注:M=tokens_per_expert 为 uniform 分摊,忽略路由负载不均"
              "(尾专家更重)→ 可能略高估加速。", flush=True)

        # DECODE 参考(卸载 no-go,仅对照):T=1,K=cold_active,M=1
        rt, dg, ig, igc = bench_k(cfg.cold_active, 1, 1)
        print(f"\n[DECODE (offload no-go, reference only)] T=1 K={cfg.cold_active}  "
              f"iGPU_cmp={igc:.1f}us  RT={rt:.1f}us", flush=True)

    c[IDX_STOP] = 1
    for s in (ctrl, in_shm, out_shm):
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768)
    ap.add_argument("--cold-experts", type=int, default=96, help="iGPU 常驻冷专家池大小")
    ap.add_argument("--hot-experts", type=int, default=4, help="恒在 dGPU 的热专家数")
    ap.add_argument("--cold-active", type=int, default=12, help="每层 active 冷专家数(可卸载池)")
    ap.add_argument("--offload-experts", type=int, default=2, help="单点卸载数(无 --sweep 时用)")
    ap.add_argument("--sweep-offload", type=str, default="", help='如 "0,1,2,3,4,6,8,12"')
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--attn-scale", type=float, default=1.0, help="attention proxy 重复次数")
    ap.add_argument("--prefill-tokens", type=int, default=2048)
    ap.add_argument("--decode-tokens", type=int, default=1)
    ap.add_argument("--mode", choices=["compute", "echo"], default="compute")
    ap.add_argument("--batched", action="store_true", help="bmm 批处理专家(少 kernel launch)")
    args = ap.parse_args()

    if args.sweep_offload.strip():
        sweep = tuple(int(x) for x in args.sweep_offload.split(",") if x.strip() != "")
    else:
        sweep = (0, args.offload_experts)  # 至少含 baseline(0) 和单点

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # in/out buffer 按 prefill 的每专家分摊行数 M 上限分配(远小于 T)
    M_pref = _tokens_per_expert(args.prefill_tokens, args.top_k,
                                args.hot_experts + args.cold_active)
    max_rows = max(M_pref, args.decode_tokens, 1)
    nbytes = max_rows * args.hidden * 2  # fp16
    ctrl = shared_memory.SharedMemory(create=True, size=CTRL_SLOTS * 8)
    in_shm = shared_memory.SharedMemory(create=True, size=nbytes)
    out_shm = shared_memory.SharedMemory(create=True, size=nbytes)
    np.ndarray((CTRL_SLOTS,), dtype=np.int64, buffer=ctrl.buf)[:] = 0

    cfg = Cfg(
        hidden=args.hidden, inter=args.inter, cold_experts=args.cold_experts,
        hot_experts=args.hot_experts, cold_active=args.cold_active, top_k=args.top_k,
        attn_scale=args.attn_scale, prefill_tokens=args.prefill_tokens,
        decode_tokens=args.decode_tokens, mode=args.mode, batched=args.batched,
        sweep=sweep, ctrl_name=ctrl.name, in_name=in_shm.name, out_name=out_shm.name,
        max_rows=max_rows,
    )

    ig = mp.Process(target=igpu_worker, args=(cfg,), name="igpu")
    dg = mp.Process(target=dgpu_worker, args=(cfg,), name="dgpu")
    ig.start(); dg.start()
    dg.join()
    ig.join(timeout=10)
    if ig.is_alive():
        ig.terminate()
    for s in (ctrl, in_shm, out_shm):
        s.close(); s.unlink()


if __name__ == "__main__":
    main()
