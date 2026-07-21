#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""最小传输基准 —— 判定 iGPU MoE 卸载落在 +7% / +15% 哪个区间。

量三件事:
  A(dGPU, mask=0):pageable vs pinned 的 D2H/H2D 真实带宽;D2H(copy engine)能否与
    dGPU 计算 overlap。
  B(iGPU, mask=1):把 x[T,H] 从 host 取进 iGPU 可用 xpu tensor 的 ingest 成本
    (随 K 固定的开销,决定 iGPU_side 的 fixed 部分)。

用法:
  ZE_AFFINITY_MASK=0 python transport_bench.py dgpu
  ZE_AFFINITY_MASK=1 python transport_bench.py igpu
"""
import os
import sys
import time

MASK = "0" if (len(sys.argv) > 1 and sys.argv[1] == "dgpu") else "1"
os.environ.setdefault("ZE_AFFINITY_MASK", MASK)

import torch  # noqa: E402
import intel_extension_for_pytorch as ipex  # noqa: E402,F401

dev = "xpu:0"
T, H = 2048, 2048
NBYTES = T * H * 2  # fp16, 8 MiB


def gbps(nbytes, us):
    return nbytes / (us * 1e-6) / 1e9


def wall(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        torch.xpu.synchronize()
        ts.append(time.perf_counter_ns() - t0)
    ts.sort()
    return ts[len(ts) // 2] / 1e3  # p50 us


def bench_dgpu():
    print(f"=== dGPU (mask={os.environ['ZE_AFFINITY_MASK']}, "
          f"{torch.xpu.get_device_name(0)}) x[{T},{H}]=8MiB ===")
    x = torch.randn(T, H, device=dev, dtype=torch.float16)
    pageable = torch.empty(T, H, dtype=torch.float16)
    pinned = torch.empty(T, H, dtype=torch.float16, pin_memory=True)
    xdev = torch.empty(T, H, device=dev, dtype=torch.float16)

    d2h_page = wall(lambda: pageable.copy_(x))
    d2h_pin = wall(lambda: pinned.copy_(x, non_blocking=True))
    h2d_page = wall(lambda: xdev.copy_(pageable))
    h2d_pin = wall(lambda: xdev.copy_(pinned, non_blocking=True))
    print(f"D2H pageable : {d2h_page:7.0f}us  {gbps(NBYTES,d2h_page):5.1f} GB/s")
    print(f"D2H pinned   : {d2h_pin:7.0f}us  {gbps(NBYTES,d2h_pin):5.1f} GB/s")
    print(f"H2D pageable : {h2d_page:7.0f}us  {gbps(NBYTES,h2d_page):5.1f} GB/s")
    print(f"H2D pinned   : {h2d_pin:7.0f}us  {gbps(NBYTES,h2d_pin):5.1f} GB/s")

    # overlap:side stream 上做 D2H(copy engine),default stream 跑 matmul。
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    side = torch.xpu.Stream()

    def compute_only():
        for _ in range(8):
            a.mm(a)

    def copy_only():
        pinned.copy_(x, non_blocking=True)

    def overlapped():
        with torch.xpu.stream(side):
            pinned.copy_(x, non_blocking=True)
        for _ in range(8):
            a.mm(a)

    c = wall(compute_only)
    cp = wall(copy_only)
    ov = wall(overlapped)
    print(f"\noverlap 测试(8×4096³ matmul ‖ D2H):")
    print(f"  compute only : {c:7.0f}us")
    print(f"  copy only    : {cp:7.0f}us")
    print(f"  overlapped   : {ov:7.0f}us   (串行应≈{c+cp:.0f}, "
          f"完美 overlap≈{max(c,cp):.0f})")
    hidden = (c + cp - ov)
    print(f"  → 被 overlap 掩盖 ≈ {hidden:.0f}us "
          f"({100*hidden/min(c,cp):.0f}% of 较小者)")


def bench_igpu():
    print(f"=== iGPU (mask={os.environ['ZE_AFFINITY_MASK']}, "
          f"{torch.xpu.get_device_name(0)}) x[{T},{H}]=8MiB ===")
    import numpy as np
    from multiprocessing import shared_memory
    # 模拟 sidecar:x 已在 host 共享内存里(dGPU D2H 写好),量 iGPU 取进 xpu 的成本。
    shm = shared_memory.SharedMemory(create=True, size=NBYTES)
    host_np = np.ndarray((T, H), dtype=np.float16, buffer=shm.buf)
    host_np[:] = np.random.randn(T, H).astype(np.float16)
    pinned = torch.empty(T, H, dtype=torch.float16, pin_memory=True)

    # 路径1:numpy(pageable shm)-> xpu(现集成代码走这条)
    ingest_np = wall(lambda: torch.from_numpy(host_np).to(dev, non_blocking=True))
    # 路径2:先进 pinned 再 -> xpu
    def via_pinned():
        pinned.copy_(torch.from_numpy(host_np))
        pinned.to(dev, non_blocking=True)
    ingest_pin = wall(via_pinned)
    # egress:iGPU 结果 -> host
    r = torch.randn(T, H, device=dev, dtype=torch.float16)
    egress_np = wall(lambda: r.to("cpu"))
    egress_pin = wall(lambda: pinned.copy_(r, non_blocking=True))
    print(f"ingest numpy-shm->xpu : {ingest_np:7.0f}us  {gbps(NBYTES,ingest_np):5.1f} GB/s")
    print(f"ingest via pinned     : {ingest_pin:7.0f}us  {gbps(NBYTES,ingest_pin):5.1f} GB/s")
    print(f"egress xpu->host np    : {egress_np:7.0f}us  {gbps(NBYTES,egress_np):5.1f} GB/s")
    print(f"egress xpu->pinned     : {egress_pin:7.0f}us  {gbps(NBYTES,egress_pin):5.1f} GB/s")
    shm.close(); shm.unlink()


if __name__ == "__main__":
    if MASK == "0":
        bench_dgpu()
    else:
        bench_igpu()
