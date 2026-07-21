# SPDX-License-Identifier: Apache-2.0
"""prefill 冷专家 iGPU 卸载 —— sidecar 进程 + dGPU 侧客户端。

背景与结论见 /llm/vllm/igpu_moe_offload/STATUS.md、OFFLOAD_TUNING.md。

核心思想(已实证):ipex ``GatedMLPMOE(experts_start_id=0)`` 做全局 top_k + 全局
renorm,只计算 ``id < n_local``(它持有权重)的专家贡献,其余归零。因此:

    full(E 专家) == hot(dGPU 上 [0, E-K) 专家)
                    + cold(iGPU 上 [E-K, E) 专家,把这些列在 router_logits 里置换到最前)

两边都复用 ipex 快核。本模块负责 cold 那一半:一个钉在 iGPU(ZE_AFFINITY_MASK)上的
sidecar 子进程,常驻各 MoE 层的冷专家权重,经 CPU 共享内存服务卸载请求。

仅用于 prefill(decode 是内存带宽 bound,不卸载 —— 见 STATUS §7)。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

# ctrl 块(int64)槽位
CTRL_SLOTS = 16
(IDX_REQ, IDX_RESP, IDX_NTOK, IDX_LAYER, IDX_STOP,
 IDX_MODE, IDX_REG_E, IDX_ERR) = range(8)
# IDX_MODE: 0=offload 计算请求, 1=注册一层冷权重
# IDX_REG_E: 注册时本层冷专家数 K(各层一致)
# IDX_ERR:   sidecar 置 1 表示出错(dGPU 侧据此回退)

_SPIN_SLEEP = 0.0  # 纯自旋;必要时可设极小 sleep 降 CPU

# --------------------------------------------------------------------------- #
# 环境变量(沿用 VLLM_XPU_IGPU_* 家族的 raw os.getenv 约定,不进 envs.py)
# --------------------------------------------------------------------------- #
_DEFAULT_OFFLOAD_FRAC = 0.30  # auto 时卸载 ~30% 专家(原型 K* 落 30-50%,见 OFFLOAD_TUNING.md)


def moe_offload_enabled() -> bool:
    return os.getenv("VLLM_XPU_IGPU_MOE", "0") == "1"


def moe_offload_debug() -> bool:
    return os.getenv("VLLM_XPU_IGPU_MOE_DEBUG", "0") == "1"


def moe_offload_mask() -> str:
    return os.getenv("VLLM_XPU_IGPU_MOE_MASK", "1").strip()


def moe_offload_min_tokens() -> int:
    return int(os.getenv("VLLM_XPU_IGPU_MOE_MIN_TOKENS", "64"))


def resolve_offload_k(global_experts: int) -> int:
    """K = 卸载到 iGPU 的专家数。VLLM_XPU_IGPU_MOE_K<0(默认 -1)= auto=30%。
    夹到 [1, E-1] 以保证 hot/cold 两边都非空。"""
    raw = int(os.getenv("VLLM_XPU_IGPU_MOE_K", "-1"))
    k = round(_DEFAULT_OFFLOAD_FRAC * global_experts) if raw < 0 else raw
    return max(1, min(global_experts - 1, k))


@dataclass
class OffloadCfg:
    hidden: int          # H
    inter: int           # I(w2 的最后一维,= intermediate_size_per_partition)
    w13_up: int          # w13 第 1 维 = 2*I(is_act_and_mul)或 I
    global_experts: int  # E
    offload_k: int       # K(卸载到 iGPU 的专家数)
    top_k: int
    renormalize: bool
    use_grouped_topk: bool
    topk_group: int
    num_expert_group: int
    scoring_func: str
    max_tokens: int      # M 上限(= max_num_batched_tokens)
    mask: str            # iGPU 亲和掩码
    debug: bool
    ctrl_name: str
    in_name: str         # 复用:注册时装权重,计算时装 x + logits
    out_name: str
    logits_name: str


def _log(cfg_debug, *a):
    if cfg_debug:
        print("[igpu-moe-sidecar]", *a, flush=True)


def cold_perm(E: int, K: int) -> np.ndarray:
    """把冷专家 [E-K, E) 置换到前,其余接后:new col j -> old expert id。"""
    return np.concatenate([np.arange(E - K, E), np.arange(0, E - K)]).astype(np.int64)


# --------------------------------------------------------------------------- #
# sidecar 进程入口(钉在 iGPU)
# --------------------------------------------------------------------------- #
def igpu_sidecar_main(cfg: OffloadCfg):
    # 必须在 import torch 之前设亲和掩码 —— 本进程里 xpu:0 == iGPU
    os.environ["ZE_AFFINITY_MASK"] = cfg.mask
    import torch
    import intel_extension_for_pytorch as ipex  # noqa: F401

    dev = "xpu:0"
    E, K, H, I, w13_up = (cfg.global_experts, cfg.offload_k, cfg.hidden,
                          cfg.inter, cfg.w13_up)
    perm = torch.from_numpy(cold_perm(E, K)).to(dev)

    ctrl_shm = _spin_open(cfg.ctrl_name)
    in_shm = _spin_open(cfg.in_name)
    out_shm = _spin_open(cfg.out_name)
    logits_shm = _spin_open(cfg.logits_name)
    c = np.ndarray((CTRL_SLOTS,), dtype=np.int64, buffer=ctrl_shm.buf)
    # in_buf 既用于注册(装 K 个专家权重)也用于计算(装 x[M,H])
    in_np = np.ndarray((in_shm.size // 2,), dtype=np.float16, buffer=in_shm.buf)
    out_np = np.ndarray((cfg.max_tokens, H), dtype=np.float16, buffer=out_shm.buf)
    logits_np = np.ndarray((cfg.max_tokens, E), dtype=np.float16, buffer=logits_shm.buf)

    silu = "silu"
    modules: dict[int, object] = {}  # layer_id -> GatedMLPMOE(cold)
    w13_numel = K * w13_up * H
    w2_numel = K * H * I

    _log(cfg.debug, f"up on iGPU (mask={cfg.mask}) E={E} K={K} H={H} I={I}")
    last = 0
    while True:
        while c[IDX_REQ] == last and c[IDX_STOP] == 0:
            if _SPIN_SLEEP:
                time.sleep(_SPIN_SLEEP)
        if c[IDX_STOP] != 0:
            break
        seq = int(c[IDX_REQ])
        layer_id = int(c[IDX_LAYER])
        mode = int(c[IDX_MODE])
        try:
            if mode == 1:
                # 注册一层冷权重:in_buf 前 w13_numel 装 w13,接着 w2_numel 装 w2
                w13 = (torch.from_numpy(in_np[:w13_numel])
                       .view(K, w13_up, H).to(dev).clone().contiguous())
                w2 = (torch.from_numpy(in_np[w13_numel:w13_numel + w2_numel])
                      .view(K, H, I).to(dev).clone().contiguous())
                modules[layer_id] = ipex.llm.modules.GatedMLPMOE(
                    w13, w2, use_prepack=True, experts_start_id=0)
                torch.xpu.synchronize()
                _log(cfg.debug, f"registered layer {layer_id} "
                     f"({len(modules)} total)")
            else:
                # offload 计算:x[M,H] + logits[M,E] -> cold partial[M,H]
                M = int(c[IDX_NTOK])
                x = torch.from_numpy(in_np[:M * H]).view(M, H).to(dev)
                logits = torch.from_numpy(logits_np[:M]).to(dev)
                lg_cold = logits.index_select(1, perm).contiguous()
                mod = modules[layer_id]
                out = mod(x, cfg.use_grouped_topk, cfg.top_k, lg_cold,
                          cfg.renormalize, cfg.topk_group, cfg.num_expert_group,
                          None, cfg.scoring_func, silu, None)
                res = out.to("cpu")
                torch.xpu.synchronize()
                out_np[:M] = res.view(M, H).numpy()
            c[IDX_ERR] = 0
        except Exception as e:  # noqa: BLE001 —— 出错不拖垮 dGPU,置错误位回退
            c[IDX_ERR] = 1
            print(f"[igpu-moe-sidecar] ERROR layer={layer_id} mode={mode}: {e}",
                  flush=True)
        c[IDX_RESP] = seq
        last = seq

    for s in (ctrl_shm, in_shm, out_shm, logits_shm):
        s.close()


def _spin_open(name, retries=4000):
    last = None
    for _ in range(retries):
        try:
            return shared_memory.SharedMemory(name=name)
        except FileNotFoundError as e:
            last = e
            time.sleep(0.005)
    raise last


# --------------------------------------------------------------------------- #
# dGPU 进程内的客户端(单例)
# --------------------------------------------------------------------------- #
class IGpuMoeSidecar:
    """管理 iGPU sidecar 子进程 + 共享内存;dGPU worker 进程内单例。"""

    _instance: "IGpuMoeSidecar | None" = None

    def __init__(self, cfg: OffloadCfg):
        self.cfg = cfg
        self._seq = 0
        self._ready = False
        self._registered: set[int] = set()
        self._proc = None
        self._shms: list = []
        # 共享内存:ctrl + in(权重/激活复用) + out + logits
        H, E, M = cfg.hidden, cfg.global_experts, cfg.max_tokens
        # in_buf 需容纳 max(注册权重, 计算激活):权重 = K*(w13_up*H + H*I)
        w_numel = cfg.offload_k * (cfg.w13_up * H + H * cfg.inter)
        in_numel = max(w_numel, M * H)
        self._ctrl = shared_memory.SharedMemory(create=True, size=CTRL_SLOTS * 8)
        self._in = shared_memory.SharedMemory(create=True, size=in_numel * 2)
        self._out = shared_memory.SharedMemory(create=True, size=M * H * 2)
        self._logits = shared_memory.SharedMemory(create=True, size=M * E * 2)
        self._shms = [self._ctrl, self._in, self._out, self._logits]
        self._c = np.ndarray((CTRL_SLOTS,), dtype=np.int64, buffer=self._ctrl.buf)
        self._c[:] = 0
        self._in_np = np.ndarray((in_numel,), dtype=np.float16, buffer=self._in.buf)
        self._out_np = np.ndarray((M, H), dtype=np.float16, buffer=self._out.buf)
        self._logits_np = np.ndarray((M, E), dtype=np.float16, buffer=self._logits.buf)
        cfg.ctrl_name = self._ctrl.name
        cfg.in_name = self._in.name
        cfg.out_name = self._out.name
        cfg.logits_name = self._logits.name

    # ---- 生命周期 ----
    @classmethod
    def get(cls, cfg_factory) -> "IGpuMoeSidecar":
        if cls._instance is None:
            cls._instance = IGpuMoeSidecar(cfg_factory())
            cls._instance.spawn()
        return cls._instance

    def spawn(self):
        import multiprocessing as mp
        import atexit
        ctx = mp.get_context("spawn")  # 必须 spawn:全新解释器,掩码在 import torch 前设
        self._proc = ctx.Process(target=igpu_sidecar_main, args=(self.cfg,),
                                 name="igpu-moe-sidecar", daemon=True)
        self._proc.start()
        atexit.register(self.shutdown)
        self._ready = True
        if self.cfg.debug:
            print(f"[igpu-moe] sidecar spawned pid={self._proc.pid}", flush=True)

    @property
    def ready(self) -> bool:
        return self._ready and self._proc is not None and self._proc.is_alive()

    def is_registered(self, layer_id: int) -> bool:
        return layer_id in self._registered

    # ---- 一次性:把某层冷权重推给 sidecar ----
    def register_layer(self, layer_id: int, cold_w13_cpu, cold_w2_cpu) -> bool:
        """cold_w13_cpu: [K, w13_up, H] fp16 cpu;cold_w2_cpu: [K, H, I] fp16 cpu。"""
        w13 = cold_w13_cpu.contiguous().view(-1).numpy()
        w2 = cold_w2_cpu.contiguous().view(-1).numpy()
        n13, n2 = w13.size, w2.size
        self._in_np[:n13] = w13
        self._in_np[n13:n13 + n2] = w2
        self._c[IDX_NTOK] = 0
        self._c[IDX_LAYER] = layer_id
        self._c[IDX_MODE] = 1
        ok = self._roundtrip()
        if ok:
            self._registered.add(layer_id)
        return ok

    # ---- 每层 prefill:send() 发激活+logits(sidecar 并行算),wait() 取 cold partial ----
    def send(self, layer_id: int, x, router_logits) -> int:
        import torch
        M = x.shape[0]
        H = self.cfg.hidden
        self._in_np[:M * H] = x.detach().to("cpu", torch.float16).view(-1).numpy()
        self._logits_np[:M] = router_logits.detach().to("cpu", torch.float16).numpy()
        self._c[IDX_NTOK] = M
        self._c[IDX_LAYER] = layer_id
        self._c[IDX_MODE] = 0
        self._seq += 1
        self._c[IDX_RESP] = self._seq - 1  # 确保不等于新 seq
        self._c[IDX_REQ] = self._seq  # 发布请求(sidecar 开始并行计算)
        return self._seq

    def wait(self, seq: int, out_device):
        import torch
        while self._c[IDX_RESP] != seq:
            if not self.ready:
                raise RuntimeError("igpu-moe sidecar died")
        if int(self._c[IDX_ERR]) != 0:
            raise RuntimeError("igpu-moe sidecar reported error")
        M = int(self._c[IDX_NTOK])
        H = self.cfg.hidden
        res = torch.from_numpy(self._out_np[:M].copy())
        return res.to(out_device)

    def _roundtrip(self) -> bool:
        self._seq += 1
        self._c[IDX_RESP] = self._seq - 1
        self._c[IDX_REQ] = self._seq
        while self._c[IDX_RESP] != self._seq:
            if not self.ready:
                return False
        return int(self._c[IDX_ERR]) == 0

    def shutdown(self):
        if self._proc is None:
            return
        try:
            self._c[IDX_STOP] = 1
        except Exception:  # noqa: BLE001
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
        for s in self._shms:
            try:
                s.close(); s.unlink()
            except Exception:  # noqa: BLE001
                pass
        self._proc = None
        self._ready = False
        IGpuMoeSidecar._instance = None
