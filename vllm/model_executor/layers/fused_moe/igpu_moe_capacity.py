# SPDX-License-Identifier: Apache-2.0
"""iGPU 冷专家「容量模式」—— 按专家 index 切分,让装不下的 MoE 模型跑起来。

与同目录 ``igpu_moe_offload.py``(性能模式)的区别是**根本性**的:

|              | perf(igpu_moe_offload) | **capacity(本模块)**            |
|--------------|------------------------|----------------------------------|
| 冷专家权重   | dGPU 上**也有**一份     | **只在 iGPU**,dGPU 从不 materialize |
| 目的         | prefill overlap 提速    | 把显存装不下的部分挪走           |
| 生效阶段     | 仅 prefill(>=阈值)     | prefill + decode **全程**        |
| sidecar 挂了 | 回退完整 ipex           | **无法回退**,直接报错            |

原理:ipex ``GatedMLPMOE`` 做**全局** top_k + **全局** renorm,只计算
``experts_start_id`` 起本地持有的那些专家,其余归零。因此::

    full(E 专家) == hot(dGPU 持有 [0, E-K))  +  cold(iGPU 持有 [E-K, E))

容量模式把这个等式用作**权重放置**而非计算调度:

- ``FusedMoE.__init__`` 里把 ``local_num_experts`` 改成 ``E-K``、``_expert_map``
  中 ``[E-K, E)`` 置 -1 → 所有 quant method 的 ``create_weights`` 自动只在独显上
  分配 E-K 份,``experts_start_id`` 自动为 0。**独显峰值就是 E-K 份,没有中间双份。**
- ``FusedMoE.weight_loader`` 里,原本因 ``expert_id == -1`` 被丢弃的冷专家分片,
  改为写进本模块的 registration 共享内存;一层齐了就推给 sidecar。
- sidecar(钉在 iGPU 的独立进程)把 bf16/fp16 冷权重量化成 fp8(可选)后常驻,
  每层 forward 时收 ``x`` + ``router_logits``,返回冷专家的 partial。

环境变量(沿用 ``VLLM_XPU_IGPU_*`` 家族的 raw os.getenv 约定,不进 envs.py):

``VLLM_XPU_IGPU_MOE_CAPACITY=1``    总开关(默认关)
``VLLM_XPU_IGPU_MOE_COLD_K``        绝对冷专家数;<0 时用 COLD_FRAC
``VLLM_XPU_IGPU_MOE_COLD_FRAC``     冷专家比例,默认 0.45
``VLLM_XPU_IGPU_MOE_COLD_QUANT``    ``fp8``(默认)或 ``none``(iGPU 上保持原 dtype)
``VLLM_XPU_IGPU_MOE_MASK``          sidecar 的 ZE_AFFINITY_MASK,默认 ``1``
``VLLM_XPU_IGPU_MOE_REG_CHUNK``     注册时 iGPU 侧一次吞多少个专家,默认 8
``VLLM_XPU_IGPU_MOE_COLD_PERM=1``   改用"把冷专家 logits 列置换到最前 +
                                    experts_start_id=0"。**默认关,不要开** ——
                                    置换会改掉 top_k 的并列打破顺序,实测
                                    E=256/top_k=8 时每 1024 token 有 ~6 个
                                    被两半选中不同专家。默认走 start_id=E-K。
``VLLM_XPU_IGPU_MOE_DEBUG=1``       打日志

显存/内存账(Qwen3.6-35B-A3B,E=256 H=2048 I=512,fp8):

    每专家 = w13[2I,H] + w2[H,I] = 3,145,728 元素 → fp8 3.0 MiB / bf16 6.0 MiB
    iGPU 系统内存 = K × 3.0 MiB × 40 层     (K=114 → 13.4 GiB)
    registration 暂存 shm = K × 6.0 MiB     (K=114 → 684 MiB,注册完即释放)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory

import numpy as np

# --------------------------------------------------------------------------- #
# 控制块(int64 槽位)
# --------------------------------------------------------------------------- #
CTRL_SLOTS = 16
(
    IDX_REQ,      # dGPU 递增发布请求
    IDX_RESP,     # sidecar 回写同值表示完成
    IDX_OP,       # 见 OP_*
    IDX_LAYER,    # 层 id
    IDX_NTOK,     # forward 的 token 数 M
    IDX_ARG0,     # 通用参数
    IDX_ERR,      # sidecar 置 1 = 出错
    IDX_STOP,     # dGPU 置 1 = 退出
    IDX_READY,    # sidecar 置 1 = 起来了
) = range(9)

OP_REGISTER = 1   # 把 reg shm 里一层的冷权重吃进 iGPU
OP_FORWARD = 2    # 算一层冷专家 partial
OP_REG_DONE = 3   # 全部层注册完,释放 reg shm

_DEFAULT_COLD_FRAC = 0.45
_DEFAULT_REG_CHUNK = 8

_ENV_PREFIX = "VLLM_XPU_IGPU_MOE"


# --------------------------------------------------------------------------- #
# 环境变量
# --------------------------------------------------------------------------- #
def capacity_mode_enabled() -> bool:
    return os.getenv(f"{_ENV_PREFIX}_CAPACITY", "0") == "1"


def capacity_debug() -> bool:
    return os.getenv(f"{_ENV_PREFIX}_DEBUG", "0") == "1"


def capacity_mask() -> str:
    return os.getenv(f"{_ENV_PREFIX}_MASK", "1").strip()


def capacity_quant() -> str:
    q = os.getenv(f"{_ENV_PREFIX}_COLD_QUANT", "fp8").strip().lower()
    if q not in ("fp8", "none"):
        raise ValueError(f"{_ENV_PREFIX}_COLD_QUANT must be 'fp8' or 'none', got {q!r}")
    return q


def capacity_reg_chunk() -> int:
    return max(1, int(os.getenv(f"{_ENV_PREFIX}_REG_CHUNK", str(_DEFAULT_REG_CHUNK))))


def capacity_use_perm() -> bool:
    """是否把冷专家的 logits 列置换到最前(而不是用 experts_start_id=E-K)。

    默认 **关**。置换会改变 top_k 的并列打破顺序和 softmax 的求和次序,实测
    E=256/top_k=8/fp16 logits 下每 1024 个 token 有 ~6 个会被两半选中不同的专家
    (整行结果错掉);``experts_start_id=E-K`` 则 bad_rows=0。
    """
    return os.getenv(f"{_ENV_PREFIX}_COLD_PERM", "0") == "1"


def resolve_cold_k(global_experts: int) -> int:
    """K = 放在 iGPU 上的冷专家数。夹到 [0, E-1] 保证 dGPU 侧非空。"""
    raw = int(os.getenv(f"{_ENV_PREFIX}_COLD_K", "-1"))
    if raw < 0:
        frac = float(os.getenv(f"{_ENV_PREFIX}_COLD_FRAC", str(_DEFAULT_COLD_FRAC)))
        raw = round(frac * global_experts)
    return max(0, min(global_experts - 1, int(raw)))


# --------------------------------------------------------------------------- #
# dtype ←→ numpy 视图
#
# numpy 没有 bf16 / fp8,所以共享内存一律用同宽度的整型视图承载,两侧再 view 回去。
# --------------------------------------------------------------------------- #
_ITEMSIZE = {
    "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
    "uint8": 1, "int8": 1, "int32": 4, "int64": 8,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}

# 直接可用的 numpy dtype;其余用等宽整型搬运
_NP_DIRECT = {
    "float16": np.float16, "float32": np.float32, "float64": np.float64,
    "uint8": np.uint8, "int8": np.int8, "int32": np.int32, "int64": np.int64,
}
_NP_CARRIER = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def dtype_name(dt) -> str:
    name = str(dt).rsplit(".", 1)[-1]
    if name not in _ITEMSIZE:
        raise ValueError(f"igpu-moe-capacity: unsupported dtype {dt}")
    return name


def np_carrier_dtype(name: str):
    """共享内存里用的 numpy dtype(可能只是等宽整型载体)。"""
    return _NP_DIRECT.get(name, _NP_CARRIER[_ITEMSIZE[name]])


def torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def as_torch(arr: np.ndarray, name: str):
    """把共享内存的 numpy 视图变成目标 dtype 的 torch 视图(零拷贝)。"""
    import torch

    t = torch.from_numpy(arr)
    want = torch_dtype(name)
    return t if t.dtype == want else t.view(want)


def _no_dispatch():
    """屏蔽外层 TorchDispatchMode。

    fp8 streaming 加载路径(``ipex_quant.XPUFp8MoEMethod``)用一个全局
    ``CopyNumelCounter`` 统计 ``aten.copy_`` 的元素数来判断"这层加载完了没"。
    冷专家往暂存区的拷贝**不属于**那份统计,漏屏蔽会让它提前触发量化,
    把只加载了一半的 w13 量化掉。
    """
    from torch.utils._python_dispatch import _disable_current_modes

    return _disable_current_modes()


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class ColdCfg:
    # 形状
    global_experts: int      # E
    cold_k: int              # K
    hidden: int              # H
    inter: int               # I  (= w2 最后一维 = intermediate_size_per_partition)
    w13_up: int              # w13 第 1 维(is_act_and_mul 时 = 2I)
    num_layers: int          # 有多少个 MoE 层要注册
    max_tokens: int          # M 上限

    # 路由(必须与 dGPU 侧逐字一致,否则两半的 top_k 会选出不同专家)
    top_k: int
    renormalize: bool
    use_grouped_topk: bool
    topk_group: int | None
    num_expert_group: int | None
    scoring_func: str
    activation: str

    # dtype(用名字传,子进程再 getattr(torch, name))
    param_dtype: str         # 冷权重在 shm 里的原始 dtype(bf16/fp16)
    act_dtype: str           # x / out
    logits_dtype: str        # router_logits,**必须原样传,不能降精度**

    # 行为
    cold_quant: str          # "fp8" | "none"
    reg_chunk: int
    use_perm: bool
    mask: str
    debug: bool

    # shm 名字(由 client 填)
    ctrl_name: str = ""
    reg_name: str = ""
    act_name: str = ""
    out_name: str = ""
    logits_name: str = ""

    # 派生
    @property
    def w13_numel(self) -> int:
        return self.cold_k * self.w13_up * self.hidden

    @property
    def w2_numel(self) -> int:
        return self.cold_k * self.hidden * self.inter

    @property
    def reg_numel(self) -> int:
        return self.w13_numel + self.w2_numel

    def signature(self) -> tuple:
        """异构层检测:形状/路由不一致的层不能共用同一个 sidecar。"""
        return (self.global_experts, self.cold_k, self.hidden, self.inter,
                self.w13_up, self.top_k, self.renormalize, self.use_grouped_topk,
                self.topk_group, self.num_expert_group, self.scoring_func,
                self.activation, self.param_dtype, self.act_dtype,
                self.logits_dtype, self.use_perm)


def cold_perm(E: int, K: int) -> np.ndarray:
    """把冷专家 [E-K, E) 置换到最前:new col j -> old expert id。"""
    return np.concatenate([np.arange(E - K, E), np.arange(0, E - K)]).astype(np.int64)


# --------------------------------------------------------------------------- #
# 层 id 分配 + sidecar 挂载
# --------------------------------------------------------------------------- #
_LAYER_COUNTER = 0


def next_layer_id() -> int:
    global _LAYER_COUNTER
    lid = _LAYER_COUNTER
    _LAYER_COUNTER += 1
    return lid


def reset_layer_ids() -> None:
    global _LAYER_COUNTER
    _LAYER_COUNTER = 0


def disable_conflicting_fast_paths() -> None:
    """关掉会绕过 ``quant_method.apply`` 直接读 ``w13_weight`` 的自定义 kernel。

    ``Qwen3NextSparseMoeBlock.forward``(qwen3.5 系列复用)在 fp8/int4 且
    ``num_tokens <= 128`` 时会走 ESIMD 融合算子,直接拿 ``self.experts.w13_weight``
    并按 ``n_routed_experts`` = **全局** 专家数索引。容量模式下独显只有 E-K 份,
    那条路要么越界要么静默算出半个 MoE —— 必须关掉,让 decode 也走 ipex + 冷专家。
    """
    if os.environ.get("DISABLE_ESIMD_MOE") != "1":
        os.environ["DISABLE_ESIMD_MOE"] = "1"
        return True
    return False


def build_cfg(
    *,
    global_experts: int,
    cold_k: int,
    hidden: int,
    inter: int,
    is_act_and_mul: bool,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool,
    topk_group,
    num_expert_group,
    scoring_func: str,
    activation: str,
    param_dtype,
    act_dtype,
    logits_dtype,
    max_tokens: int,
) -> ColdCfg:
    return ColdCfg(
        global_experts=global_experts,
        cold_k=cold_k,
        hidden=hidden,
        inter=inter,
        w13_up=(2 * inter) if is_act_and_mul else inter,
        num_layers=0,
        max_tokens=max_tokens,
        top_k=top_k,
        renormalize=bool(renormalize),
        use_grouped_topk=bool(use_grouped_topk),
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        scoring_func=scoring_func,
        activation=activation,
        param_dtype=dtype_name(param_dtype),
        act_dtype=dtype_name(act_dtype),
        logits_dtype=dtype_name(logits_dtype),
        cold_quant=capacity_quant(),
        reg_chunk=capacity_reg_chunk(),
        use_perm=capacity_use_perm(),
        mask=capacity_mask(),
        debug=capacity_debug(),
    )


def attach(cfg: ColdCfg) -> "ColdExpertSidecar":
    """第一层建 sidecar,后续层复核签名一致(异构 MoE 不支持)。"""
    sc = ColdExpertSidecar.instance()
    if sc is None:
        return ColdExpertSidecar.get_or_create(cfg)
    if sc.cfg.signature() != cfg.signature():
        raise NotImplementedError(
            "igpu cold capacity mode does not support heterogeneous MoE layers.\n"
            f"  first layer: {sc.cfg.signature()}\n"
            f"  this  layer: {cfg.signature()}")
    return sc


# =========================================================================== #
# sidecar 进程(钉在 iGPU)
# =========================================================================== #
def _sidecar_log(debug, *a):
    if debug:
        print("[igpu-cold]", *a, flush=True)


def sidecar_main(cfg: ColdCfg) -> int:
    # ZE_AFFINITY_MASK 由父进程在 Popen(env=...) 里设好,这里不再动 —— 靠 env 而不是
    # "import torch 之前 setenv" 才是可靠的(spawn 会先 import 整个 vllm 包)。
    import torch

    import intel_extension_for_pytorch as ipex  # noqa: F401

    debug = cfg.debug
    dev = "xpu:0"
    E, K = cfg.global_experts, cfg.cold_k
    H, I, w13_up = cfg.hidden, cfg.inter, cfg.w13_up

    param_dt = torch_dtype(cfg.param_dtype)
    act_dt = torch_dtype(cfg.act_dtype)
    logits_dt = torch_dtype(cfg.logits_dtype)

    ctrl_shm = _spin_open(cfg.ctrl_name)
    act_shm = _spin_open(cfg.act_name)
    out_shm = _spin_open(cfg.out_name)
    logits_shm = _spin_open(cfg.logits_name)
    reg_shm = _spin_open(cfg.reg_name)

    c = np.ndarray((CTRL_SLOTS,), dtype=np.int64, buffer=ctrl_shm.buf)
    act_np = np.ndarray((cfg.max_tokens * H,),
                        dtype=np_carrier_dtype(cfg.act_dtype), buffer=act_shm.buf)
    out_np = np.ndarray((cfg.max_tokens * H,),
                        dtype=np_carrier_dtype(cfg.act_dtype), buffer=out_shm.buf)
    logits_np = np.ndarray((cfg.max_tokens * E,),
                           dtype=np_carrier_dtype(cfg.logits_dtype),
                           buffer=logits_shm.buf)
    reg_np: np.ndarray | None = np.ndarray(
        (cfg.reg_numel,), dtype=np_carrier_dtype(cfg.param_dtype), buffer=reg_shm.buf)

    act_t = as_torch(act_np, cfg.act_dtype)
    out_t = as_torch(out_np, cfg.act_dtype)
    logits_t = as_torch(logits_np, cfg.logits_dtype)
    reg_t = as_torch(reg_np, cfg.param_dtype)

    perm = torch.from_numpy(cold_perm(E, K)).to(dev) if cfg.use_perm else None
    experts_start_id = 0 if cfg.use_perm else (E - K)

    modules: dict[int, object] = {}
    keepalive: dict[int, tuple] = {}   # 防止 GatedMLPMOE 之外的引用被 GC

    # 把实际拿到的设备名打出来 —— 掩码没生效的话 sidecar 会悄悄跑在独显上,
    # 那样既不省显存也测不出问题,必须一眼能看见。
    try:
        devname = torch.xpu.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        devname = f"<unavailable: {e}>"
    _sidecar_log(True, f"up on '{devname}' (ZE_AFFINITY_MASK={cfg.mask}) "
                       f"E={E} K={K} H={H} I={I} quant={cfg.cold_quant} "
                       f"perm={cfg.use_perm} start_id={experts_start_id}")
    if "Arc" in devname and "Pro" in devname:
        print("[igpu-cold] WARNING: the sidecar landed on what looks like the "
              "discrete GPU, not the iGPU. ZE_AFFINITY_MASK="
              f"{cfg.mask} did not take effect; cold experts would eat dGPU "
              "VRAM instead of system RAM.", flush=True)
    c[IDX_READY] = 1

    last = 0
    while True:
        while c[IDX_REQ] == last and c[IDX_STOP] == 0:
            pass
        if c[IDX_STOP] != 0:
            break
        seq = int(c[IDX_REQ])
        op = int(c[IDX_OP])
        lid = int(c[IDX_LAYER])
        try:
            if op == OP_REGISTER:
                assert reg_t is not None, "registration buffer already released"
                mod, ka = _ingest_layer(torch, ipex, cfg, reg_t, dev,
                                        experts_start_id)
                modules[lid] = mod
                keepalive[lid] = ka
                torch.xpu.synchronize()
                _sidecar_log(debug, f"layer {lid} registered "
                                    f"({len(modules)}/{cfg.num_layers})")
            elif op == OP_REG_DONE:
                reg_t = None
                reg_np = None
                reg_shm.close()
                _sidecar_log(True, f"registration done: {len(modules)} layers, "
                                   f"xpu alloc={_xpu_alloc_gib(torch):.2f} GiB")
            elif op == OP_FORWARD:
                M = int(c[IDX_NTOK])
                x = act_t[: M * H].view(M, H).to(dev, non_blocking=False)
                lg = logits_t[: M * E].view(M, E).to(dev, non_blocking=False)
                if perm is not None:
                    lg = lg.index_select(1, perm).contiguous()
                mod = modules[lid]
                y = mod(x, cfg.use_grouped_topk, cfg.top_k, lg, cfg.renormalize,
                        cfg.topk_group, cfg.num_expert_group, None,
                        cfg.scoring_func, cfg.activation, None)
                res = y.reshape(M * H).to(dtype=act_dt, device="cpu")
                torch.xpu.synchronize()
                out_t[: M * H].copy_(res)
            else:
                raise RuntimeError(f"unknown op {op}")
            c[IDX_ERR] = 0
        except BaseException as e:  # noqa: BLE001 —— 任何错都要让 dGPU 侧看见
            import traceback

            c[IDX_ERR] = 1
            print(f"[igpu-cold] ERROR op={op} layer={lid}: {e}", flush=True)
            traceback.print_exc()
        c[IDX_RESP] = seq
        last = seq

    for s in (ctrl_shm, act_shm, out_shm, logits_shm):
        try:
            s.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


def _xpu_alloc_gib(torch) -> float:
    try:
        return torch.xpu.memory_allocated() / (1 << 30)
    except Exception:  # noqa: BLE001
        return 0.0


def _ingest_layer(torch, ipex, cfg: ColdCfg, reg_t, dev: str,
                  experts_start_id: int):
    """把 reg shm 里一层的 K 个冷专家吃进 iGPU,分块以限制瞬时显存。

    返回 (GatedMLPMOE, keepalive_tuple)。
    """
    K, H, I, w13_up = cfg.cold_k, cfg.hidden, cfg.inter, cfg.w13_up
    src_w13 = reg_t[: cfg.w13_numel].view(K, w13_up, H)
    src_w2 = reg_t[cfg.w13_numel:].view(K, H, I)

    if cfg.cold_quant == "fp8":
        from vllm import _custom_ops as ops
        from vllm.platforms import current_platform

        qdt = current_platform.fp8_dtype()
        w13 = torch.empty((K, w13_up, H), dtype=qdt, device=dev)
        w2 = torch.empty((K, H, I), dtype=qdt, device=dev)
        s13 = torch.empty((K,), dtype=torch.float32, device=dev)
        s2 = torch.empty((K,), dtype=torch.float32, device=dev)
        for lo in range(0, K, cfg.reg_chunk):
            hi = min(lo + cfg.reg_chunk, K)
            # 一次只把 chunk 份 bf16 搬上 iGPU,量化完立刻丢
            b13 = src_w13[lo:hi].to(dev)
            b2 = src_w2[lo:hi].to(dev)
            for j in range(hi - lo):
                w13[lo + j], s13[lo + j] = ops.scaled_fp8_quant(b13[j])
                w2[lo + j], s2[lo + j] = ops.scaled_fp8_quant(b2[j])
            del b13, b2
        mod = ipex.llm.modules.GatedMLPMOE(
            w13, w2,
            w1_scale_inv=s13, w2_scale_inv=s2,
            a1_scale_inv=None, a2_scale_inv=None,
            use_prepack=True, experts_start_id=experts_start_id,
        )
        return mod, (w13, w2, s13, s2)

    # cold_quant == "none":原 dtype 直接常驻(不分块,瞬时占用 = 一层的 bf16 全量)
    w13 = src_w13.to(dev).contiguous()
    w2 = src_w2.to(dev).contiguous()
    mod = ipex.llm.modules.GatedMLPMOE(
        w13, w2, use_prepack=True, experts_start_id=experts_start_id)
    return mod, (w13, w2)


def _spin_open(name: str, retries: int = 6000):
    """sidecar 侧打开一段已存在的共享内存。

    顺手从**本进程**的 resource_tracker 里注销:段的所有权在父进程,子进程退出时
    再去 unlink 一次只会刷出一堆 "leaked shared_memory / No such file" 警告。
    """
    from multiprocessing import resource_tracker

    last = None
    for _ in range(retries):
        try:
            shm = shared_memory.SharedMemory(name=name)
            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:  # noqa: BLE001
                pass
            return shm
        except FileNotFoundError as e:
            last = e
            time.sleep(0.005)
    raise last


# =========================================================================== #
# dGPU 进程内的客户端(worker 进程内单例)
# =========================================================================== #
class ColdExpertSidecar:
    """管理 iGPU sidecar 子进程 + 共享内存。

    生命周期::

        get_or_create(cfg)      # FusedMoE.__init__ 第一次遇到 MoE 层时
        staging_w13 / staging_w2 + note_staged   # weight_loader 里逐个冷专家
        register_layer(lid)     # 某层冷专家齐了(note_staged 返回 True)
        finish_registration()   # 全部层齐了,释放 684 MiB 的 reg shm
        send(lid, x, logits) / wait(seq)         # 每层 forward
    """

    _instance: "ColdExpertSidecar | None" = None

    # ---------------------------------------------------------------- 构造 #
    def __init__(self, cfg: ColdCfg):
        self.cfg = cfg
        self._seq = 0
        self._proc: subprocess.Popen | None = None
        self._registered: set[int] = set()
        self._staging: dict[int, dict] = {}
        # 4 not 2: with shards walked in numeric order, Qwen3.6-35B-A3B really
        # does keep 3 layers open at once (file 8 holds layers 11/12/13's
        # gate_up while their down_proj is in file 9), so 2 fails on a healthy
        # checkpoint. Each in-flight layer costs cold_k*(w13+w2) in param dtype
        # -- ~818 MiB at K=130 -- so this is a real host-RAM knob, not a
        # formality.
        self._max_inflight = max(
            1, int(os.getenv(f"{_ENV_PREFIX}_MAX_INFLIGHT", "4")))
        self._reg_released = False
        self._warned_logits_dtype = False

        E, H, M = cfg.global_experts, cfg.hidden, cfg.max_tokens
        self._ctrl = shared_memory.SharedMemory(create=True, size=CTRL_SLOTS * 8)
        self._reg = shared_memory.SharedMemory(
            create=True, size=cfg.reg_numel * _ITEMSIZE[cfg.param_dtype])
        self._act = shared_memory.SharedMemory(
            create=True, size=M * H * _ITEMSIZE[cfg.act_dtype])
        self._out = shared_memory.SharedMemory(
            create=True, size=M * H * _ITEMSIZE[cfg.act_dtype])
        self._logits = shared_memory.SharedMemory(
            create=True, size=M * E * _ITEMSIZE[cfg.logits_dtype])

        cfg.ctrl_name = self._ctrl.name
        cfg.reg_name = self._reg.name
        cfg.act_name = self._act.name
        cfg.out_name = self._out.name
        cfg.logits_name = self._logits.name

        self._c = np.ndarray((CTRL_SLOTS,), dtype=np.int64, buffer=self._ctrl.buf)
        self._c[:] = 0

        reg_np = np.ndarray((cfg.reg_numel,), dtype=np_carrier_dtype(cfg.param_dtype),
                            buffer=self._reg.buf)
        reg_t = as_torch(reg_np, cfg.param_dtype)
        self._reg_w13 = reg_t[: cfg.w13_numel].view(cfg.cold_k, cfg.w13_up, cfg.hidden)
        self._reg_w2 = reg_t[cfg.w13_numel:].view(cfg.cold_k, cfg.hidden, cfg.inter)

        act_np = np.ndarray((M * H,), dtype=np_carrier_dtype(cfg.act_dtype),
                            buffer=self._act.buf)
        out_np = np.ndarray((M * H,), dtype=np_carrier_dtype(cfg.act_dtype),
                            buffer=self._out.buf)
        lg_np = np.ndarray((M * E,), dtype=np_carrier_dtype(cfg.logits_dtype),
                           buffer=self._logits.buf)
        self._act_t = as_torch(act_np, cfg.act_dtype)
        self._out_t = as_torch(out_np, cfg.act_dtype)
        self._logits_t = as_torch(lg_np, cfg.logits_dtype)

    @classmethod
    def get_or_create(cls, cfg: ColdCfg) -> "ColdExpertSidecar":
        if cls._instance is None:
            inst = cls(cfg)
            inst.spawn()
            cls._instance = inst
        return cls._instance

    @classmethod
    def instance(cls) -> "ColdExpertSidecar | None":
        return cls._instance

    # ------------------------------------------------------------ 生命周期 #
    def spawn(self, ready_timeout: float = 120.0) -> None:
        """用 subprocess 而不是 multiprocessing.spawn 起 sidecar。

        原因:``mp`` 的 spawn 子进程会先 import 整个 vllm 包(从而 import torch),
        之后再执行我们的函数体 —— 那时才 setenv ``ZE_AFFINITY_MASK`` 就晚了。
        直接 Popen(env=...) 能保证掩码在进程诞生前就位。
        """
        import atexit

        env = dict(os.environ)
        env["ZE_AFFINITY_MASK"] = self.cfg.mask
        # 子进程只跑 sidecar,不能再递归开一个
        env[f"{_ENV_PREFIX}_CAPACITY"] = "0"
        env["VLLM_XPU_IGPU_MOE"] = "0"
        # iGPU 没有独立显存,别让任何人按独显的假设去 reserve
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        payload = json.dumps(asdict(self.cfg))
        cmd = [sys.executable, "-m",
               "vllm.model_executor.layers.fused_moe.igpu_moe_capacity",
               "--cfg", payload]
        self._proc = subprocess.Popen(cmd, env=env)

        deadline = time.time() + ready_timeout
        while self._c[IDX_READY] == 0:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    "igpu cold-expert sidecar exited during startup with code "
                    f"{self._proc.returncode}; see its stderr above")
            if time.time() > deadline:
                self.shutdown()
                raise RuntimeError(
                    f"igpu cold-expert sidecar not ready after {ready_timeout}s")
            time.sleep(0.01)
        atexit.register(self.shutdown)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _assert_alive(self) -> None:
        if not self.alive:
            rc = self._proc.returncode if self._proc else "n/a"
            raise RuntimeError(
                "igpu cold-expert sidecar died (rc=%s). Capacity mode cannot "
                "fall back: the cold experts only exist in that process." % rc)

    def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            self._c[IDX_STOP] = 1
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self._proc.kill()
        self._proc = None
        for s in (self._ctrl, self._act, self._out, self._logits):
            _close_unlink(s)
        if not self._reg_released:
            _close_unlink(self._reg)
            self._reg_released = True
        ColdExpertSidecar._instance = None

    # -------------------------------------------------------------- 注册 #
    def cold_local_id(self, global_expert_id: int) -> int:
        """全局专家 id → 冷专家局部 id;不是冷专家返回 -1。"""
        base = self.cfg.global_experts - self.cfg.cold_k
        return global_expert_id - base if global_expert_id >= base else -1

    def _staging_for(self, lid: int) -> dict:
        """每层一份独立的 CPU 暂存。

        不能直接往 reg shm 里写:safetensors 分片的遍历顺序不保证"一层写完再写
        下一层"(本模型 40 层分在 26 个文件里,层会跨文件)。用 per-layer 字典
        天然容忍交错,代价是同时在飞的层各占一份;超过 max_inflight 就报错,
        免得病态顺序把内存吃光。
        """
        import torch

        st = self._staging.get(lid)
        if st is not None:
            return st
        if self.cfg.debug:
            print(f"[igpu-cold] staging open lid={lid} "
                  f"(in flight now: {sorted(self._staging) + [lid]})",
                  flush=True)
        if len(self._staging) >= self._max_inflight:
            raise RuntimeError(
                f"igpu cold: more than {self._max_inflight} MoE layers are "
                f"loading concurrently (in flight: {sorted(self._staging)}, "
                f"new: {lid}). Raise {_ENV_PREFIX}_MAX_INFLIGHT if you really "
                "have that much host RAM to spare.")
        cfg = self.cfg
        pdt = torch_dtype(cfg.param_dtype)
        st = {
            # device="cpu" 是必须的:模型加载期 default device 可能已经是 xpu
            "w13": torch.empty((cfg.cold_k, cfg.w13_up, cfg.hidden),
                               dtype=pdt, device="cpu"),
            "w2": torch.empty((cfg.cold_k, cfg.hidden, cfg.inter),
                              dtype=pdt, device="cpu"),
            "n13": 0,
            "n2": 0,
        }
        self._staging[lid] = st
        return st

    def staging_w13(self, lid: int, cold_id: int):
        """该冷专家的 w13 暂存视图 [w13_up, H](torch, CPU)。"""
        return self._staging_for(lid)["w13"][cold_id]

    def staging_w2(self, lid: int, cold_id: int):
        """该冷专家的 w2 暂存视图 [H, I](torch, CPU)。"""
        return self._staging_for(lid)["w2"][cold_id]

    def note_staged(self, lid: int, n_w13: int = 0, n_w2: int = 0) -> bool:
        """记账;该层的冷专家全齐了返回 True。"""
        st = self._staging_for(lid)
        st["n13"] += n_w13
        st["n2"] += n_w2
        return st["n13"] >= self.cfg.w13_numel and st["n2"] >= self.cfg.w2_numel

    def register_layer(self, lid: int) -> None:
        """把暂存的一层冷权重拷进 reg shm 并让 sidecar 吃进 iGPU。"""
        if lid in self._registered:
            raise RuntimeError(f"igpu cold: layer {lid} registered twice")
        if self._reg_released:
            raise RuntimeError(
                f"igpu cold: layer {lid} arrived after finish_registration()")
        st = self._staging.pop(lid)
        with _no_dispatch():
            self._reg_w13.copy_(st["w13"])
            self._reg_w2.copy_(st["w2"])
        del st
        self._c[IDX_LAYER] = lid
        self._c[IDX_OP] = OP_REGISTER
        self._roundtrip(spin=False)
        self._registered.add(lid)

    @property
    def num_registered(self) -> int:
        return len(self._registered)

    def is_registered(self, lid: int) -> bool:
        return lid in self._registered

    def finish_registration(self) -> None:
        """全部层注册完:释放 registration 暂存 shm(本模型 ~684 MiB)。"""
        if self._reg_released:
            return
        if self._staging:
            raise RuntimeError(
                "igpu cold: finish_registration() with layers still partially "
                f"staged: {sorted(self._staging)}. Some cold expert weights "
                "never arrived — the model would compute a partial MoE.")
        self._c[IDX_OP] = OP_REG_DONE
        self._roundtrip(spin=False)
        self._reg_w13 = None
        self._reg_w2 = None
        _close_unlink(self._reg)
        self._reg_released = True

    # ------------------------------------------------------------ forward #
    def send(self, lid: int, x, router_logits) -> int:
        """把 x / router_logits 放进 shm 并发布请求(sidecar 立刻开始算)。

        ``router_logits`` **原样**(不降精度)传过去 —— 两侧要在同一份 logits 上
        做全局 top_k,精度不一致会选出不同专家,结果就不是 full MoE 了。
        """
        import torch

        self._assert_alive()
        cfg = self.cfg
        M, H, E = x.shape[0], cfg.hidden, cfg.global_experts
        if M > cfg.max_tokens:
            raise RuntimeError(
                f"igpu cold: M={M} exceeds max_tokens={cfg.max_tokens}")

        act_dt = torch_dtype(cfg.act_dtype)
        lg_dt = torch_dtype(cfg.logits_dtype)
        if router_logits.dtype != lg_dt and not self._warned_logits_dtype:
            self._warned_logits_dtype = True
            print(f"[igpu-cold] WARNING: router_logits are {router_logits.dtype} "
                  f"but the sidecar was configured for {lg_dt}. The hot and cold "
                  f"halves must run top_k on identical values; a lossy cast here "
                  f"can make them pick different experts. Set "
                  f"{_ENV_PREFIX}_CAPACITY off or fix router_logits_dtype.",
                  flush=True)
        with _no_dispatch():
            self._act_t[: M * H].copy_(
                x.detach().reshape(-1).to(dtype=act_dt, device="cpu"))
            self._logits_t[: M * E].copy_(
                router_logits.detach().reshape(-1).to(dtype=lg_dt, device="cpu"))
        self._c[IDX_NTOK] = M
        self._c[IDX_LAYER] = lid
        self._c[IDX_OP] = OP_FORWARD
        self._seq += 1
        self._c[IDX_RESP] = self._seq - 1
        self._c[IDX_REQ] = self._seq          # 发布:sidecar 开始并行计算
        return self._seq

    def wait(self, seq: int, out_device, out_dtype):
        """自旋等 sidecar 完成,返回冷专家 partial [M, H]。"""
        import torch

        spins = 0
        while self._c[IDX_RESP] != seq:
            spins += 1
            if (spins & 0xFFFF) == 0:
                self._assert_alive()
        if int(self._c[IDX_ERR]) != 0:
            raise RuntimeError(
                "igpu cold-expert sidecar reported an error (see its log). "
                "Capacity mode cannot fall back.")
        M, H = int(self._c[IDX_NTOK]), self.cfg.hidden
        with _no_dispatch():
            res = self._out_t[: M * H].view(M, H).clone()
        return res.to(device=out_device, dtype=out_dtype)

    def forward(self, lid: int, x, router_logits):
        """同步版(不 overlap),调试用。"""
        seq = self.send(lid, x, router_logits)
        return self.wait(seq, x.device, x.dtype)

    # -------------------------------------------------------------- 内部 #
    def _roundtrip(self, spin: bool = True) -> None:
        self._assert_alive()
        self._seq += 1
        self._c[IDX_RESP] = self._seq - 1
        self._c[IDX_REQ] = self._seq
        while self._c[IDX_RESP] != self._seq:
            if not spin:
                time.sleep(0.001)
            self._assert_alive()
        if int(self._c[IDX_ERR]) != 0:
            raise RuntimeError(
                "igpu cold-expert sidecar reported an error during "
                f"op={int(self._c[IDX_OP])} layer={int(self._c[IDX_LAYER])}; "
                "see its log above.")


def _close_unlink(shm) -> None:
    try:
        shm.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        shm.unlink()
    except Exception:  # noqa: BLE001
        pass


# =========================================================================== #
# 给 quant method 用的 forward 帮手
# =========================================================================== #
def apply_with_cold(layer, hot_fn, x, router_logits):
    """capacity 模式下的统一 forward:``hot(dGPU) ‖ cold(iGPU)`` 然后相加。

    先发请求让 iGPU 开工,再提交 dGPU 的 hot 计算(ipex 是异步的),最后 wait —— 两
    边真并行。``hot_fn`` 是一个无参 callable,内部调 ``layer.ipex_fusion(...)``。
    """
    lid = layer._igpu_cold_lid
    sc = ColdExpertSidecar.instance()
    if sc is None or not sc.is_registered(lid):
        raise RuntimeError(
            f"igpu cold: layer {lid} has no registered cold experts; "
            "the model would silently compute a partial MoE. Aborting.")
    # 第一次 forward = 权重加载彻底结束,可以把 registration 暂存 shm 还给系统。
    if not sc._reg_released:
        sc.finish_registration()
    seq = sc.send(lid, x, router_logits)
    hot = hot_fn()
    cold = sc.wait(seq, hot.device, hot.dtype)
    return hot + cold


def layer_uses_cold(layer) -> bool:
    return getattr(layer, "_igpu_cold_k", 0) > 0


# =========================================================================== #
# 子进程入口
# =========================================================================== #
def _main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="igpu_moe_capacity")
    p.add_argument("--cfg", required=True, help="JSON-encoded ColdCfg")
    ns = p.parse_args(argv)
    cfg = ColdCfg(**json.loads(ns.cfg))
    return sidecar_main(cfg)


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
