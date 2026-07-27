# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from concurrent.futures import ThreadPoolExecutor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_router import FusedMoERouter
from vllm.model_executor.utils import set_weight_attrs
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from vllm.utils.math_utils import round_up

from vllm.envs import VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT, VLLM_QUANTIZE_Q40_LIB
from vllm.model_executor.layers.quantization.fp8 import (
    CopyNumelCounter, _copy_missing_attrs,
)
from vllm.logger import init_logger
import ctypes
import os
from packaging import version

logger = init_logger(__name__)

MIN_IPEX_VERSION = "2.5.0"
QK4_GROUP_SIZE: int = 128
QK4_PACK_FACTOR: int = 8

# Optional XPU-side Q4_0 quantization kernel. When available, streaming INT4
# quantization can quantize BF16 weights directly on XPU — skipping the
# transient D→H copy + CPU ggml + H→D that otherwise causes a ~1 side-sized
# BF16 peak on XPU per layer. Falls back to the CPU path silently if the
# module is missing or if VLLM_INT4_DISABLE_XPU_QUANT=1 is set.
_HAS_XPU_Q4_0 = False
try:
    import custom_esimd_kernels_vllm.q4_0_quant_ops  # noqa: F401
    _HAS_XPU_Q4_0 = True
except ImportError:
    pass


def _use_xpu_quant() -> bool:
    if not _HAS_XPU_Q4_0:
        return False
    if os.environ.get("VLLM_INT4_DISABLE_XPU_QUANT", "0") == "1":
        return False
    return True


def _xpu_q4_0_quantize(bf16_xpu: torch.Tensor):
    """Quantize a 2D BF16/FP16 XPU tensor to GGML Q4_0 layout, returning
    ``(qweight_xpu [M, K/8] int32, scale_xpu [M, K/128] float16)``.
    Allocates outputs on the same XPU device as the input."""
    assert bf16_xpu.dim() == 2, bf16_xpu.shape
    M, K = bf16_xpu.shape
    dev = bf16_xpu.device
    qweight = torch.empty(
        M, K // QK4_PACK_FACTOR, dtype=torch.int32, device=dev)
    scale = torch.empty(
        M, K // QK4_GROUP_SIZE, dtype=torch.float16, device=dev)
    torch.ops.custom_esimd_kernels_vllm.q4_0_quantize(
        bf16_xpu.contiguous(), qweight, scale)
    return qweight, scale

_QLIB_CACHE = None

def _get_quant_lib():
    """
    Lazy loads the quantization library and sets up argtypes.
    Singleton pattern to avoid reloading the DLL multiple times.
    """
    global _QLIB_CACHE
    if _QLIB_CACHE is not None:
        return _QLIB_CACHE

    try:
        clib = ctypes.CDLL(VLLM_QUANTIZE_Q40_LIB)
    except OSError as e:
        raise RuntimeError(f"Failed to load required quantization lib at {VLLM_QUANTIZE_Q40_LIB}: {e}")

    # Updated argtypes to match C signature:
    # (float *src, int32_t *qweight, ggml_fp16_t *scale, int out_features, int in_features, int block_size)
    clib.quantize_q4_0_to_qweight_and_scale.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # [New] block_size argument
    ]
    clib.quantize_q4_0_to_qweight_and_scale.restype = ctypes.c_size_t

    _QLIB_CACHE = clib
    return clib

def ggml_quantize_tensor(weight: torch.Tensor,
                         out_qweight: torch.Tensor,
                         out_scale: torch.Tensor,
                         out_features: int,
                         in_features: int,
                         block_size: int = QK4_GROUP_SIZE,
                         transpose: bool = True):
    """
    Shared implementation for quantizing a tensor using the C library.

    Args:
        transpose: If True (default), transpose the output tensors. The C code
            fills row-wise [out_features, ...], and some callers (Linear) need
            the transposed layout. MoE callers that want the original
            [out_features, ...] layout should pass transpose=False to avoid a
            redundant transpose-then-transpose-back.
    """
    # Assertions
    assert weight.dim() == 2
    # Validate shapes considering packing factor and block size
    assert out_qweight.shape == (out_features, in_features // QK4_PACK_FACTOR)
    assert out_scale.shape == (out_features, in_features // block_size)

    assert weight.dtype == torch.float32
    assert out_qweight.dtype == torch.int32
    assert out_scale.dtype == torch.float16

    assert out_qweight.is_contiguous()
    assert out_scale.is_contiguous()

    # Ctypes casting
    src = ctypes.cast(weight.data.data_ptr(), ctypes.POINTER(ctypes.c_float))
    qweight = ctypes.cast(out_qweight.data.data_ptr(), ctypes.POINTER(ctypes.c_int32))
    scale = ctypes.cast(out_scale.data.data_ptr(), ctypes.POINTER(ctypes.c_uint16))

    clib = _get_quant_lib()

    # Call C function with the new block_size parameter
    clib.quantize_q4_0_to_qweight_and_scale(src, qweight, scale, out_features, in_features, block_size)

    if transpose:
        # Transpose for callers that need column-major layout (e.g. Linear)
        out_qweight = out_qweight.transpose(0, 1).contiguous()
        out_scale = out_scale.transpose(0, 1).contiguous()

    return out_qweight, out_scale

# ==============================================================================
#  Classes
# ==============================================================================

class SymInt4Config(QuantizationConfig):
    """SYM_INT4 quantization config class which uses IPEX kernel behind the scene..."""
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> str:
        return "sym_int4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SymInt4Config":
        return cls()

    @classmethod
    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        modules_to_not_convert = ["visual", "vision", "vpm", "resampler",
                                  "shared_expert"]
        modules_to_convert=["vision_experts"]
        if any(key in prefix for key in modules_to_not_convert) and not any(key in prefix for key in modules_to_convert):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            return SymInt4LinearMethod(self)
        if isinstance(layer, FusedMoE):
            return XPUGPTQInt4LinearMoEMethod(self, layer.moe_config)
        else:
            return None


class SymInt4LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: SymInt4Config):
        self.quant_config = quant_config
        # Ensure lib is loaded on init
        _get_quant_lib()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight_dtype = params_dtype

        if VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT:
            # Legacy path: buffer BF16 weights on CPU; they will later be
            # moved to XPU by device_loading_context and quantized in
            # process_weights_after_loading.
            weight = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=weight_dtype,
                    device="cpu",
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)
            return

        # Streaming path: allocate on meta; the patched weight_loader will
        # materialize on XPU on the first shard, quantize once fully loaded,
        # and release the BF16 intermediate.
        orig_weight_loader = weight_loader
        layer._load_device = torch.get_default_device()

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            # First call: materialize the meta placeholder on the target
            # device and re-register the parameter so later loads write real
            # storage. `_streaming_patched` lets qwen3_5.load_weights route
            # tuple shard_id to us instead of bypassing via weight_loader_v2.
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0
                materialized = ModelWeightParameter(
                    data=torch.empty_like(
                        layer.weight, device=layer._load_device,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                _copy_missing_attrs(layer.weight, materialized)
                materialized._streaming_patched = True
                layer.register_parameter("weight", materialized)

            # Always refresh to the live parameter after potential
            # re-registration; stale references can come from params_dict
            # snapshots captured before materialization.
            param = layer.weight

            # Detect tuple shard_id (GDN fused projections use
            # MergedColumnParallelLinear.weight_loader_v2 which accepts
            # tuple). The INT4 v1 weight_loader does not handle tuple, so
            # we dispatch through layer.weight_loader_v2 directly (Path X).
            shard_id = kwargs.get("loaded_shard_id")
            if shard_id is None:
                shard_id = kwargs.get("shard_id")
            if shard_id is None and args:
                shard_id = args[0]

            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                if (isinstance(shard_id, tuple)
                        and hasattr(layer, "weight_loader_v2")):
                    layer.weight_loader_v2(param, loaded_weight, shard_id)
                    res = None
                else:
                    res = orig_weight_loader(
                        param, loaded_weight, *args, **kwargs,
                    )
            layer._loaded_numel += copy_numel_counter.copied_numel

            # Fully loaded: quantize in place and release the BF16 copy
            # before the next layer begins loading.
            if layer._loaded_numel >= layer.weight.numel():
                _quantize_linear_int4_inplace(layer)
                del layer._loaded_numel
                if hasattr(layer, "_load_device"):
                    del layer._load_device
                layer._already_called_process_weights_after_loading = True

            return res

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=weight_dtype,
                device="meta",
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        weight._streaming_patched = True
        layer.register_parameter("weight", weight)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # The same with the GPTQ's linear method by IPEX
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size, ))

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading",
                   False):
            return

        # Fallback for layers that never saw their streaming weight_loader
        # (e.g. tied lm_head, skipped vision branches). Materialize zeros so
        # downstream quantization does not read uninitialized memory.
        if (hasattr(layer, "weight") and layer.weight is not None
                and layer.weight.device == torch.device("meta")):
            device = getattr(layer, "_load_device", torch.device("xpu"))
            wl = getattr(layer.weight, "weight_loader",
                         lambda p, w, *a, **k: None)
            materialized = ModelWeightParameter(
                data=torch.zeros_like(layer.weight, device=device),
                input_dim=1, output_dim=0, weight_loader=wl)
            layer.register_parameter("weight", materialized)
            if hasattr(layer, "_load_device"):
                del layer._load_device

        _quantize_linear_int4_inplace(layer)


def _quantize_linear_int4_inplace(layer: Module) -> None:
    """Quantize a Linear layer's BF16/FP16 weight to INT4 and wire up IPEX.

    Works for both paths:
    - Legacy CPU offload: ``layer.weight`` is BF16 on CPU (or temporarily
      moved to XPU by ``device_loading_context``).
    - Streaming: ``layer.weight`` is BF16 on XPU; we copy to CPU for the
      CPU-only ``ggml_quantize_tensor`` and release XPU BF16 immediately.
    """
    bf16 = layer.weight.data
    out_features = bf16.shape[0]
    in_features = bf16.shape[1]

    if bf16.device.type == "xpu" and _use_xpu_quant():
        # XPU streaming path: quantize directly on device — no CPU round trip.
        # This keeps XPU peak close to OFFLOAD=1 because we never hold both
        # BF16 + FP32 nor do a D→H bounce of the full side.
        qweight_xpu, scale_xpu = _xpu_q4_0_quantize(bf16)
        layer.weight = None
        del bf16
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
    else:
        if bf16.device.type == "xpu":
            # CPU-fallback streaming path: pull to CPU before BF16→FP32 upcast
            # so the FP32 temporary never lives on XPU. Doing
            # ``bf16.float().cpu()`` instead would briefly hold BF16 + FP32
            # (= 3× BF16) on XPU for this layer.
            bf16_cpu = bf16.cpu()
            layer.weight = None
            del bf16
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass
            weight_cpu = bf16_cpu.float().contiguous()
            del bf16_cpu
        else:
            weight_cpu = bf16.float().contiguous()

        qweight = torch.zeros(
            (out_features, in_features // QK4_PACK_FACTOR),
            dtype=torch.int32, device="cpu",
        )
        scale = torch.zeros(
            (out_features, in_features // QK4_GROUP_SIZE),
            dtype=torch.float16, device="cpu",
        )
        qweight, scale = ggml_quantize_tensor(
            weight_cpu, qweight, scale, out_features, in_features,
            block_size=QK4_GROUP_SIZE, transpose=False,
        )
        del weight_cpu

        # Move quantized weights to XPU. IPEX needs [K/8, N] (transposed);
        # ESIMD needs [N, K/2] uint8.
        qweight_xpu = qweight.to("xpu")
        scale_xpu = scale.to("xpu")
        del qweight, scale

    layer.weight_esimd = Parameter(
        qweight_xpu.view(torch.uint8), requires_grad=False)
    layer.scale_esimd = Parameter(scale_xpu, requires_grad=False)
    layer.weight = Parameter(qweight_xpu.t(), requires_grad=False)
    layer.weight_scale = Parameter(scale_xpu.t(), requires_grad=False)

    try:
        import intel_extension_for_pytorch as ipex
        if version.parse(ipex.__version__) < version.parse(MIN_IPEX_VERSION):
            raise ImportError(
                f"intel_extension_for_pytorch version is wrong. "
                f"Current: {ipex.__version__}, Required: >={MIN_IPEX_VERSION}")
    except ImportError as err:
        raise ImportError(
            "Please install "
            f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
            f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
            " to use IPEX-AWQ linear method.") from err

    lowp_mode = ipex.quantization.WoqLowpMode.INT8
    weight_dtype = ipex.quantization.WoqWeightDtype.INT4
    act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK
    qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
        weight_dtype=weight_dtype,
        lowp_mode=lowp_mode,
        act_quant_mode=act_quant_mode,
        group_size=QK4_GROUP_SIZE,
    )
    layer.ipex_output_size = layer.weight.shape[-1]
    g_idx = None
    layer.ipex_qlinear = ipex.llm.quantization.woq_linear. \
        IPEXWeightOnlyQuantizedLinear.from_weight(
        layer.weight,
        layer.weight_scale,
        torch.tensor([8], device=layer.weight.device, dtype=torch.int8),
        layer.weight.size(0),
        layer.ipex_output_size,
        qconfig=qconfig,
        g_idx=g_idx,
        bias=None,
        group_size=QK4_GROUP_SIZE,
        # For GPTQ layout
        quant_method=0,
    )


def _to_cutlass_nmajor(qweight_nk: torch.Tensor):
    """Repack MoE expert weights from ggml int32 to CUTLASS uint8 N-major.

    Input:  [E, N, K/8] int32 — 8 unsigned nibbles (0-15) per word
    Output: [E, N, K/2] uint8 — byte-level view, same nibble order

    implement_zp conversion is left to xpu_fused_moe (auto on first call).
    """
    E, N, Kp8 = qweight_nk.shape
    return qweight_nk.view(torch.uint8).reshape(E, N, Kp8 * 4).contiguous()


def _esimd_prefill_moe_apply(layer, router, x, router_logits):
    """End-to-end MoE prefill path for USE_ESIMD_MOE_PREFILL=1.

    Hybrid kernel selection: ESIMD for topk and silu_mul (faster at all M),
    XPU-K for scatter/remap and gather (faster at all M, especially small M).
    CUTLASS grouped GEMM for the two INT4 GEMM stages.
    """
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as _ops
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    import vllm_xpu_kernels._moe_C  # noqa: F401

    M, H = x.shape
    TK = int(layer.top_k)
    E  = int(layer.num_experts)
    two_I = layer.w13_weight.size(1)
    I = two_I // 2
    total = M * TK

    # 1. TopK -- ESIMD (2-3x faster than XPU-K at all M)
    topk_weights, topk_ids = _ops.moe_topk_softmax(router_logits, TK, E)

    # 2. Scatter/remap -- XPU-K (20-40% faster, fused scatter+offset)
    topk_ids_i64 = topk_ids.to(torch.int64)
    x_perm = torch.empty(total, H, dtype=x.dtype, device=x.device)
    efto = torch.zeros(E + 1, dtype=torch.int64, device=x.device)
    u2p = torch.empty(M, TK, dtype=torch.int32, device=x.device)
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=x, hidden_states_scales=None,
        remapped_hidden_states=x_perm, remapped_hidden_states_scales=None,
        expert_map=None, expert_first_token_offset=efto,
        unpermuted_row_to_permuted_row=u2p,
        topk_ids=topk_ids_i64,
        total_experts_num=E, local_experts_num=E)

    # 3. GEMM1 -- CUTLASS grouped GEMM (W13, INT4 N-major)
    gate_up_perm = torch.empty(total, two_I, dtype=x.dtype, device=x.device)
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=x_perm, ptr_B=layer.w13_weight,
        ptr_scales=layer.w13_scales, ptr_bias=None,
        ptr_D=gate_up_perm, expert_first_token_offset=efto,
        N=two_I, K=H, num_experts=E,
        is_B_int4=True, is_B_mxfp4=False)

    # 4. SiLU*Mul -- ESIMD (faster than XPU-K at M<=2048)
    inter_perm = _ops.moe_prefill_silu_mul_forward(gate_up_perm)

    # 5. GEMM2 -- CUTLASS grouped GEMM (W2, INT4 N-major)
    down_perm = torch.empty(total, H, dtype=x.dtype, device=x.device)
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=inter_perm, ptr_B=layer.w2_weight,
        ptr_scales=layer.w2_scales, ptr_bias=None,
        ptr_D=down_perm, expert_first_token_offset=efto,
        N=H, K=I, num_experts=E,
        is_B_int4=True, is_B_mxfp4=False)

    # 6. Gather -- XPU-K (10x faster than ESIMD at small M)
    topk_weights_f32 = topk_weights.float()
    out = torch.empty(M, H, dtype=x.dtype, device=x.device)
    torch.ops._moe_C.moe_gather(
        out, down_perm, topk_weights_f32, u2p, efto, E)
    return out


def _quantize_moe_int4_side_inplace(layer: Module, which: str) -> None:
    """Quantize a single side (w13 or w2) of an MoE layer to INT4.

    Loads BF16 weights (wherever they currently live), sends them to CPU
    one side at a time for ``ggml_quantize_tensor``, moves the quantized
    int32 qweight + fp16 scales to XPU, and registers them back on the
    layer under ``<side>_weight`` / ``<side>_scales``. The BF16 source is
    released before the XPU allocation to minimize peak memory.
    """
    assert which in ("w13", "w2"), f"unknown side {which!r}"

    d_model = layer.hidden_size
    d_ff = layer.d_ff
    num_loop = getattr(layer, "local_num_experts", layer.num_experts)

    weight_name = f"{which}_weight"
    scale_name = f"{which}_scales"

    src = getattr(layer, weight_name).data
    if src.dim() != 3:
        raise RuntimeError(
            f"expected 3D MoE weight for {weight_name}, got {src.shape}")

    if which == "w13":
        out_features, in_features = 2 * d_ff, d_model
    else:
        out_features, in_features = d_model, d_ff

    E_all = src.shape[0]

    # XPU streaming fast path: quantize each expert directly on device, one at
    # a time, and drop its BF16 slice when done. Peak XPU footprint per layer
    # is now (accumulated int4) + 1 BF16 expert, not a full BF16 side.
    if src.device.type == "xpu" and _use_xpu_quant():
        qweight_xpu = torch.empty(
            E_all, out_features, in_features // QK4_PACK_FACTOR,
            dtype=torch.int32, device=src.device)
        scales_xpu = torch.empty(
            E_all, out_features, in_features // QK4_GROUP_SIZE,
            dtype=torch.float16, device=src.device)
        for e in range(num_loop):
            q_e, s_e = _xpu_q4_0_quantize(src[e])
            qweight_xpu[e].copy_(q_e)
            scales_xpu[e].copy_(s_e)
            del q_e, s_e
        # Experts outside [0, num_loop) are not local; their slices stay
        # zero-initialized (matches legacy path behavior).
        setattr(layer, weight_name, None)
        del src
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
        setattr(layer, weight_name, torch.nn.Parameter(
            qweight_xpu, requires_grad=False))
        setattr(layer, scale_name, torch.nn.Parameter(
            scales_xpu, requires_grad=False))
        return

    # CPU fallback path: keep the existing D→H + ggml + H→D behavior.
    # Streaming path: weights live on XPU. Move BF16 to CPU once (BF16 keeps
    # the memory footprint half of FP32), release XPU storage immediately,
    # then quantize per-expert with transient FP32 buffers — matching the
    # legacy path's per-expert peak shape.
    # Legacy path: weights already on CPU as BF16; alias directly and drop
    # the parameter slot so we can overwrite with the quantized result.
    if src.device.type == "xpu":
        src_cpu = src.cpu().contiguous()
        setattr(layer, weight_name, None)
        del src
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
    else:
        src_cpu = src

    E = src_cpu.shape[0]
    qweight = torch.empty(
        E, out_features, in_features // QK4_PACK_FACTOR,
        dtype=torch.int32, device="cpu",
    )
    scales = torch.empty(
        E, out_features, in_features // QK4_GROUP_SIZE,
        dtype=torch.float16, device="cpu",
    )

    def _quantize_expert(e):
        # Per-expert FP32 copy so peak CPU footprint is BF16_full +
        # FP32_one_expert + qweight_accum + scales_accum.
        expert_fp32 = src_cpu[e].float().contiguous()
        q_buf = torch.zeros(
            (out_features, in_features // QK4_PACK_FACTOR),
            dtype=torch.int32, device="cpu",
        )
        s_buf = torch.zeros(
            (out_features, in_features // QK4_GROUP_SIZE),
            dtype=torch.float16, device="cpu",
        )
        q, s = ggml_quantize_tensor(
            expert_fp32, q_buf, s_buf, out_features, in_features,
            block_size=QK4_GROUP_SIZE, transpose=False,
        )
        qweight[e].copy_(q)
        scales[e].copy_(s)

    with ThreadPoolExecutor() as executor:
        list(executor.map(_quantize_expert, range(num_loop)))

    # Release the BF16 source now that all experts are quantized. For the
    # legacy path this also drops the original parameter storage.
    del src_cpu
    if getattr(layer, weight_name, None) is not None:
        setattr(layer, weight_name, None)

    qweight_xpu = qweight.to("xpu")
    scales_xpu = scales.to("xpu")
    del qweight, scales

    setattr(layer, weight_name, torch.nn.Parameter(
        qweight_xpu, requires_grad=False))
    setattr(layer, scale_name, torch.nn.Parameter(
        scales_xpu, requires_grad=False))


def _setup_moe_int4_kernel(layer: Module, method) -> None:
    """Finalize MoE INT4 layout: CUTLASS N-major repack (optional) and IPEX
    fusion kernel. Idempotent — safe to call twice.
    """
    if getattr(layer, "_moe_int4_kernel_ready", False):
        return

    import intel_extension_for_pytorch as ipex

    use_esimd = os.environ.get("USE_ESIMD_MOE_PREFILL", "1") == "1"
    method._use_esimd_prefill = use_esimd

    if use_esimd:
        from vllm_xpu_kernels.fused_moe_interface import implement_zp
        E = layer.num_experts

        w13_qweight = _to_cutlass_nmajor(layer.w13_weight.data)
        w2_qweight = _to_cutlass_nmajor(layer.w2_weight.data)
        w13_tmp = torch.empty_like(w13_qweight)
        w2_tmp = torch.empty_like(w2_qweight)
        for i in range(E):
            w13_tmp[i] = implement_zp(w13_qweight[i])
            w2_tmp[i] = implement_zp(w2_qweight[i])
        layer.w13_weight = torch.nn.Parameter(
            w13_tmp.contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(
            w2_tmp.contiguous(), requires_grad=False)
        layer.ipex_fusion = None
        # Mark as already converted so xpu_fused_moe skips implement_zp.
        layer.w13_weight.xpu_fused_moe = True
        layer.w2_weight.xpu_fused_moe = True
    else:
        layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
            layer.w13_weight,
            layer.w2_weight,
            w1_scale_inv=layer.w13_scales,
            w2_scale_inv=layer.w2_scales,
            is_int4=True,
        )

    layer._moe_int4_kernel_ready = True


class XPUGPTQInt4LinearMoEMethod(FusedMoEMethodBase):
    def __init__(
        self,
        quant_config: SymInt4Config,
        moe: "FusedMoEConfig",
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.moe_config = moe
        # Ensure lib is loaded
        _get_quant_lib()

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        tp_size = get_tensor_model_parallel_world_size()
        # Round up intermediate_size_per_partition for IPEX GatedMLPMOE alignment.
        # The alignment value must match qwen3_5.py's interleaved padding logic.
        if tp_size == 4:
            intermediate_size_per_partition = round_up(intermediate_size_per_partition, 256)
        elif tp_size == 8:
            if self.moe_config.hidden_dim == 2048:
                intermediate_size_per_partition = round_up(intermediate_size_per_partition, 128)
            else:
                # For hidden_dim=3072 (122B-A10B), 4096 (235B-A22B), etc.
                intermediate_size_per_partition = round_up(intermediate_size_per_partition, 256)
        elif tp_size == 16:
            intermediate_size_per_partition = round_up(intermediate_size_per_partition, 128)
        layer.d_ff = intermediate_size_per_partition

        if VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT:
            # Legacy path: full BF16 buffers on CPU; device_loading_context
            # will move them onto XPU prior to process_weights_after_loading.
            weight_device = "cpu"
        else:
            # Streaming path: meta placeholders, materialized JIT on first
            # shard and quantized immediately when each side is full.
            weight_device = "meta"
            layer._load_device = torch.get_default_device()

        orig_weight_loader = extra_weight_attrs.get("weight_loader")

        # w13 shape: [d_ff * 2, d_model]
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
                device=weight_device,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)

        # w2 shape: [d_model, d_ff]
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
                device=weight_device,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)

        if VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT:
            set_weight_attrs(w13_weight, extra_weight_attrs)
            set_weight_attrs(w2_weight, extra_weight_attrs)
            return

        # Streaming: install a patched weight_loader that (a) materializes
        # w13 / w2 on the target device the first time its respective shard
        # arrives, (b) counts loaded elements independently, and (c) kicks
        # off per-side INT4 quantization the moment each side is fully
        # loaded — releasing the BF16 intermediate before the next layer
        # starts loading. Mirrors the FP8 MoE streaming path.
        def patched_moe_weight_loader(param, loaded_weight, *args, **kwargs):
            shard_id = kwargs.get("shard_id")
            if shard_id is None and len(args) >= 2:
                shard_id = args[1]
            is_w13 = shard_id in ("w1", "w3")

            if is_w13 and not hasattr(layer, "_w13_materialized"):
                layer._w13_materialized = True
                layer._w13_loaded_numel = 0
                new_w13 = torch.nn.Parameter(
                    torch.empty_like(
                        layer.w13_weight, device=layer._load_device,
                    ),
                    requires_grad=False,
                )
                new_attrs = dict(extra_weight_attrs)
                new_attrs["weight_loader"] = patched_moe_weight_loader
                set_weight_attrs(new_w13, new_attrs)
                layer.register_parameter("w13_weight", new_w13)

            if (not is_w13) and not hasattr(layer, "_w2_materialized"):
                layer._w2_materialized = True
                layer._w2_loaded_numel = 0
                new_w2 = torch.nn.Parameter(
                    torch.empty_like(
                        layer.w2_weight, device=layer._load_device,
                    ),
                    requires_grad=False,
                )
                new_attrs = dict(extra_weight_attrs)
                new_attrs["weight_loader"] = patched_moe_weight_loader
                set_weight_attrs(new_w2, new_attrs)
                layer.register_parameter("w2_weight", new_w2)

            if (hasattr(layer, "_w13_materialized")
                    and hasattr(layer, "_w2_materialized")
                    and hasattr(layer, "_load_device")):
                del layer._load_device

            param = layer.w13_weight if is_w13 else layer.w2_weight

            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                res = orig_weight_loader(
                    param, loaded_weight, *args, **kwargs,
                )

            if is_w13:
                layer._w13_loaded_numel += copy_numel_counter.copied_numel
                if layer._w13_loaded_numel >= layer.w13_weight.numel():
                    _quantize_moe_int4_side_inplace(layer, "w13")
                    del layer._w13_loaded_numel
            else:
                layer._w2_loaded_numel += copy_numel_counter.copied_numel
                if layer._w2_loaded_numel >= layer.w2_weight.numel():
                    _quantize_moe_int4_side_inplace(layer, "w2")
                    del layer._w2_loaded_numel

            # When both sides are fully quantized, set up the runtime kernel
            # and short-circuit the later process_weights_after_loading.
            if (not hasattr(layer, "_w13_loaded_numel")
                    and not hasattr(layer, "_w2_loaded_numel")
                    and getattr(layer, "_w13_materialized", False)
                    and getattr(layer, "_w2_materialized", False)):
                _setup_moe_int4_kernel(layer, self)
                layer._already_called_process_weights_after_loading = True

            return res

        patched_attrs = dict(extra_weight_attrs)
        patched_attrs["weight_loader"] = patched_moe_weight_loader
        set_weight_attrs(w13_weight, patched_attrs)
        set_weight_attrs(w2_weight, patched_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading",
                   False):
            # Streaming already finished quantizing and setting up the
            # kernel. Still guarantee idempotent kernel setup in case the
            # caller invoked this after a state reload.
            _setup_moe_int4_kernel(layer, self)
            return

        assert layer.hidden_size % QK4_PACK_FACTOR == 0 \
            and layer.d_ff % QK4_PACK_FACTOR == 0, \
            "INT4 packing requires feature dims % 8 == 0"
        assert layer.hidden_size % QK4_GROUP_SIZE == 0 \
            and layer.d_ff % QK4_GROUP_SIZE == 0, \
            f"group_size={QK4_GROUP_SIZE} requires dims % {QK4_GROUP_SIZE} == 0"

        # meta fallback for layers whose weight_loader was never invoked
        # (e.g. tied / skipped branches).
        for name in ("w13_weight", "w2_weight"):
            p = getattr(layer, name, None)
            if p is not None and p.device == torch.device("meta"):
                dev = getattr(layer, "_load_device", torch.device("xpu"))
                setattr(layer, name, torch.nn.Parameter(
                    torch.zeros_like(p, device=dev), requires_grad=False,
                ))
        if hasattr(layer, "_load_device"):
            del layer._load_device

        # Legacy (OFFLOAD=1) path: w13 / w2 still hold BF16. Quantize each
        # side (releasing its BF16 before moving on), then build the kernel.
        if layer.w13_weight.dtype != torch.int32:
            _quantize_moe_int4_side_inplace(layer, "w13")
        if layer.w2_weight.dtype != torch.int32:
            _quantize_moe_int4_side_inplace(layer, "w2")
        _setup_moe_int4_kernel(layer, self)


    def apply(
        self,
        layer: FusedMoE,
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self._use_esimd_prefill:
            import os
            if os.environ.get("DISABLE_CUTLASS_PREFILL", "0") == "1":
                pass  # fall through to ipex_fusion below
            else:
                return _esimd_prefill_moe_apply(layer, router, x, router_logits)

        res = layer.ipex_fusion(
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            topk_group=layer.topk_group,
            num_expert_group=layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
            scoring_func=layer.scoring_func,
        )
        return res
