# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )

elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops

    reshape_and_cache_flash = ipex_ops.reshape_and_cache_flash
    get_scheduler_metadata = ipex_ops.get_scheduler_metadata

    # Supported values: "cutlass" (default), "xetla" (ipex fallback)
    _XPU_FLASH_BACKEND = os.environ.get(
        "VLLM_XPU_FLASH_ATTN_BACKEND", "cutlass").lower()

    if _XPU_FLASH_BACKEND == "cutlass":
        try:
            from vllm_xpu_kernels import (
                flash_attn_varlen_func as _cutlass_flash_attn_varlen_func,
            )
            logger.info(
                "Using cutlass flash attention backend for XPU (TTFT).")
        except ImportError as e:
            logger.warning(
                "VLLM_XPU_FLASH_ATTN_BACKEND=cutlass but "
                "vllm_xpu_kernels not available: %s. "
                "Falling back to xetla (ipex).", e)
            _XPU_FLASH_BACKEND = "xetla"

    if _XPU_FLASH_BACKEND == "cutlass":

        def flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=None,
            causal=False,
            out=None,
            block_table=None,
            alibi_slopes=None,
            window_size=None,
            softcap=0.0,
            seqused_k=None,
            cu_seqlens_k=None,
            dropout_p=0.0,
            scheduler_metadata=None,
            fa_version=2,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            num_splits=0,
            s_aux=None,
            return_softmax_lse=False,
        ):
            result = _cutlass_flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_k=cu_seqlens_k,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap if softcap else 0.0,
                alibi_slopes=alibi_slopes,
                block_table=block_table,
                return_softmax_lse=return_softmax_lse,
                out=out,
                k_descale=k_descale,
                v_descale=v_descale,
                s_aux=s_aux,
                num_splits=num_splits,
            )
            if return_softmax_lse:
                return result
            if isinstance(result, tuple):
                return result[0]
            return result

    else:
        flash_attn_varlen_func = ipex_ops.flash_attn_varlen_func

elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Rocm platform requires upstream flash-attn "
            "to be installed. Please install flash-attn first."
        ) from e


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    # import here to avoid circular dependencies
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        fa_version = (
            3 if (device_capability.major == 9 and is_fa_version_supported(3)) else 2
        )

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major == 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def flash_attn_supports_fp8() -> bool:
    if current_platform.is_xpu():
        return True
    return (
        get_flash_attn_version() == 3
        and current_platform.is_device_capability_family(90)
    )


def flash_attn_supports_quant_query_input() -> bool:
    return not current_platform.is_xpu()


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    else:
        return get_flash_attn_version() == 3


def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            return is_fa_version_supported(
                3
            ) and current_platform.is_device_capability_family(90)
        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return current_platform.is_cuda() or current_platform.is_xpu()
