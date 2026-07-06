# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

import os
import time
import typing
from collections.abc import Callable, Iterable

import torch
from custom_esimd_kernels_vllm import esimd_qkv_split_norm_rope
from custom_esimd_kernels_vllm import esimd_gdn_conv_fused, esimd_gdn_conv_fused_seq
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from vllm.utils.math_utils import round_up
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    mamba_v2_sharded_weight_loader,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.forward_context import get_forward_context
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
    def __init__(
        self,
        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextGatedDeltaNet, self).__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # Merge QKV+Z into single projection for better performance
        self.in_proj_qkvz = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )

        # Merge B+A into single projection
        self.in_proj_ba = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.num_v_heads, self.num_v_heads],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ],
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=current_platform.current_device(),
            dtype=config.dtype,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.conv_bias_zeros = torch.zeros(
            self.conv_dim // self.tp_size,
            dtype=torch.float16,
            device=current_platform.current_device(),
        )

        # Pre-allocate decode buffers to avoid per-forward allocation overhead
        _dev = current_platform.current_device()
        # qkvz buffer: [1, (key_dim * 2 + value_dim * 2) // tp_size]
        self._decode_qkvz_buf = torch.empty(
            (1, (self.key_dim * 2 + self.value_dim * 2) // self.tp_size),
            dtype=torch.float16, device=_dev)
        # ba buffer: [1, (num_v_heads * 2) // tp_size]
        self._decode_ba_buf = torch.empty(
            (1, (self.num_v_heads * 2) // self.tp_size),
            dtype=torch.float16, device=_dev)
        self._decode_attn_out_buf = torch.zeros(
            (1, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=torch.float16, device=_dev)
        self._decode_z_out_buf = torch.empty(
            (1, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=torch.float16, device=_dev)

        # Pre-cache values that don't change between forward calls
        self._cached_conv_weights = None  # lazily cached after first forward
        self._cached_nk_tp = self.num_k_heads // self.tp_size
        self._cached_nv_tp = self.num_v_heads // self.tp_size
        self._cached_attn_scale = float(self.head_k_dim ** -0.5)

        # FP8 ESIMD projection kernels (GEMV/GEMM for input projection)
        self._is_fp8 = quant_config is not None and quant_config.get_name() == "fp8"
        # Note: can't use hasattr(weight_scale) here — it's added later by
        # process_weights_after_loading for runtime FP8 quantization.
        self._use_esimd_proj = self._is_fp8

        # INT4 detection
        self._is_sym_int4 = (
            quant_config is not None
            and quant_config.get_name() == "sym_int4"
        )
        # esimd_gemm_int4_pgrp requires N%16==0; in_proj_ba may not satisfy
        # this at higher TP (e.g. 27B TP=4 → N=24).
        _ba_n = (self.num_v_heads * 2) // self.tp_size
        self._int4_gemm_ok = self._is_sym_int4 and (_ba_n % 16 == 0)

        # Fused norm + out_proj GEMV for decode (FP8 or INT4)
        # Note: can't use hasattr(self.out_proj, 'weight_scale') here because
        # weight_scale is added later by process_weights_after_loading.
        self._use_fused_out_proj = self._is_fp8 or self._is_sym_int4
        self._decode_outproj_buf = torch.empty(
            (1, self.hidden_size),
            dtype=torch.float16, device=_dev)
        self._norm_weight_fp16 = None  # lazily cached after weights loaded
        # Disable lazy fp16 norm-weight caching only for 27B on TP=4 to work
        # around an XPU allocator bug that corrupts cached storage at certain
        # max_model_len values. Other configs keep the cache to avoid the
        # per-forward .half().clone() overhead.
        self._disable_norm_cache = (
            config.hidden_size == 5120 and self.tp_size == 4
        )

        # Pre-compute gather indices to convert GEMV output from
        # sequential [q|k|v|z] to GQA-interleaved [q_g0|k_g0|v_g0|z_g0|...]
        # This replaces the per-forward split+reshape+cat with a single gather.
        nk_tp = self._cached_nk_tp
        vpg = self.num_v_heads // self.num_k_heads
        dk = self.head_k_dim
        dv = self.head_v_dim
        qs = self.key_dim // self.tp_size
        ks = qs
        vs = self.value_dim // self.tp_size
        qkvs = qs + ks + vs  # offset where z starts
        idx_qkvz = []
        for g in range(nk_tp):
            idx_qkvz.extend(range(g * dk, (g + 1) * dk))
            idx_qkvz.extend(range(qs + g * dk, qs + (g + 1) * dk))
            idx_qkvz.extend(range(qs + ks + g * vpg * dv,
                                  qs + ks + (g + 1) * vpg * dv))
            idx_qkvz.extend(range(qkvs + g * vpg * dv,
                                  qkvs + (g + 1) * vpg * dv))
        self._gather_qkvz = torch.tensor(idx_qkvz, dtype=torch.long, device=_dev)

        nv_tp = self._cached_nv_tp
        idx_ba = []
        for g in range(nk_tp):
            idx_ba.extend(range(g * vpg, (g + 1) * vpg))
            idx_ba.extend(range(nv_tp + g * vpg, nv_tp + (g + 1) * vpg))
        self._gather_ba = torch.tensor(idx_ba, dtype=torch.long, device=_dev)

        # Pre-allocate BSZ>1 decode buffers (avoids per-forward torch.zeros/empty)
        _mb = int(os.environ.get("MAX_DECODE_BSZ", "64"))
        self._max_bsz = _mb
        self._m_qkvz = torch.empty(
            (_mb, (self.key_dim * 2 + self.value_dim * 2) // self.tp_size),
            dtype=torch.float16, device=_dev)
        self._m_ba = torch.empty(
            (_mb, (self.num_v_heads * 2) // self.tp_size),
            dtype=torch.float16, device=_dev)
        self._m_attn_out = torch.empty(
            (_mb, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=torch.float16, device=_dev)
        self._m_z = torch.empty(
            (_mb, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=torch.float16, device=_dev)
        self._m_outproj = torch.empty(
            (_mb, self.hidden_size), dtype=torch.float16, device=_dev)

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        raise NotImplementedError(
            "Qwen3.5 Series dont need to fix query key value ordering"
        )
    def fix_query_key_value_ordering(
        self,
        mixed_qkv,
        z,
        b,
        a,
    ):
        raise NotImplementedError(
            "Qwen3.5 Series dont need to fix query key value ordering"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        if current_platform.is_xpu():
            self.forward_xpu(hidden_states, output)
        else:
            self.forward_cuda(hidden_states, output)

    _attn_profile_data = {}
    _attn_profile_count = 0
    _PROFILE_ATTN = os.environ.get("PROFILE_ATTN", "0") == "1"

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass for XPU with three parts:
        1. Input projection (separate projections + rearrangement)
        2. Core attention (XPU kernel)
        3. Output projection
        """
        # Workaround: clone FP8 weight storage on first forward to prevent
        # corruption by gdn_attention XE2 chunk kernel OOB writes.
        # See PR description for root cause analysis.
        if not getattr(self, '_weights_cloned', False):
            self._weights_cloned = True
            if hasattr(self.out_proj, 'weight'):
                self.out_proj.weight.data = self.out_proj.weight.data.clone()
            if hasattr(self.in_proj_qkvz, 'weight'):
                self.in_proj_qkvz.weight.data = (
                    self.in_proj_qkvz.weight.data.clone()
                )
            if hasattr(self.in_proj_ba, 'weight'):
                self.in_proj_ba.weight.data = (
                    self.in_proj_ba.weight.data.clone()
                )

        _PROFILE_ATTN = Qwen3_5GatedDeltaNet._PROFILE_ATTN

        num_tokens = hidden_states.size(0)
        is_decode = (num_tokens == 1)

        # Resolve this layer's attn_metadata up front so Part 1 can pick the
        # right projected-states layout (sequential for ESIMD GDN vs
        # interleaved for the C++ prefill op).
        # forward_context = get_forward_context()
        # attn_metadata = forward_context.attn_metadata
        # if attn_metadata is not None:
        #     attn_metadata = attn_metadata[self.prefix]
        # _use_esimd_gdn = is_decode or (
        #     attn_metadata is not None
        #     and attn_metadata.num_prefills == 0
        #     and attn_metadata.num_decodes > 0
        #     and attn_metadata.num_decodes <= 128
        # )

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if _PROFILE_ATTN and is_decode:
            torch.xpu.synchronize()
            _t0 = time.perf_counter()
        if is_decode and self._use_esimd_proj:
            # M=1, FP8: single fused GEMV for qkvz + ba projections
            from custom_esimd_kernels_vllm import esimd_gemv_fp8_pert, esimd_gemv_fp8_pert_fused2
            qkvz_merged = self._decode_qkvz_buf
            ba_merged = self._decode_ba_buf
            esimd_gemv_fp8_pert_fused2(
                hidden_states,
                self.in_proj_qkvz.weight,
                self.in_proj_qkvz.weight_scale,
                qkvz_merged,
                self.in_proj_ba.weight,
                self.in_proj_ba.weight_scale,
                ba_merged,
            )
            projected_states_qkvz = qkvz_merged
            projected_states_ba = ba_merged
        elif is_decode and self._is_sym_int4:
            from custom_esimd_kernels_vllm import esimd_gemv_int4_fused2
            qkvz_merged = self._decode_qkvz_buf
            ba_merged = self._decode_ba_buf
            esimd_gemv_int4_fused2(
                hidden_states,
                self.in_proj_qkvz.weight_esimd,
                self.in_proj_qkvz.scale_esimd,
                qkvz_merged,
                self.in_proj_ba.weight_esimd,
                self.in_proj_ba.scale_esimd,
                ba_merged,
            )
            projected_states_qkvz = qkvz_merged
            projected_states_ba = ba_merged
        elif is_decode:
            # M=1, non-FP8: standard Linear, keep sequential layout
            # (esimd_gdn_conv_fused_seq expects sequential [q|k|v|z])
            qkvz_merged, _ = self.in_proj_qkvz(hidden_states)
            ba_merged, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = qkvz_merged
            projected_states_ba = ba_merged
        elif num_tokens <= 64 and self._use_esimd_proj:
            # M=2-64, FP8: ESIMD GEMM.
            from custom_esimd_kernels_vllm import esimd_gemm_fp8_pert
            if num_tokens <= self._max_bsz:
                qkvz_merged = self._m_qkvz[:num_tokens]
                ba_merged = self._m_ba[:num_tokens]
            else:
                N_qkvz = self.in_proj_qkvz.weight.shape[0]
                N_ba = self.in_proj_ba.weight.shape[0]
                qkvz_merged = torch.empty(
                    (num_tokens, N_qkvz), dtype=torch.float16, device=hidden_states.device)
                ba_merged = torch.empty(
                    (num_tokens, N_ba), dtype=torch.float16, device=hidden_states.device)
            esimd_gemm_fp8_pert(
                hidden_states, self.in_proj_qkvz.weight,
                self.in_proj_qkvz.weight_scale, qkvz_merged)
            esimd_gemm_fp8_pert(
                hidden_states, self.in_proj_ba.weight,
                self.in_proj_ba.weight_scale, ba_merged)
            projected_states_qkvz = qkvz_merged[:, self._gather_qkvz]
            projected_states_ba = ba_merged[:, self._gather_ba]
        elif num_tokens <= 64 and self._int4_gemm_ok and num_tokens <= self._max_bsz:
            # M=2-64, INT4: ESIMD DPAS GEMM (esimd_gemm_int4_pgrp).
            # Kernel requires N % 16 == 0 and K % 128 == 0.
            from custom_esimd_kernels_vllm import esimd_gemm_int4_pgrp
            qkvz_merged = self._m_qkvz[:num_tokens]
            ba_merged = self._m_ba[:num_tokens]
            esimd_gemm_int4_pgrp(
                hidden_states,
                self.in_proj_qkvz.weight_esimd,
                self.in_proj_qkvz.scale_esimd,
                qkvz_merged)
            esimd_gemm_int4_pgrp(
                hidden_states,
                self.in_proj_ba.weight_esimd,
                self.in_proj_ba.scale_esimd,
                ba_merged)
            projected_states_qkvz = qkvz_merged[:, self._gather_qkvz]
            projected_states_ba = ba_merged[:, self._gather_ba]
        else:
            # Fallback: standard Linear.
            # Same layout choice as above.
            qkvz_merged, _ = self.in_proj_qkvz(hidden_states)
            ba_merged, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = qkvz_merged[:, self._gather_qkvz]
            projected_states_ba = ba_merged[:, self._gather_ba]

        if _PROFILE_ATTN and is_decode:
            torch.xpu.synchronize()
            _t_proj = (time.perf_counter() - _t0) * 1e6

        # ============================================================
        # Part 2: Core Attention (XPU Kernel)
        # ============================================================
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if is_decode:
            core_attn_out = self._decode_attn_out_buf
            z_out = self._decode_z_out_buf
        elif num_tokens <= self._max_bsz:
            core_attn_out = self._m_attn_out[:num_tokens]
            z_out = self._m_z[:num_tokens]
        else:
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z_out = torch.empty_like(core_attn_out)
        if attn_metadata is not None:
            attn_metadata = attn_metadata[self.prefix]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            conv_state = self_kv_cache[0]
            ssm_state = self_kv_cache[1]

            # Cache conv_weights view (shape doesn't change)
            if self._cached_conv_weights is None:
                self._cached_conv_weights = self.conv1d.weight.view(
                    self.conv1d.weight.size(0), self.conv1d.weight.size(2)
                )
            conv_weights = self._cached_conv_weights

            if is_decode:
                # Decode: fused ESIMD kernel
                N_dec = attn_metadata.num_decodes
                state_idx = attn_metadata.non_spec_state_indices_tensor[:N_dec]

                if _PROFILE_ATTN:
                    torch.xpu.synchronize()
                    _t0 = time.perf_counter()

                esimd_gdn_conv_fused_seq(
                    projected_states_qkvz,
                    conv_state,
                    conv_weights,
                    self.conv_bias_zeros,
                    state_idx,
                    self.A_log,
                    self.dt_bias,
                    projected_states_ba,
                    ssm_state,
                    state_idx,
                    core_attn_out,
                    z_out,
                    N_dec,
                    self._cached_nk_tp,
                    self._cached_nv_tp,
                    self.head_k_dim,
                    self.head_v_dim,
                    self._cached_attn_scale,
                )

                if _PROFILE_ATTN:
                    torch.xpu.synchronize()
                    _t_gdn = (time.perf_counter() - _t0) * 1e6

            else:
                # Prefill: XPU C++ kernel
                spec_sequence_masks = attn_metadata.spec_sequence_masks
                if spec_sequence_masks is not None:
                    raise NotImplementedError(
                        "XPU gdn_attention does not yet support 'spec_sequence_masks'."
                    )

                torch.ops._xpu_C.gdn_attention(
                    core_attn_out,
                    z_out,
                    projected_states_qkvz,
                    projected_states_ba,
                    self.num_k_heads,
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                    conv_state=conv_state,
                    ssm_state=ssm_state,
                    conv_weights=conv_weights,
                    conv_bias=self.conv1d.bias,
                    activation=self.activation,
                    A_log=self.A_log.float(),
                    dt_bias=self.dt_bias,
                    num_prefills=attn_metadata.num_prefills,
                    num_decodes=attn_metadata.num_decodes,
                    has_initial_state=attn_metadata.has_initial_state,
                    non_spec_query_start_loc=attn_metadata.non_spec_query_start_loc,
                    non_spec_state_indices_tensor=attn_metadata.non_spec_state_indices_tensor,
                    num_actual_tokens=attn_metadata.num_actual_tokens,
                    tp_size=self.tp_size,
                    reorder_input=False,
                )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        if _PROFILE_ATTN and is_decode:
            torch.xpu.synchronize()
            _t0 = time.perf_counter()

        if is_decode and self._use_fused_out_proj:
            # Decode fast path: fused RMSNormGated + out_proj GEMV (single kernel)
            from vllm.distributed import tensor_model_parallel_all_reduce
            nv_tp = self._cached_nv_tp
            hv = self.head_v_dim
            if self._disable_norm_cache:
                # 27B+TP4: recompute every forward to dodge XPU allocator bug.
                norm_w_fp16 = self.norm.weight.data.half().clone().contiguous()
            else:
                if self._norm_weight_fp16 is None:
                    # .clone().contiguous() to own private storage — the XPU
                    # cache allocator has been observed reusing a shared fp16
                    # half() result's storage for other buffers mid-inference,
                    # corrupting the norm weight and producing NaN cascades.
                    self._norm_weight_fp16 = (
                        self.norm.weight.data.half().clone().contiguous()
                    )
                norm_w_fp16 = self._norm_weight_fp16
            if self._is_sym_int4:
                from custom_esimd_kernels_vllm import esimd_norm_gemv_int4_pert
                # .t() gives contiguous (N, K/8) layout for block_load
                esimd_norm_gemv_int4_pert(
                    core_attn_out.view(nv_tp, hv),
                    z_out.view(nv_tp, hv),
                    norm_w_fp16,
                    self.out_proj.weight_esimd.view(torch.int32),
                    self.out_proj.scale_esimd,
                    self._decode_outproj_buf,
                    nv_tp, hv, self.norm.eps,
                )
            else:
                from custom_esimd_kernels_vllm import esimd_norm_gemv_fp8_pert
                esimd_norm_gemv_fp8_pert(
                    core_attn_out.view(nv_tp, hv),
                    z_out.view(nv_tp, hv),
                    norm_w_fp16,
                    self.out_proj.weight,
                    self.out_proj.weight_scale,
                    self._decode_outproj_buf,
                    nv_tp, hv, self.norm.eps,
                )
            output[:1] = tensor_model_parallel_all_reduce(
                self._decode_outproj_buf)
        elif is_decode:
            # Decode non-FP8/sym_int4 fallback
            nv_tp = self._cached_nv_tp
            hv = self.head_v_dim
            core_attn_out_2d = core_attn_out.view(nv_tp, hv)
            z_out_2d = z_out.view(nv_tp, hv)
            core_attn_out_2d = self.norm(core_attn_out_2d, z_out_2d)
            core_attn_out_flat = core_attn_out_2d.view(1, nv_tp * hv)
            output[:1], _ = self.out_proj(core_attn_out_flat)
        elif num_tokens <= 64 and self._is_sym_int4:
            # BSZ=2-64: ESIMD RMSNormGated (single kernel, quantization-agnostic)
            from custom_esimd_kernels_vllm import esimd_rms_norm_gated
            from vllm.distributed import tensor_model_parallel_all_reduce
            if self._disable_norm_cache:
                # 27B+TP4: recompute every forward to dodge XPU allocator bug.
                norm_w_fp16 = self.norm.weight.data.half().clone().contiguous()
            else:
                if self._norm_weight_fp16 is None:
                    self._norm_weight_fp16 = (
                        self.norm.weight.data.half().clone().contiguous()
                    )
                norm_w_fp16 = self._norm_weight_fp16
            x_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z_flat = z_out.reshape(-1, z_out.shape[-1])
            normed = torch.empty_like(x_flat)
            esimd_rms_norm_gated(x_flat, z_flat, norm_w_fp16, normed, self.norm.eps)
            core_attn_out = normed.reshape(num_tokens, -1)
            if self._is_sym_int4:
                from custom_esimd_kernels_vllm import esimd_gemm_int4_pgrp
                out_buf = self._m_outproj[:num_tokens]
                esimd_gemm_int4_pgrp(
                    core_attn_out, self.out_proj.weight_esimd,
                    self.out_proj.scale_esimd, out_buf)
                output[:num_tokens] = tensor_model_parallel_all_reduce(out_buf)
            elif self._use_fused_out_proj:
                from custom_esimd_kernels_vllm import esimd_gemm_fp8_pert
                out_buf = self._m_outproj[:num_tokens]
                esimd_gemm_fp8_pert(
                    core_attn_out, self.out_proj.weight,
                    self.out_proj.weight_scale, out_buf)
                output[:num_tokens] = tensor_model_parallel_all_reduce(out_buf)
            else:
                core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], -1)
                output[:num_tokens], _ = self.out_proj(core_attn_out)
        else:
            # BSZ>64: fallback to PyTorch RMSNormGated
            z_shape_og = z_out.shape
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z_out = z_out.reshape(-1, z_out.shape[-1])
            core_attn_out = self.norm(core_attn_out, z_out)
            core_attn_out = core_attn_out.reshape(z_shape_og)
            core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
            output[:num_tokens], _ = self.out_proj(core_attn_out)

        if _PROFILE_ATTN and is_decode:
            torch.xpu.synchronize()
            _t_out = (time.perf_counter() - _t0) * 1e6

            key = f"L{self.layer_idx}_{self.prefix.split('.')[-1]}"
            if key not in Qwen3_5GatedDeltaNet._attn_profile_data:
                Qwen3_5GatedDeltaNet._attn_profile_data[key] = {"proj": [], "gdn": [], "out": [], "count": 0}
            pd = Qwen3_5GatedDeltaNet._attn_profile_data[key]
            pd["proj"].append(_t_proj)
            pd["gdn"].append(_t_gdn)
            pd["out"].append(_t_out)
            pd["count"] += 1

            Qwen3_5GatedDeltaNet._attn_profile_count += 1
            total_keys = len(Qwen3_5GatedDeltaNet._attn_profile_data)
            if total_keys > 0 and Qwen3_5GatedDeltaNet._attn_profile_count % (total_keys * 20) == 0:
                print(f"\n[ATTN_PROFILE] === Step {pd['count']} summary (last 20) ===")
                sum_proj = sum_gdn = sum_out = 0
                for k in sorted(Qwen3_5GatedDeltaNet._attn_profile_data.keys()):
                    d = Qwen3_5GatedDeltaNet._attn_profile_data[k]
                    n = min(len(d["proj"]), 20)
                    p = sum(d["proj"][-20:]) / n
                    g = sum(d["gdn"][-20:]) / n
                    o = sum(d["out"][-20:]) / n
                    sum_proj += p; sum_gdn += g; sum_out += o
                    print(f"[ATTN_PROFILE] {k:30s} proj={p:>7.1f}us  gdn={g:>7.1f}us  out={o:>7.1f}us  total={p+g+o:>7.1f}us")
                print(f"[ATTN_PROFILE] {'SUM':30s} proj={sum_proj:>7.1f}us  gdn={sum_gdn:>7.1f}us  out={sum_out:>7.1f}us  total={sum_proj+sum_gdn+sum_out:>7.1f}us")
                print(f"[ATTN_PROFILE] {'(ms)':30s} proj={sum_proj/1000:>7.2f}ms  gdn={sum_gdn/1000:>7.2f}ms  out={sum_out/1000:>7.2f}ms  total={(sum_proj+sum_gdn+sum_out)/1000:>7.2f}ms")
                print()

    def forward_xpu_with_precomputed_proj(
        self,
        projected_states_qkvz: torch.Tensor,
        projected_states_ba: torch.Tensor,
        output: torch.Tensor,
    ):
        """Full version matching forward_xpu decode path exactly."""

        # ---- Part 2: Core Attention (copied from forward_xpu decode) ----
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        core_attn_out = self._decode_attn_out_buf
        core_attn_out.zero_()
        z_out = self._decode_z_out_buf

        if attn_metadata is not None:
            attn_metadata = attn_metadata[self.prefix]

            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            conv_state = self_kv_cache[0]
            ssm_state = self_kv_cache[1]

            if self._cached_conv_weights is None:
                self._cached_conv_weights = self.conv1d.weight.view(
                    self.conv1d.weight.size(0), self.conv1d.weight.size(2)
                )

            N_dec = attn_metadata.num_decodes
            state_idx = attn_metadata.non_spec_state_indices_tensor[:N_dec]


            esimd_gdn_conv_fused_seq(
                projected_states_qkvz,
                conv_state,
                self._cached_conv_weights,
                self.conv_bias_zeros,
                state_idx,
                self.A_log,
                self.dt_bias,
                projected_states_ba,
                ssm_state,
                state_idx,
                core_attn_out,
                z_out,
                N_dec,
                self._cached_nk_tp,
                self._cached_nv_tp,
                self.head_k_dim,
                self.head_v_dim,
                self._cached_attn_scale,
            )

        # ---- Part 3: Output Projection (copied from forward_xpu decode) ----
        nv_tp = self._cached_nv_tp
        hv = self.head_v_dim
        if self._use_fused_out_proj:
            from vllm.distributed import tensor_model_parallel_all_reduce
            if self._disable_norm_cache:
                # 27B+TP4: recompute every forward to dodge XPU allocator bug.
                norm_w_fp16 = self.norm.weight.data.half().clone().contiguous()
            else:
                if self._norm_weight_fp16 is None:
                    # .clone().contiguous() to own private storage — the XPU
                    # cache allocator has been observed reusing a shared fp16
                    # half() result's storage for other buffers mid-inference,
                    # corrupting the norm weight and producing NaN cascades.
                    self._norm_weight_fp16 = (
                        self.norm.weight.data.half().clone().contiguous()
                    )
                norm_w_fp16 = self._norm_weight_fp16
            if self._is_sym_int4:
                from custom_esimd_kernels_vllm import esimd_norm_gemv_int4_pert
                esimd_norm_gemv_int4_pert(
                    core_attn_out.view(nv_tp, hv),
                    z_out.view(nv_tp, hv),
                    norm_w_fp16,
                    self.out_proj.weight_esimd.view(torch.int32),
                    self.out_proj.scale_esimd,
                    self._decode_outproj_buf,
                    nv_tp, hv, self.norm.eps,
                )
            else:
                from custom_esimd_kernels_vllm import esimd_norm_gemv_fp8_pert
                esimd_norm_gemv_fp8_pert(
                    core_attn_out.view(nv_tp, hv),
                    z_out.view(nv_tp, hv),
                    norm_w_fp16,
                    self.out_proj.weight,
                    self.out_proj.weight_scale,
                    self._decode_outproj_buf,
                    nv_tp, hv, self.norm.eps,
                )
            output[:1] = tensor_model_parallel_all_reduce(
                self._decode_outproj_buf)
        else:
            from vllm.distributed import tensor_model_parallel_all_reduce
            core_attn_out_2d = core_attn_out.view(nv_tp, hv)
            z_out_2d = z_out.view(nv_tp, hv)
            core_attn_out_2d = self.norm(core_attn_out_2d, z_out_2d)
            core_attn_out_flat = core_attn_out_2d.view(1, nv_tp * hv)
            output[:], _ = self.out_proj(core_attn_out_flat)


    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        qkvz_merged, _ = self.in_proj_qkvz(hidden_states)
        ba_merged, _ = self.in_proj_ba(hidden_states)

        # Split merged outputs
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        mixed_qkv = qkvz_merged[:, :qkv_size]
        z = qkvz_merged[:, qkv_size:qkv_size + self.value_dim // self.tp_size]
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                ),
            )

        # Disable lazy fp16 norm-weight caching only for 27B on TP=4 to work
        # around an XPU allocator bug that corrupts cached storage at certain
        # max_model_len values. Other configs keep the cache to avoid the
        # per-forward .half().clone() overhead.
        self._disable_norm_cache = (
            config.hidden_size == 5120
            and get_tensor_model_parallel_world_size() == 4
        )

        # Detect dense MLP with FP8 quantization for ESIMD fast path.
        # INT4 dense MLP does not currently have a working fast path (the
        # fused resadd+norm+GEMV kernel is numerically broken for the 27B
        # shape), so INT4 falls back to self.mlp(...) → IPEX qlinear.
        # This matches the pre-ec3f74a31 behavior for Qwen3.5-27B INT4.
        _quant_name = quant_config.get_name() if quant_config is not None else ""
        self._dense_mlp_fp8 = (
            isinstance(self.mlp, Qwen3NextMLP)
            and self.mlp.expert_gate is None
            and _quant_name == "fp8"
            and os.environ.get("DISABLE_ESIMD_DENSE", "0") != "1"
        )
        self._dense_mlp_is_int4 = False
        self._max_bsz = 0  # default when no FP8 dense MLP path; overwritten below
        if self._dense_mlp_fp8:
            _dev = current_platform.current_device()
            tp_size = get_tensor_model_parallel_world_size()
            _inter_tp = config.intermediate_size // tp_size
            _hidden = config.hidden_size
            # Pre-allocate decode buffers (bsz=1)
            self._dense_gate_up_buf = torch.empty(
                1, 2 * _inter_tp, dtype=torch.float16, device=_dev)
            self._dense_down_buf = torch.empty(
                1, _hidden, dtype=torch.float16, device=_dev)
            self._dense_normed_buf = torch.empty(
                1, _hidden, dtype=torch.float16, device=_dev)
            # BSZ>1 pre-allocated buffers
            _mb = int(os.environ.get("MAX_DECODE_BSZ", "64"))
            self._max_bsz = _mb
            self._m_gate_up = torch.empty(
                _mb, 2 * _inter_tp, dtype=torch.float16, device=_dev)
            self._m_down = torch.empty(
                _mb, _hidden, dtype=torch.float16, device=_dev)
            # Lazily cached after weights are loaded
            self._dense_post_norm_w_fp16 = None

        # Detect MoE with FP8/INT4 for ESIMD fused norm+router path
        self._moe_esimd_enabled = (
            isinstance(self.mlp, Qwen3NextSparseMoeBlock)
            and hasattr(self.mlp, 'gate')
            and _quant_name in ("fp8", "sym_int4")
        )
        self._moe_is_int4 = (_quant_name == "sym_int4") if self._moe_esimd_enabled else False
        if self._moe_esimd_enabled:
            _dev = current_platform.current_device()
            n_exp = self.mlp.gate.weight.shape[0]
            self._post_norm_w_fp16 = None  # lazily cached
            self._router_buf = torch.empty(
                1, n_exp, dtype=torch.float16, device=_dev)
            self._normed_buf = torch.empty(
                1, config.hidden_size, dtype=torch.float16, device=_dev)

        # Detect fused input_norm + input_proj opportunity (FP8/INT4, decode)
        self._fused_input_norm = (
            _quant_name in ("fp8", "sym_int4") and not self.layer_scale
            and os.environ.get("DISABLE_ESIMD_FUSED_INPUT", "0") != "1"
        )
        self._is_fused_int4 = (_quant_name == "sym_int4") if self._fused_input_norm else False
        if self._fused_input_norm:
            _dev = current_platform.current_device()
            _hidden = config.hidden_size
            _mb = int(os.environ.get("MAX_DECODE_BSZ", "64"))
            self._input_max_bsz = _mb
            self._input_norm_w_fp16 = None  # lazily cached
            if self.layer_type == "linear_attention":
                # GDN: norm + fused 2-GEMV (in_proj_qkvz + in_proj_ba)
                _qkvz_sz = self.linear_attn.in_proj_qkvz.weight.shape[0]
                _ba_sz = self.linear_attn.in_proj_ba.weight.shape[0]
                self._fused_qkvz_buf = torch.empty(
                    1, _qkvz_sz, dtype=torch.float16, device=_dev)
                self._fused_ba_buf = torch.empty(
                    1, _ba_sz, dtype=torch.float16, device=_dev)
                # BSZ>1 input proj buffers
                self._m_fused_qkvz = torch.empty(
                    _mb, _qkvz_sz, dtype=torch.float16, device=_dev)
                self._m_fused_ba = torch.empty(
                    _mb, _ba_sz, dtype=torch.float16, device=_dev)
            elif self.layer_type == "full_attention":
                # Full Attn: norm + qkv_proj GEMV
                _qkv_sz = self.self_attn.qkv_proj.weight.shape[0]
                self._fused_qkv_buf = torch.empty(
                    1, _qkv_sz, dtype=torch.float16, device=_dev)
                self._fused_normed_buf = torch.empty(
                    1, _hidden, dtype=torch.float16, device=_dev)
                # BSZ>1 input proj buffer
                self._m_fused_qkv = torch.empty(
                    _mb, _qkv_sz, dtype=torch.float16, device=_dev)
            # BSZ>1 output buffer (shared by both layer_types)
            self._m_attn_output = torch.empty(
                _mb, _hidden, dtype=torch.float16, device=_dev)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )

        is_padding_needed = False
        _pad_align = 0  # padding alignment for intermediate_size_per_partition
        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None or self.quant_config is not None:
            if quantization_config is not None:
                quant_method = quantization_config.get("quant_method", "").lower()
            elif self.quant_config is not None:
                quant_method = self.quant_config.get_name()
            tp_size = get_tensor_model_parallel_world_size()
            hidden_size = self.config.hidden_size if hasattr(self.config, 'hidden_size') else getattr(self.config, 'hidden_size', 0)
            if quant_method in ("sym_int4"):
                # Must match sym_int4.py XPUGPTQInt4LinearMoEMethod.create_weights round_up logic
                if tp_size == 4:
                    _pad_align = 256
                elif tp_size == 8:
                    _pad_align = 128 if hidden_size == 2048 else 256
                elif tp_size == 16:
                    _pad_align = 128
                if _pad_align > 0:
                    is_padding_needed = True
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]

                # Use weight_loader_v2 for tuple shard_id (GDN projections).
                # Exception: when FP8 streaming has patched param.weight_loader
                # (marked with _streaming_patched), dispatch through it so the
                # hook observes every shard (including tuple shard_id used by
                # GDN fused projections).
                if isinstance(shard_id, tuple):
                    if getattr(param, "_streaming_patched", False):
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                    else:
                        # For MergedColumnParallelLinear params, we need to use weight_loader_v2
                        # Get the parent module that has weight_loader_v2
                        parent_module_name = ".".join(name.split(".")[:-1])
                        if parent_module_name:
                            parent_module = dict(self.named_modules()).get(parent_module_name)
                            if parent_module and hasattr(parent_module, "weight_loader_v2"):
                                parent_module.weight_loader_v2(param, loaded_weight, shard_id)
                            else:
                                weight_loader = param.weight_loader
                                weight_loader(param, loaded_weight, shard_id)
                        else:
                            weight_loader = param.weight_loader
                            weight_loader(param, loaded_weight, shard_id)
                else:
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)

                            if is_padding_needed:
                                # Interleaved padding: reshape into TP shards, pad each shard, reassemble.
                                # This ensures each TP rank gets its correct data slice + zero padding.
                                tp_size = get_tensor_model_parallel_world_size()
                                for _idx in range(2):  # gate (0) and up (1)
                                    _w = loaded_weight[_idx]
                                    _E, _rows, _H = _w.shape
                                    _real_shard = _rows // tp_size
                                    _padded_shard = round_up(_real_shard, _pad_align)
                                    _pad = _padded_shard - _real_shard
                                    if _pad > 0:
                                        _shards = _w.reshape(_E, tp_size, _real_shard, _H)
                                        _shards = torch.nn.functional.pad(_shards, (0, 0, 0, _pad), value=0)
                                        _w = _shards.reshape(_E, tp_size * _padded_shard, _H)
                                    if _idx == 0:
                                        loaded_weight0 = _w
                                    else:
                                        loaded_weight1 = _w
                            else:
                                loaded_weight0 = loaded_weight[0]
                                loaded_weight1 = loaded_weight[1]
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight0,
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight1,
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            if is_padding_needed:
                                # Interleaved padding along last dim (d_ff)
                                tp_size = get_tensor_model_parallel_world_size()
                                _orig_shape = loaded_weight.shape  # [E, hidden, d_ff] or [hidden, d_ff]
                                _d_ff = _orig_shape[-1]
                                _real_shard = _d_ff // tp_size
                                _padded_shard = round_up(_real_shard, _pad_align)
                                _pad = _padded_shard - _real_shard
                                if _pad > 0:
                                    _shards = loaded_weight.reshape(*_orig_shape[:-1], tp_size, _real_shard)
                                    _shards = torch.nn.functional.pad(_shards, (0, _pad), value=0)
                                    loaded_weight = _shards.reshape(*_orig_shape[:-1], tp_size * _padded_shard)

                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        # with self._mark_tower_model(vllm_config, {"image", "video"}):
        self.visual = Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

        # with self._mark_language_model(vllm_config):
        self.language_model = Qwen3_5ForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        #with self._mark_tower_model(vllm_config, {"image", "video"}):
        self.visual = Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

        #with self._mark_language_model(vllm_config):
        self.language_model = Qwen3_5MoeForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
