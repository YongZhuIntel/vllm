# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base_device_communicator import DeviceCommunicatorBase


ONECCL_SUCCESS = 0
ONECCL_FLOAT16 = 6
ONECCL_FLOAT32 = 7
ONECCL_BFLOAT16 = 9
UNIQUE_ID_BYTES = 4096


class OneCCLUniqueId(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char * UNIQUE_ID_BYTES)]


class OneCCLError(RuntimeError):
    def __init__(self, fn: str, code: int):
        super().__init__(f"{fn} failed with onecclResult_t={code}")
        self.code = code


class OneCCL:
    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self._comm = ctypes.c_void_p()

        lib = self.lib
        lib.onecclGetUniqueId.argtypes = [ctypes.POINTER(OneCCLUniqueId)]
        lib.onecclGetUniqueId.restype = ctypes.c_int
        lib.onecclSetDevice.argtypes = [ctypes.c_uint]
        lib.onecclSetDevice.restype = ctypes.c_int
        lib.onecclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            OneCCLUniqueId,
            ctypes.c_int,
        ]
        lib.onecclCommInitRank.restype = ctypes.c_int
        lib.onecclCommDestroy.argtypes = [ctypes.c_void_p]
        lib.onecclCommDestroy.restype = ctypes.c_int
        lib.onecclSend.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.onecclSend.restype = ctypes.c_int
        lib.onecclRecv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.onecclRecv.restype = ctypes.c_int
        lib.onecclMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.onecclMemAlloc.restype = ctypes.c_int
        lib.onecclMemFree.argtypes = [ctypes.c_void_p]
        lib.onecclMemFree.restype = ctypes.c_int

    def _check(self, fn: str, code: int) -> None:
        if code != ONECCL_SUCCESS:
            raise OneCCLError(fn, code)

    def get_unique_id(self) -> OneCCLUniqueId:
        uid = OneCCLUniqueId()
        self._check("onecclGetUniqueId", self.lib.onecclGetUniqueId(ctypes.byref(uid)))
        return uid

    def set_device(self, index: int) -> None:
        self._check("onecclSetDevice", self.lib.onecclSetDevice(index))

    def comm_init_rank(self, nranks: int, uid: OneCCLUniqueId, rank: int) -> None:
        self._check(
            "onecclCommInitRank",
            self.lib.onecclCommInitRank(
                ctypes.byref(self._comm), nranks, uid, rank
            ),
        )

    def send(self, ptr: int, count: int, dtype: int, peer: int, stream: int) -> None:
        self._check(
            "onecclSend",
            self.lib.onecclSend(
                ctypes.c_void_p(ptr),
                count,
                dtype,
                peer,
                self._comm,
                ctypes.c_void_p(stream),
            ),
        )

    def recv(self, ptr: int, count: int, dtype: int, peer: int, stream: int) -> None:
        self._check(
            "onecclRecv",
            self.lib.onecclRecv(
                ctypes.c_void_p(ptr),
                count,
                dtype,
                peer,
                self._comm,
                ctypes.c_void_p(stream),
            ),
        )

    def mem_alloc(self, size: int) -> int:
        ptr = ctypes.c_void_p()
        self._check("onecclMemAlloc", self.lib.onecclMemAlloc(ctypes.byref(ptr), size))
        assert ptr.value is not None
        return ptr.value

    def mem_free(self, ptr: int) -> None:
        if ptr:
            self._check("onecclMemFree", self.lib.onecclMemFree(ctypes.c_void_p(ptr)))

    def destroy(self) -> None:
        if self._comm:
            self._check("onecclCommDestroy", self.lib.onecclCommDestroy(self._comm))
            self._comm = ctypes.c_void_p()


@dataclass
class _PreparedTensor:
    tensor: torch.Tensor
    ptr: int
    count: int
    ccl_dtype: int
    nbytes: int
    host_ptr: int | None = None


class OneCCLIgpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        if self.world_size != 2:
            raise NotImplementedError("VLLM_XPU_IGPU_PP currently supports PP=2 only")

        # VLLM_XPU_IGPU_PP: load the oneCCL v2 C API. CCL_PLUGIN selects
        # libccl_igpu.so; do not use torch.distributed send/recv for PP tensors.
        lib_path = os.getenv("VLLM_XPU_IGPU_PP_LIB", "libccl.so")
        self.ccl = OneCCL(lib_path)
        self.is_igpu_rank = self.rank_in_group == 1
        self._stream = torch.xpu.current_stream().sycl_queue
        self._debug = os.getenv("VLLM_XPU_IGPU_PP_DEBUG", "0") == "1"
        self._log(
            "init "
            f"rank_in_group={self.rank_in_group} global_rank={self.global_rank} "
            f"device={self.device} is_igpu_rank={self.is_igpu_rank} "
        )
        self._init_comm()

    def _log(self, msg: str) -> None:
        if self._debug:
            # VLLM_XPU_IGPU_PP: opt-in debug tracing for prototype PP transfers.
            print(f"[VLLM_XPU_IGPU_PP][rank {self.rank_in_group}] {msg}", flush=True)

    def _init_comm(self) -> None:
        if self.rank_in_group == 0:
            uid = self.ccl.get_unique_id()
            uid_bytes = bytes(uid.data)
        else:
            uid_bytes = None

        objects: list[bytes | None] = [uid_bytes]
        dist.broadcast_object_list(objects, src=self.ranks[0], group=self.cpu_group)
        raw = objects[0]
        assert raw is not None
        uid = OneCCLUniqueId()
        uid.data = raw.ljust(UNIQUE_ID_BYTES, b"\x00")[:UNIQUE_ID_BYTES]

        # VLLM_XPU_IGPU_PP: default to rank-based oneCCL device selection to
        # match plugin examples. Allow override for experiments.
        ccl_device = int(
            os.getenv("VLLM_XPU_IGPU_PP_CCL_DEVICE", str(self.rank_in_group))
        )
        self._log(f"set oneCCL device={ccl_device}")
        self.ccl.set_device(ccl_device)
        self.ccl.comm_init_rank(self.world_size, uid, self.rank_in_group)
        self._log("oneCCL communicator initialized")

    @staticmethod
    def _dtype_to_ccl(dtype: torch.dtype) -> int:
        if dtype == torch.float16:
            return ONECCL_FLOAT16
        if dtype == torch.float32:
            return ONECCL_FLOAT32
        if dtype == torch.bfloat16:
            return ONECCL_BFLOAT16
        raise NotImplementedError(f"VLLM_XPU_IGPU_PP unsupported dtype: {dtype}")

    @staticmethod
    def _dtype_name(dtype: torch.dtype) -> str:
        return str(dtype).removeprefix("torch.")

    @staticmethod
    def _dtype_from_name(name: str) -> torch.dtype:
        return getattr(torch, name)

    def _copy_tensor_to_host_ptr(self, tensor: torch.Tensor, ptr: int) -> None:
        cpu_tensor = tensor.detach().contiguous().cpu()
        ctypes.memmove(
            ptr, cpu_tensor.data_ptr(), cpu_tensor.numel() * cpu_tensor.element_size()
        )

    def _copy_host_ptr_to_tensor(self, ptr: int, tensor: torch.Tensor) -> None:
        cpu_tensor = torch.empty(tensor.size(), dtype=tensor.dtype, device="cpu")
        ctypes.memmove(
            cpu_tensor.data_ptr(), ptr, cpu_tensor.numel() * cpu_tensor.element_size()
        )
        tensor.copy_(cpu_tensor.to(device=tensor.device))

    def _release_prepared(self, item: _PreparedTensor) -> None:
        if item.host_ptr is not None:
            self.ccl.mem_free(item.ptr)

    def _prepare_send_tensor(self, tensor: torch.Tensor) -> _PreparedTensor:
        if not tensor.is_xpu:
            raise NotImplementedError("VLLM_XPU_IGPU_PP only supports XPU tensors")
        tensor = tensor.contiguous()
        ccl_dtype = self._dtype_to_ccl(tensor.dtype)
        count = tensor.numel()
        nbytes = count * tensor.element_size()

        if self.is_igpu_rank:
            # VLLM_XPU_IGPU_PP: iGPU plugin send buffers must be plugin-managed
            # USM host memory, not raw torch XPU tensor memory.
            host_ptr = self.ccl.mem_alloc(nbytes)
            self._copy_tensor_to_host_ptr(tensor, host_ptr)
            return _PreparedTensor(tensor, host_ptr, count, ccl_dtype, nbytes, host_ptr)

        return _PreparedTensor(tensor, tensor.data_ptr(), count, ccl_dtype, nbytes)

    def _prepare_recv_tensor(self, size: tuple[int, ...], dtype: torch.dtype) -> _PreparedTensor:
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        ccl_dtype = self._dtype_to_ccl(dtype)
        count = tensor.numel()
        nbytes = count * tensor.element_size()

        if self.is_igpu_rank:
            # VLLM_XPU_IGPU_PP: iGPU plugin recv buffers must be plugin-managed
            # USM host memory, then copied back into torch XPU tensors.
            host_ptr = self.ccl.mem_alloc(nbytes)
            return _PreparedTensor(tensor, host_ptr, count, ccl_dtype, nbytes, host_ptr)

        return _PreparedTensor(tensor, tensor.data_ptr(), count, ccl_dtype, nbytes)

    def _send_packed(self, packed: torch.Tensor, dst: int) -> None:
        item = self._prepare_send_tensor(packed)
        try:
            self._log(
                f"send packed count={item.count} nbytes={item.nbytes} "
                f"dst={dst} ptr=0x{item.ptr:x}"
            )
            torch.xpu.synchronize()
            self.ccl.send(item.ptr, item.count, item.ccl_dtype, dst, self._stream)
            torch.xpu.synchronize()
        finally:
            self._release_prepared(item)

    def _recv_packed(self, out: torch.Tensor, src: int) -> None:
        item = self._prepare_recv_tensor((out.numel(),), out.dtype)
        try:
            self._log(
                f"recv packed count={item.count} nbytes={item.nbytes} "
                f"src={src} ptr=0x{item.ptr:x}"
            )
            self.ccl.recv(item.ptr, item.count, item.ccl_dtype, src, self._stream)
            torch.xpu.synchronize()
            if self.is_igpu_rank:
                self._copy_host_ptr_to_tensor(item.ptr, item.tensor)
            out.copy_(item.tensor)
        finally:
            self._release_prepared(item)

    def send_tensor_dict(self, tensor_dict: dict[str, torch.Tensor | Any], dst: int) -> None:
        self._log(f"send_tensor_dict keys={list(tensor_dict.keys())} dst={dst}")
        metadata: list[tuple[str, Any]] = []
        tensor_items: list[tuple[str, torch.Tensor]] = []
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                metadata.append((key, ("tensor", tuple(value.size()), self._dtype_name(value.dtype))))
                tensor_items.append((key, value))
            else:
                metadata.append((key, value))

        dist.send_object_list([metadata], dst=self.ranks[dst], group=self.cpu_group)
        self._log("sent metadata")

        if tensor_items:
            dtypes = {tensor.dtype for _, tensor in tensor_items}
            if len(dtypes) != 1:
                raise NotImplementedError(
                    "VLLM_XPU_IGPU_PP packed tensor dict requires one dtype"
                )
            # VLLM_XPU_IGPU_PP: the iGPU plugin hangs on multiple sequential or
            # grouped one-way P2P ops in a single PP hop. Pack all tensor values
            # into one flat tensor so each PP hop performs exactly one send/recv.
            packed = torch.cat(
                [tensor.contiguous().reshape(-1) for _, tensor in tensor_items]
            )
            self._log(
                f"send packed tensors keys={[key for key, _ in tensor_items]} "
                f"count={packed.numel()} dtype={packed.dtype} "
                f"dst={dst} igpu={self.is_igpu_rank}"
            )
            self._send_packed(packed, dst)
        self._log("send_tensor_dict done")

    def recv_tensor_dict(self, src: int) -> dict[str, torch.Tensor | Any]:
        self._log(f"recv_tensor_dict src={src}")
        metadata_list: list[Any] = [None]
        dist.recv_object_list(metadata_list, src=self.ranks[src], group=self.cpu_group)
        metadata = metadata_list[0]
        assert isinstance(metadata, list)
        self._log(f"received metadata keys={[key for key, _ in metadata]}")

        result: dict[str, torch.Tensor | Any] = {}
        tensor_specs: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        for key, value in metadata:
            if isinstance(value, tuple) and len(value) == 3 and value[0] == "tensor":
                _, size, dtype_name = value
                tensor_specs.append((key, tuple(size), self._dtype_from_name(dtype_name)))
            else:
                result[key] = value

        if tensor_specs:
            dtypes = {dtype for _, _, dtype in tensor_specs}
            if len(dtypes) != 1:
                raise NotImplementedError(
                    "VLLM_XPU_IGPU_PP packed tensor dict requires one dtype"
                )
            dtype = next(iter(dtypes))
            total_numel = sum(int(torch.Size(size).numel()) for _, size, _ in tensor_specs)
            self._log(
                f"recv packed tensors keys={[key for key, _, _ in tensor_specs]} "
                f"count={total_numel} dtype={dtype} "
                f"src={src} igpu={self.is_igpu_rank}"
            )

            packed = torch.empty(total_numel, dtype=dtype, device=self.device)
            self._recv_packed(packed, src)
            offset = 0
            for key, size, _ in tensor_specs:
                numel = int(torch.Size(size).numel())
                tensor = packed.narrow(0, offset, numel).view(size)
                result[key] = tensor
                offset += numel

        self._log("recv_tensor_dict done")
        return result

    def destroy(self) -> None:
        self.ccl.destroy()
        super().destroy()
