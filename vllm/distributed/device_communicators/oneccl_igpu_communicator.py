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
# onecclDataType_t (include/oneapi/ccl/v2/types.h)
ONECCL_INT8 = 0
ONECCL_UINT8 = 1
ONECCL_INT32 = 2
ONECCL_INT64 = 4
ONECCL_FLOAT16 = 6
ONECCL_FLOAT32 = 7
ONECCL_FLOAT64 = 8
ONECCL_BFLOAT16 = 9
# onecclRedOp_t (include/oneapi/ccl/v2/types.h)
ONECCL_SUM = 0
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
        # VLLM_XPU_IGPU_TP: TP collectives use the plugin's fused
        # reduce+broadcast / gather kernels (see plugins/igpu allreduce.cpp,
        # allgather.cpp) instead of torch.distributed on the mixed dGPU/iGPU.
        lib.onecclAllReduce.argtypes = [
            ctypes.c_void_p,  # sendbuff
            ctypes.c_void_p,  # recvbuff
            ctypes.c_size_t,  # count
            ctypes.c_int,  # datatype
            ctypes.c_int,  # reduction_op
            ctypes.c_void_p,  # comm
            ctypes.c_void_p,  # stream
        ]
        lib.onecclAllReduce.restype = ctypes.c_int
        lib.onecclAllGather.argtypes = [
            ctypes.c_void_p,  # sendbuff
            ctypes.c_void_p,  # recvbuff
            ctypes.c_size_t,  # sendcount
            ctypes.c_int,  # datatype
            ctypes.c_void_p,  # comm
            ctypes.c_void_p,  # stream
        ]
        lib.onecclAllGather.restype = ctypes.c_int
        # VLLM_XPU_IGPU_TP: register reused collective buffers so the plugin
        # runs its fd handshake once (then skips it), restoring small-message
        # latency. Must be paired symmetrically across ranks.
        lib.onecclCommRegister.argtypes = [
            ctypes.c_void_p,  # comm
            ctypes.c_void_p,  # buff
            ctypes.c_size_t,  # size
            ctypes.POINTER(ctypes.c_void_p),  # handle (out)
        ]
        lib.onecclCommRegister.restype = ctypes.c_int
        lib.onecclCommDeregister.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.onecclCommDeregister.restype = ctypes.c_int

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

    def all_reduce(
        self, sendptr: int, recvptr: int, count: int, dtype: int, op: int, stream: int
    ) -> None:
        self._check(
            "onecclAllReduce",
            self.lib.onecclAllReduce(
                ctypes.c_void_p(sendptr),
                ctypes.c_void_p(recvptr),
                count,
                dtype,
                op,
                self._comm,
                ctypes.c_void_p(stream),
            ),
        )

    def all_gather(
        self, sendptr: int, recvptr: int, sendcount: int, dtype: int, stream: int
    ) -> None:
        self._check(
            "onecclAllGather",
            self.lib.onecclAllGather(
                ctypes.c_void_p(sendptr),
                ctypes.c_void_p(recvptr),
                sendcount,
                dtype,
                self._comm,
                ctypes.c_void_p(stream),
            ),
        )

    def comm_register(self, ptr: int, size: int) -> int:
        handle = ctypes.c_void_p()
        self._check(
            "onecclCommRegister",
            self.lib.onecclCommRegister(
                self._comm, ctypes.c_void_p(ptr), size, ctypes.byref(handle)
            ),
        )
        return handle.value if handle.value is not None else ptr

    def comm_deregister(self, handle: int) -> None:
        self._check(
            "onecclCommDeregister",
            self.lib.onecclCommDeregister(self._comm, ctypes.c_void_p(handle)),
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


@dataclass
class _CollBuf:
    """A stable, registered per-role collective buffer.

    ``ptr`` is what the plugin sees; ``nbytes`` is its (grow-only) capacity;
    ``handle`` is the onecclCommRegister handle. On the dGPU ``keep`` owns the
    torch device allocation (a uint8 buffer reinterpreted per call); on the
    iGPU ``keep`` is None and ``ptr`` is plugin USM-host memory.
    """

    ptr: int
    nbytes: int
    handle: int
    keep: Any = None
    registered: bool = True


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
            # The igpu plugin topology is fixed at 2 ranks (1 dGPU + 1 iGPU on
            # the same host); this holds for both PP=2 and TP=2.
            raise NotImplementedError(
                "VLLM_XPU_IGPU currently supports exactly 2 ranks "
                "(1 dGPU + 1 iGPU), i.e. PP=2 or TP=2 only"
            )

        # VLLM_XPU_IGPU_*: load the oneCCL v2 C API. CCL_PLUGIN selects
        # libccl_igpu.so; do not use torch.distributed for PP tensors or TP
        # collectives on the mixed dGPU/iGPU platform.
        lib_path = os.getenv(
            "VLLM_XPU_IGPU_LIB", os.getenv("VLLM_XPU_IGPU_PP_LIB", "libccl.so")
        )
        self.ccl = OneCCL(lib_path)
        self.is_igpu_rank = self.rank_in_group == 1
        self._stream = torch.xpu.current_stream().sycl_queue
        self._debug = os.getenv("VLLM_XPU_IGPU_DEBUG", "0") == "1" or (
            os.getenv("VLLM_XPU_IGPU_PP_DEBUG", "0") == "1"
        )
        # VLLM_XPU_IGPU_PP_REGISTER: opt-in to routing PP send/recv through the
        # stable, REGISTERED _coll_buf pool so the plugin does its pt2pt fd
        # handshake once per buffer and skips it thereafter. Default off keeps
        # the original per-op path (fresh USM-host alloc / raw tensor ptr) that
        # re-handshakes every hop. TP always registers; this makes PP match when
        # enabled.
        self._pp_register = os.getenv("VLLM_XPU_IGPU_PP_REGISTER", "0") == "1"
        # VLLM_XPU_IGPU_TP: grow-only pool of stable, REGISTERED collective
        # buffers keyed by role ("ar_send"/"ar_recv"/"ag_send"/"ag_recv"), on
        # BOTH ranks (iGPU: plugin USM-host; dGPU: torch device memory). A role
        # keeps one buffer whose pointer changes only when it must grow; the
        # buffer is registered with onecclCommRegister so the plugin runs its
        # fd handshake once and skips it thereafter (restoring small-message
        # latency). Grown-out buffers are retired -- deregistered but kept
        # alive so their address is never recycled into a stale plugin import --
        # and freed only at destroy. In practice profiling hits the max size
        # first, so no role grows after its first call.
        self._coll_bufs: dict[str, _CollBuf] = {}
        self._retired_bufs: list[_CollBuf] = []
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
        if dtype == torch.float64:
            return ONECCL_FLOAT64
        if dtype == torch.int32:
            return ONECCL_INT32
        if dtype == torch.int64:
            return ONECCL_INT64
        if dtype == torch.int8:
            return ONECCL_INT8
        if dtype == torch.uint8:
            return ONECCL_UINT8
        raise NotImplementedError(f"VLLM_XPU_IGPU unsupported dtype: {dtype}")

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
        # Registered fast path (VLLM_XPU_IGPU_PP_REGISTER=1): send from a stable,
        # REGISTERED per-role buffer (the same _coll_buf pool TP uses) so the
        # plugin runs its pt2pt fd handshake once and skips it on every later hop
        # (p2p.cpp registered fast path). Both ranks route matching hops through
        # registered buffers (sender's pp_send, receiver's pp_recv), so the skip
        # stays symmetric and the paired kvs_sendrecv never desyncs.
        if self._pp_register:
            packed = packed.contiguous()
            ccl_dtype = self._dtype_to_ccl(packed.dtype)
            count = packed.numel()
            nbytes = count * packed.element_size()
            sbuf = self._coll_buf("pp_send", nbytes)
            self._log(
                f"send packed count={count} nbytes={nbytes} dst={dst} "
                f"ptr=0x{sbuf.ptr:x} reg=1"
            )
            if self.is_igpu_rank:
                self._copy_tensor_to_host_ptr(packed, sbuf.ptr)
            else:
                self._dev_view(sbuf, count, packed.dtype).copy_(packed.reshape(-1))
                torch.xpu.synchronize()
            self.ccl.send(sbuf.ptr, count, ccl_dtype, dst, self._stream)
            torch.xpu.synchronize()
            return

        # Default per-op path: fresh USM-host alloc (iGPU) / raw tensor ptr
        # (dGPU); re-does the handshake every hop.
        item = self._prepare_send_tensor(packed)
        try:
            self._log(
                f"send packed count={item.count} nbytes={item.nbytes} "
                f"dst={dst} ptr=0x{item.ptr:x} reg=0"
            )
            torch.xpu.synchronize()
            self.ccl.send(item.ptr, item.count, item.ccl_dtype, dst, self._stream)
            torch.xpu.synchronize()
        finally:
            self._release_prepared(item)

    def _recv_packed(self, out: torch.Tensor, src: int) -> None:
        # Registered fast path (VLLM_XPU_IGPU_PP_REGISTER=1): mirror of
        # _send_packed -- receive into the stable, registered pp_recv buffer,
        # then copy out into the caller's tensor.
        if self._pp_register:
            ccl_dtype = self._dtype_to_ccl(out.dtype)
            count = out.numel()
            nbytes = count * out.element_size()
            rbuf = self._coll_buf("pp_recv", nbytes)
            self._log(
                f"recv packed count={count} nbytes={nbytes} src={src} "
                f"ptr=0x{rbuf.ptr:x} reg=1"
            )
            self.ccl.recv(rbuf.ptr, count, ccl_dtype, src, self._stream)
            torch.xpu.synchronize()
            if self.is_igpu_rank:
                self._copy_host_ptr_to_tensor(rbuf.ptr, out)
            else:
                out.copy_(self._dev_view(rbuf, count, out.dtype).reshape(out.shape))
            return

        # Default per-op path.
        item = self._prepare_recv_tensor((out.numel(),), out.dtype)
        try:
            self._log(
                f"recv packed count={item.count} nbytes={item.nbytes} "
                f"src={src} ptr=0x{item.ptr:x} reg=0"
            )
            self.ccl.recv(item.ptr, item.count, item.ccl_dtype, src, self._stream)
            torch.xpu.synchronize()
            if self.is_igpu_rank:
                self._copy_host_ptr_to_tensor(item.ptr, item.tensor)
            out.copy_(item.tensor)
        finally:
            self._release_prepared(item)

    # --- TP collectives (VLLM_XPU_IGPU_TP) ------------------------------------
    def _coll_buf(self, role: str, nbytes: int) -> _CollBuf:
        """Return a stable, registered collective buffer of >= ``nbytes`` bytes.

        Grow-only: the pointer changes only when a larger size is requested, and
        the previous buffer is deregistered + retired (kept alive) so its address
        is never recycled. Because both ranks issue the same role/size sequence
        in lockstep, they grow (and thus re-register / re-handshake) together.
        """
        cur = self._coll_bufs.get(role)
        if cur is not None and cur.nbytes >= nbytes:
            return cur
        if cur is not None:
            self.ccl.comm_deregister(cur.handle)
            cur.registered = False
            self._retired_bufs.append(cur)
        if self.is_igpu_rank:
            ptr = self.ccl.mem_alloc(nbytes)
            keep = None
        else:
            keep = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            ptr = keep.data_ptr()
        handle = self.ccl.comm_register(ptr, nbytes)
        buf = _CollBuf(ptr=ptr, nbytes=nbytes, handle=handle, keep=keep)
        self._coll_bufs[role] = buf
        return buf

    def _dev_view(self, buf: _CollBuf, count: int, dtype: torch.dtype) -> torch.Tensor:
        # Reinterpret the leading bytes of the dGPU uint8 pool buffer as a
        # 1-D tensor of ``count`` elements of ``dtype`` (data_ptr == buf.ptr).
        elsize = torch.empty(0, dtype=dtype).element_size()
        return buf.keep[: count * elsize].view(dtype)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if not input_.is_xpu:
            raise NotImplementedError("VLLM_XPU_IGPU_TP only supports XPU tensors")
        input_ = input_.contiguous()
        ccl_dtype = self._dtype_to_ccl(input_.dtype)
        count = input_.numel()
        nbytes = count * input_.element_size()
        # Fused reduce+broadcast writes both recvbuffs; return a fresh output
        # rather than reducing in place (the kernel reads peer sendbuff over PCIe
        # while writing recvbuff, and the pool recv buffer is overwritten by the
        # next call so it must not be handed to the caller).
        output = torch.empty_like(input_)
        sbuf = self._coll_buf("ar_send", nbytes)
        rbuf = self._coll_buf("ar_recv", nbytes)
        self._log(f"all_reduce count={count} nbytes={nbytes} igpu={self.is_igpu_rank}")

        if self.is_igpu_rank:
            self._copy_tensor_to_host_ptr(input_, sbuf.ptr)
            self.ccl.all_reduce(
                sbuf.ptr, rbuf.ptr, count, ccl_dtype, ONECCL_SUM, self._stream
            )
            torch.xpu.synchronize()
            self._copy_host_ptr_to_tensor(rbuf.ptr, output)
        else:
            self._dev_view(sbuf, count, input_.dtype).copy_(input_.reshape(-1))
            torch.xpu.synchronize()
            self.ccl.all_reduce(
                sbuf.ptr, rbuf.ptr, count, ccl_dtype, ONECCL_SUM, self._stream
            )
            torch.xpu.synchronize()
            output.copy_(self._dev_view(rbuf, count, input_.dtype).reshape(output.shape))
        return output

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if not input_.is_xpu:
            raise NotImplementedError("VLLM_XPU_IGPU_TP only supports XPU tensors")
        if dim < 0:
            dim += input_.dim()
        input_ = input_.contiguous()
        input_size = tuple(input_.size())
        ccl_dtype = self._dtype_to_ccl(input_.dtype)
        sendcount = input_.numel()
        nbytes = sendcount * input_.element_size()
        total = sendcount * self.world_size
        self._log(f"all_gather sendcount={sendcount} dim={dim} igpu={self.is_igpu_rank}")

        # Gather rank-major into a flat device tensor, then reshape like the base
        # communicator (concat-style all-gather, torch.compile compatible).
        gathered = torch.empty(total, dtype=input_.dtype, device=self.device)
        sbuf = self._coll_buf("ag_send", nbytes)
        rbuf = self._coll_buf("ag_recv", nbytes * self.world_size)
        if self.is_igpu_rank:
            self._copy_tensor_to_host_ptr(input_, sbuf.ptr)
            self.ccl.all_gather(sbuf.ptr, rbuf.ptr, sendcount, ccl_dtype, self._stream)
            torch.xpu.synchronize()
            self._copy_host_ptr_to_tensor(rbuf.ptr, gathered)
        else:
            self._dev_view(sbuf, sendcount, input_.dtype).copy_(input_.reshape(-1))
            torch.xpu.synchronize()
            self.ccl.all_gather(sbuf.ptr, rbuf.ptr, sendcount, ccl_dtype, self._stream)
            torch.xpu.synchronize()
            gathered.copy_(self._dev_view(rbuf, total, input_.dtype))

        output = gathered.reshape((self.world_size,) + input_size)
        output = output.movedim(0, dim)
        output = output.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output

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
        # Deregister live buffers (retired ones already were) and free the iGPU
        # USM-host allocations; dGPU torch buffers are freed by GC. Both ranks
        # tear down the same roles in the same order, so this stays symmetric.
        for buf in list(self._coll_bufs.values()) + self._retired_bufs:
            if buf.registered:
                try:
                    self.ccl.comm_deregister(buf.handle)
                except OneCCLError:
                    pass
                buf.registered = False
            if buf.keep is None and buf.ptr:
                self.ccl.mem_free(buf.ptr)
        self._coll_bufs.clear()
        self._retired_bufs.clear()
        self.ccl.destroy()
        super().destroy()
