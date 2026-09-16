#!/usr/bin/env python3
"""Measure host memcpy and pinned CUDA transfer bandwidth without extra packages.

This helper intentionally uses only Python's standard library and the CUDA
driver already required by an NVIDIA backend. JSON is the only stdout output;
ggrun validates and hardware-keys it before caching anything.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import multiprocessing
import os
import queue
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


MIB = 1024 * 1024
MIN_BYTES = 16 * MIB
MAX_BYTES = 1024 * MIB
MIN_SAMPLE_SECONDS = 0.25
MAX_ITERATIONS = 64


class ProbeError(RuntimeError):
    """A concise user-facing probe failure."""


def _timed_copy(
    copy_once: Callable[[], None],
    byte_count: int,
    min_iterations: int,
    synchronize: Optional[Callable[[], None]] = None,
) -> Tuple[int, int]:
    """Return conservative one-way MB/s and the number of measured copies."""
    for _ in range(2):
        copy_once()
    if synchronize is not None:
        synchronize()

    started = time.perf_counter()
    iterations = 0
    while iterations < min_iterations or (
        time.perf_counter() - started < MIN_SAMPLE_SECONDS
        and iterations < MAX_ITERATIONS
    ):
        copy_once()
        iterations += 1
    if synchronize is not None:
        synchronize()
    elapsed = time.perf_counter() - started
    if elapsed <= 0:
        raise ProbeError("timer returned a non-positive interval")
    mbps = int((byte_count * iterations / 1_000_000.0) / elapsed)
    if mbps <= 0:
        raise ProbeError("measured bandwidth was zero")
    return mbps, iterations


def _load_memmove() -> Callable[..., object]:
    candidates = [None]
    if os.name == "nt":
        candidates.extend(("msvcrt.dll", "ucrtbase.dll"))
    else:
        discovered = ctypes.util.find_library("c")
        if discovered:
            candidates.append(discovered)
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate) if candidate else ctypes.CDLL(None)
        except OSError:
            continue
        function = getattr(library, "memmove", None)
        if function is None:
            function = getattr(library, "memcpy", None)
        if function is not None:
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            function.restype = ctypes.c_void_p
            return function
    raise ProbeError("C runtime memmove is unavailable")


def _host_copy_worker(
    byte_count: int,
    min_iterations: int,
    start_event: object,
    ready_queue: object,
    result_queue: object,
) -> None:
    source = None
    destination = None
    ready_sent = False
    try:
        source = mmap.mmap(-1, byte_count, access=mmap.ACCESS_WRITE)
        destination = mmap.mmap(-1, byte_count, access=mmap.ACCESS_WRITE)
        source_addr = ctypes.addressof(ctypes.c_char.from_buffer(source))
        destination_addr = ctypes.addressof(ctypes.c_char.from_buffer(destination))
        # First-touch inside the worker gives NUMA-aware allocation on Linux.
        ctypes.memset(source_addr, 0xA5, byte_count)
        ctypes.memset(destination_addr, 0, byte_count)
        memmove = _load_memmove()
        for _ in range(2):
            memmove(destination_addr, source_addr, byte_count)
        ready_queue.put("")
        ready_sent = True
        start_event.wait()
        started = time.perf_counter()
        iterations = 0
        while iterations < min_iterations or (
            time.perf_counter() - started < MIN_SAMPLE_SECONDS
            and iterations < 4096
        ):
            memmove(destination_addr, source_addr, byte_count)
            iterations += 1
        elapsed = time.perf_counter() - started
        result_queue.put((byte_count * iterations, elapsed, iterations, ""))
    except Exception as exc:  # child must report rather than silently disappear
        if not ready_sent:
            ready_queue.put(str(exc))
        result_queue.put((0, 0.0, 0, str(exc)))
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()


def measure_host_copy(
    byte_count: int, min_iterations: int, requested_workers: int
) -> Tuple[int, int, int]:
    """Measure NUMA-aware parallel memcpy as aggregate read+write traffic."""
    # Keep at least 16 MiB per worker. The combined source+destination working
    # set then exceeds ordinary CPU caches instead of timing L3 bandwidth.
    workers = max(1, min(requested_workers, byte_count // (16 * MIB)))
    context_name = "spawn" if os.name == "nt" else "fork"
    context = multiprocessing.get_context(context_name)
    start_event = context.Event()
    ready_queue = context.Queue()
    result_queue = context.Queue()
    chunk = byte_count // workers
    processes = []
    for worker in range(workers):
        size = byte_count - chunk * worker if worker == workers - 1 else chunk
        process = context.Process(
            target=_host_copy_worker,
            args=(size, min_iterations, start_event, ready_queue, result_queue),
        )
        process.start()
        processes.append(process)
    try:
        readiness = [ready_queue.get(timeout=30) for _ in processes]
        readiness_errors = [error for error in readiness if error]
        if readiness_errors:
            raise ProbeError(f"host memcpy worker failed: {readiness_errors[0]}")
        start_event.set()
        results = [result_queue.get(timeout=30) for _ in processes]
        errors = [result[3] for result in results if result[3]]
        if errors:
            raise ProbeError(f"host memcpy worker failed: {errors[0]}")
        elapsed = max(result[1] for result in results)
        if elapsed <= 0:
            raise ProbeError("host memcpy timer returned a non-positive interval")
        copied = sum(result[0] for result in results)
        iterations = sum(result[2] for result in results)
        # memcpy consumes one source read and one destination write. Reporting
        # both produces the memory-controller traffic ceiling that a read-heavy
        # CPU weight stream should be compared against.
        return int((2 * copied / 1_000_000.0) / elapsed), iterations, workers
    except queue.Empty as exc:
        raise ProbeError("host memcpy workers timed out") from exc
    finally:
        start_event.set()
        for process in processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)


def _load_cuda_driver() -> ctypes.CDLL:
    candidates: List[str] = []
    discovered = ctypes.util.find_library("cuda")
    if discovered:
        candidates.append(discovered)
    if os.name == "nt":
        candidates.append("nvcuda.dll")
    else:
        candidates.extend(("libcuda.so.1", "libcuda.so"))
    errors: List[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) if errors else "no CUDA driver library candidate"
    raise ProbeError(f"CUDA driver library unavailable ({detail})")


class CudaDriver:
    def __init__(self) -> None:
        self.lib = _load_cuda_driver()
        self.init = self._bind(("cuInit",), [ctypes.c_uint])
        self.device_by_bus = self._bind(
            ("cuDeviceGetByPCIBusId",),
            [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p],
        )
        self.ctx_create = self._bind(
            ("cuCtxCreate_v2", "cuCtxCreate"),
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
        )
        self.ctx_destroy = self._bind(
            ("cuCtxDestroy_v2", "cuCtxDestroy"), [ctypes.c_void_p]
        )
        self.ctx_synchronize = self._bind(("cuCtxSynchronize",), [])
        self.host_alloc = self._bind(
            ("cuMemHostAlloc",),
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
        )
        self.host_free = self._bind(("cuMemFreeHost",), [ctypes.c_void_p])
        self.device_alloc = self._bind(
            ("cuMemAlloc_v2", "cuMemAlloc"),
            [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
        )
        self.device_free = self._bind(
            ("cuMemFree_v2", "cuMemFree"), [ctypes.c_uint64]
        )
        self.copy_h2d = self._bind(
            ("cuMemcpyHtoD_v2", "cuMemcpyHtoD"),
            [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
        )
        self.copy_d2h = self._bind(
            ("cuMemcpyDtoH_v2", "cuMemcpyDtoH"),
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
        )
        self.error_string = self._bind_optional(
            ("cuGetErrorString",),
            [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        )
        self.check(self.init(0), "cuInit")

    def _bind(
        self, names: Sequence[str], argtypes: List[object]
    ) -> Callable[..., int]:
        function = self._bind_optional(names, argtypes)
        if function is None:
            raise ProbeError(f"CUDA driver is missing {'/'.join(names)}")
        return function

    def _bind_optional(
        self, names: Sequence[str], argtypes: List[object]
    ) -> Optional[Callable[..., int]]:
        for name in names:
            function = getattr(self.lib, name, None)
            if function is not None:
                function.argtypes = argtypes
                function.restype = ctypes.c_int
                return function
        return None

    def check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        detail = ""
        if self.error_string is not None:
            value = ctypes.c_char_p()
            if self.error_string(result, ctypes.byref(value)) == 0 and value.value:
                detail = f": {value.value.decode('utf-8', errors='replace')}"
        raise ProbeError(f"{operation} failed with CUDA error {result}{detail}")

    def device_for_bus(self, bus_id: str) -> ctypes.c_int:
        alternatives = [bus_id]
        try:
            domain, remainder = bus_id.split(":", 1)
            canonical = f"{int(domain, 16):04x}:{remainder}"
            if canonical not in alternatives:
                alternatives.append(canonical)
        except (ValueError, TypeError):
            pass
        last_result = -1
        for candidate in alternatives:
            device = ctypes.c_int()
            last_result = self.device_by_bus(
                ctypes.byref(device), candidate.encode("ascii", errors="strict")
            )
            if last_result == 0:
                return device
        self.check(last_result, f"cuDeviceGetByPCIBusId({bus_id})")
        raise AssertionError("unreachable")


@dataclass
class DeviceBuffers:
    context: ctypes.c_void_p
    host: ctypes.c_void_p
    device: ctypes.c_uint64


def measure_gpu(
    cuda: CudaDriver, bus_id: str, byte_count: int, min_iterations: int
) -> Dict[str, Union[int, str]]:
    device = cuda.device_for_bus(bus_id)
    buffers = DeviceBuffers(ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_uint64())
    try:
        cuda.check(
            cuda.ctx_create(ctypes.byref(buffers.context), 0, device),
            f"cuCtxCreate({bus_id})",
        )
        cuda.check(
            cuda.host_alloc(ctypes.byref(buffers.host), byte_count, 0),
            f"cuMemHostAlloc({bus_id})",
        )
        cuda.check(
            cuda.device_alloc(ctypes.byref(buffers.device), byte_count),
            f"cuMemAlloc({bus_id})",
        )
        ctypes.memset(buffers.host.value, 0x5A, byte_count)

        def synchronize() -> None:
            cuda.check(cuda.ctx_synchronize(), f"cuCtxSynchronize({bus_id})")

        h2d_mbps, h2d_iterations = _timed_copy(
            lambda: cuda.check(
                cuda.copy_h2d(buffers.device, buffers.host, byte_count),
                f"cuMemcpyHtoD({bus_id})",
            ),
            byte_count,
            min_iterations,
            synchronize,
        )
        d2h_mbps, d2h_iterations = _timed_copy(
            lambda: cuda.check(
                cuda.copy_d2h(buffers.host, buffers.device, byte_count),
                f"cuMemcpyDtoH({bus_id})",
            ),
            byte_count,
            min_iterations,
            synchronize,
        )
        return {
            "pci_bus_id": bus_id,
            "h2d_mbps": h2d_mbps,
            "d2h_mbps": d2h_mbps,
            "h2d_iterations": h2d_iterations,
            "d2h_iterations": d2h_iterations,
        }
    finally:
        if buffers.device.value:
            cuda.device_free(buffers.device)
        if buffers.host.value:
            cuda.host_free(buffers.host)
        if buffers.context.value:
            cuda.ctx_destroy(buffers.context)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure host memcpy and pinned CUDA H2D/D2H bandwidth"
    )
    parser.add_argument("--gpu-bus-id", action="append", default=[])
    parser.add_argument("--bytes", type=int, default=128 * MIB)
    parser.add_argument("--min-iterations", type=int, default=4)
    parser.add_argument(
        "--host-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    args = parser.parse_args(argv)
    if not args.gpu_bus_id:
        parser.error("at least one --gpu-bus-id is required")
    if not MIN_BYTES <= args.bytes <= MAX_BYTES:
        parser.error(
            f"--bytes must be between {MIN_BYTES} and {MAX_BYTES} (16 MiB to 1 GiB)"
        )
    if not 1 <= args.min_iterations <= MAX_ITERATIONS:
        parser.error(f"--min-iterations must be between 1 and {MAX_ITERATIONS}")
    if not 1 <= args.host_workers <= 256:
        parser.error("--host-workers must be between 1 and 256")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        host_mbps, host_iterations, host_workers = measure_host_copy(
            args.bytes, args.min_iterations, args.host_workers
        )
        cuda = CudaDriver()
        gpus = [
            measure_gpu(cuda, bus_id, args.bytes, args.min_iterations)
            for bus_id in args.gpu_bus_id
        ]
        json.dump(
            {
                "bytes": args.bytes,
                "min_iterations": args.min_iterations,
                "host_copy_mbps": host_mbps,
                "host_copy_iterations": host_iterations,
                "host_copy_workers": host_workers,
                "gpus": gpus,
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0
    except (OSError, ProbeError) as exc:
        print(f"bandwidth probe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
