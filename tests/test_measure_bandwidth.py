"""Unit tests for the dependency-free hardware bandwidth helper."""

import ctypes
import importlib.util
import multiprocessing
import os
import pathlib
import queue
import sys
import threading
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "go" / "pkg" / "detect" / "scripts" / "measure_bandwidth.py"
SPEC = importlib.util.spec_from_file_location("measure_bandwidth", SCRIPT)
assert SPEC and SPEC.loader
measure_bandwidth = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = measure_bandwidth
SPEC.loader.exec_module(measure_bandwidth)


def test_timed_copy_honors_minimum_iterations(monkeypatch):
    monkeypatch.setattr(measure_bandwidth, "MIN_SAMPLE_SECONDS", 0)
    calls = []
    mbps, iterations = measure_bandwidth._timed_copy(
        lambda: calls.append(1), 1_000_000, 3
    )
    assert iterations == 3
    assert len(calls) == 5  # two warmups plus the three measured copies
    assert mbps > 0


def test_cuda_bus_lookup_normalizes_nvidia_domain_width():
    driver = object.__new__(measure_bandwidth.CudaDriver)
    attempted = []

    def device_by_bus(device_pointer, encoded_bus):
        attempted.append(encoded_bus)
        if encoded_bus != b"0000:17:00.0":
            return 1
        ctypes.cast(device_pointer, ctypes.POINTER(ctypes.c_int))[0] = 7
        return 0

    driver.device_by_bus = device_by_bus
    device = driver.device_for_bus("00000000:17:00.0")
    assert device.value == 7
    assert attempted == [b"00000000:17:00.0", b"0000:17:00.0"]


def test_parser_requires_a_gpu_bus_id():
    with pytest.raises(SystemExit):
        measure_bandwidth.parse_args([])


MIB = measure_bandwidth.MIB
WINDOWS_CRT = ["vcruntime140.dll", "ucrtbase.dll", "msvcrt.dll"]
MEMMOVE_ARGTYPES = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]


class _FakeExport:
    pass


class _FakeLibrary:
    def __init__(self, exports):
        for name in exports:
            setattr(self, name, _FakeExport())


def _windows_loader(attempted, available):
    """Mimic LoadLibrary: None is a TypeError, unknown DLLs are not found."""

    def fake_cdll(name, *args, **kwargs):
        attempted.append(name)
        if name is None:
            raise TypeError("LoadLibrary() argument 1 must be str, not None")
        if name not in available:
            raise FileNotFoundError(f"Could not find module '{name}'")
        return available[name]

    return fake_cdll


def _as_windows(monkeypatch, attempted, available):
    # Patch only the script's own os binding, not the global os module.
    monkeypatch.setattr(measure_bandwidth, "os", types.SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        measure_bandwidth.ctypes, "CDLL", _windows_loader(attempted, available)
    )


def test_windows_memmove_never_passes_none_and_tries_crt_dlls(monkeypatch):
    attempted = []
    ucrt = _FakeLibrary(["memmove"])
    _as_windows(monkeypatch, attempted, {"ucrtbase.dll": ucrt})
    function = measure_bandwidth._load_memmove()
    assert attempted[0] is not None, "ctypes.CDLL(None) is a TypeError on Windows"
    assert attempted == ["vcruntime140.dll", "ucrtbase.dll"]
    assert function is ucrt.memmove
    assert function.argtypes == MEMMOVE_ARGTYPES
    assert function.restype is ctypes.c_void_p


def test_windows_memmove_uses_memcpy_export_when_memmove_missing(monkeypatch):
    attempted = []
    msvcrt = _FakeLibrary(["memcpy"])
    _as_windows(
        monkeypatch,
        attempted,
        {"vcruntime140.dll": _FakeLibrary([]), "msvcrt.dll": msvcrt},
    )
    assert measure_bandwidth._load_memmove() is msvcrt.memcpy
    assert attempted == WINDOWS_CRT


def test_windows_memmove_fails_closed_naming_every_candidate(monkeypatch):
    attempted = []
    monkeypatch.setattr(measure_bandwidth, "os", types.SimpleNamespace(name="nt"))

    def always_typeerror(name, *args, **kwargs):
        attempted.append(name)
        raise TypeError("LoadLibrary() argument 1 must be str, not None")

    monkeypatch.setattr(measure_bandwidth.ctypes, "CDLL", always_typeerror)
    with pytest.raises(measure_bandwidth.ProbeError, match="vcruntime140.dll") as info:
        measure_bandwidth._load_memmove()
    for name in WINDOWS_CRT:
        assert name in str(info.value)
    assert attempted == WINDOWS_CRT


@pytest.mark.skipif(os.name == "nt", reason="dlopen(NULL) exists only on POSIX")
def test_linux_memmove_candidates_unchanged(monkeypatch):
    monkeypatch.setattr(measure_bandwidth, "os", types.SimpleNamespace(name="posix"))
    monkeypatch.setattr(
        measure_bandwidth.ctypes.util,
        "find_library",
        lambda name: "libc.so.6" if name == "c" else None,
    )
    assert measure_bandwidth._memmove_library_candidates() == [None, "libc.so.6"]

    attempted = []
    real_cdll = ctypes.CDLL

    def spy(name, *args, **kwargs):
        attempted.append(name)
        return real_cdll(name, *args, **kwargs)

    monkeypatch.setattr(measure_bandwidth.ctypes, "CDLL", spy)
    function = measure_bandwidth._load_memmove()
    assert attempted == [None]
    assert function.restype is ctypes.c_void_p


@pytest.mark.skipif(os.name == "nt", reason="POSIX failure path")
def test_linux_memmove_still_fails_closed_without_libc(monkeypatch):
    def unavailable(name, *args, **kwargs):
        raise OSError("not found")

    monkeypatch.setattr(measure_bandwidth.ctypes, "CDLL", unavailable)
    with pytest.raises(measure_bandwidth.ProbeError, match="memmove is unavailable"):
        measure_bandwidth._load_memmove()


def test_host_copy_worker_copies_with_windows_loader(monkeypatch):
    # The child in issue #73 died with the TypeError before any copy ran.
    prototype = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
    )
    crt = types.SimpleNamespace(
        memmove=prototype(ctypes.cast(ctypes.memmove, ctypes.c_void_p).value)
    )
    monkeypatch.setattr(measure_bandwidth, "MIN_SAMPLE_SECONDS", 0)
    _as_windows(monkeypatch, [], {"ucrtbase.dll": crt})
    start = threading.Event()
    start.set()
    ready, results = queue.Queue(), queue.Queue()
    measure_bandwidth._host_copy_worker(16 * MIB, 1, start, ready, results)
    copied, elapsed, iterations, error = results.get_nowait()
    assert error == ""
    assert ready.get_nowait() == ""
    assert iterations >= 1
    assert copied == 16 * MIB * iterations
    assert elapsed > 0


def test_load_memmove_copies_bytes():
    source = ctypes.create_string_buffer(bytes(range(256)) * 64)
    destination = ctypes.create_string_buffer(len(source))
    memmove = measure_bandwidth._load_memmove()
    memmove(ctypes.addressof(destination), ctypes.addressof(source), len(source))
    assert destination.raw == source.raw


def test_measure_host_copy_runs_under_spawn(monkeypatch):
    # Windows always uses spawn: children re-import the script by module name.
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    real_get_context = multiprocessing.get_context
    monkeypatch.setattr(
        measure_bandwidth.multiprocessing,
        "get_context",
        lambda name=None: real_get_context("spawn"),
    )
    mbps, iterations, workers = measure_bandwidth.measure_host_copy(32 * MIB, 1, 2)
    assert workers == 2
    assert iterations >= 2
    assert mbps > 0


@pytest.mark.skipif(os.name != "nt", reason="native Windows CRT only")
def test_native_windows_memmove_comes_from_crt_dll(monkeypatch):
    loaded = []
    real_cdll = ctypes.CDLL

    def spy(name, *args, **kwargs):
        library = real_cdll(name, *args, **kwargs)
        loaded.append(name)
        return library

    monkeypatch.setattr(measure_bandwidth.ctypes, "CDLL", spy)
    memmove = measure_bandwidth._load_memmove()
    assert loaded and loaded[-1] in WINDOWS_CRT
    source = ctypes.create_string_buffer(bytes(range(256)) * 64)
    destination = ctypes.create_string_buffer(len(source))
    memmove(ctypes.addressof(destination), ctypes.addressof(source), len(source))
    assert destination.raw == source.raw


@pytest.mark.skipif(os.name != "nt", reason="native Windows spawn path only")
def test_native_windows_host_copy(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    mbps, iterations, workers = measure_bandwidth.measure_host_copy(16 * MIB, 1, 1)
    assert workers == 1
    assert iterations >= 1
    assert mbps > 0
