"""Unit tests for the dependency-free hardware bandwidth helper."""

import ctypes
import importlib.util
import pathlib
import sys

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
