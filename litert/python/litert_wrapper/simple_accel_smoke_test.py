# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal smoke test for CPU, GPU, and GPU+CPU fallback.

Example:
  python simple_accel_smoke_test.py \
    --model_path ../../../test/testdata/mobilenet_v2_1.0_224.tflite \
    --iterations 1
"""

from __future__ import annotations

import argparse
import pprint
import sys
import time
from types import MappingProxyType

from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.environment import Environment
from ai_edge_litert.environment import EnvironmentOptions
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import CpuOptions
from ai_edge_litert.options import GpuOptions
from ai_edge_litert.options import Options
from ai_edge_litert.tensor_buffer import TensorBuffer
import numpy as np

_READABLE_DTYPES = MappingProxyType({
    np.dtype(np.float32): np.float32,
    np.dtype(np.int32): np.int32,
    np.dtype(np.int8): np.int8,
})


def _normalize_dtype(dtype_value) -> np.dtype:
  if isinstance(dtype_value, np.dtype):
    return dtype_value
  return np.dtype(dtype_value)


def _readable_dtype(dtype_value):
  np_dtype = _normalize_dtype(dtype_value)
  return _READABLE_DTYPES.get(np_dtype)


def _create_environment() -> Environment:
  return Environment.create(options=EnvironmentOptions())


def _create_model_options(
    hardware_accel: HardwareAccelerator, cpu_num_threads: int
) -> Options:
  gpu_options = (
      GpuOptions() if hardware_accel & HardwareAccelerator.GPU else None
  )
  return Options(
      hardware_accelerators=hardware_accel,
      cpu_options=CpuOptions(num_threads=cpu_num_threads),
      gpu_options=gpu_options,
  )


def _make_io_maps(
    model: CompiledModel, signature_index: int = 0
) -> tuple[str, dict[str, TensorBuffer], dict[str, TensorBuffer]]:
  sig_info = model.get_signature_by_index(signature_index)
  signature_key = sig_info.get("key")
  if not signature_key:
    raise ValueError("Missing signature key for index 0")

  input_names = sig_info.get("inputs", [])
  output_names = sig_info.get("outputs", [])
  if not input_names or not output_names:
    raise ValueError("Missing input/output names in signature info")

  input_details = model.get_input_tensor_details(signature_key)
  input_map: dict[str, TensorBuffer] = {}
  for name in input_names:
    detail = input_details[name]
    data = np.zeros(detail["shape"], dtype=detail["dtype"])
    buf = model.create_input_buffer_by_name(signature_key, name)
    buf.write(data)
    input_map[name] = buf

  output_map = {
      name: model.create_output_buffer_by_name(signature_key, name)
      for name in output_names
  }
  return signature_key, input_map, output_map


def _summarize_outputs(
    model: CompiledModel,
    signature_key: str,
    output_map: dict[str, TensorBuffer],
) -> dict[str, dict[str, object]]:
  get_output_details = getattr(model, "get_output_tensor_details", None)
  output_details = (
      get_output_details(signature_key) if callable(get_output_details) else {}
  )
  summaries = {}
  for name, buf in output_map.items():
    detail = output_details.get(name)
    if detail is None:
      summaries[name] = {
          "shape": None,
          "dtype": "unknown",
          "verification": (
              "skipped (wheel does not expose output tensor details)"
          ),
      }
      continue
    np_dtype = _normalize_dtype(detail["dtype"])
    shape = list(detail["shape"])
    summary: dict[str, object] = {
        "shape": shape,
        "dtype": np_dtype.name,
    }
    num_elements = int(np.prod(shape, dtype=np.int64))
    readable_dtype = _readable_dtype(np_dtype)
    if num_elements and readable_dtype is not None:
      data = buf.read(num_elements, readable_dtype).reshape(shape)
      if np.issubdtype(data.dtype, np.floating):
        summary["sum"] = float(np.sum(data, dtype=np.float64))
        summary["max"] = float(np.max(data))
      else:
        summary["sum"] = int(np.sum(data, dtype=np.int64))
        summary["max"] = int(np.max(data))
    summaries[name] = summary
  return summaries


def _run_once(
    model_path: str,
    environment: Environment,
    options: Options,
    label: str,
    *,
    iterations: int,
    warmup_iterations: int,
):
  print(
      "\n==="
      f" {label} (hardware_accelerators={options.hardware_accelerators})"
      " ==="
  )
  try:
    with CompiledModel.from_file(
        model_path,
        environment=environment,
        options=options,
    ) as model:
      signature_key, input_map, output_map = _make_io_maps(model)
      for _ in range(warmup_iterations):
        model.run_by_name(signature_key, input_map, output_map)

      times_ms = []
      for _ in range(iterations):
        start = time.perf_counter()
        model.run_by_name(signature_key, input_map, output_map)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

      output_summaries = _summarize_outputs(model, signature_key, output_map)
      avg_ms = sum(times_ms) / len(times_ms)
      min_ms = min(times_ms)
      max_ms = max(times_ms)
      status_fields = [
          f"avg={avg_ms:.3f} ms",
          f"min={min_ms:.3f} ms",
          f"max={max_ms:.3f} ms",
      ]
      is_fully_accelerated = getattr(model, "is_fully_accelerated", None)
      if callable(is_fully_accelerated):
        status_fields.insert(0, f"fully_accelerated={is_fully_accelerated()}")
      print(f"OK: {', '.join(status_fields)}")
      print(f"Output verification: {pprint.pformat(output_summaries)}")
      return True, avg_ms
  except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")
    return False, None


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Smoke test LiteRT CPU/GPU accelerators"
  )
  parser.add_argument(
      "--model_path",
      required=True,
      help="Path to a .tflite model file",
  )
  parser.add_argument(
      "--iterations",
      type=int,
      default=1,
      help="Number of inference runs per accelerator (default: 1)",
  )
  parser.add_argument(
      "--warmup_iterations",
      type=int,
      default=1,
      help="Number of warmup runs before timing (default: 1)",
  )
  parser.add_argument(
      "--cpu_num_threads",
      type=int,
      default=1,
      help="Number of CPU threads to configure in per-model CpuOptions",
  )
  args = parser.parse_args()

  with _create_environment() as shared_environment:
    print(
        "Created shared LiteRT environment once and reusing it across all"
        " CompiledModel instances. CPU threads are set per model"
        f" (cpu_num_threads={args.cpu_num_threads})."
    )

    results = {}
    results["CPU"] = _run_once(
        args.model_path,
        shared_environment,
        _create_model_options(HardwareAccelerator.CPU, args.cpu_num_threads),
        "CPU",
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
    )
    results["GPU"] = _run_once(
        args.model_path,
        shared_environment,
        _create_model_options(HardwareAccelerator.GPU, args.cpu_num_threads),
        "GPU",
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
    )
    results["GPU|CPU"] = _run_once(
        args.model_path,
        shared_environment,
        _create_model_options(
            HardwareAccelerator.GPU | HardwareAccelerator.CPU,
            args.cpu_num_threads,
        ),
        "GPU|CPU (fallback)",
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
    )

    print("\n=== SUMMARY ===")
    for name, (ok, avg_ms) in results.items():
      if ok and avg_ms is not None:
        print(f"{name}: OK, avg={avg_ms:.3f} ms")
      else:
        print(f"{name}: FAILED")

    failed = [name for name, (ok, _) in results.items() if not ok]
    if failed:
      print(
          "Required accelerator smoke case(s) failed: " + ", ".join(failed),
          file=sys.stderr,
      )
      return 1
    return 0


if __name__ == "__main__":
  raise SystemExit(main())
