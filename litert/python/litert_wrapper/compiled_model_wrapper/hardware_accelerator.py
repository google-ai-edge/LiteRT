# Copyright 2025 Google LLC.
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

"""Python wrapper for Hardware Accelerator enums."""

import enum


# TODO(b/410257592): Deprecate this class and use pybind's flag enum.
class HardwareAccelerator(enum.IntFlag):
  """Constants representing hardware acceleration types.

  These values correspond to the C++ LiteRtHwAccelerators enum defined in
  litert/c/litert_common.h. They are bit flags that can be combined using
  bitwise OR (|).

  IMPORTANT: Using GPU or NPU alone may fail if some ops are not supported
  by that accelerator. For robust execution, combine with CPU as fallback:
    hardware_accel=HardwareAccelerator.GPU | HardwareAccelerator.CPU

  Attributes:
    CPU: Use CPU for inference (value: 1, bit 0). Always works.
    GPU: Use GPU for inference with WebGPU/OpenCL/Metal backend (value: 2, bit
      1). May fail if model has ops unsupported by GPU; combine with CPU for
      fallback.
    NPU: Use NPU/TPU for inference if available (value: 4, bit 2).

  Example usage:
    # CPU only (safe default)
    model = CompiledModel.from_file("model.tflite",
        hardware_accel=HardwareAccelerator.CPU)

    # GPU only (may fail for some models)
    model = CompiledModel.from_file("model.tflite",
        hardware_accel=HardwareAccelerator.GPU)

    # GPU with CPU fallback (recommended for GPU acceleration)
    model = CompiledModel.from_file("model.tflite",
        hardware_accel=HardwareAccelerator.GPU | HardwareAccelerator.CPU)

    # NPU with GPU and CPU fallback
    model = CompiledModel.from_file("model.tflite",
        hardware_accel=HardwareAccelerator.NPU | HardwareAccelerator.GPU |
                       HardwareAccelerator.CPU)
  """

  CPU = enum.auto()
  GPU = enum.auto()
  NPU = enum.auto()
