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

"""Grouped Python options for LiteRT compiled models."""

import dataclasses
import os
from typing import Any, Optional

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("ai_edge_litert", "options")
):
  from litert.python.litert_wrapper.compiled_model_wrapper import (
      hardware_accelerator,
  )
  from litert.python.litert_wrapper.compiled_model_wrapper import (
      cpu_options as cpu_options_lib,
  )
  from litert.python.litert_wrapper.compiled_model_wrapper import (
      gpu_options as gpu_options_lib,
  )
  from litert.python.litert_wrapper.compiled_model_wrapper import (
      intel_openvino_options as intel_openvino_options_lib,
  )
  from litert.python.litert_wrapper.compiled_model_wrapper import (
      qualcomm_options as qualcomm_options_lib,
  )

  HardwareAccelerator = hardware_accelerator.HardwareAccelerator
else:
  from ai_edge_litert.hardware_accelerator import HardwareAccelerator
  from ai_edge_litert import cpu_options as cpu_options_lib
  from ai_edge_litert import gpu_options as gpu_options_lib
  from ai_edge_litert import intel_openvino_options as intel_openvino_options_lib
  from ai_edge_litert import qualcomm_options as qualcomm_options_lib
# pylint: enable=g-import-not-at-top

CpuOptions = cpu_options_lib.CpuOptions
CpuKernelMode = cpu_options_lib.CpuKernelMode
GpuOptions = gpu_options_lib.GpuOptions
QualcommOptions = qualcomm_options_lib.QualcommOptions
QualcommLogLevel = qualcomm_options_lib.QualcommLogLevel
QualcommHtpPerformanceMode = qualcomm_options_lib.QualcommHtpPerformanceMode
QualcommDspPerformanceMode = qualcomm_options_lib.QualcommDspPerformanceMode
QualcommProfiling = qualcomm_options_lib.QualcommProfiling
QualcommOptimizationLevel = qualcomm_options_lib.QualcommOptimizationLevel
QualcommGraphPriority = qualcomm_options_lib.QualcommGraphPriority
QualcommBackend = qualcomm_options_lib.QualcommBackend
QualcommGraphIOTensorMemType = qualcomm_options_lib.QualcommGraphIOTensorMemType
IntelOpenVinoOptions = intel_openvino_options_lib.IntelOpenVinoOptions
IntelOpenVinoDeviceType = intel_openvino_options_lib.IntelOpenVinoDeviceType
IntelOpenVinoPerformanceMode = (
    intel_openvino_options_lib.IntelOpenVinoPerformanceMode
)


@dataclasses.dataclass
class Options:
  """Per-model compilation options for LiteRT CompiledModel creation."""

  hardware_accelerators: HardwareAccelerator = HardwareAccelerator.CPU
  _cpu_options: Optional[CpuOptions] = None
  _gpu_options: Optional[GpuOptions] = None
  _qualcomm_options: Optional[QualcommOptions] = None
  _intel_openvino_options: Optional[IntelOpenVinoOptions] = None

  def __init__(
      self,
      hardware_accelerators: HardwareAccelerator = HardwareAccelerator.CPU,
      cpu_options: Optional[CpuOptions] = None,
      gpu_options: Optional[GpuOptions] = None,
      qualcomm_options: Optional[QualcommOptions] = None,
      intel_openvino_options: Optional[IntelOpenVinoOptions] = None,
  ):
    self.hardware_accelerators = hardware_accelerators
    self._cpu_options = cpu_options
    self._gpu_options = gpu_options
    self._qualcomm_options = qualcomm_options
    self._intel_openvino_options = intel_openvino_options

  @classmethod
  def create(cls) -> "Options":
    """Creates a new options object, mirroring the native C++ factory."""
    return cls()

  @property
  def cpu_options(self) -> CpuOptions:
    """Returns mutable CPU options, creating them if needed."""
    if self._cpu_options is None:
      self._cpu_options = CpuOptions()
    return self._cpu_options

  @property
  def gpu_options(self) -> GpuOptions:
    """Returns mutable GPU options, creating them if needed."""
    if self._gpu_options is None:
      self._gpu_options = GpuOptions()
    return self._gpu_options

  @property
  def qualcomm_options(self) -> QualcommOptions:
    """Returns mutable Qualcomm options, creating them if needed."""
    if self._qualcomm_options is None:
      self._qualcomm_options = QualcommOptions()
    return self._qualcomm_options

  @property
  def intel_openvino_options(self) -> IntelOpenVinoOptions:
    """Returns mutable Intel OpenVINO options, creating them if needed."""
    if self._intel_openvino_options is None:
      self._intel_openvino_options = IntelOpenVinoOptions()
    return self._intel_openvino_options

  def _as_flat_kwargs(self) -> dict[str, Any]:
    """Returns kwargs for the internal pybind wrapper."""
    kwargs = {
        "hardware_accel": int(self.hardware_accelerators),
        "cpu_num_threads": 0,
        "cpu_kernel_mode": -1,
        "xnnpack_flags": -1,
        "xnnpack_weight_cache_path": "",
        "gpu_enforce_f32": False,
        "gpu_share_constant_tensors": False,
        "enable_constant_tensor_sharing": False,
        "enable_infinite_float_capping": False,
        "enable_benchmark_mode": False,
        "enable_allow_src_quantized_fc_conv_ops": False,
        "enable_hint_waiting_for_completion": False,
        "qualcomm_log_level": -1,
        "qualcomm_htp_performance_mode": -1,
        "qualcomm_dsp_performance_mode": -1,
        "qualcomm_use_int64_bias_as_int32": -1,
        "qualcomm_enable_weight_sharing": -1,
        "qualcomm_use_conv_hmx": -1,
        "qualcomm_use_fold_relu": -1,
        "qualcomm_profiling": -1,
        "qualcomm_has_dump_tensor_ids": False,
        "qualcomm_dump_tensor_ids": [],
        "qualcomm_ir_json_dir": "",
        "qualcomm_dlc_dir": "",
        "qualcomm_vtcm_size": -1,
        "qualcomm_num_hvx_threads": -1,
        "qualcomm_optimization_level": -1,
        "qualcomm_graph_priority": -1,
        "qualcomm_backend": -1,
        "qualcomm_saver_output_dir": "",
        "qualcomm_graph_io_tensor_mem_type": -1,
        "intel_openvino_device_type": -1,
        "intel_openvino_performance_mode": -1,
        "intel_openvino_configs_map": {},
    }
    if self._cpu_options is not None:
      kwargs.update(self._cpu_options._as_flat_kwargs())
    if self._gpu_options is not None:
      kwargs.update(self._gpu_options._as_flat_kwargs())
    if self._qualcomm_options is not None:
      kwargs.update(self._qualcomm_options._as_flat_kwargs())
    if self._intel_openvino_options is not None:
      kwargs.update(self._intel_openvino_options._as_flat_kwargs())
    return kwargs
