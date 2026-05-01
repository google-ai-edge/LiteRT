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

"""Python wrapper for LiteRT environments."""

import os
from typing import Optional

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("ai_edge_litert", "environment")
):
  from litert.python.litert_wrapper.environment_wrapper import (
      _pywrap_litert_environment_wrapper as _env,
  )
else:
  from ai_edge_litert import _pywrap_litert_environment_wrapper as _env
# pylint: enable=g-import-not-at-top


class Environment:
  """Python wrapper for a shared LiteRT environment."""

  def __init__(
      self,
      capsule,
      cpu_num_threads: int = 0,
      gpu_enforce_f32: bool = False,
      gpu_share_constant_tensors: bool = False,
      cpu_kernel_mode: int = -1,
      xnnpack_flags: int = -1,
      xnnpack_weight_cache_path: str = "",
      enable_constant_tensor_sharing: bool = False,
      enable_infinite_float_capping: bool = False,
      enable_benchmark_mode: bool = False,
      enable_allow_src_quantized_fc_conv_ops: bool = False,
  ):
    self._capsule = capsule
    self.cpu_num_threads = cpu_num_threads
    self.gpu_enforce_f32 = gpu_enforce_f32
    self.gpu_share_constant_tensors = gpu_share_constant_tensors
    self.cpu_kernel_mode = cpu_kernel_mode
    self.xnnpack_flags = xnnpack_flags
    self.xnnpack_weight_cache_path = xnnpack_weight_cache_path
    self.enable_constant_tensor_sharing = enable_constant_tensor_sharing
    self.enable_infinite_float_capping = enable_infinite_float_capping
    self.enable_benchmark_mode = enable_benchmark_mode
    self.enable_allow_src_quantized_fc_conv_ops = (
        enable_allow_src_quantized_fc_conv_ops
    )

  @classmethod
  def create(
      cls,
      runtime_path: Optional[str] = None,
      compiler_plugin_path: str = "",
      dispatch_library_path: str = "",
      cpu_num_threads: int = 1,
      gpu_enforce_f32: bool = False,
      gpu_share_constant_tensors: bool = False,
      cpu_kernel_mode: int = -1,
      xnnpack_flags: int = -1,
      xnnpack_weight_cache_path: str = "",
      enable_constant_tensor_sharing: bool = False,
      enable_infinite_float_capping: bool = False,
      enable_benchmark_mode: bool = False,
      enable_allow_src_quantized_fc_conv_ops: bool = False,
  ) -> "Environment":
    """Creates a reusable LiteRT environment.

    Args:
      runtime_path: Optional path to the LiteRT runtime library directory.
        Defaults to the directory containing the Python wheel modules.
      compiler_plugin_path: Optional path to compiler plugin libraries.
      dispatch_library_path: Optional path to dispatch libraries.
      cpu_num_threads: Number of threads for CPU execution.
      gpu_enforce_f32: Enforce F32 precision on GPU.
      gpu_share_constant_tensors: Share constant tensors among subgraphs on GPU.
      cpu_kernel_mode: CPU kernel mode option.
      xnnpack_flags: XNNPACK flags option.
      xnnpack_weight_cache_path: XNNPACK weight cache path option.
      enable_constant_tensor_sharing: Enable constant tensor sharing on GPU.
      enable_infinite_float_capping: Enable infinite float capping on GPU.
      enable_benchmark_mode: Enable benchmark mode on GPU.
      enable_allow_src_quantized_fc_conv_ops: Enable allow src quantized fc conv
        ops on GPU.

    Returns:
      A new Environment instance.
    """
    if runtime_path is None:
      runtime_path = os.path.dirname(os.path.abspath(__file__))
    capsule = _env.CreateEnvironment(
        runtime_path=runtime_path,
        compiler_plugin_path=compiler_plugin_path,
        dispatch_library_path=dispatch_library_path,
    )
    return cls(
        capsule,
        cpu_num_threads,
        gpu_enforce_f32,
        gpu_share_constant_tensors,
        cpu_kernel_mode,
        xnnpack_flags,
        xnnpack_weight_cache_path,
        enable_constant_tensor_sharing,
        enable_infinite_float_capping,
        enable_benchmark_mode,
        enable_allow_src_quantized_fc_conv_ops,
    )

  @property
  def capsule(self):
    if self._capsule is None:
      raise ValueError("Environment is no longer valid")
    return self._capsule
