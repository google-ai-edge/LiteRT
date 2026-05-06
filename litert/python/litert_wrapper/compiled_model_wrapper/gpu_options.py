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

"""GPU-specific options for LiteRT compiled models."""

import dataclasses
from typing import Any


@dataclasses.dataclass
class GpuOptions:
  """GPU-specific options for a LiteRT compiled model."""

  enforce_f32: bool = False
  constant_tensor_sharing: bool = False
  infinite_float_capping: bool = False
  benchmark_mode: bool = False
  allow_src_quantized_fc_conv_ops: bool = False
  hint_waiting_for_completion: bool = False

  def _as_flat_kwargs(self) -> dict[str, Any]:
    """Returns kwargs for the internal pybind wrapper."""
    return {
        "gpu_enforce_f32": self.enforce_f32,
        "gpu_share_constant_tensors": self.constant_tensor_sharing,
        "enable_constant_tensor_sharing": self.constant_tensor_sharing,
        "enable_infinite_float_capping": self.infinite_float_capping,
        "enable_benchmark_mode": self.benchmark_mode,
        "enable_allow_src_quantized_fc_conv_ops": (
            self.allow_src_quantized_fc_conv_ops
        ),
        "enable_hint_waiting_for_completion": self.hint_waiting_for_completion,
    }
