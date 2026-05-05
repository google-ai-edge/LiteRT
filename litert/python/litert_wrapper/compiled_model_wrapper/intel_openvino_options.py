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

"""Intel OpenVINO-specific options for LiteRT compiled models."""

import dataclasses
import enum
import os
from typing import Any, Optional

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("ai_edge_litert", "intel_openvino_options")
):
  from litert.python.litert_wrapper.compiled_model_wrapper import option_utils
else:
  from ai_edge_litert import option_utils
# pylint: enable=g-import-not-at-top


class IntelOpenVinoDeviceType(enum.IntEnum):
  """Intel OpenVINO device types."""

  CPU = 0
  GPU = 1
  NPU = 2
  AUTO = 3


class IntelOpenVinoPerformanceMode(enum.IntEnum):
  """Intel OpenVINO performance modes."""

  LATENCY = 0
  THROUGHPUT = 1
  CUMULATIVE_THROUGHPUT = 2


@dataclasses.dataclass
class IntelOpenVinoOptions:
  """Intel OpenVINO-specific options for a LiteRT compiled model."""

  DEVICE_TYPE = IntelOpenVinoDeviceType
  PERFORMANCE_MODE = IntelOpenVinoPerformanceMode

  device_type: Optional[IntelOpenVinoDeviceType] = None
  performance_mode: Optional[IntelOpenVinoPerformanceMode] = None
  configs_map: dict[str, str] = dataclasses.field(default_factory=dict)

  def _as_flat_kwargs(self) -> dict[str, Any]:
    """Returns kwargs for the internal pybind wrapper."""
    return {
        "intel_openvino_device_type": option_utils.optional_enum_to_int(
            self.device_type
        ),
        "intel_openvino_performance_mode": option_utils.optional_enum_to_int(
            self.performance_mode
        ),
        "intel_openvino_configs_map": dict(self.configs_map),
    }
