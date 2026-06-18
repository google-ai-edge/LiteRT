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

"""Qualcomm-specific options for LiteRT compiled models."""

from collections.abc import Sequence
import dataclasses
import enum
import os
from typing import Any, Optional

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("ai_edge_litert", "qualcomm_options")
):
  from litert.python.litert_wrapper.compiled_model_wrapper import option_utils
else:
  from ai_edge_litert import option_utils
# pylint: enable=g-import-not-at-top


class QualcommLogLevel(enum.IntEnum):
  """Qualcomm SDK log levels."""

  OFF = 0
  ERROR = 1
  WARN = 2
  INFO = 3
  VERBOSE = 4
  DEBUG = 5


class QualcommHtpPerformanceMode(enum.IntEnum):
  """Qualcomm HTP performance modes."""

  DEFAULT = 0
  SUSTAINED_HIGH_PERFORMANCE = 1
  BURST = 2
  HIGH_PERFORMANCE = 3
  POWER_SAVER = 4
  LOW_POWER_SAVER = 5
  HIGH_POWER_SAVER = 6
  LOW_BALANCED = 7
  BALANCED = 8
  EXTREME_POWER_SAVER = 9


class QualcommDspPerformanceMode(enum.IntEnum):
  """Qualcomm DSP performance modes."""

  DEFAULT = 0
  SUSTAINED_HIGH_PERFORMANCE = 1
  BURST = 2
  HIGH_PERFORMANCE = 3
  POWER_SAVER = 4
  LOW_POWER_SAVER = 5
  HIGH_POWER_SAVER = 6
  LOW_BALANCED = 7
  BALANCED = 8


class QualcommProfiling(enum.IntEnum):
  """Qualcomm profiling levels."""

  OFF = 0
  BASIC = 1
  DETAILED = 2
  LINTING = 3
  OPTRACE = 4


class QualcommOptimizationLevel(enum.IntEnum):
  """Qualcomm graph optimization levels."""

  OPTIMIZE_FOR_INFERENCE = 0
  OPTIMIZE_FOR_PREPARE = 1
  OPTIMIZE_FOR_INFERENCE_O3 = 2


class QualcommGraphPriority(enum.IntEnum):
  """Qualcomm graph priorities."""

  DEFAULT = 0
  LOW = 1
  NORMAL = 2
  NORMAL_HIGH = 3
  HIGH = 4


class QualcommBackend(enum.IntEnum):
  """Qualcomm backend choices."""

  UNDEFINED = 0
  GPU = 1
  HTP = 2
  DSP = 3
  IR = 4


class QualcommGraphIOTensorMemType(enum.IntEnum):
  """Qualcomm graph I/O tensor memory types."""

  RAW = 0
  MEM_HANDLE = 1


@dataclasses.dataclass
class QualcommOptions:
  """Qualcomm-specific options for a LiteRT compiled model."""

  LOG_LEVEL = QualcommLogLevel
  HTP_PERFORMANCE_MODE = QualcommHtpPerformanceMode
  DSP_PERFORMANCE_MODE = QualcommDspPerformanceMode
  PROFILING = QualcommProfiling
  OPTIMIZATION_LEVEL = QualcommOptimizationLevel
  GRAPH_PRIORITY = QualcommGraphPriority
  BACKEND = QualcommBackend
  GRAPH_IO_TENSOR_MEM_TYPE = QualcommGraphIOTensorMemType

  log_level: Optional[QualcommLogLevel] = None
  htp_performance_mode: Optional[QualcommHtpPerformanceMode] = None
  dsp_performance_mode: Optional[QualcommDspPerformanceMode] = None
  use_int64_bias_as_int32: Optional[bool] = None
  enable_weight_sharing: Optional[bool] = None
  use_conv_hmx: Optional[bool] = None
  use_fold_relu: Optional[bool] = None
  profiling: Optional[QualcommProfiling] = None
  dump_tensor_ids: Optional[Sequence[int]] = None
  ir_json_dir: str = ""
  dlc_dir: str = ""
  vtcm_size: Optional[int] = None
  num_hvx_threads: Optional[int] = None
  optimization_level: Optional[QualcommOptimizationLevel] = None
  graph_priority: Optional[QualcommGraphPriority] = None
  backend: Optional[QualcommBackend] = None
  saver_output_dir: str = ""
  graph_io_tensor_mem_type: Optional[QualcommGraphIOTensorMemType] = None

  def _as_flat_kwargs(self) -> dict[str, Any]:
    """Returns kwargs for the internal pybind wrapper."""
    return {
        "qualcomm_log_level": option_utils.optional_enum_to_int(self.log_level),
        "qualcomm_htp_performance_mode": option_utils.optional_enum_to_int(
            self.htp_performance_mode
        ),
        "qualcomm_dsp_performance_mode": option_utils.optional_enum_to_int(
            self.dsp_performance_mode
        ),
        "qualcomm_use_int64_bias_as_int32": option_utils.optional_bool_to_int(
            self.use_int64_bias_as_int32
        ),
        "qualcomm_enable_weight_sharing": option_utils.optional_bool_to_int(
            self.enable_weight_sharing
        ),
        "qualcomm_use_conv_hmx": option_utils.optional_bool_to_int(
            self.use_conv_hmx
        ),
        "qualcomm_use_fold_relu": option_utils.optional_bool_to_int(
            self.use_fold_relu
        ),
        "qualcomm_profiling": option_utils.optional_enum_to_int(self.profiling),
        "qualcomm_has_dump_tensor_ids": self.dump_tensor_ids is not None,
        "qualcomm_dump_tensor_ids": list(self.dump_tensor_ids or []),
        "qualcomm_ir_json_dir": self.ir_json_dir,
        "qualcomm_dlc_dir": self.dlc_dir,
        "qualcomm_vtcm_size": -1 if self.vtcm_size is None else self.vtcm_size,
        "qualcomm_num_hvx_threads": (
            -1 if self.num_hvx_threads is None else self.num_hvx_threads
        ),
        "qualcomm_optimization_level": option_utils.optional_enum_to_int(
            self.optimization_level
        ),
        "qualcomm_graph_priority": option_utils.optional_enum_to_int(
            self.graph_priority
        ),
        "qualcomm_backend": option_utils.optional_enum_to_int(self.backend),
        "qualcomm_saver_output_dir": self.saver_output_dir,
        "qualcomm_graph_io_tensor_mem_type": option_utils.optional_enum_to_int(
            self.graph_io_tensor_mem_type
        ),
    }
