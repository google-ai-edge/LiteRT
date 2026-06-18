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

"""CPU-specific options for LiteRT compiled models."""

import dataclasses
import os
from typing import Any, Optional

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("ai_edge_litert", "cpu_options")
):
  from litert.python.litert_wrapper.compiled_model_wrapper import (
      cpu_kernel_mode,
  )

  CpuKernelMode = cpu_kernel_mode.CpuKernelMode
else:
  from ai_edge_litert.cpu_kernel_mode import CpuKernelMode
# pylint: enable=g-import-not-at-top


@dataclasses.dataclass
class CpuOptions:
  """CPU-specific options for a LiteRT compiled model.

  Attributes:
    num_threads: The number of threads to use for CPU inference. If 0, the
      LiteRT runtime will decide.
    kernel_mode: The CPU kernel mode to use. See `CpuKernelMode` for details.
    xnnpack_flags: Flags to pass to XNNPACK. Consult XNNPACK documentation for
      available flags.
    xnnpack_weight_cache_path: Path to a directory for caching XNNPACK weights.
  """

  num_threads: int = 0
  kernel_mode: Optional[CpuKernelMode] = None
  xnnpack_flags: Optional[int] = None
  xnnpack_weight_cache_path: str = ""

  def _as_flat_kwargs(self) -> dict[str, Any]:
    """Returns kwargs for the internal pybind wrapper."""
    return {
        "cpu_num_threads": self.num_threads,
        "cpu_kernel_mode": (
            -1 if self.kernel_mode is None else int(self.kernel_mode)
        ),
        "xnnpack_flags": (
            -1 if self.xnnpack_flags is None else self.xnnpack_flags
        ),
        "xnnpack_weight_cache_path": self.xnnpack_weight_cache_path,
    }
