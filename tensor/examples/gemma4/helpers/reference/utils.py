# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for reference Python implementations."""

import numpy as np


def generate_angles(head_dim: int) -> np.ndarray:
  """Generate angles based on the head dimension.

  Args:
    head_dim: Input tensor last dimension.

  Returns:
    An array holding the angles to pass to the RoPE function.
  """
  half_dim = head_dim // 2
  return np.tile(
      [(i + 1) * np.pi / (half_dim * 3) for i in range(half_dim)],
      2,
  )


def cpp_print(*args, **kwargs) -> None:
  """Forwards all arguments to `print`.

  Flattens and joins all np.ndarray into a string that can be copied into a C++
  program.

  Args:
    *args: Position arguments forwarded to `print`.
    **kwargs: Named arguments forwarded to `print`.
  """
  print(
      *(
          a
          if not isinstance(a, np.ndarray)
          else ", ".join(f"{v:.7f}f" for v in a.flat)
          for a in args
      ),
      **kwargs,
  )
