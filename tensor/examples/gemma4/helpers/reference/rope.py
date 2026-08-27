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

"""Reference Python implementation for Gemma 4 RoPE.

Computes the expected reference values for RoPE unit tests.
"""

import numpy as np
from tensor.examples.gemma4.helpers.reference import utils


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
  """Computes split-half Rotary Position Embedding (RoPE).

  Args:
    x: Input tensor.
    cos: Cosine embedding tensor.
    sin: Sine embedding tensor.

  Returns:
    Rotated output tensor.
  """
  x1, x2 = np.split(x, 2, axis=-1)
  rotated = np.concatenate([-x2, x1], axis=-1)
  return x * cos + rotated * sin


def main() -> None:
  x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
  angles = utils.generate_angles(x.shape[-1])
  cos = np.cos(angles)
  sin = np.sin(angles)

  output = rope(x, cos, sin)

  utils.cpp_print("Input:", x)
  utils.cpp_print("Angles:", angles)
  utils.cpp_print("Cos:", cos)
  utils.cpp_print("Sin:", sin)
  utils.cpp_print("\nOutput:", output)


if __name__ == "__main__":
  main()
