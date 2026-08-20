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

"""Reference Python implementation for Gemma 4 Feed Forward Network.

Computes the expected reference values for Feed Forward Network unit tests.
"""

import numpy as np
from tensor.examples.gemma4.helpers.reference import utils


def gelu_approx(x: np.ndarray) -> np.ndarray:
  """Computes approximate GELU activation function using Tanh approximation.

  Args:
    x: Input tensor.

  Returns:
    Output tensor after approximate GELU activation.
  """
  return (
      0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))
  )


def feed_forward_network(
    x: np.ndarray,
    gate_proj: np.ndarray,
    up_proj: np.ndarray,
    down_proj: np.ndarray,
) -> np.ndarray:
  """Computes Feed Forward Network (FFN) layer output.

  Args:
    x: Input tensor of shape [..., embed_dim].
    gate_proj: Gate projection matrix of shape [hidden_dim, embed_dim].
    up_proj: Up projection matrix of shape [hidden_dim, embed_dim].
    down_proj: Down projection matrix of shape [embed_dim, hidden_dim].

  Returns:
    Output tensor of shape [..., embed_dim].
  """
  up = np.matmul(x, up_proj.T)
  gate_proj_out = np.matmul(x, gate_proj.T)
  gate = gelu_approx(gate_proj_out)
  mul_out = up * gate
  return np.matmul(mul_out, down_proj.T)


def main() -> None:
  x = np.array([1.0, 2.0], dtype=np.float32)
  gate_proj = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float32)
  up_proj = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
  down_proj = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32)

  output = feed_forward_network(x, gate_proj, up_proj, down_proj)

  utils.cpp_print("Input:", x)
  utils.cpp_print("Gate Proj:", gate_proj)
  utils.cpp_print("Up Proj:", up_proj)
  utils.cpp_print("Down Proj:", down_proj)
  utils.cpp_print("\nOutput:", output)


if __name__ == "__main__":
  main()
