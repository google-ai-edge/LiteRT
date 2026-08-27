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

"""Reference Python implementation for Gemma 4 RmsNorm.

Computes the expected reference values for RmsNorm unit tests.
"""

import numpy as np


def rms_norm(
    x: np.ndarray, scale: np.ndarray | None = None, eps: float = 1e-6
) -> tuple[float, np.ndarray]:
  """Computes RMSNorm given input array x, scale array, and epsilon.

  Args:
    x: Input.
    scale: Scale factors.
    eps: Small constant for numerical stability.

  Returns:
    A tuple of (rms_denominator, normalized_output_array).
  """
  var = np.mean(x**2, axis=-1, keepdims=True)
  rms = np.sqrt(var + eps)
  output = (x / rms) * (scale if scale is not None else 1.0)
  return float(rms.flat[0]), output


def main() -> None:
  input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
  scale_data = np.array([1.0, 1.3, 0.9, 1.5], dtype=np.float32)
  eps = 1e-6

  # With Scale (RmsNormTest, RmsNormWithAttributeEpsilonTest)
  rms_with_scale, output_with_scale = rms_norm(input_data, scale_data, eps)

  print("--- With Scale (RmsNormTest, RmsNormWithAttributeEpsilonTest) ---")
  print(f"Input: {input_data.tolist()}")
  print(f"Scale: {scale_data.tolist()}")
  print(f"Epsilon: {eps}")
  print(f"RMS Denominator: {rms_with_scale}")
  print(
      f"C++ array: {{{', '.join([f'{o:.6f}f' for o in output_with_scale])}}};")
  print()

  # Without Scale (RmsNormNoScaleTest)
  rms_no_scale, output_no_scale = rms_norm(input_data, None, eps)

  print("---  Without Scale (RmsNormNoScaleTest) ---")
  print(f"Input: {input_data.tolist()}")
  print(f"Epsilon: {eps}")
  print(f"RMS Denominator: {rms_no_scale}")
  print(f"C++ array: {{{', '.join([f'{o:.6f}f' for o in output_no_scale])}}};")


if __name__ == "__main__":
  main()
