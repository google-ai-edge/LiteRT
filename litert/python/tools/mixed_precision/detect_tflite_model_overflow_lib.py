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
"""Core logic for detecting TFLite model overflow."""

import numpy as np


def generate_tensor_data(shape, dtype):
  """Generates a numpy array with data based on shape and dtype."""
  # If shape is empty or contains None, default to a single element for scalar
  # pylint: disable=g-explicit-length-test
  if shape is None or len(shape) == 0:
    total_elements = 1
    shape_to_reshape = ()
  elif None in shape:
    shape_to_reshape = [1 if d is None else d for d in shape]
    total_elements = int(np.prod(shape_to_reshape))
  else:
    total_elements = int(np.prod(shape))
    shape_to_reshape = shape

  if np.issubdtype(dtype, np.floating):
    rng = np.random.RandomState(7)
    # Use sin() to generate floating point values strictly between -1 and 1
    data = np.sin(rng.randn(total_elements) * 100.0)
  elif dtype == np.int32:
    rng = np.random.RandomState(7)
    data = rng.randint(0, 10, size=total_elements)
  elif dtype == np.int16:
    data = np.arange(total_elements) % 2048
  elif dtype == np.int64:
    data = np.arange(total_elements) % 2048
  elif dtype == np.int8:
    data = np.arange(total_elements) % 256 - 128
  elif dtype == np.uint8:
    data = np.arange(total_elements) % 256
  elif dtype == np.bool_:
    rng = np.random.RandomState(7)
    data = rng.randint(0, 2, size=total_elements, dtype=bool)
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')

  return data.astype(dtype).reshape(shape_to_reshape)


def check_tensor_overflow(tensor, name):
  """Checks if a tensor contains inf, nan, or huge values."""
  if not np.issubdtype(tensor.dtype, np.floating):
    return False
  has_nan = np.isnan(tensor).any()
  has_inf = np.isinf(tensor).any()
  # Using an arbitrary threshold for overflow
  # (e.g. 1e10 is usually very large for normalized ML floats)
  # cast to float32 to avoid RuntimeWarning when comparing float16 with 1e10
  tensor_f32 = tensor.astype(np.float32)
  has_large = np.max(np.abs(tensor_f32)) > 1e10 if tensor.size > 0 else False
  if has_nan or has_inf or has_large:
    print(f'Overflow detected in tensor: {name}')
    if has_nan:
      print('  - Contains NaN')
    if has_inf:
      print('  - Contains Inf')
    if has_large:
      print(f'  - Large values detected, max abs: {np.max(np.abs(tensor))}')
    return True
  return False
