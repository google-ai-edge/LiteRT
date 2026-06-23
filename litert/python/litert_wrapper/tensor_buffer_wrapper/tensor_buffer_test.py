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

"""Unit tests for LiteRT TensorBuffer dtype handling."""

import unittest

import ml_dtypes
import numpy as np

from litert.python.litert_wrapper.tensor_buffer_wrapper import tensor_buffer


_LITERT_HOST_MEMORY_BUFFER_ALIGNMENT = 64


def _aligned_array(shape, dtype):
  dtype = np.dtype(dtype)
  num_bytes = int(np.prod(shape)) * dtype.itemsize
  storage = np.empty(
      num_bytes + _LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, dtype=np.uint8
  )
  offset = (-storage.ctypes.data) % _LITERT_HOST_MEMORY_BUFFER_ALIGNMENT
  return np.ndarray(shape, dtype=dtype, buffer=storage, offset=offset)


class TensorBufferDTypeTest(unittest.TestCase):

  def assert_array_bytes_equal(self, actual, expected):
    self.assertEqual(actual.dtype, expected.dtype)
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))

  def test_normalize_ml_dtypes(self):
    cases = (
        (ml_dtypes.bfloat16, "bfloat16"),
        (np.dtype(ml_dtypes.bfloat16), "bfloat16"),
        (ml_dtypes.float8_e4m3fn, "float8_e4m3fn"),
        (np.dtype(ml_dtypes.float8_e4m3fn), "float8_e4m3fn"),
        (ml_dtypes.float8_e5m2, "float8_e5m2"),
        (np.dtype(ml_dtypes.float8_e5m2), "float8_e5m2"),
    )

    for dtype, expected_name in cases:
      with self.subTest(dtype=dtype):
        dtype_name, np_dtype = tensor_buffer.TensorBuffer._normalize_dtype(
            dtype
        )

        self.assertEqual(dtype_name, expected_name)
        self.assertEqual(np_dtype, np.dtype(dtype))

  def test_normalize_ml_dtype_array_for_zero_copy(self):
    cases = (
        (ml_dtypes.bfloat16, "bfloat16"),
        (ml_dtypes.float8_e4m3fn, "float8_e4m3fn"),
        (ml_dtypes.float8_e5m2, "float8_e5m2"),
    )

    for dtype, expected_name in cases:
      with self.subTest(dtype=dtype):
        data = np.zeros((2, 3), dtype=dtype)

        array, dtype_name = (
            tensor_buffer.TensorBuffer._normalize_numpy_array(
                data, zero_copy=True
            )
        )

        self.assertIs(array, data)
        self.assertEqual(dtype_name, expected_name)

  def test_sub_byte_ml_dtypes_are_not_tensor_buffer_dtypes_yet(self):
    for dtype in (
        ml_dtypes.int2,
        ml_dtypes.uint2,
        ml_dtypes.int4,
        ml_dtypes.uint4,
    ):
      with self.subTest(dtype=dtype):
        with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
          tensor_buffer.TensorBuffer._normalize_dtype(dtype)

  def test_create_from_host_memory_preserves_byte_aligned_ml_dtypes(self):
    cases = (
        (ml_dtypes.bfloat16, "bfloat16"),
        (ml_dtypes.float8_e4m3fn, "float8_e4m3fn"),
        (ml_dtypes.float8_e5m2, "float8_e5m2"),
    )

    for dtype, expected_name in cases:
      with self.subTest(dtype=dtype):
        data = _aligned_array((3,), dtype)
        data[...] = np.array([1.0, -2.0, 0.5], dtype=dtype)
        tb = tensor_buffer.TensorBuffer.create_from_host_memory(data)
        self.addCleanup(tb.destroy)

        self.assertEqual(
            tb.get_tensor_details(), {"dtype": expected_name, "shape": [3]}
        )
        self.assert_array_bytes_equal(tb.read(data.size, dtype), data)

  def test_write_and_read_preserve_byte_aligned_ml_dtypes(self):
    for dtype in (
        ml_dtypes.bfloat16,
        ml_dtypes.float8_e4m3fn,
        ml_dtypes.float8_e5m2,
    ):
      with self.subTest(dtype=dtype):
        backing = _aligned_array((4,), dtype)
        backing[...] = np.zeros((4,), dtype=dtype)
        tb = tensor_buffer.TensorBuffer.create_from_host_memory(backing)
        self.addCleanup(tb.destroy)
        expected = np.array([1.0, -1.0, 0.5, -0.5], dtype=dtype)

        tb.write(expected)

        self.assert_array_bytes_equal(tb.read(expected.size, dtype), expected)


if __name__ == "__main__":
  unittest.main()
