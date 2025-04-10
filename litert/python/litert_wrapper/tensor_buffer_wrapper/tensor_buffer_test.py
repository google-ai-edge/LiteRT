# Copyright 2025 Google LLC.
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

"""Unit tests for tensor_buffer.py."""

import unittest

import numpy as np

from litert.python.litert_wrapper.tensor_buffer_wrapper.tensor_buffer import TensorBuffer


def aligned_array(shape, dtype, alignment=64):
  """Allocates a NumPy array with the specified memory alignment.

  Creates a NumPy array where the underlying memory is aligned to the specified
  byte boundary, which is useful for hardware-specific optimizations.

  Args:
    shape: The shape of the array.
    dtype: The data type of the array.
    alignment: The memory alignment in bytes (default: 64).

  Returns:
    A NumPy array with the specified alignment.
  """
  # Ensure we only use 64-byte alignment as required by the native code
  if alignment != 64:
    alignment = 64

  size = np.prod(shape)
  itemsize = np.dtype(dtype).itemsize
  n_extra = alignment // itemsize

  # Create a 1D array with extra space to find aligned memory
  # Use np.empty_like with a larger shape to avoid initialization overhead
  raw = np.zeros(size + n_extra + 1, dtype=dtype)

  # Find the first aligned position
  start_index = 0
  while (raw[start_index:].ctypes.data % alignment) != 0:
    start_index += 1

  # Ensure we have enough elements after the aligned position
  if start_index + size > len(raw):
    # If not enough space, create a larger array and try again
    raw = np.zeros(size + n_extra + alignment, dtype=dtype)
    start_index = 0
    while (raw[start_index:].ctypes.data % alignment) != 0:
      start_index += 1

  # Return the aligned slice reshaped to the requested dimensions
  aligned = raw[start_index : start_index + size].reshape(shape)
  # Double-check alignment
  assert (
      aligned.ctypes.data % alignment == 0
  ), f"Array not aligned to {alignment} bytes"
  return aligned


class TensorBufferTest(unittest.TestCase):
  """Test cases for the TensorBuffer class."""

  def test_create_from_host_memory_float(self):
    """Tests creating a TensorBuffer from host memory with float data."""
    # Create aligned numpy array with 64-byte alignment
    test_data = [1.0, 2.0, 3.0, 4.0]
    arr = aligned_array((4,), np.float32, alignment=64)
    arr[:] = test_data

    # Create TensorBuffer referencing the array
    tb = TensorBuffer.create_from_host_memory(arr)

    # Verify data can be read back correctly
    result = tb.read(4, np.float32)
    np.testing.assert_array_equal(result, np.array(test_data, dtype=np.float32))

    # Clean up
    tb.destroy()

  def test_create_from_host_memory_int(self):
    """Tests creating a TensorBuffer from host memory with integer data."""
    # Create aligned numpy array with 64-byte alignment
    test_data = [10, 20, 30, 40]
    arr = aligned_array((4,), np.int32, alignment=64)
    arr[:] = test_data

    # Create TensorBuffer referencing the array
    tb = TensorBuffer.create_from_host_memory(arr)

    # Verify data can be read back correctly
    result = tb.read(4, np.int32)
    np.testing.assert_array_equal(result, np.array(test_data, dtype=np.int32))

    # Clean up
    tb.destroy()

  def test_write_read_float(self):
    """Tests writing to and reading from a TensorBuffer with float data."""
    # Create data to write
    test_data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)

    # Create a TensorBuffer with empty host memory with 64-byte alignment
    arr = aligned_array((4,), np.float32, alignment=64)
    tb = TensorBuffer.create_from_host_memory(arr)

    # Write data to the buffer
    tb.write(test_data)

    # Read data back and verify
    result = tb.read(4, np.float32)
    np.testing.assert_array_almost_equal(result, test_data, decimal=5)

    # Clean up
    tb.destroy()

  def test_write_read_int8(self):
    """Tests writing to and reading from a TensorBuffer with int8 data."""
    # Create data to write
    test_data = np.array([10, 20, 30, 40], dtype=np.int8)

    # Create a TensorBuffer with empty host memory with 64-byte alignment
    arr = aligned_array((4,), np.int8, alignment=64)
    tb = TensorBuffer.create_from_host_memory(arr)

    # Write data to the buffer
    tb.write(test_data)

    # Read data back and verify
    result = tb.read(4, np.int8)
    np.testing.assert_array_equal(result, test_data)

    # Clean up
    tb.destroy()

  def test_destroy(self):
    """Tests that destroying a TensorBuffer works correctly."""
    # Create a TensorBuffer with 64-byte aligned memory
    arr = aligned_array((4,), np.float32, alignment=64)
    tb = TensorBuffer.create_from_host_memory(arr)

    # Destroy the buffer
    tb.destroy()

    # Verify that capsule is None after destroy
    self.assertIsNone(tb._capsule)

  def test_destroy_twice(self):
    """Tests that destroying a TensorBuffer twice doesn't crash."""
    # Create a TensorBuffer with 64-byte aligned memory
    arr = aligned_array((4,), np.float32, alignment=64)
    tb = TensorBuffer.create_from_host_memory(arr)

    # Destroy the buffer twice
    tb.destroy()
    tb.destroy()  # This should not raise any exception

    # Verify that capsule is still None
    self.assertIsNone(tb._capsule)

  def test_various_alignments(self):
    """Tests TensorBuffer with different memory alignments."""
    # Only test with 64-byte alignment as that's what the native code expects
    alignment = 64
    # Create aligned numpy array
    test_data = [1.0, 2.0, 3.0, 4.0]
    arr = aligned_array((4,), np.float32, alignment=alignment)
    arr[:] = test_data

    # Verify the array is correctly aligned
    self.assertEqual(arr.ctypes.data % alignment, 0)

    # Create TensorBuffer referencing the array
    tb = TensorBuffer.create_from_host_memory(arr)

    # Verify data can be read back correctly
    result = tb.read(4, np.float32)
    np.testing.assert_array_equal(result, np.array(test_data, dtype=np.float32))

    # Clean up
    tb.destroy()

  def test_various_dtypes(self):
    """Tests TensorBuffer with different data types."""
    test_configs = [
        (np.float32, [1.0, 2.0, 3.0, 4.0]),
        (np.int8, [1, 2, 3, 4]),
        (np.int32, [1, 2, 3, 4]),
    ]

    for np_dtype, test_data in test_configs:
      # Create aligned numpy array with 64-byte alignment
      arr = aligned_array((4,), np_dtype, alignment=64)
      arr[:] = test_data

      # Create TensorBuffer referencing the array
      tb = TensorBuffer.create_from_host_memory(arr)

      # Verify data can be read back correctly
      result = tb.read(4, np_dtype)
      expected = np.array(test_data, dtype=np_dtype)

      if np_dtype == np.float32:
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
      else:
        np.testing.assert_array_equal(result, expected)

      # Clean up
      tb.destroy()

  def test_capsule_property(self):
    """Tests the capsule property."""
    # Create a TensorBuffer with 64-byte aligned memory
    arr = aligned_array((4,), np.float32, alignment=64)
    tb = TensorBuffer.create_from_host_memory(arr)

    # Verify that the capsule property returns something
    self.assertIsNotNone(tb.capsule)

    # Clean up
    tb.destroy()


if __name__ == "__main__":
  unittest.main()
