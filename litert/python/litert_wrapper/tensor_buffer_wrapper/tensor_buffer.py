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

"""Python wrapper for LiteRT tensor buffer."""

import os
import numpy as np

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("ai_edge_litert", "tensor_buffer")
):
  # This file is part of litert package.
  from litert.python.litert_wrapper.tensor_buffer_wrapper import _pywrap_litert_tensor_buffer_wrapper as _tb
else:
  # This file is part of ai_edge_litert package.
  from ai_edge_litert import _pywrap_litert_tensor_buffer_wrapper as _tb
# pylint: enable=g-import-not-at-top


class TensorBuffer:
  """Python wrapper for LiteRtTensorBuffer.

  This class provides a high-level interface to the underlying C++
  LiteRtTensorBuffer
  implementation, allowing for creation, reading, writing, and management of
  tensor
  buffers in Python.
  """

  def __init__(self, capsule):
    """Initializes a TensorBuffer with the provided PyCapsule.

    Args:
      capsule: A PyCapsule containing a pointer to a LiteRtTensorBuffer.
    """
    self._capsule = capsule

  @staticmethod
  def _dtype_to_str(np_dtype):
    """Converts a numpy dtype to a string representation.

    Args:
      np_dtype: A numpy dtype (e.g., np.float32, np.int8).

    Returns:
      String representation of the data type (e.g., "float32", "int8").

    Raises:
      ValueError: If the dtype is not supported.
    """
    if np_dtype == np.float32:
      return "float32"
    elif np_dtype == np.int32:
      return "int32"
    elif np_dtype == np.int8:
      return "int8"
    else:
      raise ValueError(f"Unsupported dtype: {np_dtype}")

  @classmethod
  def create_from_host_memory(cls, data_array):
    """Creates a new TensorBuffer referencing existing host memory.

    Args:
      data_array: A NumPy array (e.g., np.array([[1.0, 2.0, 3.0, 4.0]],
        dtype=np.float32)). The dtype of the array is used.

    Returns:
      A new TensorBuffer instance.

    Raises:
      ValueError: If the input is not a NumPy array or has an unsupported dtype.
    """
    if not isinstance(data_array, np.ndarray):
      raise ValueError("data_array must be a NumPy array")

    dtype_str = cls._dtype_to_str(data_array.dtype)
    num_elements = data_array.size

    cap = _tb.CreateTensorBufferFromHostMemory(
        data_array, dtype_str, num_elements
    )
    return cls(cap)

  def write(self, data_array):
    """Writes data to this tensor buffer.

    Args:
      data_array: A NumPy array (e.g., np.array([[1.0, 2.0, 3.0, 4.0]],
        dtype=np.float32)). The dtype of the array is used.

    Example:
      # Using NumPy array
      test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
      tensor_buffer.write(test_input)

    Raises:
      ValueError: If the input is not a NumPy array or has an unsupported dtype.
    """
    if not isinstance(data_array, np.ndarray):
      raise ValueError("data_array must be a NumPy array")

    dtype_str = self._dtype_to_str(data_array.dtype)
    _tb.WriteTensor(self._capsule, data_array.flatten().tolist(), dtype_str)

  def read(self, num_elements: int, output_dtype):
    """Reads data from this tensor buffer.

    Args:
      num_elements: Number of elements to read.
      output_dtype: NumPy dtype for the output (e.g., np.float32, np.int8).

    Returns:
      A NumPy array containing the tensor data.

    Example:
      # Get output as NumPy array
      output_array = tensor_buffer.read(4, np.float32).reshape((1, 4))

    Raises:
      ValueError: If output_dtype is not a NumPy dtype or is not supported.
    """
    if not isinstance(output_dtype, type) or not hasattr(
        np, output_dtype.__name__
    ):
      raise ValueError(f"output_dtype must be a NumPy dtype (e.g., np.float32)")

    dtype_str = self._dtype_to_str(output_dtype)
    data_list = _tb.ReadTensor(self._capsule, num_elements, dtype_str)
    return np.array(data_list, dtype=output_dtype)

  def destroy(self):
    """Explicitly releases resources associated with this tensor buffer.

    After calling this method, the tensor buffer is no longer valid for use.
    """
    _tb.DestroyTensorBuffer(self._capsule)
    self._capsule = None

  @property
  def capsule(self):
    """Returns the underlying PyCapsule for direct C++ interoperability.

    When the capsule is used with compiled_model methods, the ownership remains
    with this TensorBuffer instance. This avoids double-free when both objects
    try to destroy the same underlying tensor buffer.

    Returns:
      The PyCapsule containing the pointer to the LiteRtTensorBuffer, with
      ownership remaining with this TensorBuffer instance.
    """
    # Once destroyed, shouldn't try to access the capsule again.
    if self._capsule is None:
      raise ValueError("TensorBuffer has been destroyed")
    return self._capsule
