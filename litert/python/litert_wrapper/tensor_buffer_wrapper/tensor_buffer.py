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
  from litert.python.litert_wrapper.tensor_buffer_wrapper import (
      _pywrap_litert_tensor_buffer_wrapper as _tb,
  )
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

  _DTYPE_STRINGS = {
      np.float32: "float32",
      np.dtype(np.float32): "float32",
      np.float16: "float16",
      np.dtype(np.float16): "float16",
      np.int32: "int32",
      np.dtype(np.int32): "int32",
      np.int8: "int8",
      np.dtype(np.int8): "int8",
      np.uint8: "uint8",
      np.dtype(np.uint8): "uint8",
  }

  def __init__(self, capsule, environment=None):
    """Initializes a TensorBuffer with the provided PyCapsule.

    Args:
      capsule: A PyCapsule containing a pointer to a LiteRtTensorBuffer.
      environment: Optional Environment object retained to keep the shared
        LiteRT runtime context alive for buffers created by a CompiledModel.
    """
    self._capsule = capsule
    self._environment = environment

  @classmethod
  def _normalize_dtype(cls, dtype):
    """Normalizes a supported NumPy dtype.

    Args:
      dtype: A NumPy dtype or NumPy scalar type.

    Returns:
      Tuple of `(dtype_str, numpy_dtype)`.

    Raises:
      ValueError: If the dtype is not supported.
    """
    dtype_str = cls._DTYPE_STRINGS.get(dtype)
    if dtype_str is None:
      try:
        np_dtype = np.dtype(dtype)
      except TypeError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc
      dtype_str = cls._DTYPE_STRINGS.get(np_dtype)
      if dtype_str is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    else:
      np_dtype = np.dtype(dtype)
    return dtype_str, np_dtype

  @classmethod
  def _normalize_numpy_array(
      cls, data_array: np.ndarray, *, zero_copy: bool
  ) -> tuple[np.ndarray, str]:
    """Normalizes a NumPy array to supported contiguous host memory.

    Args:
      data_array: NumPy array.
      zero_copy: If True, reject inputs that cannot be exposed zero-copy.

    Returns:
      Tuple of `(array, dtype_str)`.
    """
    if not isinstance(data_array, np.ndarray):
      raise ValueError("data_array must be a NumPy array")

    dtype_str, _ = cls._normalize_dtype(data_array.dtype)
    if zero_copy and not data_array.flags.c_contiguous:
      raise ValueError(
          "data_array must be C-contiguous for zero-copy host-memory binding"
      )
    if zero_copy:
      return data_array, dtype_str
    return np.ascontiguousarray(data_array), dtype_str

  @classmethod
  def create_from_host_memory(cls, data_array: np.ndarray):
    """Creates a new TensorBuffer referencing existing host memory.

    Args:
      data_array: A C-contiguous NumPy array (e.g., np.array([[1.0, 2.0, 3.0,
        4.0]], dtype=np.float32)). The dtype of the array is used. Inputs that
        cannot be exposed zero-copy are rejected.

    Returns:
      A new TensorBuffer instance.

    Raises:
      ValueError: If the input is not a supported zero-copy NumPy array or has
        an unsupported dtype.
    """
    array, dtype_str = cls._normalize_numpy_array(data_array, zero_copy=True)
    num_elements = array.size

    cap = _tb.CreateTensorBufferFromHostMemory(
        array, dtype_str, num_elements
    )
    return cls(cap)

  def write(self, data_array: np.ndarray):
    """Writes data to this tensor buffer.

    Args:
      data_array: A NumPy array. Non-contiguous arrays are normalized onto a
        contiguous buffer before copying into the TensorBuffer.

    Example:
      # Using NumPy array
      test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
      tensor_buffer.write(test_input)

    Raises:
      ValueError: If the input is not a NumPy array or has an unsupported
        dtype.
    """
    array, dtype_str = self._normalize_numpy_array(data_array, zero_copy=False)
    _tb.WriteTensorBuffer(self._capsule, array.reshape(-1), dtype_str)

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
      ValueError: If output_dtype is not supported.
    """
    dtype_str, np_dtype = self._normalize_dtype(output_dtype)
    result = np.empty(num_elements, dtype=np_dtype)
    _tb.ReadTensorToBuffer(self._capsule, result, dtype_str)
    return result

  def get_tensor_details(self):
    """Returns tensor metadata for this buffer.

    Returns:
      A dictionary containing the buffer tensor `dtype` and `shape`.
    """
    return _tb.GetTensorDetails(self._capsule)

  def destroy(self):
    """Explicitly releases resources associated with this tensor buffer.

    After calling this method, the tensor buffer is no longer valid for use.
    """
    _tb.DestroyTensorBuffer(self._capsule)
    self._capsule = None
    self._environment = None

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
