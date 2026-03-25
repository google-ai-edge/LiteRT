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

  @staticmethod
  def _torch_module():
    try:
      import torch  # pylint: disable=g-import-not-at-top
    except ImportError:
      return None
    return torch

  @classmethod
  def _normalize_dtype(cls, dtype):
    """Normalizes a NumPy or PyTorch dtype.

    Args:
      dtype: A NumPy dtype, NumPy scalar type, or supported PyTorch dtype.

    Returns:
      Tuple of `(dtype_str, numpy_dtype, torch_dtype_or_none)`.

    Raises:
      ValueError: If the dtype is not supported.
    """
    try:
      np_dtype = np.dtype(dtype)
    except TypeError:
      np_dtype = None

    if np_dtype == np.dtype(np.float32):
      return "float32", np.float32, None
    if np_dtype == np.dtype(np.float16):
      return "float16", np.float16, None
    if np_dtype == np.dtype(np.int32):
      return "int32", np.int32, None
    if np_dtype == np.dtype(np.int8):
      return "int8", np.int8, None

    torch = cls._torch_module()
    if torch is not None:
      torch_to_numpy = {
          torch.float32: np.float32,
          torch.float16: np.float16,
          torch.int32: np.int32,
          torch.int8: np.int8,
      }
      if dtype in torch_to_numpy:
        np_dtype = torch_to_numpy[dtype]
        dtype_str, np_dtype, _ = cls._normalize_dtype(np_dtype)
        return dtype_str, np_dtype, dtype

    raise ValueError(f"Unsupported dtype: {dtype}")

  @classmethod
  def _dtype_to_str(cls, dtype):
    """Converts a supported dtype to the low-level string representation."""
    dtype_str, _, _ = cls._normalize_dtype(dtype)
    return dtype_str

  @classmethod
  def _normalize_host_array_like(cls, data_array, *, zero_copy: bool):
    """Normalizes NumPy/PyTorch array-likes to a contiguous host NumPy array.

    Args:
      data_array: NumPy array or supported PyTorch tensor.
      zero_copy: If True, reject inputs that cannot be exposed zero-copy.

    Returns:
      A contiguous NumPy array backed by host memory.
    """
    if isinstance(data_array, np.ndarray):
      cls._dtype_to_str(data_array.dtype)
      if zero_copy:
        if not data_array.flags.c_contiguous:
          raise ValueError(
              "data_array must be C-contiguous for zero-copy host-memory "
              "binding"
          )
        return data_array
      return np.ascontiguousarray(data_array)

    torch = cls._torch_module()
    if torch is not None and isinstance(data_array, torch.Tensor):
      cls._dtype_to_str(data_array.dtype)
      tensor = data_array.detach()
      if zero_copy:
        if tensor.device.type != "cpu":
          raise ValueError(
              "PyTorch tensor must be on CPU for zero-copy host-memory binding"
          )
        if not tensor.is_contiguous():
          raise ValueError(
              "PyTorch tensor must be contiguous for zero-copy host-memory "
              "binding"
          )
        return tensor.numpy()
      if tensor.device.type != "cpu":
        tensor = tensor.cpu()
      tensor = tensor.contiguous()
      return tensor.numpy()

    raise ValueError(
        "data_array must be a NumPy array or a supported PyTorch tensor"
    )

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
      data_array: A C-contiguous NumPy array or a CPU-contiguous PyTorch tensor
        (e.g., np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)). The dtype
        of the array/tensor is used. Inputs that cannot be exposed zero-copy
        are rejected.

    Returns:
      A new TensorBuffer instance.

    Raises:
      ValueError: If the input is not a supported zero-copy host array/tensor or
        has an unsupported dtype.
    """
    array = cls._normalize_host_array_like(data_array, zero_copy=True)
    dtype_str = cls._dtype_to_str(array.dtype)
    num_elements = array.size

    cap = _tb.CreateTensorBufferFromHostMemory(
        array, dtype_str, num_elements
    )
    return cls(cap)

  def write(self, data_array: np.ndarray):
    """Writes data to this tensor buffer.

    Args:
      data_array: A NumPy array or supported PyTorch tensor. Non-contiguous
        arrays/tensors and non-CPU PyTorch tensors are normalized onto a
        contiguous CPU buffer before copying into the TensorBuffer.

    Example:
      # Using NumPy array
      test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
      tensor_buffer.write(test_input)

    Raises:
      ValueError: If the input type or dtype is unsupported.
    """
    array = self._normalize_host_array_like(data_array, zero_copy=False)
    dtype_str = self._dtype_to_str(array.dtype)
    flat = np.ascontiguousarray(array.reshape(-1))
    _tb.WriteTensorBuffer(self._capsule, flat, dtype_str)

  def read(self, num_elements: int, output_dtype):
    """Reads data from this tensor buffer.

    Args:
      num_elements: Number of elements to read.
      output_dtype: NumPy dtype or supported PyTorch dtype for the output
        (e.g., np.float32, np.int8, torch.float16).

    Returns:
      A NumPy array containing the tensor data, or a CPU PyTorch tensor when a
      PyTorch dtype is requested.

    Example:
      # Get output as NumPy array
      output_array = tensor_buffer.read(4, np.float32).reshape((1, 4))

    Raises:
      ValueError: If output_dtype is not supported.
    """
    dtype_str, np_dtype, torch_dtype = self._normalize_dtype(output_dtype)
    result = np.empty(num_elements, dtype=np_dtype)
    _tb.ReadTensorToBuffer(self._capsule, result, dtype_str)
    if torch_dtype is not None:
      torch = self._torch_module()
      return torch.from_numpy(result)
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
