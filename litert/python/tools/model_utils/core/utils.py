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
"""ModelUtils core utilities."""

from typing import Any
import ml_dtypes
from litert.python.mlir import ir
import numpy as np

__all__ = [
    'dtype_to_ir_type',
    'ir_type_to_dtype',
]


def tree_flatten(value: Any) -> list[Any]:
  """Flattens nested list or tuple into a list."""
  result = []
  if isinstance(value, (list, tuple)):
    for item in value:
      result.extend(tree_flatten(item))
  else:
    result.append(value)
  return result


def dtype_to_ir_type(dtype: np.dtype | np.generic) -> ir.Type:
  """Converts a NumPy dtype into a corresponding MLIR IR Type."""
  # Standardize the input to a np.dtype object
  assert isinstance(dtype, (np.dtype, np.generic)), type(dtype)
  dtype = np.dtype(dtype)

  ir_type: ir.Type

  # Integer Types (Signed/Signless)
  if dtype == np.dtype(np.bool_):
    ir_type = ir.IntegerType.get_signless(1)
  elif dtype == np.dtype(ml_dtypes.int2):
    ir_type = ir.IntegerType.get_signless(2)
  elif dtype == np.dtype(ml_dtypes.int4):
    ir_type = ir.IntegerType.get_signless(4)
  elif dtype == np.dtype(np.int8):
    ir_type = ir.IntegerType.get_signless(8)
  elif dtype == np.dtype(np.int16):
    ir_type = ir.IntegerType.get_signless(16)
  elif dtype == np.dtype(np.int32):
    ir_type = ir.IntegerType.get_signless(32)
  elif dtype == np.dtype(np.int64):
    ir_type = ir.IntegerType.get_signless(64)

  # Integer Types (Unsigned)
  elif dtype == np.dtype(ml_dtypes.uint2):
    ir_type = ir.IntegerType.get_unsigned(2)
  elif dtype == np.dtype(ml_dtypes.uint4):
    ir_type = ir.IntegerType.get_unsigned(4)
  elif dtype == np.dtype(np.uint8):
    ir_type = ir.IntegerType.get_unsigned(8)
  elif dtype == np.dtype(np.uint16):
    ir_type = ir.IntegerType.get_unsigned(16)
  elif dtype == np.dtype(np.uint32):
    ir_type = ir.IntegerType.get_unsigned(32)
  elif dtype == np.dtype(np.uint64):
    ir_type = ir.IntegerType.get_unsigned(64)

  # Floating Point Types
  elif dtype == np.dtype(ml_dtypes.bfloat16):
    ir_type = ir.BF16Type.get()
  elif dtype == np.dtype(np.float16):
    ir_type = ir.F16Type.get()
  elif dtype == np.dtype(np.float32):
    ir_type = ir.F32Type.get()
  elif dtype == np.dtype(np.float64):
    ir_type = ir.F64Type.get()

  else:
    raise TypeError(f'No handler to convert to MLIR IR Type for dtype: {dtype}')

  return ir_type


def ir_type_to_dtype(ir_type: ir.Type) -> np.dtype:
  """Converts an MLIR IR Type to the corresponding NumPy dtype."""

  # Handle direct types (BF16, F16, F32, F64)
  if isinstance(ir_type, ir.BF16Type):
    return np.dtype(ml_dtypes.bfloat16)
  elif isinstance(ir_type, ir.F16Type):
    return np.dtype(np.float16)
  elif isinstance(ir_type, ir.F32Type):
    return np.dtype(np.float32)
  elif isinstance(ir_type, ir.F64Type):
    return np.dtype(np.float64)

  # Handle Integer Types (using nested if for width/sign)
  elif isinstance(ir_type, ir.IntegerType):

    # Check if the type is signless (or signed)
    if not ir_type.is_unsigned:
      # Nested check for width
      if ir_type.width == 1:
        return np.dtype(np.bool_)
      elif ir_type.width == 2:
        return np.dtype(ml_dtypes.int2)
      elif ir_type.width == 4:
        return np.dtype(ml_dtypes.int4)
      elif ir_type.width == 8:
        return np.dtype(np.int8)
      elif ir_type.width == 16:
        return np.dtype(np.int16)
      elif ir_type.width == 32:
        return np.dtype(np.int32)
      elif ir_type.width == 64:
        return np.dtype(np.int64)
      else:
        raise TypeError(f'Unsupported signless integer width: {ir_type.width}')

    # Check if the type is unsigned
    else:
      # Nested check for width
      if ir_type.width == 2:
        return np.dtype(ml_dtypes.uint2)
      elif ir_type.width == 4:
        return np.dtype(ml_dtypes.uint4)
      elif ir_type.width == 8:
        return np.dtype(np.uint8)
      elif ir_type.width == 16:
        return np.dtype(np.uint16)
      elif ir_type.width == 32:
        return np.dtype(np.uint32)
      elif ir_type.width == 64:
        return np.dtype(np.uint64)
      else:
        raise TypeError(f'Unsupported unsigned integer width: {ir_type.width}')

  else:
    raise TypeError(
        f'No handler to convert to NumPy dtype for IR Type: {ir_type}'
    )


def numpy_array_to_mlir_compatible(arr: np.ndarray) -> np.ndarray:
  """Converts a tensor to a numpy array.

  The numpy array is compatible with MLIR contiguity and endianness.

  Args:
    arr: The numpy array to convert.

  Returns:
    The numpy array compatible with MLIR.
  """
  if arr.dtype == bool or arr.dtype == np.bool_:
    # packbits returns uint8; bitorder='little' is crucial for MLIR
    return np.packbits(arr, axis=None, bitorder='little')

  target_dtype = {
      # Floating point
      np.float16: '<f2',
      np.float32: '<f4',
      np.float64: '<f8',
      # Signed Integers
      np.int8: '<i1',
      np.int16: '<i2',
      np.int32: '<i4',
      np.int64: '<i8',
      # Unsigned Integers
      np.uint8: '<u1',
      np.uint16: '<u2',
      np.uint32: '<u4',
      np.uint64: '<u8',
  }.get(arr.dtype.type)

  if target_dtype is None:
    raise TypeError(f'Unsupported dtype for MLIR conversion: {arr.dtype}')

  # Ensure C-contiguity and the specific bit-width/endianness
  return np.ascontiguousarray(arr, dtype=target_dtype)
