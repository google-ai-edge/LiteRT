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
"""Utility functions for creating TFL dialect op classes and builders."""

import functools

import numpy as np
import xdsl.irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir


SSAValue = xdsl.irdl.SSAValue


def op_builder_wraps(op_cls):
  """Creates a decorator to wrap op builder function with metadata from OpDef class."""

  def wraps(builder):
    assigned = {"__module__", "__doc__", "__annotations__"}
    if builder.__doc__:
      assigned.remove("__doc__")

    builder = functools.wraps(op_cls, assigned=assigned)(builder)
    return builder

  return wraps


def to_int(x: int | mlir.IntegerAttr | np.generic) -> int:
  if isinstance(x, mlir.IntegerAttr):
    return int(x.data)
  elif isinstance(x, np.generic):
    return int(x.item())
  else:
    return int(x)


def to_float(x: float | mlir.FloatAttr | np.generic) -> float:
  if isinstance(x, mlir.FloatAttr):
    return float(x.data)
  elif isinstance(x, np.generic):
    return float(x.item())
  else:
    return float(x)


def to_str(x: str | mlir.StringAttr) -> str:
  if isinstance(x, mlir.StringAttr):
    return str(x.data)
  else:
    return str(x)


def to_bool(x: bool | mlir.BoolAttr) -> bool:
  if isinstance(x, mlir.BoolAttr):
    return bool(x.data)
  else:
    return bool(x)


def get_tensor_type(x: SSAValue | core.MlirOpBase) -> mlir.RankedTensorType:
  x = SSAValue.get(x)
  ty = x.type
  if not isinstance(ty, mlir.RankedTensorType):
    raise ValueError("Cannot get shape of non-RankedTensorType value.")
  return ty


def np_64bit_to_32bit(data: np.array) -> np.array:
  """Converts a numpy array of 64-bit dtype to 32-bit dtype."""
  data = np.array(data)
  if data.dtype == np.int64:
    data = data.astype(np.int32)
  elif data.dtype == np.float64:
    data = data.astype(np.float32)
  return data
