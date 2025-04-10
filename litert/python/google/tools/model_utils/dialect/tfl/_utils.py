import jax
import jax.numpy as jnp
import numpy as np
import xdsl.irdl

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir


SSAValue = xdsl.irdl.SSAValue


def tensor_type_to_jax(ty: mlir.RankedTensorType) -> jax.core.ShapedArray:
  if not isinstance(ty, mlir.RankedTensorType):
    raise ValueError("Cannot convert non-RankedTensorType value to JAX")

  elty = ty.element_type
  if "quant" in str(elty)[:100]:
    raise ValueError("Cannot convert quant RankedTensorType to JAX")

  dtype = {
      "i8": jnp.int8,
      "i32": jnp.int32,
      "i64": jnp.int64,
      "f32": jnp.float32,
      "f64": jnp.float64,
  }[elty]
  return jax.core.ShapedArray(shape=ty.shape, dtype=dtype)


def jax_to_tensor_type(x: jax.core.ShapedArray):
  dtype = np.dtype(x.dtype)
  shape = x.shape

  elty = {
      np.dtype(np.int8): "i8",
      np.dtype(np.int32): "i32",
      np.dtype(np.int64): "i64",
      np.dtype(np.float32): "f32",
      np.dtype(np.float64): "f64",
  }[dtype]
  return mlir.RankedTensorType(shape=shape, element_type=elty)


def infer_result_types(aval, *args, **kwargs):
  def to_jax(x):
    if not isinstance(x, SSAValue):
      return x
    return tensor_type_to_jax(x.type)

  args, kwargs = jax.tree.map(to_jax, [args, kwargs])
  results = jax.eval_shape(aval, *args, **kwargs)
  return jax.tree.map(jax_to_tensor_type, results)


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
