import jax
import numpy as np


def dtype_to_ir_type(dtype: np.dtype):
  return jax._src.interpreters.mlir.dtype_to_ir_type(dtype)
