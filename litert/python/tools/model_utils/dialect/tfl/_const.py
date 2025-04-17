from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import arith

ConstOp = arith.ConstantOp
ConstantOp = arith.ConstantOp
PseudoConstOp = arith.ConstantOp


# Overload arith.constant to be tfl.pseudo_const
core.register_mlir_transform("tfl.pseudo_const")(arith.ConstantOp)


def pseudo_const(*args, **kwargs):
  return arith.constant(*args, **kwargs)


def const(*args, **kwargs):
  return arith.constant(*args, **kwargs)


def constant(*args, **kwargs):
  return arith.constant(*args, **kwargs)
