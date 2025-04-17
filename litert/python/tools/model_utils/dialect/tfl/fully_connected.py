from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.fully_connected")
@core.overload_cls_attrs
@irdl_op_definition
class FullyConnectedOp(core.MlirOpBase):
  name = "tfl.fully_connected"

  input = operand_def()
  filter = operand_def()
  bias = opt_operand_def()
  output = result_def()

  fused_activation_function = opt_attr_def(mlir.StringAttr)
  weights_format = opt_attr_def(mlir.StringAttr)
  keep_num_dims = opt_attr_def(mlir.BoolAttr)
  asymmetric_quantize_inputs = opt_attr_def(mlir.BoolAttr)

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      filter: SSAValue | core.MlirOpBase,
      bias: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      fused_activation_function: str | mlir.StringAttr = "NONE",
      weights_format: str | mlir.StringAttr = "DEFAULT",
      keep_num_dims: bool | mlir.BoolAttr = False,
      asymmetric_quantize_inputs: bool | mlir.BoolAttr = False,
      location=None,
  ):
    input = SSAValue.get(input)
    filter = SSAValue.get(filter)
    bias = SSAValue.get(bias)

    fused_activation_function = _utils.to_str(fused_activation_function)
    weights_format = _utils.to_str(weights_format)
    keep_num_dims = mlir.BoolAttr(keep_num_dims)
    asymmetric_quantize_inputs = mlir.BoolAttr(asymmetric_quantize_inputs)

    result_types = (
        [result_type]
        if result_type is not None
        else [self._infer_result_type(input, filter, bias)]
    )

    operands = [input, filter, bias]
    super().__init__(
        operands=operands,
        result_types=result_types,
        attributes={
            "fused_activation_function": mlir.StringAttr(
                fused_activation_function
            ),
            "weights_format": mlir.StringAttr(weights_format),
            "keep_num_dims": keep_num_dims,
            "asymmetric_quantize_inputs": asymmetric_quantize_inputs,
        },
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      filter: SSAValue | core.MlirOpBase,
      bias: SSAValue | core.MlirOpBase | None = None,
  ):
    raise NotImplementedError(
        "Cannot infer result type for FullyConnectedOp. Please specify"
        " result_type."
    )

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        fused_activation_function=mlir.StringAttr.op_attribute_accessor(
            "fused_activation_function"
        ),
        weights_format=mlir.StringAttr.op_attribute_accessor("weights_format"),
        keep_num_dims=mlir.BoolAttr.op_attribute_accessor("keep_num_dims"),
        asymmetric_quantize_inputs=mlir.BoolAttr.op_attribute_accessor(
            "asymmetric_quantize_inputs"
        ),
    )


def fully_connected(*args, **kwargs):
  return FullyConnectedOp(*args, **kwargs).output
