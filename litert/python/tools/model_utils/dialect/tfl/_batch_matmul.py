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
"""tfl.batch_matmul operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.batch_matmul")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class BatchMatMulOp(core.MlirOpBase):
  """Batch Matrix Multiply Operator.

  Performs a batched matrix multiplication on the inputs. Follows the
  conventions of TensorFlow BatchMatMulV2, with support for unknown dimensions
  in the batch dimensions and broadcasting.

  Inputs:
    `inputs[0]`: required: input LHS
    `inputs[1]`: required: input RHS
    `adjoint_lhs`: optional: Transpose LHS (default false)
    `adjoint_rhs`: optional: Transpose RHS (default false)
  """

  name = "tfl.batch_matmul"

  x = irdl.operand_def()
  y = irdl.operand_def()
  adj_x = irdl.opt_attr_def(mlir.BoolAttr)
  adj_y = irdl.opt_attr_def(mlir.BoolAttr)
  asymmetric_quantize_inputs = irdl.opt_attr_def(mlir.BoolAttr)
  output = irdl.result_def()

  def __init__(
      self,
      x: SSAValue | core.MlirOpBase,
      y: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      adj_x: bool | mlir.BoolAttr = False,
      adj_y: bool | mlir.BoolAttr = False,
      asymmetric_quantize_inputs: bool | mlir.BoolAttr = False,
      location=None,
  ):
    x = SSAValue.get(x)
    y = SSAValue.get(y)
    adj_x = mlir.BoolAttr(adj_x)
    adj_y = mlir.BoolAttr(adj_y)
    asymmetric_quantize_inputs = mlir.BoolAttr(asymmetric_quantize_inputs)

    result_types = [result_type or self._infer_result_type(x, y, adj_x, adj_y)]

    super().__init__(
        operands=[x, y],
        result_types=result_types,
        attributes={
            "adj_x": adj_x,
            "adj_y": adj_y,
            "asymmetric_quantize_inputs": asymmetric_quantize_inputs,
        },
        location=location,
    )

  def _infer_result_type(
      self,
      x: SSAValue | core.MlirOpBase,
      y: SSAValue | core.MlirOpBase,
      adj_x: bool | mlir.BoolAttr,
      adj_y: bool | mlir.BoolAttr,
  ):
    x_type = _utils.get_tensor_type(x)
    y_type = _utils.get_tensor_type(y)

    x_shape = list(x_type.shape)
    y_shape = list(y_type.shape)

    if len(x_shape) < 2 or len(y_shape) < 2:
      raise ValueError("Inputs must have rank at least 2.")

    if x_type.element_type != y_type.element_type:
      raise ValueError("Inputs must have the same element type.")

    adj_x_val = _utils.to_bool(adj_x)
    adj_y_val = _utils.to_bool(adj_y)

    x_rows = x_shape[-2] if not adj_x_val else x_shape[-1]
    x_cols = x_shape[-1] if not adj_x_val else x_shape[-2]
    y_rows = y_shape[-2] if not adj_y_val else y_shape[-1]
    y_cols = y_shape[-1] if not adj_y_val else y_shape[-2]

    if x_cols != y_rows:
      raise ValueError(
          "Incompatible dimensions for matrix multiplication:"
          f" x_cols={x_cols} and y_rows={y_rows}."
      )

    batch_dims_x = x_shape[:-2]
    batch_dims_y = y_shape[:-2]

    batch_dims = np.broadcast_shapes(batch_dims_x, batch_dims_y)
    output_shape = [*batch_dims, x_rows, y_cols]
    return mlir.RankedTensorType(output_shape, x_type.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        adj_x=mlir.BoolAttr.op_attribute_accessor("adj_x"),
        adj_y=mlir.BoolAttr.op_attribute_accessor("adj_y"),
        asymmetric_quantize_inputs=mlir.BoolAttr.op_attribute_accessor(
            "asymmetric_quantize_inputs"
        ),
    )


@_utils.op_builder_wraps(BatchMatMulOp)
def batch_matmul(*args, **kwargs):
  return BatchMatMulOp(*args, **kwargs).output
