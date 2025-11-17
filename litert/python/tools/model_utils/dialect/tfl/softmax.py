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
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.softmax")
@core.overload_cls_attrs
@irdl_op_definition
class SoftmaxOp(core.MlirOpBase):
  """Softmax operator

  Computes element-wise softmax activations with the following formula

  exp(input * beta) / tf.reduce_sum(exp(input * beta), dim)
  """

  name = "tfl.softmax"

  input = operand_def()
  beta = opt_attr_def(mlir.FloatAttr)
  output = result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      beta: float | mlir.FloatAttr = 1.0,
      location=None,
  ):
    input = SSAValue.get(input)
    beta = mlir.FloatAttr(beta)

    result_types = [result_type or self._infer_result_type(input)]

    super().__init__(
        operands=[input],
        result_types=result_types,
        attributes={"beta": beta},
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(input)

    return mlir.RankedTensorType(input_type.shape, input_type.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        beta=mlir.FloatAttr.op_attribute_accessor("beta"),
    )


def softmax(
    input: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    beta: float | mlir.FloatAttr = 1.0,
    location=None,
):
  """Softmax operator

  Computes element-wise softmax activations with the following formula

  exp(input * beta) / tf.reduce_sum(exp(input * beta), dim)
  """
  return SoftmaxOp(
      input, result_type=result_type, beta=beta, location=location
  ).output
