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
"""tfl.gather_nd operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

# pylint: disable=redefined-builtin

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.gather_nd")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class GatherNdOp(core.MlirOpBase):
  """_Gathernd operator.

  Gather slices from params into a Tensor with shape specified by indices.
  """

  name = "tfl.gather_nd"

  params = irdl.operand_def()
  indices = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      params: SSAValue | core.MlirOpBase,
      indices: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    params_val = SSAValue.get(params)
    indices_val = SSAValue.get(indices)

    result_types = [
        result_type or self._infer_result_type(params_val, indices_val)
    ]

    super().__init__(
        operands=[params_val, indices_val],
        result_types=result_types,
        attributes={},  # No attributes
        location=location,
    )

  def _infer_result_type(
      self,
      params_val: SSAValue | core.MlirOpBase,
      indices_val: SSAValue | core.MlirOpBase,
  ):
    params_type = _utils.get_tensor_type(params_val)
    indices_type = _utils.get_tensor_type(indices_val)
    params_shape = list(params_type.shape)
    indices_shape = list(indices_type.shape)
    params_rank = len(params_shape)
    indices_rank = len(indices_shape)

    index_depth = indices_shape[-1] if indices_rank > 0 else 0

    if index_depth > params_rank:
      raise ValueError(
          f"Last dimension of indices ({index_depth}) must not exceed params"
          f" rank ({params_rank})"
      )

    # Calculate output shape based on TF/TFLite GatherNd semantics
    # Output shape: indices.shape[:-1] + params.shape[indices.shape[-1]:]
    output_shape = indices_shape[:-1] + params_shape[index_depth:]

    # Result element type is the same as the params element type.
    return mlir.RankedTensorType(output_shape, params_type.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(GatherNdOp)
def gather_nd(*args, **kwargs):
  return GatherNdOp(*args, **kwargs).output
