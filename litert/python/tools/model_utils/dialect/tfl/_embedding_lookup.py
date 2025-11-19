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
"""tfl.embedding_lookup operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

# pylint: disable=redefined-builtin

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.embedding_lookup")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class EmbeddingLookupOp(core.MlirOpBase):
  """Embedding lookup operator.

  Looks up ids in a list of embedding tensors.
  """

  name = "tfl.embedding_lookup"

  lookup = irdl.operand_def()  # IDs
  value = irdl.operand_def()  # Embedding table
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      lookup: SSAValue | core.MlirOpBase,
      value: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    lookup_val = SSAValue.get(lookup)
    value_val = SSAValue.get(value)
    result_types = [
        result_type or self._infer_result_type(lookup_val, value_val)
    ]
    super().__init__(
        operands=[lookup_val, value_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      lookup_val: SSAValue | core.MlirOpBase,
      value_val: SSAValue | core.MlirOpBase,
  ):
    lookup_type = _utils.get_tensor_type(lookup_val)
    value_type = _utils.get_tensor_type(value_val)

    value_rank = len(value_type.shape)
    if value_rank < 1:  # Spec implies rank >= 1, usually >= 2 for embeddings
      raise ValueError(
          f"Value tensor must have rank at least 1, got {value_rank}"
      )

    # Output shape is lookup_shape + value_shape[1:]
    output_shape = list(lookup_type.shape) + list(value_type.shape[1:])
    value_elem_type = value_type.element_type
    output_elem_type = value_elem_type

    return mlir.RankedTensorType(output_shape, output_elem_type)

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(EmbeddingLookupOp)
def embedding_lookup(*args, **kwargs):
  return EmbeddingLookupOp(*args, **kwargs).output
