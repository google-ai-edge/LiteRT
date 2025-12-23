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
"""TFLite external_const op."""


from litert.python.mlir import ir
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

irdl_op_definition = irdl.irdl_op_definition
attr_def = irdl.attr_def
result_def = irdl.result_def


@core.register_mlir_transform('tfl.external_const')
@irdl_op_definition
class ExternalConstOp(core.MlirOpBase):
  """TFLite external_const op."""

  name = 'tfl.external_const'

  output = result_def()

  def __init__(
      self,
      group_name: str,
      offset: int,
      length: int,
      packing: str,
      result_type: mlir.RankedTensorType,
      location: ir.Location | None = None,
  ):

    super().__init__(
        result_types=[result_type],
        location=location,
    )
    self.set_external_buffer(group_name, offset, length, packing)

  def set_external_buffer(
      self,
      group_name: str,
      offset: int,
      length: int,
      packing: str,
  ):
    """Sets the external buffer attribute."""
    group_name = str(group_name).replace('"', '\\"')
    packing = str(packing).replace('"', '\\"')
    offset = int(offset)
    length = int(length)

    ser = f"""#tfl.external_buffer<group_name = "{group_name}", offset = {offset}, length = {length}, packing = "{packing}">"""
    attr = mlir.MlirAttribute(ir.Attribute.parse(ser))
    self.attributes['external_buffer'] = attr


def external_const(*args, **kwargs):
  """Creates an external_const op."""
  return ExternalConstOp(*args, **kwargs).output
