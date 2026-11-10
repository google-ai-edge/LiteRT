# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Duplicate buffer transformation."""

from litert_quantizer import qtyping
from litert_quantizer.transformations import transformation_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


def duplicate_buffer(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Duplicates the buffer of the tensor."""
  tensor_id = transformation_input.tensor_id
  tensor = transformation_input.subgraph.tensors[tensor_id]
  buffer_data = transformation_input.model.buffers[tensor.buffer].data
  if buffer_data is None:
    tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
    raise ValueError(
        'Duplicate Buffer transformation supports only constant tensors.'
        f' Tensor {tensor_name} is not constant.'
    )

  duplicated_buffer_id = transformation_utils.get_constant_buffer(
      data=buffer_data,
      model=transformation_input.model,
      force_duplicate_buffer=True,
  )
  tensor.buffer = duplicated_buffer_id

  return qtyping.TransformationInfo(
      op_id=0, num_ops_added=0, output_tensor_id=tensor_id
  )
