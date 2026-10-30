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

"""Apply dequantization transformations to the given op/tensor.

Inserts dequantize node after the given tensor to enable float execution of
the tensor consumer
"""

from litert_quantizer import qtyping
from litert_quantizer.transformations import quantize_tensor
from litert_quantizer.transformations import transformation_utils


def insert_dequant(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Insert dequant op after the given tensor in the subgraph.

  Args:
    transformation_input: input structure that contains all information needed
      for the transformation.

  Returns:
    TransformationInfo:
      op_id: the index where the dequant op is added
      num_ops_added: the total number of ops inserted by this operation, which
        is 1
  """
  dequant_op_code_idx = transformation_utils.add_op_code(
      qtyping.BuiltinOperator.DEQUANTIZE,
      transformation_input.model.operatorCodes,
  )
  # create output tensor for the dequant op
  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  new_tensor_id = transformation_utils.add_new_activation_tensor(
      tensor.name + b'_dequant',
      tensor.shape,
      qtyping.TensorType.FLOAT32,
      transformation_input.subgraph,
  )

  # create dequantize_op
  dequant_op = qtyping.OperatorT()
  dequant_op.opcodeIndex = dequant_op_code_idx
  dequant_op.outputs = [new_tensor_id]
  dequant_op.inputs = [transformation_input.tensor_id]

  # quantize the source tensor
  quantize_tensor.quantize_tensor(transformation_input)

  # update the original consumers of the op to take the dequant op,
  # and find the first consumer of the new tensor
  first_consumer_id = min(transformation_input.consumers)
  for consumer_id in transformation_input.consumers:
    op = transformation_input.subgraph.operators[consumer_id]
    for input_idx in range(len(op.inputs)):
      if op.inputs[input_idx] == transformation_input.tensor_id:
        op.inputs[input_idx] = new_tensor_id

  # if the output is also an output to the graph, we need to update that as well
  for output_idx, output in enumerate(transformation_input.subgraph.outputs):
    if output == transformation_input.tensor_id:
      transformation_input.subgraph.outputs[output_idx] = new_tensor_id

  # add dequant into the subgraph op list,
  # must insert the op right before it's first consumer
  # in the case of output goes to graph output, we need to ensure the dequant
  # op is inserted after the producer
  op_id = max(transformation_input.producer + 1, first_consumer_id)
  transformation_input.subgraph.operators.insert(op_id, dequant_op)
  return qtyping.TransformationInfo(
      op_id=op_id, num_ops_added=1, output_tensor_id=new_tensor_id
  )
