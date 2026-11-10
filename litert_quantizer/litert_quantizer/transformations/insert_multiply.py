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

"""Insert multiply pattern transformation."""

import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.transformations import transformation_utils


def insert_multiply(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Inserts an elementwise MUL op on this tensor for activation scaling.

  This function works for float32 tensors. It inserts a constant multiplier
  tensor and a BuiltinOperator.MUL op:
    x' = tfl.mul(x, multiplier)
  and rewires consumer fully_connected ops to read x'.

  Args:
    transformation_input: The transformation input to insert the op on.

  Returns:
    The transformation info of the inserted op.

  Raises:
    ValueError: If the transformation input is not uniform quantization.
    ValueError: If custom_algorithm_param or multiplier is not set.
    ValueError: If the tensor is not a float32 tensor.
    ValueError: If no supported ops were found as the tensor's consumers.
  """
  if not isinstance(
      transformation_input.quant_params, qtyping.UniformQuantParams
  ):
    raise ValueError('Insert multiply supports uniform quantization only.')

  if (
      transformation_input.quant_params.custom_algorithm_param is None
      or 'multiplier'
      not in transformation_input.quant_params.custom_algorithm_param
  ):
    raise ValueError(
        'Custom algorithm parameter "multiplier" is not set but multiply op'
        ' insertion is requested.'
    )

  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  if tensor.type != qtyping.TensorType.FLOAT32:
    raise ValueError(
        'The insert multiply op supports float32 tensors only. Got'
        f' {tensor.type} tensor.'
    )

  multiplier_data = np.asarray(
      transformation_input.quant_params.custom_algorithm_param['multiplier'],
      dtype=np.float32,
  )

  # Create constant tensor for multiplier.
  multiplier_tensor_id = transformation_utils.add_new_constant_tensor(
      tensor_name=tensor.name + b'_multiplier',
      data=multiplier_data,
      tensor_type=qtyping.TensorType.FLOAT32,
      subgraph=transformation_input.subgraph,
      model=transformation_input.model,
      allow_tensor_sharing=True,
  )

  # Create output activation tensor.
  output_shape = list(
      tensor.shapeSignature
      if tensor.shapeSignature is not None
      else tensor.shape
  )
  mul_output_tensor_id = transformation_utils.add_new_activation_tensor(
      tensor_name=tensor.name + b'_scaled',
      shape=output_shape,
      tensor_type=qtyping.TensorType.FLOAT32,
      subgraph=transformation_input.subgraph,
  )

  # Create MUL operator.
  mul_op_code_idx = transformation_utils.add_op_code(
      qtyping.BuiltinOperator.MUL,
      transformation_input.model.operatorCodes,
      'MUL',
  )
  mul_op = qtyping.OperatorT()
  mul_op.opcodeIndex = mul_op_code_idx
  mul_op.inputs = [transformation_input.tensor_id, multiplier_tensor_id]
  mul_op.outputs = [mul_output_tensor_id]
  mul_options = qtyping.MulOptionsT()
  mul_options.fusedActivationFunction = qtyping.ActivationFunctionType.NONE
  mul_op.builtinOptionsType = qtyping.BuiltinOptions.MulOptions
  mul_op.builtinOptions = mul_options

  # Update consumer fully connected ops.
  if not transformation_utils.update_fully_connected_consumers(
      transformation_input, mul_output_tensor_id
  ):
    raise ValueError(
        'The insert multiply op supports fully connected consumers only, but'
        ' no such ops were found.'
    )

  # Find insertion point.
  first_consumer_id = min(transformation_input.consumers)
  op_id = max(transformation_input.producer + 1, first_consumer_id)

  transformation_input.subgraph.operators.insert(op_id, mul_op)

  return qtyping.TransformationInfo(
      op_id=op_id,
      num_ops_added=1,
      output_tensor_id=mul_output_tensor_id,
  )
