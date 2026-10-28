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

"""Hadamard rotation decomposed pattern transformation."""

from flatbuffers import flexbuffers
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.transformations import transformation_utils


def _to_flexbuffer(
    hadamard_size: int,
    random_binary_vector: list[np.int8],
) -> bytes:
  """Converts hadamard_size to flexbuffer."""
  fbb = flexbuffers.Builder()
  with fbb.Map():
    fbb.Int('hadamard_size', hadamard_size)
    fbb.VectorFromElements('random_binary_vector', random_binary_vector)
  return fbb.Finish()


def _update_embedding_lookup_consumers(
    transformation: transformation_utils.TransformationInput,
    new_tensor_id: int,
) -> bool:  # pyrefly: ignore[bad-return]
  """Updates the consumers of the embedding lookup op to use the new tensor.

  Args:
    transformation: The transformation input to update the consumers of.
    new_tensor_id: The new tensor id to use as the input to the embedding lookup
      consumers.
  """
  for consumer in transformation.consumers:
    # If the consumer is a graph output and not an op, we can ignore it here
    # since the graph output will be updated later.
    if consumer == -1:
      continue
    consumer_op = transformation.subgraph.operators[consumer]
    # Find the input that was attached to the insertion point, and replace it
    # with the new tensor.
    for i in range(len(consumer_op.inputs)):
      if consumer_op.inputs[i] == transformation.tensor_id:
        consumer_op.inputs[i] = new_tensor_id


def _make_hadamard_matrix(
    size: int,
    random_binary_vector: np.ndarray | None = None,
):
  """Generates a (optionally sign-flipped) Hadamard matrix of the given size.

  Args:
    size: The size of the Hadamard matrix. Must be a power of 2. This represents
      a single dimension. E.g. if size is 4, then the Hadamard matrix is a 4x4
      matrix.
    random_binary_vector: Optional Rademacher `+/-1` sign vector of length
      `size` (or length 1) applied as `diag(random_binary_vector) @ H`.

  Returns:
    The Hadamard matrix.

  Raises:
    ValueError: If the size is not a power of 2.
  """
  if size <= 0 or (size & (size - 1)) != 0:
    raise ValueError('Hadamard matrix size must be a power of 2. ')
  h = h2 = np.array([[1, 1], [1, -1]])
  current_size = 2
  while current_size < size:
    h = np.kron(h, h2)
    current_size *= 2
  h_norm = h / np.sqrt(size)
  if random_binary_vector is not None:
    signs = np.asarray(random_binary_vector, dtype=np.float32).reshape(-1, 1)
    h_norm = signs * h_norm
  return h_norm


def _make_reshape_op(
    input_tensor_id: int,
    target_shape: list[int],
    shape_name,
    output_name,
    subgraph: qtyping.SubGraphT,
    model: qtyping.ModelT,
) -> tuple[qtyping.OperatorT, int]:
  """Creates a TFLite RESHAPE op and returns `(op, output_tensor_id)`."""
  shape_tensor_id = transformation_utils.add_new_constant_tensor(
      shape_name,
      np.array(target_shape, dtype=np.int32),
      qtyping.TensorType.INT32,
      subgraph,
      model,
  )
  output_tensor_id = transformation_utils.add_new_activation_tensor(
      output_name,
      target_shape,
      qtyping.TensorType.FLOAT32,
      subgraph,
  )
  reshape_op = qtyping.OperatorT()
  reshape_op.opcodeIndex = transformation_utils.add_op_code(
      qtyping.BuiltinOperator.RESHAPE,
      model.operatorCodes,
      'RESHAPE',
  )
  reshape_op.inputs = [input_tensor_id, shape_tensor_id]
  reshape_op.outputs = [output_tensor_id]
  return reshape_op, output_tensor_id


def _make_transpose_op(
    input_tensor_id: int,
    perm: list[int],
    output_shape: list[int],
    perm_name,
    output_name,
    subgraph: qtyping.SubGraphT,
    model: qtyping.ModelT,
) -> tuple[qtyping.OperatorT, int]:
  """Creates a TFLite TRANSPOSE op and returns `(op, output_tensor_id)`."""
  perm_tensor_id = transformation_utils.add_new_constant_tensor(
      perm_name,
      np.array(perm, dtype=np.int32),
      qtyping.TensorType.INT32,
      subgraph,
      model,
      allow_tensor_sharing=True,
  )
  output_tensor_id = transformation_utils.add_new_activation_tensor(
      output_name,
      output_shape,
      qtyping.TensorType.FLOAT32,
      subgraph,
  )
  transpose_op = qtyping.OperatorT()
  transpose_op.opcodeIndex = transformation_utils.add_op_code(
      qtyping.BuiltinOperator.TRANSPOSE,
      model.operatorCodes,
      'TRANSPOSE',
  )
  transpose_op.inputs = [input_tensor_id, perm_tensor_id]
  transpose_op.outputs = [output_tensor_id]
  return transpose_op, output_tensor_id


def _make_fc_op(
    input_tensor_id: int,
    weight_matrix: np.ndarray,
    weight_name,
    output_shape: list[int],
    output_name,
    subgraph: qtyping.SubGraphT,
    model: qtyping.ModelT,
    keep_num_dims: bool = False,
) -> tuple[qtyping.OperatorT, int]:
  """Creates a TFLite FULLY_CONNECTED op and returns `(op, output_tensor_id)`."""
  weight_tensor_id = transformation_utils.add_new_constant_tensor(
      tensor_name=weight_name,
      data=np.ascontiguousarray(weight_matrix, dtype=np.float32),
      tensor_type=qtyping.TensorType.FLOAT32,
      subgraph=subgraph,
      model=model,
      allow_tensor_sharing=True,
  )
  output_tensor_id = transformation_utils.add_new_activation_tensor(
      output_name,
      output_shape,
      qtyping.TensorType.FLOAT32,
      subgraph,
  )
  fc_op = qtyping.OperatorT()
  fc_op.opcodeIndex = transformation_utils.add_op_code(
      qtyping.BuiltinOperator.FULLY_CONNECTED,
      model.operatorCodes,
      'FULLY_CONNECTED',
  )
  fc_op.inputs = [input_tensor_id, weight_tensor_id]
  fc_op.outputs = [output_tensor_id]
  fc_options = qtyping.FullyConnectedOptionsT()
  fc_options.fusedActivationFunction = qtyping.ActivationFunctionType.NONE
  if keep_num_dims:
    fc_options.keepNumDims = True
  fc_op.builtinOptionsType = qtyping.BuiltinOptions.FullyConnectedOptions
  fc_op.builtinOptions = fc_options
  return fc_op, output_tensor_id


def insert_decomposed_hadamard_rotation(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Inserts a decomposed pattern of Hadamard or Kronecker rotation on a tensor.

  When `cross_block_matrix` is None (default), inserts 3 built-in TFLite ops:
    x' = reshape(x, (-1, hadamard_size))
    x' = fully_connected(x', H(hadamard_size))
    x' = reshape(x', x.shape)

  When `cross_block_matrix` (`Q_1` of shape `(m, m)`) is set, inserts 6 built-in
  TFLite ops so both rotation matrices are standard quantizable FULLY_CONNECTED
  weights:
    x' = reshape(x, (B, m, hadamard_size))
    x' = fully_connected(x', H(hadamard_size), keep_num_dims=True)
    x' = transpose(x', (0, 2, 1))
    x' = fully_connected(x', Q_1.T if is_embedding else Q_1, keep_num_dims=True)
    x' = transpose(x', (0, 2, 1))
    x' = reshape(x', x.shape)

  Args:
    transformation_input: The transformation input to insert the ops on.

  Returns:
    The transformation info of the inserted ops.

  Raises:
    ValueError: If the transformation input is not a uniform quantization
    transformation.
    ValueError: If the Hadamard quantization params are not set.
    ValueError: If the tensor is not a float32 tensor.
    ValueError: If no supported ops were found as the tensor's producer or
    consumers.
  """
  if not isinstance(
      transformation_input.quant_params, qtyping.UniformQuantParams
  ):
    raise ValueError('Hadamard rotation supports uniform quantization only')

  if transformation_input.quant_params.hadamard is None:
    raise ValueError(
        'Hadamard rotation quantization params are not set but op insertion is'
        ' requested.'
    )

  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  if tensor.type != qtyping.TensorType.FLOAT32:
    raise ValueError(
        'The Hadamard rotation op supports float32 tensors only. Got'
        f' {tensor.type} tensor.'
    )

  hadamard_size = transformation_input.quant_params.hadamard.hadamard_size
  cross_block_matrix = (
      transformation_input.quant_params.hadamard.cross_block_matrix
  )
  is_embedding = (
      transformation_utils.get_producer_schema_op_id(transformation_input)
      == qtyping.BuiltinOperator.EMBEDDING_LOOKUP
  )
  tensor_size = int(np.prod(tensor.shape))
  hadamard_matrix = _make_hadamard_matrix(
      hadamard_size,
      transformation_input.quant_params.hadamard.random_binary_vector,
  )
  hadamard_weight = hadamard_matrix if is_embedding else hadamard_matrix.T

  if cross_block_matrix is None:
    prerotate_shape = [tensor_size // hadamard_size, hadamard_size]
    prerotate_op, prerotate_tensor_id = _make_reshape_op(
        transformation_input.tensor_id,
        prerotate_shape,
        tensor.name + b'_prerotate_shape',
        tensor.name + b'_prerotate_reshaped',
        transformation_input.subgraph,
        transformation_input.model,
    )
    fc_op, rotated_tensor_id = _make_fc_op(
        prerotate_tensor_id,
        hadamard_weight,
        tensor.name + b'_hadamard_matrix',
        prerotate_shape,
        tensor.name + b'_rotated',
        transformation_input.subgraph,
        transformation_input.model,
        keep_num_dims=False,
    )
    post_reshape_op, post_reshape_output_tensor_id = _make_reshape_op(
        rotated_tensor_id,
        list(tensor.shape),
        tensor.name + b'_postrotate_shape',
        tensor.name + b'_postrotate_reshaped',
        transformation_input.subgraph,
        transformation_input.model,
    )
    inserted_ops = [prerotate_op, fc_op, post_reshape_op]
  else:
    num_blocks = int(cross_block_matrix.shape[0])
    batch_dim = tensor_size // (num_blocks * hadamard_size)
    shape_bmd = [batch_dim, num_blocks, hadamard_size]
    shape_bdm = [batch_dim, hadamard_size, num_blocks]
    cross_matrix = cross_block_matrix.T if is_embedding else cross_block_matrix

    prerotate_op, prerotate_tensor_id = _make_reshape_op(
        transformation_input.tensor_id,
        shape_bmd,
        tensor.name + b'_prerotate_shape',
        tensor.name + b'_prerotate_reshaped',
        transformation_input.subgraph,
        transformation_input.model,
    )
    fc_h_op, after_h_tensor_id = _make_fc_op(
        prerotate_tensor_id,
        hadamard_weight,
        tensor.name + b'_hadamard_matrix',
        shape_bmd,
        tensor.name + b'_hadamard_rotated',
        transformation_input.subgraph,
        transformation_input.model,
        keep_num_dims=True,
    )
    transpose_pre_op, pre_q_tensor_id = _make_transpose_op(
        after_h_tensor_id,
        [0, 2, 1],
        shape_bdm,
        tensor.name + b'_transpose_perm',
        tensor.name + b'_transposed_pre_q',
        transformation_input.subgraph,
        transformation_input.model,
    )
    fc_q_op, after_q_tensor_id = _make_fc_op(
        pre_q_tensor_id,
        cross_matrix,
        tensor.name + b'_cross_block_matrix',
        shape_bdm,
        tensor.name + b'_cross_rotated',
        transformation_input.subgraph,
        transformation_input.model,
        keep_num_dims=True,
    )
    transpose_post_op, rotated_tensor_id = _make_transpose_op(
        after_q_tensor_id,
        [0, 2, 1],
        shape_bmd,
        tensor.name + b'_transpose_perm',
        tensor.name + b'_rotated',
        transformation_input.subgraph,
        transformation_input.model,
    )
    post_reshape_op, post_reshape_output_tensor_id = _make_reshape_op(
        rotated_tensor_id,
        list(tensor.shape),
        tensor.name + b'_postrotate_shape',
        tensor.name + b'_postrotate_reshaped',
        transformation_input.subgraph,
        transformation_input.model,
    )
    inserted_ops = [
        prerotate_op,
        fc_h_op,
        transpose_pre_op,
        fc_q_op,
        transpose_post_op,
        post_reshape_op,
    ]

  # Update the users of this tensor to use the new tensor.
  if is_embedding:
    _update_embedding_lookup_consumers(
        transformation_input, post_reshape_output_tensor_id
    )
    for i, output in enumerate(transformation_input.subgraph.outputs):
      if output == transformation_input.tensor_id:
        transformation_input.subgraph.outputs[i] = post_reshape_output_tensor_id
        if getattr(transformation_input.model, 'signatureDefs', None):
          for sig_def in transformation_input.model.signatureDefs:
            for sig_out in getattr(sig_def, 'outputs', None) or []:
              if sig_out.tensorIndex == transformation_input.tensor_id:
                sig_out.tensorIndex = post_reshape_output_tensor_id
  elif not transformation_utils.update_fully_connected_consumers(
      transformation_input, post_reshape_output_tensor_id
  ):
    raise ValueError(
        'The Hadamard rotation op supports embedding lookup and fully connected'
        ' ops only, but no such ops were found.'
    )

  # Find the actual insertion point. The insertion point should be after the
  # producer op and before the first consumer op. The max() operation ensures
  # that we're not using -1 as the insertion point.
  first_consumer_id = min(transformation_input.consumers)
  op_id = max(transformation_input.producer + 1, first_consumer_id)

  for offset, op in enumerate(inserted_ops):
    transformation_input.subgraph.operators.insert(op_id + offset, op)

  return qtyping.TransformationInfo(
      op_id=op_id,
      num_ops_added=len(inserted_ops),
      output_tensor_id=post_reshape_output_tensor_id,
  )
