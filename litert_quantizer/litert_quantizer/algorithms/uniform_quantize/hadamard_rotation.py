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

"""Implements the Hadamard Rotation quantization.

Hadamard rotation mitigates the quantization loss caused by the large
eigenvalues of the weight matrix (outliers) by rotating the weights with a
Hadamard matrix (H). The rotation distributes the larger eigenvalues more evenly
across the Hadamard transformed weight matrix. In other words, it "smears" the
outliers across the entire tensor, making the distribution more uniform.

Matrix H is a square matrix with entries +1 or -1 and orthogonal rows and cols
(H * H^T = n * I, where n is the dimension). The rotation is often performed
using a normalized Hadamard matrix (1/sqrt(n) * H), which is orthogonal.

A fully connected linear layer with weight W and input X can be written as:
Y = X * W^T = (X * H) * (W * H)^T
- The input X is rotated by H (updated online during inference)
- The weight W is rotated by H (updated offline during quantization, with OCTAV)
"""

from typing import Any, Optional
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import octav
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


CUSTOM_OP_ALGORITHM_KEY = "HADAMARD_ROTATION"
DECOMPOSED_ALGORITHM_KEY = "DECOMPOSED_HADAMARD_ROTATION"

_HADAMARD_MATRIX_CACHE = {}


def _make_hadamard_matrix(size: int) -> np.ndarray:
  """Generates a Hadamard matrix of the given size.

  Hadamard matrices are square matrices with entries +1 or -1 and orthogonal
  rows and columns. They are defined recursively:

    H_1 = [[1]]
    H_{2n} = [[H_n, H_n], [H_n, -H_n]]

  Args:
    size: The size of the Hadamard matrix. Must be a power of 2. This represents
      a single dimension. E.g. if size is 4, then the Hadamard matrix is a 4x4
      matrix.

  Returns:
    The Hadamard matrix.

  Raises:
    ValueError: If the size is not a power of 2.
  """
  if not isinstance(size, int):
    size = int(size)

  if size <= 0 or (size & (size - 1)) != 0:
    raise ValueError("Hadamard matrix size must be a power of 2. ")

  # Check whether we already have a Hadamard matrix for this size.
  if size in _HADAMARD_MATRIX_CACHE:
    return _HADAMARD_MATRIX_CACHE[size]

  # Otherwise, create one.
  h_int8 = h2 = np.array([[1, 1], [1, -1]], dtype=np.int8)
  h_norm = h_int8 / np.sqrt(2, dtype=np.float32)
  current_size = 2
  while current_size < size:
    h_int8 = np.kron(h_int8, h2)
    current_size *= 2
    if current_size not in _HADAMARD_MATRIX_CACHE:
      h_norm = h_int8 / np.sqrt(current_size, dtype=np.float32)
      _HADAMARD_MATRIX_CACHE[current_size] = h_norm
    else:
      h_norm = _HADAMARD_MATRIX_CACHE[current_size]
  return h_norm


def _rotate_with_diagonal_hadamard(
    tensor_content: np.ndarray,
    axis: int,
    max_size: int | None = None,
):
  """Quantizes the given float array using the diagonal Hadamard algorithm.

  Args:
    tensor_content: The float array to quantize.
    axis: The axis of the tensor to rotate.
    max_size: The maximum size of the Hadamard matrix.

  Returns:
    A tuple containing the quantized array and the recovered array.

  Raises:
    ValueError: If the axis is not the last axis of tensor_content. To support
      other axes, please add support to the matrix multiplication.
  """
  if axis != tensor_content.ndim - 1:
    raise ValueError(
        "Hadamard rotation is only supported for tensors with quantized"
        " dimension 0 (rotate last dimension)."
    )

  # Use the largest power of 2 that is a factor of the dimension and then
  # tile this Hadamard matrix along the diagonal. 2**30 is just a large power
  # of 2 to calculate this factor.
  hadamard_size = np.gcd(tensor_content.shape[axis], 2**30)
  if max_size:
    hadamard_size = min(hadamard_size, 1 << (max_size.bit_length() - 1))
  random_vector = np.ones(hadamard_size, dtype=np.int8)

  # Use a canonical Hadamard matrix.
  hadamard = _make_hadamard_matrix(hadamard_size)
  reshaped_tensor = tensor_content.reshape((-1, hadamard_size), copy=False)
  w_rotated = np.matmul(reshaped_tensor, hadamard)
  return (
      w_rotated.reshape(tensor_content.shape, copy=False),
      hadamard_size,
      random_vector,
  )


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Returns the quantization parameters for a tensor.

  This function will rotate the tensor with a Hadamard matrix and then
  quantize it with OCTAV.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor. When None, it means the tensor is
      not a weight tensor (e.g. static quantization).
    tensor_qsv: A dictionary containing the min/max of the tensor.

  Raises:
    ValueError: If the blockwise quantization is requested.
    ValueError: If the asymmetric quantization is requested.
    ValueError: `tensor_qsv` must contain min/max values, or `tensor_content`
      must be provided so that they can be inferred.
  """
  if tensor_content is None:
    raise ValueError("Hadamard rotation is only supported for weight tensors.")

  if tensor_qsv is not None:
    raise ValueError(
        "Hadamard rotation is not supported for static quantization."
    )

  if tensor_content.ndim < 2:
    raise ValueError(
        "Hadamard rotation is only supported for tensors with rank >= 2."
    )

  # Reduction axis is the last non-quantized dimension. Since we only support
  # quantized_dim of 0 (or 1 for blockwise), the reduction axis is the last
  # axis.
  reduce_axis = tensor_content.ndim - 1

  # Rotate the tensor with a Hadamard matrix.
  w_rotated, hadamard_size, random_vector = _rotate_with_diagonal_hadamard(
      tensor_content,
      axis=reduce_axis,
      max_size=tensor_quant_config.algorithm_params.get("max_hadamard_size"),
  )

  # Get the quantized values of the rotated tensor.
  qparams = octav.get_tensor_quant_params(
      op_info, tensor_quant_config, w_rotated, tensor_qsv
  )

  return qtyping.UniformQuantParams(
      quantized_dimension=qparams.quantized_dimension,
      num_bits=qparams.num_bits,
      scale=qparams.scale,
      zero_point=qparams.zero_point,
      symmetric=qparams.symmetric,
      quantized_data=qparams.quantized_data,
      block_size=qparams.block_size,
      hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
          random_binary_vector=random_vector,
          hadamard_size=hadamard_size,
      ),
  )


def _materialize_fully_connected(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    is_decomposed: bool = False,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,  # pylint: disable=unused-argument
) -> list[qtyping.TensorTransformationParams]:
  """Materialize the fully_connected op.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.
    is_decomposed: Whether to use decomposed Hadamard rotation ops or a custom
      op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  if op_info.op_quant_config.weight_tensor_config is None:
    raise ValueError(
        "Weight tensor quantization config is not provided for Hadamard"
        " Rotation quantization."
    )

  op_tensor_params = []

  # Materialize weight.
  weight_tensor_index = 1
  weight_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[weight_tensor_index]
  ]
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      weight_tensor, graph_info.buffers
  )

  if not (
      quant_params := tensor_quant_params_cache.lookup(
          weight_tensor.buffer, op_info.op_quant_config.weight_tensor_config
      )
  ):
    # quant_params contains the rotated and quantized weights done by
    # get_tensor_quant_params().
    quant_params = get_tensor_quant_params(
        op_info,
        op_info.op_quant_config.weight_tensor_config,
        tensor_data,
        None,
    )
    tensor_quant_params_cache.insert(
        weight_tensor.buffer,
        op_info.op_quant_config.weight_tensor_config,
        quant_params,
    )

  transformations = [qtyping.QuantTransformation.QUANTIZE_TENSOR]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  weight_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(weight_tensor),
      consumers=[op2tensor_params],
  )

  # Materialize input. A hadamard rotation op should be inserted on the input
  # tensor to do the inverse of the weight's transformation.
  input_tensor_index = 0
  input_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[input_tensor_index]
  ]
  transformations = [
      qtyping.QuantTransformation.INSERT_DECOMPOSED_HADAMARD_ROTATION
      if is_decomposed
      else qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION,
  ]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  input_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(input_tensor),
      consumers=[op2tensor_params],
  )
  op_tensor_params.append(input_transformation_params)
  op_tensor_params.append(weight_transformation_params)

  # Materialize bias. Since static quantization is not supported, we do not
  # quantize the bias tensor.
  bias_tensor_index = 2
  bias_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[bias_tensor_index]
  ]
  no_quant_tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
  )
  bias_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(bias_tensor),
      consumers=[no_quant_tensor_params],
  )
  op_tensor_params.append(bias_transformation_params)

  # Materialize output. Since static quantization is not supported, we do not
  # quantize the output tensor.
  output_tensor_index = 0
  output_tensor = graph_info.subgraph_tensors[
      op_info.op.outputs[output_tensor_index]
  ]
  no_quant_tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
  )
  output_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(output_tensor),
      producer=no_quant_tensor_params,
  )
  op_tensor_params.append(output_transformation_params)

  return op_tensor_params


def materialize_fully_connected_custom_op(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,
) -> list[qtyping.TensorTransformationParams]:
  return _materialize_fully_connected(
      op_info,
      graph_info,
      is_decomposed=False,
      tensor_name_to_qsv=tensor_name_to_qsv,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_fully_connected_decomposed(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,
) -> list[qtyping.TensorTransformationParams]:
  return _materialize_fully_connected(
      op_info,
      graph_info,
      is_decomposed=True,
      tensor_name_to_qsv=tensor_name_to_qsv,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def _materialize_embedding_lookup(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    is_decomposed: bool = False,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,  # pylint: disable=unused-argument
) -> list[qtyping.TensorTransformationParams]:
  """Materialize the embedding_lookup op.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.
    is_decomposed: Whether to use decomposed Hadamard rotation ops or a custom
      op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  op_tensor_params = []

  # Materialize lookup.
  lookup_tensor_index = 0
  lookup_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[lookup_tensor_index]
  ]
  transformations = [
      qtyping.QuantTransformation.NO_QUANTIZE,
  ]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=None,
      transformations=transformations,
  )
  lookup_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(lookup_tensor),
      consumers=[op2tensor_params],
  )
  op_tensor_params.append(lookup_transformation_params)

  # Materialize embedding. The embedding table should be rotated and then
  # quantized.
  embedding_tensor_index = 1
  embedding_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[embedding_tensor_index]
  ]
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      embedding_tensor, graph_info.buffers
  )

  if tensor_quant_params_cache is None or not (
      quant_params := tensor_quant_params_cache.lookup(
          embedding_tensor.buffer, op_info.op_quant_config.weight_tensor_config  # pyrefly: ignore[bad-argument-type]
      )
  ):
    quant_params = get_tensor_quant_params(
        op_info,
        op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
        tensor_data,
        None,
    )
    if tensor_quant_params_cache:
      tensor_quant_params_cache.insert(
          embedding_tensor.buffer,
          op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
          quant_params,
      )

  transformations = [qtyping.QuantTransformation.QUANTIZE_TENSOR]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  weight_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(embedding_tensor),
      consumers=[op2tensor_params],
  )
  op_tensor_params.append(weight_transformation_params)

  # Materialize output. A hadamard rotation op should be inserted on the output
  # tensor to do the inverse of the embedding's transformation.
  output_tensor_index = 0
  output_tensor = graph_info.subgraph_tensors[
      op_info.op.outputs[output_tensor_index]
  ]
  transformations = [
      qtyping.QuantTransformation.INSERT_DECOMPOSED_HADAMARD_ROTATION
      if is_decomposed
      else qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION,
  ]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  output_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(output_tensor),
      producer=op2tensor_params,
  )
  op_tensor_params.append(output_transformation_params)

  return op_tensor_params


def materialize_embedding_lookup_custom_op(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,
) -> list[qtyping.TensorTransformationParams]:
  return _materialize_embedding_lookup(
      op_info,
      graph_info,
      is_decomposed=False,
      tensor_name_to_qsv=tensor_name_to_qsv,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_embedding_lookup_decomposed(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,
) -> list[qtyping.TensorTransformationParams]:
  return _materialize_embedding_lookup(
      op_info,
      graph_info,
      is_decomposed=True,
      tensor_name_to_qsv=tensor_name_to_qsv,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )
