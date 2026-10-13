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

import dataclasses
from typing import Any, Optional
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import octav
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


CUSTOM_OP_ALGORITHM_KEY = "HADAMARD_ROTATION"
DECOMPOSED_ALGORITHM_KEY = "DECOMPOSED_HADAMARD_ROTATION"

# Theoretical crossover dimension `m` (`D = m * d`) when `m` is not a power of 2
# between skew-symmetric Rademacher Cayley (`m <= 48`) and Polar SVD (`m > 48`):
#   * Rademacher Cayley (`Q = (I - A)(I + A)^{-1}`, `A^T = -A`,
#     `|a_ij| = 1 / (sqrt(m) + 1)`): `(I + A)^T (I + A) = I + A^T A >= I`
#     bounds all singular values below by `1` and keeps off-diagonal entries
#     concentrated at `O(1 / sqrt(m))` without Gaussian tail spikes. However,
#     by Wigner's semicircle Stieltjes transform on `2(I + A)^{-1} - I`, the
#     diagonal entries retain a non-vanishing `O(1)` floor as `m -> inf`:
#       `E[Q_ii] = (sqrt(1 + 4 * alpha_m^2) - 1 - alpha_m^2) / alpha_m^2`
#       `-> sqrt(5) - 2 ~= 0.236` (`alpha_m = sqrt(m) / (sqrt(m) + 1)`).
#   * Polar SVD (`Q = U @ V^T` of `R in {-1, +1}^{m x m}`): zero diagonal bias
#     (`E[Q_ii^2] = 1 / m`), so expected column peak magnitude decays to zero
#     as `sqrt(2 * ln(m) / m)`. Below `m ~= 48`, sub-Gaussian tail spikes
#     (`sqrt(2 * ln(m) / m) > 0.40`) and small-`m` singular value dispersion
#     exceed Cayley's entries; above `m ~= 48`, `sqrt(2 * ln(m) / m)` drops
#     into and below Cayley's non-vanishing diagonal floor (`~0.33 -> 0.236`),
#     giving Polar SVD lower worst-case outlier coherence for all larger `m`.
_HYBRID_MAX_DIM = 48
VALID_Q_INIT_MODES = ("none", "auto")
VALID_SIZING_MODES = ("min_compute", "max_power_of_two")
VALID_SCALE_REFINEMENTS = ("none", "mse_grid")

_HADAMARD_MATRIX_CACHE = {}
_CROSS_BLOCK_MATRIX_CACHE = {}


def _select_kronecker_hadamard_size(
    dim: int,
    max_size: Optional[int] = None,
    sizing_mode: str = "min_compute",
) -> int:
  """Selects power-of-2 divisor `d | dim` using the specified `sizing_mode`."""
  if sizing_mode not in VALID_SIZING_MODES:
    raise ValueError(
        f"Unsupported sizing_mode: {sizing_mode!r}. Expected one of"
        f" {VALID_SIZING_MODES}."
    )
  max_pow2 = int(np.gcd(dim, 2**30))
  if max_size:
    max_pow2 = min(max_pow2, 1 << (max_size.bit_length() - 1))
  if max_pow2 < 2:
    return 1
  if sizing_mode == "max_power_of_two":
    return max_pow2

  best_d = 2
  best_cost = (dim // 2) + 2
  d = 2
  while d <= max_pow2:
    cost = (dim // d) + d
    if cost <= best_cost:
      best_cost = cost
      best_d = d
    d *= 2
  return best_d


def _make_rademacher_cayley_matrix(
    m: int, rng: np.random.Generator
) -> np.ndarray:
  """Generates orthogonal `Q_1 = (I - A)(I + A)^{-1}` with Rademacher skew `A`."""
  scale = 1.0 / (np.sqrt(float(m)) + 1.0)
  num_upper = (m * (m - 1)) // 2
  signs = rng.choice([-scale, scale], size=num_upper).astype(np.float64)
  skew = np.zeros((m, m), dtype=np.float64)
  rows, cols = np.triu_indices(m, k=1)
  skew[rows, cols] = signs
  skew -= skew.T
  eye = np.eye(m, dtype=np.float64)
  q1 = np.linalg.solve(eye + skew.T, eye - skew.T).T
  return q1.astype(np.float32)


def _make_polar_svd_matrix(m: int, rng: np.random.Generator) -> np.ndarray:
  """Generates orthogonal `Q_1 = U @ V^T` via SVD of a Rademacher matrix."""
  signs = rng.choice([-1.0, 1.0], size=(m, m)).astype(np.float64)
  u, _, vh = np.linalg.svd(signs, full_matrices=False)
  return (u @ vh).astype(np.float32)


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


def _make_cross_block_matrix(
    m: int, q_init_mode: str, seed: int = 0
) -> Optional[np.ndarray]:
  """Constructs cross-block orthogonal matrix `Q_1` of shape `(m, m)`."""
  if q_init_mode not in VALID_Q_INIT_MODES:
    raise ValueError(
        f"Unsupported q_init_mode: {q_init_mode!r}. Expected one of"
        f" {VALID_Q_INIT_MODES}."
    )
  if q_init_mode == "none" or m <= 1:
    return None

  if (m & (m - 1)) == 0:
    return _make_hadamard_matrix(m).astype(np.float32)

  resolved_mode = "rademacher" if m <= _HYBRID_MAX_DIM else "polar_svd"
  cache_key = (m, resolved_mode, seed)
  if cache_key in _CROSS_BLOCK_MATRIX_CACHE:
    return _CROSS_BLOCK_MATRIX_CACHE[cache_key]

  rng = np.random.default_rng(seed)
  if resolved_mode == "rademacher":
    q1 = _make_rademacher_cayley_matrix(m, rng)
  else:
    q1 = _make_polar_svd_matrix(m, rng)

  _CROSS_BLOCK_MATRIX_CACHE[cache_key] = q1
  return q1


def _make_random_binary_vector(size: int, seed: int = 0) -> np.ndarray:
  """Generates a deterministic Rademacher (+/-1) sign vector of length `size`."""
  rng = np.random.default_rng([int(size), int(seed), 1])
  return rng.choice(np.array([-1, 1], dtype=np.int8), size=int(size))


def _rotate_with_diagonal_hadamard(
    tensor_content: np.ndarray,
    axis: int,
    max_size: int | None = None,
    q_init_mode: str = "none",
    q_seed: int = 0,
    randomize: bool = False,
    sizing_mode: str = "min_compute",
):
  """Rotates the float array using single Hadamard or Kronecker `(Q_1, H_d)`.

  Args:
    tensor_content: The float array to rotate before quantization.
    axis: The axis of the tensor to rotate.
    max_size: The maximum size of the Hadamard matrix.
    q_init_mode: Cross-block orthogonal matrix initializer (`"none"` or
      `"auto"`).
    q_seed: Random seed for constructing the cross-block orthogonal matrix and
      random sign flips.
    randomize: Whether to apply deterministic Rademacher `+/-1` sign flips
      (`diag(random_vector) @ H_d`) before the Hadamard transform.
    sizing_mode: Strategy for choosing Hadamard block size `d` when `q_init_mode
      != "none"` (`"min_compute"` or `"max_power_of_two"`).

  Returns:
    A tuple `(w_rotated, hadamard_size, random_vector, cross_block_matrix)`.

  Raises:
    ValueError: If the axis is not the last axis of tensor_content. To support
      other axes, please add support to the matrix multiplication.
  """
  if axis != tensor_content.ndim - 1:
    raise ValueError(
        "Hadamard rotation is only supported for tensors with quantized"
        " dimension 0 (rotate last dimension)."
    )

  dim = tensor_content.shape[axis]
  if q_init_mode == "none":
    hadamard_size = int(np.gcd(dim, 2**30))
    if max_size:
      hadamard_size = min(hadamard_size, 1 << (max_size.bit_length() - 1))
    cross_block_matrix = None
  else:
    hadamard_size = _select_kronecker_hadamard_size(
        dim, max_size, sizing_mode=sizing_mode
    )
    num_blocks = dim // hadamard_size
    cross_block_matrix = _make_cross_block_matrix(
        num_blocks, q_init_mode=q_init_mode, seed=q_seed
    )

  if randomize:
    random_vector = _make_random_binary_vector(hadamard_size, seed=q_seed)
  else:
    random_vector = np.ones(hadamard_size, dtype=np.int8)
  hadamard = random_vector.astype(np.float32)[
      :, np.newaxis
  ] * _make_hadamard_matrix(hadamard_size)

  if cross_block_matrix is None:
    reshaped_tensor = tensor_content.reshape((-1, hadamard_size), copy=False)
    w_rotated = np.matmul(reshaped_tensor, hadamard)
  else:
    num_blocks = cross_block_matrix.shape[0]
    reshaped_2d = tensor_content.reshape((-1, hadamard_size), copy=False)
    w_rotated = np.matmul(reshaped_2d, hadamard).reshape(
        (-1, num_blocks, hadamard_size), copy=False
    )
    w_rotated = np.ascontiguousarray(
        np.tensordot(cross_block_matrix, w_rotated, axes=([1], [1])).transpose(
            1, 0, 2
        )
    )

  return (
      w_rotated.reshape(tensor_content.shape, copy=False),
      hadamard_size,
      random_vector,
      cross_block_matrix,
  )


def _refine_quant_params_mse(
    w_rotated: np.ndarray,
    qparams: qtyping.UniformQuantParams,
) -> qtyping.UniformQuantParams:
  """Refines per-channel symmetric scale and quantized_data to minimize L2 MSE."""
  if (
      not qparams.symmetric
      or qparams.block_size != 0
      or qparams.quantized_dimension != 0
      or w_rotated.ndim != 2
      or qparams.quantized_data is None
  ):
    return qparams

  m_rows, d_cols = w_rotated.shape
  orig_scale_shape = qparams.scale.shape
  orig_scale_dtype = qparams.scale.dtype
  orig_q_dtype = qparams.quantized_data.dtype

  qmax = float((1 << (qparams.num_bits - 1)) - 1)
  narrow = qparams.symmetric and qparams.num_bits >= 8
  qmin = float(-(1 << (qparams.num_bits - 1)) + (1 if narrow else 0))

  base_scale = qparams.scale.reshape(m_rows, 1).astype(np.float32)
  best_scale = base_scale.copy()
  best_q = qparams.quantized_data.reshape(m_rows, d_cols).copy()
  if narrow:
    best_q = np.clip(best_q, int(qmin), int(qmax))

  num_alphas = 15 if m_rows > 16384 else 25
  alphas = np.linspace(0.80, 1.20, num_alphas, dtype=np.float32)
  chunk_size = 4096
  for start in range(0, m_rows, chunk_size):
    end = min(start + chunk_size, m_rows)
    w_chunk = w_rotated[start:end].astype(np.float32, copy=False)
    s_cur = best_scale[start:end].copy()
    q_cur = best_q[start:end].astype(np.float32)
    err_cur = np.sum((w_chunk - q_cur * s_cur) ** 2, axis=1, keepdims=True)

    s_base_chunk = base_scale[start:end]
    for alpha in alphas:
      s_cand = np.maximum(s_base_chunk * alpha, 1e-9)
      q_cand = np.clip(np.rint(w_chunk / s_cand), qmin, qmax)
      err_cand = np.sum((w_chunk - q_cand * s_cand) ** 2, axis=1, keepdims=True)
      improved = err_cand < err_cur
      err_cur = np.where(improved, err_cand, err_cur)
      s_cur = np.where(improved, s_cand, s_cur)
      q_cur = np.where(improved, q_cand, q_cur)

    # Perform 2 final alternating least-squares steps with exact re-rounding
    s_cand = s_cur
    for _ in range(2):
      qq = np.sum(q_cur * q_cur, axis=1, keepdims=True)
      wq = np.sum(w_chunk * q_cur, axis=1, keepdims=True)
      s_cand = np.where(qq > 0.0, wq / np.maximum(qq, 1e-12), s_cand)
      s_cand = np.maximum(s_cand, 1e-9)
      q_cand = np.clip(np.rint(w_chunk / s_cand), qmin, qmax)
      err_cand = np.sum((w_chunk - q_cand * s_cand) ** 2, axis=1, keepdims=True)
      improved = err_cand < err_cur
      err_cur = np.where(improved, err_cand, err_cur)
      s_cur = np.where(improved, s_cand, s_cur)
      q_cur = np.where(improved, q_cand, q_cur)

    best_scale[start:end] = s_cur
    best_q[start:end] = q_cur.astype(orig_q_dtype)

  return dataclasses.replace(
      qparams,
      scale=best_scale.reshape(orig_scale_shape).astype(orig_scale_dtype),
      quantized_data=best_q.reshape(w_rotated.shape),
  )


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
    is_decomposed: bool = False,
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
    is_decomposed: Whether decomposed Hadamard rotation ops are used.

  Raises:
    ValueError: If the blockwise quantization is requested.
    ValueError: If the asymmetric quantization is requested.
    ValueError: `tensor_qsv` must contain min/max values, or `tensor_content`
      must be provided so that they can be inferred.
  """
  del is_decomposed
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
  alg_params = tensor_quant_config.algorithm_params or {}

  # Rotate the tensor with a Hadamard (or Kronecker) orthogonal transformation.
  (
      w_rotated,
      hadamard_size,
      random_vector,
      cross_block_matrix,
  ) = _rotate_with_diagonal_hadamard(
      tensor_content,
      axis=reduce_axis,
      max_size=alg_params.get("max_hadamard_size"),
      q_init_mode=alg_params.get("q_init_mode", "none"),
      q_seed=alg_params.get("q_seed", 0),
      randomize=alg_params.get("randomize", False),
      sizing_mode=alg_params.get("sizing_mode", "min_compute"),
  )

  # Get the quantized values of the rotated tensor.
  qparams = octav.get_tensor_quant_params(
      op_info, tensor_quant_config, w_rotated, tensor_qsv
  )

  scale_refinement = alg_params.get("scale_refinement", "none")
  if scale_refinement not in VALID_SCALE_REFINEMENTS:
    raise ValueError(
        f"Unsupported scale_refinement: {scale_refinement!r}. Expected one of"
        f" {VALID_SCALE_REFINEMENTS}."
    )
  if scale_refinement == "mse_grid":
    qparams = _refine_quant_params_mse(w_rotated, qparams)

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
          cross_block_matrix=cross_block_matrix,
      ),
  )


def _materialize_fully_connected(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    is_decomposed: bool = False,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,  # pylint: disable=unused-argument
) -> list[qtyping.TensorTransformationParams]:
  """Materialize the fully_connected op."""
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
    quant_params = get_tensor_quant_params(
        op_info,
        op_info.op_quant_config.weight_tensor_config,
        tensor_data,
        None,
        is_decomposed=is_decomposed,
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

  # Materialize input.
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

  # Materialize bias.
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

  # Materialize output.
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
  """Materialize the embedding_lookup op."""
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

  # Materialize embedding.
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
        is_decomposed=is_decomposed,
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

  # Materialize output.
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
