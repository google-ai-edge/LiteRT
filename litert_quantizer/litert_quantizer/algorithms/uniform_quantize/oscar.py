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

"""Implements OSCAR quantization: activation-aware scaling and optimal clipping.

OSCAR (Optimal Scaling and Channel ARrangement) minimizes the
activation-weighted weight error E||(Q(W) - W) x||^2 under a diagonal activation
model.

For each FULLY_CONNECTED op:
1. `calibrate` collects per-input-channel activation second moments (mu2).
2. `materialize_fully_connected` computes optimal per-channel scales s, scales
   weights offline (W' = W * s), quantizes W' with activation-weighted optimal
   clipping bounds, and inserts an elementwise MUL op (x' = x * (1/s)) on the
   activation input.

Scope: FULLY_CONNECTED only.
"""

from collections.abc import MutableMapping, Sequence
import dataclasses
from typing import Any

from absl import logging
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


ALGORITHM_KEY = "OSCAR"
_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation


_EPS = 1e-12
_SCALE_CLAMP = (1e-4, 1e4)


def _floor_positive(mu2: np.ndarray) -> np.ndarray:
  """Guards against zero/negative masses (dead channels)."""
  mu2 = np.asarray(mu2, np.float64)
  return np.maximum(mu2, float(np.max(mu2)) * 1e-8 + _EPS)


def _optimal_group_clip(a: np.ndarray, m: np.ndarray, qmax: int) -> np.ndarray:
  """Exact per-row optimal clip bound via breakpoint scan.

  Minimizes, per row, E(c) = c^2 * M / (12 qmax^2)
                            + sum_j max(a_j - c, 0)^2 * m_j
  where M = sum_j m_j. Between consecutive sorted magnitudes the objective
  is quadratic in c, so each segment has a closed-form minimizer and the
  scan over segments is exact.

  Args:
    a: (n, g) weight magnitudes within one group.
    m: (g,) per-column masses (activation second moments).
    qmax: The largest quantized level, 2^(bits-1) - 1.

  Returns:
    (n,) optimal clip bound per row.
  """
  n, _ = a.shape
  order = np.argsort(-a, axis=1)
  a_s = np.take_along_axis(a, order, 1)  # descending
  m_s = m[order]
  mass = float(m.sum()) + _EPS

  s_m = np.cumsum(m_s, 1)
  s_am = np.cumsum(a_s * m_s, 1)
  s_a2m = np.cumsum(a_s * a_s * m_s, 1)

  # With exactly k clipped channels, dE/dc = 0 gives:
  c_k = 2.0 * s_am / (mass / (6.0 * qmax * qmax) + 2.0 * s_m)
  lower = np.concatenate([a_s[:, 1:], np.zeros((n, 1))], 1)
  c_k = np.clip(c_k, lower, a_s)  # keep each candidate inside its segment
  e_k = (
      (c_k**2) * (mass / (12.0 * qmax * qmax))
      + s_a2m
      - 2.0 * c_k * s_am
      + (c_k**2) * s_m
  )
  c0 = a_s[:, :1]  # no-clip candidate
  e0 = (c0**2) * (mass / (12.0 * qmax * qmax))

  cand_c = np.concatenate([c0, c_k], 1)
  cand_e = np.concatenate([e0, e_k], 1)
  return cand_c[np.arange(n), np.argmin(cand_e, 1)]


def _weight_layout(
    op_name: _TFLOpName,
    tensor_content: np.ndarray,
    mu2: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
  """Views the FC weight as (rows, cols) and maps mu2 onto the columns.

  OSCAR supports FULLY_CONNECTED only.

  Args:
    op_name: The TFL op that consumes the weight.
    tensor_content: The weight tensor.
    mu2: Per-input-channel activation second moments, or None for uniform
      weighting (weight-only fallback: exact weight-MSE clipping).

  Returns:
    (matrix, col_mu2, channelwise_param_shape) where matrix rows are output
    channels, columns are the reduction (input-channel) axis, col_mu2 holds one
    mass per column, and channelwise_param_shape is the keepdims shape that
    channelwise scale/zero_point must have (matching
    common_quantize.init_tensor_min_max).
  """
  if op_name != _TFLOpName.FULLY_CONNECTED:
    raise ValueError(
        f"OSCAR supports FULLY_CONNECTED only, got: {op_name}"
    )
  w = np.asarray(tensor_content, np.float64)
  if w.ndim != 2:
    raise ValueError(
        f"OSCAR expects 2-D weights for {op_name}, got {w.shape}"
    )
  out_ch, in_ch = w.shape
  if mu2 is None:
    col_mu2 = np.ones(in_ch)
  else:
    col_mu2 = np.asarray(mu2, np.float64).ravel()
    if col_mu2.size != in_ch:
      raise ValueError(
          f"OSCAR: activation mu2 has {col_mu2.size} channels but"
          f" {op_name} weights of shape {w.shape} expect {in_ch}."
          " The calibration statistics do not match this tensor."
      )
  return w, col_mu2, (out_ch, 1)


def _check_blockwise_validity(
    op_name: qtyping.TFLOperationName,
    block_size: int,
    reduction_dim: int,
    tensor_shape: tuple[int, ...],
) -> None:
  """Validates blockwise quantization parameters."""
  if (
      op_name
      not in tfl_flatbuffer_utils.TFL_OP_TO_BLOCKWISE_WEIGHT_QUANTIZED_DIM
  ):
    raise ValueError(
        f"Blockwise granularity is not supported for op: {op_name}"
    )
  if reduction_dim % block_size != 0:
    raise ValueError(
        f"Block size {block_size} must divide the reduction dimension "
        f"{reduction_dim} of {op_name} weights with shape {tensor_shape}."
    )


def _channel_scale_objective(
    w: np.ndarray,
    s: np.ndarray,
    mu2: np.ndarray,
    block_size: int,
) -> float:
  """Computes total quantization proxy objective for scales s on weight w."""
  m = mu2 / (s * s)
  a = np.abs(w) * s
  d = w.shape[1]
  g = block_size if (block_size and d % block_size == 0) else d
  total = 0.0
  for gi in range(d // g):
    cols = slice(gi * g, (gi + 1) * g)
    mx = a[:, cols].max(1)
    total += float((mx * mx).sum()) * float(m[cols].sum())
  return total


def _compute_channel_scales(
    w: np.ndarray,
    mu2: np.ndarray,
    block_size: int = 0,
    num_iters: int = 3,
) -> tuple[np.ndarray | None, float]:
  """Computes optimal per-input-channel scales s for a fully connected op.

  Args:
    w: The 2-D weight matrix of shape (out_ch, in_ch).
    mu2: Per-input-channel activation second moments E[x_j^2] of shape (in_ch,).
    block_size: Block size along the reduction dimension, or 0 for
      tensor/channelwise.
    num_iters: Number of alternating fixed-point iterations to run.

  Returns:
    A tuple (s, gain):
      s: The optimal per-input-channel scale vector of shape (in_ch,), or None
        if scaling does not improve over identity (s = 1).
      gain: Ratio of identity objective to best objective achieved.
  """
  in_ch = mu2.size
  mu2 = _floor_positive(mu2)
  mu = np.sqrt(mu2)

  def normalized(v):
    v = v / np.exp(np.mean(np.log(v)))
    return np.clip(v, *_SCALE_CLAMP)

  a_base = (w * w).sum(0) + _EPS

  identity_loss = _channel_scale_objective(w, np.ones(in_ch), mu2, block_size)
  s = normalized(np.sqrt(mu / np.sqrt(a_base)))
  best = (_channel_scale_objective(w, s, mu2, block_size), s)

  # Alternating fixed-point optimization:
  for _ in range(num_iters):
    # 1. Estimate effective weight magnitude per channel (a_eff) based on
    #    maximum scaled weight within each block/group.
    a_eff = np.zeros(in_ch)
    d = w.shape[1]
    g = block_size if (block_size and d % block_size == 0) else d
    ws_abs = np.abs(w) * s
    rows = np.arange(w.shape[0])
    for gi in range(d // g):
      j_star = gi * g + np.argmax(ws_abs[:, gi * g : (gi + 1) * g], 1)
      np.add.at(a_eff, j_star, w[rows, j_star] ** 2)
    a_eff = np.maximum(a_eff, 0.25 * a_base)

    # 2. Update scales s to balance activation deviation (sqrt(mu)) against
    #    effective weight deviation (sqrt(a_eff)), damped via geometric mean.
    s_cand = normalized(np.sqrt(mu / np.sqrt(a_eff)))
    s = normalized(np.sqrt(s * s_cand))
    loss = _channel_scale_objective(w, s, mu2, block_size)
    if loss < best[0]:
      best = (loss, s)

  if best[0] >= identity_loss:
    return None, 1.0
  return best[1], identity_loss / max(best[0], _EPS)


def calibrate(
    tfl_op: qtyping.OperatorT,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: MutableMapping[str, np.ndarray],
    inputs_to_ignore: Sequence[int] | None = None,
    outputs_to_ignore: Sequence[int] | None = None,
    valid_range: tuple[float, float] = (-3e38, 3e38),
) -> dict[str, qtyping.QSV]:
  """Collects min/max plus per-channel activation second moments (mu2).

  Mirrors gptq.calibrate, storing only the Hessian diagonal:
  mu2 = mean(x*x) over all rows, per trailing-axis channel.

  Args:
    tfl_op: The tfl operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.
    valid_range: The valid range for tensor content for min/max collection.

  Returns:
    A dictionary mapping tensor names to the collected QSVs.
  """
  op_qsvs = {}
  min_val, max_val = valid_range

  tensor_ids = common_quantize.get_tensor_indices_requiring_calibration(
      tfl_op, graph_info, inputs_to_ignore, outputs_to_ignore
  )
  for tensor_idx in tensor_ids:
    result = common_quantize.collect_activation_tensor_statistics(
        tensor_idx,
        graph_info,
        tensor_content_map,
        valid_float_range_min=min_val,
        valid_float_range_max=max_val,
    )
    if result is None:
      continue

    tensor_name, tensor_content, tensor_qsvs = result

    # Per-channel second moment over the trailing (channel) axis. This is
    # the diagonal of GPTQ's Hessian (up to its factor 2) at O(d) memory.
    x = np.asarray(tensor_content, np.float64).reshape(
        [-1, tensor_content.shape[-1]]
    )
    tensor_qsvs["mu2"] = np.mean(x * x, axis=0)
    op_qsvs[tensor_name] = tensor_qsvs

  return op_qsvs


def get_clip_bounds(
    op_name: _TFLOpName,
    tensor_content: np.ndarray,
    mu2: np.ndarray | None,
    num_bits: int,
    granularity: qtyping.QuantGranularity,
) -> np.ndarray:
  """Computes activation-weighted optimal symmetric clip bounds.

  The returned array has exactly the shape that min/max QSVs have for this
  op and granularity (see common_quantize.init_tensor_min_max), so it can be
  fed as min=-bounds, max=+bounds into tensor_zp_scale_from_min_max.

  Args:
    op_name: The TFL op that consumes the weight.
    tensor_content: The weight tensor.
    mu2: Per-input-channel activation second moments, or None for uniform
      weighting (weight-only fallback: exact weight-MSE clipping).
    num_bits: Bit width of the target integer type.
    granularity: The quantization granularity.

  Returns:
    Clip bounds: scalar keepdims shape for TENSORWISE, keepdims channel
    shape for CHANNELWISE, (rows, cols // block_size) for BLOCKWISE.
  """
  qmax = 2 ** (num_bits - 1) - 1
  matrix, col_mu2, param_shape = _weight_layout(op_name, tensor_content, mu2)
  col_mu2 = _floor_positive(col_mu2)
  n, d = matrix.shape
  a = np.abs(matrix)

  if granularity == qtyping.QuantGranularity.TENSORWISE:
    # One shared bound: solve the same problem on a single flattened row.
    bound = _optimal_group_clip(a.reshape(1, n * d), np.tile(col_mu2, n), qmax)
    return bound.reshape((1,) * tensor_content.ndim)

  if granularity == qtyping.QuantGranularity.CHANNELWISE:
    bounds = _optimal_group_clip(a, col_mu2, qmax)
    return bounds.reshape(param_shape)

  if uniform_quantize_tensor.is_blockwise(granularity):
    block_size = uniform_quantize_tensor.extract_block_size_from_granularity(
        granularity
    )
    _check_blockwise_validity(
        op_name, block_size, reduction_dim=d, tensor_shape=tensor_content.shape
    )
    n_blocks = d // block_size
    bounds = np.empty((n, n_blocks))
    for block_idx in range(n_blocks):
      cols = slice(block_idx * block_size, (block_idx + 1) * block_size)
      bounds[:, block_idx] = _optimal_group_clip(
          a[:, cols], col_mu2[cols], qmax
      )
    return bounds

  raise ValueError(f"Unsupported granularity: {granularity}")


def _extract_mu2(tensor_qsv: dict[str, Any] | None) -> np.ndarray | None:
  """Extracts activation second moments (mu2) from tensor_qsv dictionary."""
  if not tensor_qsv:
    return None
  if "mu2" in tensor_qsv:
    return tensor_qsv["mu2"]
  if (
      "activation_tensor_qsv" in tensor_qsv
      and tensor_qsv["activation_tensor_qsv"]
  ):
    return tensor_qsv["activation_tensor_qsv"].get("mu2")
  return None


def _compute_oscar_weight_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    w: np.ndarray,
    mu2: np.ndarray | None,
) -> qtyping.UniformQuantParams:
  """Computes OSCAR scales, scales weight, and quantizes with optimal bounds."""
  _, in_ch = w.shape
  granularity = tensor_quant_config.granularity
  block_size = (
      uniform_quantize_tensor.extract_block_size_from_granularity(granularity)
      if uniform_quantize_tensor.is_blockwise(granularity)
      else 0
  )

  if mu2 is not None:
    mu2_arr = np.asarray(mu2, np.float64).ravel()
    if mu2_arr.size != in_ch:
      raise ValueError(
          f"OSCAR: activation mu2 has {mu2_arr.size} channels but"
          f" {op_info.op_name} weights of shape {w.shape} expect {in_ch}."
      )
    s, _ = _compute_channel_scales(w, mu2_arr, block_size)
    if s is None:
      s = np.ones(in_ch, dtype=np.float64)
  else:
    logging.warning(
        "OSCAR: no activation second moments (mu2) found for op %s"
        " (index %d); falling back to unscaled optimal clipping.",
        op_info.op_name,
        op_info.subgraph_op_index,
    )
    s = np.ones(in_ch, dtype=np.float64)

  # Scale weight: W' = W * s
  w_scaled = w * s
  mu2_scaled = (
      (np.asarray(mu2, np.float64).ravel() / (s * s))
      if mu2 is not None
      else None
  )

  # Compute optimal clip bounds on scaled weight.
  bounds = get_clip_bounds(
      op_info.op_name,
      w_scaled,
      mu2_scaled,
      tensor_quant_config.num_bits,
      granularity,
  )
  zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
      -bounds,
      bounds,
      tensor_quant_config.num_bits,
      tensor_quant_config.symmetric,
      granularity,
      None,
  )
  quantized_dim = common_utils.get_weight_quantized_dim(
      op_info, w_scaled, granularity
  )
  multiplier = (1.0 / s).astype(np.float32)

  base_quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
      block_size=block_size,
      custom_algorithm_param={"multiplier": multiplier},
  )
  is_blockwise = uniform_quantize_tensor.is_blockwise(granularity)
  quantized_vars = uniform_quantize_tensor.uniform_quantize(
      w_scaled, base_quant_params, is_blockwise
  )
  return dataclasses.replace(base_quant_params, quantized_data=quantized_vars)


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: np.ndarray | None = None,
    tensor_qsv: dict[str, Any] | None = None,
) -> qtyping.UniformQuantParams:
  """Returns the quantization parameters for a tensor.

  For weight tensors, replaces the max-abs bound with OSCAR's
  activation-weighted optimal clip bound and derives scale/zero_point and
  quantized data through the standard min/max machinery -- so the result is
  standard-representable by construction. Non-weight tensors fall back to
  naive_min_max_quantize.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor. None means the tensor is not
      a weight tensor.
    tensor_qsv: A dictionary containing the tensor QSVs. It may contain
      "mu2" or "activation_tensor_qsv" with the "mu2" statistic collected by
      calibrate(); if absent, OSCAR degrades gracefully to uniform
      weighting (exact weight-MSE clipping) with a warning.

  Returns:
    UniformQuantParams with quantized weights and custom_algorithm_param
    multiplier.

  Raises:
    ValueError: If asymmetric weight quantization is requested, or blockwise
      is requested for an unsupported op, or the mu2 statistic exists but
      does not match the weight shape.
  """
  # Fallback to naive_min_max_quantize for non-weight tensors.
  if tensor_content is None:
    res = naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )
    if not isinstance(res, qtyping.UniformQuantParams):
      raise TypeError(
          "Expected UniformQuantParams for uniform quantize, got"
          f" {type(res)}"
      )
    return res

  if not tensor_quant_config.symmetric:
    raise ValueError(
        "OSCAR supports symmetric weight quantization only, got asymmetric"
        f" config for op {op_info.op_name}."
    )

  if op_info.op_name != _TFLOpName.FULLY_CONNECTED:
    raise ValueError(
        f"OSCAR supports FULLY_CONNECTED only, got: {op_info.op_name}"
    )

  mu2 = _extract_mu2(tensor_qsv)
  w = np.asarray(tensor_content, np.float64)
  return _compute_oscar_weight_quant_params(
      op_info, tensor_quant_config, w, mu2
  )


def _get_or_compute_weight_quant_params(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    mu2: np.ndarray | None,
) -> qtyping.UniformQuantParams:
  """Gets cached or computes OSCAR weight quantization parameters."""
  weight_config = op_info.op_quant_config.weight_tensor_config
  if weight_config is None:
    raise ValueError(
        "Weight tensor quantization config is not provided for OSCAR"
        " quantization."
    )
  weight_tensor = graph_info.subgraph_tensors[op_info.op.inputs[1]]
  if quant_params := tensor_quant_params_cache.lookup(
      weight_tensor.buffer, weight_config
  ):
    assert isinstance(quant_params, qtyping.UniformQuantParams)
    return quant_params
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      weight_tensor, graph_info.buffers
  )
  quant_params = get_tensor_quant_params(
      op_info,
      weight_config,
      tensor_data,
      tensor_qsv={"mu2": mu2},
  )
  tensor_quant_params_cache.insert(
      weight_tensor.buffer,
      weight_config,
      quant_params,
  )
  return quant_params


def _materialize_input_tensor(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    quant_params: qtyping.UniformQuantParams,
) -> qtyping.TensorTransformationParams:
  """Materializes the activation input tensor for INSERT_MULTIPLY."""
  input_tensor = graph_info.subgraph_tensors[op_info.op.inputs[0]]
  op2input_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=[qtyping.QuantTransformation.INSERT_MULTIPLY],
  )
  return qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(input_tensor),
      consumers=[op2input_params],
  )


def _materialize_weight_tensor(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    quant_params: qtyping.UniformQuantParams,
) -> qtyping.TensorTransformationParams:
  """Materializes the weight tensor for QUANTIZE_TENSOR."""
  weight_tensor = graph_info.subgraph_tensors[op_info.op.inputs[1]]
  op2weight_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=[qtyping.QuantTransformation.QUANTIZE_TENSOR],
  )
  return qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(weight_tensor),
      consumers=[op2weight_params],
  )


def _materialize_bias_tensor(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
) -> qtyping.TensorTransformationParams | None:
  """Materializes the bias tensor if present."""
  if len(op_info.op.inputs) <= 2 or op_info.op.inputs[2] < 0:
    return None
  bias_tensor = graph_info.subgraph_tensors[op_info.op.inputs[2]]
  no_quant_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
  )
  return qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(bias_tensor),
      consumers=[no_quant_params],
  )


def _materialize_output_tensor(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
) -> qtyping.TensorTransformationParams:
  """Materializes the output tensor with NO_QUANTIZE transformation."""
  output_tensor = graph_info.subgraph_tensors[op_info.op.outputs[0]]
  no_quant_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
  )
  return qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(output_tensor),
      producer=no_quant_params,
  )


def materialize_fully_connected(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
    tensor_name_to_qsv: dict[str, Any] | None = None,
) -> list[qtyping.TensorTransformationParams]:
  """Materializes the fully_connected op for OSCAR.

  Inserts an elementwise multiplier transformation on the activation input
  tensor and quantizes the scaled weight tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_quant_params_cache: Cache of already computed quantization
      parameters.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.

  Returns:
    Quantization configuration for the tensors associated with the op.
  """
  if op_info.op_quant_config.weight_tensor_config is None:
    raise ValueError(
        "Weight tensor quantization config is not provided for OSCAR"
        " quantization."
    )

  if op_info.op_name != _TFLOpName.FULLY_CONNECTED:
    raise ValueError(
        f"OSCAR supports FULLY_CONNECTED only, got: {op_info.op_name}"
    )

  input_tensor = graph_info.subgraph_tensors[op_info.op.inputs[0]]
  input_tensor_name = tfl_flatbuffer_utils.get_tensor_name(input_tensor)
  mu2 = None
  if tensor_name_to_qsv and input_tensor_name in tensor_name_to_qsv:
    mu2 = tensor_name_to_qsv[input_tensor_name].get("mu2")

  quant_params = _get_or_compute_weight_quant_params(
      op_info, graph_info, tensor_quant_params_cache, mu2
  )

  # TODO(b/542923497): Support upstream scale folding to fold multiplier into
  # preceding operations instead of inserting runtime MUL ops. Please check out
  # cl/962751575 for a pointer on how to do that.
  op_tensor_params = [
      _materialize_input_tensor(op_info, graph_info, quant_params),
      _materialize_weight_tensor(op_info, graph_info, quant_params),
  ]
  if bias_params := _materialize_bias_tensor(op_info, graph_info):
    op_tensor_params.append(bias_params)
  op_tensor_params.append(_materialize_output_tensor(op_info, graph_info))

  return op_tensor_params
