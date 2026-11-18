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

"""Implements the OCTAV quantization."""

import dataclasses
from typing import Any, Optional
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils

ALGORITHM_KEY = "OCTAV"
_CHUNK_BYTES = 32 * 1024 * 1024


def _guess_clipping_with_octav(
    x: np.ndarray,
    bits: int,
    axis: int,
    max_iterations: int,
    exponent_divisor: float,
    early_stop: bool = True,
    mid_riser: bool = False,
) -> np.ndarray:
  """Returns a tensor of absolute clipping constants for a tensor using OCTAV.

  This method implements equation (6) from the OCTAV paper:
  https://arxiv.org/abs/2206.06501

  Args:
    x: Tensor data to return guesses for.
    bits: Number of bits used during quantization.
    axis: Axis to reduce the tensor along to get the guesses.
    max_iterations: Number of Newton-Raphson iterations to use.
    exponent_divisor: Divisor scaling the quantization noise weight 1 / (4 *
      exponent_divisor * half_range_steps^2) (which equals 4^-bits /
      exponent_divisor when mid_riser=True). In the paper, 3.0 is optimal for
      symmetric signed ints and 12.0 for asymmetric/unsigned ints.
    early_stop: If True, stop the iteration if the guess doesn't change.
    mid_riser: If True, symmetric mid-riser quantization without a zero level
      (e.g. int2_in_int4 {-3, -1, 1, 3}) has step clip / 2^(bits-1), so [0,
      clip] spans 2^(bits-1) step widths. If False (default), symmetric signed
      quantization with zero_point=0 is mid-tread with step clip / (2^(bits-1) -
      1), i.e. the positive half-range spans 2^(bits-1) - 1 step widths (for
      bits < 8, uniform_quantize keeps the extra -2^(bits-1) negative level,
      making the symmetric [-clip, clip] clipping model slightly approximate on
      the negative tail while still using step clip / (2^(bits-1) - 1)).

  Returns:
    A tensor of shape [num_channels] with clipping constant guesses.

  Raises:
    ValueError: If `bits` is less than 2 for mid-tread or less than 1 for
      mid-riser.
  """
  min_bits = 1 if mid_riser else 2
  if bits < min_bits:
    raise ValueError(
        f"OCTAV requires bits >= {min_bits} when mid_riser={mid_riser},"
        f" got bits={bits}."
    )
  if axis is not None:
    axis = (axis,) if isinstance(axis, int) else axis  # pyrefly: ignore[bad-assignment]
    shape_reduced = tuple(1 if k in axis else d for k, d in enumerate(x.shape))  # pyrefly: ignore[not-iterable]
    count = np.prod([x.shape[d] for d in axis])  # pyrefly: ignore[not-iterable]
  else:
    shape_reduced = (1,)
    count = x.size
  guess = np.ones(shape_reduced, dtype=np.float32)
  guess_mask = np.zeros(x.shape, dtype=np.bool)
  # With half-range step count M = half_range_steps and step Delta = clip / M,
  # uniform within-bin MSE is Delta^2 / 12 = clip^2 / (12 * M^2), giving noise
  # weight 1 / (4 * exponent_divisor * M^2) (= 1 / (12 * M^2) for symmetric
  # signed quantization with exponent_divisor=3.0: 1/12 for B=2, 1/588 for B=4,
  # 1/193548 for B=8, and 1/48 for 2-bit mid-riser M=2).
  half_range_steps = (1 << (bits - 1)) if mid_riser else (1 << (bits - 1)) - 1
  scale = np.asarray(
      1.0 / (4.0 * exponent_divisor * (half_range_steps**2)),
      dtype=np.float32,
  )
  for _ in range(max_iterations):
    old_guess = guess

    # Note: Computing `guess_mask = np.greater_equal(np.abs(x), old_guess)`
    # requires making a temporary copy of `x` for the result of `np.abs(x)`.
    # We therefore compute the mask _twice_, i.e. once for `x >= guess_mask`,
    # and once for `x <= -guess_mask`, which does not require creating a copy
    # of `x`.

    # Values larger than `guess`.
    guess_mask = np.greater_equal(x, old_guess, out=guess_mask)
    denominator = np.count_nonzero(guess_mask, axis=axis, keepdims=True).astype(
        np.float32
    )
    guess = np.sum(
        x,
        axis=axis,
        where=guess_mask,
        keepdims=True,
        dtype=guess.dtype,
    )

    # Values smaller than `-guess`.
    guess_mask = np.less_equal(x, -old_guess, out=guess_mask)
    denominator = np.add(
        denominator,
        np.count_nonzero(guess_mask, axis=axis, keepdims=True),
        out=denominator,
    )
    guess = np.subtract(
        guess,
        np.sum(
            x,
            axis=axis,
            where=guess_mask,
            keepdims=True,
        ),
        out=guess,
    )

    # Update the `guess` values.
    denominator = np.multiply(denominator, 1.0 - scale, out=denominator)
    denominator = np.add(denominator, scale * count, out=denominator)
    guess = np.divide(guess, denominator, out=guess)
    if early_stop and np.allclose(old_guess, guess):
      break

  return guess


def _to_odd_int4_codes(data: np.ndarray, scale: np.ndarray) -> np.ndarray:
  """Rounds `data` to the nearest of {-3, -1, 1, 3} * `scale` as int8."""
  s_2bit = np.float32(2.0) * np.maximum(
      scale.astype(np.float32, copy=False), np.float32(1e-9)
  )
  q = 2.0 * np.clip(np.rint(data / s_2bit - 0.5), -2.0, 1.0) + 1.0
  return q.astype(np.int8)


def _quantize_int2_in_int4(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
    is_blockwise_quant: bool = False,
) -> np.ndarray:
  """Quantizes tensor to 4-level 2-bit values [-3, -1, 1, 3] stored in INT4.

  Args:
    tensor_data: Floating-point tensor data to quantize.
    quantization_params: Symmetric 4-bit quantization parameters with `scale =
      clip / 4` (adjacent mid-riser levels are `2 * scale` apart).
    is_blockwise_quant: Whether `quantization_params` uses blockwise scales.

  Returns:
    Quantized `np.int8` array containing signed INT4 codes in `{-3, -1, 1, 3}`.
  """
  if is_blockwise_quant:
    quantization_params = (
        uniform_quantize_tensor.broadcast_scale_zp_for_blockwise(
            tensor_data, quantization_params
        )
    )
  quantization_params = uniform_quantize_tensor.fix_quantization_params_rank(
      tensor_data, quantization_params
  )
  scales = quantization_params.scale

  if tensor_data.ndim > 1 and tensor_data.nbytes > _CHUNK_BYTES:
    output_shape = tensor_data.shape
    flat_data = tensor_data.reshape([-1, output_shape[-1]])
    last_scale_dim = scales.shape[-1]
    flat_scale = np.broadcast_to(
        scales, (*output_shape[:-1], last_scale_dim)
    ).reshape([-1, last_scale_dim])
    ret = np.zeros(shape=flat_data.shape, dtype=np.int8)
    chunk_rows = max(
        1,
        _CHUNK_BYTES // (output_shape[-1] * tensor_data.dtype.itemsize),
    )
    for k in range(0, flat_data.shape[0], chunk_rows):
      end = min(k + chunk_rows, flat_data.shape[0])
      ret[k:end, :] = _to_odd_int4_codes(
          flat_data[k:end, :], flat_scale[k:end, :]
      )
    return np.reshape(ret, output_shape)

  return _to_odd_int4_codes(tensor_data, scales)


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Returns the quantization parameters for a tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor. Supports
      `algorithm_params={'int2_in_int4': True}` when `num_bits=4` and
      `symmetric=True` to quantize weights to the 4-level 2-bit mid-riser
      codebook `{-3, -1, 1, 3} * scale` stored in signed INT4 (for evaluating or
      executing 2-bit-quantized weights on INT4 kernels).
    tensor_content: The content of the tensor. When None, it means the tensor is
      not a weight tensor (e.g. static quantization) so we fallback to using
      naive_min_max_quantize.
    tensor_qsv: A dictionary containing the min/max of the tensor.

  Returns:
    The computed `UniformQuantParams` including quantized tensor data.

  Raises:
    ValueError: If unsigned or asymmetric quantization is requested, if
      `num_bits < 2`, if `int2_in_int4` is invalid, or if min/max values
      cannot be determined from `tensor_qsv` or `tensor_content`.
  """
  # Fallback to naive_min_max_quantize.py for non-weight tensors.
  if tensor_content is None:
    return naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )

  if tensor_quant_config.dtype == qtyping.TensorDataType.UINT:
    raise ValueError("Unsigned quantization is not supported by OCTAV.")

  if not tensor_quant_config.symmetric:
    raise ValueError(
        f"Unsupported symmetry: {tensor_quant_config.symmetric}. OCTAV"
        " supports symmetric quantization only for now."
    )

  if tensor_quant_config.num_bits < 2:
    raise ValueError(
        "OCTAV requires num_bits >= 2, got"
        f" num_bits={tensor_quant_config.num_bits}."
    )

  int2_in_int4 = tensor_quant_config.algorithm_params.get("int2_in_int4", False)
  if not isinstance(int2_in_int4, bool):
    raise ValueError(f"int2_in_int4 must be a boolean, got {int2_in_int4!r}.")
  if int2_in_int4 and tensor_quant_config.num_bits != 4:
    raise ValueError(
        "int2_in_int4 requires num_bits=4, got"
        f" num_bits={tensor_quant_config.num_bits}."
    )

  # Use tensor_content if QSV lacks min/max (e.g., including GPTQ case).
  if not tensor_qsv or "min" not in tensor_qsv:
    # We need min/max to calculate quantization parameters, which
    # should be collected during the calibration process. However,
    # weight-only and DRQ do not require calibration, thus it is
    # possible that this information is missing here. In that case we
    # collect min/max on the spot.
    tensor_min_max = common_quantize.init_tensor_min_max(
        tensor_content,
        op_info,
    )
  else:
    tensor_min_max = tensor_qsv

  if "min" not in tensor_min_max or "max" not in tensor_min_max:
    raise ValueError(
        "min and max must be provided to produce tensor quantization"
        " parameters. Check if the correct calibration results are passed into"
        " the ParamsGenerator."
    )

  quantized_dim = common_utils.get_weight_quantized_dim(
      op_info, tensor_content, tensor_quant_config.granularity
  )
  is_blockwise_quant = uniform_quantize_tensor.is_blockwise(
      tensor_quant_config.granularity
  )
  if is_blockwise_quant:
    reshaped_data, reduce_dims = (
        uniform_quantize_tensor.reshape_data_for_blockwise(
            tensor_content,
            op_info.op_name,
            tensor_quant_config.granularity,
        )
    )
  else:
    reshaped_data = tensor_content
    reduce_dims = common_utils.get_reduce_dims(
        quantized_dim, tensor_content.shape
    )

  octav_bits = 2 if int2_in_int4 else tensor_quant_config.num_bits
  clipping_constants = _guess_clipping_with_octav(
      reshaped_data,
      octav_bits,
      reduce_dims,  # pyrefly: ignore[bad-argument-type]
      max_iterations=10,
      exponent_divisor=3.0 if tensor_quant_config.symmetric else 12.0,
      mid_riser=int2_in_int4,
  )
  # We created a new dimension in order to reduce properly for blockwise
  # quantization, so we need to reshape the clipping constants back to the
  # min/max shape for the next step.
  if is_blockwise_quant:
    clipping_constants = clipping_constants.reshape(tensor_min_max["min"].shape)

  min_val = tensor_min_max["min"]
  max_val = tensor_min_max["max"]
  clip_val = clipping_constants
  if int2_in_int4:
    # Pre-scale by 7 / 4 in float64 before tensor_zp_scale_from_min_max divides
    # by qmax=7 so the resulting scale is bound / 4 (adjacent levels in
    # {-3, -1, 1, 3} are 2 * scale apart, spanning half-range 4 * scale) and
    # blockwise bfloat16/float16 rounding is applied once to bound / 4.
    if is_blockwise_quant:
      clip_val = np.minimum(clip_val, np.float32(65280.0 * 4.0))
    min_val = min_val.astype(np.float64) * (7.0 / 4.0)
    max_val = max_val.astype(np.float64) * (7.0 / 4.0)
    clip_val = clip_val.astype(np.float64) * (7.0 / 4.0)

  zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
      min_val,
      max_val,
      tensor_quant_config.num_bits,
      tensor_quant_config.symmetric,
      tensor_quant_config.granularity,
      clip_val,
  )
  if int2_in_int4 and not is_blockwise_quant:
    scale = scale.astype(np.float32)

  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
      block_size=uniform_quantize_tensor.extract_block_size_from_granularity(
          tensor_quant_config.granularity
      ),
  )

  if int2_in_int4:
    quantized_vars = _quantize_int2_in_int4(
        tensor_content,
        quant_params,
        is_blockwise_quant=is_blockwise_quant,
    )
  elif op_info is not None and getattr(op_info, "stream_weights", False):
    return quant_params
  else:
    quantized_vars = uniform_quantize_tensor.uniform_quantize(
        tensor_content,
        quant_params,
        is_blockwise_quant=is_blockwise_quant,
    )

  return dataclasses.replace(quant_params, quantized_data=quantized_vars)
