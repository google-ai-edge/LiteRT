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

"""Uniform quantize in tensor level."""

import dataclasses
from typing import Any, Optional, Sequence
import ml_dtypes
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.utils import tfl_flatbuffer_utils


@dataclasses.dataclass(frozen=True)
class IntType:
  num_bits: int
  signed: bool


def is_blockwise(granularity: qtyping.QuantGranularity) -> bool:
  """Checks if the quantization granularity is blockwise."""
  return "BLOCKWISE" in str(granularity)


def get_quantized_range(qtype: IntType) -> tuple[float, float]:
  """Calculates range of the quantized type."""
  if qtype.signed:
    qmax = 2 ** (qtype.num_bits - 1) - 1
    qmin = -(2 ** (qtype.num_bits - 1))
  else:
    qmax = (2**qtype.num_bits) - 1
    qmin = 0
  return float(qmin), float(qmax)


def extract_block_size_from_granularity(
    granularity: qtyping.QuantGranularity,
) -> int:
  """Get the block size for blockwise quantization."""
  if granularity == qtyping.QuantGranularity.BLOCKWISE_32:
    return 32
  elif granularity == qtyping.QuantGranularity.BLOCKWISE_64:
    return 64
  elif granularity == qtyping.QuantGranularity.BLOCKWISE_128:
    return 128
  elif granularity == qtyping.QuantGranularity.BLOCKWISE_256:
    return 256
  else:
    return 0


def _round_and_clip_inplace(
    tensor: np.ndarray, qtype: IntType, narrow: bool
) -> np.ndarray:
  """Round and clip the tensor to the given type in-place, but don't cast it."""
  qmin, qmax = get_quantized_range(qtype)
  if narrow:
    if qtype.signed:
      return np.clip(
          np.rint(tensor, out=tensor if not np.isscalar(tensor) else None),
          qmin + 1,
          qmax,
          out=tensor if not np.isscalar(tensor) else None,
      )
    else:
      raise ValueError("Unsigned data type should not have narrow range.")
  else:
    return np.clip(
        np.rint(tensor, out=tensor if not np.isscalar(tensor) else None),
        qmin,
        qmax,
        out=tensor if not np.isscalar(tensor) else None,
    )


def _get_numpy_dtype(qtype: IntType) -> type[Any]:
  if qtype.num_bits <= 8:
    return np.int8 if qtype.signed else np.uint8
  elif qtype.num_bits <= 16:
    return np.int16 if qtype.signed else np.uint16
  elif qtype.num_bits <= 32:
    return np.int32 if qtype.signed else np.uint32
  else:
    return np.int64 if qtype.signed else np.uint64


def assign_quantized_type(tensor: np.ndarray, qtype: IntType) -> np.ndarray:
  """Cast the tensor to the quantized type."""
  if qtype.num_bits <= 8:
    qtype = np.int8 if qtype.signed else np.uint8  # pyrefly: ignore[bad-assignment]
  elif qtype.num_bits <= 16:
    qtype = np.int16 if qtype.signed else np.uint16  # pyrefly: ignore[bad-assignment]
  elif qtype.num_bits <= 32:
    qtype = np.int32 if qtype.signed else np.uint32  # pyrefly: ignore[bad-assignment]
  else:
    qtype = np.int64 if qtype.signed else np.uint64  # pyrefly: ignore[bad-assignment]
  return tensor.astype(qtype, copy=False)  # pyrefly: ignore[no-matching-overload]


def fix_quantization_params_rank(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
) -> qtyping.UniformQuantParams:
  """Fix the rank of quantization parameters (scale/zero points).

  Scale and zero points need to be the same rank as tensor_data to avoid
  ambiguous broadcasting.

  Args:
    tensor_data: The tensor to be quantized.
    quantization_params: The quantization parameters.

  Returns:
    quantization_params with broadcasted scales and zero_points.
  """
  scales, zero_points = (
      quantization_params.scale,
      quantization_params.zero_point,
  )
  if tensor_data.ndim == scales.ndim:
    return quantization_params

  if tensor_data.ndim == 0:
    # Scalar tensor requires scalar scale and zero_point.
    if scales.size != 1 or zero_points.size != 1:
      raise ValueError(
          "Scale and zero_point must contain single element for scalar tensor."
          f" Got scale: {scales}, zero_point: {zero_points}"
      )
    scales = np.array(scales.item())
    zero_points = np.array(zero_points.item())
  else:
    dims = [
        dim
        for dim in range(tensor_data.ndim)
        if dim != quantization_params.quantized_dimension
    ]
    scales = np.expand_dims(scales, axis=dims)  # pyrefly: ignore[no-matching-overload]
    zero_points = np.expand_dims(zero_points, axis=dims)  # pyrefly: ignore[no-matching-overload]

  return qtyping.UniformQuantParams(
      scale=scales,
      zero_point=zero_points,
      num_bits=quantization_params.num_bits,
      symmetric=quantization_params.symmetric,
      quantized_dimension=quantization_params.quantized_dimension,
      quantized_data=quantization_params.quantized_data,
      block_size=quantization_params.block_size,
      signed=quantization_params.signed,
  )


def _get_tensor_shape_for_blockwise(
    tensor_shape: Sequence[int], quantized_dim: int, block_size: int
) -> list[int]:
  """Get the tensor shape for blockwise quantization.

  This function splits the quantize dimension of the tensor into blocks and the
  dim/blocks. Hence, min/max of the tensor can be calculated for each block
  using existing functions.

  Args:
    tensor_shape: The original shape of the tensor.
    quantized_dim: The dimension to be quantized blockwise.
    block_size: The size of the block.

  Returns:
    The new tensor shape for calculating scale and zp for blockwise
    quantization.
  """
  new_shape = []
  for index, val in enumerate(tensor_shape):
    if index == quantized_dim:
      if val % block_size != 0:
        raise ValueError(
            f"Quantized dimension {val} in tensor shape {tensor_shape} is not"
            f" divisible by block size {block_size}."
        )
      new_shape.append(int(val / block_size))
      new_shape.append(block_size)
    else:
      new_shape.append(val)
  return new_shape


def reshape_data_for_blockwise(
    tensor_data: np.ndarray,
    op_name: qtyping.TFLOperationName,
    granularity: qtyping.QuantGranularity,
) -> tuple[np.ndarray, int]:
  """Reshapes data for blockwise quantization.

  Args:
    tensor_data: The original tensor data.
    op_name: The name of the TFL op.
    granularity: The quantization granularity for the tensor.

  Returns:
    A tuple containing the reshaped tensor data and the new reduce dimension.
  """
  quantized_dim = tfl_flatbuffer_utils.TFL_OP_TO_BLOCKWISE_WEIGHT_QUANTIZED_DIM[
      op_name
  ]
  block_size = extract_block_size_from_granularity(granularity)
  new_shape = _get_tensor_shape_for_blockwise(
      tensor_data.shape, quantized_dim, block_size
  )
  return tensor_data.reshape(new_shape), quantized_dim + 1


def _broadcast_scale_zp_for_blockwise(
    tensor_content: np.ndarray,
    quant_params: qtyping.UniformQuantParams,
) -> qtyping.UniformQuantParams:
  """Broadcasts scale and zp for blockwise quantization.

  Args:
    tensor_content: The original tensor data.
    quant_params: The quantization parameters.
      `quant_params.quantized_dimension` must be specified.
      `quant_params.block_size` must be specified and positive.

  Returns:
    The updated quantization parameters with broadcasted scale and zp for
    correct constant quantization.
  """
  if quant_params.quantized_dimension is None:
    raise ValueError("Quantized dimension must be specified.")
  if quant_params.block_size is None or quant_params.block_size <= 0:
    raise ValueError("Block size must be specified and positive.")
  quantized_dim = quant_params.quantized_dimension
  expanded_tensor_shape = _get_tensor_shape_for_blockwise(
      tensor_content.shape, quantized_dim, quant_params.block_size
  )
  expanded_scale = np.reshape(
      np.broadcast_to(
          np.expand_dims(quant_params.scale, quantized_dim + 1),
          expanded_tensor_shape,
      ),
      tensor_content.shape,
  )
  if quant_params.zero_point is None or quant_params.zero_point.size == 0:
    expanded_zp = np.zeros_like(tensor_content, dtype=np.int32)
  else:
    expanded_zp = np.reshape(
        np.broadcast_to(
            np.expand_dims(quant_params.zero_point, quantized_dim + 1),
            expanded_tensor_shape,
        ),
        tensor_content.shape,
    )
  return qtyping.UniformQuantParams(
      scale=expanded_scale,
      zero_point=expanded_zp,
      num_bits=quant_params.num_bits,
      symmetric=quant_params.symmetric,
      quantized_dimension=quantized_dim,
      block_size=quant_params.block_size,
      signed=quant_params.signed,
  )


def uniform_quantize(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
    is_blockwise_quant: bool = False,
):
  """Uniform quantize a tensor.

  Args:
    tensor_data: The tensor to be quantized.
    quantization_params: The quantization parameters.
    is_blockwise_quant: Whether the tensor is blockwise quantized.

  Returns:
    The quantized tensor.
  """
  # The reshaping for blockwise quantization is unique hence we do this here
  # to avoid unexpected broadcast behavior downstream.
  if is_blockwise_quant:
    quantization_params = _broadcast_scale_zp_for_blockwise(
        tensor_data, quantization_params
    )

  # quant params in flatbuffer is flattened, expand the rank to be the same
  # as the tensor rank to avoid ambiguous broadcasting.
  quantization_params = fix_quantization_params_rank(
      tensor_data, quantization_params
  )
  _is_valid_quantization_params(tensor_data, quantization_params)
  scales, zero_points = (
      quantization_params.scale,
      quantization_params.zero_point,
  )
  qtype = IntType(
      quantization_params.num_bits,
      signed=quantization_params.signed,
  )
  # For quantization with more than 8 bits, symmetric narrow-range quantization
  # is required due to assumptions made by legacy TFLite kernels. However, this
  # method is not ideal for low-bit quantization (e.g., 2-bit quantization,
  # which only has 4 bins), as it wastes a bin and there are no kernel
  # requirements for a narrow range when < 8 bits because the data is unpacked
  # to int8 before being used in the kernel.
  narrow_range = (
      quantization_params.symmetric and quantization_params.num_bits >= 8
  )
  if not np.issubdtype(zero_points.dtype, np.integer):
    raise ValueError(f"zero_points need to be integer. Got {zero_points.dtype}")
  qmin, qmax = get_quantized_range(qtype)
  if np.any(zero_points < qmin) or np.any(zero_points > qmax):
    raise ValueError(
        f"zero_points values must be within [{qmin}, {qmax}]."
        f" But got zero_points: {zero_points}"
    )

  # For large tensors, quantize the data in chunks of the leading dimension to
  # avoid creating large intermediate values.
  if (
      tensor_data.ndim > 1
      and tensor_data.nbytes > 32 * 1024 * 1024
      and tensor_data.dtype.itemsize * 8 != qtype.num_bits
  ):
    # Reshape the `tensor_data`, `scales`, and `zero_points` to just two
    # dimensions.
    output_shape = tensor_data.shape
    tensor_data = tensor_data.reshape([-1, output_shape[-1]])
    scales = np.broadcast_to(scales, tensor_data.shape)
    zero_points = np.broadcast_to(zero_points, tensor_data.shape)
    ret = np.zeros(shape=tensor_data.shape, dtype=_get_numpy_dtype(qtype))

    # Use a chunk size corresponding to 32 MiB of input data.
    chunk_size = (32 * 1024 * 1024) // (
        output_shape[-1] * tensor_data.dtype.itemsize
    )

    # Loop over `tensor_data` in chunks.
    for k in range(0, tensor_data.size, chunk_size):
      end = min(k + chunk_size, tensor_data.size)

      # Quantize the chunk.
      q = np.divide(tensor_data[k:end, :], scales[k:end, :])
      q = np.add(q, zero_points[k:end, :], out=q)
      q = _round_and_clip_inplace(q, qtype, narrow_range)
      ret[k:end, :] = assign_quantized_type(q, qtype)

    # Reshape the flat output.
    ret = np.reshape(ret, output_shape)

  else:
    ret = np.divide(tensor_data, scales)
    ret = np.add(ret, zero_points, out=ret if not np.isscalar(ret) else None)
    ret = _round_and_clip_inplace(ret, qtype, narrow_range)
    ret = assign_quantized_type(ret, qtype)

  return ret


def uniform_dequantize(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
):
  """Uniform dequantize a tensor.

  Args:
    tensor_data: The tensor to be dequantized.
    quantization_params: The quantization parameters.

  Returns:
    The dequantized tensor.
  """
  if quantization_params.block_size != 0:
    # b/443830202: The quantized dimension is currently increased by 1 because
    # AEQ expects 1 and XNNPack expects 0.
    # We only increment it if it is XNNPack style (0) to avoid double increment
    # if the caller already passed the actual AEQ style dimension (1).
    if quantization_params.quantized_dimension == 0:
      quantization_params = dataclasses.replace(
          quantization_params,
          quantized_dimension=quantization_params.quantized_dimension + 1,
      )
    scale_shape = list(tensor_data.shape)
    scale_shape[quantization_params.quantized_dimension] = (  # pyrefly: ignore[unsupported-operation]
        scale_shape[quantization_params.quantized_dimension]  # pyrefly: ignore[bad-index]
        // quantization_params.block_size
    )
    quantization_params = dataclasses.replace(
        quantization_params,
        scale=quantization_params.scale.reshape(scale_shape),
    )
    quantization_params = _broadcast_scale_zp_for_blockwise(
        tensor_data, quantization_params
    )

  # quant params in flatbuffer is flattened, expand the rank to be the same
  # as the tensor rank to avoid ambiguous broadcasting.
  quantization_params = fix_quantization_params_rank(
      tensor_data, quantization_params
  )
  _is_valid_quantization_params(tensor_data, quantization_params)
  return np.multiply(
      tensor_data - quantization_params.zero_point, quantization_params.scale
  )


def symmetric_quantize_bias_tensor(
    bias_content: np.ndarray,
    input_tensor_quant_params: qtyping.UniformQuantParams,
    weight_tensor_quant_params: qtyping.UniformQuantParams,
    # TODO: b/477623376 - Have a debugging mode to check bias.
    check_error: bool = False,
) -> qtyping.UniformQuantParams:
  """Quantize bias tensor (symmetrically, i.e., zero_point = 0).

  We quantize bias to a much higher bit width, e.g., int32 for int8 weights. We
  can afford the cost of being symmetric all the time. This configuration fits
  TFL kernel designs.

  Args:
    bias_content: The bias content.
    input_tensor_quant_params: The quantization parameters of input tensor.
    weight_tensor_quant_params: The quantization parameters of weight tensor.
    check_error: Whether to check if the quantization error (the difference
      between the original and dequantized bias) is larger than the quantization
      scale. This check is important because bias quantization parameters are
      fixed (bias_scale = input_scale * weight_scale), which can lead to large
      quantization errors. Raising an error when the quantization error is
      larger than the scale helps to identify unexpected numerical issues.

  Returns:
    The quantized bias tensor.
  """
  input_tensor_scale = input_tensor_quant_params.scale
  weight_tensor_scale = weight_tensor_quant_params.scale
  # Bias is always 1D, make sure the scale has 1D shape as well.
  effective_output_scale = np.squeeze(input_tensor_scale * weight_tensor_scale)
  # Squeeze can produce scalar, but we want 1D tensor.
  if not effective_output_scale.shape:
    effective_output_scale = np.expand_dims(effective_output_scale, axis=0)

  # symmetric
  bias_zp = np.zeros_like(effective_output_scale, dtype=np.int32)
  # Fixed to 32 bits since most of the accelerators use int32 accumulator.
  bias_number_bits = 32
  symmetric = True
  quantized_dimension = None if len(effective_output_scale) == 1 else 0
  bias_quant_params = qtyping.UniformQuantParams(
      scale=effective_output_scale,
      zero_point=bias_zp,
      num_bits=bias_number_bits,
      symmetric=symmetric,
      quantized_dimension=quantized_dimension,
  )

  quantized_vars = uniform_quantize(bias_content, bias_quant_params)
  if check_error:
    dequantized_bias = uniform_dequantize(quantized_vars, bias_quant_params)
    max_quant_error = np.max(np.abs(dequantized_bias - bias_content))
    error_tolerance = np.maximum(1e-6, np.max(effective_output_scale))
    if max_quant_error > error_tolerance:
      raise ValueError(
          "Quantization error is too large for bias tensor quantization. Max"
          f" quantization error is {max_quant_error}, which exceed"
          f" the threshold {error_tolerance}"
      )

  # Save the int32 quantized bias as int64 if the input tensor is quantized to
  # 16 bits. This is to assume the matmul is using int64 accumulator (safe from
  # overflow). For accelerators with int32 accumulator, it is safe to cast int64
  # back to int32.
  if input_tensor_quant_params.num_bits == 16:
    quantized_vars = quantized_vars.astype(np.int64)
    bias_number_bits = 64

  # UniformQuantParams is frozen dataclass, need to recreate.
  return qtyping.UniformQuantParams(
      scale=effective_output_scale,
      zero_point=bias_zp,
      num_bits=bias_number_bits,
      quantized_dimension=quantized_dimension,
      symmetric=symmetric,
      quantized_data=quantized_vars,
  )


def tensor_zp_scale_from_min_max(
    min_value,
    max_value,
    num_bits: int,
    symmetric: bool,
    granularity: qtyping.QuantGranularity,
    clipping_values: Optional[np.ndarray] = None,
    signed: bool = True,
):
  """Get zero point and scale from min and max value.

  Args:
    min_value: The minimum value of the tensor (channelwise and blockwise
      supported).
    max_value: The maximum value of the tensor (channelwise and blockwise
      supported).
    num_bits: The number of bits of the tensor.
    symmetric: Whether the tensor is symmetric.
    granularity: The granularity of the tensor.
    clipping_values: Absolute clipping values to apply to the tensor. This will
      clip the tensors to the range [-clipping_values, clipping_values]. This
      should be the same shape as min_value and max_value. If None, no clipping
      will be applied.
    signed: Whether the tensor is signed.

  Returns:
    The zero point and scale of the tensor.
  """

  qtype = IntType(
      num_bits,
      signed=signed,
  )
  qmin, qmax = get_quantized_range(qtype)
  min_bound = 1e-9  # Avoid zero scale.
  pos_clipping_values = None if clipping_values is None else clipping_values
  neg_clipping_values = None if clipping_values is None else -clipping_values

  if is_blockwise(granularity):
    # Blockwise quantization uses float16 scale,
    # with 7 bit mantissa, so the maximum scale value is 65280 and maximum
    # representable range is [-65280 * (2 ** num_bits),
    # 65280 * (2 ** num_bits - 1)].
    # Note that we have one extra value on the negative side.
    float16_max = np.broadcast_to(
        np.array(65280) * (2**num_bits - 1), max_value.shape
    )
    float16_min = np.broadcast_to(
        np.array(-65280) * (2**num_bits), min_value.shape
    )
    pos_clipping_values = (
        float16_max
        if pos_clipping_values is None
        else np.minimum(pos_clipping_values, float16_max)
    )
    neg_clipping_values = (
        float16_min
        if neg_clipping_values is None
        else np.maximum(neg_clipping_values, float16_min)
    )

  if symmetric:
    bound = np.maximum(np.abs(min_value), np.abs(max_value))
    bound = np.maximum(bound, min_bound)
    if clipping_values is not None:
      bound = np.clip(bound, neg_clipping_values, pos_clipping_values)
    if not qtype.signed:
      half_q = (qmax - 1) / 2
      scale = bound / half_q
      zp = np.ones_like(scale) * (half_q + 1)
    else:
      scale = bound / qmax
      zp = np.zeros_like(scale, dtype=np.int32)
  else:
    # Include 0 to the range to support zero-padding.
    # See: https://arxiv.org/pdf/1712.05877.pdf
    # This ensures bound_min <= 0 <= bound_max.
    bound_max = np.maximum(max_value, np.zeros_like(max_value))
    bound_min = np.minimum(min_value, np.zeros_like(min_value))
    bound = np.maximum(bound_max - bound_min, min_bound)
    if clipping_values is not None:
      bound = np.clip(bound, -clipping_values, clipping_values)
    scale = bound / (qmax - qmin)
    zp = qmin - bound_min / scale
    zp = np.rint(zp)

  if is_blockwise(granularity):
    # Round the scale values to 7 bit mantissa.
    scale = (
        scale.astype(ml_dtypes.bfloat16).astype(np.float16).astype(np.float32)
    )

  # It's safe to cast zp to qtype without clipping because we can infer
  # qmin <= zp <= qmax from bound_min <= 0 <= bound_max.
  zp = assign_quantized_type(zp, qtype)
  return zp, scale


def _is_valid_quantization_params(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
) -> None:
  """Checks if the quantization parameters are valid.

  A valid quantization params requires:
    1. scale and zero point either have the same shape or the zero point is a
    scalar.
    2. scale and zero point have the same rank as the tensor content (avoid
    ambiguous broadcasting).

  Args:
    tensor_data: The tensor to be quantized.
    quantization_params: The quantization parameters.

  Returns:
    True if the quantization parameters are valid.
  """
  if (
      quantization_params.scale.shape != quantization_params.zero_point.shape
      and quantization_params.zero_point.size != 1
  ):
    raise ValueError(
        "scale and zero_point must have the same shape or zero_point must have"
        f" only one element. Got {quantization_params.scale.shape} and"
        f" {quantization_params.zero_point.shape}"
    )

  tensor_rank = tensor_data.ndim
  scale_rank = quantization_params.scale.ndim
  zero_point_rank = quantization_params.zero_point.ndim
  if tensor_rank != scale_rank or (tensor_rank != zero_point_rank):
    raise ValueError(
        f"Ranks of scales ({scale_rank}) and zps"
        f" ({zero_point_rank}) must be the same as the tensor rank"
        f" ({tensor_rank})."
    )
  if (
      quantization_params.block_size != 0
      and tensor_data.shape[quantization_params.quantized_dimension]  # pyrefly: ignore[bad-index]
      % quantization_params.block_size
      != 0
  ):
    raise ValueError(
        # pyrefly: ignore[bad-index]
        "Tensor dimension must be divisible by block size. Got dimension:"
        f" {tensor_data.shape[quantization_params.quantized_dimension]} and"
        f" block size: {quantization_params.block_size}"
    )
