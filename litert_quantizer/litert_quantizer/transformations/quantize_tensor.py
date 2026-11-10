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

"""quantize a given tensor."""

from collections.abc import Sequence
import logging
from typing import Optional

import ml_dtypes
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.transformations import transformation_utils


# TODO: b/335014051 - Support distinguishing INT, FLOAT & UINT, BFLOAT.
def quant_params_to_tflite_type(
    bitwidth: int,
    signed: bool = True,
) -> Optional[qtyping.TensorType]:
  """Returns the TFLite dtype for the given bit width.

  Args:
    bitwidth: Bit width from UniformQuantParams.
    signed: Whether the tensor is signed.

  Returns:
    The corresponding TFLite tensor type.
  """
  match bitwidth:
    case 2:
      return qtyping.TensorType.INT2
    case 4:
      return qtyping.TensorType.INT4
    case bits if 1 < bits <= 8:
      return qtyping.TensorType.INT8 if signed else qtyping.TensorType.UINT8
    case bits if 8 < bits <= 16:
      return qtyping.TensorType.INT16
    case bits if 16 < bits <= 32:
      return qtyping.TensorType.INT32
    case bits if 32 < bits <= 64:
      return qtyping.TensorType.INT64
    case _:
      raise ValueError(f"Unsupported bitwidth {bitwidth}.I")


def nonlinear_quant_params_to_tflite_type(
    bitwidth: int,
) -> Optional[qtyping.TensorType]:
  """Returns the TFLite dtype for the given bit width.

  Args:
    bitwidth: bitwidth from NonLinearQuantParams

  Returns:
    the corresponding tflite tensortype
  """
  if bitwidth == 16:
    return qtyping.TensorType.FLOAT16
  elif bitwidth == 32:
    return qtyping.TensorType.FLOAT32
  else:
    raise ValueError(f"Unsupported nonlinear params: {bitwidth}")


def _perform_channelwise_quantization(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.QuantizationParametersT:
  """Perform channelwise quantization and fill the quantization parameters.

  Args:
    transformation_input: Input structure that contains all information needed
      for the transformation.

  Returns:
    The quantization parameters.
  """
  assert isinstance(
      transformation_input.quant_params, qtyping.UniformQuantParams
  )
  flatbuffer_quantization = qtyping.QuantizationParametersT()
  flatbuffer_quantization.scale = np.ravel(
      transformation_input.quant_params.scale
  ).astype(np.float32, copy=False)
  if transformation_input.quant_params.zero_point is not None:
    flatbuffer_quantization.zeroPoint = np.ravel(
        transformation_input.quant_params.zero_point
    ).astype(np.int64, copy=False)
  if transformation_input.quant_params.quantized_dimension is not None:
    flatbuffer_quantization.quantizedDimension = (
        transformation_input.quant_params.quantized_dimension
    )

  return flatbuffer_quantization


def _perform_blockwise_quantization(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.QuantizationParametersT:
  """Perform blockwise quantization and fill the quantization parameters.

  Args:
    transformation_input: Input structure that contains all information needed
      for the transformation.

  Returns:
    The quantization parameters.
  """
  assert isinstance(
      transformation_input.quant_params, qtyping.UniformQuantParams
  )
  flatbuffer_quantization = qtyping.QuantizationParametersT()
  flatbuffer_quantization.detailsType = (
      qtyping.QuantizationDetails.BlockwiseQuantization
  )
  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  blockwise_details = qtyping.BlockwiseQuantizationT()
  # Downcast and round the scale to fp16 with 7 bit mantissa.
  scale_tensor_id = transformation_utils.add_new_constant_tensor(
      tensor.name + b"_scales",
      transformation_input.quant_params.scale.astype(ml_dtypes.bfloat16).astype(
          np.float16
      ),
      qtyping.TensorType.FLOAT16,
      transformation_input.subgraph,
      transformation_input.model,
  )
  blockwise_details.scales = scale_tensor_id
  # Blockwise quantization does not support zero point yet, so this points to
  # a -1 buffer index.
  # TODO: b/404909258 - Add optional zero point to blockwise quantization.
  blockwise_details.zeroPoints = -1
  blockwise_details.blockSize = transformation_input.quant_params.block_size
  flatbuffer_quantization.details = blockwise_details
  # TODO: b/443830202 - Hardcoding to 0 for now.
  flatbuffer_quantization.quantizedDimension = 0
  return flatbuffer_quantization


def _get_multi_axis_block_shape(
    tensor_shape: Sequence[int],
    quantized_dimensions: Sequence[int],
    block_size: int,
) -> list[int]:
  """Returns `block_shape`: how many elements share one scale per dimension.

  Each quantized dimension gets 1 in the block shape, and each non-quantized
  dimension gets its original size. For blockwise quantization, the last
  dimension gets `block_size` instead of its original size.

  E.g. for an MoE weight of shape [E, N, K] with `quantized_dimensions` (0, 1),
  this is [1, 1, K] for channelwise and [1, 1, 32] for blocks of 32.

  Args:
    tensor_shape: The shape of the quantized tensor.
    quantized_dimensions: The dimensions with one scale per index.
    block_size: The block size along the last dimension, or 0 if not blockwise.

  Returns:
    The block shape, with one entry per tensor dimension.
  """
  last_dim = len(tensor_shape) - 1
  block_shape = []
  for dim, dim_size in enumerate(tensor_shape):
    if dim in quantized_dimensions:
      block_shape.append(1)
    elif block_size > 0 and dim == last_dim:
      block_shape.append(block_size)
    else:
      block_shape.append(dim_size)
  return block_shape


def _perform_multi_axis_quantization(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.QuantizationParametersT:
  """Perform multi-axis quantization and fill the quantization parameters.

  Multi-axis quantization is stored as `BlockwiseQuantization` with a
  `block_shape` that gives the block extent along every dimension.

  Args:
    transformation_input: Input structure that contains all information needed
      for the transformation.

  Returns:
    The quantization parameters.
  """
  quant_params = transformation_input.quant_params
  assert isinstance(quant_params, qtyping.UniformQuantParams)
  assert quant_params.quantized_dimensions is not None
  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  tensor_shape = [int(dim) for dim in tensor.shape]
  block_shape = _get_multi_axis_block_shape(
      tensor_shape, quant_params.quantized_dimensions, quant_params.block_size
  )
  scale = quant_params.scale.astype(np.float32)
  expected_scale_shape = tuple(
      (dim + block - 1) // block
      for dim, block in zip(tensor_shape, block_shape)
  )
  if scale.shape != expected_scale_shape:
    raise ValueError(
        f"Scale shape {scale.shape} of tensor {tensor.name.decode()} does not"
        f" match the expected shape {expected_scale_shape} for block_shape"
        f" {block_shape}."
    )

  flatbuffer_quantization = qtyping.QuantizationParametersT()
  flatbuffer_quantization.detailsType = (
      qtyping.QuantizationDetails.BlockwiseQuantization
  )
  blockwise_details = qtyping.BlockwiseQuantizationT()
  blockwise_details.scales = transformation_utils.add_new_constant_tensor(
      tensor.name + b"_scales",
      scale,
      qtyping.TensorType.FLOAT32,
      transformation_input.subgraph,
      transformation_input.model,
  )
  # Multi-axis quantization only supports symmetric quantization where zero
  # points are 0, so this points to -1 (no separate zero-point tensor).
  blockwise_details.zeroPoints = -1
  blockwise_details.blockShape = block_shape
  # Also set `block_size` for readers that ignore `block_shape`. It can only
  # express blocks that run along the last dimension.
  blockwise_details.blockSize = (
      block_shape[-1] if all(block == 1 for block in block_shape[:-1]) else 0
  )
  flatbuffer_quantization.details = blockwise_details
  return flatbuffer_quantization


def quantize_tensor(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Quantize the tensor at the tensor_id in the given subgraph.

  Args:
    transformation_input: Input structure that contains all information needed
      for the transformation.

  Returns:
    TransformationInfo:
      op_id: The producer index for tensor.
      num_ops_added: The total number of ops inserted by this operation, which
        is 0.
  """
  tensor: qtyping.TensorT = transformation_input.subgraph.tensors[
      transformation_input.tensor_id
  ]
  buffer_id = tensor.buffer
  # TODO: b/336385820 - Suppport quantize buffer directly when quantized_data
  # is not provided.
  if (
      buffer_id
      and (quant_params := transformation_input.quant_params).quantized_data
      is not None
  ):
    if (
        origin := transformation_input.buffer_origin.get(buffer_id)
    ) and origin is quant_params:
      logging.debug(
          "Quantized data for tensor %s (%s bytes) has already been packed to"
          " buffer %s, skipping packing",
          tensor.name.decode(),
          quant_params.quantized_data.size,  # pyrefly: ignore[missing-attribute]
          buffer_id,
      )
    else:
      if origin is not None:
        logging.warning(
            "Quantized data for tensor %s is overriding other previously"
            " quantized data in buffer %s.",
            tensor.name.decode(),
            buffer_id,
        )
      transformation_input.buffer_origin[buffer_id] = quant_params
      transformation_input.model.buffers[buffer_id].data = (
          transformation_utils.pack_data(
              quant_params.num_bits,
              np.ravel(np.asarray(quant_params.quantized_data)).view(np.uint8),
          )
      )

  if isinstance(transformation_input.quant_params, qtyping.UniformQuantParams):
    if transformation_input.quant_params.quantized_dimensions is not None:
      flatbuffer_quantization = _perform_multi_axis_quantization(
          transformation_input
      )
    elif transformation_input.quant_params.block_size == 0:
      flatbuffer_quantization = _perform_channelwise_quantization(
          transformation_input
      )
    else:
      flatbuffer_quantization = _perform_blockwise_quantization(
          transformation_input
      )
    tensor.quantization = flatbuffer_quantization
    tensor.type = quant_params_to_tflite_type(
        transformation_input.quant_params.num_bits,
        transformation_input.quant_params.signed,
    )
  if isinstance(
      transformation_input.quant_params, qtyping.NonLinearQuantParams
  ):
    tensor.type = nonlinear_quant_params_to_tflite_type(
        transformation_input.quant_params.num_bits
    )

  return qtyping.TransformationInfo(
      0, num_ops_added=0, output_tensor_id=transformation_input.tensor_id
  )
