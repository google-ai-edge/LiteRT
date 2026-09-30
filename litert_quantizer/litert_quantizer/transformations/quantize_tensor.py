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
    if transformation_input.quant_params.block_size == 0:
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
