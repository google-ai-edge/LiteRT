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

"""Performs float casting quantization."""

from typing import Any, Optional

import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

ALGORITHM_KEY = "float_casting"
_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation

# "Fake" quantization config for float casting.
_FP16_QUANT_CONFIG = qtyping.TensorQuantizationConfig(
    num_bits=16, dtype=qtyping.TensorDataType.FLOAT
)


# Ops that support weight quantization config (e.g., support Weight-only).
SUPPORTED_WEIGHT_QUANT_OPS = frozenset([
    _TFLOpName.FULLY_CONNECTED,
    _TFLOpName.CONV_2D,
    _TFLOpName.DEPTHWISE_CONV_2D,
    _TFLOpName.CONV_2D_TRANSPOSE,
    _TFLOpName.EMBEDDING_LOOKUP,
])


def check_op_quantization_config(
    op_name: _TFLOpName,
    op_quant_config: qtyping.OpQuantizationConfig,
    config_check_policy: Optional[qtyping.ConfigCheckPolicyDict] = None,
) -> None:
  """Checks if the op is valid for float casting quantization.

  Args:
    op_name: The name of the op.
    op_quant_config: The quantization config for the op.
    config_check_policy: The policy to check the quantization config.

  Raises:
    ValueError: If the op is not supported or the compute_precision is not
      FLOAT.
  """
  # TODO: b/353780772 - Add config check policy for float casting quantization.
  if config_check_policy is not None and config_check_policy:
    raise ValueError(f"Config check isn't implemented yet for op: {op_name}.")

  # Check if WEIGHT_ONLY.
  if op_quant_config.compute_precision != qtyping.ComputePrecision.FLOAT:
    raise ValueError(
        "Currently, only Weight-Only is supported for float casting"
        " quantization. Got unsupported execution mode:"
        f" {op_quant_config.compute_precision} for op: {op_name}"
    )
  if op_quant_config.activation_tensor_config is not None:
    raise ValueError(
        "Activation tensor quantization is not supported for float casting"
        " quantization."
    )
  if op_name not in SUPPORTED_WEIGHT_QUANT_OPS:
    raise ValueError(
        f"Unsupported op: {op_name} for float casting quantization."
    )
  if op_quant_config.weight_tensor_config is None:
    raise ValueError(
        "Weight tensor quantization config is required for float casting"
        " quantization."
    )
  if (
      op_quant_config.weight_tensor_config.num_bits != 16
      or op_quant_config.weight_tensor_config.dtype
      != qtyping.TensorDataType.FLOAT
  ):
    raise ValueError(
        "Currently, float casting quantization config requires number of bits"
        " to be set as 16, dtype as float, got"
        f" {op_quant_config.weight_tensor_config.num_bits} and"
        f" {op_quant_config.weight_tensor_config.dtype} ."
    )


def materialize_fc_conv(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],  # pylint: disable=unused-argument
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in fully_connected, conv_2d and depthwise_conv_2d ops.

  This function is called by the quantization pipeline to materialize
  quantization parameters for the weight tensor of the op.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters
      (unused).
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    Quantization configuration for the weight tensor of the op.

  Raises:
    ValueError: If the op is not supported or the compute precision is not
      FLOAT.
  """

  input_tensor, weight_tensor, bias_tensor, output_tensor = (
      tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
          op_info.op, graph_info.subgraph_tensors
      )
  )
  op_tensor_params = []
  # Input tensor.
  input_quant_params = _config_no_quantize_tensor(
      op_info, input_tensor, is_inbounding_tensor=True
  )
  op_tensor_params.append(input_quant_params)
  # Weight tensor.
  weight_content = tfl_flatbuffer_utils.get_tensor_data(
      weight_tensor,
      graph_info.buffers,
  )
  if weight_content is None:
    # If the weight data is not stored in the flatbuffer (e.g. dynamic weights),
    # skip quantization for this tensor.
    op2weight_params = qtyping.OpToTensorParams(
        subgraph_op_id=op_info.subgraph_op_index,
        transformations=[_QuantTransformation.NO_QUANTIZE],
    )
  else:
    if not (
        quant_params := tensor_quant_params_cache.lookup(
            weight_tensor.buffer, _FP16_QUANT_CONFIG
        )
    ):
      quant_params = qtyping.NonLinearQuantParams(
          num_bits=16, quantized_data=weight_content.astype(np.float16)
      )
      tensor_quant_params_cache.insert(
          weight_tensor.buffer, _FP16_QUANT_CONFIG, quant_params
      )
    op2weight_params = qtyping.OpToTensorParams(
        subgraph_op_id=op_info.subgraph_op_index,
        parameters=quant_params,
        transformations=[_QuantTransformation.ADD_DEQUANTIZE],
    )
    quant_params = qtyping.NonLinearQuantParams(
        num_bits=16, quantized_data=weight_content.astype(np.float16)  # pytype: disable=attribute-error
    )
    op2weight_params = qtyping.OpToTensorParams(
        subgraph_op_id=op_info.subgraph_op_index,
        parameters=quant_params,
        transformations=[_QuantTransformation.ADD_DEQUANTIZE],
    )
  op_tensor_params.append(
      qtyping.TensorTransformationParams(
          tensor_name=tfl_flatbuffer_utils.get_tensor_name(weight_tensor),
          consumers=[op2weight_params],
      )
  )
  # Output tensor.
  output_quant_params = _config_no_quantize_tensor(
      op_info, output_tensor, is_inbounding_tensor=False
  )
  op_tensor_params.append(output_quant_params)
  # Bias tensor.
  if bias_tensor is not None:
    bias_quant_params = _config_no_quantize_tensor(
        op_info, bias_tensor, is_inbounding_tensor=True
    )
    op_tensor_params.append(bias_quant_params)
  return op_tensor_params


def materialize_embedding_lookup(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
) -> list[qtyping.TensorTransformationParams]:
  return materialize_fc_conv(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_conv2d_transpose(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],  # pylint: disable=unused-argument
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in fully_connected, conv_2d and depthwise_conv_2d ops.

  This function is called by the quantization pipeline to materialize
  quantization parameters for the weight tensor of the op.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters
      (unused).
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    Quantization configuration for the weight tensor of the op.

  Raises:
    ValueError: If the op is not supported or the execution mode is not
      WEIGHT_ONLY.
  """

  input_tensor, weight_tensor, bias_tensor, output_tensor = (
      tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
          op_info.op,
          graph_info.subgraph_tensors,
          input_index=2,
          weight_index=1,
          bias_index=3,
          output_index=0,
      )
  )
  op_tensor_params = []
  # Input tensor.
  input_quant_params = _config_no_quantize_tensor(
      op_info, input_tensor, is_inbounding_tensor=True
  )
  op_tensor_params.append(input_quant_params)
  # Weight tensor.
  weight_content = tfl_flatbuffer_utils.get_tensor_data(
      weight_tensor,
      graph_info.buffers,
  )
  if weight_content is None:
    # If the weight data is not stored in the flatbuffer (e.g. dynamic weights),
    # skip quantization for this tensor.
    op2weight_params = qtyping.OpToTensorParams(
        subgraph_op_id=op_info.subgraph_op_index,
        transformations=[_QuantTransformation.NO_QUANTIZE],
    )
  else:
    if not (
        quant_params := tensor_quant_params_cache.lookup(
            weight_tensor.buffer, _FP16_QUANT_CONFIG
        )
    ):
      quant_params = qtyping.NonLinearQuantParams(
          num_bits=16, quantized_data=weight_content.astype(np.float16)
      )
      tensor_quant_params_cache.insert(
          weight_tensor.buffer, _FP16_QUANT_CONFIG, quant_params
      )
    op2weight_params = qtyping.OpToTensorParams(
        subgraph_op_id=op_info.subgraph_op_index,
        parameters=quant_params,
        transformations=[_QuantTransformation.ADD_DEQUANTIZE],
    )
  op_tensor_params.append(
      qtyping.TensorTransformationParams(
          tensor_name=tfl_flatbuffer_utils.get_tensor_name(weight_tensor),
          consumers=[op2weight_params],
      )
  )
  # Output tensor.
  output_quant_params = _config_no_quantize_tensor(
      op_info, output_tensor, is_inbounding_tensor=False
  )
  op_tensor_params.append(output_quant_params)
  # Bias tensor.
  if bias_tensor is not None:
    bias_quant_params = _config_no_quantize_tensor(
        op_info, bias_tensor, is_inbounding_tensor=True
    )
    op_tensor_params.append(bias_quant_params)
  return op_tensor_params


def _config_no_quantize_tensor(
    op_info: qtyping.OpInfo,
    tensor: Any,
    is_inbounding_tensor: bool,
) -> qtyping.TensorTransformationParams:
  """Configures a tensor to be not quantized.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor: The tensor to be configured.
    is_inbounding_tensor: Whether the tensor is an inbounding tensor.

  Returns:
    TensorTransformationParams for the tensor.
  """
  tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      transformations=[_QuantTransformation.NO_QUANTIZE],
  )
  if is_inbounding_tensor:
    return qtyping.TensorTransformationParams(
        tensor_name=tensor_name,
        consumers=[op2tensor_params],
    )
  return qtyping.TensorTransformationParams(
      tensor_name=tensor_name, producer=op2tensor_params
  )


def init_qsvs(*_) -> qtyping.QSV:
  """Currently calibration free. Placeholder for AlgorithmManager."""
  return {}


def calibrate(*_) -> dict[str, qtyping.QSV]:
  """Currently calibration free. Placeholder for AlgorithmManager."""
  return {}
