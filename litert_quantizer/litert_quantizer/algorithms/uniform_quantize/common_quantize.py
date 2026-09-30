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

"""Quantization helpers common to all uniform quantization algorithms.

This file contains quantization helpers common to all uniform quantization
algorithms. The materialize_op functions require algorithm-specific logic to
produce the quantization parameters (e.g. scale, zero point) for each tensor,
which is encapsulated in get_tensor_quant_params_fn. Each algorithm is required
to implement the get_tensor_quant_params_fn with the
qtyping.GetTensorQuantParamsFuncSignature signature.
"""

from collections.abc import MutableMapping
from typing import Any, Optional, Sequence
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation
_OpQuantConstraint = common_utils.OpQuantConstraint
_TensorQuantParamsCache = common_utils.TensorQuantParamsCache
_ComputePrecision = qtyping.ComputePrecision
_TensorTransformationParams = qtyping.TensorTransformationParams


def check_if_quantized(tensor: Any) -> bool:
  """Checks if the tensor is quantized."""
  return (
      tensor.quantization is not None and tensor.quantization.scale is not None
  )


def check_op_quantization_config(
    op_name: _TFLOpName,
    op_quant_config: qtyping.OpQuantizationConfig,
    config_check_policy: qtyping.ConfigCheckPolicyDict,
) -> None:
  """Checks the op quantization config.

  Args:
    op_name: The name of the op.
    op_quant_config: The quantization config for the op.
    config_check_policy: The policy to check the op quantization config.

  Raises:
    ValueError: If the op quantization config is invalid.
  """
  if op_name in common_utils.DRQ_OR_WEIGHT_ONLY_OPS:
    if op_quant_config.weight_tensor_config is None:
      raise ValueError(
          "Weight tensor quantization is required for min/max uniform"
          " quantization."
      )
    if op_quant_config.weight_tensor_config.dtype != qtyping.TensorDataType.INT:
      raise ValueError(
          "Weights need to have integer type for min/max uniform quantization."
          " If you wish to perform float casting quantization (e.g., fp16"
          " weight only), please set algorithm key as 'float_casting'."
      )

  if op_quant_config.min_weight_elements < 0:
    raise ValueError(
        f"min_weight_elements must be non-negative for op: {op_name} with"
        f" config: {op_quant_config}."
    )

  if op_quant_config.compute_precision in [
      _ComputePrecision.INTEGER,
      _ComputePrecision.FLOAT,
  ]:
    # Use policy-based mechanism to validate op.
    common_utils.check_if_valid_op_config(
        op_name, op_quant_config, config_check_policy
    )
  common_utils.check_subchannel_config(op_name, op_quant_config)


def materialize_input(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in the virtual input op."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_output(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in the virtual output op."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_composite(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in the virtual output op."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_add(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.add."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_sub(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.sub."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_mul(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.mul."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_softmax_and_logistic(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.softmax and tfl.logistic."""
  # Hard code scales and zp values as they are hard coded in TFL kernels.
  # Softmax:
  #   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/activations.cc#L548
  # Logistic:
  #   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/activations.cc#L421
  output_activation_constraints = {
      8: qtyping.UniformQuantParams(
          num_bits=8,
          quantized_dimension=None,
          scale=np.array(1.0 / 256),
          zero_point=np.array(-128),
          symmetric=False,
      ),
      16: qtyping.UniformQuantParams(
          num_bits=16,
          quantized_dimension=None,
          scale=np.array(1.0 / 32768),
          zero_point=np.array(0),
      ),
  }

  return common_utils.materialize_op_with_output_activation_constraint(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      output_activation_constraints,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_batch_matmul(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.batch_matmul."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_embedding_lookup(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.embedding_lookup."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      inputs_to_ignore=[0],  # Lookup index does not need to be quantized.
  )


def materialize_reshape(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.reshape."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Shape tensor does not need to be quantized.
  )


def materialize_average_pool_2d(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.average_pool_2d."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
  )


def _materialize_bias_for_fc_conv_ops(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    op_tensor_params: list[_TensorTransformationParams],
    op_input_index: int = 0,
    op_weight_index: int = 1,
    op_bias_index: int = 2,
):
  """Materializes bias tensors in conv ops by updating `op_tensor_params`.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    op_tensor_params: Partially populated quantization configuration for the
      tensors associated with the op in the order of input, weight, output.
    op_input_index: Index for the input tensor in the op.
    op_weight_index: Index for the weight tensor in the op.
    op_bias_index: Index for the bias tensor in the op.
  """
  _, weight_tensor, bias_tensor, _ = (
      tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
          op_info.op,
          graph_info.subgraph_tensors,
          op_input_index,
          op_weight_index,
          op_bias_index,
      )
  )
  if bias_tensor is not None and not check_if_quantized(bias_tensor):
    bias_quant_params = None
    # Fused bias needs to be quantized for SRQ.
    # Check if SRQ.
    if (
        op_info.op_quant_config.compute_precision == _ComputePrecision.INTEGER
        and op_info.op_quant_config.activation_tensor_config is not None
    ):
      bias_content = tfl_flatbuffer_utils.get_tensor_data(
          bias_tensor,
          graph_info.buffers,
      )
      input_consumer_params = (
          op_tensor_params[op_input_index].consumers[0].parameters  # pyrefly: ignore[unsupported-operation]
      )
      weight_consumer_params = (
          op_tensor_params[op_weight_index].consumers[0].parameters  # pyrefly: ignore[unsupported-operation]
      )
      if weight_consumer_params is None and check_if_quantized(weight_tensor):
        quant_params = weight_tensor.quantization
        if op_info.op_quant_config.weight_tensor_config is None:
          raise ValueError(
              "weight_tensor_config cannot be None when weight tensor is"
              " quantized."
          )
        weight_consumer_params = qtyping.UniformQuantParams(
            num_bits=op_info.op_quant_config.weight_tensor_config.num_bits,
            scale=quant_params.scale,
            zero_point=quant_params.zeroPoint,
            quantized_dimension=quant_params.quantizedDimension,
        )
      try:
        # Bias quantization is using fixed quantization scale:
        # input_scale * weight_scale. To avoid hidden numerics error, we check
        # the quantization error in bias quantization.
        bias_quant_params = (
            uniform_quantize_tensor.symmetric_quantize_bias_tensor(
                bias_content,  # pyrefly: ignore[bad-argument-type]
                input_consumer_params,  # pyrefly: ignore[bad-argument-type]
                weight_consumer_params,  # pyrefly: ignore[bad-argument-type]
            )
        )
      except ValueError as e:
        raise ValueError(
            f"Failed to quantize bias tensor for op {op_info.op_name} with op"
            f" id {op_info.subgraph_op_index}."
        ) from e
    # We only quantize bias under SRQ. Setting is_constant=True for SRQ only
    # to avoid quantize bias for DRQ and weight-only cases.
    is_constant = (
        # Check if SRQ.
        op_info.op_quant_config.compute_precision == _ComputePrecision.INTEGER
        and op_info.op_quant_config.activation_tensor_config is not None
    )
    op_tensor_params[op_bias_index] = (
        common_utils.get_tensor_transformation_params(
            tfl_flatbuffer_utils.get_tensor_name(bias_tensor),
            op_info,
            is_inbounding_tensor=True,
            quant_params=bias_quant_params,
            is_constant=is_constant,
        )
    )


def _are_weights_too_small(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    weight_index: int,
) -> bool:
  """Checks if weights are too small to be quantized."""
  tensor = graph_info.subgraph_tensors[op_info.op.inputs[weight_index]]
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      tensor,
      graph_info.buffers,
  )
  return (
      tensor_data is not None
      and np.size(tensor_data) < op_info.op_quant_config.min_weight_elements
  )


def materialize_slice(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.slice."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[
          1,
          2,
      ],  # Begin and size indices do not need to be quantized.
  )


def materialize_select(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.select."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
      inputs_to_ignore=[
          0,
      ],  # Condition tensor does not need to be quantized.
  )


def materialize_select_v2(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.select_v2."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
      inputs_to_ignore=[
          0,
      ],  # Condition tensor does not need to be quantized.
  )


def materialize_dynamic_update_slice(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.dynamic_update_slice."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
      inputs_to_ignore=[
          2,
      ],  # start_indices do not need to be quantized.
  )


def materialize_sum(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.sum."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      inputs_to_ignore=[1],  # Axis index does not need to be quantized.
  )


def materialize_fc_conv(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
    input_index: int = 0,
    weight_index: int = 1,
    bias_index: int = 2,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in fully_connected, conv_2d and depthwise_conv_2d.

  Args:
    get_tensor_quant_params_fn: A function to get the quantization parameters
      for a tensor.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.
    input_index: Index for the input tensor in the op.
    weight_index: Index for the weight tensor in the op.
    bias_index: Index for the bias tensor in the op.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  ignored_inputs = [bias_index]  # Bias tensor is quantized separately.
  should_ignore_weight = False
  if graph_info:
    w_tensor = graph_info.subgraph_tensors[op_info.op.inputs[weight_index]]
    should_ignore_weight = check_if_quantized(w_tensor)
  if should_ignore_weight or _are_weights_too_small(
      op_info, graph_info, weight_index
  ):
    ignored_inputs.append(weight_index)

  op_tensor_params = common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      inputs_to_ignore=ignored_inputs,
  )

  _materialize_bias_for_fc_conv_ops(
      op_info,
      graph_info,
      op_tensor_params,
      op_input_index=input_index,
      op_weight_index=weight_index,
      op_bias_index=bias_index,
  )

  return op_tensor_params


def materialize_conv2d_transpose(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.conv2d_transpose.

  Args:
    get_tensor_quant_params_fn: A function to get the quantization parameters
      for a tensor.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  ignored_shape_index = 0
  weight_index = 1
  input_index = 2
  bias_index = 3

  ignored_inputs = [
      ignored_shape_index,
      bias_index,  # Bias tensor is quantized separately.
  ]
  if _are_weights_too_small(op_info, graph_info, weight_index):
    ignored_inputs.append(weight_index)

  op_tensor_params = common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      inputs_to_ignore=ignored_inputs,
  )
  if len(op_tensor_params) < 2:
    raise ValueError(
        "Materialize standard op should return at least two tensors for"
        " conv2d_transpose."
    )
  _materialize_bias_for_fc_conv_ops(
      op_info,
      graph_info,
      op_tensor_params,
      op_input_index=input_index,
      op_weight_index=weight_index,
      op_bias_index=bias_index,
  )

  return op_tensor_params


def materialize_tanh(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.tanh."""
  # Hard code scales and zero point values as they are hard coded in:
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/ir/tfl_ops.td#L3430
  output_activation_constraints = {}
  for num_bits in [8, 16]:
    output_activation_constraints[num_bits] = qtyping.UniformQuantParams(
        num_bits=num_bits,
        quantized_dimension=None,
        scale=np.array(1.0 / (1 << (num_bits - 1))),
        zero_point=np.array(0),
        # Activation is always asymmetric for 8 bit and symmetric for 16 bits.
        symmetric=num_bits == 16,
    )
  return common_utils.materialize_op_with_output_activation_constraint(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      output_activation_constraints,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_transpose(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.transpose."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Permutation tensor does not need to be quantized.
  )


def materialize_gelu(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.gelu."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_strided_slice(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.strided_slice."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1, 2, 3],  # Ignore the begin, end, and strides tensors.
  )


def materialize_mean(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.mean."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      inputs_to_ignore=[1],  # Axis tensor does not need to be quantized.
  )


def materialize_rsqrt(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.rsqrt."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_concatenation(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.concatenation."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
  )


def materialize_split(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.split."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[0],  # Split dimension does not need to be quantized.
  )


def materialize_pad(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.pad."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Paddings tensor does not need to be quantized.
  )


def materialize_padv2(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.padv2."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
      inputs_to_ignore=[1],  # Paddings tensor does not need to be quantized.
  )


def materialize_mirror_pad(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.mirror_pad.

  Args:
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    A list of `_TensorTransformationParams` for the tensors in the op.
  """
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Paddings tensor does not need to be quantized.
  )


def materialize_space_to_depth(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.space_to_depth.

  Args:
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    A list of `_TensorTransformationParams` for the tensors in the op.
  """
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
  )


def materialize_squared_difference(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.squared_difference."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_max_pool_2d(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.max_pool_2d."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
  )


def materialize_resize_bilinear(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.resize_bilinear."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Resize size does not need to be quantized.
  )


def materialize_resize_nearest_neighbor(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.resize_nearest_neighbor."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Resize size does not need to be quantized.
  )


def materialize_gather_nd(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.gather_nd."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Gather indices do not need to be quantized.
  )


def materialize_maximum(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.maximum."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
  )


def materialize_pack(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.pack."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
  )


def materialize_unpack(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.unpack."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
  )


def materialize_div(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.div."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_broadcast_to(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.broadcast_to."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Shape tensor does not need to be quantized.
  )


def materialize_sqrt(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.sqrt."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_hard_swish(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.hard_swish."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_gather(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.gather."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Indices do not need to be quantized.
  )


def materialize_reduce_min(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.reduce_min."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Axis index does not need to be quantized.
  )


def materialize_equal(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.equal."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_not_equal(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.not_equal."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def materialize_relu(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    tensor_quant_params_cache: _TensorQuantParamsCache,
) -> list[_TensorTransformationParams]:
  """Materialize tensors in tfl.relu."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
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
      new_shape.append(int(val / block_size))
      new_shape.append(block_size)
    else:
      new_shape.append(val)
  return new_shape


def _reshape_data_for_blockwise(
    tensor_data: np.ndarray,
    quantized_dim: int,
    block_size: int,
) -> tuple[np.ndarray, int]:
  """Reshapes data for blockwise quantization.

  Args:
    tensor_data: The original tensor data.
    quantized_dim: The dimension to be quantized blockwise.
    block_size: The size of the block. `block_size must be a multiple of 32. `
      `The tensor quantized dimension shape must be divisible by block_size.

  Returns:
    A tuple containing the reshaped tensor data and the new reduce dimension.
  """

  # TODO: b/417508018 - create AEQ specific error class instead of
  # using generic ValueError.
  if tensor_data.shape[quantized_dim] % block_size != 0:
    raise ValueError(
        "Tensor quantization dimension must be divisible by block size for"
        " blockwise quantization."
    )
  new_shape = _get_tensor_shape_for_blockwise(
      tensor_data.shape, quantized_dim, block_size
  )
  reshaped_data = tensor_data.reshape(new_shape)
  return reshaped_data, quantized_dim + 1


def broadcast_scale_zp_for_blockwise(
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


def init_tensor_min_max(
    tensor_data: Optional[np.ndarray],
    op_info: qtyping.OpInfo,
) -> qtyping.QSV:
  """Initialize the min/max for a tensor.

  This function initializes the min/max values for a tensor.

  Args:
    tensor_data: The tensor data.
    op_info: Aggregated information about the op (e.g., quantization config).

  Returns:
    A dictionary containing the min/max values for the tensor, or an empty
    dictionary if the tensor data is None.
  """
  weight_tensor_config = op_info.op_quant_config.weight_tensor_config
  if tensor_data is None or weight_tensor_config is None:
    return {}
  else:
    # Get reduce dimension for min/max calculation based on quantization
    # granularity.
    granularity = weight_tensor_config.granularity
    if granularity == qtyping.QuantGranularity.TENSORWISE:
      reduce_dims = None
      keep_dims = True
    elif granularity == qtyping.QuantGranularity.CHANNELWISE:
      quantized_dim = common_utils.get_weight_quantized_dim(
          op_info, tensor_data, weight_tensor_config.granularity
      )
      reduce_dims = common_utils.get_reduce_dims(
          quantized_dim, tensor_data.shape
      )
      keep_dims = True
    elif uniform_quantize_tensor.is_blockwise(granularity):
      tensor_data, reduce_dims = (
          uniform_quantize_tensor.reshape_data_for_blockwise(
              tensor_data,
              op_info.op_name,
              granularity,
          )
      )
      keep_dims = False
    else:
      raise ValueError(f"Unsupported granularity: {granularity}")
    return {
        "min": np.min(tensor_data, axis=reduce_dims, keepdims=keep_dims),
        "max": np.max(tensor_data, axis=reduce_dims, keepdims=keep_dims),
    }


def get_activation_min_max(
    tensor_content: np.ndarray,
    valid_float_range_min: float | None = None,
    valid_float_range_max: float | None = None,
) -> dict[str, np.ndarray]:
  """Gets activation QSV (min and max boundaries) for the tensor.

  Args:
    tensor_content: The Numpy array containing the activation tensor values.
    valid_float_range_min: The minimum valid boundary for floating point data.
      Values below this are ignored for min computation unless all are below.
    valid_float_range_max: The maximum valid boundary for floating point data.
      Values above this are ignored for max computation unless all are above.

  Returns:
    A dictionary containing 'min' and 'max' QSV numpy arrays with the
    boundaries.
  """
  qsv_shape = (1,) * tensor_content.ndim

  if np.issubdtype(tensor_content.dtype, np.integer):
    t_min = np.min(tensor_content)
    t_max = np.max(tensor_content)
  else:
    if valid_float_range_min is not None:
      t_min = np.min(
          tensor_content,
          where=tensor_content > valid_float_range_min,
          initial=np.inf,
          axis=None,
      )
      if t_min == np.inf:
        t_min = np.min(tensor_content)
    else:
      t_min = np.min(tensor_content)

    if valid_float_range_max is not None:
      t_max = np.max(
          tensor_content,
          where=tensor_content < valid_float_range_max,
          initial=-np.inf,
          axis=None,
      )
      if t_max == -np.inf:
        t_max = np.max(tensor_content)
    else:
      t_max = np.max(tensor_content)

  return {
      "min": np.reshape(t_min, qsv_shape),
      "max": np.reshape(t_max, qsv_shape),
  }


def collect_activation_tensor_statistics(
    tensor_idx: int,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: MutableMapping[str, np.ndarray],
    valid_float_range_min: float | None = None,
    valid_float_range_max: float | None = None,
) -> tuple[str, np.ndarray, dict[str, np.ndarray]] | None:
  """Collects activation statistics (QSV) for the specified tensor.

  Collected statistics include: `min`, `max`, and `num_samples`.

  Args:
    tensor_idx: The index of the tensor to evaluate.
    graph_info: Graph information holding the tensors sequence.
    tensor_content_map: Map from tensor name to numeric np.ndarray content.
    valid_float_range_min: The valid minimum bound float threshold.
    valid_float_range_max: The valid maximum bound float threshold.

  Returns:
    A tuple of (tensor_name, tensor_content_array, qsv) or None if the tensor
    is a constant and should be ignored.
  """
  tensor = graph_info.subgraph_tensors[tensor_idx]
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      tensor, graph_info.buffers
  )
  # Skip constant tensors.
  if tensor_data is not None:
    return None

  tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
  tensor_content = tensor_content_map[tensor_name]
  qsv = get_activation_min_max(
      tensor_content,
      valid_float_range_min=valid_float_range_min,
      valid_float_range_max=valid_float_range_max,
  )
  qsv["num_samples"] = np.array(
      tensor_content.shape[0] if tensor_content.ndim > 0 else 1
  )
  return tensor_name, tensor_content, qsv


def get_tensor_indices_requiring_calibration(
    tfl_op: qtyping.OperatorT,
    graph_info: qtyping.GraphInfo,
    inputs_to_ignore: Sequence[int] | None = None,
    outputs_to_ignore: Sequence[int] | None = None,
) -> list[int]:
  """Gets tensor indices that require calibration statistics collection.

  Args:
    tfl_op: The TFLite operator being processed.
    graph_info: The model's graph information holding the tensors list.
    inputs_to_ignore: A sequence of input tensor positional indices to skip.
    outputs_to_ignore: A sequence of output tensor positional indices to skip.

  Returns:
    A list of global tensor indices within the graph that require calibration
    logic.
  """
  inputs_to_ignore_set = set(inputs_to_ignore or [])
  # Ignore any already quantized inputs.
  inputs_to_ignore_set.update(
      opr_idx
      for opr_idx, tensor_idx in enumerate(tfl_op.inputs)
      if check_if_quantized(graph_info.subgraph_tensors[tensor_idx])
  )
  outputs_to_ignore_set = set(outputs_to_ignore or [])

  tensor_ids = [
      tid
      for k, tid in enumerate(tfl_op.inputs)
      if k not in inputs_to_ignore_set and tid != -1
  ] + [
      tid
      for k, tid in enumerate(tfl_op.outputs)
      if k not in outputs_to_ignore_set and tid != -1
  ]
  return tensor_ids
