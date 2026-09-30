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

"""Test utils for uniform quantize op architecture tests."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, Optional

from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_ComputePrecision = qtyping.ComputePrecision
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation


@dataclasses.dataclass
class OpTestInfo:
  """Aggregate op test information.

  Attributes:
    test_model: The test model.
    op_tensor_names: A map of op tensor names with keys for input(s) and output.
      Note for multiple inputs, the keys will follow the pattern of "input",
      "input2", "input3" etc. for ops tested with _test_no_weights_op. For ops
      tested with _test_fc_bmm_conv, the format for non-ignored inputs is the
      same, but the ignored tensors should have keys "ignored_input",
      "ignored_input2", "ignored_input3" etc.
    input_range: A tuple of input min and max.
    output_range: A tuple of output min and max.
    quantized_dimension: Quantized dimension.
  """

  test_model: Any
  op_tensor_names: dict[str, str]
  input_range: tuple[np.ndarray, np.ndarray]
  output_range: tuple[np.ndarray, np.ndarray]
  quantized_dimension: int = 0


DEFAULT_ACTIVATION_QUANT_SETTING = _TensorQuantConfig(
    num_bits=8,
    symmetric=False,
    granularity=qtyping.QuantGranularity.TENSORWISE,
)
DEFAULT_WEIGHT_QUANT_SETTING = _TensorQuantConfig(
    num_bits=8,
    symmetric=True,
    granularity=qtyping.QuantGranularity.CHANNELWISE,
)


class BaseQuantizeTest(parameterized.TestCase):
  """Base test class for uniform quantize op architecture tests.

  This class provides test utils for uniform quantize op architecture tests.
  """

  def setUp(self):
    super().setUp()
    self._tensor_name_to_qsv = {}

  def _setup_op_test_config(
      self,
      compute_precision,
      is_drq,
      op_test_info,
      num_inputs=1,
      num_outputs=1,
  ):
    """Helper to set up qsv for op test."""
    # SRQ requires QSVs (min/max).
    input_min, input_max = op_test_info.input_range
    output_min, output_max = op_test_info.output_range
    # Check if SRQ.
    if compute_precision == _ComputePrecision.INTEGER and not is_drq:
      input_qsv = {
          "min": input_min,
          "max": input_max,
      }
      output_qsv = {
          "min": output_min,
          "max": output_max,
      }
      for i in range(num_inputs):
        input_name = "input"
        if i > 0:
          input_name = f"input{i+1}"
        self._tensor_name_to_qsv[op_test_info.op_tensor_names[input_name]] = (
            input_qsv
        )
      for i in range(num_outputs):
        output_name = "output"
        if i > 0:
          output_name = f"output{i+1}"
        self._tensor_name_to_qsv[op_test_info.op_tensor_names[output_name]] = (
            output_qsv
        )

  def _test_same_direction_tensors(
      self,
      tensor_quant_params,
      op_quant_config,
      op_info,
      op_test_info,
      num_tensors,
      indices_to_ignore,
      is_inbounding_tensor,
      constant_inputs=None,
  ):
    """Tests all input or output tensors in provided quant params.

    Args:
      tensor_quant_params: Tensor transformation parameters.
      op_quant_config: Op quantization config.
      op_info: OpInfo object.
      op_test_info: OpTestInfo object.
      num_tensors: Number of tensors to test.
      indices_to_ignore: Indices of tensors to ignore.
      is_inbounding_tensor: Whether to test all inbounding tensors.
      constant_inputs: Constant inputs indices.
    """
    if not is_inbounding_tensor and constant_inputs is not None:
      raise ValueError(
          "Constant inputs should only be used for inbounding tensors."
      )
    constant_inputs = constant_inputs or []
    tensor_base_name = "input" if is_inbounding_tensor else "output"
    tensor_names = [tensor_base_name] + [
        f"{tensor_base_name}{i+2}" for i in range(num_tensors - 1)
    ]
    for i in range(num_tensors):
      if (
          # Check if SRQ.
          op_quant_config.compute_precision == _ComputePrecision.INTEGER
          and op_quant_config.activation_tensor_config is not None
          and i not in indices_to_ignore
      ):
        if is_inbounding_tensor:
          if i in constant_inputs:
            transformations = [_QuantTransformation.QUANTIZE_TENSOR]
          else:
            transformations = [_QuantTransformation.ADD_QUANTIZE]
        else:
          if i in constant_inputs:
            transformations = []
          else:
            transformations = [_QuantTransformation.ADD_DEQUANTIZE]
      else:
        transformations = [_QuantTransformation.NO_QUANTIZE]
      self._test_tensor_transformation_params(
          op_test_info.op_tensor_names[tensor_names[i]],
          op_info.subgraph_op_index,
          is_inbounding_tensor=is_inbounding_tensor,
          tensor_quant_config=op_quant_config.activation_tensor_config,
          transformation_params=tensor_quant_params[i],
          desired_transformations=transformations,
      )

  def _test_same_input_output_params(
      self,
      tensor_quant_params,
      num_inputs,
      num_outputs,
      inputs_to_ignore,
      outputs_to_ignore,
  ):
    """Tests input and output tensor transformation parameters are the same.

    Assumes that each of non-ignored input tensors has exactly one consumer.

    Args:
      tensor_quant_params: Tensor transformation parameters.
      num_inputs: Number of inputs in materialization function result.
      num_outputs: Number of outputs in materialization function result.
      inputs_to_ignore: Inputs to ignore.
      outputs_to_ignore: Outputs to ignore.
    """
    num_quant_inputs = num_inputs - len(inputs_to_ignore)
    num_quant_outputs = num_outputs - len(outputs_to_ignore)
    self.assertTrue(num_quant_inputs == 1 or num_quant_outputs == 1)

    # Test inputs. Ignoring `quantized_data` in comparison because inputs can be
    # constants and therefore have different quantized data.
    inputs_to_ignore = inputs_to_ignore or []
    input_quant_params = [
        dataclasses.replace(
            tensor_quant_params[i].consumers[0].parameters, quantized_data=None  # pytype: disable=attribute-error
        )
        for i in range(num_inputs)
        if i not in inputs_to_ignore
    ]
    expected_params = input_quant_params[0]
    for i in range(1, num_quant_inputs):
      self.assertEqual(input_quant_params[i], expected_params)

    # Test outputs.
    outputs_to_ignore = outputs_to_ignore or []
    for i in range(num_outputs):
      if i not in outputs_to_ignore:
        output_tensor_quant_params = tensor_quant_params[
            i + num_inputs
        ].producer.parameters  # pytype: disable=attribute-error
        self.assertEqual(output_tensor_quant_params, expected_params)

  def _test_no_weights_op(
      self,
      op_info,
      graph_info,
      op_test_info,
      materialization_func,
      get_tensor_quant_params_func,
      same_input_output_params=False,
      inputs_to_ignore=None,
      outputs_to_ignore=None,
      constant_inputs=None,
  ):
    """Test an op without weights and bias.

    Args:
      op_info: OpInfo object.
      graph_info: GraphInfo object.
      op_test_info: OpTestInfo object. In op_tensor_names field of this object,
        op tensor input names have to follow the pattern of "input", "input2",
        "input3" etc.
      materialization_func: Function to materialize tensor transformation
        parameters.
      get_tensor_quant_params_func: Function to get tensor quantization
        parameters.
      same_input_output_params: Whether the input and output tensor
        transformation parameters are the same.
      inputs_to_ignore: Inputs to ignore.
      outputs_to_ignore: Outputs to ignore.
      constant_inputs: Indices of constant inputs.
    """
    num_inputs = len(op_info.op.inputs)
    num_outputs = len(op_info.op.outputs)
    op_quant_config = op_info.op_quant_config
    self._setup_op_test_config(
        compute_precision=op_quant_config.compute_precision,
        is_drq=op_quant_config.activation_tensor_config is None,
        op_test_info=op_test_info,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )
    tensor_quant_params = materialization_func(
        get_tensor_quant_params_func,
        op_info,
        graph_info,
        tensor_name_to_qsv=self._tensor_name_to_qsv,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
    )
    self.assertLen(tensor_quant_params, num_inputs + num_outputs)

    # Test input tensor settings.
    inputs_to_ignore = inputs_to_ignore or []
    self._test_same_direction_tensors(
        tensor_quant_params[:num_inputs],
        op_quant_config,
        op_info,
        op_test_info,
        num_inputs,
        inputs_to_ignore,
        is_inbounding_tensor=True,
        constant_inputs=constant_inputs,
    )
    # Test output tensor settings.
    outputs_to_ignore = outputs_to_ignore or []
    self._test_same_direction_tensors(
        tensor_quant_params[num_inputs:],
        op_quant_config,
        op_info,
        op_test_info,
        num_outputs,
        outputs_to_ignore,
        is_inbounding_tensor=False,
    )
    # Test same input and output params.
    if same_input_output_params:
      self._test_same_input_output_params(
          tensor_quant_params,
          num_inputs,
          num_outputs,
          inputs_to_ignore,
          outputs_to_ignore,
      )
    # Test ignored inputs and outputs.
    self._test_ignored_inputs_and_outputs(
        tensor_quant_params,
        op_info,
        op_test_info,
        inputs_to_ignore,
        outputs_to_ignore,
    )

  def _test_fc_bmm_conv(
      self,
      op_info,
      graph_info,
      op_test_info,
      materialization_func,
      get_tensor_quant_params_func,
      bias_quantized_dim: int = 0,
      input_index: int = 0,
      weight_index: int = 1,
      bias_index: int = 2,
      inputs_to_ignore: Optional[Sequence[int]] = None,
      outputs_to_ignore: Optional[Sequence[int]] = None,
  ):
    """Tests fully connected, batch matmul and conv ops.

    Assumes that the op has one non-ignored output, which comes last in the
    materialization function result.

    Args:
      op_info: OpInfo object.
      graph_info: GraphInfo object.
      op_test_info: OpTestInfo object. In op_tensor_names field of this object,
        not ignored input tensors' names have to follow the pattern of "input",
        "input2", "input3" etc. Ignored input tensors' names have to follow the
        pattern of "ignored_input", "ignored_input2", "ignored_input3" etc.
      materialization_func: Function to materialize tensor transformation
        parameters.
      get_tensor_quant_params_func: Function to get tensor quantization
        parameters.
      bias_quantized_dim: Quantized dimension for bias.
      input_index: Input tensor index in the result of materialization function.
      weight_index: Weight tensor index in the result of materialization
        function.
      bias_index: Bias tensor index in the result of materialization function.
      inputs_to_ignore: Inputs to ignore.
      outputs_to_ignore: Outputs to ignore.
    """
    op_quant_config = op_info.op_quant_config
    self._setup_op_test_config(
        compute_precision=op_quant_config.compute_precision,
        is_drq=op_quant_config.activation_tensor_config is None,
        op_test_info=op_test_info,
    )
    tensor_quant_params = materialization_func(
        get_tensor_quant_params_func,
        op_info,
        graph_info,
        tensor_name_to_qsv=self._tensor_name_to_qsv,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
    )

    _, weight_tensor, bias_tensor, _ = (
        tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
            op_info.op,
            graph_info.subgraph_tensors,
        )
    )
    num_configs = 4 if bias_tensor is not None else 3
    if inputs_to_ignore:
      num_configs += len(inputs_to_ignore)
    if outputs_to_ignore:
      num_configs += len(outputs_to_ignore)
    self.assertLen(tensor_quant_params, num_configs)
    bias_tensor_data = None
    if bias_tensor is not None:
      bias_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
          bias_tensor,
          op_test_info.test_model.buffers,
      )
    # Test input tensor settings
    transformations = [_QuantTransformation.NO_QUANTIZE]
    # Check if SRQ.
    if (
        op_quant_config.compute_precision == _ComputePrecision.INTEGER
        and op_quant_config.activation_tensor_config is not None
    ):
      transformations = [_QuantTransformation.ADD_QUANTIZE]
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["input"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=True,
        tensor_quant_config=op_quant_config.activation_tensor_config,
        transformation_params=tensor_quant_params[input_index],
        desired_transformations=transformations,
    )

    # Test weight tensor settings.
    transformations = [_QuantTransformation.QUANTIZE_TENSOR]
    # Check if WEIGHT_ONLY.
    if (
        op_quant_config.compute_precision == _ComputePrecision.FLOAT
        and op_quant_config.explicit_dequantize
    ):
      transformations = [
          _QuantTransformation.ADD_DEQUANTIZE,
      ]
    weight_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        weight_tensor,
        op_test_info.test_model.buffers,
    )
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["weight"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=True,
        tensor_quant_config=op_quant_config.weight_tensor_config,
        transformation_params=tensor_quant_params[weight_index],
        desired_transformations=transformations,
        tensor_data=weight_tensor_data,
        quantized_dimension=op_test_info.quantized_dimension,
    )

    # Test output tensor settings.
    transformations = [_QuantTransformation.NO_QUANTIZE]
    # Check if SRQ.
    if (
        op_quant_config.compute_precision == _ComputePrecision.INTEGER
        and op_quant_config.activation_tensor_config is not None
    ):
      transformations = [_QuantTransformation.ADD_DEQUANTIZE]
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["output"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=False,
        tensor_quant_config=op_quant_config.activation_tensor_config,
        transformation_params=tensor_quant_params[-1],
        desired_transformations=transformations,
    )

    # Test bias tensor settings.
    if bias_tensor is not None:
      bias_bit_width = 32
      transformations = [_QuantTransformation.NO_QUANTIZE]
      # Check if SRQ.
      if (
          op_quant_config.compute_precision == _ComputePrecision.INTEGER
          and op_quant_config.activation_tensor_config is not None
      ):
        transformations = [_QuantTransformation.QUANTIZE_TENSOR]
        if op_quant_config.activation_tensor_config.num_bits == 16:  # pytype: disable=attribute-error
          bias_bit_width = 64
      bias_config = qtyping.TensorQuantizationConfig(
          num_bits=bias_bit_width,
          symmetric=True,
          granularity=op_quant_config.weight_tensor_config.granularity,
      )
      self._test_tensor_transformation_params(
          op_test_info.op_tensor_names["bias"],
          op_info.subgraph_op_index,
          is_inbounding_tensor=True,
          tensor_quant_config=bias_config,
          transformation_params=tensor_quant_params[bias_index],
          desired_transformations=transformations,
          tensor_data=bias_tensor_data,
          quantized_dimension=bias_quantized_dim,
      )

    # Test ignored inputs and outputs.
    self._test_ignored_inputs_and_outputs(
        tensor_quant_params,
        op_info,
        op_test_info,
        inputs_to_ignore,
        outputs_to_ignore,
        is_op_with_weight=True,
    )

  def _get_ignore_tensor_name(
      self,
      ignored_tensor_index: int,
      is_op_with_weight: bool,
      is_inbounding_tensor: bool,
  ) -> str:
    """Gets the tensor name for ignored tensors.

    Args:
      ignored_tensor_index: Index of the ignored tensor in list of
        inputs/outputs.
      is_op_with_weight: Whether the op has weight.
      is_inbounding_tensor: Whether the tensor is an inbounding tensor.

    Returns:
      Tensor name.
    """
    base_name = "input" if is_inbounding_tensor else "output"
    if is_op_with_weight:
      base_name = f"ignored_{base_name}"
    if ignored_tensor_index == 0:
      return base_name
    return f"{base_name}{ignored_tensor_index+1}"

  def _test_ignored_inputs_and_outputs(
      self,
      tensor_quant_params,
      op_info,
      op_test_info,
      inputs_to_ignore: Optional[Sequence[int]] = None,
      outputs_to_ignore: Optional[Sequence[int]] = None,
      is_op_with_weight: bool = False,
  ):
    """Tests ignored inputs and outputs.

    Args:
      tensor_quant_params: Tensor transformation parameters.
      op_info: OpInfo object.
      op_test_info: OpTestInfo object.
      inputs_to_ignore: Inputs to ignore.
      outputs_to_ignore: Outputs to ignore.
      is_op_with_weight: Whether the op has weights.
    """
    op_quant_config = op_info.op_quant_config
    # Use activation tensor config just to pass something.
    tensor_quant_config = op_quant_config.activation_tensor_config
    inputs_to_ignore = inputs_to_ignore or []
    outputs_to_ignore = outputs_to_ignore or []
    # Test ignored inputs.
    for i in inputs_to_ignore:
      tensor_name = self._get_ignore_tensor_name(
          i, is_op_with_weight=is_op_with_weight, is_inbounding_tensor=True
      )
      self._test_tensor_transformation_params(
          op_test_info.op_tensor_names[tensor_name],
          op_info.subgraph_op_index,
          is_inbounding_tensor=True,
          tensor_quant_config=tensor_quant_config,
          transformation_params=tensor_quant_params[i],
          desired_transformations=[_QuantTransformation.NO_QUANTIZE],
      )
    # Test ignored outputs.
    for i in outputs_to_ignore:
      quant_params_index = len(op_info.op.inputs) + i
      tensor_name = self._get_ignore_tensor_name(
          i, is_op_with_weight=is_op_with_weight, is_inbounding_tensor=False
      )
      self._test_tensor_transformation_params(
          op_test_info.op_tensor_names[tensor_name],
          op_info.subgraph_op_index,
          is_inbounding_tensor=False,
          tensor_quant_config=tensor_quant_config,
          transformation_params=tensor_quant_params[quant_params_index],
          desired_transformations=[_QuantTransformation.NO_QUANTIZE],
      )

  def _test_tensor_transformation_params(
      self,
      tensor_name,
      subgraph_op_id,
      is_inbounding_tensor,
      tensor_quant_config,
      transformation_params,
      desired_transformations,
      tensor_data=None,
      expected_tensor_max=None,
      quantized_dimension=0,
  ):
    """Tests tensor transformation parameters.

    Args:
      tensor_name: Tensor name.
      subgraph_op_id: Subgraph op id.
      is_inbounding_tensor: Whether the tensor is an inbounding tensor.
      tensor_quant_config: Tensor quantization config.
      transformation_params: Tensor transformation parameters.
      desired_transformations: Desired transformations.
      tensor_data: Tensor data.
      expected_tensor_max: Expected tensor max.
      quantized_dimension: Quantized dimension.
    """
    self.assertEqual(transformation_params.tensor_name, tensor_name)
    if is_inbounding_tensor:
      self.assertIsNone(transformation_params.producer)
      self.assertLen(transformation_params.consumers, 1)
      op_params = transformation_params.consumers[0]
    else:
      self.assertIsNone(transformation_params.consumers)
      op_params = transformation_params.producer
      self.assertIsNotNone(op_params)
    self.assertEqual(op_params.subgraph_op_id, subgraph_op_id)
    self.assertSequenceEqual(op_params.transformations, desired_transformations)
    quantization_params = op_params.parameters
    if desired_transformations == [_QuantTransformation.NO_QUANTIZE]:
      self.assertIsNone(quantization_params)
    else:
      self.assertIsNotNone(quantization_params)
      if (
          tensor_quant_config.granularity
          is qtyping.QuantGranularity.CHANNELWISE
      ):
        self.assertEqual(
            quantization_params.quantized_dimension, quantized_dimension
        )
      else:
        self.assertIsNone(quantization_params.quantized_dimension)
      self.assertEqual(
          quantization_params.num_bits, tensor_quant_config.num_bits
      )
      # Test correctness of quantization parameters.
      if tensor_data is not None:
        expected_quantized_data = uniform_quantize_tensor.uniform_quantize(
            tensor_data, quantization_params
        )
        self.assertSequenceEqual(
            np.ravel(expected_quantized_data).tolist(),
            np.ravel(quantization_params.quantized_data).tolist(),  # pytype: disable=attribute-error
        )
      elif expected_tensor_max:
        max_q = 2**tensor_quant_config.num_bits / 2 - 1
        calculated_tensor_max = quantization_params.scale[0] * (
            max_q - quantization_params.zero_point[0]
        )
        self.assertAlmostEqual(
            calculated_tensor_max, expected_tensor_max, delta=5e-2
        )

  def _test_materialize_fn_quantizes_weights_larger_than_min_weight_elements_for_w8_afp32(
      self,
      op_name: qtyping.TFLOperationName,
      subgraph_op_id: int,
      min_weight_elements: int,
      graph_info: qtyping.GraphInfo,
      op_test_info: OpTestInfo,
      materialization_func: Callable[..., Any],
      get_tensor_quant_params_func: Callable[..., Any],
      expect_weights_quantized: bool,
  ):
    """Tests weights are quantized if they are larger than min_weight_elements.

    Args:
      op_name: Op name.
      subgraph_op_id: Subgraph op id.
      min_weight_elements: Minimum number of weight elements.
      graph_info: GraphInfo object.
      op_test_info: OpTestInfo object.
      materialization_func: Function to materialize tensor transformation
        parameters.
      get_tensor_quant_params_func: Function to get tensor quantization
        parameters.
      expect_weights_quantized: Whether the weights are expected to be
        quantized.
    """
    subgraph0 = op_test_info.test_model.subgraphs[0]
    op = subgraph0.operators[subgraph_op_id]
    op_info = qtyping.OpInfo(
        op=op,
        op_name=op_name,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
            compute_precision=_ComputePrecision.INTEGER,
            explicit_dequantize=False,
            skip_checks=False,
            min_weight_elements=min_weight_elements,
        ),
    )
    _, weight_quant_params, *_ = materialization_func(
        get_tensor_quant_params_func,
        op_info,
        graph_info,
        tensor_name_to_qsv=self._tensor_name_to_qsv,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
    )
    self.assertEqual(
        weight_quant_params.tensor_name, op_test_info.op_tensor_names["weight"]
    )
    self.assertLen(weight_quant_params.consumers, 1)
    quantization_params = weight_quant_params.consumers[0].parameters
    transformations = weight_quant_params.consumers[0].transformations
    if expect_weights_quantized:
      self.assertIsNotNone(quantization_params)
      self.assertEqual(
          transformations,
          [_QuantTransformation.QUANTIZE_TENSOR],
      )
    else:
      self.assertIsNone(quantization_params)
      self.assertEqual(
          transformations,
          [_QuantTransformation.NO_QUANTIZE],
      )
