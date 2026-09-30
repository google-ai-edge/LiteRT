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

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.algorithms.uniform_quantize import octav
from litert_quantizer.algorithms.uniform_quantize.op_architecture_tests import test_utils as op_test_utils
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_ComputePrecision = qtyping.ComputePrecision
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_OpTestInfo = op_test_utils.OpTestInfo


_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(
    "../../../tests/models"
)


class InputOutputTest(op_test_utils.BaseQuantizeTest):

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "single_transpose.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    self._model_qsv = {
        "serving_default_input_2:0": {
            "min": np.array([[-10]]),
            "max": np.array([[8]]),
        },
        "PartitionedCall:0": {"min": np.array([[10]]), "max": np.array([[88]])},
    }
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._input_op = qtyping.IOOperator(
        inputs=[],
        outputs=self._test_model.subgraphs[0].inputs,
        op_key=_TFLOpName.INPUT,
    )
    self._output_op = qtyping.IOOperator(
        inputs=self._test_model.subgraphs[0].outputs,
        outputs=[],
        op_key=_TFLOpName.OUTPUT,
    )

  @parameterized.product(
      weight_num_bits=[4, 8],
      symmetric_weight=[True, False],
      granularity=[
          qtyping.QuantGranularity.CHANNELWISE,
          qtyping.QuantGranularity.TENSORWISE,
      ],
  )
  def test_materialize_input_float(
      self, weight_num_bits, symmetric_weight, granularity
  ):
    weight_config = _TensorQuantConfig(
        num_bits=weight_num_bits,
        symmetric=symmetric_weight,
        granularity=granularity,
    )
    op_quant_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=None,
        weight_tensor_config=weight_config,
        compute_precision=_ComputePrecision.FLOAT,
        explicit_dequantize=True,
    )
    op_info = qtyping.OpInfo(
        op=self._input_op,
        op_name=qtyping.TFLOperationName.INPUT,
        subgraph_op_index=-1,  # Virtual op, no real id.
        op_quant_config=op_quant_config,
    )
    quantization_params = common_quantize.materialize_input(
        naive_min_max_quantize.get_tensor_quant_params,
        op_info,
        self._graph_info,
        self._model_qsv,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
    )
    # Only one input tensor for the test model.
    self.assertLen(quantization_params, 1)
    quant_param = quantization_params[0]
    self.assertEqual(quant_param.tensor_name, "serving_default_input_2:0")
    # Produced by the virtual Input op.
    self.assertIsNotNone(quant_param.producer)
    producer_info = quant_param.producer
    self.assertEqual(producer_info.subgraph_op_id, -1)
    self.assertListEqual(
        producer_info.transformations, [qtyping.QuantTransformation.NO_QUANTIZE]
    )
    self.assertIsNone(producer_info.parameters)

  @parameterized.product(
      get_tensor_quant_params_func=(
          naive_min_max_quantize.get_tensor_quant_params,
          octav.get_tensor_quant_params,
      ),
      weight_num_bits=[4, 8],
      symmetric_weight=[True, False],
      granularity=[
          qtyping.QuantGranularity.CHANNELWISE,
          qtyping.QuantGranularity.TENSORWISE,
      ],
  )
  def test_materialize_output_float(
      self,
      get_tensor_quant_params_func,
      weight_num_bits,
      symmetric_weight,
      granularity,
  ):
    weight_config = _TensorQuantConfig(
        num_bits=weight_num_bits,
        symmetric=symmetric_weight,
        granularity=granularity,
    )
    op_quant_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=None,
        weight_tensor_config=weight_config,
        compute_precision=_ComputePrecision.FLOAT,
        explicit_dequantize=True,
    )
    op_info = qtyping.OpInfo(
        op=self._output_op,
        op_name=qtyping.TFLOperationName.OUTPUT,
        subgraph_op_index=-1,  # Virtual op, no real id.
        op_quant_config=op_quant_config,
    )
    quantization_params = common_quantize.materialize_output(
        get_tensor_quant_params_func,
        op_info,
        self._graph_info,
        self._model_qsv,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
    )
    # Only one output tensor for the test model.
    self.assertLen(quantization_params, 1)
    quant_param = quantization_params[0]
    self.assertEqual(quant_param.tensor_name, "PartitionedCall:0")
    # Consumed by the virtual Output op.
    self.assertLen(quant_param.consumers, 1)
    consumer_info = quant_param.consumers[0]
    self.assertEqual(consumer_info.subgraph_op_id, -1)
    self.assertListEqual(
        consumer_info.transformations, [qtyping.QuantTransformation.NO_QUANTIZE]
    )
    self.assertIsNone(consumer_info.parameters)

  @parameterized.product(
      get_tensor_quant_params_func=(
          naive_min_max_quantize.get_tensor_quant_params,
          octav.get_tensor_quant_params,
      ),
      act_num_bits=[8, 16],
      weight_num_bits=[4, 8],
      granularity=[
          qtyping.QuantGranularity.CHANNELWISE,
          qtyping.QuantGranularity.TENSORWISE,
      ],
  )
  def test_materialize_input_integer(
      self,
      get_tensor_quant_params_func,
      act_num_bits,
      weight_num_bits,
      granularity,
  ):
    activation_config = _TensorQuantConfig(
        num_bits=act_num_bits,
        symmetric=True,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    weight_config = _TensorQuantConfig(
        num_bits=weight_num_bits,
        symmetric=True,
        granularity=granularity,
    )
    op_quant_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=activation_config,
        weight_tensor_config=weight_config,
        compute_precision=_ComputePrecision.INTEGER,
    )
    op_info = qtyping.OpInfo(
        op=self._input_op,
        op_name=qtyping.TFLOperationName.INPUT,
        subgraph_op_index=-1,  # Virtual op, no real id.
        op_quant_config=op_quant_config,
    )
    quantization_params = common_quantize.materialize_input(
        get_tensor_quant_params_func,
        op_info,
        self._graph_info,
        self._model_qsv,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
    )
    # Only one input tensor for the test model.
    self.assertLen(quantization_params, 1)
    quant_param = quantization_params[0]
    self.assertEqual(quant_param.tensor_name, "serving_default_input_2:0")
    # Produced by the virtual Input op.
    self.assertIsNotNone(quant_param.producer)
    producer_info = quant_param.producer
    self.assertEqual(producer_info.subgraph_op_id, -1)
    self.assertListEqual(
        producer_info.transformations,
        [qtyping.QuantTransformation.ADD_DEQUANTIZE],
    )
    self.assertIsNotNone(producer_info.parameters)
    quant_params = producer_info.parameters
    self.assertEqual(quant_params.num_bits, act_num_bits)

  @parameterized.product(
      get_tensor_quant_params_func=(
          naive_min_max_quantize.get_tensor_quant_params,
          octav.get_tensor_quant_params,
      ),
      act_num_bits=[8, 16],
      weight_num_bits=[4, 8],
      granularity=[
          qtyping.QuantGranularity.CHANNELWISE,
          qtyping.QuantGranularity.TENSORWISE,
      ],
  )
  def test_materialize_output_integer(
      self,
      get_tensor_quant_params_func,
      act_num_bits,
      weight_num_bits,
      granularity,
  ):
    activation_config = _TensorQuantConfig(
        num_bits=act_num_bits,
        symmetric=True,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    weight_config = _TensorQuantConfig(
        num_bits=weight_num_bits,
        symmetric=True,
        granularity=granularity,
    )
    op_quant_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=activation_config,
        weight_tensor_config=weight_config,
        compute_precision=_ComputePrecision.INTEGER,
    )
    op_info = qtyping.OpInfo(
        op=self._output_op,
        op_name=qtyping.TFLOperationName.OUTPUT,
        subgraph_op_index=-1,  # Virtual op, no real id.
        op_quant_config=op_quant_config,
    )
    quantization_params = common_quantize.materialize_output(
        get_tensor_quant_params_func,
        op_info,
        self._graph_info,
        self._model_qsv,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
    )
    # Only one output tensor for the test model.
    self.assertLen(quantization_params, 1)
    quant_param = quantization_params[0]
    self.assertEqual(quant_param.tensor_name, "PartitionedCall:0")
    # Consumed by the virtual Output op.
    self.assertLen(quant_param.consumers, 1)
    consumer_info = quant_param.consumers[0]
    self.assertEqual(consumer_info.subgraph_op_id, -1)
    self.assertListEqual(
        consumer_info.transformations,
        [qtyping.QuantTransformation.ADD_QUANTIZE],
    )
    self.assertIsNotNone(consumer_info.parameters)
    quant_params = consumer_info.parameters
    self.assertEqual(quant_params.num_bits, act_num_bits)


if __name__ == "__main__":
  absltest.main()
