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

from collections.abc import Generator
import pathlib
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import calibrator
from litert_quantizer import params_generator
from litert_quantizer import qtyping
from litert_quantizer import recipe_manager
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils
from litert_quantizer.utils import tfl_interpreter_utils


_ComputePrecision = qtyping.ComputePrecision
_TensorDataType = qtyping.TensorDataType
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_AlgorithmName = recipe_manager.AlgorithmName
_QuantGranularity = qtyping.QuantGranularity
_QTransf = qtyping.QuantTransformation


TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('')
_PARAMS_8BIT = qtyping.UniformQuantParams(8, None, np.array([1]), np.array([0]))


def _single_fc_model_representative_dataset_gen(num_samples=5):
  for _ in range(num_samples):
    yield {'input_1': np.random.rand(1, 8).astype(np.float32)}


def _int_transpose_model_representative_dataset_gen(num_samples=5):
  data = []
  for _ in range(num_samples):
    data.append({'input_2': np.random.rand(1, 2, 3, 4).astype(np.int32)})
  return data


def _get_calibration_data(
    dataset_gen: Generator[dict[str, Any], Any, None],
) -> dict[str, Any]:
  calibration_samples = [sample for sample in dataset_gen]
  calibration_data = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
  }
  return calibration_data


def _get_test_consumers(
    transformations_per_consumer: list[list[_QTransf]],
    params_per_consumer: list[qtyping.OpToTensorParams],
) -> list[qtyping.OpToTensorParams]:
  return [
      qtyping.OpToTensorParams(
          subgraph_op_id=i + 1,
          transformations=transformations_per_consumer[i],
          parameters=params_per_consumer[i],
      )
      for i in range(len(transformations_per_consumer))
  ]


class ParamsGeneratorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/conv_fc_mnist.tflite'
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    self._params_generator = params_generator.ParamsGenerator(self._test_model)
    self._recipe_manager = recipe_manager.RecipeManager()

  def test_update_model_quant_results(self):
    params_from_target = qtyping.TensorTransformationParams(
        tensor_name='test_tensor0',
        consumers=[
            qtyping.OpToTensorParams(
                subgraph_op_id=3,
                transformations=[
                    _QuantTransformation.ADD_QUANTIZE,
                    _QuantTransformation.ADD_DEQUANTIZE,
                ],
            )
        ],
    )
    # Test add new tensor from target
    self._params_generator._update_model_quant_results([params_from_target])
    self.assertIsNotNone(
        'test_tensor0' in self._params_generator.model_quant_results
    )
    tensor_params = self._params_generator.model_quant_results['test_tensor0']
    self.assertEqual(
        tensor_params,
        params_from_target,
    )
    # Test update new tensor from source
    params_from_source = qtyping.TensorTransformationParams(
        tensor_name='test_tensor0',
        producer=qtyping.OpToTensorParams(
            subgraph_op_id=3,
            transformations=[
                _QuantTransformation.ADD_DEQUANTIZE,
            ],
        ),
    )
    self._params_generator._update_model_quant_results([params_from_source])
    tensor_params = self._params_generator.model_quant_results['test_tensor0']
    self.assertEqual(
        tensor_params.producer,
        params_from_source.producer,
    )

    # We can have multiple target op params
    params_from_target2 = qtyping.TensorTransformationParams(
        tensor_name='test_tensor0',
        consumers=[
            qtyping.OpToTensorParams(
                subgraph_op_id=3,
                transformations=[
                    _QuantTransformation.NO_QUANTIZE,
                ],
            )
        ],
    )
    self._params_generator._update_model_quant_results([params_from_target2])
    tensor_params = self._params_generator.model_quant_results['test_tensor0']
    self.assertSequenceEqual(
        tensor_params.consumers,
        params_from_target.consumers + params_from_target2.consumers,
    )

    # but only a single source op params
    error_message = (
        'received multiple quantization parameters from the source op'
    )
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      self._params_generator._update_model_quant_results([params_from_source])

  def test_generate_config_global(self):
    # Quantize all fully_connected.
    global_recipe = [
        {
            'regex': '.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 8,
                    'symmetric': False,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # Equivalent to WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
        },
    ]
    self._recipe_manager.load_quantization_recipe(global_recipe)
    tensor_quantization_params = (
        self._params_generator.generate_quantization_parameters(
            self._recipe_manager
        )
    )
    # Every tensor in the model will have their params!
    flatbuffer_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    tensors = flatbuffer_model.subgraphs[0].tensors
    self.assertLen(tensor_quantization_params, len(tensors))

    # Input tensor
    tensor_name = 'serving_default_conv2d_input:0'
    self._test_tensor_transformation_params(
        0,
        tensor_quantization_params,
        tensor_name,
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )
    # Input tensor is produced from the virtual Input op.
    transformation_config = tensor_quantization_params[tensor_name]
    self.assertIsNotNone(transformation_config.producer)
    self.assertEqual(transformation_config.producer.subgraph_op_id, -1)

    # Intermediate tensor will have no_quantize at the both end
    tensor_name = 'sequential/average_pooling2d/AvgPool'
    self._test_tensor_transformation_params(
        1,
        tensor_quantization_params,
        tensor_name,
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )  # output from average pool
    self._test_tensor_transformation_params(
        2,
        tensor_quantization_params,
        tensor_name,
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )  # input to Reshape

    # First FC
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'sequential/flatten/Reshape',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )  # input tensor

    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'arith.constant1',
        [
            _QuantTransformation.ADD_DEQUANTIZE,
        ],
        is_inbounding_tensor=True,
        num_bits=8,
        granularity=_QuantGranularity.CHANNELWISE,
        symmetric=False,
    )  # weight tensor
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'arith.constant2',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )  # bias tensor
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )  # output tensor

    # Second FC
    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )  # input tensor
    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'arith.constant',
        [
            _QuantTransformation.ADD_DEQUANTIZE,
        ],
        is_inbounding_tensor=True,
        num_bits=8,
        granularity=_QuantGranularity.CHANNELWISE,
        symmetric=False,
    )  # weight tensor
    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'sequential/dense_1/MatMul',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )  # output tensor

    # Model output tensor
    tensor_name = 'StatefulPartitionedCall:0'
    self._test_tensor_transformation_params(
        5,
        tensor_quantization_params,
        tensor_name,
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )
    # Output tensor is consumed by the virtual Output op.
    transformation_config = tensor_quantization_params[tensor_name]
    self.assertLen(transformation_config.consumers, 1)
    consumer = transformation_config.consumers[0]
    self.assertEqual(consumer.subgraph_op_id, -1)

  # TODO: b/330770656 - expand the test to cover mixed activation precision.
  def test_generate_config_selective(self):
    # Choose scope regex using Model Explorer
    selective_quantization_recipe = [
        {
            'regex': '.*/dense/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # Equivalent to DRQ.
                'compute_precision': _ComputePrecision.INTEGER,
                'explicit_dequantize': False,
            },
        },
        {
            'regex': '.*/dense_1/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 4,
                    'symmetric': False,
                    'granularity': _QuantGranularity.TENSORWISE,
                },
                # Equivalent to WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
        },
    ]
    self._recipe_manager.load_quantization_recipe(selective_quantization_recipe)
    tensor_quantization_params = (
        self._params_generator.generate_quantization_parameters(
            self._recipe_manager
        )
    )
    # FC weights for scope "dense"
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'arith.constant1',
        [_QuantTransformation.QUANTIZE_TENSOR],
        is_inbounding_tensor=True,
        num_bits=8,
        granularity=_QuantGranularity.CHANNELWISE,
        symmetric=True,
    )

    # FC weights for scope "dense1"
    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'arith.constant',
        [
            _QuantTransformation.ADD_DEQUANTIZE,
        ],
        is_inbounding_tensor=True,
        num_bits=4,
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=False,
    )

  def test_generate_config_edge_cases(self):

    selective_quantization_recipe = [
        # Use the tensor name as scope directly.
        {
            'regex': 'sequential/dense_1/MatMul',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # Equivalent to DRQ.
                'compute_precision': _ComputePrecision.INTEGER,
            },
        },
        # Scope that does not exist in the model.
        {
            'regex': '.*/dense_3/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'num_bits': 4,
                    'symmetric': False,
                    'granularity': _QuantGranularity.TENSORWISE,
                },
                # Equivalent to WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
        },
    ]
    self._recipe_manager.load_quantization_recipe(selective_quantization_recipe)
    tensor_quantization_params = (
        self._params_generator.generate_quantization_parameters(
            self._recipe_manager
        )
    )
    # Only the second FC will be quantized
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'arith.constant1',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )

    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'arith.constant',
        [_QuantTransformation.QUANTIZE_TENSOR],
        is_inbounding_tensor=True,
        num_bits=8,
        granularity=_QuantGranularity.CHANNELWISE,
        symmetric=True,
    )

  @parameterized.parameters(
      (True, _QuantGranularity.CHANNELWISE),
      (True, _QuantGranularity.TENSORWISE),
      (False, _QuantGranularity.CHANNELWISE),
      (False, _QuantGranularity.TENSORWISE),
  )
  def test_generate_config_int8xint8_single_fc(
      self, act_symmetric, channelwise_weight
  ):
    single_fc_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / 'tests/models/single_fc.tflite'
    )
    single_fc_model = tfl_flatbuffer_utils.read_model(single_fc_model_path)
    params_generator_single_fc = params_generator.ParamsGenerator(
        single_fc_model
    )
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=act_symmetric
            ),
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=True, granularity=channelwise_weight
            ),
            # Equivalent to SRQ.
            compute_precision=_ComputePrecision.INTEGER,
        ),
    )

    # Raise error when missing QSVs.
    error_message = 'Model quantization statistics values (QSVs) are required'
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      params_generator_single_fc.generate_quantization_parameters(
          self._recipe_manager
      )

    # Calibrate then quantize
    model_calibrator = calibrator.Calibrator(single_fc_model_path)
    calibration_data = _get_calibration_data(
        _single_fc_model_representative_dataset_gen()
    )
    model_calibrator.calibrate(calibration_data, self._recipe_manager)
    model_qsvs = model_calibrator.get_model_qsvs()
    quant_params = params_generator_single_fc.generate_quantization_parameters(
        self._recipe_manager,
        model_qsvs,
    )
    self.assertLen(quant_params, 4)

    # Input tensor producer (from the virtual input op).
    self._test_tensor_transformation_params(
        -1,  # virtual input op.
        quant_params,
        'serving_default_input_1:0',
        [_QuantTransformation.ADD_DEQUANTIZE],
        num_bits=8,
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=act_symmetric,
        is_inbounding_tensor=False,
    )
    # Input tensor consumer.
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'serving_default_input_1:0',
        [_QuantTransformation.ADD_QUANTIZE],
        num_bits=8,
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=act_symmetric,
        is_inbounding_tensor=True,
    )

    # output tensor producer.
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'StatefulPartitionedCall:0',
        [_QuantTransformation.ADD_DEQUANTIZE],
        num_bits=8,
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=act_symmetric,
        is_inbounding_tensor=False,
    )
    # output tensor consumer (into the virtual output op).
    self._test_tensor_transformation_params(
        -1,  # virtual output op.
        quant_params,
        'StatefulPartitionedCall:0',
        [_QuantTransformation.ADD_QUANTIZE],
        num_bits=8,
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=act_symmetric,
        is_inbounding_tensor=True,
    )

    # weights
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'sequential/dense/MatMul',
        [_QuantTransformation.QUANTIZE_TENSOR],
        num_bits=8,
        granularity=channelwise_weight,
        symmetric=True,
        is_inbounding_tensor=True,
    )

    # bias
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'sequential/dense/BiasAdd/ReadVariableOp',
        [_QuantTransformation.QUANTIZE_TENSOR],
        num_bits=32,
        granularity=channelwise_weight,
        symmetric=True,
        is_inbounding_tensor=True,
    )

  @parameterized.parameters('weight_only', 'DRQ')
  def test_generate_params_buffer_sharing_graphs_succeeds(
      self, the_other_fc_difference
  ):
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/weight_sharing_fcs.tflite'
    )
    model = tfl_flatbuffer_utils.read_model(model_path)
    pg = params_generator.ParamsGenerator(model)
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
            # Equivalent to WEIGHT_ONLY.
            compute_precision=_ComputePrecision.FLOAT,
            explicit_dequantize=True,
        ),
    )
    if the_other_fc_difference == 'DRQ':
      self._recipe_manager.add_quantization_config(
          regex='PartitionedCall_1:0',
          operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
          op_config=qtyping.OpQuantizationConfig(
              # Equivalent to DRQ.
              compute_precision=_ComputePrecision.INTEGER,
          ),
      )
    quant_params = pg.generate_quantization_parameters(
        self._recipe_manager,
    )
    self.assertLen(quant_params, 6)

  @parameterized.named_parameters(
      dict(
          testcase_name='different_quant_config_fc2_no_quant',
          fc_2_num_bits=None,
          expected_tensor_with_buffer_duplication='BatchMatMulV3',
      ),
      dict(
          testcase_name='different_quant_config_fc2_int4',
          fc_2_num_bits=4,
          expected_tensor_with_buffer_duplication='BatchMatMulV3',
      ),
      dict(
          testcase_name='same_quant_config',
          fc_2_num_bits=8,
          expected_tensor_with_buffer_duplication=None,
      ),
  )
  def test_generate_params_marks_correct_buffers_for_duplication_when_distinct_tensors_share_constant_buffer(
      self,
      fc_2_num_bits,
      expected_tensor_with_buffer_duplication,
  ):
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/weight_sharing_fcs.tflite'
    )
    model = tfl_flatbuffer_utils.read_model(model_path)
    pg = params_generator.ParamsGenerator(model)

    # Setup the quantization config for the first FC.
    self._recipe_manager.add_quantization_config(
        regex='PartitionedCall:0',
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8, granularity=qtyping.QuantGranularity.CHANNELWISE
            ),
            compute_precision=_ComputePrecision.INTEGER,
        ),
    )
    # Setup the quantization config for the second FC (weight shared with the
    # first FC).
    if fc_2_num_bits is not None:
      self._recipe_manager.add_quantization_config(
          regex='PartitionedCall_1:0',
          operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
          op_config=qtyping.OpQuantizationConfig(
              weight_tensor_config=_TensorQuantConfig(
                  num_bits=fc_2_num_bits,
                  granularity=qtyping.QuantGranularity.CHANNELWISE,
              ),
              compute_precision=_ComputePrecision.INTEGER,
          ),
      )

    quant_params = pg.generate_quantization_parameters(
        self._recipe_manager,
    )
    self.assertLen(quant_params, 6)

    # Check that the expected tensor has buffer duplication transformation as
    # the first one to be applied. And no other tensor has buffer duplication
    # transformation at all.
    for tensor_name in quant_params:
      if tensor_name == expected_tensor_with_buffer_duplication:
        self.assertIsNotNone(quant_params[tensor_name].consumers)
        for consumer in quant_params[tensor_name].consumers:
          self.assertNotEmpty(consumer.transformations)
          self.assertEqual(
              consumer.transformations[0],
              _QTransf.DUPLICATE_BUFFER,
          )
          self.assertNotIn(
              _QTransf.DUPLICATE_BUFFER, consumer.transformations[1:]
          )
      elif quant_params[tensor_name].consumers is not None:
        for consumer in quant_params[tensor_name].consumers:
          self.assertNotIn(_QTransf.DUPLICATE_BUFFER, consumer.transformations)

  def _get_fc_recipe_entry(self, regex: str, num_bits: int):
    return {
        'regex': regex,
        'operation': 'FULLY_CONNECTED',
        'algorithm_key': 'min_max_uniform_quantize',
        'op_config': {
            'weight_tensor_config': {
                'num_bits': num_bits,
                'symmetric': True,
                'granularity': 'CHANNELWISE',
                'dtype': 'INT',
                'block_size': 0,
            },
            'compute_precision': 'INTEGER',
            'explicit_dequantize': False,
            'skip_checks': False,
            'min_weight_elements': 0,
        },
    }

  @parameterized.named_parameters(
      dict(
          testcase_name='fc1_quant_fc2_no_quant',
          fc1_num_bits=8,
          fc2_num_bits=None,
      ),
      dict(
          testcase_name='fc1_no_quant_fc2_quant',
          fc1_num_bits=None,
          fc2_num_bits=8,
      ),
      dict(
          testcase_name='fc1_quant_fc2_quant_different_params',
          fc1_num_bits=8,
          fc2_num_bits=4,
      ),
  )
  def test_generate_params_marks_correct_buffers_tensors_for_duplication(
      self,
      fc1_num_bits,
      fc2_num_bits,
  ):
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/constant_tensor_and_buffer_only_sharing_weight_fcs.tflite',
    )
    model = tfl_flatbuffer_utils.read_model(model_path)
    pg = params_generator.ParamsGenerator(model)
    
    sig1_fc1_regex = 'BatchMatMulV3;'
    sig1_fc2_regex = 'PartitionedCall:0;'
    recipe = []
    if fc1_num_bits is not None:
      recipe.append(self._get_fc_recipe_entry(sig1_fc1_regex, fc1_num_bits))
    if fc2_num_bits is not None:
      recipe.append(self._get_fc_recipe_entry(sig1_fc2_regex, fc2_num_bits))
    self._recipe_manager.load_quantization_recipe(recipe)
    quant_params = pg.generate_quantization_parameters(self._recipe_manager)

    expected_tensor = 'arith.constant'
    consumers = quant_params[expected_tensor].consumers
    self.assertLen(consumers, 2)

    # Check FC1 transformations.
    if fc1_num_bits is None:
      fc1_quant_transformation = _QTransf.NO_QUANTIZE
    else:
      fc1_quant_transformation = _QTransf.QUANTIZE_TENSOR
    self.assertEqual(
        consumers[0].transformations,
        [
            _QTransf.DUPLICATE_TENSOR,
            _QTransf.DUPLICATE_BUFFER,
            fc1_quant_transformation,
        ],
    )
    # Check FC2 transformations.
    if fc2_num_bits is None:
      fc2_quant_transformation = _QTransf.NO_QUANTIZE
    else:
      fc2_quant_transformation = _QTransf.QUANTIZE_TENSOR
    self.assertEqual(
        consumers[1].transformations,
        [
            _QTransf.DUPLICATE_TENSOR,
            _QTransf.DUPLICATE_BUFFER,
            fc2_quant_transformation,
        ],
    )
    # Check that no other tensor has tensor or buffer duplication
    # transformations.
    for tensor_name, params in quant_params.items():
      if tensor_name == expected_tensor:
        continue
      for consumer in params.consumers:
        self.assertNotIn(_QTransf.DUPLICATE_TENSOR, consumer.transformations)
        self.assertNotIn(_QTransf.DUPLICATE_BUFFER, consumer.transformations)

  def test_generate_params_returns_valid_results_when_multiple_tensor_duplication_for_one_buffer(
      self,
  ):
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/constant_tensor_and_buffer_only_sharing_weight_fcs.tflite',
    )
    model = tfl_flatbuffer_utils.read_model(model_path)
    pg = params_generator.ParamsGenerator(model)

    sig1_fc1_regex = 'BatchMatMulV3;'
    sig1_fc2_regex = 'PartitionedCall:0;'
    sig2_fc1_regex = 'BatchMatMulV31;'
    sig2_fc2_regex = 'PartitionedCall_1:0;'
    recipe = [
        self._get_fc_recipe_entry(sig1_fc1_regex, num_bits=8),
        self._get_fc_recipe_entry(sig1_fc2_regex, num_bits=4),
        self._get_fc_recipe_entry(sig2_fc1_regex, num_bits=8),
        self._get_fc_recipe_entry(sig2_fc2_regex, num_bits=4),
    ]
    self._recipe_manager.load_quantization_recipe(recipe)
    quant_params = pg.generate_quantization_parameters(self._recipe_manager)

    # Check transformations for sig1.
    sig1_expected_tensor = 'arith.constant'
    sig1_consumers = quant_params[sig1_expected_tensor].consumers
    self.assertLen(sig1_consumers, 2)
    sig1_expected_transformations = [
        _QTransf.DUPLICATE_TENSOR,
        _QTransf.DUPLICATE_BUFFER,
        _QTransf.QUANTIZE_TENSOR,
    ]
    for sig1_consumer in sig1_consumers:
      self.assertEqual(
          sig1_consumer.transformations,
          sig1_expected_transformations,
      )
    # Check transformations for sig2.
    sig2_expected_tensor = 'arith.constant1'
    sig2_consumers = quant_params[sig2_expected_tensor].consumers
    self.assertLen(sig2_consumers, 2)
    sig2_expected_transformations = [
        _QTransf.DUPLICATE_TENSOR,
        _QTransf.QUANTIZE_TENSOR,
    ]
    for sig2_consumer in sig2_consumers:
      self.assertEqual(
          sig2_consumer.transformations,
          sig2_expected_transformations,
      )
    # Check that no other tensor has tensor or buffer duplication
    # transformations.
    for tensor_name, params in quant_params.items():
      if tensor_name in [sig1_expected_tensor, sig2_expected_tensor]:
        continue
      for consumer in params.consumers:
        self.assertNotIn(_QTransf.DUPLICATE_TENSOR, consumer.transformations)
        self.assertNotIn(_QTransf.DUPLICATE_BUFFER, consumer.transformations)

  @parameterized.named_parameters(
      dict(
          testcase_name='producer_incompatible',
          param1=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[_QTransf.ADD_DEQUANTIZE],
                  parameters=_PARAMS_8BIT,
              ),
              consumers=_get_test_consumers(
                  transformations_per_consumer=[
                      [_QTransf.ADD_QUANTIZE],
                      [_QTransf.ADD_QUANTIZE, _QTransf.ADD_DEQUANTIZE],
                      [_QTransf.ADD_QUANTIZE, _QTransf.ADD_DEQUANTIZE],
                      [_QTransf.NO_QUANTIZE],
                  ],
                  params_per_consumer=[_PARAMS_8BIT] * 4,
              ),
          ),
          param2=qtyping.TensorTransformationParams(
              tensor_name='tfl.other_quantize',
              producer=qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[_QTransf.NO_QUANTIZE],
                  parameters=_PARAMS_8BIT,
              ),
              consumers=_get_test_consumers(
                  transformations_per_consumer=[
                      [_QTransf.ADD_QUANTIZE],
                      [_QTransf.ADD_QUANTIZE, _QTransf.ADD_DEQUANTIZE],
                      [_QTransf.ADD_QUANTIZE, _QTransf.ADD_DEQUANTIZE],
                  ],
                  params_per_consumer=[_PARAMS_8BIT] * 4,
              ),
          ),
          expected=False,
      ),
      dict(
          testcase_name='compatible',
          param1=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=None,
              consumers=_get_test_consumers(
                  transformations_per_consumer=[
                      [_QTransf.ADD_QUANTIZE],
                      [_QTransf.NO_QUANTIZE, _QTransf.ADD_QUANTIZE],
                      [_QTransf.NO_QUANTIZE],
                  ],
                  params_per_consumer=[_PARAMS_8BIT] * 4,
              ),
          ),
          param2=qtyping.TensorTransformationParams(
              tensor_name='tfl.other_quantize',
              producer=None,
              consumers=_get_test_consumers(
                  transformations_per_consumer=[
                      [_QTransf.ADD_QUANTIZE],
                      [_QTransf.ADD_QUANTIZE, _QTransf.ADD_DEQUANTIZE],
                      [_QTransf.ADD_QUANTIZE, _QTransf.ADD_DEQUANTIZE],
                      [_QTransf.ADD_QUANTIZE],
                  ],
                  params_per_consumer=[_PARAMS_8BIT] * 4,
              ),
          ),
          expected=True,
      ),
      dict(
          testcase_name='compatible_no_numeric_check',
          param1=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=None,
              consumers=_get_test_consumers(
                  transformations_per_consumer=[
                      [_QTransf.ADD_QUANTIZE],
                      [_QTransf.ADD_QUANTIZE],
                  ],
                  params_per_consumer=[
                      qtyping.UniformQuantParams(
                          8, None, np.array([0.00028806]), np.array([0])
                      ),
                      qtyping.UniformQuantParams(
                          8, None, np.array([0.00027501]), np.array([0])
                      ),
                  ],
              ),
          ),
          param2=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=None,
              consumers=_get_test_consumers(
                  transformations_per_consumer=[
                      [_QTransf.ADD_QUANTIZE],
                      [_QTransf.ADD_QUANTIZE],
                  ],
                  params_per_consumer=[
                      qtyping.UniformQuantParams(
                          8, None, np.array([0.00028806]), np.array([0])
                      ),
                      qtyping.UniformQuantParams(
                          8, None, np.array([0.00027501]), np.array([0])
                      ),
                  ],
              ),
          ),
          expected=True,
      ),
  )
  def test__are_self_compatible_tensors_compatible_to_each_other(
      self, param1, param2, expected
  ):
    self.assertEqual(
        params_generator._are_self_compatible_tensors_compatible_to_each_other(
            param1, param2
        ),
        expected,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='consumer_incompatible',
          params=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[_QTransf.NO_QUANTIZE],
                  parameters=_PARAMS_8BIT,
              ),
              consumers=_get_test_consumers(
                  transformations_per_consumer=[
                      [_QTransf.ADD_QUANTIZE],
                      [_QTransf.ADD_QUANTIZE, _QTransf.ADD_DEQUANTIZE],
                      [_QTransf.ADD_QUANTIZE, _QTransf.ADD_DEQUANTIZE],
                      [_QTransf.QUANTIZE_TENSOR],
                  ],
                  params_per_consumer=[_PARAMS_8BIT] * 4,
              ),
          ),
          expected=False,
      ),
      dict(
          testcase_name='compatible',
          params=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=None,
              consumers=_get_test_consumers(
                  transformations_per_consumer=[
                      [_QTransf.ADD_QUANTIZE, _QTransf.ADD_DEQUANTIZE],
                      [_QTransf.ADD_QUANTIZE],
                      [_QTransf.NO_QUANTIZE, _QTransf.ADD_QUANTIZE],
                      [_QTransf.NO_QUANTIZE],
                  ],
                  params_per_consumer=[_PARAMS_8BIT] * 4,
              ),
          ),
          expected=True,
      ),
      dict(
          testcase_name='compatible_no_numeric_check',
          params=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=None,
              consumers=_get_test_consumers(
                  transformations_per_consumer=[
                      [_QTransf.ADD_QUANTIZE],
                      [_QTransf.ADD_QUANTIZE],
                  ],
                  params_per_consumer=[
                      qtyping.UniformQuantParams(
                          8, None, np.array([0.00028806]), np.array([0])
                      ),
                      qtyping.UniformQuantParams(
                          8, None, np.array([0.00027501]), np.array([0])
                      ),
                  ],
              ),
          ),
          expected=True,
      ),
  )
  def test__are_tensor_consumer_params_compatible(self, params, expected):
    self.assertEqual(
        params_generator._are_tensor_consumer_params_compatible(params),
        expected,
    )

  def test_model_with_duplicated_tensor_names_fails(self):
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/duplicated_tensor_names.tflite'
    )
    model = tfl_flatbuffer_utils.read_model(model_path)
    error_message = 'Tensor name test_same_name is not unique in the model.'
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      params_generator.ParamsGenerator(model)

  def test_quantize_integer_input_output(self):
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/single_transpose_int32.tflite'
    )
    model = tfl_flatbuffer_utils.read_model(model_path)
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=False
            ),
            weight_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
            # Equivalent to SRQ.
            compute_precision=_ComputePrecision.INTEGER,
        ),
    )
    pg = params_generator.ParamsGenerator(model)

    # Calibrate then quantize.
    model_calibrator = calibrator.Calibrator(model_path)
    calibration_data = _get_calibration_data(
        _int_transpose_model_representative_dataset_gen()
    )
    model_calibrator.calibrate(calibration_data, self._recipe_manager)
    model_qsvs = model_calibrator.get_model_qsvs()
    quant_params = pg.generate_quantization_parameters(
        self._recipe_manager,
        model_qsvs,
    )
    self.assertLen(quant_params, 3)

    self._test_tensor_transformation_params(
        -1,  # virtual input op.
        quant_params,
        'serving_default_input_2:0',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )
    # Input tensor consumer.
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'serving_default_input_2:0',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )

    # Output tensor producer.
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'PartitionedCall:0',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )
    # output tensor consumer (into the virtual output op).
    self._test_tensor_transformation_params(
        -1,  # virtual output op.
        quant_params,
        'PartitionedCall:0',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )

    # perm
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'sequential_1/permute_1/transpose/perm',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )

  def _test_tensor_transformation_params(
      self,
      subgraph_op_id,
      quant_params,
      tensor_name,
      transformations,
      is_inbounding_tensor,
      num_bits=8,
      granularity=_QuantGranularity.TENSORWISE,
      symmetric=True,
      quantized_dimension=0,
  ):
    """Helper function to test tensor transformation parameters are correct."""
    self.assertIn(tensor_name, quant_params)
    transformation_config = quant_params[tensor_name]
    self.assertEqual(transformation_config.tensor_name, tensor_name)
    if is_inbounding_tensor:
      self.assertLen(transformation_config.consumers, 1)
      op_config = transformation_config.consumers[0]
    else:
      op_config = transformation_config.producer
    self.assertIsNotNone(op_config)
    self.assertEqual(op_config.subgraph_op_id, subgraph_op_id)
    self.assertSequenceEqual(op_config.transformations, transformations)
    if transformations == [_QuantTransformation.NO_QUANTIZE]:
      self.assertIsNone(op_config.parameters)
    else:
      quantization_params = op_config.parameters
      self.assertIsNotNone(quantization_params)
      if granularity is _QuantGranularity.CHANNELWISE:
        self.assertEqual(
            quantization_params.quantized_dimension, quantized_dimension
        )
      else:
        self.assertIsNone(quantization_params.quantized_dimension)
      self.assertEqual(quantization_params.num_bits, num_bits)
      if symmetric:
        self.assertEqual(np.sum(abs(quantization_params.zero_point)), 0)
      else:
        self.assertEqual(
            len(quantization_params.scale),
            len(quantization_params.zero_point),
        )


class ParamsGeneratorAlreadyQuantizedModelTest(absltest.TestCase):

  def test_check_is_float_model_succeeds_when_model_is_float(self):
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/conv_fc_mnist.tflite'
    )
    test_model = tfl_flatbuffer_utils.read_model(test_model_path)
    _ = params_generator.ParamsGenerator(test_model)

  def test_check_is_quantized_model_succeeds_when_model_is_quantized(self):
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/mnist_quantized.tflite'
    )
    test_model = tfl_flatbuffer_utils.read_model(test_model_path)
    _ = params_generator.ParamsGenerator(test_model)


if __name__ == '__main__':
  absltest.main()
