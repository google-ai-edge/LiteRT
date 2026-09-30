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

import os
import io
from litert_quantizer import algorithm_manager
from litert_quantizer import calibrator
from litert_quantizer import default_policy
from litert_quantizer import model_validator
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import recipe_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils

_ComputePrecision = qtyping.ComputePrecision
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_TensorDataType = qtyping.TensorDataType
_AlgorithmName = quantizer.AlgorithmName
_CalibrationMode = calibrator.CalibrationMode

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('')
_MULTI_SIGNATURE_CALIBRATION_DATASET = {
    'add': [{'x': np.array([2.0], dtype=np.float32)}],
    'multiply': [{'x': np.array([1.0], dtype=np.float32)}],
}
_RNG = np.random.default_rng(66)


def _get_calibration_data(num_samples: int = 16):
  calibration_samples = []
  for _ in range(num_samples):
    calibration_samples.append(
        {'conv2d_input': _RNG.uniform(size=(1, 28, 28, 1)).astype(np.float32)}
    )
  calibration_data = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
  }
  return calibration_data


def _is_all_signature_defs_inputs_float(model_content: bytes):
  tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(model_content)
  for signature_key in tfl_interpreter.get_signature_list():
    input_details = tfl_interpreter.get_signature_runner(
        signature_key
    ).get_input_details()
    for tensor_details in input_details.values():
      if tensor_details['dtype'] != np.float32:
        return False
  return True


def _is_all_signature_defs_outputs_float(model_content: bytes):
  tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(model_content)
  for signature_key in tfl_interpreter.get_signature_list():
    output_details = tfl_interpreter.get_signature_runner(
        signature_key
    ).get_output_details()
    for tensor_details in output_details.values():
      if tensor_details['dtype'] != np.float32:
        return False
  return True


class QuantizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp_save_path = self.create_tempdir().full_path
    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/conv_fc_mnist.tflite'
    )
    self._test_recipe = recipe_utils.resolve_recipe('default_af32w8float')
    self._quantizer = quantizer.Quantizer(
        self._test_model_path, self._test_recipe
    )

  def test_update_quantization_recipe_succeeds(self):
    self._quantizer.load_quantization_recipe(self._test_recipe)
    scope_regex = '.*/Dense/.*'
    new_op_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=_TensorQuantConfig(num_bits=4, symmetric=True),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    self._quantizer.update_quantization_recipe(
        regex=scope_regex,
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=new_op_config,
    )
    updated_recipe = self._quantizer.get_quantization_recipe()
    self.assertLen(updated_recipe, 2)

    added_config = updated_recipe[-1]
    self.assertEqual(added_config['regex'], scope_regex)
    self.assertEqual(
        added_config['op_config']['compute_precision'],
        new_op_config.compute_precision,
    )

  def test_add_dynamic_config_succeeds(self):
    self._quantizer.load_quantization_recipe(self._test_recipe)
    scope_regex = '.*/Dense/.*'
    self._quantizer.add_dynamic_config(
        regex=scope_regex,
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        num_bits=8,
    )
    updated_recipe = self._quantizer.get_quantization_recipe()
    self.assertLen(updated_recipe, 2)

    added_config = updated_recipe[-1]
    self.assertEqual(added_config['regex'], scope_regex)
    self.assertEqual(
        added_config['op_config']['compute_precision'],
        qtyping.ComputePrecision.INTEGER,
    )
    self.assertFalse(added_config['op_config']['explicit_dequantize'])
    self.assertEqual(
        added_config['op_config']['weight_tensor_config']['num_bits'], 8
    )

  def test_add_weight_only_config_succeeds(self):
    self._quantizer.load_quantization_recipe(self._test_recipe)
    scope_regex = '.*/Dense/.*'
    self._quantizer.add_weight_only_config(
        regex=scope_regex,
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        num_bits=4,
    )
    updated_recipe = self._quantizer.get_quantization_recipe()
    self.assertLen(updated_recipe, 2)

    added_config = updated_recipe[-1]
    self.assertEqual(added_config['regex'], scope_regex)
    self.assertEqual(
        added_config['op_config']['compute_precision'],
        qtyping.ComputePrecision.FLOAT,
    )
    self.assertTrue(added_config['op_config']['explicit_dequantize'])
    self.assertEqual(
        added_config['op_config']['weight_tensor_config']['num_bits'], 4
    )

  def test_add_static_config_succeeds(self):
    self._quantizer.load_quantization_recipe(self._test_recipe)
    scope_regex = '.*/Dense/.*'
    self._quantizer.add_static_config(
        regex=scope_regex,
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        activation_num_bits=8,
        weight_num_bits=4,
    )
    updated_recipe = self._quantizer.get_quantization_recipe()
    self.assertLen(updated_recipe, 2)

    added_config = updated_recipe[-1]
    self.assertEqual(added_config['regex'], scope_regex)
    self.assertEqual(
        added_config['op_config']['compute_precision'],
        qtyping.ComputePrecision.INTEGER,
    )
    self.assertFalse(added_config['op_config']['explicit_dequantize'])
    self.assertEqual(
        added_config['op_config']['activation_tensor_config']['num_bits'], 8
    )
    self.assertEqual(
        added_config['op_config']['weight_tensor_config']['num_bits'], 4
    )

  def test_load_quantization_recipe_succeeds(self):
    qt = quantizer.Quantizer(self._test_model_path, None)
    qt.load_quantization_recipe(self._test_recipe)
    self.assertEqual(qt.get_quantization_recipe(), self._test_recipe)

    # Load a different recipe.
    new_recipe = recipe_utils.resolve_recipe('dynamic_wi8_afp32')
    qt.load_quantization_recipe('dynamic_wi8_afp32')
    self.assertEqual(qt.get_quantization_recipe(), new_recipe)

  @parameterized.product(
      recipe_name=[
          'default_a8w8',
          'default_a16w8',
      ],
      mode=[
          _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
          _CalibrationMode.CALIBRATION_PROFILER_BASED,
      ],
  )
  def test_calibrate_required_recipe_succeeds(self, recipe_name, mode):
    self._quantizer.load_quantization_recipe(recipe_name)
    self.assertTrue(self._quantizer.need_calibration)
    # Calibrate with empty state.
    calib_data = _get_calibration_data()
    calibration_result = self._quantizer.calibrate(calib_data, mode=mode)
    self.assertLen(calibration_result, 7)

  @parameterized.product(
      recipe_name=[
          'default_a8w8',
          'default_a16w8',
      ],
      mode=[
          _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
          _CalibrationMode.CALIBRATION_PROFILER_BASED,
      ],
  )
  def test_reloaded_calibration_succeeds(self, recipe_name, mode):
    self._quantizer.load_quantization_recipe(recipe_name)
    calib_data = _get_calibration_data()
    calibration_result = self._quantizer.calibrate(calib_data, mode=mode)
    # Load and calibrate again.
    updated_calibration_result = self._quantizer.calibrate(
        calib_data, previous_calibration_result=calibration_result, mode=mode
    )
    self.assertLen(updated_calibration_result, 7)
    self.assertNotEqual(
        calibration_result['StatefulPartitionedCall:0'],
        updated_calibration_result['StatefulPartitionedCall:0'],
    )

  def test_calibrate_unsupported_mode_raises_error(self):
    self._quantizer.load_quantization_recipe('default_a8w8')
    with self.assertRaisesRegex(
        ValueError, r'Unsupported calibration mode: .*INFERENCE'
    ):
      self._quantizer.calibrate(
          _get_calibration_data(), mode=_CalibrationMode.INFERENCE
      )

  @parameterized.product(
      recipe_name=[
          'dynamic_legacy_wi8_afp32',
          'dynamic_wi8_afp32',
          'default_af32w8float',
      ],
      mode=[
          _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
          _CalibrationMode.CALIBRATION_PROFILER_BASED,
      ],
  )
  def test_calibrate_nonrequired_recipe_succeeds(self, recipe_name, mode):
    self._quantizer.load_quantization_recipe(recipe_name)
    self.assertFalse(self._quantizer.need_calibration)
    # Empty calibration result if no calibration is required.
    calibration_result = self._quantizer.calibrate(
        _get_calibration_data(), mode=mode
    )
    self.assertEmpty(calibration_result)

  def test_quantize_no_calibration_succeeds(self):
    self._quantizer.load_quantization_recipe(self._test_recipe)
    self.assertIsNone(self._quantizer._result.quantized_model)
    quant_result = self._quantizer.quantize()
    self.assertEqual(quant_result.recipe, self._test_recipe)
    self.assertIsNotNone(quant_result.quantized_model)

  @parameterized.product(
      recipe_name=[
          'default_a8w8',
          'default_a16w8',
      ],
      mode=[
          _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
          _CalibrationMode.CALIBRATION_PROFILER_BASED,
      ],
  )
  def test_quantize_calibration_needed_succeeds(self, recipe_name, mode):
    recipe = recipe_utils.resolve_recipe(recipe_name)

    self._quantizer.load_quantization_recipe(recipe_name)
    self.assertTrue(self._quantizer.need_calibration)
    calibration_result = self._quantizer.calibrate(
        _get_calibration_data(), mode=mode
    )

    self.assertIsNone(self._quantizer._result.quantized_model)
    quant_result = self._quantizer.quantize(calibration_result)
    self.assertEqual(quant_result.recipe, recipe)
    self.assertIsNotNone(quant_result.quantized_model)

  @parameterized.parameters(
      'default_a8w8',
      'default_a16w8',
  )
  def test_quantize_calibration_needed_raise_error(self, recipe_name):
    self._quantizer.load_quantization_recipe(recipe_name)
    self.assertTrue(self._quantizer.need_calibration)
    error_message = (
        'Model quantization statistics values (QSVs) are required for the input'
        ' recipe.'
    )
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      self._quantizer.quantize()

  def test_quantize_no_recipe_raise_error(self):
    qt = quantizer.Quantizer(self._test_model_path, None)
    error_message = 'Can not quantize without a quantization recipe.'
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      qt.quantize()

  def test_save_succeeds(self):
    model_name = 'test_model'
    self._quantizer.load_quantization_recipe(self._test_recipe)
    result = self._quantizer.quantize()
    result.save(self._tmp_save_path, model_name)
    saved_recipe_path = str(
        pathlib.Path(self._tmp_save_path) / (model_name + '_recipe.json')
    )
    saved_recipe = recipe_utils.resolve_recipe(saved_recipe_path)
    self.assertEqual(saved_recipe, self._test_recipe)

  def test_saved_legacy_recipe_lacks_block_size(self):
    model_name = 'test_model'
    legacy_recipe_name = 'dynamic_legacy_wi8_afp32'
    self._quantizer.load_quantization_recipe(legacy_recipe_name)
    result = self._quantizer.quantize()
    result.save(self._tmp_save_path, model_name)
    saved_recipe_path = str(
        pathlib.Path(self._tmp_save_path) / (model_name + '_recipe.json')
    )
    saved_recipe = recipe_utils.resolve_recipe(saved_recipe_path)
    legacy_recipe = recipe_utils.resolve_recipe(legacy_recipe_name)

    self.assertNotEqual(saved_recipe, legacy_recipe)

    # Verify that the default test recipe contains 'block_size'.
    has_block_size = False
    for config in legacy_recipe:
      op_config = config.get('op_config')
      if op_config:
        weight_config = op_config.get('weight_tensor_config')
        if weight_config and 'block_size' in weight_config:
          has_block_size = True
          break
    self.assertTrue(has_block_size)

    # Verify that the saved recipe does not have 'block_size'.
    for config in saved_recipe:
      op_config = config.get('op_config')
      if op_config:
        weight_config = op_config.get('weight_tensor_config')
        if weight_config:
          self.assertNotIn('block_size', weight_config)

  def test_save_no_quantize_raise_error(self):
    error_message = 'No quantized model to save.'
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      self._quantizer._result.save(self._tmp_save_path, 'test_model')

  def test_export_model_succeeds(self):
    model_name = 'exported_model'
    self._quantizer.load_quantization_recipe(self._test_recipe)
    result = self._quantizer.quantize()

    exported_model_path = str(
        pathlib.Path(self._tmp_save_path) / (model_name + '.tflite')
    )
    self.assertFalse(pathlib.Path(exported_model_path).exists())
    result.export_model(exported_model_path)
    self.assertTrue(pathlib.Path(exported_model_path).exists())

  def test_compare_succeeds(self):
    self._quantizer.quantize()
    validation_result = self._quantizer.validate()
    validation_result = validation_result.get_signature_comparison_result()
    self.assertIsNotNone(validation_result)
    self.assertIn(
        'sequential/dense_1/MatMul', validation_result.intermediate_tensors
    )

  def test_validate_output_tensors_only_succeeds(self):
    self._quantizer.quantize()
    validation_result = self._quantizer.validate(
        validate_output_tensors_only=True
    )
    validation_result = validation_result.get_signature_comparison_result()
    self.assertIsNotNone(validation_result)
    self.assertEmpty(validation_result.input_tensors)
    self.assertEmpty(validation_result.constant_tensors)
    self.assertEmpty(validation_result.intermediate_tensors)
    self.assertNotEmpty(validation_result.output_tensors)
    self.assertIn('StatefulPartitionedCall:0', validation_result.output_tensors)

  def test_validate_with_quantized_model_arg_succeeds(self):
    self._quantizer.quantize()
    quantized_model = self._quantizer._result.quantized_model
    self.assertIsNotNone(quantized_model)

    new_quantizer = quantizer.Quantizer(
        self._test_model_path, previous_quantized_model=quantized_model
    )
    validation_result = new_quantizer.validate()
    validation_result = validation_result.get_signature_comparison_result()
    self.assertIsNotNone(validation_result)
    self.assertIn(
        'sequential/dense_1/MatMul', validation_result.intermediate_tensors
    )

  def test_load_custom_policies_succeeds(self):

    test_op_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=_TensorQuantConfig(num_bits=4, symmetric=True),
        compute_precision=_ComputePrecision.INTEGER,
    )

    # Check if the quant config is supported by default policy.
    self._quantizer.update_quantization_recipe(
        regex='.*/Dense/.*',
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        op_config=test_op_config,
    )

    # Check if the quant config fails on dummy policy.
    dummy_policy_path = test_utils.get_path_to_datafile(
        'policies/dummy_config_policy.json'
    )
    self._quantizer.load_config_policy(dummy_policy_path)
    with self.assertRaisesRegex(
        ValueError, 'Unsupported op for .*FULLY_CONNECTED'
    ):
      self._quantizer.update_quantization_recipe(
          regex='.*/Dense/.*',
          operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
          algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          op_config=test_op_config,
      )

    # Check if the quant config is supported by example policy in *.json file.
    default_policy_path = test_utils.get_path_to_datafile(
        'policies/example_config_policy.json'
    )
    self._quantizer.load_config_policy(default_policy_path)
    self._quantizer.update_quantization_recipe(
        regex='.*/Dense/.*',
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=test_op_config,
    )

    # Restore default policy to avoid side effects on other tests.
    algorithm_manager.register_config_check_policy_func(
        algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        default_policy.DEFAULT_CONFIG_CHECK_POLICY,
    )

  @parameterized.parameters(
      _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      _CalibrationMode.CALIBRATION_PROFILER_BASED,
  )
  def test_two_pass_quantization_with_conv_and_fc_succeeds(self, mode):
    float_model_path = self._test_model_path

    drq_recipe_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'recipes/dynamic_wi8_afp32_hadamard_recipe.json'
    )
    drq_quantizer = quantizer.Quantizer(float_model_path)
    drq_quantizer.load_quantization_recipe(drq_recipe_path)
    drq_result = drq_quantizer.quantize()
    drq_model_path = str(pathlib.Path(self._tmp_save_path) / 'drq_model.tflite')
    drq_result.export_model(drq_model_path)

    srq_recipe_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / 'recipes/default_a8w8_recipe.json'
    )
    srq_quantizer = quantizer.Quantizer(drq_model_path)
    srq_quantizer.load_quantization_recipe(srq_recipe_path)
    representative_dataset = (
        tfl_interpreter_utils.create_random_normal_input_data(
            drq_model_path, num_samples=1
        )
    )
    calibration_result = srq_quantizer.calibrate(
        representative_dataset, mode=mode
    )
    srq_result = srq_quantizer.quantize(calibration_result)
    srq_model_path = str(pathlib.Path(self._tmp_save_path) / 'srq_model.tflite')
    srq_result.export_model(srq_model_path)


class QuantizerBytearrayInputs(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/conv_fc_mnist.tflite'
    )
    with open(self._test_model_path, 'rb') as f:
      model_content = bytearray(f.read())
    self._test_recipe = recipe_utils.resolve_recipe('default_af32w8float')
    self._quantizer = quantizer.Quantizer(model_content, self._test_recipe)

  def test_quantize_compare_succeeds(self):
    quant_result = self._quantizer.quantize()
    self.assertEqual(quant_result.recipe, self._test_recipe)
    self.assertIsNotNone(quant_result.quantized_model)
    validation_result = self._quantizer.validate()
    validation_result = validation_result.get_signature_comparison_result()
    self.assertIsNotNone(validation_result)
    self.assertIn(
        'sequential/dense_1/MatMul', validation_result.intermediate_tensors
    )

  def test_compare_succeeds(self):
    self._quantizer.quantize()
    validation_result = self._quantizer.validate()
    validation_result = validation_result.get_signature_comparison_result()
    self.assertIsNotNone(validation_result)
    self.assertIn(
        'sequential/dense_1/MatMul', validation_result.intermediate_tensors
    )


# TODO: b/364974841 - Add more tests after multiple signatures are supported
# for calibrate and quantize.
class QuantizerMultiSignatureModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp_save_path = self.create_tempdir().full_path
    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/two_signatures.tflite'
    )
    self._test_recipe = recipe_utils.resolve_recipe('default_a8w8')
    self._calibration_result = {
        'add_x:0': {'min': -2.0, 'max': 2.0},
        'PartitionedCall:0': {'min': -8.0, 'max': 12.0},
        'multiply_x:0': {'min': -2.0, 'max': 2.0},
        'PartitionedCall_1:0': {'min': -20.0, 'max': 20.0},
    }
    self._quantizer = quantizer.Quantizer(
        self._test_model_path, self._test_recipe
    )

  @parameterized.named_parameters(
      ('default_random_data', None),
      ('specific_data', _MULTI_SIGNATURE_CALIBRATION_DATASET),
  )
  def test_validate_multiple_signatures_succeeds(self, test_data):
    self._quantizer.quantize(self._calibration_result)
    validation_result = self._quantizer.validate(test_data)
    available_signatures = validation_result.available_signature_keys()
    self.assertLen(available_signatures, 2)

    add_result = validation_result.get_signature_comparison_result('add')
    self.assertIn(quantizer.ValidationErrorMetric.MSE, add_result.error_metrics)
    self.assertIn('add_x:0', add_result.input_tensors)
    self.assertIn('PartitionedCall:0', add_result.output_tensors)
    self.assertIn('Add/y', add_result.constant_tensors)
    self.assertEmpty(add_result.intermediate_tensors)

    mul_result = validation_result.get_signature_comparison_result('multiply')
    self.assertIn(quantizer.ValidationErrorMetric.MSE, mul_result.error_metrics)
    self.assertIn('multiply_x:0', mul_result.input_tensors)
    self.assertIn('PartitionedCall_1:0', mul_result.output_tensors)
    self.assertIn('Mul/y', mul_result.constant_tensors)
    self.assertEmpty(mul_result.intermediate_tensors)

  def test_validate_add_signature_succeeds(self):
    test_data = {'add': [{'x': np.array([2.0]).astype(np.float32)}]}
    self._quantizer.quantize(self._calibration_result)
    validation_result = self._quantizer.validate(test_data)
    available_signatures = validation_result.available_signature_keys()
    self.assertLen(available_signatures, 1)
    self.assertIn('add', available_signatures)
    add_result = validation_result.get_signature_comparison_result('add')
    self.assertIn(quantizer.ValidationErrorMetric.MSE, add_result.error_metrics)
    self.assertIn('add_x:0', add_result.input_tensors)
    self.assertIn('PartitionedCall:0', add_result.output_tensors)
    self.assertIn('Add/y', add_result.constant_tensors)
    self.assertEmpty(add_result.intermediate_tensors)

  def test_validate_multiply_signature_succeeds(self):
    test_data = {'multiply': [{'x': np.array([1.0]).astype(np.float32)}]}
    self._quantizer.quantize(self._calibration_result)
    validation_result = self._quantizer.validate(test_data)
    available_signatures = validation_result.available_signature_keys()
    self.assertLen(available_signatures, 1)
    self.assertIn('multiply', available_signatures)
    mul_result = validation_result.get_signature_comparison_result('multiply')
    self.assertIn(quantizer.ValidationErrorMetric.MSE, mul_result.error_metrics)
    self.assertIn('multiply_x:0', mul_result.input_tensors)
    self.assertIn('PartitionedCall_1:0', mul_result.output_tensors)
    self.assertIn('Mul/y', mul_result.constant_tensors)
    self.assertEmpty(mul_result.intermediate_tensors)

  @parameterized.parameters(
      _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      _CalibrationMode.CALIBRATION_PROFILER_BASED,
  )
  def test_validate_quantize_after_calibration_succeeds(self, mode):
    calib_result = self._quantizer.calibrate(
        _MULTI_SIGNATURE_CALIBRATION_DATASET, mode=mode
    )
    self._quantizer.quantize(calib_result)
    validation_result = self._quantizer.validate(
        _MULTI_SIGNATURE_CALIBRATION_DATASET
    )
    available_signatures = validation_result.available_signature_keys()
    self.assertLen(available_signatures, 2)

  @parameterized.parameters(
      _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      _CalibrationMode.CALIBRATION_PROFILER_BASED,
  )
  def test_constant_buffer_shared_by_tensors_with_different_quantization_params_succeeds(
      self, mode
  ):
    recipe = [
        dict({
            'regex': '.*',
            'operation': 'ADD',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'activation_tensor_config': {
                    'num_bits': 8,
                    'symmetric': False,
                    'granularity': 'TENSORWISE',
                    'dtype': 'INT',
                },
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': 'CHANNELWISE',
                    'dtype': 'INT',
                },
                'compute_precision': 'INTEGER',
                'explicit_dequantize': False,
                'skip_checks': False,
            },
        })
    ]
    qt = quantizer.Quantizer(self._test_model_path, recipe)
    calib_result = qt.calibrate(_MULTI_SIGNATURE_CALIBRATION_DATASET, mode=mode)
    self.assertIsNotNone(qt.quantize(calib_result).quantized_model)

  @parameterized.parameters(
      _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      _CalibrationMode.CALIBRATION_PROFILER_BASED,
  )
  def test_quantization_with_insufficient_calibration(self, mode):
    # Run calibration for one signature only.
    scarce_calibration_dataset = {
        'add': [{'x': np.array([2.0], dtype=np.float32)}],
    }
    calib_result = self._quantizer.calibrate(
        scarce_calibration_dataset, mode=mode
    )

    # Quantize and expect an error about missing signature in calibration data.
    error_message = 'MUL(index: 0) not found in tensor_name_to_qsv'
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._quantizer.quantize(calib_result)


class QuantizerToyGemma2Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp_save_path = self.create_tempdir().full_path
    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / 'tests/models/toy_model_with_kv_cache_multi_signature.tflite',
    )

    self._toy_gemma2_calibration_dataset = {
        'signature_1': [{
            'cache_0': _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            'cache_1': _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            'positions': (
                _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
            ),
            'tokens': (
                _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
            ),
        }],
        'signature_2': [{
            'cache_0': _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            'cache_1': _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            'positions': (
                _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
            ),
            'tokens': (
                _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
            ),
        }],
    }

    self._test_recipe = recipe_utils.resolve_recipe('default_a8w8')

    self._quantizer = quantizer.Quantizer(
        self._test_model_path, self._test_recipe
    )

    self._quantizer.update_quantization_recipe(
        regex='.*',
        operation_name=qtyping.TFLOperationName.OUTPUT,
        algorithm_key=_AlgorithmName.NO_QUANTIZE,
    )

  @parameterized.parameters(
      _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      _CalibrationMode.CALIBRATION_PROFILER_BASED,
  )
  def test_toy_gemma2_quantization_succeeds(self, mode):
    calib_result = self._quantizer.calibrate(
        self._toy_gemma2_calibration_dataset, mode=mode
    )
    self.assertIsNotNone(calib_result)
    self._quantizer.quantize(calib_result)
    self.assertIsNotNone(self._quantizer._result.quantized_model)

  @parameterized.parameters(
      _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      _CalibrationMode.CALIBRATION_PROFILER_BASED,
  )
  def test_toy_gemma2_update_signature_defs_succeeds(self, mode):

    self.assertTrue(
        _is_all_signature_defs_outputs_float(
            open(self._test_model_path, 'rb').read()
        )
    )
    calib_result = self._quantizer.calibrate(
        self._toy_gemma2_calibration_dataset, mode=mode
    )
    self.assertIsNotNone(calib_result)
    self._quantizer.quantize(calib_result)
    self.assertIsNotNone(self._quantizer._result.quantized_model)
    self.assertTrue(
        _is_all_signature_defs_outputs_float(
            self._quantizer._result.quantized_model
        )
    )


class QuantizerFullyConnectedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp_save_path = self.create_tempdir().full_path
    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / 'tests/models/single_fc.tflite',
    )

    self._test_recipe = recipe_utils.resolve_recipe('default_a8w8')

    self._quantizer = quantizer.Quantizer(
        self._test_model_path, self._test_recipe
    )

    self._quantizer.update_quantization_recipe(
        regex='.*',
        operation_name=qtyping.TFLOperationName.INPUT,
        algorithm_key=_AlgorithmName.NO_QUANTIZE,
    )
    self._quantizer.update_quantization_recipe(
        regex='.*',
        operation_name=qtyping.TFLOperationName.OUTPUT,
        algorithm_key=_AlgorithmName.NO_QUANTIZE,
    )

  @parameterized.parameters(
      _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      _CalibrationMode.CALIBRATION_PROFILER_BASED,
  )
  def test_fully_connected_quantization_succeeds(self, mode):
    calib_result = self._quantizer.calibrate(
        tfl_interpreter_utils.create_random_normal_input_data(
            self._test_model_path, num_samples=4
        ),
        mode=mode,
    )
    self.assertIsNotNone(calib_result)
    self._quantizer.quantize(calib_result)
    self.assertIsNotNone(self._quantizer._result.quantized_model)

  @parameterized.parameters(
      _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      _CalibrationMode.CALIBRATION_PROFILER_BASED,
  )
  def test_fully_connected_quantization_update_signature_defs_succeeds(
      self, mode
  ):

    model_content = open(self._test_model_path, 'rb').read()
    self.assertTrue(_is_all_signature_defs_inputs_float(model_content))
    self.assertTrue(_is_all_signature_defs_outputs_float(model_content))

    calib_result = self._quantizer.calibrate(
        tfl_interpreter_utils.create_random_normal_input_data(
            self._test_model_path, num_samples=4
        ),
        mode=mode,
    )
    self.assertIsNotNone(calib_result)
    quant_result = self._quantizer.quantize(calib_result)
    self.assertIsNotNone(quant_result.quantized_model)

    self.assertTrue(
        _is_all_signature_defs_inputs_float(quant_result.quantized_model)
    )
    self.assertTrue(
        _is_all_signature_defs_outputs_float(quant_result.quantized_model)
    )


class CalibrationModesParityTest(parameterized.TestCase):

  @parameterized.parameters(
      'single_fc.tflite',
      'conv_fc_mnist.tflite',
      'two_signatures.tflite',
  )
  def test_parity_between_calibration_modes(self, model_name):
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / 'tests/models' / model_name
    )
    input_data = tfl_interpreter_utils.create_random_normal_input_data(
        model_path, random_seed=0
    )
    _test_recipe = recipe_utils.resolve_recipe('default_a8w8')
    qt = quantizer.Quantizer(model_path, _test_recipe)
    self.assertTrue(qt.need_calibration)

    # 1. Run CALIBRATION_PRESERVE_ALL_TENSORS
    calib_preserve = qt.calibrate(
        input_data, mode=_CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS
    )
    model_preserve = qt.quantize(calib_preserve).quantized_model

    # 2. Run CALIBRATION_PROFILER_BASED
    calib_profiler = qt.calibrate(
        input_data, mode=_CalibrationMode.CALIBRATION_PROFILER_BASED
    )
    model_profiler = qt.quantize(calib_profiler).quantized_model

    # 3. Compare models by running inference. Note that quantized models
    # might be binary different due to slight floating point differences in
    # QSVs.
    validation_result = model_validator.compare_model(
        model_preserve,
        model_profiler,
        input_data,
        error_metrics=[quantizer.ValidationErrorMetric.MSE],
    )

    for sig_key in validation_result.available_signature_keys():
      sig_result = validation_result.get_signature_comparison_result(sig_key)
      for output_name, errors in sig_result.output_tensors.items():
        self.assertLess(
            errors[quantizer.ValidationErrorMetric.MSE.value],
            1e-10,
            f'Output mismatch for {sig_key}:{output_name}',
        )


if __name__ == '__main__':
  absltest.main()
