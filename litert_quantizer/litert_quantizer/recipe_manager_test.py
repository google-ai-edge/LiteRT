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

"""Tests for recipe_manager.py."""

import copy

from absl.testing import absltest
from absl.testing import parameterized

from litert_quantizer import algorithm_manager
from litert_quantizer import algorithm_manager_api
from litert_quantizer import qtyping
from litert_quantizer import recipe_manager

_ComputePrecision = qtyping.ComputePrecision
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_TensorDataType = qtyping.TensorDataType
_AlgorithmName = recipe_manager.AlgorithmName
_QuantGranularity = qtyping.QuantGranularity


def _sample_check_op_config_func(op_name, op_config, _):
  if (
      op_config.weight_tensor_config is not None
      and op_config.weight_tensor_config.num_bits == 17
  ):
    raise ValueError(f'Unsupported number of bits for op: {op_name}.')


def _add_default_int8xint8_integer_recipe(recipe_manager_object):
  recipe_manager_object.add_quantization_config(
      regex='.*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      op_config=qtyping.OpQuantizationConfig(
          activation_tensor_config=_TensorQuantConfig(
              num_bits=8, symmetric=False
          ),
          weight_tensor_config=_TensorQuantConfig(num_bits=8),
          compute_precision=_ComputePrecision.INTEGER,  # SRQ.
      ),
  )


# register some currently unsupported ops for testing purposes
def _register_testing_op(algorithm_key, tfl_op):
  # Sample functions for test cases.
  def _sample_init_qsvs(*_, **__):
    return {'name': dict()}

  def _sample_calibration_func(*_, **__):
    return {'name2': dict()}

  def _sample_materialize_func(*_, **__):
    return []

  algorithm_manager.register_op_quant_config_validation_func(
      algorithm_key, _sample_check_op_config_func  # pyrefly: ignore[bad-argument-type]
  )
  algorithm_manager.register_quantized_op(
      algorithm_key,
      tfl_op,
      _sample_init_qsvs,
      _sample_calibration_func,
      _sample_materialize_func,
  )


class ConfiguratorTest(parameterized.TestCase):
  """Test cases for the flax quantizer Configurator."""

  def setUp(self):
    super().setUp()
    self._setup_algorithm_manager()
    self._recipe_manager = recipe_manager.RecipeManager()
    self._testing_ops = [
        _TFLOpName.BATCH_MATMUL,
        _TFLOpName.FULLY_CONNECTED,
        _TFLOpName.DEPTHWISE_CONV_2D,
    ]
    for op in self._testing_ops:
      _register_testing_op(_AlgorithmName.MIN_MAX_UNIFORM_QUANT, op)
      _register_testing_op('GPTQ', op)

  def tearDown(self):
    self._restore_algorithm_manager()
    super().tearDown()

  def _setup_algorithm_manager(self):
    self._original_alg_manager_instance = (
        algorithm_manager._alg_manager_instance
    )
    self._new_alg_manager_instance = algorithm_manager_api.AlgorithmManagerApi()

    # Deep copy the state from original instance
    self._new_alg_manager_instance._config_check_registry = (
        self._original_alg_manager_instance._config_check_registry.copy()
    )
    self._new_alg_manager_instance._config_check_policy_registry = (
        self._original_alg_manager_instance._config_check_policy_registry.copy()
    )
    self._new_alg_manager_instance._algorithm_registry = {}
    for k, v in self._original_alg_manager_instance._algorithm_registry.items():
      # v is QuantizationAlgorithmInfo
      new_v = copy.copy(v)
      new_v.quantized_ops = v.quantized_ops.copy()
      self._new_alg_manager_instance._algorithm_registry[k] = new_v

    algorithm_manager._alg_manager_instance = self._new_alg_manager_instance
    self._original_funcs = {}
    for func_name in [
        'get_quantization_func',
        'get_supported_ops',
        'get_init_qsv_func',
        'register_op_quant_config_validation_func',
        'register_config_check_policy_func',
        'register_quantized_op',
        'is_op_registered',
        'is_algorithm_registered',
        'check_op_quantization_config',
    ]:
      self._original_funcs[func_name] = getattr(algorithm_manager, func_name)
      if hasattr(self._new_alg_manager_instance, func_name):
        setattr(
            algorithm_manager,
            func_name,
            getattr(self._new_alg_manager_instance, func_name),
        )
      elif func_name == 'register_config_check_policy_func':
        setattr(
            algorithm_manager,
            func_name,
            self._new_alg_manager_instance.register_config_check_policy,
        )

  def _restore_algorithm_manager(self):
    algorithm_manager._alg_manager_instance = (
        self._original_alg_manager_instance
    )
    for func_name, func in self._original_funcs.items():
      setattr(algorithm_manager, func_name, func)

  def test_add_get_quantization_config(self):
    # Int8 DRQ all ops under "Dense".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.INTEGER,  # DRQ.
        ),
    )

    # Int8 weight-only FullyConnected configuration under "Dense_3".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
            explicit_dequantize=True,
        ),
    )
    # Int4 DRQ BatchMatmul configuration under "Dense_3".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.BATCH_MATMUL,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=4),
            compute_precision=_ComputePrecision.INTEGER,  # DRQ.
        ),
    )

    # Return NO_QUANT if not match.
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_1/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.NO_QUANTIZE)
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.DEPTHWISE_CONV_2D, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.NO_QUANTIZE)

    # Check _TFLOperationKey.ALL
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.DEPTHWISE_CONV_2D, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    # DRQ check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)

    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    # DRQ check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)

    # Check conflicts handling.
    # Int8 Weight-only for FC under "Dense", this should only overwrite FC but
    # leave others unchanged.
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
        ),
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    # WEIGHT_ONLY check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.FLOAT)
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)

    # Reset all ops, this time with 4 bits DRQ.
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=4),
            compute_precision=_ComputePrecision.INTEGER,  # DRQ.
        ),
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    # DRQ check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)
    self.assertEqual(weight_tensor_config.num_bits, 4)
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    # DRQ check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)
    self.assertEqual(weight_tensor_config.num_bits, 4)

    # Overwrite all FC.
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=3),
        ),
    )
    # FC config is overridden.
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, 3)
    # No overridden for batch matmul.
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, 4)

  def test_add_unsupported_quantization_config(self):
    error_message = 'Unsupported operation'
    # Add unregistered operations.
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._recipe_manager.add_quantization_config(
          regex='.*/Dense/.*',
          operation_name=_TFLOpName.CUSTOM_OP,
          algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          op_config=qtyping.OpQuantizationConfig(
              weight_tensor_config=_TensorQuantConfig(num_bits=8),
              compute_precision=_ComputePrecision.INTEGER,  # DRQ.
          ),
      )

  def test_add_unsupported_num_bits_raise_error(self):
    test_op_name = _TFLOpName.FULLY_CONNECTED
    error_message = f'Unsupported number of bits for op: {test_op_name}.'
    # Add unregistered operation
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._recipe_manager.add_quantization_config(
          regex='.*/Dense/.*',
          operation_name=test_op_name,
          algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          op_config=qtyping.OpQuantizationConfig(
              weight_tensor_config=_TensorQuantConfig(num_bits=17),
          ),
      )

  def test_add_unsupported_skip_successful(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=17),
            compute_precision=_ComputePrecision.INTEGER,  # DRQ.
            skip_checks=True,
        ),
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, 17)
    # DRQ check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)

  def test_add_unsupported_algorithm_key_raise_error(self):
    error_message = 'Unsupported algorithm key'
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._recipe_manager.add_quantization_config(
          regex='.*/Dense/.*',
          operation_name=_TFLOpName.FULLY_CONNECTED,
          algorithm_key='decomposed_hadamard',
          op_config=qtyping.OpQuantizationConfig(
              weight_tensor_config=_TensorQuantConfig(num_bits=8),
          ),
      )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._recipe_manager.add_quantization_config(
          regex='.*/Dense/.*',
          operation_name=_TFLOpName.ALL_SUPPORTED,
          algorithm_key='decomposed_hadamard',
          op_config=qtyping.OpQuantizationConfig(
              weight_tensor_config=_TensorQuantConfig(num_bits=8),
          ),
      )

  def test_add_dynamic_config(self):
    self._recipe_manager.add_dynamic_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        num_bits=8,
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)
    self.assertFalse(op_config.explicit_dequantize)
    self.assertIsNone(op_config.activation_tensor_config)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, 8)
    self.assertTrue(weight_tensor_config.symmetric)
    self.assertEqual(
        weight_tensor_config.granularity,
        _QuantGranularity.CHANNELWISE,
    )

  @parameterized.parameters(4, 8)
  def test_add_weight_only_config_int(self, num_bits):
    self._recipe_manager.add_weight_only_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        num_bits=num_bits,
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.compute_precision, _ComputePrecision.FLOAT)
    self.assertTrue(op_config.explicit_dequantize)
    self.assertIsNone(op_config.activation_tensor_config)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, num_bits)
    self.assertTrue(weight_tensor_config.symmetric)
    self.assertEqual(
        weight_tensor_config.granularity,
        _QuantGranularity.CHANNELWISE,
    )
    self.assertEqual(weight_tensor_config.dtype, _TensorDataType.INT)

  def test_add_weight_only_config_fp16(self):
    self._recipe_manager.add_weight_only_config(
        regex='.*/Dense2/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        num_bits=16,
        algorithm_key=_AlgorithmName.FLOAT_CASTING,
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense2/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.FLOAT_CASTING)
    self.assertEqual(op_config.compute_precision, _ComputePrecision.FLOAT)
    self.assertTrue(op_config.explicit_dequantize)
    self.assertIsNone(op_config.activation_tensor_config)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, 16)
    self.assertTrue(weight_tensor_config.symmetric)
    self.assertEqual(
        weight_tensor_config.granularity,
        _QuantGranularity.CHANNELWISE,
    )
    self.assertEqual(weight_tensor_config.dtype, _TensorDataType.FLOAT)

  def test_add_weight_only_config_fp8_raise_error(self):
    error_message = (
        'float casting quantization config requires number of bits to be set'
        ' as 16'
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._recipe_manager.add_weight_only_config(
          regex='.*/Dense2/.*',
          operation_name=_TFLOpName.FULLY_CONNECTED,
          num_bits=8,
          algorithm_key=_AlgorithmName.FLOAT_CASTING,
      )

  def test_add_static_config(self):
    self._recipe_manager.add_static_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        activation_num_bits=8,
        weight_num_bits=4,
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)
    self.assertFalse(op_config.explicit_dequantize)
    activation_tensor_config = op_config.activation_tensor_config
    self.assertIsNotNone(activation_tensor_config)
    self.assertEqual(activation_tensor_config.num_bits, 8)
    self.assertFalse(activation_tensor_config.symmetric)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, 4)
    self.assertTrue(weight_tensor_config.symmetric)
    self.assertEqual(
        weight_tensor_config.granularity,
        _QuantGranularity.CHANNELWISE,
    )

  def test_set_full_integer_quantization_config(self):
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    # Full integer setting is global
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    op_act_config = op_config.activation_tensor_config
    self.assertIsNotNone(op_act_config)
    self.assertEqual(op_act_config.num_bits, 8)
    self.assertEqual(op_act_config.symmetric, False)
    self.assertEqual(
        op_act_config.granularity,
        _QuantGranularity.TENSORWISE,
    )
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, 8)
    self.assertEqual(weight_tensor_config.symmetric, True)
    self.assertEqual(
        weight_tensor_config.granularity,
        _QuantGranularity.TENSORWISE,
    )

    # Change weight settings for Dense_3 FC
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=3),
            compute_precision=_ComputePrecision.INTEGER,  # DRQ.
        ),
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, 3)
    # WEIGHT_ONLY check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)

    # Change the global setting to  int16
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=_TensorQuantConfig(
                num_bits=16, symmetric=True
            ),
            weight_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
            compute_precision=_ComputePrecision.INTEGER,  # SRQ.
        ),
    )
    # This does not impact the special dense_3 case
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(weight_tensor_config.num_bits, 3)
    # WEIGHT_ONLY check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)

    # All the others will be int16
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.CONV_2D, 'model/Dense_31/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    op_act_config = op_config.activation_tensor_config
    self.assertIsNotNone(op_act_config)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    self.assertEqual(op_act_config.num_bits, 16)
    self.assertEqual(op_act_config.symmetric, True)
    self.assertEqual(
        op_act_config.granularity,
        _QuantGranularity.TENSORWISE,
    )
    self.assertEqual(weight_tensor_config.num_bits, 8)
    self.assertEqual(weight_tensor_config.symmetric, True)
    self.assertEqual(
        weight_tensor_config.granularity,
        _QuantGranularity.TENSORWISE,
    )

  def test_get_full_quantization_config(self):
    # Int8 asymetric full integer model.
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    # Default all BMM.
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=_TFLOpName.BATCH_MATMUL,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
            explicit_dequantize=True,
        ),
    )

    # Int8 DRQ FULLY_CONNECTED ops under "Dense".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.INTEGER,  # DRQ.
        ),
    )

    # Overwrite DRQ ALL ops under "Dense".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=4),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
            explicit_dequantize=True,
        ),
    )

    # Overwrite "Dense_1" to only quantize FullyConnected.
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_1/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=6),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
            explicit_dequantize=True,
        ),
    )

    # Add BMM to "Dense_1".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_1/.*',
        operation_name=_TFLOpName.BATCH_MATMUL,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=3),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
            explicit_dequantize=True,
        ),
    )

    expected_full_quantization_config = [
        {
            'regex': '.*',
            'operation': _TFLOpName.ALL_SUPPORTED,
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'activation_tensor_config': {
                    'num_bits': 8,
                    'symmetric': False,
                    'granularity': _QuantGranularity.TENSORWISE,
                    'dtype': _TensorDataType.INT,
                },
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': _QuantGranularity.TENSORWISE,
                    'dtype': _TensorDataType.INT,
                },
                # WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.INTEGER,
                'explicit_dequantize': False,
                'skip_checks': False,
                'min_weight_elements': 0,
            },
        },
        {
            'regex': '.*',
            'operation': _TFLOpName.BATCH_MATMUL,
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': _QuantGranularity.TENSORWISE,
                },
                # WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
                'skip_checks': False,
                'min_weight_elements': 0,
            },
        },
        {
            'regex': '.*/Dense/.*',
            'operation': _TFLOpName.ALL_SUPPORTED,
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 4,
                    'symmetric': True,
                    'granularity': _QuantGranularity.TENSORWISE,
                },
                # WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
                'skip_checks': False,
                'min_weight_elements': 0,
            },
        },
        {
            'regex': '.*/Dense_1/.*',
            'operation': _TFLOpName.FULLY_CONNECTED,
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 6,
                    'symmetric': True,
                    'granularity': _QuantGranularity.TENSORWISE,
                },
                # WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
                'skip_checks': False,
                'min_weight_elements': 0,
            },
        },
        {
            'regex': '.*/Dense_1/.*',
            'operation': _TFLOpName.BATCH_MATMUL,
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 3,
                    'symmetric': True,
                    'granularity': _QuantGranularity.TENSORWISE,
                },
                # WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
                'skip_checks': False,
                'min_weight_elements': 0,
            },
        },
    ]
    self.assertEqual(
        expected_full_quantization_config,
        self._recipe_manager.get_quantization_recipe(),
    )

  def test_get_quantization_configs_with_no_quantize_overwrite(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
        ),
    )
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.NO_QUANTIZE,
    )

    # Fully connected will be overwritten to no quantization.
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.NO_QUANTIZE)
    # Other ops will be quantized.
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.CONV_2D, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    # WEIGHT_ONLY check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.FLOAT)
    self.assertEqual(weight_tensor_config.num_bits, 8)

  def test_load_from_full_quantization_config(self):
    full_quantization_config = [
        {
            'regex': '.*',
            'operation': 'BATCH_MATMUL',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': False,
            },
        },
        {
            'regex': '.*/Dense/.*',
            'operation': '*',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 4,
                    'symmetric': False,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # DRQ.
                'compute_precision': _ComputePrecision.INTEGER,
            },
        },
    ]
    self._recipe_manager.load_quantization_recipe(full_quantization_config)

    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense10/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    # WEIGHT_ONLY check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.FLOAT)
    self.assertEqual(weight_tensor_config.num_bits, 8)

    # Dense will be overwritten by the last setting
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    # DRQ check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)
    self.assertEqual(weight_tensor_config.num_bits, 4)

  def test_get_unsupported_op_fall_back_to_default(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=17),
        ),
    )
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense10/op'
    )
    # int17 is not supported, fall back to float.
    self.assertEqual(alg_key, _AlgorithmName.NO_QUANTIZE)

  def test_load_from_full_quantization_config_full_integer(self):
    full_quantization_config = [
        {
            'regex': '.*',
            'operation': '*',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'activation_tensor_config': {
                    'num_bits': 8,
                    'symmetric': False,
                    'granularity': _QuantGranularity.TENSORWISE,
                    'dtype': 'INT',
                },
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': _QuantGranularity.TENSORWISE,
                    'dtype': 'INT',
                },
                # SRQ.
                'compute_precision': _ComputePrecision.INTEGER,
            },
        },
        {
            'regex': '.*',
            'operation': 'BATCH_MATMUL',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
        },
        {
            'regex': '.*/Dense/.*',
            'operation': '*',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 4,
                    'symmetric': False,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # DRQ.
                'compute_precision': _ComputePrecision.INTEGER,
            },
        },
    ]
    self._recipe_manager.load_quantization_recipe(full_quantization_config)

    # BMMs will be overridden to weight-only
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense10/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    # WEIGHT_ONLY check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.FLOAT)
    self.assertEqual(weight_tensor_config.num_bits, 8)

    # Dense will be overwritten by the last setting
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    # DRQ check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)
    self.assertEqual(weight_tensor_config.num_bits, 4)

    # Other ops will have default quantization settings
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.CONV_2D, 'model/Dense11/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    op_act_config = op_config.activation_tensor_config
    self.assertIsNotNone(op_act_config)
    self.assertEqual(op_act_config.num_bits, 8)
    weight_tensor_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_tensor_config)
    # SRQ check.
    self.assertEqual(op_config.compute_precision, _ComputePrecision.INTEGER)
    self.assertEqual(weight_tensor_config.num_bits, 8)

  def test_need_calibration_false(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_1/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.INTEGER,  # DRQ.
        ),
    )
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_2/.*',
        operation_name=_TFLOpName.CONV_2D,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
            explicit_dequantize=True,
        ),
    )
    self.assertFalse(self._recipe_manager.need_calibration())

  def test_need_calibration_true(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_1/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.INTEGER,  # DRQ.
        ),
    )
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_2/.*',
        operation_name=_TFLOpName.CONV_2D,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
        ),
    )
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.BATCH_MATMUL,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            activation_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.INTEGER,  # SRQ.
        ),
    )
    self.assertTrue(self._recipe_manager.need_calibration())

  def test_need_calibration_gptq_true(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_1/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.GPTQ,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
            skip_checks=True,
        ),
    )
    self.assertTrue(self._recipe_manager.need_calibration())

  def test_get_hadamard_with_max_size(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.HADAMARD_ROTATION,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8, algorithm_params={'max_hadamard_size': 1024}
            ),
            compute_precision=_ComputePrecision.INTEGER,
        ),
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.HADAMARD_ROTATION)
    weight_tensor_config = op_config.weight_tensor_config
    assert weight_tensor_config is not None
    self.assertEqual(
        weight_tensor_config.algorithm_params['max_hadamard_size'], 1024
    )


if __name__ == '__main__':
  absltest.main()
