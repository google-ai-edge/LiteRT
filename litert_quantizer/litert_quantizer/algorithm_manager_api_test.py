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

"""Tests for algorithm_manager_api."""

from collections.abc import MutableMapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import algorithm_manager_api
from litert_quantizer import default_policy
from litert_quantizer import qtyping
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import qsv_utils

_TFLOpName = qtyping.TFLOperationName


# Sample functions for test cases.
def _sample_init_qsv(
    op_info: qtyping.OpInfo,  # pylint: disable=unused-argument
    graph_info: qtyping.GraphInfo,  # pylint: disable=unused-argument
    inputs_to_ignore: Sequence[int] | None,  # pylint: disable=unused-argument
    outputs_to_ignore: Sequence[int] | None,  # pylint: disable=unused-argument
) -> qtyping.QSV:
  return {"_sample_init_qsv": None}


def _sample_calibration_func(
    tfl_op: qtyping.OperatorT,  # pylint: disable=unused-argument
    graph_info: qtyping.GraphInfo,  # pylint: disable=unused-argument
    tensor_name_to_qsv: MutableMapping[str, np.ndarray],  # pylint: disable=unused-argument
    inputs_to_ignore: Sequence[int] | None = None,  # pylint: disable=unused-argument
    outputs_to_ignore: Sequence[int] | None = None,  # pylint: disable=unused-argument
) -> dict[str, qtyping.QSV]:
  return {"_sample_calibration_func": {"dummy": None}}


def _sample_materialize_func(
    op_info: qtyping.OpInfo,  # pylint: disable=unused-argument
    graph_info: qtyping.GraphInfo,  # pylint: disable=unused-argument
    tensor_name_to_qsv: MutableMapping[str, qtyping.QSV],  # pylint: disable=unused-argument
    tensor_quant_params_cache: common_utils.TensorQuantParamsCache,  # pylint: disable=unused-argument
) -> list[qtyping.TensorTransformationParams]:
  return []


def _sample_update_qsv_func(qsv: qtyping.QSV, new_qsv: qtyping.QSV) -> qtyping.QSV:  # pylint: disable=unused-argument
  return {"_sample_update_qsv_func": None}


def _sample_check_op_config_func(
    op_name: _TFLOpName,  # pylint: disable=unused-argument
    op_quant_config: qtyping.OpQuantizationConfig,
    config_check_policy: qtyping.ConfigCheckPolicyDict,  # pylint: disable=unused-argument
) -> None:
  if (
      op_quant_config.weight_tensor_config
      and op_quant_config.weight_tensor_config.num_bits == 17
  ):
    raise ValueError("Unsupported number of bits.")


class AlgorithmManagerApiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._alg_manager = algorithm_manager_api.AlgorithmManagerApi()

  def test_register_op_quant_config_validation_func_succeeds(self):
    self.assertEmpty(self._alg_manager._config_check_registry)
    test_algorithm_name = "test_algorithm"
    self._alg_manager.register_op_quant_config_validation_func(
        test_algorithm_name, _sample_check_op_config_func
    )
    self.assertIn(test_algorithm_name, self._alg_manager._config_check_registry)
    check_func = self._alg_manager._config_check_registry[test_algorithm_name]
    self.assertEqual(check_func, _sample_check_op_config_func)

  def test_register_quantized_op(self):
    self._alg_manager.register_quantized_op(
        algorithm_key="ptq",
        tfl_op_name=_TFLOpName.FULLY_CONNECTED,
        init_qsv_func=_sample_init_qsv,  # pyrefly: ignore[bad-argument-type]
        calibration_func=_sample_calibration_func,  # pyrefly: ignore[bad-argument-type]
        materialize_func=_sample_materialize_func,  # pyrefly: ignore[bad-argument-type]
    )
    self._alg_manager.register_quantized_op(
        algorithm_key="gptq",
        tfl_op_name=_TFLOpName.CONV_2D,
        init_qsv_func=_sample_init_qsv,  # pyrefly: ignore[bad-argument-type]
        calibration_func=_sample_calibration_func,  # pyrefly: ignore[bad-argument-type]
        materialize_func=_sample_materialize_func,  # pyrefly: ignore[bad-argument-type]
    )
    self.assertTrue(self._alg_manager.is_algorithm_registered("ptq"))
    self.assertTrue(self._alg_manager.is_algorithm_registered("gptq"))
    self.assertTrue(
        self._alg_manager.is_op_registered("ptq", _TFLOpName.FULLY_CONNECTED)
    )
    self.assertTrue(
        self._alg_manager.is_op_registered("gptq", _TFLOpName.CONV_2D)
    )
    self.assertFalse(
        self._alg_manager.is_op_registered("gptq", _TFLOpName.DEPTHWISE_CONV_2D)
    )

  def test_get_supported_ops(self):
    algorithm_key = "ptq"
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=_TFLOpName.FULLY_CONNECTED,
        init_qsv_func=_sample_init_qsv,  # pyrefly: ignore[bad-argument-type]
        calibration_func=_sample_calibration_func,  # pyrefly: ignore[bad-argument-type]
        materialize_func=_sample_materialize_func,  # pyrefly: ignore[bad-argument-type]
    )
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=_TFLOpName.CONV_2D,
        init_qsv_func=_sample_init_qsv,  # pyrefly: ignore[bad-argument-type]
        calibration_func=_sample_calibration_func,  # pyrefly: ignore[bad-argument-type]
        materialize_func=_sample_materialize_func,  # pyrefly: ignore[bad-argument-type]
    )
    supported_ops = self._alg_manager.get_supported_ops(algorithm_key)
    self.assertIn(_TFLOpName.CONV_2D, supported_ops)
    self.assertIn(_TFLOpName.FULLY_CONNECTED, supported_ops)
    self.assertNotIn(_TFLOpName.DEPTHWISE_CONV_2D, supported_ops)

  def test_get_quantization_func(self):
    algorithm_key = "ptq"
    tfl_op = _TFLOpName.FULLY_CONNECTED
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=tfl_op,
        init_qsv_func=_sample_init_qsv,  # pyrefly: ignore[bad-argument-type]
        calibration_func=_sample_calibration_func,  # pyrefly: ignore[bad-argument-type]
        materialize_func=_sample_materialize_func,  # pyrefly: ignore[bad-argument-type]
    )
    materialize_func = self._alg_manager.get_quantization_func(
        algorithm_key,
        tfl_op,
        qtyping.QuantizeMode.MATERIALIZE,
    )
    self.assertIs(_sample_materialize_func, materialize_func)
    calibration_func = self._alg_manager.get_quantization_func(
        algorithm_key,
        tfl_op,
        qtyping.QuantizeMode.CALIBRATE,
    )
    self.assertIs(_sample_calibration_func, calibration_func)

    # Query for unsupported operation.
    error_message = "Unsupported operation"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._alg_manager.get_quantization_func(
          algorithm_key,
          _TFLOpName.BATCH_MATMUL,
          qtyping.QuantizeMode.MATERIALIZE,
      )

    # Query for unregisted algorithm.
    error_message = "Unregistered algorithm"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._alg_manager.get_quantization_func(
          "gptq",
          tfl_op,
          qtyping.QuantizeMode.MATERIALIZE,
      )

  def test_get_init_qsv_func(self):
    algorithm_key = "ptq"
    tfl_op = _TFLOpName.FULLY_CONNECTED
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=tfl_op,
        init_qsv_func=_sample_init_qsv,  # pyrefly: ignore[bad-argument-type]
        calibration_func=_sample_calibration_func,  # pyrefly: ignore[bad-argument-type]
        materialize_func=_sample_materialize_func,  # pyrefly: ignore[bad-argument-type]
    )
    init_qsv_func = self._alg_manager.get_init_qsv_func(algorithm_key, tfl_op)
    self.assertIs(_sample_init_qsv, init_qsv_func)

    # Query for unsupported operation.
    error_message = "Unsupported operation"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._alg_manager.get_init_qsv_func(
          algorithm_key,
          _TFLOpName.BATCH_MATMUL,
      )

    # Query for unregisted algorithm.
    error_message = "Unregistered algorithm"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._alg_manager.get_init_qsv_func(
          "gptq",
          tfl_op,
      )

  def test_register_config_check_policy_succeeds(self):
    self.assertEmpty(self._alg_manager._config_check_policy_registry)
    test_algorithm_name = "test_algorithm"
    test_config_check_policy = qtyping.ConfigCheckPolicyDict({  # pyrefly: ignore[no-matching-overload]
        _TFLOpName.FULLY_CONNECTED: {
            qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(
                    num_bits=1
                )
            )
        }
    })
    self._alg_manager.register_config_check_policy(
        test_algorithm_name, test_config_check_policy
    )
    self.assertIn(
        test_algorithm_name, self._alg_manager._config_check_policy_registry
    )
    self.assertIsNotNone(
        self._alg_manager._config_check_policy_registry[test_algorithm_name]
    )

  def test_default_policy_not_empty(self):
    """Tests that the default policy is not empty & no empty policy is generated."""
    self.assertNotEmpty(default_policy.DEFAULT_CONFIG_CHECK_POLICY)
    for policy in default_policy.DEFAULT_CONFIG_CHECK_POLICY.values():
      self.assertNotEmpty(policy)

  def test_get_update_qsv_func_default(self):
    algorithm_key = "ptq"
    tfl_op = _TFLOpName.FULLY_CONNECTED
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=tfl_op,
        init_qsv_func=_sample_init_qsv,  # pyrefly: ignore[bad-argument-type]
        calibration_func=_sample_calibration_func,  # pyrefly: ignore[bad-argument-type]
        materialize_func=_sample_materialize_func,  # pyrefly: ignore[bad-argument-type]
    )
    update_qsv_func = self._alg_manager.get_update_qsv_func(
        algorithm_key, tfl_op
    )
    self.assertEqual(update_qsv_func, qsv_utils.moving_average_update)

  def test_get_update_qsv_func_custom(self):
    algorithm_key = "ptq"
    tfl_op = _TFLOpName.FULLY_CONNECTED
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=tfl_op,
        init_qsv_func=_sample_init_qsv,  # pyrefly: ignore[bad-argument-type]
        calibration_func=_sample_calibration_func,  # pyrefly: ignore[bad-argument-type]
        materialize_func=_sample_materialize_func,  # pyrefly: ignore[bad-argument-type]
        update_qsv_func=_sample_update_qsv_func,  # pyrefly: ignore[bad-argument-type]
    )
    update_qsv_func = self._alg_manager.get_update_qsv_func(
        algorithm_key, tfl_op
    )
    self.assertEqual(update_qsv_func, _sample_update_qsv_func)


if __name__ == "__main__":
  absltest.main()
