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

"""Tests for calibrator."""

from collections.abc import Generator, Mapping
import json
import pathlib
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import os
import io
from litert_quantizer import calibrator
from litert_quantizer import qtyping
from litert_quantizer import recipe_manager
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils

_ComputePrecision = qtyping.ComputePrecision
_AlgorithmName = recipe_manager.AlgorithmName
_CalibrationMode = calibrator.CalibrationMode

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("")
_TENSOR_QUANT_CONFIG = qtyping.TensorQuantizationConfig

TEST_MIN_VAL, TEST_MAX_VAL = -1, 1

_RNG = np.random.default_rng(66)


def _representative_dataset_gen(size=(1, 8), num_samples=10):
  for _ in range(num_samples):
    vals = np.random.rand(*size).astype(np.float32)
    vals[0][0], vals[0][1] = (
        TEST_MIN_VAL,
        TEST_MAX_VAL,
    )  # fix min/max for testing
    yield {"input_1": vals}


def _get_calibration_data(
    dataset_gen: Generator[dict[str, Any], Any, None],
) -> dict[str, Any]:
  calibration_samples = [sample for sample in dataset_gen]
  calibration_data = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
  }
  return calibration_data


def _add_default_int8xint8_integer_recipe(recipe_manager_object):
  recipe_manager_object.add_quantization_config(
      regex=".*",
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      op_config=qtyping.OpQuantizationConfig(
          activation_tensor_config=_TENSOR_QUANT_CONFIG(
              num_bits=8, symmetric=False
          ),
          weight_tensor_config=_TENSOR_QUANT_CONFIG(num_bits=8, symmetric=True),
          compute_precision=_ComputePrecision.INTEGER,
      ),
  )


class CalibratorTestBase(absltest.TestCase):

  mode: _CalibrationMode | None = None
  _recipe_manager: recipe_manager.RecipeManager
  _representative_dataset: dict[str, Any]

  def setUp(self):
    super().setUp()
    if self.mode is None:
      self.skipTest("Base class should not run directly.")
    np.random.seed(0)
    self._recipe_manager = recipe_manager.RecipeManager()
    dataset_gen = _representative_dataset_gen()
    self._representative_dataset = _get_calibration_data(dataset_gen)

  def _single_fc_model_init(self) -> None:
    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / "tests/models/single_fc.tflite"
    )
    self._calibrator = calibrator.Calibrator(
        self._test_model_path,
        mode=self.mode,  # pytype: disable=wrong-arg-types
    )

  def test_calibrator_state_manipulation(self):
    self._single_fc_model_init()
    # load/get qsvs
    sample_qsv = {"serving_default_input_1:0": {"min": -10, "max": 8}}
    self._calibrator.load_model_qsvs(sample_qsv)  # pyrefly: ignore[bad-argument-type]
    model_tensor_qsvs = self._calibrator.get_model_qsvs()
    self.assertLen(model_tensor_qsvs, 1)
    self.assertIn("serving_default_input_1:0", model_tensor_qsvs)  # input
    input_qsv = model_tensor_qsvs["serving_default_input_1:0"]
    self.assertEqual(input_qsv["min"], -10)
    self.assertEqual(input_qsv["max"], 8)

    # reset qsvs
    self._calibrator.reset_model_qsvs()
    model_tensor_qsvs = self._calibrator.get_model_qsvs()
    self.assertEmpty(model_tensor_qsvs)

  def test_calibrate_single_fc_success(self):
    self._single_fc_model_init()
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager
    )
    model_tensor_qsvs = self._calibrator.get_model_qsvs()

    self.assertLen(model_tensor_qsvs, 2)
    self.assertIn("serving_default_input_1:0", model_tensor_qsvs)  # input
    input_qsv = model_tensor_qsvs["serving_default_input_1:0"]
    self.assertSequenceAlmostEqual(
        input_qsv["min"].flatten(), [TEST_MIN_VAL], delta=1e-5
    )
    self.assertSequenceAlmostEqual(
        input_qsv["max"].flatten(), [TEST_MAX_VAL], delta=1e-5
    )
    self.assertIn("StatefulPartitionedCall:0", model_tensor_qsvs)  # output
    output_qsv = model_tensor_qsvs["StatefulPartitionedCall:0"]
    # Relu, only check the min
    self.assertSequenceAlmostEqual(output_qsv["min"].flatten(), [0])

  def test_calibration_cache_is_empty_when_off(self):
    self._single_fc_model_init()
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self.assertEmpty(self._calibrator.get_cached_output())
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager, cache_output=False
    )
    self.assertEmpty(self._calibrator.get_cached_output())

  def test_calibration_cache_when_on(self):
    self._single_fc_model_init()
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self.assertEmpty(self._calibrator.get_cached_output())
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager, cache_output=True
    )
    self.assertLen(self._calibrator.get_cached_output(), 10)

  def test_calibration_cache_is_empty_after_reset(self):
    self._single_fc_model_init()
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager, cache_output=True
    )
    self._calibrator.clear_cached_output()
    self.assertEmpty(self._calibrator.get_cached_output())

  def test_calibrate_unsupported_ops_success(self):
    # Many ops in the following model are not supported currently.
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "tests/models/branching_conv_fc.tflite"
    )
    test_calibrator = calibrator.Calibrator(test_model_path, mode=self.mode)  # pytype: disable=wrong-arg-types
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    dataset_gen = _representative_dataset_gen(size=(3, 4, 4, 1))
    test_calibrator.calibrate(
        _get_calibration_data(dataset_gen),
        self._recipe_manager,
        cache_output=True,
    )
    self.assertLen(test_calibrator.get_cached_output(), 10)

  def test_calibrate_reshape_with_empty_shape_success(self):
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "tests/models/reshape_with_empty_shape.tflite"
    )
    test_calibrator = calibrator.Calibrator(test_model_path, mode=self.mode)  # pytype: disable=wrong-arg-types
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    calib_data = tfl_interpreter_utils.create_random_normal_input_data(
        test_model_path, num_samples=4
    )
    test_calibrator.calibrate(calib_data, self._recipe_manager)
    self.assertNotEmpty(test_calibrator.get_model_qsvs())

  def test_save_and_load_calibration_result(self):
    self._single_fc_model_init()
    # Setup some QSV
    sample_qsv = {
        "serving_default_input_1:0": {
            "min": np.array([-10.0]),
            "max": np.array([8.0]),
        }
    }
    self._calibrator.load_model_qsvs(sample_qsv)  # pyrefly: ignore[bad-argument-type]

    # Save
    temp_file = self.create_tempfile().full_path
    self._calibrator.save_calibration_result(temp_file)

    # Reset
    self._calibrator.reset_model_qsvs()
    self.assertEmpty(self._calibrator.get_model_qsvs())

    # Load
    self._calibrator.load_model_qsvs(temp_file)

    # Verify
    model_tensor_qsvs = self._calibrator.get_model_qsvs()
    self.assertLen(model_tensor_qsvs, 1)
    self.assertIn("serving_default_input_1:0", model_tensor_qsvs)
    input_qsv = model_tensor_qsvs["serving_default_input_1:0"]
    self.assertSequenceAlmostEqual(input_qsv["min"], [-10.0])
    self.assertSequenceAlmostEqual(input_qsv["max"], [8.0])

  def test_calibrate_updates_num_samples_calibrated(self):
    self._single_fc_model_init()
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    # The representative dataset has 10 samples.
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager
    )
    # Check if internal counter is updated.
    self.assertEqual(self._calibrator._metadata["num_samples_calibrated"], 10)

  def test_save_and_load_calibration_result_with_metadata(self):
    self._single_fc_model_init()
    # Setup some QSV
    sample_qsv = {
        "serving_default_input_1:0": {
            "min": np.array([-10.0]),
            "max": np.array([8.0]),
        }
    }
    self._calibrator.load_model_qsvs(sample_qsv)  # pyrefly: ignore[bad-argument-type]
    self._calibrator._metadata["num_samples_calibrated"] = 123

    # Save
    temp_file = self.create_tempfile().full_path
    self._calibrator.save_calibration_result(temp_file)

    # Reset
    self._calibrator.reset_model_qsvs()
    self.assertEmpty(self._calibrator.get_model_qsvs())
    self.assertEqual(self._calibrator._metadata["num_samples_calibrated"], 0)

    # Load
    self._calibrator.load_model_qsvs(temp_file)

    # Verify
    model_tensor_qsvs = self._calibrator.get_model_qsvs()
    self.assertLen(model_tensor_qsvs, 1)
    self.assertEqual(self._calibrator._metadata["num_samples_calibrated"], 123)

  def test_load_legacy_calibration_result(self):
    self._single_fc_model_init()
    # Create a legacy format file (just the dict)
    legacy_data = {
        "serving_default_input_1:0": {
            "min": [-10.0],
            "max": [8.0],
        }
    }
    temp_file = self.create_tempfile().full_path
    with open(temp_file, "w") as f:
      json.dump(legacy_data, f)

    self._calibrator.load_model_qsvs(temp_file)

    model_tensor_qsvs = self._calibrator.get_model_qsvs()
    self.assertLen(model_tensor_qsvs, 1)
    self.assertEqual(self._calibrator._metadata["num_samples_calibrated"], 0)

  def test_preserve_metadata_on_save(self):
    self._single_fc_model_init()
    # Create a file with custom metadata
    initial_data = {
        "model_qsvs": {
            "serving_default_input_1:0": {
                "min": [-10.0],
                "max": [8.0],
            }
        },
        "metadata": {
            "num_samples_calibrated": 10,
            "custom_field": "custom_value",
        },
    }
    temp_file = self.create_tempfile().full_path
    with open(temp_file, "w") as f:
      json.dump(initial_data, f)

    # Load
    self._calibrator.load_model_qsvs(temp_file)
    self.assertEqual(self._calibrator._metadata["num_samples_calibrated"], 10)

    # Perform more calibration (update num samples)
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager
    )
    # 10 initial + 10 new samples = 20
    self.assertEqual(self._calibrator._metadata["num_samples_calibrated"], 20)

    # Save
    self._calibrator.save_calibration_result(temp_file)

    # Reload and verify
    with open(temp_file, "r") as f:
      saved_data = json.load(f)

    self.assertEqual(saved_data["metadata"]["num_samples_calibrated"], 20)
    self.assertEqual(saved_data["metadata"]["custom_field"], "custom_value")

  def test_save_with_extra_metadata(self):
    self._single_fc_model_init()
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager
    )
    temp_file = self.create_tempfile().full_path
    extra_metadata = {"model_name": "test_model", "version": "1.0"}
    self._calibrator.save_calibration_result(
        temp_file, extra_metadata=extra_metadata
    )

    with open(temp_file, "r") as f:
      saved_data = json.load(f)

    self.assertEqual(saved_data["metadata"]["num_samples_calibrated"], 10)
    self.assertEqual(saved_data["metadata"]["model_name"], "test_model")
    self.assertEqual(saved_data["metadata"]["version"], "1.0")

  def test_reset_model_qsvs_resets_counter(self):
    self._single_fc_model_init()
    self._calibrator._metadata["num_samples_calibrated"] = 100
    self._calibrator.reset_model_qsvs()
    self.assertEqual(self._calibrator._metadata["num_samples_calibrated"], 0)


class CalibratorPreserveAllTensorsTest(CalibratorTestBase):
  mode = _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS


class CalibratorProfilerBasedTest(CalibratorTestBase):
  mode = _CalibrationMode.CALIBRATION_PROFILER_BASED


class CalibratorAlreadyQuantizedModelTest(parameterized.TestCase):

  @parameterized.parameters(
      (_CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,),
      (_CalibrationMode.CALIBRATION_PROFILER_BASED,),
  )
  def test_check_is_float_model_succeeds_when_model_is_float(self, mode):
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "tests/models/conv_fc_mnist.tflite"
    )
    _ = calibrator.Calibrator(test_model_path, mode=mode)

  @parameterized.parameters(
      (_CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,),
      (_CalibrationMode.CALIBRATION_PROFILER_BASED,),
  )
  def test_check_is_quantized_model_succeeds_when_model_is_quantized(
      self, mode
  ):
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "tests/models/mnist_quantized.tflite"
    )
    _ = calibrator.Calibrator(test_model_path, mode=mode)


class CalibratorToyGemma2Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "tests/models/toy_model_with_kv_cache_multi_signature.tflite"
    )

    self._toy_gemma2_calibration_dataset = {
        "signature_1": [{
            "cache_0": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            "cache_1": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            "positions": (
                _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
            ),
            "tokens": (
                _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
            ),
        }],
        "signature_2": [{
            "cache_0": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            "cache_1": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            "positions": (
                _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
            ),
            "tokens": (
                _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
            ),
        }],
    }

  @parameterized.named_parameters(
      dict(
          testcase_name="preserve_all_tensors",
          mode=_CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
          expected_length=202,
      ),
      dict(
          testcase_name="profiler_based",
          mode=_CalibrationMode.CALIBRATION_PROFILER_BASED,
          expected_length=200,  # 2 fewer than preserve_all_tensors because
          # constant and bool tensors are skipped in profiler based mode.
      ),
  )
  def test_toy_gemma2_calibration_success(self, mode, expected_length):
    calib = calibrator.Calibrator(self._test_model_path, mode=mode)
    recipe_mngr = recipe_manager.RecipeManager()
    _add_default_int8xint8_integer_recipe(recipe_mngr)
    calib.calibrate(
        self._toy_gemma2_calibration_dataset,
        model_recipe_manager=recipe_mngr,
    )
    self.assertLen(calib.get_model_qsvs(), expected_length)


class CalibrationInterpreterTestBase(absltest.TestCase):

  mode: _CalibrationMode | None = None
  _test_model_path: str

  def setUp(self):
    super().setUp()
    if self.mode is None:
      self.skipTest("Base class")
    np.random.seed(0)
    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / "tests/models/single_fc.tflite"
    )

  def test_initialization(self):
    interpreter = calibrator.CalibrationInterpreter(
        self._test_model_path,
        mode=self.mode,  # pytype: disable=wrong-arg-types
    )
    self.assertIsInstance(interpreter, calibrator.CalibrationInterpreter)

  def test_calibration_mode(self):
    interpreter = calibrator.CalibrationInterpreter(
        self._test_model_path,
        mode=self.mode,  # pytype: disable=wrong-arg-types
    )
    runner = interpreter.get_signature_runner()

    # Run inference which triggers calibration
    input_data = np.random.rand(1, 8).astype(np.float32)
    output = runner(input_1=input_data)

    # Check results
    qsvs = interpreter.get_calibration_results()
    self.assertNotEmpty(qsvs)

    # Verify input tensor qsv
    self.assertIn("serving_default_input_1:0", qsvs)
    self.assertIsNotNone(output)

  def test_save_calibration_result(self):
    interpreter = calibrator.CalibrationInterpreter(
        self._test_model_path,
        mode=self.mode,  # pytype: disable=wrong-arg-types
    )
    runner = interpreter.get_signature_runner()
    input_data = np.random.rand(1, 8).astype(np.float32)
    runner(input_1=input_data)

    temp_file = self.create_tempfile().full_path
    interpreter.save_calibration_result(temp_file)

    # Verify file exists and has content
    with open(temp_file, "r") as f:
      content = f.read()
      self.assertNotEmpty(content)

  def test_get_signature_list(self):
    interpreter = calibrator.CalibrationInterpreter(
        self._test_model_path,
        mode=self.mode,  # pytype: disable=wrong-arg-types
    )
    signatures = interpreter.get_signature_list()
    self.assertNotEmpty(signatures)
    self.assertIn("serving_default", signatures)

  def test_runner_details(self):
    interpreter = calibrator.CalibrationInterpreter(
        self._test_model_path,
        mode=self.mode,  # pytype: disable=wrong-arg-types
    )
    runner = interpreter.get_signature_runner()
    input_details = runner.get_input_details()
    output_details = runner.get_output_details()

    self.assertNotEmpty(input_details)
    self.assertNotEmpty(output_details)
    self.assertIn("input_1", input_details)

  def test_output_match_original_interpreter(self):
    # Run calibration interpreter
    interpreter = calibrator.CalibrationInterpreter(
        self._test_model_path,
        mode=self.mode,  # pytype: disable=wrong-arg-types
    )
    calib_runner = interpreter.get_signature_runner()
    input_data = np.random.rand(1, 8).astype(np.float32)
    calib_output = calib_runner(input_1=input_data)

    # Run original interpreter
    original_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    original_runner = original_interpreter.get_signature_runner()
    original_output = original_runner(input_1=input_data)

    # Compare
    self.assertEqual(calib_output.keys(), original_output.keys())
    for key in calib_output:
      np.testing.assert_allclose(
          calib_output[key], original_output[key], rtol=1e-5, atol=1e-5
      )


class CalibrationInterpreterPreserveAllTensorsTest(
    CalibrationInterpreterTestBase
):
  mode = _CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS


class CalibrationInterpreterProfilerBasedTest(CalibrationInterpreterTestBase):
  mode = _CalibrationMode.CALIBRATION_PROFILER_BASED


class CalibrationModesComparisonTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self._recipe_manager: recipe_manager.RecipeManager = (
        recipe_manager.RecipeManager()
    )
    _add_default_int8xint8_integer_recipe(self._recipe_manager)

  def _get_random_input_data(
      self, model_path: str
  ) -> dict[str, list[dict[str, np.ndarray]]]:
    with open(model_path, "rb") as f:
      tflite_model = f.read()
    return tfl_interpreter_utils.create_random_normal_input_data(
        tflite_model, num_samples=1, random_seed=0
    )

  def _compare_qsvs(
      self,
      qsvs_preserve: Mapping[str, qtyping.QSV],
      qsvs_profiler: Mapping[str, qtyping.QSV],
  ) -> None:
    """Compares Quantization Statistics Values (QSVs) from two calibration modes.

    This function compares the min/max values in the QSV dictionaries generated
    by CALIBRATION_PRESERVE_ALL_TENSORS and CALIBRATION_PROFILER_BASED modes.
    It checks for non-empty results and asserts that the keys of the QSV
    dictionaries match exactly, flattening the min/max arrays for robust
    comparison.

    Args:
      qsvs_preserve: QSV dictionary from CALIBRATION_PRESERVE_ALL_TENSORS mode.
      qsvs_profiler: QSV dictionary from CALIBRATION_PROFILER_BASED mode.
    """
    self.assertNotEmpty(qsvs_preserve)
    self.assertNotEmpty(qsvs_profiler)
    self.assertCountEqual(qsvs_preserve, qsvs_profiler)
    for tensor_name in qsvs_preserve.keys():
      stats_preserve = qsvs_preserve[tensor_name]
      stats_profiler = qsvs_profiler[tensor_name]
      # Both calibrators produce scalar min/max values, but may store them with
      # different shapes (e.g., (1, 1, 1, 1) vs (1, 1) vs (1,)) depending on
      # operator and rank of the tensor. We flatten them here for robust
      # comparison.
      np.testing.assert_allclose(
          stats_preserve["min"].flatten(),
          stats_profiler["min"].flatten(),
          rtol=1e-5,
          atol=1e-5,
      )
      np.testing.assert_allclose(
          stats_preserve["max"].flatten(),
          stats_profiler["max"].flatten(),
          rtol=1e-5,
          atol=1e-5,
      )

  @parameterized.named_parameters(
      dict(testcase_name="single_fc", model_name="single_fc.tflite"),
      dict(testcase_name="conv_mnist", model_name="conv_fc_mnist.tflite"),
      dict(testcase_name="two_signatures", model_name="two_signatures.tflite"),
  )
  def test_modes_results_match_with_calibrator_api(
      self, model_name: str
  ) -> None:
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / "tests/models" / model_name
    )
    calib_preserve = calibrator.Calibrator(
        model_path, mode=_CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS
    )
    calib_profiler = calibrator.Calibrator(
        model_path, mode=_CalibrationMode.CALIBRATION_PROFILER_BASED
    )
    input_data_map = self._get_random_input_data(model_path)
    calib_preserve.calibrate(input_data_map, self._recipe_manager)
    calib_profiler.calibrate(input_data_map, self._recipe_manager)

    self._compare_qsvs(
        calib_preserve.get_model_qsvs(), calib_profiler.get_model_qsvs()
    )

  @parameterized.named_parameters(
      dict(testcase_name="single_fc", model_name="single_fc.tflite"),
      dict(testcase_name="conv_mnist", model_name="conv_fc_mnist.tflite"),
      dict(testcase_name="two_signatures", model_name="two_signatures.tflite"),
  )
  def test_modes_results_match_with_calibration_interpreter_api(
      self, model_name: str
  ) -> None:
    model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / "tests/models" / model_name
    )
    interpreter_preserve = calibrator.CalibrationInterpreter(
        model_path, mode=_CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS
    )
    interpreter_profiler = calibrator.CalibrationInterpreter(
        model_path, mode=_CalibrationMode.CALIBRATION_PROFILER_BASED
    )
    input_data_map = self._get_random_input_data(model_path)
    for sig_key, input_samples in input_data_map.items():
      runner_preserve = interpreter_preserve.get_signature_runner(sig_key)
      runner_profiler = interpreter_profiler.get_signature_runner(sig_key)
      input_data = input_samples[0]
      runner_preserve(**input_data)
      runner_profiler(**input_data)

    self._compare_qsvs(
        interpreter_preserve.get_calibration_results(),
        interpreter_profiler.get_calibration_results(),
    )


class CalibrationInterpreterInferenceModeTest(absltest.TestCase):

  def test_inference_mode_raises_error_on_get_calibration_results(self):
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / "tests/models/single_fc.tflite"
    )
    interpreter = calibrator.CalibrationInterpreter(
        test_model_path, mode=_CalibrationMode.INFERENCE
    )
    runner = interpreter.get_signature_runner()

    input_data = np.random.rand(1, 8).astype(np.float32)
    output = runner(input_1=input_data)

    with self.assertRaisesRegex(
        ValueError, "Calibration results are not available in INFERENCE mode."
    ):
      interpreter.get_calibration_results()
    self.assertIsNotNone(output)


if __name__ == "__main__":
  absltest.main()
