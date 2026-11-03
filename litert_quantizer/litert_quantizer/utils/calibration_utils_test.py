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

import json

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import quantizer
from litert_quantizer import recipe
from litert_quantizer.utils import calibration_utils
from litert_quantizer.utils import histogram_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils


_RNG = np.random.default_rng(66)

_CALIBRATION_DATASET = {
    "signature_1": [{
        "cache_0": np.zeros(shape=(1, 100, 4, 4), dtype=np.float32),
        "cache_1": np.zeros(shape=(1, 100, 4, 4), dtype=np.float32),
        "positions": np.zeros(shape=(1, 100), dtype=np.int32),
        "tokens": np.zeros(shape=(1, 100), dtype=np.int32),
    }],
    "signature_2": [{
        "cache_0": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
        "cache_1": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
        "positions": (
            _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
        ),
        "tokens": _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32),
    }],
}


def _get_quant_parameters(
    quantized_model: bytes, signature_data: dict[str, list[str]]
) -> list[np.ndarray]:
  """Returns the quantization parameters from the quantized model."""
  quant_params = []
  tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
      quantized_model
  )
  for signature_key, signature_names in signature_data.items():
    signature_runner = tfl_interpreter.get_signature_runner(signature_key)

    for signature_name in signature_names:
      input_details = signature_runner.get_input_details()
      output_details = signature_runner.get_output_details()
      if signature_name in input_details.keys():
        quant_param = input_details[signature_name]["quantization_parameters"][
            "scales"
        ].squeeze()
        quant_params.append(quant_param)
      elif signature_name in output_details.keys():
        output_details = signature_runner.get_output_details()
        quant_param = output_details[signature_name]["quantization_parameters"][
            "scales"
        ].squeeze()
        quant_params.append(quant_param)
      else:
        raise ValueError(
            f"Signature name {signature_name} not found in the model."
        )
  return quant_params


class CalibrationQsvAlignmentUtilsTest(parameterized.TestCase):

  def test_load_calibration_results(self):
    temp_file = self.create_tempfile()
    temp_file.write_text(
        '{"model_qsvs": {"tensor1": {"min": [-1.0], "max": [1.0]}}, "metadata":'
        " {}}"
    )
    results, _ = calibration_utils.load_calibration_results(temp_file.full_path)
    self.assertIn("tensor1", results)
    self.assertSequenceAlmostEqual(results["tensor1"]["min"], [-1.0])
    self.assertSequenceAlmostEqual(results["tensor1"]["max"], [1.0])

  def test_load_legacy_calibration_results(self):
    temp_file = self.create_tempfile()
    temp_file.write_text('{"tensor1": {"min": [-1.0], "max": [1.0]}}')
    results, metadata = calibration_utils.load_calibration_results(
        temp_file.full_path
    )
    self.assertIn("tensor1", results)
    self.assertSequenceAlmostEqual(results["tensor1"]["min"], [-1.0])
    self.assertSequenceAlmostEqual(results["tensor1"]["max"], [1.0])
    self.assertEmpty(metadata)

  def test_calibration_utils_init_fails(self):
    model_path = "non_existent_model.tflite"
    with self.assertRaisesWithPredicateMatch(
        Exception, lambda err: f"{model_path}" in str(err)
    ):
      calibration_utils.CalibrationQsvAlignmentUtils(model_path)

  def test_calibration_utils_init_succeeds(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/single_add.tflite"
    )
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    self.assertNotEmpty(calib_utils._signature_runners)
    self.assertNotEmpty(calib_utils._same_as_input_scale_ops)

  def test_search_tensor_by_signature_name_succeeds_on_unconstrained_op(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/single_add.tflite"
    )
    expected_tensor_name = "PartitionedCall:0"
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    tensor_name = calib_utils._search_tensor_by_signature_name(
        "serving_default", "add"
    )
    self.assertEqual(tensor_name, [expected_tensor_name])

  def test_search_tensor_by_signature_name_succeeds_on_constrained_op(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/single_slice.tflite"
    )
    expected_tensor_name = "slice_input_tensor:0"
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    tensor_name = calib_utils._search_tensor_by_signature_name(
        "slice", "output_0"
    )
    self.assertEqual(tensor_name, [expected_tensor_name])

  def test_align_quant_stats_succeeds(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/toy_model_with_kv_cache_multi_signature.tflite"
    )
    recipe_path = test_utils.get_path_to_datafile(
        "../recipes/default_a8w8_recipe.json"
    )
    signature_data = {
        "signature_1": ["output_1_1"],
        "signature_2": ["output_1_1"],
    }

    # Obtain the calibration results.
    qt = quantizer.Quantizer(model_path, recipe_path)
    qsv = qt.calibrate(_CALIBRATION_DATASET)

    # First quantize the model without aligning the quantization parameters.
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertFalse(
        all(x == quant_params[0] for x in quant_params)
    )  # not equal quantization params.

    # Align the quantization parameters and quantize again.
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    calib_utils.align_quant_stats(qsv, signature_data)
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertTrue(
        all(x == quant_params[0] for x in quant_params)
    )  # equal quantization params.

  def test_update_quant_stats_succeeds(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/toy_model_with_kv_cache_multi_signature.tflite"
    )
    recipe_path = test_utils.get_path_to_datafile(
        "../recipes/default_a8w8_recipe.json"
    )
    signature_data = {
        "signature_1": ["output_1_1"],
        "signature_2": ["output_1_1"],
    }

    # Obtain the calibration results.
    qt = quantizer.Quantizer(model_path, recipe_path)
    qsv = qt.calibrate(_CALIBRATION_DATASET)

    # First quantize the model without updating the `signature_1`.
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertFalse(
        all(x == quant_params[0] for x in quant_params)
    )  # not equal quantization params.

    # Update the `signature_1` with stats from `signature_2`.
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    min_val, max_val = calib_utils.align_quant_stats(  # for min and max only.
        qsv,
        {
            "signature_2": ["output_1_1"],
        },
    )
    calib_utils.update_quant_stats(
        qsv,
        {
            "signature_1": ["output_1_1"],
        },
        min_val,
        max_val,
    )
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertTrue(
        all(x == quant_params[0] for x in quant_params)
    )  # equal quantization params.

  def test_save_and_load_calibration_results_with_histogram(self):
    hist = histogram_utils.DynamicHistogram(max_tensor_bins=64)
    hist.add(np.linspace(-2.0, 4.0, 100))
    payload = {
        "model_qsvs": {
            "tensor1": {
                "min": np.array([[-2.0]]),
                "max": np.array([[4.0]]),
                "num_samples": np.array(100),
                "histogram": hist.to_dict(),
            }
        },
        "metadata": {"num_samples_calibrated": 100},
    }
    temp_file = self.create_tempfile()
    temp_file.write_text(
        json.dumps(payload, cls=calibration_utils.NumpyEncoder)
    )
    loaded_qsvs, metadata = calibration_utils.load_calibration_results(
        temp_file.full_path
    )
    self.assertEqual(metadata, {"num_samples_calibrated": 100})
    np.testing.assert_allclose(loaded_qsvs["tensor1"]["min"], [[-2.0]])
    np.testing.assert_allclose(loaded_qsvs["tensor1"]["max"], [[4.0]])
    self.assertEqual(int(loaded_qsvs["tensor1"]["num_samples"]), 100)
    loaded_hist = histogram_utils.DynamicHistogram.from_dict(
        loaded_qsvs["tensor1"]["histogram"]
    )
    np.testing.assert_array_equal(loaded_hist.counts, hist.counts)
    self.assertAlmostEqual(loaded_hist.lower_bound, hist.lower_bound)
    self.assertAlmostEqual(loaded_hist.bin_width, hist.bin_width)

  def test_align_quant_stats_merges_histograms(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/toy_model_with_kv_cache_multi_signature.tflite"
    )
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    signature_data = {
        "signature_1": ["output_1_1"],
        "signature_2": ["output_1_1"],
    }
    tensor_1 = calib_utils._search_tensor_by_signature_name(
        "signature_1", "output_1_1"
    )[0]
    tensor_2 = calib_utils._search_tensor_by_signature_name(
        "signature_2", "output_1_1"
    )[0]
    self.assertNotEqual(tensor_1, tensor_2)
    hist_1 = histogram_utils.DynamicHistogram(max_tensor_bins=64)
    hist_1.add(np.array([-1.0, 0.0, 1.0]))
    hist_2 = histogram_utils.DynamicHistogram(max_tensor_bins=64)
    hist_2.add(np.array([2.0, 3.0]))
    qsv = {
        tensor_1: {
            "min": np.array(-1.0),
            "max": np.array(1.0),
            "histogram": hist_1.to_dict(),
        },
        tensor_2: {
            "min": np.array(2.0),
            "max": np.array(3.0),
            "histogram": hist_2.to_dict(),
        },
    }

    min_val, max_val = calib_utils.align_quant_stats(qsv, signature_data)

    self.assertEqual(min_val, -1.0)
    self.assertEqual(max_val, 3.0)
    for tensor_name in (tensor_1, tensor_2):
      merged = histogram_utils.DynamicHistogram.from_dict(
          qsv[tensor_name]["histogram"]
      )
      self.assertEqual(np.sum(merged.counts), 5)
      self.assertAlmostEqual(float(merged.global_min[0]), -1.0)
      self.assertAlmostEqual(float(merged.global_max[0]), 3.0)
    # Each tensor gets its own copy of the merged histogram.
    self.assertIsNot(qsv[tensor_1]["histogram"], qsv[tensor_2]["histogram"])

  def test_update_quant_stats_sets_or_drops_histogram(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/toy_model_with_kv_cache_multi_signature.tflite"
    )
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    tensor_name = calib_utils._search_tensor_by_signature_name(
        "signature_1", "output_1_1"
    )[0]
    stale_hist = histogram_utils.DynamicHistogram(max_tensor_bins=64)
    stale_hist.add(np.array([5.0, 6.0]))
    qsv = {
        tensor_name: {
            "min": np.array(5.0),
            "max": np.array(6.0),
            "histogram": stale_hist.to_dict(),
        }
    }
    new_hist = histogram_utils.DynamicHistogram(max_tensor_bins=64)
    new_hist.add(np.array([0.0, 0.5, 1.0]))
    new_hist_dict = new_hist.to_dict()
    target = {"signature_1": ["output_1_1"]}

    # With a histogram: min/max and a copy of the histogram are applied.
    calib_utils.update_quant_stats(
        qsv, target, np.array(0.0), np.array(1.0), histogram=new_hist_dict
    )
    self.assertEqual(qsv[tensor_name]["min"], 0.0)
    self.assertEqual(qsv[tensor_name]["max"], 1.0)
    self.assertIsNot(qsv[tensor_name]["histogram"], new_hist_dict)
    applied = histogram_utils.DynamicHistogram.from_dict(
        qsv[tensor_name]["histogram"]
    )
    np.testing.assert_array_equal(applied.counts, new_hist.counts)

    # Without a histogram: any existing histogram is dropped so quantization
    # falls back to the aligned min/max.
    calib_utils.update_quant_stats(qsv, target, np.array(0.0), np.array(1.0))
    self.assertNotIn("histogram", qsv[tensor_name])

  def test_align_quant_stats_with_histogram_succeeds(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/toy_model_with_kv_cache_multi_signature.tflite"
    )
    signature_data = {
        "signature_1": ["output_1_1"],
        "signature_2": ["output_1_1"],
    }

    qt = quantizer.Quantizer(
        model_path,
        recipe.static_wi8_ai8(
            algorithm_key=quantizer.AlgorithmName.HISTOGRAM_MSE
        ),
    )
    qsv = qt.calibrate(_CALIBRATION_DATASET)

    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    calib_utils.align_quant_stats(qsv, signature_data)
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertTrue(all(x == quant_params[0] for x in quant_params))

  def test_update_quant_stats_with_histogram_succeeds(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/toy_model_with_kv_cache_multi_signature.tflite"
    )
    signature_data = {
        "signature_1": ["output_1_1"],
        "signature_2": ["output_1_1"],
    }
    qt = quantizer.Quantizer(
        model_path,
        recipe.static_wi8_ai8(
            algorithm_key=quantizer.AlgorithmName.HISTOGRAM_MSE
        ),
    )
    qsv = qt.calibrate(_CALIBRATION_DATASET)
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    src_tensor = calib_utils._search_tensor_by_signature_name(
        "signature_2", "output_1_1"
    )[0]
    self.assertIn("histogram", qsv[src_tensor])

    min_val, max_val = calib_utils.align_quant_stats(
        qsv, {"signature_2": ["output_1_1"]}
    )
    calib_utils.update_quant_stats(
        qsv,
        {"signature_1": ["output_1_1"]},
        min_val,
        max_val,
        histogram=qsv[src_tensor]["histogram"],
    )
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertTrue(all(x == quant_params[0] for x in quant_params))


if __name__ == "__main__":
  absltest.main()
