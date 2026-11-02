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
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np

import os
import io
from litert_quantizer import algorithm_manager
from litert_quantizer import calibrator
from litert_quantizer.examples.selfie_segmentation import quantize_selfie_segmentation
from litert_quantizer.utils import histogram_range_utils
from litert_quantizer.utils import test_utils

AlgorithmName = algorithm_manager.AlgorithmName
CalibrationMode = calibrator.CalibrationMode


class QuantizeSelfieSegmentationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._model_path = str(
        pathlib.Path(test_utils.get_path_to_datafile('../../tests/models'))
        / 'conv_fc_mnist.tflite'
    )
    self._image_dir = test_utils.get_path_to_datafile('data')

  def test_load_and_preprocess_images_produces_expected_shapes(self):
    input_name, input_shape, _ = (
        quantize_selfie_segmentation.get_model_input_spec(self._model_path)
    )
    self.assertNotEmpty(input_name)
    calib_samples, eval_samples = (
        quantize_selfie_segmentation.load_and_preprocess_images(
            self._image_dir,
            target_shape=input_shape,
            num_augmented_samples=10,
        )
    )
    self.assertLen(calib_samples, 10)
    self.assertGreaterEqual(len(eval_samples), 6)
    for arr in calib_samples:
      self.assertEqual(arr.shape, input_shape)
      self.assertEqual(arr.dtype, np.float32)
      self.assertGreaterEqual(float(np.min(arr)), 0.0)
      self.assertLessEqual(float(np.max(arr)), 1.0)

  @parameterized.named_parameters(
      ('min_max_profiler', AlgorithmName.MIN_MAX_UNIFORM_QUANT),
      ('histogram_percentile_profiler', AlgorithmName.HISTOGRAM_PERCENTILE),
      ('histogram_mse_profiler', AlgorithmName.HISTOGRAM_MSE),
  )
  def test_calibrate_quantize_and_evaluate_a8w8(self, algorithm_key):
    if algorithm_key in (
        AlgorithmName.HISTOGRAM_PERCENTILE,
        AlgorithmName.HISTOGRAM_MSE,
    ) and not hasattr(calibrator._pywrap_tfl_calibration, 'CalibrationSession'):
      self.skipTest('Requires _pywrap_tfl_calibration.CalibrationSession')
    input_name, input_shape, _ = (
        quantize_selfie_segmentation.get_model_input_spec(self._model_path)
    )
    calib_samples, eval_samples = (
        quantize_selfie_segmentation.load_and_preprocess_images(
            self._image_dir,
            target_shape=input_shape,
            num_augmented_samples=8,
        )
    )
    calib_dataset = quantize_selfie_segmentation.build_calibration_dataset(
        input_name, calib_samples
    )
    float_outputs = quantize_selfie_segmentation._run_model_on_samples(
        self._model_path, eval_samples
    )
    out_dir = self.create_tempdir().full_path

    quant_bytes, summary, report = (
        quantize_selfie_segmentation.calibrate_and_quantize_a8w8(
            float_model_path=self._model_path,
            calibration_data=calib_dataset,
            eval_samples=eval_samples,
            float_outputs=float_outputs,
            algorithm_key=algorithm_key,
            calibration_mode=CalibrationMode.CALIBRATION_PROFILER_BASED,
            output_dir=out_dir,
            model_name=f'test_{algorithm_key.value}',
        )
    )
    self.assertNotEmpty(quant_bytes)
    self.assertGreater(report.compression_ratio, 1.0)
    self.assertGreater(report.cosine_similarity, 0.90)

    if algorithm_key in (
        AlgorithmName.HISTOGRAM_PERCENTILE,
        AlgorithmName.HISTOGRAM_MSE,
    ):
      self.assertGreater(summary.num_histogram_tensors, 0)
      self.assertLessEqual(summary.mean_range_ratio, 1.0)
      self.assertGreaterEqual(summary.mean_bit_gain, 0.0)
    else:
      self.assertEqual(summary.num_histogram_tensors, 0)

  def test_uniform_quant_mse_decomposition(self):
    decompose = (
        quantize_selfie_segmentation._compute_uniform_quant_mse_decomposition
    )
    scale = 2.0 / 255.0
    granular, overload, total = decompose(
        np.array([-0.5, 0.0, 0.5, 3.0]), np.ones(4), clip_min=-1.0, clip_max=1.0
    )
    # Three in-range bins contribute uniform rounding noise; the outlier at 3.0
    # is clamped to ~1.0 (within one quantization step).
    self.assertAlmostEqual(granular, 0.75 * scale**2 / 12.0, delta=1e-9)
    self.assertAlmostEqual(overload, (3.0 - 1.0) ** 2 / 4.0, delta=2 * scale)
    self.assertAlmostEqual(total, granular + overload)

  def test_analyze_tail_fraction_counts_only_values_actually_clipped(self):
    # Mirrors a ReLU output from the selfie model: raw_min == 0, most of the
    # mass in the zero bin, and one far outlier. The MSE search can return a
    # clip_min just above zero, but quantization always extends the range to
    # include zero, so the zero bin is not clipped.
    counts = np.zeros(200, dtype=np.int64)
    counts[0] = 600
    counts[1:40] = 10
    counts[199] = 1
    qsv = {
        'histogram': {
            'axis': None,
            'min': np.array([0.0]),
            'max': np.array([1.99]),
            'channels': [{
                'hist_counts': counts,
                'lower_bound': -0.005,
                'bin_width': 0.01,
                'min': np.array([0.0]),
                'max': np.array([1.99]),
            }],
        }
    }
    with mock.patch.object(
        histogram_range_utils,
        'mse_range_from_histogram',
        autospec=True,
        return_value=(0.01, 1.0),
    ):
      summary = quantize_selfie_segmentation.analyze_histogram_qsvs(
          {'t': qsv},
          algorithm_key=AlgorithmName.HISTOGRAM_MSE,
          calibration_mode=CalibrationMode.CALIBRATION_PROFILER_BASED,
          wall_time_sec=0.0,
      )
    (analysis,) = summary.top_outlier_tensors
    self.assertAlmostEqual(
        analysis.tail_clip_fraction, 1.0 / np.sum(counts), places=9
    )

  def test_evaluate_identical_model_gives_perfect_agreement(self):
    _, input_shape, _ = quantize_selfie_segmentation.get_model_input_spec(
        self._model_path
    )
    _, eval_samples = quantize_selfie_segmentation.load_and_preprocess_images(
        self._image_dir, target_shape=input_shape, num_augmented_samples=6
    )
    float_outputs = quantize_selfie_segmentation._run_model_on_samples(
        self._model_path, eval_samples
    )
    with open(self._model_path, 'rb') as f:
      model_bytes = f.read()
    report = quantize_selfie_segmentation.evaluate_segmentation_quality(
        float_outputs=float_outputs,
        quant_model_bytes=model_bytes,
        eval_samples=eval_samples,
        model_name='identity',
        algorithm_key='none',
        calibration_mode=CalibrationMode.CALIBRATION_PROFILER_BASED,
        float_model_size_bytes=len(model_bytes),
    )
    self.assertEqual(report.output_mse, 0.0)
    self.assertAlmostEqual(report.cosine_similarity, 1.0)
    self.assertAlmostEqual(report.kl_divergence, 0.0)
    self.assertEqual(report.pixel_top1_agreement, 1.0)
    self.assertEqual(report.mean_iou, 1.0)
    self.assertNotEmpty(report.per_class_iou)
    self.assertTrue(all(v == 1.0 for v in report.per_class_iou.values()))
    self.assertEqual(report.compression_ratio, 1.0)

  @absltest.skipIf(
      not hasattr(calibrator._pywrap_tfl_calibration, 'CalibrationSession'),
      'Requires _pywrap_tfl_calibration.CalibrationSession',
  )
  def test_main_cli_execution_with_flags(self):
    out_dir = self.create_tempdir().full_path
    out_tflite = str(pathlib.Path(out_dir) / 'model_mse.tflite')
    with flagsaver.flagsaver(
        model_path=self._model_path,
        dataset_dir=self._image_dir,
        calibration_algorithm='HISTOGRAM_MSE',
        calibration_method='CALIBRATION_PROFILER_BASED',
        output_tflite_path=out_tflite,
        num_calibration_steps=4,
        output_dir=out_dir,
    ):
      quantize_selfie_segmentation.main(['quantize_selfie_segmentation.py'])

    self.assertTrue(pathlib.Path(out_tflite).exists())
    self.assertGreater(pathlib.Path(out_tflite).stat().st_size, 0)


if __name__ == '__main__':
  absltest.main()
