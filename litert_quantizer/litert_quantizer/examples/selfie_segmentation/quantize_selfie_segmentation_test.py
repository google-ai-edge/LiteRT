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

# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import pathlib
from unittest import mock
import urllib.request

from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np
from PIL import Image

import os
import io
from litert_quantizer import calibrator
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.examples.selfie_segmentation import quantize_selfie_segmentation
from litert_quantizer.utils import test_utils

RangeMethod = naive_min_max_quantize.ActivationRangeMethod
CalibrationMode = calibrator.CalibrationMode


def _write_synthetic_images(image_dir: str, num_images: int = 2) -> list[str]:
  """Writes random RGB JPEG images and returns their paths."""
  rng = np.random.default_rng(0)
  paths = []
  for i in range(num_images):
    pixels = rng.integers(0, 256, size=(40, 48, 3), dtype=np.uint8)
    path = str(pathlib.Path(image_dir) / f'image_{i}.jpg')
    Image.fromarray(pixels).save(path)
    paths.append(path)
  return paths


class QuantizeSelfieSegmentationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._model_path = str(
        pathlib.Path(test_utils.get_path_to_datafile('../../tests/models'))
        / 'conv_fc_mnist.tflite'
    )
    self._image_paths = _write_synthetic_images(self.create_tempdir().full_path)

  def test_mse_range_beats_min_max_on_selfie_model(self):
    if not os.path.exists(quantize_selfie_segmentation.LOCAL_SAMPLE_DIR):
      self.skipTest('Requires internal async_segmentation sample data.')
    # Regression guard for histogram range quality on the real selfie model
    # (A8W8, 32 calibration samples, profiler-based calibration). README
    # results: min_max 8.24 dB SQNR and 64.43% mIoU, histogram_mse 9.72 dB and
    # 68.44% mIoU.
    model_path = quantize_selfie_segmentation.LOCAL_MODEL_PATH
    input_name, input_shape = quantize_selfie_segmentation.get_model_input_spec(
        model_path
    )
    calib_samples, eval_samples = (
        quantize_selfie_segmentation.load_and_preprocess_images(
            [quantize_selfie_segmentation.LOCAL_IMAGE_PATH],
            target_shape=input_shape,
            num_augmented_samples=32,
        )
    )
    calib_dataset = quantize_selfie_segmentation.build_calibration_dataset(
        input_name, calib_samples
    )
    float_outputs = quantize_selfie_segmentation.run_model_on_samples(
        model_path, eval_samples
    )
    reports = {}
    for method in (RangeMethod.MIN_MAX, RangeMethod.HISTOGRAM_MSE):
      reports[method] = (
          quantize_selfie_segmentation.calibrate_and_quantize_a8w8(
              float_model_path=model_path,
              calibration_data=calib_dataset,
              eval_samples=eval_samples,
              float_outputs=float_outputs,
              range_method=method,
              calibration_mode=CalibrationMode.CALIBRATION_PROFILER_BASED,
              output_dir=self.create_tempdir().full_path,
              model_name=f'selfie_{method.value}',
          )
      )
    min_max = reports[RangeMethod.MIN_MAX]
    mse = reports[RangeMethod.HISTOGRAM_MSE]
    self.assertGreater(mse.output_sqnr_db, min_max.output_sqnr_db + 1.0)
    self.assertGreater(mse.mean_iou, min_max.mean_iou + 0.02)

  def test_resolve_inputs_downloads_defaults_when_sample_dir_missing(self):
    temp_root = pathlib.Path(self.create_tempdir().full_path)
    out_dir = self.create_tempdir().full_path
    self.enter_context(
        mock.patch.object(
            quantize_selfie_segmentation,
            'LOCAL_SAMPLE_DIR',
            str(temp_root / 'missing_async_segmentation'),
        )
    )
    mock_open = self.enter_context(
        mock.patch.object(
            urllib.request.OpenerDirector,
            'open',
            autospec=True,
            side_effect=lambda *args, **kwargs: io.BytesIO(b'downloaded-data'),
        )
    )

    model_path, image_paths = quantize_selfie_segmentation.resolve_inputs(
        '', [], out_dir
    )

    self.assertEqual(
        model_path,
        str(pathlib.Path(out_dir) / 'selfie_multiclass_256x256.tflite'),
    )
    self.assertSequenceEqual(
        image_paths, (str(pathlib.Path(out_dir) / 'portrait.jpg'),)
    )
    self.assertEqual(mock_open.call_count, 2)
    with open(model_path, 'rb') as f:
      self.assertEqual(f.read(), b'downloaded-data')
    with open(image_paths[0], 'rb') as f:
      self.assertEqual(f.read(), b'downloaded-data')

  def test_main_runs_default_configs(self):
    out_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver(
        float_model_path=self._model_path,
        image_paths=self._image_paths,
        num_calib_samples=4,
        output_dir=out_dir,
    ):
      quantize_selfie_segmentation.main(['quantize_selfie_segmentation.py'])

    for model_name in (
        'selfie_a8w8_minmax_profiler',
        'selfie_a8w8_hist_pct_profiler',
        'selfie_a8w8_hist_mse_profiler',
        'selfie_a8w8_hist_mse_preserve',
    ):
      tflite_path = pathlib.Path(out_dir) / f'{model_name}.tflite'
      self.assertTrue(tflite_path.exists())
      self.assertGreater(tflite_path.stat().st_size, 0)


if __name__ == '__main__':
  absltest.main()
