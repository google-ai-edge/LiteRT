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

"""Test for validation_utils."""

from absl.testing import absltest
import numpy as np

from litert_quantizer.utils import validation_utils


class ValidationUtilTest(absltest.TestCase):

  def test_preprocess_same_size_arrays_nan(self):
    data1 = [1, 2, np.nan]
    data2 = [1, 2, 3]
    processed_data1, processed_data2 = (
        validation_utils._preprocess_same_size_arrays(data1, data2)
    )
    self.assertAlmostEqual(processed_data1[2], 1e-9)
    self.assertListEqual(list(processed_data2), data2)

  def test_preprocess_same_size_arrays_inf(self):
    data1 = [1, 2, -np.inf]
    data2 = [1, 2, np.inf]
    processed_data1, processed_data2 = (
        validation_utils._preprocess_same_size_arrays(data1, data2)
    )
    self.assertAlmostEqual(processed_data1[2], -1e9)
    self.assertAlmostEqual(processed_data2[2], 1e9)

  def test_preprocess_same_size_arrays_fails_with_incorrect_shape(self):
    data1 = [1, 2, 3]
    data2 = [1, 2]
    self.assertRaises(
        ValueError, validation_utils._preprocess_same_size_arrays, data1, data2
    )

  def test_mean_squared_difference(self):
    data1 = [1, 2, 3]
    data2 = [1, 2, 3]
    result = validation_utils.mean_squared_difference(data1, data2)
    self.assertEqual(result, 0)

  def test_mean_squared_difference_multidim(self):
    data1 = [[1, 2], [4, 5]]
    data2 = [[1, 3], [2, 2]]
    result = validation_utils.mean_squared_difference(data1, data2)
    self.assertAlmostEqual(result, 3.5)

  def test_mean_squared_difference_0d(self):
    data1 = []
    data2 = []
    result = validation_utils.mean_squared_difference(data1, data2)
    self.assertEqual(result, 0)

  def test_median_diff_ratio(self):
    data1 = [1, 2, 3]
    data2 = [1, 2, 3]
    result = validation_utils.median_diff_ratio(data1, data2)
    self.assertEqual(result, 0)

  def test_median_diff_ratio_multidim(self):
    data1 = [[1, 2], [4, 5]]
    data2 = [[1, 3], [2, 2]]
    result = validation_utils.median_diff_ratio(data1, data2)
    self.assertAlmostEqual(result, 0.6666664)

  def test_median_diff_ratio_0d(self):
    data1 = []
    data2 = []
    result = validation_utils.median_diff_ratio(data1, data2)
    self.assertEqual(result, 0)

  def test_cosine_similarity(self):
    data1 = [1, 2, 3]
    data2 = [1, 2, 3]
    result = validation_utils.cosine_similarity(data1, data2)
    self.assertAlmostEqual(result, 1.0, 6)

  def test_cosine_similarity_perpendicular(self):
    data1 = [1, 0, 0]
    data2 = [0, 1, 0]
    result = validation_utils.cosine_similarity(data1, data2)
    self.assertAlmostEqual(result, 0.0, 6)

  def test_cosine_similarity_multidim(self):
    data1 = [[1, 2], [4, 5]]
    data2 = [[1, 3], [2, 2]]
    result = validation_utils.cosine_similarity(data1, data2)
    self.assertAlmostEqual(result, 0.86881, 6)

  def test_cosine_similarity_0d(self):
    data1 = []
    data2 = []
    result = validation_utils.cosine_similarity(data1, data2)
    self.assertEqual(result, 0)

  def test_kl_divergence(self):
    data1 = [0.5, 0.5]
    data2 = [0.1, 0.9]
    result = validation_utils.kl_divergence(data1, data2)
    self.assertAlmostEqual(result, 0.36808, 4)

  def test_kl_divergence_zero_in_q(self):
    data1 = [0, 1]
    data2 = [1, 0]
    result = validation_utils.kl_divergence(data1, data2)
    self.assertAlmostEqual(result, 20.7232658, 4)

  def test_kl_divergence_negative_values(self):
    data1 = [-1, 1]
    data2 = [1, -1]
    result = validation_utils.kl_divergence(data1, data2)
    self.assertAlmostEqual(result, 20.7232658, 4)

  def test_kl_divergence_0d(self):
    data1 = []
    data2 = []
    result = validation_utils.kl_divergence(data1, data2)
    self.assertEqual(result, 0)

  def test_get_validation_func_kl_divergence(self):
    func = validation_utils.get_validation_func(
        validation_utils.ValidationErrorMetric.KL_DIVERGENCE
    )
    self.assertEqual(func, validation_utils.kl_divergence)

  def test_signal_to_noise_ratio_0d(self):
    data1 = []
    data2 = []
    result = validation_utils.signal_to_noise_ratio(data1, data2)
    self.assertEqual(result, 0)

  def test_signal_to_noise_ratio_identical(self):
    data1 = [1, 2, 3]
    data2 = [1, 2, 3]
    result = validation_utils.signal_to_noise_ratio(data1, data2)
    self.assertGreater(result, 1e8)  # mse=0, so snr should be large

  def test_signal_to_noise_ratio_with_noise(self):
    data1 = [2, 3, 4]
    data2 = [1, 2, 3]
    result = validation_utils.signal_to_noise_ratio(data1, data2)
    self.assertAlmostEqual(result, 14 / 3, places=5)

  def test_signal_to_noise_ratio_simple(self):
    data1 = [1, 1]
    data2 = [1, 0]
    result = validation_utils.signal_to_noise_ratio(data1, data2)
    self.assertAlmostEqual(result, 1.0, places=5)

  def test_get_validation_func_snr(self):
    func = validation_utils.get_validation_func(
        validation_utils.ValidationErrorMetric.SNR
    )
    self.assertEqual(func, validation_utils.signal_to_noise_ratio)


if __name__ == "__main__":
  absltest.main()
