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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer.utils import histogram_range_utils
from litert_quantizer.utils import histogram_utils


def _create_tf_golden_histogram() -> np.ndarray:
  """Creates the 501-bin TF calibration test histogram."""
  hist_freq = np.zeros(501, dtype=np.int64)
  # Outliers at both ends.
  hist_freq[0] = 1
  hist_freq[-1] = 1
  # Triangular distribution centered at bin 250 (bins 50..450).
  hist_freq[250] = 1000
  for i in range(1, 201):
    hist_freq[250 - i] = 1000 - i
    hist_freq[250 + i] = 1000 - i
  return hist_freq


class HistogramRangeUtilsTest(parameterized.TestCase):

  def test_get_hist_mids(self):
    mids = histogram_range_utils.get_hist_mids(
        num_bins=4, lower_bound=0.0, bin_width=2.0
    )
    np.testing.assert_allclose(mids, [1.0, 3.0, 5.0, 7.0])
    self.assertEqual(
        histogram_range_utils.get_hist_mids(0, 0.0, 1.0).size, 0
    )

  def test_clamp_and_snap_zero_bin(self):
    mids = np.array([-2.0, -0.1, 0.4, 2.0])
    clamped = histogram_range_utils.clamp_and_snap_zero_bin(
        mids, bin_width=0.5, data_min=-1.5, data_max=1.5
    )
    np.testing.assert_allclose(clamped, [-1.5, 0.0, 0.4, 1.5])
    # Returns mids unchanged when data_min or data_max is None or inverted.
    np.testing.assert_array_equal(
        histogram_range_utils.clamp_and_snap_zero_bin(mids, 0.5, None, 1.5),
        mids,
    )
    np.testing.assert_array_equal(
        histogram_range_utils.clamp_and_snap_zero_bin(mids, 0.5, 2.0, -2.0),
        mids,
    )

  def test_percentile_range_matches_tf_golden(self):
    hist_freq = _create_tf_golden_histogram()
    min_val, max_val = histogram_range_utils.percentile_range(
        hist_freq,
        lower_bound=0.0,
        bin_width=1.0,
    )
    # Exact TF golden values: bin 50 midpoint (50.5) and bin 450 midpoint
    # (450.5).
    self.assertAlmostEqual(min_val, 50.5)
    self.assertAlmostEqual(max_val, 450.5)

  def test_mse_range_matches_tf_golden(self):
    hist_freq = _create_tf_golden_histogram()
    min_val, max_val = histogram_range_utils.mse_range(
        hist_freq,
        lower_bound=0.0,
        bin_width=1.0,
    )
    # Matches TF's assertAllInRange(min_value, 49, 51) and
    # (max_value, 449, 451).
    self.assertBetween(min_val, 49.0, 51.0)
    self.assertBetween(max_val, 449.0, 451.0)

  def test_percentile_range_uniform_distribution(self):
    rng = np.random.default_rng(42)
    data = rng.uniform(-10.0, 10.0, size=100_000)
    hist = histogram_utils.DynamicHistogram(max_tensor_bins=2048)
    hist.add(data)

    min_val, max_val = histogram_range_utils.percentile_range_from_histogram(
        hist, min_percentile=5.0, max_percentile=95.0
    )
    self.assertAlmostEqual(min_val, -9.0, delta=0.1)
    self.assertAlmostEqual(max_val, 9.0, delta=0.1)

  @parameterized.named_parameters(
      ('asymmetric_signed', False, True),
      ('symmetric_signed', True, True),
      ('symmetric_unsigned', True, False),
  )
  def test_mse_range_with_outliers(self, symmetric, signed):
    rng = np.random.default_rng(123)
    inliers = rng.uniform(-10.0, 10.0, size=500_000)
    outliers = np.array([-14.0, 14.0])

    hist = histogram_utils.DynamicHistogram(max_tensor_bins=512)
    hist.add(inliers)
    hist.add(outliers)

    clip_min, clip_max = histogram_range_utils.mse_range_from_histogram(
        hist,
        symmetric=symmetric,
        signed=signed,
    )
    # Clipped range must be narrower than the raw outlier range [-14, 14].
    self.assertGreater(clip_min, -12.0)
    self.assertLess(clip_max, 12.0)
    if symmetric:
      self.assertAlmostEqual(clip_min, -clip_max)

    # Verify reconstruction MSE on the histogram is smaller with MSE range than
    # with naive min/max range [-14, 14].
    assert hist.bin_width is not None
    mids = histogram_range_utils.get_hist_mids(
        len(hist.counts), hist.lower_bound, hist.bin_width
    )
    freq = hist.counts.astype(np.float64)
    # pylint: disable=protected-access
    errors = histogram_range_utils._compute_weighted_mse_from_moments(
        histogram_range_utils._precompute_histogram_moments(mids, freq),
        np.array([clip_min, -14.0]),
        np.array([clip_max, 14.0]),
        num_bits=8,
        symmetric=symmetric,
        signed=signed,
    )
    # pylint: enable=protected-access
    self.assertLess(errors[0], errors[1])

  @parameterized.named_parameters(
      (
          'one_sided',
          0.0,
          10.0,
          [13.5, 14.0],
          (0.0, 0.2),
          (9.5, 12.0),
      ),
      (
          'skewed',
          -5.0,
          10.0,
          [-5.2, 13.5, 14.0],
          (-5.2, -4.5),
          (9.5, 12.0),
      ),
  )
  def test_mse_range_one_sided_and_skewed_distributions(
      self, low, high, outliers, expected_min_bounds, expected_max_bounds
  ):
    rng = np.random.default_rng(7)
    inliers = rng.uniform(low, high, size=500_000)
    hist = histogram_utils.DynamicHistogram(max_tensor_bins=1024)
    hist.add(inliers)
    hist.add(np.array(outliers))

    clip_min, clip_max = histogram_range_utils.mse_range_from_histogram(hist)
    self.assertBetween(clip_min, *expected_min_bounds)
    self.assertBetween(clip_max, *expected_max_bounds)

  def test_mse_and_percentile_range_with_relu_zero_spike(self):
    rng = np.random.default_rng(19)
    # Simulate a sparse ReLU activation tensor where 85% of elements are exact
    # zeros and 15% are positive in [0.0, 1.0], followed by a second batch with
    # a rare outlier at 2.1 that triggers DynamicHistogram bin doubling.
    batch1 = np.concatenate([
        np.zeros(425_000, dtype=np.float32),
        rng.uniform(0.0, 1.0, size=75_000).astype(np.float32),
    ])
    batch2 = np.array([0.0, 0.5, 2.1], dtype=np.float32)

    hist = histogram_utils.DynamicHistogram(max_tensor_bins=512)
    hist.add(batch1)
    hist.add(batch2)

    p_min, p_max = histogram_range_utils.percentile_range_from_histogram(hist)
    m_min, m_max = histogram_range_utils.mse_range_from_histogram(hist)
    # Both percentile and MSE range must snap the 0.0-straddling bin to 0.0
    # and preserve the [0.0, 1.0] active positive body without over-clipping.
    self.assertEqual(p_min, 0.0)
    self.assertEqual(m_min, 0.0)
    self.assertBetween(p_max, 0.95, 1.15)
    self.assertBetween(m_max, 0.90, 1.35)

  def test_mse_range_16bit_symmetric(self):
    hist_freq = _create_tf_golden_histogram()
    clip_min, clip_max = histogram_range_utils.mse_range(
        hist_freq,
        lower_bound=-250.0,
        bin_width=1.0,
        num_bits=16,
        symmetric=True,
    )
    self.assertAlmostEqual(clip_min, -clip_max)
    self.assertGreater(clip_max, 200.0)

  def test_edge_cases_empty_and_single_bin(self):
    # Empty / all-zero counts with fallback.
    self.assertEqual(
        histogram_range_utils.percentile_range(
            np.zeros(10, dtype=np.int64),
            lower_bound=-1.0,
            bin_width=0.1,
            fallback_min=-2.0,
            fallback_max=3.0,
        ),
        (-2.0, 3.0),
    )
    self.assertEqual(
        histogram_range_utils.mse_range(
            np.zeros(10, dtype=np.int64),
            lower_bound=-1.0,
            bin_width=0.1,
            fallback_min=-2.0,
            fallback_max=3.0,
        ),
        (-2.0, 3.0),
    )
    # Single-bin histogram.
    self.assertEqual(
        histogram_range_utils.mse_range(
            np.array([10], dtype=np.int64),
            lower_bound=1.0,
            bin_width=0.5,
            fallback_min=1.2,
            fallback_max=1.2,
        ),
        (1.2, 1.2),
    )
    # Uninitialized DynamicHistogram falls back cleanly.
    uninit_hist = histogram_utils.DynamicHistogram()
    self.assertEqual(
        histogram_range_utils.percentile_range_from_histogram(
            uninit_hist, fallback_min=-1.0, fallback_max=1.0
        ),
        (-1.0, 1.0),
    )
    self.assertEqual(
        histogram_range_utils.mse_range_from_histogram(
            uninit_hist, fallback_min=-1.0, fallback_max=1.0
        ),
        (-1.0, 1.0),
    )

  def test_edge_cases_without_fallback(self):
    zeros = np.zeros(10, dtype=np.int64)
    self.assertEqual(
        histogram_range_utils.percentile_range(zeros, -1.0, 0.1), (-1.0, -1.0)
    )
    self.assertEqual(
        histogram_range_utils.mse_range(zeros, -1.0, 0.1), (-1.0, -1.0)
    )
    # bin_width=None (uninitialized histogram) is treated as empty.
    counts = np.array([1, 2, 3], dtype=np.int64)
    self.assertEqual(
        histogram_range_utils.percentile_range(counts, 2.0, None), (2.0, 2.0)
    )
    self.assertEqual(
        histogram_range_utils.mse_range(counts, 2.0, None), (2.0, 2.0)
    )
    # A single non-zero bin returns that bin's midpoint.
    self.assertEqual(
        histogram_range_utils.mse_range(
            np.array([0, 0, 5, 0], dtype=np.int64), 0.0, 1.0
        ),
        (2.5, 2.5),
    )
    uninit_hist = histogram_utils.DynamicHistogram()
    self.assertEqual(
        histogram_range_utils.percentile_range_from_histogram(uninit_hist),
        (0.0, 0.0),
    )
    self.assertEqual(
        histogram_range_utils.mse_range_from_histogram(uninit_hist),
        (0.0, 0.0),
    )

  def test_mse_range_with_two_bins_snapped_to_zero(self):
    # 0.0 is a bin edge, so the two bins around it both snap to 0.0 and form a
    # degenerate (0.0, 0.0) candidate.
    min_val, max_val = histogram_range_utils.mse_range(
        np.array([1, 1, 1, 1], dtype=np.int64),
        lower_bound=-1.0,
        bin_width=0.5,
        data_min=-1.0,
        data_max=1.0,
    )
    self.assertEqual((min_val, max_val), (-0.75, 0.75))

  @parameterized.named_parameters(
      (
          'negative_min_percentile',
          histogram_range_utils.percentile_range,
          {'min_percentile': -1.0, 'max_percentile': 99.0},
          'Percentiles must satisfy',
      ),
      (
          'max_percentile_above_100',
          histogram_range_utils.percentile_range,
          {'min_percentile': 1.0, 'max_percentile': 101.0},
          'Percentiles must satisfy',
      ),
      (
          'min_percentile_above_max_percentile',
          histogram_range_utils.percentile_range,
          {'min_percentile': 90.0, 'max_percentile': 10.0},
          'Percentiles must satisfy',
      ),
      (
          'num_bits_less_than_one',
          histogram_range_utils.mse_range,
          {'num_bits': 0},
          'num_bits must be >= 1',
      ),
  )
  def test_invalid_arguments_raise_value_error(self, fn, kwargs, regex):
    counts = np.array([1, 2, 3], dtype=np.int64)
    with self.assertRaisesRegex(ValueError, regex):
      fn(counts, 0.0, 1.0, **kwargs)


if __name__ == '__main__':
  absltest.main()
