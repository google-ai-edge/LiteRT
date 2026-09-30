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

"""Tests for dynamic histogram utility."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer.utils import histogram_utils


class DynamicHistogramTensorTest(parameterized.TestCase):
  """Tests for DynamicHistogram in per-tensor mode (axis=None)."""

  def test_initialization(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=0.1
    )
    self.assertFalse(hist.initialized)

    hist.add(np.array([0.0, 0.5]))
    self.assertTrue(hist.initialized)
    self.assertEqual(hist.bin_width, 0.1)
    self.assertEqual(hist.lower_bound, 0.0)
    self.assertLen(hist.counts, 5)
    self.assertEqual(hist.global_min, 0.0)
    self.assertEqual(hist.global_max, 0.5)

  def test_add_within_range(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist.add(np.array([0.5, 1.5, 2.5]))
    np.testing.assert_array_equal(hist.counts, [1, 2])

  def test_add_multidimensional_data(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist.add(np.array([[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]]))
    np.testing.assert_array_equal(hist.counts, [2, 4])

  def test_expand_right(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist.add(np.array([0.0]))
    self.assertLen(hist.counts, 1)

    hist.add(np.array([2.5]))
    self.assertLen(hist.counts, 3)
    self.assertEqual(hist.lower_bound, 0.0)
    np.testing.assert_array_equal(hist.counts, [1, 0, 1])

  def test_expand_left(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist.add(np.array([2.0]))

    hist.add(np.array([0.5]))
    self.assertLen(hist.counts, 3)
    self.assertEqual(hist.lower_bound, 0.0)
    np.testing.assert_array_equal(hist.counts, [1, 0, 1])

  def test_expand_left_and_right(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist.add(np.array([2.0]))
    self.assertLen(hist.counts, 1)
    self.assertEqual(hist.lower_bound, 2.0)

    hist.add(np.array([0.0, 5.0]))
    self.assertLen(hist.counts, 5)
    self.assertEqual(hist.lower_bound, 0.0)
    np.testing.assert_array_equal(hist.counts, [1, 0, 1, 0, 1])

  def test_doubling_on_left(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=4, initial_bin_width=1.0
    )
    hist.add(np.array([2.0]))

    hist.add(np.array([-2.0]))
    self.assertEqual(hist.bin_width, 2.0)
    self.assertEqual(hist.lower_bound, -2.0)
    self.assertLen(hist.counts, 3)
    np.testing.assert_array_equal(hist.counts, [1, 0, 1])

  def test_serialization(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist.add(np.array([0.5, 1.5, 2.5]))

    d = hist.to_dict()
    self.assertIn('min', d)
    self.assertIn('max', d)
    self.assertIn('axis', d)
    self.assertIn('channels', d)
    self.assertIsNone(d['axis'])
    self.assertLen(d['channels'], 1)

    channel_d = d['channels'][0]
    self.assertIn('hist_counts', channel_d)
    self.assertIn('bin_width', channel_d)
    self.assertIn('lower_bound', channel_d)
    self.assertIn('min', channel_d)
    self.assertIn('max', channel_d)

    hist2 = histogram_utils.DynamicHistogram.from_dict(d, max_tensor_bins=10)
    self.assertEqual(hist2.bin_width, hist.bin_width)
    self.assertEqual(hist2.lower_bound, hist.lower_bound)
    np.testing.assert_array_equal(hist2.global_min, hist.global_min)
    np.testing.assert_array_equal(hist2.global_max, hist.global_max)
    np.testing.assert_array_equal(hist2.counts, hist.counts)

  def test_merge_uninitialized(self):
    hist1 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist2 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )

    # hist1 uninitialized, hist2 uninitialized.
    hist1.merge(hist2)
    self.assertFalse(hist1.initialized)

    # hist1 initialized, hist2 uninitialized.
    hist1.add(np.array([1.0]))
    hist1.merge(hist2)
    self.assertTrue(hist1.initialized)
    np.testing.assert_array_equal(hist1.counts, [1])

    # hist1 uninitialized, hist2 initialized.
    hist3 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist4 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist4.add(np.array([2.0]))
    hist3.merge(hist4)
    self.assertTrue(hist3.initialized)
    np.testing.assert_array_equal(hist3.counts, [1])
    self.assertEqual(hist3.lower_bound, 2.0)

  def test_merge_same_bin_width_and_range(self):
    hist1 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist1.add(np.array([0.0, 1.5]))
    self.assertEqual(hist1.lower_bound, 0.0)
    np.testing.assert_array_equal(hist1.counts, [1, 1])

    hist2 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist2.add(np.array([0.1, 1.6]))
    self.assertEqual(hist2.lower_bound, 0.1)
    np.testing.assert_array_equal(hist2.counts, [1, 1])

    hist1.merge(hist2)
    np.testing.assert_array_equal(hist1.counts, [2, 2, 0])
    self.assertEqual(hist1.lower_bound, 0.0)

  def test_merge_different_bin_widths(self):
    hist1 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist1.add(np.array([0.0, 1.5]))

    hist2 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=2.0
    )
    hist2.add(np.array([0.0, 3.0]))

    hist1.merge(hist2)
    self.assertEqual(hist1.bin_width, 2.0)
    self.assertEqual(hist1.lower_bound, 0.0)
    np.testing.assert_array_equal(hist1.counts, [3, 1])

  def test_nan_inf_filtering(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0
    )
    hist.add(np.array([0.5, np.nan, 1.5, np.inf, 2.5, -np.inf]))
    # Only finite values [0.5, 1.5, 2.5] should be added
    np.testing.assert_array_equal(hist.counts, [1, 2])
    self.assertEqual(hist.global_min, 0.5)
    self.assertEqual(hist.global_max, 2.5)


class DynamicHistogramChannelTest(parameterized.TestCase):
  """Tests for DynamicHistogram in per-channel mode (axis is not None)."""

  def test_per_channel_initialization_and_add(self):
    # 2 channels, axis=-1.
    hist = histogram_utils.DynamicHistogram(max_tensor_bins=10, axis=-1)
    self.assertFalse(hist.initialized)

    # Data shape (3, 2) -> 2 channels, 3 samples per channel.
    # Channel 0: [0.0, 1.0, 2.0].
    # Channel 1: [10.0, 11.0, 12.0].
    data = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    hist.add(data)
    self.assertTrue(hist.initialized)

    # Check bounds are arrays
    np.testing.assert_array_equal(hist.global_min, [0.0, 10.0])
    np.testing.assert_array_equal(hist.global_max, [2.0, 12.0])

    # Check that we have 2 sub-histograms.
    self.assertLen(hist._impls, 2)

  def test_per_channel_with_initial_width(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0, axis=-1
    )
    data = np.array([
        [0.5, 10.5],
        [1.6, 11.6],
    ])
    hist.add(data)
    self.assertTrue(hist.initialized)

    # Channel 0 data: [0.5, 1.5] -> counts should be [1, 1]
    # (since bin_width=1.0, lower_bound=0.5).
    # Channel 1 data: [10.5, 11.5] -> counts should be [1, 1]
    # (bin_width=1.0, lower_bound=10.5).
    np.testing.assert_array_equal(hist._impls[0].counts, [1, 1])
    np.testing.assert_array_equal(hist._impls[1].counts, [1, 1])
    self.assertEqual(hist._impls[0].lower_bound, 0.5)
    self.assertEqual(hist._impls[1].lower_bound, 10.5)

  def test_per_channel_serialization(self):
    hist = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0, axis=-1
    )
    data = np.array([
        [0.5, 10.5],
        [1.6, 11.6],
    ])
    hist.add(data)

    d = hist.to_dict()
    self.assertIn('min', d)
    self.assertIn('max', d)
    self.assertEqual(d['axis'], -1)
    self.assertLen(d['channels'], 2)

    hist2 = histogram_utils.DynamicHistogram.from_dict(d, max_tensor_bins=10)
    self.assertEqual(hist2.axis, -1)
    self.assertTrue(hist2.initialized)
    np.testing.assert_array_equal(hist2.global_min, hist.global_min)
    np.testing.assert_array_equal(hist2.global_max, hist.global_max)
    np.testing.assert_array_equal(hist2._impls[0].counts, hist._impls[0].counts)
    np.testing.assert_array_equal(hist2._impls[1].counts, hist._impls[1].counts)

  def test_per_channel_merge(self):
    hist1 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0, axis=-1
    )
    hist2 = histogram_utils.DynamicHistogram(
        max_tensor_bins=10, initial_bin_width=1.0, axis=-1
    )

    # Hist1: Channel 0: [0.5], Channel 1: [10.5].
    hist1.add(np.array([[0.5, 10.5]]))
    # Hist2: Channel 0: [1.5], Channel 1: [11.5].
    hist2.add(np.array([[1.5, 11.5]]))

    hist1.merge(hist2)

    # Channel 0 merged: [0.5, 1.5] -> counts [1, 1] (lower_bound 0.5).
    # Channel 1 merged: [10.5, 11.5] -> counts [1, 1] (lower_bound 10.5).
    np.testing.assert_array_equal(hist1._impls[0].counts, [1, 1])
    np.testing.assert_array_equal(hist1._impls[1].counts, [1, 1])
    np.testing.assert_array_equal(hist1.global_min, [0.5, 10.5])
    np.testing.assert_array_equal(hist1.global_max, [1.5, 11.5])


if __name__ == '__main__':
  absltest.main()
