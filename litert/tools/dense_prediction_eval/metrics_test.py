# Copyright 2026 Google LLC.
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
"""Small hand-computed dense prediction metric tests."""
import unittest

import numpy as np

from litert.tools.dense_prediction_eval import metrics


class MetricsTest(unittest.TestCase):

  def test_global_confusion_and_ignore(self):
    metric = metrics.SegmentationMetrics(3)
    metric.update(np.array([[0, 1, 1, 0]]), np.array([[0, 0, 1, 255]]))
    metric.update(np.array([[1]]), np.array([[1]]))
    self.assertAlmostEqual(metric.result()['miou'], (1 / 2 + 2 / 3) / 2)
    self.assertEqual(metric.result()['valid_pixels'], 4)

  def test_invalid_labels_rejected(self):
    with self.assertRaises(ValueError):
      metrics.SegmentationMetrics(2).update(np.array([[0]]), np.array([[2]]))

  def test_uint8_labels_do_not_overflow_confusion_indices(self):
    m = metrics.SegmentationMetrics(150)
    labels = np.array([[0, 3, 89, 149]], dtype=np.uint8)
    m.update(labels, labels)
    self.assertEqual(m.result()['miou'], 1.)
    self.assertEqual(m.confusion[149, 149], 1)

  def test_logit_ties_choose_lowest_class(self):
    m = metrics.SegmentationMetrics(3)
    m.update(np.ones((3, 1, 1)), np.zeros((2, 2), dtype=np.int64))
    self.assertEqual(m.result()['miou'], 1.)

  def test_half_pixel_resize(self):
    np.testing.assert_allclose(
        metrics.resize_bilinear(np.array([[0., 4.]]), (1, 4)),
        [[0., 1., 3., 4.]])

  def test_logits_resize_before_argmax(self):
    m = metrics.SegmentationMetrics(2)
    m.update(np.array([[[10., 0.]], [[0., 1.]]]), np.array([[0, 0, 0, 1]]))
    self.assertEqual(m.result()['miou'], 1.)

  def test_depth_values_and_empty_mask(self):
    m = metrics.DepthMetrics()
    m.update(np.array([[2., 4., 8.]]), np.array([[1., 4., 0.]]))
    self.assertAlmostEqual(m.result()['abs_rel'], .5)
    self.assertAlmostEqual(m.result()['rmse'], np.sqrt(.5))
    self.assertEqual(m.result()['delta1'], .5)
    m.update(np.ones((1, 1)), np.zeros((1, 1)))
    self.assertEqual(m.result()['empty_images'], 1)

  def test_normal_orientation_zero_and_invalid_target(self):
    m = metrics.NormalMetrics()
    truth = np.array([[[1., 1., 1., 0.]], [[0., 0., 0., 0.]], [[0., 0., 0.,
                                                                0.]]])
    pred = np.array([[[1., -1., 0., 0.]], [[0., 0., 0., 0.]], [[0., 0., 0.,
                                                                0.]]])
    m.update(pred, truth)
    self.assertEqual(m.result()['valid_pixels'], 3)
    self.assertEqual(m.result()['mean_degrees'], 90.)
    self.assertEqual(m.result()['median_degrees'], 90.)
    self.assertAlmostEqual(m.result()['within_30_degrees'], 1 / 3)

  def test_nonfinite_predictions_rejected(self):
    with self.assertRaises(ValueError):
      metrics.DepthMetrics().update(np.array([[np.nan]]), np.array([[1.]]))

  def test_empty_results_are_explicit(self):
    self.assertIsNone(metrics.DepthMetrics().result()['rmse'])
    self.assertIsNone(metrics.NormalMetrics().result()['mean_degrees'])
    self.assertIsNone(metrics.SegmentationMetrics().result()['miou'])


if __name__ == '__main__':
  unittest.main()
