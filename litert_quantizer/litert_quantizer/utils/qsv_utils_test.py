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

from litert_quantizer.utils import qsv_utils


class QsvUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_smoothing_factor",
          smoothing_factor=0,
          expected_vals={"min": -1000.0, "max": 800.0},
      ),
      dict(
          testcase_name="one_smoothing_factor",
          smoothing_factor=1,
          expected_vals={"min": -10.0, "max": 8.0},
      ),
      dict(
          testcase_name="normal_smoothing_factor",
          smoothing_factor=0.99,
          expected_vals={"min": -19.9, "max": 15.92},
      ),
  )
  def test_moving_average_update(self, smoothing_factor, expected_vals):
    old_qsv = {"min": -10.0, "max": 8.0}
    # Large values to mimic outlier.
    new_qsv = {"min": -1000.0, "max": 800.0}
    updated_qsv = qsv_utils.moving_average_update(
        old_qsv, new_qsv, smoothing_factor=smoothing_factor
    )
    self.assertAlmostEqual(updated_qsv["min"], expected_vals["min"])
    self.assertAlmostEqual(updated_qsv["max"], expected_vals["max"])

  def test_moving_average_update_with_array(self):
    old_qsv = {"min": np.array([-10.0, -20.0]), "max": np.array([8.0, 10.0])}
    new_qsv = {
        "min": np.array([-1000.0, -2000.0]),
        "max": np.array([800.0, 1000.0]),
    }
    updated_qsv = qsv_utils.moving_average_update(
        old_qsv, new_qsv, smoothing_factor=0.99
    )
    np.testing.assert_array_almost_equal(
        updated_qsv["min"], np.array([-19.9, -39.8])
    )
    np.testing.assert_array_almost_equal(
        updated_qsv["max"], np.array([15.92, 19.9])
    )

  def test_moving_average_update_no_old_qsv(self):
    new_qsv = {"min": -1000.0, "max": 800.0}
    updated_qsv = qsv_utils.moving_average_update(None, new_qsv)
    self.assertEqual(updated_qsv, new_qsv)

  @parameterized.named_parameters(
      dict(
          testcase_name="scalar",
          old_qsv={"min": -10.0, "max": 8.0},
          new_qsv={"min": -1000.0, "max": 1.0},
          expected_qsv={"min": -1000.0, "max": 8.0},
      ),
      dict(
          testcase_name="2darray",
          old_qsv={
              "min": np.array([[-19.0], [20.0]]),
              "max": np.array([[21.0], [250.0]]),
          },
          new_qsv={
              "min": np.array([[-1000.0], [25.0]]),
              "max": np.array([[33.0], [100.0]]),
          },
          expected_qsv={
              "min": np.array([[-1000.0], [20.0]]),
              "max": np.array([[33.0], [250.0]]),
          },
      ),
  )
  def test_min_max_update(self, old_qsv, new_qsv, expected_qsv):
    updated_qsv = qsv_utils.min_max_update(old_qsv, new_qsv)
    np.testing.assert_array_equal(updated_qsv["min"], expected_qsv["min"])
    np.testing.assert_array_equal(updated_qsv["max"], expected_qsv["max"])

  def test_min_max_update_no_old_qsv(self):
    new_qsv = {"min": -1000.0, "max": 800.0}
    updated_qsv = qsv_utils.min_max_update(None, new_qsv)
    self.assertEqual(updated_qsv, new_qsv)

  def test_gptq_and_moving_average_update_no_old_qsv(self):
    new_qsv = {"min": -1000.0, "max": 800.0}
    updated_qsv = qsv_utils.gptq_and_moving_average_update(None, new_qsv)
    self.assertEqual(updated_qsv, new_qsv)

  def test_gptq_and_moving_average_update(self):
    qsv = {
        "min": 1.0,
        "max": 10.0,
        "hessian": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "num_samples": 10,
    }
    new_qsv = {
        "min": 0.0,
        "max": 12.0,
        "hessian": np.array([[3.0, 0.0], [0.0, 3.0]]),
        "num_samples": 10,
    }
    updated_qsv = qsv_utils.gptq_and_moving_average_update(qsv, new_qsv)

    # moving_average_update check with smoothing_factor=0.95
    expected_min = 0.95 * 1.0 + 0.05 * 0.0
    expected_max = 0.95 * 10.0 + 0.05 * 12.0
    self.assertAlmostEqual(updated_qsv["min"], expected_min)
    self.assertAlmostEqual(updated_qsv["max"], expected_max)

    # gptq merge hessian check
    expected_hessian = (10 * qsv["hessian"] + 10 * new_qsv["hessian"]) / 20
    np.testing.assert_array_almost_equal(
        updated_qsv["hessian"], expected_hessian
    )
    self.assertEqual(updated_qsv["num_samples"], 20)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_samples_old",
          old_ns=0,
          new_ns=10,
          expected_h=np.array([[3.0, 0.0], [0.0, 3.0]]),
          expected_ns=10,
      ),
      dict(
          testcase_name="zero_samples_new",
          old_ns=10,
          new_ns=0,
          expected_h=np.array([[1.0, 0.0], [0.0, 1.0]]),
          expected_ns=10,
      ),
      dict(
          testcase_name="zero_samples_both",
          old_ns=0,
          new_ns=0,
          expected_h=np.array([[3.0, 0.0], [0.0, 3.0]]),
          expected_ns=0,
      ),
  )
  def test_gptq_and_moving_average_update_zero_samples(
      self, old_ns, new_ns, expected_h, expected_ns
  ):
    qsv = {
        "min": 1.0,
        "max": 10.0,
        "hessian": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "num_samples": old_ns,
    }
    new_qsv = {
        "min": 0.0,
        "max": 12.0,
        "hessian": np.array([[3.0, 0.0], [0.0, 3.0]]),
        "num_samples": new_ns,
    }
    updated_qsv = qsv_utils.gptq_and_moving_average_update(qsv, new_qsv)

    np.testing.assert_array_almost_equal(updated_qsv["hessian"], expected_h)
    self.assertEqual(updated_qsv["num_samples"], expected_ns)

  def test_oscar_and_moving_average_update(self):
    old_qsv = {
        "min": np.array([-1.0]),
        "max": np.array([2.0]),
        "mu2": np.array([2.0, 4.0]),
        "num_samples": 10,
    }
    new_qsv = {
        "min": np.array([-2.0]),
        "max": np.array([3.0]),
        "mu2": np.array([4.0, 8.0]),
        "num_samples": 10,
    }
    updated = qsv_utils.oscar_and_moving_average_update(old_qsv, new_qsv)
    np.testing.assert_array_almost_equal(updated["mu2"], np.array([3.0, 6.0]))
    self.assertEqual(updated["num_samples"], 20)

  @parameterized.named_parameters(
      dict(
          testcase_name="first_qsv_missing_mu2",
          qsv1={"min": np.array([-1.0]), "max": np.array([2.0])},
          qsv2={
              "min": np.array([-2.0]),
              "max": np.array([3.0]),
              "mu2": np.array([4.0, 8.0]),
              "num_samples": 10,
          },
          expected_mu2=np.array([4.0, 8.0]),
          expected_samples=10,
      ),
      dict(
          testcase_name="second_qsv_missing_mu2",
          qsv1={
              "min": np.array([-2.0]),
              "max": np.array([3.0]),
              "mu2": np.array([4.0, 8.0]),
              "num_samples": 10,
          },
          qsv2={"min": np.array([-1.0]), "max": np.array([2.0])},
          expected_mu2=np.array([4.0, 8.0]),
          expected_samples=10,
      ),
      dict(
          testcase_name="both_qsvs_missing_mu2",
          qsv1={"min": np.array([-1.0]), "max": np.array([2.0])},
          qsv2={"min": np.array([-2.0]), "max": np.array([3.0])},
          expected_mu2=None,
          expected_samples=0,
      ),
  )

  def test_oscar_and_moving_average_update_missing_mu2(
      self, qsv1, qsv2, expected_mu2, expected_samples
  ):
    updated = qsv_utils.oscar_and_moving_average_update(qsv1, qsv2)
    if expected_mu2 is None:
      self.assertIsNone(updated["mu2"])
    else:
      np.testing.assert_array_equal(updated["mu2"], expected_mu2)
    self.assertEqual(updated["num_samples"], expected_samples)


if __name__ == "__main__":
  absltest.main()
