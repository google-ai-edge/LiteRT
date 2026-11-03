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

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import histogram_quantize
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.utils import histogram_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../../tests/models")
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


class HistogramQuantizeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "conv_fc_mnist.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._fc_op = self._test_model.subgraphs[0].operators[3]
    self._op_info = qtyping.OpInfo(
        op=self._fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=3,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=False
            ),
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=True
            ),
            compute_precision=qtyping.ComputePrecision.INTEGER,
        ),
    )

  def test_histogram_calibrate_and_update_roundtrip(self):
    input_name = "sequential/flatten/Reshape"
    output_name = (
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    )
    rng = np.random.default_rng(42)
    batch1 = {
        input_name: rng.uniform(-5.0, 5.0, size=(4, 16)).astype(np.float32),
        output_name: rng.uniform(0.0, 10.0, size=(4, 8)).astype(np.float32),
    }
    batch2 = {
        input_name: rng.uniform(-6.0, 6.0, size=(4, 16)).astype(np.float32),
        output_name: rng.uniform(0.0, 12.0, size=(4, 8)).astype(np.float32),
    }

    qsv1 = histogram_quantize.histogram_calibrate(
        self._fc_op,
        self._graph_info,
        batch1,
        inputs_to_ignore=[1, 2],
        outputs_to_ignore=[],
    )
    qsv2 = histogram_quantize.histogram_calibrate(
        self._fc_op,
        self._graph_info,
        batch2,
        inputs_to_ignore=[1, 2],
        outputs_to_ignore=[],
    )

    self.assertIn(input_name, qsv1)
    self.assertIn("histogram", qsv1[input_name])
    self.assertNotIn("arith.constant1", qsv1)

    merged_qsv = histogram_quantize.histogram_update_qsv(
        qsv1[input_name], qsv2[input_name]
    )
    self.assertEqual(merged_qsv["num_samples"], 8)
    merged_hist = histogram_utils.DynamicHistogram.from_dict(
        merged_qsv["histogram"]
    )
    self.assertAlmostEqual(
        int(np.sum(merged_hist.counts)),
        batch1[input_name].size + batch2[input_name].size,
        delta=2,
    )

  def test_histogram_calibrate_filters_out_of_range_values(self):
    input_name = "sequential/flatten/Reshape"
    output_name = (
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    )
    bfloat16_inf = 3.39e38
    tensor_content_map = {
        input_name: np.array(
            [[-np.inf, 1.0, 2.0, 5.0, np.inf, bfloat16_inf]], dtype=np.float32
        ),
        output_name: np.array(
            [[6.0, 7.0, -bfloat16_inf, 9.0, np.nan]], dtype=np.float32
        ),
    }
    op_qsvs = histogram_quantize.histogram_calibrate(
        self._fc_op,
        self._graph_info,
        tensor_content_map,
        inputs_to_ignore=[1, 2],
        outputs_to_ignore=[],
    )
    in_hist = histogram_utils.DynamicHistogram.from_dict(
        op_qsvs[input_name]["histogram"]
    )
    self.assertEqual(np.sum(in_hist.counts), 3)
    self.assertAlmostEqual(float(in_hist.global_min[0]), 1.0)
    self.assertAlmostEqual(float(in_hist.global_max[0]), 5.0)

    out_hist = histogram_utils.DynamicHistogram.from_dict(
        op_qsvs[output_name]["histogram"]
    )
    self.assertEqual(np.sum(out_hist.counts), 3)
    self.assertAlmostEqual(float(out_hist.global_min[0]), 6.0)
    self.assertAlmostEqual(float(out_hist.global_max[0]), 9.0)

  def test_histogram_calibrate_skips_boolean_tensors(self):
    input_name = "sequential/flatten/Reshape"
    output_name = (
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    )
    tensor_content_map = {
        input_name: np.array([[True, False, True]], dtype=bool),
        output_name: np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    }
    op_qsvs = histogram_quantize.histogram_calibrate(
        self._fc_op,
        self._graph_info,
        tensor_content_map,
        inputs_to_ignore=[1, 2],
        outputs_to_ignore=[],
    )
    self.assertNotIn("histogram", op_qsvs[input_name])
    self.assertIn("histogram", op_qsvs[output_name])

  @parameterized.named_parameters(
      (
          "percentile",
          histogram_quantize.get_tensor_quant_params_percentile,
      ),
      (
          "mse",
          histogram_quantize.get_tensor_quant_params_mse,
      ),
  )
  def test_outlier_activation_gives_smaller_scale_than_min_max(
      self, get_params_fn
  ):
    rng = np.random.default_rng(99)
    inliers = rng.uniform(-5.0, 5.0, size=500_000).astype(np.float32)
    outliers = np.array([-8.0, 8.0], dtype=np.float32)

    hist = histogram_utils.DynamicHistogram(max_tensor_bins=512)
    hist.add(inliers)
    hist.add(outliers)

    qsv = {
        "min": np.array([[-8.0]], dtype=np.float32),
        "max": np.array([[8.0]], dtype=np.float32),
        "histogram": hist.to_dict(),
    }
    act_config = _TensorQuantConfig(num_bits=8, symmetric=False)

    hist_params = get_params_fn(
        self._op_info, act_config, tensor_content=None, tensor_qsv=qsv
    )
    minmax_params = naive_min_max_quantize.get_tensor_quant_params(
        self._op_info, act_config, tensor_content=None, tensor_qsv=qsv
    )
    self.assertLess(
        float(np.squeeze(hist_params.scale)),
        float(np.squeeze(minmax_params.scale)),
    )

  @parameterized.named_parameters(
      (
          "percentile",
          histogram_quantize.get_tensor_quant_params_percentile,
      ),
      (
          "mse",
          histogram_quantize.get_tensor_quant_params_mse,
      ),
  )
  def test_weight_tensors_match_naive_min_max(self, get_params_fn):
    rng = np.random.default_rng(77)
    weights = rng.normal(size=(16, 32)).astype(np.float32)
    weight_config = _TensorQuantConfig(
        num_bits=8,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
    )
    hist_params = get_params_fn(
        self._op_info, weight_config, tensor_content=weights
    )
    minmax_params = naive_min_max_quantize.get_tensor_quant_params(
        self._op_info, weight_config, tensor_content=weights
    )
    np.testing.assert_array_equal(hist_params.scale, minmax_params.scale)
    np.testing.assert_array_equal(
        hist_params.zero_point, minmax_params.zero_point
    )
    np.testing.assert_array_equal(
        hist_params.quantized_data, minmax_params.quantized_data
    )

  @parameterized.named_parameters(
      (
          "percentile",
          histogram_quantize.get_tensor_quant_params_percentile,
      ),
      (
          "mse",
          histogram_quantize.get_tensor_quant_params_mse,
      ),
  )
  def test_missing_histogram_falls_back_to_min_max(self, get_params_fn):
    qsv_no_hist = {
        "min": np.array([[-3.0]], dtype=np.float32),
        "max": np.array([[5.0]], dtype=np.float32),
    }
    act_config = _TensorQuantConfig(num_bits=8, symmetric=False)
    hist_params = get_params_fn(
        self._op_info, act_config, tensor_content=None, tensor_qsv=qsv_no_hist
    )
    minmax_params = naive_min_max_quantize.get_tensor_quant_params(
        self._op_info, act_config, tensor_content=None, tensor_qsv=qsv_no_hist
    )
    np.testing.assert_allclose(hist_params.scale, minmax_params.scale)
    np.testing.assert_array_equal(
        hist_params.zero_point, minmax_params.zero_point
    )

  @parameterized.named_parameters(
      (
          "percentile",
          histogram_quantize.get_tensor_quant_params_percentile,
      ),
      (
          "mse",
          histogram_quantize.get_tensor_quant_params_mse,
      ),
  )
  def test_fixed_output_scale_constraint_overrides_histogram(
      self, get_params_fn
  ):
    rng = np.random.default_rng(123)
    softmax_acts = rng.uniform(0.0, 0.12, size=10_000).astype(np.float32)
    hist = histogram_utils.DynamicHistogram(max_tensor_bins=512)
    hist.add(softmax_acts)

    # Simulate common_utils.materialize_op_with_output_activation_constraint
    # overwriting min/max with 0D scalars for Softmax (scale=1/256, zp=-128).
    qsv_with_fixed_constraint = {
        "min": np.float64(0.0),
        "max": np.float64(255.0 / 256.0),
        "histogram": hist.to_dict(),
    }
    act_config = _TensorQuantConfig(num_bits=8, symmetric=False)
    hist_params = get_params_fn(
        self._op_info,
        act_config,
        tensor_content=None,
        tensor_qsv=qsv_with_fixed_constraint,
    )
    minmax_params = naive_min_max_quantize.get_tensor_quant_params(
        self._op_info,
        act_config,
        tensor_content=None,
        tensor_qsv=qsv_with_fixed_constraint,
    )
    np.testing.assert_allclose(hist_params.scale, minmax_params.scale)
    np.testing.assert_array_equal(
        hist_params.zero_point, minmax_params.zero_point
    )

  def test_histogram_calibrate_float16_integer_and_all_out_of_range(self):
    input_name = "sequential/flatten/Reshape"
    output_name = (
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    )
    for content, valid_range, expected_min, expected_max in (
        (np.array([[1.0, 2.0, 4.0]], dtype=np.float16), (-10.0, 10.0), 1, 4),
        (
            np.array([[-1.0, 2.0, 8.0]], dtype=ml_dtypes.bfloat16),
            (-10.0, 10.0),
            -1,
            8,
        ),
        (np.array([[-3, 0, 7]], dtype=np.int32), (-1.0, 1.0), -3, 7),
        # All values outside valid_range: histogram uses the full content,
        # mirroring the min/max fallback in get_activation_min_max.
        (np.array([[2.0, 3.0]], dtype=np.float32), (-1.0, 1.0), 2, 3),
    ):
      with self.subTest(dtype=content.dtype.name, valid_range=valid_range):
        op_qsvs = histogram_quantize.histogram_calibrate(
            self._fc_op,
            self._graph_info,
            {input_name: content, output_name: content},
            inputs_to_ignore=[1, 2],
            outputs_to_ignore=[],
            valid_range=valid_range,
        )
        hist = histogram_utils.DynamicHistogram.from_dict(
            op_qsvs[input_name]["histogram"]
        )
        self.assertEqual(np.sum(hist.counts), content.size)
        self.assertAlmostEqual(float(hist.global_min[0]), expected_min)
        self.assertAlmostEqual(float(hist.global_max[0]), expected_max)

  @parameterized.named_parameters(
      (
          "percentile",
          histogram_quantize.get_tensor_quant_params_percentile,
      ),
      (
          "mse",
          histogram_quantize.get_tensor_quant_params_mse,
      ),
  )
  def test_bfloat16_activation_calibrate_then_clip(self, get_params_fn):
    input_name = "sequential/flatten/Reshape"
    output_name = (
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    )
    rng = np.random.default_rng(5)
    content = (
        np.concatenate([rng.uniform(-1.0, 1.0, size=200_000), [-50.0, 50.0]])
        .astype(ml_dtypes.bfloat16)
        .reshape(1, -1)
    )
    qsv = dict(
        histogram_quantize.histogram_calibrate(
            self._fc_op,
            self._graph_info,
            {input_name: content, output_name: content},
            inputs_to_ignore=[1, 2],
            outputs_to_ignore=[],
        )[input_name]
    )
    self.assertEqual(qsv["min"].dtype, ml_dtypes.bfloat16)
    act_config = _TensorQuantConfig(num_bits=8, symmetric=False)

    hist_params = get_params_fn(self._op_info, act_config, tensor_qsv=qsv)
    minmax_params = naive_min_max_quantize.get_tensor_quant_params(
        self._op_info, act_config, tensor_qsv=qsv
    )
    self.assertEqual(hist_params.scale.shape, minmax_params.scale.shape)
    self.assertLess(
        float(np.squeeze(hist_params.scale)),
        float(np.squeeze(minmax_params.scale)),
    )

  def test_histogram_calibrate_omits_histogram_for_all_nan_tensor(self):
    input_name = "sequential/flatten/Reshape"
    output_name = (
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    )
    op_qsvs = histogram_quantize.histogram_calibrate(
        self._fc_op,
        self._graph_info,
        {
            input_name: np.full((1, 4), np.nan, dtype=np.float32),
            output_name: np.array([[1.0, 2.0]], dtype=np.float32),
        },
        inputs_to_ignore=[1, 2],
        outputs_to_ignore=[],
    )
    self.assertIn("min", op_qsvs[input_name])
    self.assertNotIn("histogram", op_qsvs[input_name])

  @parameterized.named_parameters(
      (
          "percentile",
          histogram_quantize.get_tensor_quant_params_percentile,
      ),
      (
          "mse",
          histogram_quantize.get_tensor_quant_params_mse,
      ),
  )
  def test_uninitialized_histogram_falls_back_to_min_max(self, get_params_fn):
    qsv = {
        "min": np.array([[-3.0]], dtype=np.float32),
        "max": np.array([[5.0]], dtype=np.float32),
        "histogram": {"axis": None, "channels": [{}]},
    }
    act_config = _TensorQuantConfig(num_bits=8, symmetric=False)
    with self.assertLogs(level="WARNING") as logs:
      hist_params = get_params_fn(
          self._op_info, act_config, tensor_content=None, tensor_qsv=qsv
      )
    self.assertIn("Uninitialized histogram", logs.output[0])
    minmax_params = naive_min_max_quantize.get_tensor_quant_params(
        self._op_info, act_config, tensor_content=None, tensor_qsv=qsv
    )
    np.testing.assert_allclose(hist_params.scale, minmax_params.scale)
    np.testing.assert_array_equal(
        hist_params.zero_point, minmax_params.zero_point
    )

  @parameterized.named_parameters(
      ("contracted_within_qsv", (-8.0, 8.0), (-10.0, 4.0), (-8.0, 4.0)),
      ("disjoint_falls_back_to_qsv", (10.0, 20.0), (-5.0, 5.0), (10.0, 20.0)),
      ("inverted_qsv_uses_clip", (5.0, -5.0), (-1.0, 2.0), (-1.0, 2.0)),
  )
  def test_build_clipped_qsv(self, qsv_range, clip_range, expected_range):
    qsv = {
        "min": np.full((1, 1), qsv_range[0], dtype=np.float32),
        "max": np.full((1, 1), qsv_range[1], dtype=np.float32),
    }
    clipped = histogram_quantize._build_clipped_qsv(  # pylint: disable=protected-access
        qsv, *clip_range
    )
    np.testing.assert_array_equal(clipped["min"], [[expected_range[0]]])
    np.testing.assert_array_equal(clipped["max"], [[expected_range[1]]])

  def test_multi_sample_moving_average_qsv_does_not_truncate_histogram_range(
      self,
  ):
    rng = np.random.default_rng(2026)
    hist = histogram_utils.DynamicHistogram(max_tensor_bins=512)
    hist.add(rng.uniform(-1.0, 1.0, size=1_000).astype(np.float32))
    hist.add(rng.uniform(-10.0, 10.0, size=100_000).astype(np.float32))
    # Moving average of min/max across samples is [-5.5, 5.5], while the
    # true empirical distribution in hist spans [-10.0, 10.0].
    qsv = {
        "min": np.array([[-5.5]], dtype=np.float32),
        "max": np.array([[5.5]], dtype=np.float32),
        "histogram": hist.to_dict(),
    }
    act_config = _TensorQuantConfig(num_bits=16, symmetric=True)
    mse_params = histogram_quantize.get_tensor_quant_params_mse(
        self._op_info, act_config, tensor_qsv=qsv
    )
    minmax_params = naive_min_max_quantize.get_tensor_quant_params(
        self._op_info, act_config, tensor_qsv=qsv
    )
    # MSE params should reflect the ~[-10, 10] histogram distribution rather
    # than being clamped down to the [-5.5, 5.5] moving average.
    self.assertGreater(
        float(np.squeeze(mse_params.scale)),
        1.5 * float(np.squeeze(minmax_params.scale)),
    )

  def test_range_kwargs_are_forwarded(self):
    hist = histogram_utils.DynamicHistogram(max_tensor_bins=256)
    hist.add(np.linspace(0.0, 10.0, 10_001, dtype=np.float32))
    qsv = {
        "min": np.array([[0.0]], dtype=np.float32),
        "max": np.array([[10.0]], dtype=np.float32),
        "histogram": hist.to_dict(),
    }
    act_config = _TensorQuantConfig(num_bits=8, symmetric=False)
    default_params = histogram_quantize.get_tensor_quant_params_percentile(
        self._op_info, act_config, tensor_qsv=qsv
    )
    narrow_params = histogram_quantize.get_tensor_quant_params_percentile(
        self._op_info, act_config, tensor_qsv=qsv, max_percentile=50.0
    )
    self.assertLess(
        float(np.squeeze(narrow_params.scale)),
        0.6 * float(np.squeeze(default_params.scale)),
    )
    with self.assertRaisesRegex(ValueError, "Unsupported MSE search"):
      histogram_quantize.get_tensor_quant_params_mse(
          self._op_info,
          act_config,
          tensor_qsv=qsv,
          search="not_a_search",  # pyrefly: ignore[bad-argument-type]
      )


if __name__ == "__main__":
  absltest.main()

