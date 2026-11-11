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

import logging
import pathlib
from typing import cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import octav
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


class OctavQuantizeTest(parameterized.TestCase):
  """Tests for general functions for OCTAV."""

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(test_utils.get_path_to_datafile("../../tests/models"))
        / "conv_fc_mnist.tflite",
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = {}
    subgraph0 = self._test_model.subgraphs[0]
    self._subgraph_op_index = 3
    self._fc_op = subgraph0.operators[self._subgraph_op_index]
    self._fc_op_info = self._get_fc_op_info(None)

  def _get_fc_op_info(
      self, tensor_config: qtyping.TensorQuantizationConfig | None
  ) -> qtyping.OpInfo:
    return qtyping.OpInfo(
        op=self._fc_op,
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        subgraph_op_index=self._subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=tensor_config,
        ),
    )

  def test_get_tensor_quant_params_unsupported_symmetry(self):
    err_msg = "Unsupported symmetry"
    test_data = np.array([[-7, 7], [4, -4], [4, -4], [7, 7]])
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: err_msg in str(err)
    ):
      _ = octav.get_tensor_quant_params(
          op_info=self._fc_op_info,
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              symmetric=False,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          tensor_content=test_data,
      )

  def test_get_tensor_quant_params_success_with_qsv(self):
    # Fall back to naive_min_max_quantize.py for non-weight tensors.
    tensor_quant_params = octav.get_tensor_quant_params(
        op_info=self._fc_op_info,
        tensor_quant_config=qtyping.TensorQuantizationConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        tensor_qsv={
            "min": np.array([-1]),
            "max": np.array([1]),
        },
    )

    self.assertIsNone(tensor_quant_params.quantized_dimension)
    scale = tensor_quant_params.scale
    self.assertEqual(scale.shape, (1,))
    self.assertSequenceAlmostEqual(scale.flatten(), [1 / 127])

    # Zero point should be zero for symmetric quantization.
    zp = tensor_quant_params.zero_point
    self.assertEqual(np.sum(zp), 0)
    self.assertEqual(zp.shape, (1,))

  def test_get_tensor_quant_params_sanity_tensorwise(self):
    test_data = np.array([
        [-1e5, 25, -50, 75, -100, 125],
        [25, -30, 50, -75, 1e5, -125],
        [50, -60, 70, -80, 90, -100],
    ])
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    quant_params = octav.get_tensor_quant_params(
        op_info=self._get_fc_op_info(tensor_config),
        tensor_quant_config=tensor_config,
        tensor_content=test_data,
    )
    adjusted_test_data = quant_params.quantized_data * quant_params.scale
    real_max = np.max(np.abs(test_data))
    adjusted_max = np.max(np.abs(adjusted_test_data))

    # Check that some clipping occurred.
    with self.subTest(name="CheckClipping"):
      self.assertLess(adjusted_max, real_max)

    with self.subTest(name="CheckQuantParamsShapes"):
      self.assertEqual(quant_params.zero_point.shape, (1, 1))
      self.assertEqual(quant_params.scale.shape, (1, 1))
      self.assertIsNone(quant_params.quantized_dimension)
      self.assertIsNotNone(quant_params.quantized_data)
      self.assertTupleEqual(
          cast(np.ndarray, quant_params.quantized_data).shape, test_data.shape
      )

    with self.subTest(name="CheckQuantParamsValues"):
      self.assertTrue(np.all(quant_params.zero_point == 0))

  def test_get_tensor_quant_params_sanity_channelwise(self):
    # Test that the call generates quant params that are appropriately shaped,
    # have some clipping, and correct config values without checking the
    # actual values numerically.
    test_data = np.array([
        [-1e5, 25, -50, 75, -100, 125],
        [25, -30, 50, -75, 1e5, -125],
        [50, -60, 70, -80, 90, -100],
    ])
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
    )
    quant_params = octav.get_tensor_quant_params(
        op_info=self._get_fc_op_info(tensor_config),
        tensor_quant_config=tensor_config,
        tensor_content=test_data,
    )
    # Dequantize output to compare with the original test data.
    adjusted_test_data = quant_params.quantized_data * quant_params.scale

    for i, row in enumerate(test_data):
      real_max = np.max(np.abs(row))
      adjusted_max = np.max(np.abs(adjusted_test_data[i]))
      # Check that some clipping occurred.
      with self.subTest(name="CheckClipping"):
        self.assertLess(adjusted_max, real_max)

    with self.subTest(name="CheckQuantParamsShapes"):
      self.assertEqual(quant_params.zero_point.shape, (test_data.shape[0], 1))
      self.assertEqual(quant_params.scale.shape, (test_data.shape[0], 1))
      self.assertIsNotNone(quant_params.quantized_data)
      self.assertTupleEqual(
          cast(np.ndarray, quant_params.quantized_data).shape, test_data.shape
      )

    with self.subTest(name="CheckQuantParamsValues"):
      self.assertTrue(np.all(quant_params.zero_point == 0))
      self.assertEqual(quant_params.quantized_dimension, 0)

  def test_get_tensor_quant_params_sanity_blockwise(self):
    # Test that the call generates quant params that are appropriately shaped,
    # have some clipping, and correct config values without checking the
    # actual values numerically.
    test_data = np.random.randint(0, 1024, size=(32, 128))
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.BLOCKWISE_32,
    )
    quant_params = octav.get_tensor_quant_params(
        op_info=self._get_fc_op_info(tensor_config),
        tensor_quant_config=tensor_config,
        tensor_content=test_data,
    )

    with self.subTest(name="CheckQuantParamsShapes"):
      # Check that quant params have appropriate shapes.
      self.assertEqual(quant_params.zero_point.shape, (32, 4))
      self.assertEqual(quant_params.scale.shape, (32, 4))
      self.assertIsNotNone(quant_params.quantized_data)
      self.assertTupleEqual(
          cast(np.ndarray, quant_params.quantized_data).shape, test_data.shape
      )

    scales = np.repeat(quant_params.scale, 32, axis=1)
    adjusted_test_data = quant_params.quantized_data * scales
    for i, row in enumerate(test_data):
      real_max = np.max(np.abs(row))
      adjusted_max = np.max(np.abs(adjusted_test_data[i]))
      # Check that some clipping occurred.
      with self.subTest(name="CheckClipping"):
        self.assertLess(adjusted_max, real_max)

    with self.subTest(name="CheckQuantParamsValues"):
      self.assertTrue(np.all(quant_params.zero_point == 0))
      # See TFL_OP_TO_BLOCKWISE_WEIGHT_QUANTIZED_DIM.
      self.assertEqual(quant_params.quantized_dimension, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="channelwise",
          granularity=qtyping.QuantGranularity.CHANNELWISE,
          expected_quantized_dim=0,
          expected_scale=[[6.0 / 17.0], [12.0 / 17.0]],
          expected_q=[
              [-3, -1, 1, 3, -3, 3, -1, 1],
              [3, -3, 1, -1, 1, -1, 3, -3],
          ],
      ),
      dict(
          testcase_name="tensorwise",
          granularity=qtyping.QuantGranularity.TENSORWISE,
          expected_quantized_dim=None,
          expected_scale=[[36.0 / 55.0]],
          expected_q=[
              [-3, -1, 1, 3, -1, 1, -1, 1],
              [3, -3, 1, -1, 1, -1, 3, -3],
          ],
      ),
  )
  def test_get_tensor_quant_params_int2_in_int4(
      self,
      granularity: qtyping.QuantGranularity,
      expected_quantized_dim: int | None,
      expected_scale: list[list[float]],
      expected_q: list[list[int]],
  ):
    test_data = np.array(
        [
            [-1.5, -0.4, 0.4, 1.5, -0.9, 0.9, -0.1, 0.1],
            [3.0, -3.0, 1.1, -1.1, 0.2, -0.2, 2.5, -2.5],
        ],
        dtype=np.float32,
    )
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        symmetric=True,
        granularity=granularity,
        algorithm_params={"int2_in_int4": True},
    )
    quant_params = octav.get_tensor_quant_params(
        op_info=self._get_fc_op_info(tensor_config),
        tensor_quant_config=tensor_config,
        tensor_content=test_data,
    )
    self.assertEqual(quant_params.num_bits, 4)
    self.assertEqual(quant_params.quantized_dimension, expected_quantized_dim)
    self.assertEqual(quant_params.scale.dtype, np.float32)
    np.testing.assert_array_equal(quant_params.zero_point, 0)
    np.testing.assert_allclose(
        quant_params.scale,
        np.array(expected_scale, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    q_data = cast(np.ndarray, quant_params.quantized_data)
    np.testing.assert_array_equal(q_data, np.array(expected_q, dtype=np.int8))

  @parameterized.named_parameters(
      dict(
          testcase_name="int2_mid_tread",
          bits=2,
          mid_riser=False,
          expected_guess=0.023738872,
      ),
      dict(
          testcase_name="int4_mid_tread",
          bits=4,
          mid_riser=False,
          expected_guess=0.74102079,
      ),
      dict(
          testcase_name="int8_mid_tread",
          bits=8,
          mid_riser=False,
          expected_guess=1.98973,
      ),
      dict(
          testcase_name="int2_mid_riser",
          bits=2,
          mid_riser=True,
          expected_guess=0.091690544,
      ),
  )
  def test_guess_clipping_with_octav_step_scaling(
      self, bits: int, mid_riser: bool, expected_guess: float
  ):
    x = np.zeros(1000, dtype=np.float32)
    x[0] = 2.0
    guess = octav._guess_clipping_with_octav(  # pylint: disable=protected-access
        x,
        bits=bits,
        axis=0,
        max_iterations=1,
        exponent_divisor=3.0,
        early_stop=False,
        mid_riser=mid_riser,
    )
    np.testing.assert_allclose(guess, [expected_guess], rtol=1e-6, atol=1e-6)

  @parameterized.parameters(1, 0, -1)
  def test_get_tensor_quant_params_rejects_num_bits_less_than_2(
      self, invalid_bits: int
  ):
    test_data = np.array([[-1.0, 1.0], [0.5, -0.5]], dtype=np.float32)
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=invalid_bits,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
    )
    with self.assertRaisesRegex(ValueError, "num_bits >= 2"):
      octav.get_tensor_quant_params(
          op_info=self._get_fc_op_info(tensor_config),
          tensor_quant_config=tensor_config,
          tensor_content=test_data,
      )

  def test_guess_clipping_with_octav_rejects_invalid_bits(self):
    x = np.array([1.0, -1.0], dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "bits >= 2"):
      octav._guess_clipping_with_octav(  # pylint: disable=protected-access
          x, bits=1, axis=0, max_iterations=1, exponent_divisor=3.0
      )
    with self.assertRaisesRegex(ValueError, "bits >= 1"):
      octav._guess_clipping_with_octav(  # pylint: disable=protected-access
          x,
          bits=0,
          axis=0,
          max_iterations=1,
          exponent_divisor=3.0,
          mid_riser=True,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="int8",
          num_bits=8,
          alg_params={"int2_in_int4": True},
      ),
      dict(
          testcase_name="int2",
          num_bits=2,
          alg_params={"int2_in_int4": True},
      ),
      dict(
          testcase_name="non_bool_flag",
          num_bits=4,
          alg_params={"int2_in_int4": "false"},
      ),
  )
  def test_get_tensor_quant_params_rejects_invalid_int2_in_int4(
      self,
      num_bits: int,
      alg_params: dict[str, object],
  ):
    test_data = np.array([[-1.0, 1.0], [0.5, -0.5]], dtype=np.float32)
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=num_bits,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params=alg_params,
    )
    with self.assertRaisesRegex(ValueError, "int2_in_int4"):
      octav.get_tensor_quant_params(
          op_info=self._get_fc_op_info(tensor_config),
          tensor_quant_config=tensor_config,
          tensor_content=test_data,
      )

  def test_get_tensor_quant_params_int2_in_int4_blockwise(self):
    base_pattern = np.array(
        [-1.5, -0.4, 0.4, 1.5, -0.9, 0.9, -0.1, 0.1], dtype=np.float32
    )
    blk_pos = np.tile(base_pattern, 4)
    blk_neg = -blk_pos
    # Shape (2, 64) with BLOCKWISE_32 -> (2, 2) blocks, including a large
    # outlier block (2e5) that exceeds float16 max if scaled after rounding.
    test_data = np.array(
        [
            np.concatenate(
                [blk_pos * np.float32(1.37), blk_pos * np.float32(1.91)]
            ),
            np.concatenate(
                [blk_neg * np.float32(2.13), blk_neg * np.float32(2e5)]
            ),
        ],
        dtype=np.float32,
    )
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.BLOCKWISE_32,
        algorithm_params={"int2_in_int4": True},
    )
    quant_params = octav.get_tensor_quant_params(
        op_info=self._get_fc_op_info(tensor_config),
        tensor_quant_config=tensor_config,
        tensor_content=test_data,
    )
    self.assertEqual(quant_params.num_bits, 4)
    self.assertEqual(quant_params.block_size, 32)
    self.assertEqual(quant_params.quantized_dimension, 1)
    self.assertEqual(quant_params.scale.shape, (2, 2))
    self.assertEqual(quant_params.scale.dtype, np.float32)
    np.testing.assert_array_equal(quant_params.zero_point, 0)

    # Analytical clip_val / 4 = (6 / 17) * mult (clamped to 65280 for 2e5),
    # single-rounded to bfloat16 -> float16 -> float32.
    expected_scale = np.array(
        [[0.484375, 0.67578125], [0.75, 65280.0]], dtype=np.float32
    )
    np.testing.assert_array_equal(quant_params.scale, expected_scale)
    self.assertTrue(np.all(np.isfinite(quant_params.scale.astype(np.float16))))

    # Verify quant_params.scale is already invariant under bfloat16 -> float16
    # FlatBuffer serialization rounding.
    u32 = quant_params.scale.view(np.uint32)
    bf16_u32 = (u32 + np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))) & (
        np.uint32(0xFFFF0000)
    )
    serialized_scale = (
        bf16_u32.view(np.float32).astype(np.float16).astype(np.float32)
    )
    np.testing.assert_array_equal(quant_params.scale, serialized_scale)

    q_data = cast(np.ndarray, quant_params.quantized_data)
    expected_q_pos = np.tile(
        np.array([-3, -1, 1, 3, -3, 3, -1, 1], dtype=np.int8), 8
    )
    expected_q = np.stack([expected_q_pos, -expected_q_pos], axis=0)
    np.testing.assert_array_equal(q_data, expected_q)

  def test_get_tensor_quant_params_int2_in_int4_chunked(self):
    # (6, 16) float32 tensor (384 bytes) with _CHUNK_BYTES=64 exercises the
    # chunked quantization path at 1 row per chunk.
    two_rows = np.array(
        [
            [-1.5, -0.4, 0.4, 1.5, -0.9, 0.9, -0.1, 0.1],
            [3.0, -3.0, 1.1, -1.1, 0.2, -0.2, 2.5, -2.5],
        ],
        dtype=np.float32,
    )
    chunked_data = np.tile(two_rows, (3, 2))
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={"int2_in_int4": True},
    )
    with mock.patch.object(octav, "_CHUNK_BYTES", 64):
      chunked_params = octav.get_tensor_quant_params(
          op_info=self._get_fc_op_info(tensor_config),
          tensor_quant_config=tensor_config,
          tensor_content=chunked_data,
      )
    expected_two_scales = np.array(
        [[6.0 / 17.0], [12.0 / 17.0]], dtype=np.float32
    )
    np.testing.assert_allclose(
        chunked_params.scale,
        np.tile(expected_two_scales, (3, 1)),
        rtol=1e-6,
        atol=1e-6,
    )
    chunked_q = cast(np.ndarray, chunked_params.quantized_data)
    expected_two_q = np.array(
        [
            [-3, -1, 1, 3, -3, 3, -1, 1],
            [3, -3, 1, -1, 1, -1, 3, -3],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(chunked_q, np.tile(expected_two_q, (3, 2)))

  def test_quantize_int2_in_int4_chunked_higher_rank_matches_unchunked(self):
    rng = np.random.default_rng(123)
    for shape, q_dim, num_channels in [
        ((3, 4, 5, 8), 0, 3),
        ((2, 6, 8), 2, 8),
    ]:
      data = rng.uniform(-3.0, 3.0, size=shape).astype(np.float32)
      scales = np.linspace(0.25, 1.25, num=num_channels, dtype=np.float32)
      quant_params = qtyping.UniformQuantParams(
          scale=scales,
          zero_point=np.zeros(num_channels, dtype=np.int32),
          num_bits=4,
          symmetric=True,
          quantized_dimension=q_dim,
      )
      # pylint: disable-next=protected-access
      unchunked = octav._quantize_int2_in_int4(data, quant_params)
      with mock.patch.object(octav, "_CHUNK_BYTES", 128):
        # pylint: disable-next=protected-access
        chunked = octav._quantize_int2_in_int4(data, quant_params)
      np.testing.assert_array_equal(chunked, unchunked)

  @parameterized.product(
      distribution=["gaussian", "laplace"],
      num_bits=[2, 4],
  )
  def test_continuous_weights_mid_tread_beats_legacy_and_min_max(
      self, distribution: str, num_bits: int
  ):
    rng = np.random.default_rng(2026)
    channel_scales = rng.uniform(0.5, 1.5, size=(64, 1)).astype(np.float32)
    if distribution == "gaussian":
      base_weights = rng.normal(0.0, 0.25, size=(64, 1024)).astype(np.float32)
    else:
      base_weights = rng.laplace(0.0, 0.2, size=(64, 1024)).astype(np.float32)
    weights = base_weights * channel_scales

    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=num_bits,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={"centered": False} if num_bits == 2 else {},
    )
    fc_op_info = self._get_fc_op_info(tensor_config)

    # True mid-tread OCTAV: M = 2^(B-1) - 1 (noise weight 1/12 for INT2,
    # 1/588 for INT4).
    mid_tread_params = octav.get_tensor_quant_params(
        op_info=fc_op_info,
        tensor_quant_config=tensor_config,
        tensor_content=weights,
    )
    mid_tread_dequant = (
        cast(np.ndarray, mid_tread_params.quantized_data)
        * mid_tread_params.scale
    )

    # Legacy OCTAV step scaling on the same mid-tread integer grid:
    # M = 2^(B-1) (noise weight 1/48 for INT2, 1/768 for INT4).
    orig_guess = octav._guess_clipping_with_octav  # pylint: disable=protected-access
    with mock.patch.object(
        octav,
        "_guess_clipping_with_octav",
        side_effect=lambda *args, **kwargs: orig_guess(
            *args, **(kwargs | {"mid_riser": True})
        ),
    ):
      legacy_params = octav.get_tensor_quant_params(
          op_info=fc_op_info,
          tensor_quant_config=tensor_config,
          tensor_content=weights,
      )
    legacy_dequant = (
        cast(np.ndarray, legacy_params.quantized_data) * legacy_params.scale
    )

    # Naive min-max on the same mid-tread integer grid: scale = max|w| / M.
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    min_max_scale = np.max(np.abs(weights), axis=1, keepdims=True) / np.float32(
        qmax
    )
    min_max_dequant = (
        np.clip(np.rint(weights / min_max_scale), qmin, qmax) * min_max_scale
    )

    weights_f64 = weights.astype(np.float64)
    mid_tread_mse = float(
        np.mean((mid_tread_dequant.astype(np.float64) - weights_f64) ** 2)
    )
    legacy_mse = float(
        np.mean((legacy_dequant.astype(np.float64) - weights_f64) ** 2)
    )
    min_max_mse = float(
        np.mean((min_max_dequant.astype(np.float64) - weights_f64) ** 2)
    )

    logging.info(
        "Continuous %s INT%d MSE: mid_tread=%.6e, legacy=%.6e, min_max=%.6e",
        distribution,
        num_bits,
        mid_tread_mse,
        legacy_mse,
        min_max_mse,
    )
    self.assertLess(
        mid_tread_mse,
        legacy_mse,
        f"Expected mid-tread MSE ({mid_tread_mse:.6e}) < legacy"
        f" ({legacy_mse:.6e}) for {distribution} INT{num_bits}.",
    )
    self.assertLess(
        legacy_mse,
        min_max_mse,
        f"Expected legacy MSE ({legacy_mse:.6e}) < min-max"
        f" ({min_max_mse:.6e}) for {distribution} INT{num_bits}.",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="channelwise",
          granularity=qtyping.QuantGranularity.CHANNELWISE,
          expected_quantized_dim=0,
          expected_scale=[[12.0 / 17.0], [24.0 / 17.0]],
          expected_q=[
              [-2, -1, 0, 1, -2, 1, -1, 0],
              [1, -2, 0, -1, 0, -1, 1, -2],
          ],
      ),
      dict(
          testcase_name="tensorwise",
          granularity=qtyping.QuantGranularity.TENSORWISE,
          expected_quantized_dim=None,
          expected_scale=[[72.0 / 55.0]],
          expected_q=[
              [-2, -1, 0, 1, -1, 0, -1, 0],
              [1, -2, 0, -1, 0, -1, 1, -2],
          ],
      ),
  )
  def test_get_tensor_quant_params_centered_int2(
      self,
      granularity: qtyping.QuantGranularity,
      expected_quantized_dim: int | None,
      expected_scale: list[list[float]],
      expected_q: list[list[int]],
  ):
    test_data = np.array(
        [
            [-1.5, -0.4, 0.4, 1.5, -0.9, 0.9, -0.1, 0.1],
            [3.0, -3.0, 1.1, -1.1, 0.2, -0.2, 2.5, -2.5],
        ],
        dtype=np.float32,
    )
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=2,
        granularity=granularity,
    )
    quant_params = octav.get_tensor_quant_params(
        op_info=self._get_fc_op_info(tensor_config),
        tensor_quant_config=tensor_config,
        tensor_content=test_data,
    )
    self.assertEqual(quant_params.num_bits, 2)
    self.assertEqual(quant_params.quantized_dimension, expected_quantized_dim)
    self.assertEqual(quant_params.scale.dtype, np.float32)
    self.assertEqual(quant_params.zero_point.dtype, np.float32)
    np.testing.assert_array_equal(quant_params.zero_point, [[-0.5]])
    np.testing.assert_allclose(
        quant_params.scale,
        np.array(expected_scale, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    q_data = cast(np.ndarray, quant_params.quantized_data)
    np.testing.assert_array_equal(q_data, np.array(expected_q, dtype=np.int8))

  def test_centered_int2_matches_int2_in_int4_and_beats_mid_tread(self):
    rng = np.random.default_rng(2028)
    weights = rng.laplace(0.0, 0.2, size=(16, 256)).astype(np.float32)
    centered_cfg = qtyping.TensorQuantizationConfig(
        num_bits=2,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
    )
    mid_tread_cfg = qtyping.TensorQuantizationConfig(
        num_bits=2,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={"centered": False},
    )
    emulated_cfg = qtyping.TensorQuantizationConfig(
        num_bits=4,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={"int2_in_int4": True},
    )

    centered_qp = octav.get_tensor_quant_params(
        self._get_fc_op_info(centered_cfg), centered_cfg, weights
    )
    mid_tread_qp = octav.get_tensor_quant_params(
        self._get_fc_op_info(mid_tread_cfg), mid_tread_cfg, weights
    )
    emulated_qp = octav.get_tensor_quant_params(
        self._get_fc_op_info(emulated_cfg), emulated_cfg, weights
    )

    centered_dequant = (
        cast(np.ndarray, centered_qp.quantized_data).astype(np.float32) + 0.5
    ) * centered_qp.scale
    emulated_dequant = (
        cast(np.ndarray, emulated_qp.quantized_data).astype(np.float32)
        * emulated_qp.scale
    )
    mid_tread_dequant = (
        cast(np.ndarray, mid_tread_qp.quantized_data).astype(np.float32)
        * mid_tread_qp.scale
    )
    np.testing.assert_allclose(
        centered_dequant, emulated_dequant, rtol=1e-6, atol=1e-6
    )
    centered_mse = float(np.mean((centered_dequant - weights) ** 2))
    mid_tread_mse = float(np.mean((mid_tread_dequant - weights) ** 2))
    self.assertLess(centered_mse, mid_tread_mse)

  @parameterized.named_parameters(
      dict(
          testcase_name="non_bool_centered",
          num_bits=2,
          alg_params={"centered": "false"},
          expected_error="centered must be a boolean",
      ),
      dict(
          testcase_name="int4_centered",
          num_bits=4,
          alg_params={"centered": True},
          expected_error="centered=True requires num_bits=2",
      ),
  )
  def test_get_tensor_quant_params_rejects_invalid_centered(
      self, num_bits: int, alg_params: dict[str, object], expected_error: str
  ):
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=num_bits,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params=alg_params,
    )
    with self.assertRaisesRegex(ValueError, expected_error):
      octav.get_tensor_quant_params(
          op_info=self._get_fc_op_info(tensor_config),
          tensor_quant_config=tensor_config,
          tensor_content=np.ones((2, 4), dtype=np.float32),
      )


if __name__ == "__main__":
  absltest.main()
