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

"""Test Hadamard rotation materialization."""

import dataclasses
import pathlib
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import hadamard_rotation
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../../tests/models")
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


class HadamardRotationFullyConnectedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(888)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "conv_fc_mnist.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = None
    self._subgraph = self._test_model.subgraphs[0]
    self._fc_subgraph_op_index = 3
    self._fc_op = self._subgraph.operators[self._fc_subgraph_op_index]
    self._fc_buffer_id = self._subgraph.tensors[self._fc_op.inputs[1]].buffer
    self._op_info = qtyping.OpInfo(
        op=self._fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=self._fc_subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
        ),
    )

  def test_materialize_fully_connected_basic(self):
    params = hadamard_rotation.materialize_fully_connected_custom_op(
        self._op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv=self._tensor_name_to_qsv,
    )
    fc_input = params[0]
    weight = params[1]
    bias = params[2]
    output = params[3]

    self.assertLen(params, 4)
    self.assertIsNone(fc_input.producer)
    self.assertIsNotNone(fc_input.consumers)
    self.assertIsNone(weight.producer)
    self.assertIsNotNone(weight.consumers)
    self.assertIsNone(bias.producer)
    self.assertIsNotNone(bias.consumers)
    self.assertIsNotNone(output.producer)
    self.assertIsNone(output.consumers)
    self.assertEqual(
        fc_input.consumers[0].transformations,
        [qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION],
    )
    self.assertEqual(
        weight.consumers[0].transformations,
        [qtyping.QuantTransformation.QUANTIZE_TENSOR],
    )
    self.assertEqual(
        bias.consumers[0].transformations,
        [qtyping.QuantTransformation.NO_QUANTIZE],
    )
    if output.producer is not None:
      self.assertEqual(
          output.producer.transformations,
          [qtyping.QuantTransformation.NO_QUANTIZE],
      )

  def test_fully_connected_tensorwise_supported(self):
    self._op_info = qtyping.OpInfo(
        op=self._fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=self._fc_subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.TENSORWISE,
            ),
        ),
    )
    params = hadamard_rotation.materialize_fully_connected_custom_op(
        self._op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv=self._tensor_name_to_qsv,
    )
    self.assertLen(params, 4)
    fc_input = params[0]
    self.assertIsNotNone(fc_input)
    self.assertIsNotNone(fc_input.consumers)
    self.assertIsNotNone(fc_input.consumers[0].parameters)
    self.assertIsInstance(
        fc_input.consumers[0].parameters, qtyping.UniformQuantParams
    )
    if isinstance(
        fc_input.consumers[0].parameters, qtyping.UniformQuantParams
    ):
      self.assertIsNone(fc_input.consumers[0].parameters.quantized_dimension)
    weight = params[1]
    self.assertIsNotNone(weight)
    self.assertIsNotNone(weight.consumers)
    self.assertIsNotNone(weight.consumers[0].parameters)
    self.assertIsInstance(
        weight.consumers[0].parameters, qtyping.UniformQuantParams
    )
    if isinstance(
        weight.consumers[0].parameters, qtyping.UniformQuantParams
    ):
      self.assertIsNone(weight.consumers[0].parameters.quantized_dimension)

  def test_fully_connected_blockwise_supported(self):
    self._op_info = qtyping.OpInfo(
        op=self._fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=self._fc_subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.BLOCKWISE_32,
            ),
        ),
    )
    params = hadamard_rotation.materialize_fully_connected_custom_op(
        self._op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv=self._tensor_name_to_qsv,
    )
    self.assertLen(params, 4)
    fc_input = params[0]
    self.assertIsNotNone(fc_input)
    self.assertIsNotNone(fc_input.consumers)
    self.assertIsNotNone(fc_input.consumers[0].parameters)
    self.assertIsInstance(
        fc_input.consumers[0].parameters, qtyping.UniformQuantParams
    )
    if isinstance(
        fc_input.consumers[0].parameters, qtyping.UniformQuantParams
    ):
      self.assertEqual(fc_input.consumers[0].parameters.quantized_dimension, 1)
    weight = params[1]
    self.assertIsNotNone(weight)
    self.assertIsNotNone(weight.consumers)
    self.assertIsNotNone(weight.consumers[0].parameters)
    self.assertIsInstance(
        weight.consumers[0].parameters, qtyping.UniformQuantParams
    )
    if isinstance(
        weight.consumers[0].parameters, qtyping.UniformQuantParams
    ):
      self.assertEqual(weight.consumers[0].parameters.quantized_dimension, 1)

  def test_materialize_fully_connected_decomposed(self):
    params = hadamard_rotation.materialize_fully_connected_decomposed(
        self._op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv=self._tensor_name_to_qsv,
    )
    fc_input = params[0]
    weight = params[1]
    bias = params[2]
    output = params[3]

    self.assertLen(params, 4)
    self.assertEqual(
        fc_input.consumers[0].transformations,  # pyrefly: ignore[unsupported-operation]
        [qtyping.QuantTransformation.INSERT_DECOMPOSED_HADAMARD_ROTATION],
    )
    self.assertEqual(
        weight.consumers[0].transformations,  # pyrefly: ignore[unsupported-operation]
        [qtyping.QuantTransformation.QUANTIZE_TENSOR],
    )
    self.assertEqual(
        bias.consumers[0].transformations,  # pyrefly: ignore[unsupported-operation]
        [qtyping.QuantTransformation.NO_QUANTIZE],
    )
    if output.producer is not None:
      self.assertEqual(
          output.producer.transformations,
          [qtyping.QuantTransformation.NO_QUANTIZE],
      )

  def test_get_tensor_quant_params_basic(self):
    input_tensor = self._subgraph.tensors[self._fc_op.inputs[1]]
    buffer = self._graph_info.buffers[self._fc_buffer_id]
    np_buffer = np.frombuffer(buffer.data, dtype=np.float32).reshape(
        input_tensor.shape
    )
    qparams = hadamard_rotation.get_tensor_quant_params(
        self._op_info,
        self._op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
        np_buffer,
        self._tensor_name_to_qsv,
    )
    self.assertEqual(qparams.num_bits, 8)
    self.assertEqual(qparams.zero_point.all(), 0)
    self.assertEqual(qparams.symmetric, True)
    self.assertIsNotNone(qparams.quantized_data)
    self.assertEqual(qparams.block_size, 0)
    self.assertIsNotNone(qparams.hadamard)
    if qparams.hadamard is not None:
      self.assertEqual(qparams.hadamard.hadamard_size, 32)

  def test_get_tensor_quant_params_max_size(self):
    input_tensor = self._subgraph.tensors[self._fc_op.inputs[1]]
    buffer = self._graph_info.buffers[self._fc_buffer_id]
    np_buffer = np.frombuffer(buffer.data, dtype=np.float32).reshape(
        input_tensor.shape
    )
    # The original dimension is 32. The largest power of 2 factor is 32.
    # If we set algorithm_params to {'max_hadamard_size': 16}, then it should be
    # 16.
    new_op_quant_config = dataclasses.replace(
        self._op_info.op_quant_config,
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=8,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
            algorithm_params={"max_hadamard_size": 16},
        ),
    )
    self._op_info = dataclasses.replace(
        self._op_info, op_quant_config=new_op_quant_config
    )
    qparams = hadamard_rotation.get_tensor_quant_params(
        self._op_info,
        self._op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
        np_buffer,
        self._tensor_name_to_qsv,
    )
    self.assertIsNotNone(qparams.hadamard)
    self.assertEqual(qparams.hadamard.hadamard_size, 16)

  def test_get_tensor_quant_params_golden_1(self):
    test_data = np.ones((6, 6))
    # expected:
    #   [[127   0 127   0 127   0]
    #    [127   0 127   0 127   0]
    #    [127   0 127   0 127   0]
    #    [127   0 127   0 127   0]
    #    [127   0 127   0 127   0]
    #    [127   0 127   0 127   0]]
    expected = np.tile([127, 0], [6, 3])
    qparams = hadamard_rotation.get_tensor_quant_params(
        self._op_info,
        self._op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
        test_data,
        self._tensor_name_to_qsv,
    )
    self.assertIsNotNone(qparams.quantized_data)
    np.testing.assert_array_equal(
        np.array(qparams.quantized_data), expected
    )

  def test_get_tensor_quant_params_golden_2(self):
    # test_data:
    #   [[1 2 1 2 1 2]
    #    [3 4 3 4 3 4]
    #    [1 2 1 2 1 2]
    #    [3 4 3 4 3 4]
    #    [1 2 1 2 1 2]
    #    [3 4 3 4 3 4]]
    test_data = np.tile([[1, 2], [3, 4]], [3, 3])
    # expected:
    #   [[127 -42 127 -42 127 -42]
    #    [127 -18 127 -18 127 -18]
    #    [127 -42 127 -42 127 -42]
    #    [127 -18 127 -18 127 -18]
    #    [127 -42 127 -42 127 -42]
    #    [127 -18 127 -18 127 -18]]
    expected = np.tile([[127, -42], [127, -18]], [3, 3])
    qparams = hadamard_rotation.get_tensor_quant_params(
        self._op_info,
        self._op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
        test_data,
        self._tensor_name_to_qsv,
    )
    self.assertIsNotNone(qparams.quantized_data)
    np.testing.assert_array_equal(
        np.array(qparams.quantized_data), expected
    )

  def test_get_tensor_quant_params_golden_3(self):
    # test_data:
    #   [[[1 2 1 2 1 2]
    #     [3 4 3 4 3 4]
    #     [1 2 1 2 1 2]]
    #    [[3 4 3 4 3 4]
    #     [1 2 1 2 1 2]
    #     [3 4 3 4 3 4]]]
    test_data = np.tile([[1, 2], [3, 4]], [3, 3])
    test_data = np.reshape(test_data, (2, 3, 6))
    # expected:
    #   [[[ 54 -18  54 -18  54 -18]
    #     [127 -18 127 -18 127 -18]
    #     [ 54 -18  54 -18  54 -18]]
    #    [[127 -18 127 -18 127 -18]
    #     [ 54 -18  54 -18  54 -18]
    #     [127 -18 127 -18 127 -18]]]
    expected = np.tile([[54, -18], [127, -18]], [3, 3])
    expected = np.reshape(expected, (2, 3, 6))
    qparams = hadamard_rotation.get_tensor_quant_params(
        self._op_info,
        self._op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
        test_data,
        self._tensor_name_to_qsv,
    )
    self.assertIsNotNone(qparams.quantized_data)
    np.testing.assert_array_equal(
        np.array(qparams.quantized_data), expected
    )

  def test_raise_missing_tensor_content(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: "weight tensor" in str(err)
    ):
      hadamard_rotation.get_tensor_quant_params(
          self._op_info,
          self._op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
          None,
          self._tensor_name_to_qsv,
      )

  def test_raise_qsv_set(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: "static quantization" in str(err)
    ):
      hadamard_rotation.get_tensor_quant_params(
          self._op_info,
          self._op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
          self._graph_info.buffers[self._fc_buffer_id],
          self._graph_info.buffers[self._fc_buffer_id],
      )

  def test_raise_1d_constant(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: "rank >= 2" in str(err)
    ):
      hadamard_rotation.get_tensor_quant_params(
          self._op_info,
          self._op_info.op_quant_config.weight_tensor_config,  # pyrefly: ignore[bad-argument-type]
          np.array([1.0, 2.0, 3.0]),
          self._tensor_name_to_qsv,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="block_hadamard_odd_dim",
          weight_shape=(4, 7),
          algorithm_params={"q_init_mode": "none"},
          expected_error="dim=7, max_hadamard_size=None",
      ),
      dict(
          testcase_name="kronecker_odd_dim",
          weight_shape=(4, 7),
          algorithm_params={"q_init_mode": "auto"},
          expected_error="dim=7, max_hadamard_size=None",
      ),
      dict(
          testcase_name="kronecker_max_hadamard_size_one",
          weight_shape=(4, 8),
          algorithm_params={"q_init_mode": "auto", "max_hadamard_size": 1},
          expected_error="dim=8, max_hadamard_size=1",
      ),
  )
  def test_raise_no_hadamard_size_of_at_least_two(
      self, weight_shape, algorithm_params, expected_error
  ):
    qconfig = _TensorQuantConfig(
        num_bits=8,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params=algorithm_params,
    )
    with self.assertRaisesRegex(ValueError, expected_error):
      hadamard_rotation.get_tensor_quant_params(
          self._op_info, qconfig, np.ones(weight_shape, dtype=np.float32), None
      )

  def test_make_hadamard_matrix_raises_for_size_one(self):
    with self.assertRaisesRegex(ValueError, "at least 2, got 1"):
      hadamard_rotation._make_hadamard_matrix(1)

  @parameterized.parameters(
      (32, 8, 4),
      (24, 4, 6),
      (98, 2, 49),
  )
  def test_kronecker_auto_initializer_orthogonal_and_weight_rotation(
      self, dim, expected_d, expected_m
  ):
    rng = np.random.default_rng(123)
    weight = rng.standard_normal((8, dim)).astype(np.float32)
    qconfig = _TensorQuantConfig(
        num_bits=8,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={"q_init_mode": "auto", "q_seed": 7},
    )
    qparams = hadamard_rotation.get_tensor_quant_params(
        self._op_info, qconfig, weight, None
    )
    self.assertIsNotNone(qparams.hadamard)
    self.assertEqual(qparams.hadamard.hadamard_size, expected_d)
    q1 = qparams.hadamard.cross_block_matrix
    self.assertIsNotNone(q1)
    self.assertEqual(q1.shape, (expected_m, expected_m))
    np.testing.assert_allclose(
        q1.T @ q1, np.eye(expected_m, dtype=np.float32), atol=1e-5
    )

    # Verify exact Kronecker weight rotation `W @ (Q_1 (x) H_d)^T`.
    w_rot, d, rand_vec, q1_rot = (
        hadamard_rotation._rotate_with_diagonal_hadamard(
            weight, axis=1, q_init_mode="auto", q_seed=7, randomize=True
        )
    )
    self.assertTrue(np.all(np.isin(rand_vec, [-1, 1])))
    self.assertFalse(np.all(rand_vec == 1))
    h_d = rand_vec.astype(np.float32)[
        :, np.newaxis
    ] * hadamard_rotation._make_hadamard_matrix(d)
    expected_rot = weight @ np.kron(q1_rot, h_d.T).T
    np.testing.assert_allclose(w_rot, expected_rot, atol=1e-5)

  def test_auto_selects_hadamard_rademacher_or_polar_svd_by_m(self):
    # Power-of-2 m=8 resolves to exact Sylvester Hadamard H_8.
    q_pow2 = hadamard_rotation.make_cross_block_matrix(8, "auto", seed=11)
    np.testing.assert_array_equal(
        q_pow2, hadamard_rotation._make_hadamard_matrix(8).astype(np.float32)
    )

    # Non-power-of-2 m=48 (`<= 48`) resolves to Rademacher Cayley.
    q_small = hadamard_rotation.make_cross_block_matrix(48, "auto", seed=11)
    q_rad = hadamard_rotation._make_rademacher_cayley_matrix(
        48, np.random.default_rng(11)
    )
    np.testing.assert_array_equal(q_small, q_rad)

    # Non-power-of-2 m=50 (`> 48`) resolves to Polar SVD.
    q_large = hadamard_rotation.make_cross_block_matrix(50, "auto", seed=11)
    q_svd = hadamard_rotation._make_polar_svd_matrix(
        50, np.random.default_rng(11)
    )
    np.testing.assert_array_equal(q_large, q_svd)

  def test_invalid_q_init_mode_raises(self):
    with self.assertRaisesRegex(ValueError, "is not a valid QInitMode"):
      hadamard_rotation.make_cross_block_matrix(8, "invalid_mode")

  def test_kronecker_max_power_of_two_sizing_mode(self):
    rng = np.random.default_rng(42)
    weight = rng.standard_normal((4, 1280)).astype(np.float32)
    qconfig = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={
            "q_init_mode": "auto",
            "max_hadamard_size": 256,
            "sizing_mode": "max_power_of_two",
            "q_seed": 0,
        },
    )
    qparams = hadamard_rotation.get_tensor_quant_params(
        self._op_info, qconfig, weight, None
    )
    self.assertIsNotNone(qparams.hadamard)
    self.assertEqual(qparams.hadamard.hadamard_size, 256)
    q1 = qparams.hadamard.cross_block_matrix
    self.assertIsNotNone(q1)
    self.assertEqual(q1.shape, (5, 5))
    np.testing.assert_allclose(
        q1.T @ q1, np.eye(5, dtype=np.float32), atol=1e-5
    )

  def test_invalid_sizing_mode_raises(self):
    with self.assertRaisesRegex(ValueError, "is not a valid SizingMode"):
      hadamard_rotation._select_kronecker_hadamard_size(
          1280, max_size=256, sizing_mode="invalid_sizing"
      )

  def test_scale_refinement_reduces_or_matches_weight_mse(self):
    rng = np.random.default_rng(2026)
    weight = rng.standard_normal((16, 64)).astype(np.float32)
    base_cfg = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={"max_hadamard_size": 64, "randomize": False},
    )
    ref_cfg = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={
            "max_hadamard_size": 64,
            "randomize": False,
            "scale_refinement": "mse_grid",
        },
    )
    qp_base = hadamard_rotation.get_tensor_quant_params(
        self._op_info, base_cfg, weight, None
    )
    qp_ref = hadamard_rotation.get_tensor_quant_params(
        self._op_info, ref_cfg, weight, None
    )
    w_rot, _, _, _ = hadamard_rotation._rotate_with_diagonal_hadamard(
        weight, axis=1, max_size=64, randomize=False
    )
    assert qp_base.quantized_data is not None
    assert qp_ref.quantized_data is not None
    deq_base = qp_base.quantized_data.astype(
        np.float32
    ) * qp_base.scale.reshape(-1, 1)
    deq_ref = qp_ref.quantized_data.astype(np.float32) * qp_ref.scale.reshape(
        -1, 1
    )
    mse_base = float(np.mean((w_rot - deq_base) ** 2))
    mse_ref = float(np.mean((w_rot - deq_ref) ** 2))
    self.assertLessEqual(mse_ref, mse_base)

  def test_invalid_scale_refinement_raises(self):
    weight = np.ones((4, 8), dtype=np.float32)
    qconfig = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={"scale_refinement": "invalid_refinement"},
    )
    with self.assertRaisesRegex(ValueError, "Unsupported scale_refinement"):
      hadamard_rotation.get_tensor_quant_params(
          self._op_info, qconfig, weight, None
      )

  def test_mse_grid_int2_in_int4_keeps_codebook(self):
    rng = np.random.default_rng(2027)
    w_rot = rng.standard_normal((16, 64)).astype(np.float32)
    init_scale = (np.max(np.abs(w_rot), axis=1) / 4.5).astype(np.float32)
    s_2bit = 2.0 * init_scale.reshape(-1, 1)
    init_q = (
        2.0 * np.clip(np.rint(w_rot / s_2bit - 0.5), -2.0, 1.0) + 1.0
    ).astype(np.int8)
    init_qparams = qtyping.UniformQuantParams(
        num_bits=4,
        quantized_dimension=0,
        scale=init_scale,
        zero_point=np.zeros(16, dtype=np.int64),
        symmetric=True,
        quantized_data=init_q,
        block_size=0,
    )
    refined = hadamard_rotation._refine_quant_params_mse(
        w_rot, init_qparams, int2_in_int4=True
    )
    assert refined.quantized_data is not None
    self.assertEqual(
        set(np.unique(refined.quantized_data).tolist()), {-3, -1, 1, 3}
    )
    mse_init = float(
        np.mean((w_rot - init_q.astype(np.float32) * init_scale[:, None]) ** 2)
    )
    mse_refined = float(
        np.mean(
            (
                w_rot
                - refined.quantized_data.astype(np.float32)
                * refined.scale.reshape(-1, 1)
            )
            ** 2
        )
    )
    self.assertLess(mse_refined, mse_init)

    # Verify end-to-end parameter plumbing via get_tensor_quant_params for both
    # scale_refinement="none" and scale_refinement="mse_grid".
    weight = rng.standard_normal((16, 64)).astype(np.float32)
    qconfig_no_refine = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={
            "max_hadamard_size": 64,
            "randomize": False,
            "int2_in_int4": True,
            "scale_refinement": "none",
        },
    )
    qp_no_refine = hadamard_rotation.get_tensor_quant_params(
        self._op_info, qconfig_no_refine, weight, None
    )
    assert qp_no_refine.quantized_data is not None
    self.assertEqual(
        set(np.unique(qp_no_refine.quantized_data).tolist()), {-3, -1, 1, 3}
    )

    qconfig = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={
            "max_hadamard_size": 64,
            "randomize": False,
            "int2_in_int4": True,
            "scale_refinement": "mse_grid",
        },
    )
    qp_e2e = hadamard_rotation.get_tensor_quant_params(
        self._op_info, qconfig, weight, None
    )
    assert qp_e2e.quantized_data is not None
    self.assertEqual(
        set(np.unique(qp_e2e.quantized_data).tolist()), {-3, -1, 1, 3}
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="non_bool_value",
          num_bits=4,
          int2_in_int4=1,
          expected_error="int2_in_int4 must be a boolean",
      ),
      dict(
          testcase_name="non_4bit",
          num_bits=8,
          int2_in_int4=True,
          expected_error="int2_in_int4 requires num_bits=4",
      ),
      dict(
          testcase_name="asymmetric_4bit",
          num_bits=4,
          symmetric=False,
          int2_in_int4=True,
          expected_error="Unsupported symmetry",
      ),
  )
  def test_invalid_int2_in_int4_raises(
      self, num_bits, int2_in_int4, expected_error, symmetric=True
  ):
    weight = np.ones((4, 8), dtype=np.float32)
    qconfig = _TensorQuantConfig(
        num_bits=num_bits,
        symmetric=symmetric,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={
            "int2_in_int4": int2_in_int4,
            "scale_refinement": "mse_grid",
        },
    )
    with self.assertRaisesRegex(ValueError, expected_error):
      hadamard_rotation.get_tensor_quant_params(
          self._op_info, qconfig, weight, None
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="none_uncapped",
          dim=2560,
          max_hadamard_size=None,
          q_init_mode=hadamard_rotation.QInitMode.NONE,
          sizing_mode=hadamard_rotation.SizingMode.MIN_COMPUTE,
          expected_block_size=512,
      ),
      dict(
          testcase_name="none_zero_cap_treated_as_uncapped",
          dim=2560,
          max_hadamard_size=0,
          q_init_mode="none",
          sizing_mode="min_compute",
          expected_block_size=512,
      ),
      dict(
          testcase_name="none_capped_ignores_sizing_mode",
          dim=2560,
          max_hadamard_size=256,
          q_init_mode="none",
          sizing_mode="min_compute",
          expected_block_size=256,
      ),
      dict(
          testcase_name="auto_min_compute",
          dim=2560,
          max_hadamard_size=256,
          q_init_mode=hadamard_rotation.QInitMode.AUTO,
          sizing_mode=hadamard_rotation.SizingMode.MIN_COMPUTE,
          expected_block_size=64,
      ),
      dict(
          testcase_name="auto_max_power_of_two",
          dim=2560,
          max_hadamard_size=256,
          q_init_mode=hadamard_rotation.QInitMode.AUTO,
          sizing_mode=hadamard_rotation.SizingMode.MAX_POWER_OF_TWO,
          expected_block_size=256,
      ),
      dict(
          testcase_name="numpy_int64_scalars",
          dim=np.int64(384),
          max_hadamard_size=np.int64(256),
          q_init_mode="auto",
          sizing_mode="min_compute",
          expected_block_size=16,
      ),
  )
  def test_get_hadamard_block_size_modes(
      self,
      dim,
      max_hadamard_size,
      q_init_mode,
      sizing_mode,
      expected_block_size,
  ):
    self.assertEqual(
        hadamard_rotation.get_hadamard_block_size(
            dim,
            max_hadamard_size=max_hadamard_size,
            q_init_mode=q_init_mode,
            sizing_mode=sizing_mode,
        ),
        expected_block_size,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="odd_dim",
          dim=7,
          max_hadamard_size=None,
          q_init_mode="none",
          sizing_mode="min_compute",
          expected_error="dim=7, max_hadamard_size=None",
      ),
      dict(
          testcase_name="zero_dim",
          dim=0,
          max_hadamard_size=None,
          q_init_mode="auto",
          sizing_mode="min_compute",
          expected_error="dim=0, max_hadamard_size=None",
      ),
      dict(
          testcase_name="negative_dim",
          dim=-8,
          max_hadamard_size=None,
          q_init_mode="auto",
          sizing_mode="min_compute",
          expected_error="dim=-8, max_hadamard_size=None",
      ),
      dict(
          testcase_name="max_hadamard_size_one",
          dim=8,
          max_hadamard_size=1,
          q_init_mode="auto",
          sizing_mode="min_compute",
          expected_error="dim=8, max_hadamard_size=1",
      ),
      dict(
          testcase_name="negative_max_hadamard_size",
          dim=8,
          max_hadamard_size=-4,
          q_init_mode="auto",
          sizing_mode="min_compute",
          expected_error="dim=8, max_hadamard_size=-4",
      ),
      dict(
          testcase_name="bool_dim",
          dim=True,
          max_hadamard_size=None,
          q_init_mode="auto",
          sizing_mode="min_compute",
          expected_error="must be a non-boolean integer",
      ),
      dict(
          testcase_name="bool_max_hadamard_size",
          dim=8,
          max_hadamard_size=True,
          q_init_mode="auto",
          sizing_mode="min_compute",
          expected_error="must be a non-boolean integer",
      ),
      dict(
          testcase_name="bool_false_max_hadamard_size",
          dim=8,
          max_hadamard_size=False,
          q_init_mode="auto",
          sizing_mode="min_compute",
          expected_error="must be a non-boolean integer",
      ),
      dict(
          testcase_name="invalid_q_init_mode",
          dim=8,
          max_hadamard_size=None,
          q_init_mode="invalid",
          sizing_mode="min_compute",
          expected_error="is not a valid QInitMode",
      ),
      dict(
          testcase_name="invalid_sizing_mode",
          dim=8,
          max_hadamard_size=None,
          q_init_mode="none",
          sizing_mode="invalid",
          expected_error="is not a valid SizingMode",
      ),
  )
  def test_get_hadamard_block_size_invalid_args_raise(
      self, dim, max_hadamard_size, q_init_mode, sizing_mode, expected_error
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      hadamard_rotation.get_hadamard_block_size(
          dim,
          max_hadamard_size=max_hadamard_size,
          q_init_mode=q_init_mode,
          sizing_mode=sizing_mode,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="q_init_mode_none",
          m=8,
          q_init_mode=hadamard_rotation.QInitMode.NONE,
      ),
      dict(
          testcase_name="single_block_m1",
          m=1,
          q_init_mode=hadamard_rotation.QInitMode.AUTO,
      ),
  )
  def test_make_cross_block_matrix_returns_none(self, m, q_init_mode):
    self.assertIsNone(hadamard_rotation.make_cross_block_matrix(m, q_init_mode))

  @parameterized.named_parameters(
      dict(
          testcase_name="sylvester_hadamard_m8",
          m=8,
          seed=5,
          expected_kind="hadamard",
      ),
      dict(
          testcase_name="cached_np_int64_m6",
          m=np.int64(6),
          seed=np.int64(0),
          expected_kind="rademacher",
      ),
      dict(
          testcase_name="rademacher_cayley_m48",
          m=48,
          seed=5,
          expected_kind="rademacher",
      ),
      dict(
          testcase_name="polar_svd_m50",
          m=50,
          seed=5,
          expected_kind="polar_svd",
      ),
  )
  def test_make_cross_block_matrix_orthogonal_modes(
      self, m, seed, expected_kind
  ):
    q1 = hadamard_rotation.make_cross_block_matrix(
        m, hadamard_rotation.QInitMode.AUTO, seed=seed
    )
    assert q1 is not None
    m_int = int(m)
    seed_int = int(seed)
    self.assertEqual(q1.shape, (m_int, m_int))
    self.assertFalse(q1.flags.writeable)
    with self.assertRaises(ValueError):
      q1[0, 0] = 99.0
    np.testing.assert_allclose(
        q1.T @ q1, np.eye(m_int, dtype=np.float32), atol=1e-5
    )
    if expected_kind == "hadamard":
      expected_q1 = hadamard_rotation._make_hadamard_matrix(m_int).astype(
          np.float32
      )
    elif expected_kind == "rademacher":
      expected_q1 = hadamard_rotation._make_rademacher_cayley_matrix(
          m_int, np.random.default_rng(seed_int)
      )
    else:
      expected_q1 = hadamard_rotation._make_polar_svd_matrix(
          m_int, np.random.default_rng(seed_int)
      )
    np.testing.assert_array_equal(q1, expected_q1)

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_q_init_mode",
          m=8,
          q_init_mode="invalid",
          seed=0,
          expected_error="is not a valid QInitMode",
      ),
      dict(
          testcase_name="zero_m",
          m=0,
          q_init_mode="auto",
          seed=0,
          expected_error="m must be >= 1",
      ),
      dict(
          testcase_name="negative_m",
          m=-3,
          q_init_mode="auto",
          seed=0,
          expected_error="m must be >= 1",
      ),
      dict(
          testcase_name="bool_m",
          m=True,
          q_init_mode="auto",
          seed=0,
          expected_error="must be a non-boolean integer",
      ),
      dict(
          testcase_name="bool_seed",
          m=6,
          q_init_mode="auto",
          seed=True,
          expected_error="must be a non-boolean integer",
      ),
  )
  def test_make_cross_block_matrix_invalid_args_raise(
      self, m, q_init_mode, seed, expected_error
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      hadamard_rotation.make_cross_block_matrix(m, q_init_mode, seed=seed)

  def test_mse_grid_centered_int2_matches_int2_in_int4_and_reduces_mse(self):
    rng = np.random.default_rng(2029)
    weight = rng.standard_normal((16, 64)).astype(np.float32)

    qconfig_no_refine = _TensorQuantConfig(
        num_bits=2,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={
            "max_hadamard_size": 64,
            "scale_refinement": "none",
        },
    )
    qconfig_refine = _TensorQuantConfig(
        num_bits=2,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={
            "max_hadamard_size": 64,
            "scale_refinement": "mse_grid",
        },
    )
    qconfig_emulated = _TensorQuantConfig(
        num_bits=4,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        algorithm_params={
            "max_hadamard_size": 64,
            "int2_in_int4": True,
            "scale_refinement": "mse_grid",
        },
    )

    qp_no_refine = hadamard_rotation.get_tensor_quant_params(
        self._op_info, qconfig_no_refine, weight, None
    )
    qp_refine = hadamard_rotation.get_tensor_quant_params(
        self._op_info, qconfig_refine, weight, None
    )
    qp_emulated = hadamard_rotation.get_tensor_quant_params(
        self._op_info, qconfig_emulated, weight, None
    )

    self.assertIsNotNone(qp_no_refine.quantized_data)
    self.assertIsNotNone(qp_refine.quantized_data)
    self.assertIsNotNone(qp_emulated.quantized_data)
    q_no_refine = cast(np.ndarray, qp_no_refine.quantized_data)
    q_refine = cast(np.ndarray, qp_refine.quantized_data)
    q_emulated = cast(np.ndarray, qp_emulated.quantized_data)
    np.testing.assert_array_equal(qp_refine.zero_point, [[-0.5]])
    self.assertSetEqual(set(np.unique(q_refine).tolist()), {-2, -1, 0, 1})

    dequant_no_refine = (
        q_no_refine.astype(np.float32) + 0.5
    ) * qp_no_refine.scale.reshape(-1, 1)
    dequant_refine = (
        q_refine.astype(np.float32) + 0.5
    ) * qp_refine.scale.reshape(-1, 1)
    dequant_emulated = q_emulated.astype(
        np.float32
    ) * qp_emulated.scale.reshape(-1, 1)
    np.testing.assert_allclose(
        dequant_refine, dequant_emulated, rtol=1e-6, atol=1e-6
    )

    # Rotate weight identically to verify MSE reduction in rotated space.
    w_rot, _, _, _ = hadamard_rotation._rotate_with_diagonal_hadamard(
        weight, axis=1, max_size=64
    )
    mse_no_refine = float(np.mean((w_rot - dequant_no_refine) ** 2))
    mse_refine = float(np.mean((w_rot - dequant_refine) ** 2))
    self.assertLess(mse_refine, mse_no_refine)


class HadamardRotationEmbeddingLookupTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(888)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "embedding_lookup.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = None

  def test_materialize_embedding_lookup_basic(self):
    subgraph = self._test_model.subgraphs[0]
    embedding_subgraph_op_index = 0
    embedding_op = subgraph.operators[embedding_subgraph_op_index]
    op_info = qtyping.OpInfo(
        op=embedding_op,
        op_name=_TFLOpName.EMBEDDING_LOOKUP,
        subgraph_op_index=embedding_subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
        ),
    )
    params = hadamard_rotation.materialize_embedding_lookup_custom_op(
        op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv=self._tensor_name_to_qsv,
    )
    self.assertLen(params, 3)
    lookup = params[0]
    value = params[1]
    output = params[2]
    self.assertIsNone(lookup.producer)
    self.assertIsNotNone(lookup.consumers)
    self.assertIsNone(value.producer)
    self.assertIsNotNone(value.consumers)
    self.assertIsNotNone(output.producer)
    self.assertIsNone(output.consumers)
    self.assertEqual(
        lookup.consumers[0].transformations,
        [qtyping.QuantTransformation.NO_QUANTIZE],
    )
    self.assertEqual(
        value.consumers[0].transformations,
        [qtyping.QuantTransformation.QUANTIZE_TENSOR],
    )
    if output.producer is not None:
      self.assertEqual(
          output.producer.transformations,
          [qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION],
      )

  def test_materialize_embedding_lookup_decomposed(self):
    subgraph = self._test_model.subgraphs[0]
    embedding_subgraph_op_index = 0
    embedding_op = subgraph.operators[embedding_subgraph_op_index]
    op_info = qtyping.OpInfo(
        op=embedding_op,
        op_name=_TFLOpName.EMBEDDING_LOOKUP,
        subgraph_op_index=embedding_subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
        ),
    )
    params = hadamard_rotation.materialize_embedding_lookup_decomposed(
        op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv=self._tensor_name_to_qsv,
    )
    self.assertLen(params, 3)
    lookup = params[0]
    value = params[1]
    output = params[2]
    self.assertEqual(
        lookup.consumers[0].transformations,  # pyrefly: ignore[unsupported-operation]
        [qtyping.QuantTransformation.NO_QUANTIZE],
    )
    self.assertEqual(
        value.consumers[0].transformations,  # pyrefly: ignore[unsupported-operation]
        [qtyping.QuantTransformation.QUANTIZE_TENSOR],
    )
    if output.producer is not None:
      self.assertEqual(
          output.producer.transformations,
          [qtyping.QuantTransformation.INSERT_DECOMPOSED_HADAMARD_ROTATION],
      )


if __name__ == "__main__":
  absltest.main()
