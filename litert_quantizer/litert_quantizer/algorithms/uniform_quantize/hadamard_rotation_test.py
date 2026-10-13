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
    q_pow2 = hadamard_rotation._make_cross_block_matrix(8, "auto", seed=11)
    np.testing.assert_array_equal(
        q_pow2, hadamard_rotation._make_hadamard_matrix(8).astype(np.float32)
    )

    # Non-power-of-2 m=48 (`<= 48`) resolves to Rademacher Cayley.
    q_small = hadamard_rotation._make_cross_block_matrix(48, "auto", seed=11)
    q_rad = hadamard_rotation._make_rademacher_cayley_matrix(
        48, np.random.default_rng(11)
    )
    np.testing.assert_array_equal(q_small, q_rad)

    # Non-power-of-2 m=50 (`> 48`) resolves to Polar SVD.
    q_large = hadamard_rotation._make_cross_block_matrix(50, "auto", seed=11)
    q_svd = hadamard_rotation._make_polar_svd_matrix(
        50, np.random.default_rng(11)
    )
    np.testing.assert_array_equal(q_large, q_svd)

  def test_invalid_q_init_mode_raises(self):
    with self.assertRaisesRegex(ValueError, "Unsupported q_init_mode"):
      hadamard_rotation._make_cross_block_matrix(8, "invalid_mode")

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
        self._op_info, qconfig, weight, None, is_decomposed=True
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
    with self.assertRaisesRegex(ValueError, "Unsupported sizing_mode"):
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
        self._op_info, base_cfg, weight, None, is_decomposed=True
    )
    qp_ref = hadamard_rotation.get_tensor_quant_params(
        self._op_info, ref_cfg, weight, None, is_decomposed=True
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
