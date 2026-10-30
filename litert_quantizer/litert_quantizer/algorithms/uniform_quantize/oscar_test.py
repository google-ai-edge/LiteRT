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

"""Tests for OSCAR optimal clipping and materialization."""

import pathlib
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import oscar
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('../../tests/models')
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


def _op_info(op_name: _TFLOpName) -> qtyping.OpInfo:
  return qtyping.OpInfo(
      op=qtyping.OperatorT(),
      op_name=op_name,
      op_quant_config=qtyping.OpQuantizationConfig(),
      subgraph_op_index=-1,
  )


def _tensor_qsv(mu2: Any) -> dict[str, Any]:
  return {'activation_tensor_qsv': {'mu2': mu2, 'num_samples': 128}}


def _dequantize(quant_params: qtyping.UniformQuantParams) -> np.ndarray:
  if quant_params.quantized_data is None:
    raise ValueError('quantized_data cannot be None')
  return uniform_quantize_tensor.uniform_dequantize(
      quant_params.quantized_data, quant_params
  )


class OscarQuantParamsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng(0)

  def _fc_problem(self, out_ch=16, in_ch=64):
    w = self._rng.normal(size=(out_ch, in_ch)).astype(np.float32)
    w[:, :4] *= 20.0
    mu2 = np.exp(self._rng.normal(size=in_ch))
    mu2[4:8] *= 100.0
    return (w, mu2)

  @parameterized.named_parameters(
      dict(
          testcase_name='channelwise',
          in_ch=32,
          out_ch=16,
          granularity=qtyping.QuantGranularity.CHANNELWISE,
          expected_scale_shape=(16, 1),
          expected_block_size=0,
          expected_quantized_dimension=0,
      ),
      dict(
          testcase_name='blockwise_32',
          in_ch=64,
          out_ch=8,
          granularity=qtyping.QuantGranularity.BLOCKWISE_32,
          expected_scale_shape=(8, 2),
          expected_block_size=32,
          expected_quantized_dimension=1,
      ),
  )
  def test_fc_shapes_and_reference_equivalence(
      self,
      in_ch: int,
      out_ch: int,
      granularity: qtyping.QuantGranularity,
      expected_scale_shape: tuple[int, int],
      expected_block_size: int,
      expected_quantized_dimension: int,
  ):
    """Verifies generated params match expected shape, block size, and dim."""
    w, mu2 = self._fc_problem(out_ch=out_ch, in_ch=in_ch)
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=granularity,
        ),
        w,
        _tensor_qsv(mu2),
    )
    self.assertEqual(quant_params.scale.shape, expected_scale_shape)
    self.assertEqual(
        quant_params.quantized_dimension, expected_quantized_dimension
    )
    self.assertEqual(quant_params.block_size, expected_block_size)
    np.testing.assert_array_equal(
        quant_params.zero_point, np.zeros(expected_scale_shape, dtype=np.int32)
    )
    self.assertIsNotNone(quant_params.quantized_data)
    q_data = cast(np.ndarray, quant_params.quantized_data)
    self.assertEqual(q_data.shape, w.shape)
    self.assertEqual(q_data.dtype, np.int8)
    self.assertGreaterEqual(q_data.min(), -8)
    self.assertLessEqual(q_data.max(), 7)

  def test_unsupported_op_raises(self):
    """Verifies attempting OSCAR quantization on CONV_2D raises ValueError."""
    w = self._rng.normal(size=(16, 3, 3, 32)).astype(np.float32)
    with self.assertRaisesRegex(ValueError, 'FULLY_CONNECTED only'):
      oscar.get_tensor_quant_params(
          _op_info(_TFLOpName.CONV_2D),
          _TensorQuantConfig(
              num_bits=4,
              symmetric=True,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          w,
          None,
      )

  def test_tensorwise_produces_scalar_params(self):
    """Verifies TENSORWISE granularity produces scalar scale and zero point."""
    w, mu2 = self._fc_problem()
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=8,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        w,
        _tensor_qsv(mu2),
    )
    self.assertEqual(quant_params.scale.shape, (1, 1))
    self.assertLessEqual(
        float(quant_params.scale.max()), float(np.abs(w).max()) / 127 + 1e-09
    )

  def test_missing_mu2_falls_back_to_uniform_weighting(self):
    """Verifies missing mu2 statistics trigger warning and uniform fallback."""
    w, _ = self._fc_problem(out_ch=16, in_ch=32)
    op_n = _op_info(_TFLOpName.FULLY_CONNECTED)
    with self.assertLogs(level='WARNING') as logs:
      quant_params = oscar.get_tensor_quant_params(
          op_n,
          _TensorQuantConfig(
              num_bits=4,
              symmetric=True,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          w,
          None,
      )
    self.assertTrue(
        any('OSCAR: no activation second moments' in log for log in logs.output)
    )
    self.assertEqual(quant_params.scale.shape, (16, 1))
    self.assertIsNotNone(quant_params.quantized_data)

  def test_clipping_improves_activation_weighted_error(self):
    """Verifies optimal clipping improves activation error vs min/max."""
    w, mu2 = self._fc_problem()
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        w,
        _tensor_qsv(mu2),
    )
    self.assertIsNotNone(quant_params.custom_algorithm_param)
    mult = quant_params.custom_algorithm_param['multiplier']
    dequant = _dequantize(quant_params) * mult
    bound = np.abs(w).max(axis=1, keepdims=True)
    zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
        -bound, bound, 4, True, qtyping.QuantGranularity.CHANNELWISE, None
    )
    rtn_params = qtyping.UniformQuantParams(
        scale=scale,
        zero_point=zp,
        num_bits=4,
        symmetric=True,
        quantized_dimension=0,
        block_size=0,
    )
    rtn_codes = uniform_quantize_tensor.uniform_quantize(w, rtn_params)
    rtn_dequant = uniform_quantize_tensor.uniform_dequantize(
        rtn_codes, rtn_params
    )

    def weighted_err(w_hat):
      diff = w_hat - w
      return float((diff * diff * mu2[None, :]).sum())

    self.assertLess(weighted_err(dequant), weighted_err(rtn_dequant))

  def test_asymmetric_weight_config_raises(self):
    """Verifies requesting asymmetric weight quantization raises ValueError."""
    w, mu2 = self._fc_problem()
    with self.assertRaisesRegex(ValueError, 'symmetric'):
      oscar.get_tensor_quant_params(
          _op_info(_TFLOpName.FULLY_CONNECTED),
          _TensorQuantConfig(
              num_bits=8,
              symmetric=False,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          w,
          _tensor_qsv(mu2),
      )

  def test_mismatched_mu2_size_raises(self):
    """Verifies mu2 with mismatched channel count raises ValueError."""
    w, _ = self._fc_problem(in_ch=64)
    with self.assertRaisesRegex(ValueError, 'channels'):
      oscar.get_tensor_quant_params(
          _op_info(_TFLOpName.FULLY_CONNECTED),
          _TensorQuantConfig(
              num_bits=4,
              symmetric=True,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          w,
          _tensor_qsv(np.ones(32)),
      )

  def test_channel_scale_objective(self):
    """Verifies _channel_scale_objective computes expected objective value."""
    w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    s = np.array([1.0, 2.0], dtype=np.float64)
    mu2 = np.array([1.0, 4.0], dtype=np.float64)
    obj = oscar._channel_scale_objective(w, s, mu2, block_size=0)
    self.assertAlmostEqual(obj, 160.0)

  def test_get_tensor_quant_params_mu2_scaling(self):
    """Verifies get_tensor_quant_params computes scales and clipping bounds."""
    w = np.array([[10.0, 1.0], [1.0, 10.0]], dtype=np.float32)
    mu2 = np.array([100.0, 1.0], dtype=np.float32)
    op_info = _op_info(_TFLOpName.FULLY_CONNECTED)
    config = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
    )
    quant_params = oscar.get_tensor_quant_params(
        op_info, config, w, _tensor_qsv(mu2)
    )
    s, _ = oscar._compute_channel_scales(w, mu2, 0)
    self.assertIsNotNone(s)
    w_scaled = w * s
    mu2_scaled = mu2 / (s * s)
    expected_bounds = oscar.get_clip_bounds(
        _TFLOpName.FULLY_CONNECTED,
        w_scaled,
        mu2_scaled,
        4,
        qtyping.QuantGranularity.CHANNELWISE,
    )
    _, expected_scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
        -expected_bounds,
        expected_bounds,
        4,
        True,
        qtyping.QuantGranularity.CHANNELWISE,
        None,
    )
    np.testing.assert_allclose(quant_params.scale, expected_scale, rtol=1e-5)

  def test_non_weight_tensor_falls_back_to_min_max(self):
    """Verifies non-weight tensors fall back to min/max quantization."""
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=8,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        tensor_content=None,
        tensor_qsv={'min': np.array([[-1.0]]), 'max': np.array([[2.0]])},
    )
    self.assertIsNone(quant_params.quantized_data)

  def test_calibrate(self):
    """Verifies calibrate computes activation second moments (mu2)."""
    tensor_content = np.array([[-1.0, 2.0], [3.0, -4.0]])
    mock_op = qtyping.OperatorT()
    mock_info = mock.create_autospec(
        qtyping.GraphInfo, instance=True, spec_set=True
    )

    self.enter_context(
        mock.patch.object(
            common_quantize,
            'get_tensor_indices_requiring_calibration',
            autospec=True,
            spec_set=True,
            return_value=[0, 1],
        )
    )
    self.enter_context(
        mock.patch.object(
            common_quantize,
            'collect_activation_tensor_statistics',
            autospec=True,
            spec_set=True,
            side_effect=[
                (
                    'tensor_0',
                    tensor_content,
                    {'min': np.array([-1.0]), 'max': np.array([3.0])},
                ),
                None,  # simulate constant or ignored tensor
            ],
        )
    )
    result = oscar.calibrate(
        tfl_op=mock_op,
        graph_info=mock_info,
        tensor_content_map={'tensor_0': tensor_content},
    )
    self.assertIn('tensor_0', result)
    self.assertIn('mu2', result['tensor_0'])
    # mu2 is mean(x*x, axis=0). x = [[-1, 2], [3, -4]]
    # x*x = [[1, 4], [9, 16]]
    # mean(x*x, axis=0) = [5.0, 10.0]
    np.testing.assert_array_equal(result['tensor_0']['mu2'], [5.0, 10.0])


class OscarMaterializeFullyConnectedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / 'conv_fc_mnist.tflite'
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    self._subgraph = self._test_model.subgraphs[0]
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._subgraph.tensors,
        buffers=self._test_model.buffers,
    )
    self._fc_subgraph_op_index = 3
    self._fc_op = self._subgraph.operators[self._fc_subgraph_op_index]
    input_tensor = self._subgraph.tensors[self._fc_op.inputs[0]]
    input_tensor_name = tfl_flatbuffer_utils.get_tensor_name(input_tensor)
    self._in_ch = input_tensor.shape[1]
    mu2 = np.ones(self._in_ch, dtype=np.float32)
    mu2[: self._in_ch // 2] = 100.0
    self._tensor_name_to_qsv = {
        input_tensor_name: {'mu2': mu2}
    }
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

  def test_materialize_fully_connected(self):
    """Verifies materialize_fully_connected attaches MULTIPLY and QUANTIZE."""
    params = oscar.materialize_fully_connected(
        self._op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv=self._tensor_name_to_qsv,
    )
    self.assertLen(params, 4)
    fc_input, weight, bias, output = params

    with self.subTest('input_activation_tensor'):
      self.assertIsNotNone(fc_input.consumers)
      self.assertEqual(
          fc_input.consumers[0].transformations,
          [qtyping.QuantTransformation.INSERT_MULTIPLY],
      )
      inp_params = fc_input.consumers[0].parameters
      self.assertIsInstance(inp_params, qtyping.UniformQuantParams)
      custom_params = inp_params.custom_algorithm_param
      self.assertIsNotNone(custom_params)
      self.assertIn('multiplier', custom_params)
      self.assertEqual(custom_params['multiplier'].shape, (self._in_ch,))
      self.assertFalse(
          np.allclose(
              custom_params['multiplier'],
              np.ones(self._in_ch, dtype=np.float32),
          )
      )

    with self.subTest('weight_tensor'):
      self.assertIsNotNone(weight.consumers)
      self.assertEqual(
          weight.consumers[0].transformations,
          [qtyping.QuantTransformation.QUANTIZE_TENSOR],
      )
      w_params = weight.consumers[0].parameters
      self.assertIsInstance(w_params, qtyping.UniformQuantParams)
      self.assertIsNotNone(w_params.quantized_data)

    with self.subTest('bias_tensor'):
      self.assertIsNotNone(bias.consumers)
      self.assertEqual(
          bias.consumers[0].transformations,
          [qtyping.QuantTransformation.NO_QUANTIZE],
      )

    with self.subTest('output_tensor'):
      self.assertIsNotNone(output.producer)
      self.assertEqual(
          output.producer.transformations,
          [qtyping.QuantTransformation.NO_QUANTIZE],
      )

  def test_materialize_fully_connected_no_mu2_falls_back(self):
    """Verifies missing QSV statistics trigger unit multiplier fallback."""
    params = oscar.materialize_fully_connected(
        self._op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv=None,
    )
    self.assertLen(params, 4)
    fc_input = params[0]
    self.assertIsNotNone(fc_input.consumers)
    inp_params = fc_input.consumers[0].parameters
    self.assertIsInstance(inp_params, qtyping.UniformQuantParams)
    custom_params = inp_params.custom_algorithm_param
    self.assertIsNotNone(custom_params)
    self.assertIn('multiplier', custom_params)
    np.testing.assert_allclose(
        custom_params['multiplier'], np.ones(self._in_ch, dtype=np.float32)
    )

  def test_materialize_fully_connected_input_tensor_not_in_qsv(self):
    """Verifies missing input tensor in QSV triggers unit scale fallback."""
    params = oscar.materialize_fully_connected(
        self._op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv={'unrelated_tensor_name': {'mu2': np.ones(10)}},
    )
    self.assertLen(params, 4)
    fc_input = params[0]
    self.assertIsNotNone(fc_input.consumers)
    inp_params = fc_input.consumers[0].parameters
    self.assertIsInstance(inp_params, qtyping.UniformQuantParams)
    custom_params = inp_params.custom_algorithm_param
    self.assertIsNotNone(custom_params)
    self.assertIn('multiplier', custom_params)
    np.testing.assert_allclose(
        custom_params['multiplier'], np.ones(self._in_ch, dtype=np.float32)
    )

  def test_materialize_fully_connected_without_bias(self):
    """Verifies op without bias returns 3 transformation params not 4."""
    fc_op_without_bias = qtyping.OperatorT()
    fc_op_without_bias.opcodeIndex = self._fc_op.opcodeIndex
    fc_op_without_bias.inputs = [
        self._fc_op.inputs[0],
        self._fc_op.inputs[1],
        -1,
    ]
    fc_op_without_bias.outputs = list(self._fc_op.outputs)
    fc_op_without_bias.builtinOptions = self._fc_op.builtinOptions

    op_info = qtyping.OpInfo(
        op=fc_op_without_bias,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=self._fc_subgraph_op_index,
        op_quant_config=self._op_info.op_quant_config,
    )
    params = oscar.materialize_fully_connected(
        op_info,
        self._graph_info,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
        tensor_name_to_qsv=self._tensor_name_to_qsv,
    )
    self.assertLen(params, 3)
    self.assertEqual(
        params[0].tensor_name,
        tfl_flatbuffer_utils.get_tensor_name(
            self._subgraph.tensors[self._fc_op.inputs[0]]
        ),
    )
    self.assertEqual(
        params[1].tensor_name,
        tfl_flatbuffer_utils.get_tensor_name(
            self._subgraph.tensors[self._fc_op.inputs[1]]
        ),
    )
    self.assertEqual(
        params[2].tensor_name,
        tfl_flatbuffer_utils.get_tensor_name(
            self._subgraph.tensors[self._fc_op.outputs[0]]
        ),
    )

  def test_materialize_unsupported_op_raises(self):
    """Verifies materializing a non-FULLY_CONNECTED op raises ValueError."""
    conv_op_info = qtyping.OpInfo(
        op=self._fc_op,
        op_name=_TFLOpName.CONV_2D,
        subgraph_op_index=0,
        op_quant_config=self._op_info.op_quant_config,
    )
    with self.assertRaisesRegex(ValueError, 'FULLY_CONNECTED only'):
      oscar.materialize_fully_connected(
          conv_op_info,
          self._graph_info,
          tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
          tensor_name_to_qsv=self._tensor_name_to_qsv,
      )


if __name__ == '__main__':
  absltest.main()

