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

import collections
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from litert_quantizer import default_policy
from litert_quantizer import qtyping
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


_ComputePrecision = qtyping.ComputePrecision
_QuantTransformation = qtyping.QuantTransformation
_TFLOpName = qtyping.TFLOperationName
_OpQuantConfig = qtyping.OpQuantizationConfig
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_DEFAULT_CONFIG_CHECK_POLICY = default_policy.DEFAULT_CONFIG_CHECK_POLICY


# TODO: b/335008966 - increase test coverage.
class MinMaxQuantizeUtilsTest(parameterized.TestCase):

  @parameterized.product(
      test_case=[
          # Tuple holds computation precision, whether to use SRQ and whether
          # to use explicit dequantize.
          (_ComputePrecision.FLOAT, False, True),  # WEIGHT_ONLY.
          (_ComputePrecision.INTEGER, False, False),  # DRQ.
          (_ComputePrecision.INTEGER, True, False),  # SRQ.
      ],
      is_inbounding_tensor=[True, False],
      is_constant=[True, False],
  )
  def test_get_tensor_transformations(
      self, test_case, is_inbounding_tensor, is_constant
  ):
    compute_precision, is_srq, explicit_dequantize = test_case
    weight_tensor_config = _TensorQuantConfig(num_bits=8)
    op_quant_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=weight_tensor_config if is_srq else None,
        compute_precision=compute_precision,
        explicit_dequantize=explicit_dequantize,
    )
    transformations = common_utils.get_tensor_transformations(
        op_quant_config, is_inbounding_tensor, is_constant
    )
    # Check if WEIGHT_ONLY.
    if (
        compute_precision == _ComputePrecision.FLOAT
        and op_quant_config.explicit_dequantize
    ):
      if is_inbounding_tensor and is_constant:
        self.assertSequenceEqual(
            transformations,
            [
                _QuantTransformation.ADD_DEQUANTIZE,
            ],
        )
      else:
        self.assertSequenceEqual(
            transformations,
            [_QuantTransformation.NO_QUANTIZE],
        )

    # Check if DRQ.
    if compute_precision == _ComputePrecision.INTEGER and not is_srq:
      if is_inbounding_tensor and is_constant:
        self.assertSequenceEqual(
            transformations, [_QuantTransformation.QUANTIZE_TENSOR]
        )
      else:
        self.assertSequenceEqual(
            transformations,
            [_QuantTransformation.NO_QUANTIZE],
        )

    # Check if SRQ.
    if compute_precision == _ComputePrecision.INTEGER and is_srq:
      if is_inbounding_tensor:
        if is_constant:
          self.assertSequenceEqual(
              transformations, [_QuantTransformation.QUANTIZE_TENSOR]
          )
        else:
          self.assertSequenceEqual(
              transformations, [_QuantTransformation.ADD_QUANTIZE]
          )
      else:
        self.assertSequenceEqual(
            transformations, [_QuantTransformation.ADD_DEQUANTIZE]
        )

  @parameterized.parameters((_TFLOpName.FULLY_CONNECTED), (_TFLOpName.CONV_2D))
  def test_check_weight_only_config_succeeds(self, op_name):
    self.assertIn(op_name, _DEFAULT_CONFIG_CHECK_POLICY.keys())

  @parameterized.parameters((_TFLOpName.RESHAPE), (_TFLOpName.AVERAGE_POOL_2D))
  def test_check_weight_only_config_raises_when_invalid_config(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
        ),
        compute_precision=_ComputePrecision.FLOAT,
    )
    error_message = (
        f"Quantization config {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      common_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  @parameterized.product(
      op_name=(_TFLOpName.FULLY_CONNECTED, _TFLOpName.CONV_2D),
      weight_num_bits=(2, 4, 8),
      granularity=(
          qtyping.QuantGranularity.TENSORWISE,
          qtyping.QuantGranularity.CHANNELWISE,
      ),
  )
  def test_check_drq_config_succeeds(
      self, op_name, weight_num_bits, granularity
  ):
    # TODO: b/353365054 - Remove this check after int4 DRQ is supported for
    # conv2d.
    if op_name == _TFLOpName.CONV_2D and weight_num_bits == 4:
      return
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_num_bits,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    common_utils.check_if_valid_op_config(
        op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
    )

  @parameterized.parameters((_TFLOpName.RESHAPE), (_TFLOpName.AVERAGE_POOL_2D))
  def test_check_drq_config_unsupported_op_raise_error(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    error_message = (
        f"Quantization config {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      common_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  @parameterized.parameters((_TFLOpName.FULLY_CONNECTED), (_TFLOpName.CONV_2D))
  def test_check_drq_config_wrong_bits_raise_error(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=3,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    error_message = (
        f"Quantization config {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      common_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  @parameterized.parameters((_TFLOpName.FULLY_CONNECTED), (_TFLOpName.CONV_2D))
  def test_check_drq_config_asymmetric_weights_raise_error(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            symmetric=False,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    error_message = (
        f"Quantization config {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      common_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  def test_check_drq_config_with_non_default_min_weight_elements_succeeds(self):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
        min_weight_elements=100,
    )
    common_utils.check_if_valid_op_config(
        _TFLOpName.CONV_2D, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
    )

  def test_check_config_with_non_default_algorithm_params_succeeds(self):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
            algorithm_params={"max_hadamard_size": 1024},
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    common_utils.check_if_valid_op_config(
        _TFLOpName.FULLY_CONNECTED,
        op_quant_config,
        _DEFAULT_CONFIG_CHECK_POLICY,
    )

  @parameterized.product(
      op_name=(_TFLOpName.FULLY_CONNECTED, _TFLOpName.CONV_2D),
      act_num_bits=(8, 16),
      weight_num_bits=(4, 8),
      granularity=(
          qtyping.QuantGranularity.TENSORWISE,
          qtyping.QuantGranularity.CHANNELWISE,
      ),
      symmetric_act=(True, False),
  )
  def test_check_srq_config_succeeds(
      self,
      op_name,
      act_num_bits,
      weight_num_bits,
      granularity,
      symmetric_act,
  ):
    # Asym int16 activation is not supported.
    if not symmetric_act and act_num_bits == 16:
      return
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=act_num_bits, symmetric=symmetric_act
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_num_bits,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    common_utils.check_if_valid_op_config(
        op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
    )

  def test_check_srq_config_unsupported_op_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Unsupported op for {op_quant_config.compute_precision}:"
        f" {_TFLOpName.CUSTOM_OP}"
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      common_utils.check_if_valid_op_config(
          _TFLOpName.CUSTOM_OP, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  def test_check_srq_config_wrong_act_bits_config_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=14, symmetric=True
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Quantization config {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      common_utils.check_if_valid_op_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          _DEFAULT_CONFIG_CHECK_POLICY,
      )

  def test_check_srq_config_asym_int16_act_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=16, symmetric=False
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Quantization config {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      common_utils.check_if_valid_op_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          _DEFAULT_CONFIG_CHECK_POLICY,
      )

  def test_check_srq_config_wrong_weight_bits_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=16, symmetric=True
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=3,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Quantization config {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      common_utils.check_if_valid_op_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          _DEFAULT_CONFIG_CHECK_POLICY,
      )

  def test_check_srq_config_asym_weight_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            symmetric=False,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Quantization config {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      common_utils.check_if_valid_op_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          _DEFAULT_CONFIG_CHECK_POLICY,
      )

  @parameterized.product(
      op_name=[
          _TFLOpName.FULLY_CONNECTED,
          _TFLOpName.CONV_2D,
      ],
      activation_tensor_config=[
          None,
          _TensorQuantConfig(num_bits=8, symmetric=False),
          _TensorQuantConfig(num_bits=16, symmetric=True),
      ],
      compute_precision=[
          _ComputePrecision.FLOAT,
          _ComputePrecision.INTEGER,
      ],
  )
  def test_check_supported_int4_config_succeeds(
      self, op_name, activation_tensor_config, compute_precision
  ):
    # Exclude invalid SRQ config.
    if (
        activation_tensor_config is not None
        and compute_precision != _ComputePrecision.INTEGER
    ) or (
        activation_tensor_config is None
        and compute_precision == _ComputePrecision.FLOAT
    ):
      return
    # TODO: b/353365054 - Remove this check after int4 DRQ is supported for
    # conv2d.
    if (
        # Check if DRQ and CONV_2D.
        compute_precision == _ComputePrecision.INTEGER
        and activation_tensor_config is None
        and op_name == _TFLOpName.CONV_2D
    ):
      return
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=activation_tensor_config,
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=compute_precision,
    )
    # Raise error if the config is not supported.
    # Check if DRQ.
    if (
        compute_precision == _ComputePrecision.INTEGER
        and op_quant_config.activation_tensor_config is None
    ):
      common_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )
    # Check if WEIGHT_ONLY.
    elif (
        compute_precision == _ComputePrecision.FLOAT
        and op_quant_config.explicit_dequantize
    ):
      self.assertIn(op_name, _DEFAULT_CONFIG_CHECK_POLICY.keys())
    # Check if SRQ.
    if (
        compute_precision == _ComputePrecision.INTEGER
        and op_quant_config.activation_tensor_config is not None
    ):
      common_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  @parameterized.product(
      op_name=[_TFLOpName.BATCH_MATMUL],
      activation_tensor_config=[
          None,
          _TensorQuantConfig(num_bits=8, symmetric=False),
          _TensorQuantConfig(num_bits=16, symmetric=True),
      ],
      test_case=[
          # Tuple holds compute precision and whether to use drq.
          (_ComputePrecision.INTEGER, True),
          (_ComputePrecision.INTEGER, False),
      ],
  )
  def test_check_unsupported_int4_config_raise_error(
      self, op_name, activation_tensor_config, test_case
  ):
    compute_precision, is_drq = test_case
    # Exclude invalid SRQ config.
    if (activation_tensor_config is not None and is_drq) or (
        activation_tensor_config is None and not is_drq
    ):
      return
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=activation_tensor_config,
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=compute_precision,
    )

    with self.assertRaises(ValueError):
      if is_drq:
        common_utils.check_if_valid_op_config(
            op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
        )
      elif not is_drq:
        common_utils.check_if_valid_op_config(
            op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
        )

  def test_materialize_op_with_output_activation_constraint_fails_for_multiple_output_op(
      self,
  ):
    # Create a mock op with multiple outputs.
    MockOp = collections.namedtuple("MockOp", ["outputs"])
    mock_op_info = qtyping.OpInfo(
        op=MockOp(outputs=[1, 2]),
        op_name=_TFLOpName.SOFTMAX,
        subgraph_op_index=0,
        op_quant_config=_OpQuantConfig(),
    )

    with self.assertRaisesRegex(
        ValueError, "only supports ops with a single output tensor"
    ):
      common_utils.materialize_op_with_output_activation_constraint(
          op_info=mock_op_info,
          graph_info=qtyping.GraphInfo([], []),
          tensor_name_to_qsv={},
          output_activation_constraints={},
          get_tensor_quant_params_fn=lambda *args: [],  # pyrefly: ignore[bad-argument-type]
          tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
      )

  def test_wrapper_passes_activation_qsv(
      self,
  ):
    """Tests if activation qsv is passed to get_tensor_quant_params_fn."""
    mock_act_tensor = mock.create_autospec(
        qtyping.TensorT, instance=True, spec_set=False
    )
    mock_act_tensor.name = b"activation"
    mock.seal(mock_act_tensor)
    mock_weight_tensor = mock.create_autospec(
        qtyping.TensorT, instance=True, spec_set=False
    )
    mock_weight_tensor.name = b"weight"
    mock_weight_tensor.buffer = 1
    mock.seal(mock_weight_tensor)

    mock_op = mock.create_autospec(
        qtyping.OperatorT, instance=True, spec_set=False
    )
    mock_op.inputs = [0, 1]  # input 0=activation, input 1=weight
    mock_op.outputs = []
    mock.seal(mock_op)
    mock_op_info = qtyping.OpInfo(
        op=mock_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=0,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=8),
            compute_precision=qtyping.ComputePrecision.INTEGER,
        ),
    )
    mock_graph_info = qtyping.GraphInfo(
        subgraph_tensors=[mock_act_tensor, mock_weight_tensor], buffers=[]
    )

    def dummy_get_tensor_params(
        op_info, tensor_quant_config, tensor_data, tensor_qsv
    ):
      del op_info, tensor_quant_config, tensor_data, tensor_qsv

    mock_get_tensor_params_fn = mock.create_autospec(
        dummy_get_tensor_params, spec_set=True
    )
    tensor_name_to_qsv = {
        "activation": {"min": -1, "max": 1, "hessian": 0.5},
        "weight": {"min": -10, "max": 10},
    }

    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_name",
            side_effect=lambda x: x.name.decode("utf-8"),
            autospec=True,
            spec_set=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_data",
            return_value=np.array([1]),
            autospec=True,
            spec_set=True,
        )
    )
    common_utils._get_tensor_transformation_params_wrapper(
        tensor=mock_weight_tensor,
        is_inbounding_tensor=True,
        op_info=mock_op_info,
        graph_info=mock_graph_info,
        tensor_name_to_qsv=tensor_name_to_qsv,
        get_tensor_quant_params_fn=mock_get_tensor_params_fn,
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
    )

    expected_tensor_qsv = dict(tensor_name_to_qsv["weight"])
    expected_tensor_qsv["activation_tensor_qsv"] = tensor_name_to_qsv[  # pyrefly: ignore[unsupported-operation]
        "activation"
    ]
    mock_get_tensor_params_fn.assert_called_once_with(
        mock_op_info,
        mock_op_info.op_quant_config.weight_tensor_config,
        mock.ANY,  # tensor_data
        expected_tensor_qsv,  # tensor_qsv
    )


class CommonUtilsTest(parameterized.TestCase):

  def test_reshape_to_blocks_symmetric_2d(self):
    # Shape (2, 6), quantized_dimension=1, block_size=3.
    # Result shape should be (4, 3).
    tensor = np.arange(12, dtype=np.float32).reshape(2, 6)
    expected = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float32
    )
    actual = common_utils.reshape_to_blocks(
        tensor, quantized_dimension=1, block_size=3
    )
    np.testing.assert_array_equal(actual, expected)

  def test_reshape_to_blocks_transposed_2d(self):
    # Shape (6, 2), quantized_dimension=0, block_size=3.
    # Result shape should be (4, 3).
    tensor = np.arange(12, dtype=np.float32).reshape(6, 2)
    expected = np.array(
        [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]], dtype=np.float32
    )
    actual = common_utils.reshape_to_blocks(
        tensor, quantized_dimension=0, block_size=3
    )
    np.testing.assert_array_equal(actual, expected)

  def test_reshape_to_blocks_3d(self):
    # Shape (2, 2, 4), quantized_dimension=2, block_size=2.
    # Result shape should be (8, 2).
    tensor = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    expected = np.array(
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
        dtype=np.float32,
    )
    actual = common_utils.reshape_to_blocks(
        tensor, quantized_dimension=2, block_size=2
    )
    np.testing.assert_array_equal(actual, expected)

  def test_reshape_to_blocks_not_divisible_raises_error(self):
    tensor = np.arange(10, dtype=np.float32).reshape(2, 5)
    with self.assertRaisesRegex(ValueError, "is not divisible by block size"):
      common_utils.reshape_to_blocks(
          tensor, quantized_dimension=1, block_size=3
      )


if __name__ == "__main__":
  absltest.main()
