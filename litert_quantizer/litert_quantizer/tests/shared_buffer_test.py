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

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized

from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils


_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig


class SharedBufferTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # This two-signature model demonstrates both constant tensor and constant
    # buffer sharing. Each signature has two consecutive fully connected (FC)
    # layers that share a weight tensor: the first signature uses
    # `arith.constant` and the second uses `arith.constant1`, demonstrating
    # constant tensor sharing. Furthermore, 'arith.constant' and
    # 'arith.constant1' share a buffer, demonstrating constant buffer sharing.
    self.float_model_path = test_utils.get_path_to_datafile(
        'models/constant_tensor_and_buffer_only_sharing_weight_fcs.tflite'
    )
    self.sig1_output_tensor_name = 'PartitionedCall:0'
    self.sig2_output_tensor_name = 'PartitionedCall_1:0'

  def _get_fc_recipe_entry(
      self, regex: str, num_bits: int, granularity: str, srq: bool = False
  ):
    op_config = {
        'weight_tensor_config': {
            'num_bits': num_bits,
            'symmetric': True,
            'granularity': granularity,
            'dtype': 'INT',
            'block_size': 0,
        },
        'compute_precision': 'INTEGER',
        'explicit_dequantize': False,
        'skip_checks': False,
        'min_weight_elements': 0,
    }
    if srq:
      op_config['activation_tensor_config'] = {
          'num_bits': 8,
          'symmetric': False,
          'granularity': 'TENSORWISE',
          'dtype': 'INT',
      }
    return {
        'regex': regex,
        'operation': 'FULLY_CONNECTED',
        'algorithm_key': 'min_max_uniform_quantize',
        'op_config': op_config,
    }

  def _check_comparison_result(
      self,
      output_tensor_names: Sequence[str],
      comparison_result,
      output_tolerance: float,
  ):
    _all_results = comparison_result.get_all_tensor_results()
    metric = 'mean_squared_difference'
    with self.subTest(error_metric=metric):
      tensors_results = {k: v.get(metric, 0.0) for k, v in _all_results.items()}
      for output_tensor_name in output_tensor_names:
        self.assertLess(
            tensors_results[output_tensor_name],
            output_tolerance,
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='sig1_fcs_quant_sig2_fcs_no_quant',
          sig1_num_bits=8,
          sig2_num_bits=None,
          output_tolerance=1e-2,
      ),
      dict(
          testcase_name='sig1_fcs_no_quant_sig2_fcs_quant',
          sig1_num_bits=None,
          sig2_num_bits=8,
          output_tolerance=1e-2,
      ),
      dict(
          testcase_name='sig1_fcs_quant_sig2_fcs_quant_different_params',
          sig1_num_bits=8,
          sig2_num_bits=4,
          output_tolerance=0.5,
      ),
  )
  def test_quantization_succeeds_for_distinct_constant_tensors_with_shared_buffer_and_different_quantization_params(
      self, sig1_num_bits, sig2_num_bits, output_tolerance
  ):
    # This test checks a constant buffer shared by tensors `arith.constant`
    # and `arith.constant1` from FCs from two different signatures.
    qt = quantizer.Quantizer(self.float_model_path)

    sig1_fc1_regex = 'BatchMatMulV3;'
    sig1_fc2_regex = f'{self.sig1_output_tensor_name};'
    sig2_fc1_regex = 'BatchMatMulV31;'
    sig2_fc2_regex = f'{self.sig2_output_tensor_name};'
    recipe = []
    # Use same quantization params for FCs within each signature to ensure we
    # don't get a tensor sharing error.
    if sig1_num_bits is not None:
      recipe.append(
          self._get_fc_recipe_entry(
              sig1_fc1_regex, sig1_num_bits, 'CHANNELWISE'
          )
      )
      recipe.append(
          self._get_fc_recipe_entry(
              sig1_fc2_regex, sig1_num_bits, 'CHANNELWISE'
          )
      )
    if sig2_num_bits is not None:
      recipe.append(
          self._get_fc_recipe_entry(
              sig2_fc1_regex, sig2_num_bits, 'CHANNELWISE'
          )
      )
      recipe.append(
          self._get_fc_recipe_entry(
              sig2_fc2_regex, sig2_num_bits, 'CHANNELWISE'
          )
      )
    qt.load_quantization_recipe(recipe)
    quantized_model = qt.quantize().quantized_model
    self.assertIsNotNone(quantized_model)

    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        quantized_model, num_samples=4
    )
    comparison_result = qt.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE], test_data=test_data
    )
    self._check_comparison_result(
        output_tensor_names=[
            self.sig1_output_tensor_name,
            self.sig2_output_tensor_name,
        ],
        comparison_result=comparison_result,
        output_tolerance=output_tolerance,
    )

  def test_decomposed_hadamard_matrices_share_buffer(self):
    qt = quantizer.Quantizer(self.float_model_path)
    recipe = [
        {
            'regex': '.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'DECOMPOSED_HADAMARD_ROTATION',
            'op_config': {
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': 'CHANNELWISE',
                    'dtype': 'INT',
                },
                'compute_precision': 'INTEGER',
                'explicit_dequantize': False,
                'skip_checks': False,
                'min_weight_elements': 0,
            },
        }
    ]
    qt.load_quantization_recipe(recipe)
    quantized_model_bytes = qt.quantize().quantized_model
    self.assertIsNotNone(quantized_model_bytes)

    model = qtyping.Model.GetRootAsModel(quantized_model_bytes, 0)
    subgraph = model.Subgraphs(0)

    hadamard_tensor_ids = []
    for i in range(subgraph.OperatorsLength()):
      op = subgraph.Operators(i)
      if (
          model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
          == qtyping.BuiltinOperator.FULLY_CONNECTED
      ):
        # Weight tensor is input at index 1
        weight_tensor_id = op.Inputs(1)
        weight_tensor = subgraph.Tensors(weight_tensor_id)
        if b'_hadamard_matrix' in weight_tensor.Name():
          hadamard_tensor_ids.append(weight_tensor_id)

    # We expect that there are multiple Hadamard FC operations inserted.
    self.assertGreaterEqual(len(hadamard_tensor_ids), 2)
    # Verify that all of these Hadamard FC ops point to the same tensor ID!
    self.assertLen(set(hadamard_tensor_ids), 1)

  def test_decomposed_hadamard_with_max_size(self):
    qt = quantizer.Quantizer(self.float_model_path)
    recipe = [
        {
            'regex': '.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'DECOMPOSED_HADAMARD_ROTATION',
            'op_config': {
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': 'CHANNELWISE',
                    'dtype': 'INT',
                    'algorithm_params': {
                        'max_hadamard_size': 2,
                    },
                },
                'compute_precision': 'INTEGER',
                'explicit_dequantize': False,
                'skip_checks': False,
                'min_weight_elements': 0,
            },
        }
    ]
    qt.load_quantization_recipe(recipe)
    quantized_model_bytes = qt.quantize().quantized_model
    self.assertIsNotNone(quantized_model_bytes)

    model = qtyping.Model.GetRootAsModel(quantized_model_bytes, 0)
    subgraph = model.Subgraphs(0)

    hadamard_sizes = []
    for i in range(subgraph.TensorsLength()):
      tensor = subgraph.Tensors(i)
      if b'_hadamard_matrix' in tensor.Name():
        hadamard_sizes.append(list(tensor.ShapeAsNumpy()))

    self.assertNotEmpty(hadamard_sizes)
    for shape in hadamard_sizes:
      self.assertEqual(shape, [2, 2])

  @parameterized.named_parameters(
      dict(
          testcase_name='fc1_quant_fc2_no_quant',
          fc1_num_bits=8,
          fc2_num_bits=None,
          fc1_granularity='CHANNELWISE',
          fc2_granularity='CHANNELWISE',
          fc1_srq=False,
          fc2_srq=False,
          output_tolerance=0.0015,
      ),
      dict(
          testcase_name='fc1_no_quant_fc2_quant',
          fc1_num_bits=None,
          fc2_num_bits=8,
          fc1_granularity='CHANNELWISE',
          fc2_granularity='CHANNELWISE',
          fc1_srq=False,
          fc2_srq=False,
          output_tolerance=0.0015,
      ),
      dict(
          testcase_name='fc1_quant_fc2_quant_different_params',
          fc1_num_bits=8,
          fc2_num_bits=4,
          fc1_granularity='CHANNELWISE',
          fc2_granularity='CHANNELWISE',
          fc1_srq=False,
          fc2_srq=False,
          output_tolerance=0.25,
      ),
      dict(
          testcase_name='fc1_tensorwise_fc2_channelwise',
          fc1_num_bits=8,
          fc2_num_bits=8,
          fc1_granularity='TENSORWISE',
          fc2_granularity='CHANNELWISE',
          fc1_srq=False,
          fc2_srq=False,
          output_tolerance=0.25,
      ),
      dict(
          testcase_name='fc1_channelwise_fc2_tensorwise',
          fc1_num_bits=8,
          fc2_num_bits=8,
          fc1_granularity='CHANNELWISE',
          fc2_granularity='TENSORWISE',
          fc1_srq=False,
          fc2_srq=False,
          output_tolerance=0.25,
      ),
      dict(
          testcase_name='fc1_tensorwise_fc2_tensorwise_different_bits',
          fc1_num_bits=8,
          fc2_num_bits=4,
          fc1_granularity='TENSORWISE',
          fc2_granularity='TENSORWISE',
          fc1_srq=False,
          fc2_srq=False,
          output_tolerance=0.30,
      ),
      dict(
          testcase_name='fc1_srq_fc2_drq',
          fc1_num_bits=8,
          fc2_num_bits=8,
          fc1_granularity='CHANNELWISE',
          fc2_granularity='CHANNELWISE',
          fc1_srq=True,
          fc2_srq=False,
          output_tolerance=0.60,
      ),
      dict(
          testcase_name='fc1_drq_fc2_srq',
          fc1_num_bits=8,
          fc2_num_bits=8,
          fc1_granularity='CHANNELWISE',
          fc2_granularity='CHANNELWISE',
          fc1_srq=False,
          fc2_srq=True,
          output_tolerance=0.60,
      ),
      dict(
          testcase_name='fc1_srq_fc2_srq',
          fc1_num_bits=8,
          fc2_num_bits=8,
          fc1_granularity='CHANNELWISE',
          fc2_granularity='TENSORWISE',
          fc1_srq=True,
          fc2_srq=True,
          output_tolerance=0.60,
      ),
  )
  def test_quantization_succeeds_for_a_constant_tensor_with_different_quantization_params(
      self,
      fc1_num_bits,
      fc2_num_bits,
      output_tolerance,
      fc1_granularity,
      fc2_granularity,
      fc1_srq,
      fc2_srq,
  ):
    # This test checks a constant tensor `arith.constant` shared by FCs in the
    # first signature.
    qt = quantizer.Quantizer(self.float_model_path)

    sig1_fc1_regex = 'BatchMatMulV3'
    sig1_fc2_regex = 'PartitionedCall:0'

    recipe = []
    if fc1_num_bits is not None:
      recipe.append(
          self._get_fc_recipe_entry(
              sig1_fc1_regex, fc1_num_bits, fc1_granularity, srq=fc1_srq
          )
      )
    if fc2_num_bits is not None:
      recipe.append(
          self._get_fc_recipe_entry(
              sig1_fc2_regex, fc2_num_bits, fc2_granularity, srq=fc2_srq
          )
      )
    qt.load_quantization_recipe(recipe)

    calibration_result = None
    if qt.need_calibration:
      calib_data = tfl_interpreter_utils.create_random_normal_input_data(
          self.float_model_path, num_samples=5
      )
      calibration_result = qt.calibrate(calib_data)
    quantized_model = qt.quantize(
        calibration_result=calibration_result
    ).quantized_model
    self.assertIsNotNone(quantized_model)

    model = qtyping.Model.GetRootAsModel(quantized_model, 0)
    subgraph = model.Subgraphs(0)
    fc_ops = []
    for i in range(subgraph.OperatorsLength()):
      op = subgraph.Operators(i)
      op_code = model.OperatorCodes(op.OpcodeIndex())
      if op_code.BuiltinCode() == qtyping.BuiltinOperator.FULLY_CONNECTED:
        fc_ops.append(op)

    sig1_fc1_op, sig1_fc2_op = None, None
    for op in fc_ops:
      output_tensor = subgraph.Tensors(op.Outputs(0))
      if output_tensor.Name() == b'PartitionedCall:0':
        sig1_fc2_op = op
      elif output_tensor.Name() == b'BatchMatMulV3':
        sig1_fc1_op = op

    self.assertIsNotNone(sig1_fc1_op)
    self.assertIsNotNone(sig1_fc2_op)

    if fc1_num_bits is not None:
      fc1_weight = subgraph.Tensors(sig1_fc1_op.Inputs(1))
      self.assertIsNotNone(fc1_weight.Quantization())
      self.assertIsNotNone(fc1_weight.Quantization().ScaleAsNumpy())
      if fc1_granularity == 'TENSORWISE':
        self.assertEqual(fc1_weight.Quantization().ScaleAsNumpy().size, 1)
      elif fc1_granularity == 'CHANNELWISE':
        self.assertEqual(fc1_weight.Quantization().ScaleAsNumpy().size, 4)

    if fc2_num_bits is not None:
      fc2_weight = subgraph.Tensors(sig1_fc2_op.Inputs(1))
      self.assertIsNotNone(fc2_weight.Quantization())
      self.assertIsNotNone(fc2_weight.Quantization().ScaleAsNumpy())
      if fc2_granularity == 'TENSORWISE':
        self.assertEqual(fc2_weight.Quantization().ScaleAsNumpy().size, 1)
      elif fc2_granularity == 'CHANNELWISE':
        self.assertEqual(fc2_weight.Quantization().ScaleAsNumpy().size, 4)

    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        quantized_model, num_samples=4
    )
    comparison_result = qt.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE], test_data=test_data
    )
    self._check_comparison_result(
        output_tensor_names=[
            self.sig1_output_tensor_name,
            self.sig2_output_tensor_name,
        ],
        comparison_result=comparison_result,
        output_tolerance=output_tolerance,
    )


if __name__ == '__main__':
  absltest.main()
