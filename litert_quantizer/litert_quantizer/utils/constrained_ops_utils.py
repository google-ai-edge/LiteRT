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

"""Utils for handling operators with quantization constraints."""

from litert_quantizer import algorithm_manager
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.utils import common_utils


_OpQuantConstraint = common_utils.OpQuantConstraint


def get_constrained_op_list(
    quant_constraint: _OpQuantConstraint, verbose: bool = False
) -> list[str]:
  """Constructs and returns a list of constrained operators.

  This is achieved by invoking all materialization functions and extracting
  the constraint argument, using monkey patching to redirect logic to wrapper
  functions.

  Args:
    quant_constraint: The quantization constraint to filter operators by.
    verbose: Flag to enable verbose output.

  Returns:
    A list containing operators with the specified constraint.
  """
  constrained_ops = []

  def materialize_standard_op_wrapper(
      op_info: qtyping.OpInfo,
      *_args,
      constraint: _OpQuantConstraint = _OpQuantConstraint.NO_CONSTRAIN,
      **_kwargs,
  ) -> list[qtyping.TensorTransformationParams]:
    if constraint == quant_constraint:
      constrained_ops.append(op_info.op_name)
    # Return dummy values to avoid exceptions.
    dummy_value = [qtyping.TensorTransformationParams("")] * 2
    return dummy_value

  # Dummy implementation of the `_are_weights_too_small` function to support
  # `materialize_standard_op_wrapper` above.
  def are_weights_too_small_wrapper(*_args, **_kwargs) -> bool:
    return False

  # Dummy implementation of the `_materialize_bias_for_fc_conv_ops` function to
  # support `materialize_standard_op_wrapper` above.
  def materialize_bias_for_fc_conv_ops_wrapper(*_args, **_kwargs):
    return

  # Do monkey patch to intercept the `materialize_standard_op` function to
  # support `materialize_standard_op_wrapper` above.
  original_materialize_standard_op = common_utils.materialize_standard_op
  original_are_weights_too_small = common_quantize._are_weights_too_small  # pylint: disable=protected-access
  original_materialize_bias_for_fc_conv_ops = (
      common_quantize._materialize_bias_for_fc_conv_ops  # pylint: disable=protected-access
  )
  common_utils.materialize_standard_op = materialize_standard_op_wrapper
  common_quantize._are_weights_too_small = are_weights_too_small_wrapper  # pylint: disable=protected-access
  common_quantize._materialize_bias_for_fc_conv_ops = (  # pylint: disable=protected-access
      materialize_bias_for_fc_conv_ops_wrapper
  )
  minmax_func_dict = algorithm_manager.MIN_MAX_OP_NAME_MATERIALIZE_FUNC_DICT
  tensor_quant_params_cache = common_utils.TensorQuantParamsCache()

  # Loop over all available materialization functions to build up a list of
  # ops with the given constraint.
  for op, materialize_fn in minmax_func_dict.items():
    # Create a dummy op info to trigger the materialization.
    mock_op = qtyping.OperatorT()
    mock_op.inputs = [0]
    mock_op.outputs = [0]
    op_info = qtyping.OpInfo(
        op=mock_op,
        op_name=op,
        subgraph_op_index=0,
        op_quant_config=qtyping.OpQuantizationConfig(),
    )
    materialize_fn(
        get_tensor_quant_params_fn=None,  # pyrefly: ignore[bad-argument-type]
        op_info=op_info,
        graph_info=None,  # pyrefly: ignore[bad-argument-type]
        tensor_name_to_qsv=None,  # pyrefly: ignore[bad-argument-type]
        tensor_quant_params_cache=tensor_quant_params_cache,
    )

  if verbose:
    print(f"  {quant_constraint} op list: {constrained_ops}")

  # Restore the original functions.
  common_utils.materialize_standard_op = original_materialize_standard_op
  common_quantize._are_weights_too_small = original_are_weights_too_small  # pylint: disable=protected-access
  common_quantize._materialize_bias_for_fc_conv_ops = (  # pylint: disable=protected-access
      original_materialize_bias_for_fc_conv_ops
  )
  return constrained_ops
