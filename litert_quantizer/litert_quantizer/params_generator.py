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

"""Generate model tensor level quantization config."""

from collections.abc import Sequence
import copy
from typing import Any, Optional
import warnings

from ai_edge_litert.tools import mmap_utils
from litert_quantizer import algorithm_manager
from litert_quantizer import default_policy as policy
from litert_quantizer import qtyping
from litert_quantizer import recipe_manager
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import progress_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_QuantTrans = qtyping.QuantTransformation
_OpName = qtyping.TFLOperationName


class ParamsGenerator:
  """Generate model tensor level quantization parameters."""

  def __init__(self, float_tflite: qtyping.ModelT):
    self.float_model: qtyping.ModelT = float_tflite

    if not tfl_flatbuffer_utils.is_float_model(self.float_model):
      warnings.warn(
          'Input model is already partially quantized. Proceeding with'
          ' re-quantization; existing quantized tensors will remain unchanged'
          ' regardless of the selected recipe.'
      )

    self._check_tensor_names_are_unique()
    self.buffer_to_tensors: dict[int, list[Any]] = (
        tfl_flatbuffer_utils.buffer_to_tensors(self.float_model)
    )
    self.model_quant_results: dict[str, qtyping.TensorTransformationParams] = {}
    self._tensor_quant_params_cache = common_utils.TensorQuantParamsCache()

  def _get_total_operations(self) -> int:
    """Returns the total number of operations.

    This is the total number of operations to process in order to generate
    quantization parameters.
    """
    total_ops = sum(
        len(subgraph.operators)
        + 2  # Input and output operators.
        for subgraph in self.float_model.subgraphs
    )
    return total_ops

  def generate_quantization_parameters(
      self,
      model_recipe_manager: recipe_manager.RecipeManager,
      model_qsvs: Optional[dict[str, qtyping.QSV]] = None,
      enable_progress_bar: bool | None = None,
  ) -> dict[str, qtyping.TensorTransformationParams]:
    """Generate the quantization parameters for the model.

    Args:
      model_recipe_manager: The recipe manager for the model.
      model_qsvs: Quantization statistics values (QSVs) for the model. This is
        obtained through calibration process.
      enable_progress_bar: Whether to enable the progress bar. By default, it is
        disabled for smaller models and enabled for larger models.

    Returns:
      model_quant_results: The quantization parameters for tensors in the model.

    Raises:
      RuntimeError: If the calibration dataset is required but not provided.
    """
    if model_recipe_manager.need_calibration() and not model_qsvs:
      raise RuntimeError(
          'Model quantization statistics values (QSVs) are required for the'
          ' input recipe. This can be obtained by running calibration on sample'
          ' dataset.'
      )

    if model_qsvs is None:
      model_qsvs = {}

    skip_subgraphs = set()
    op_codes = self.float_model.operatorCodes

    total_ops = self._get_total_operations()
    with progress_utils.ProgressBar(
        total_ops,
        'Generating Quantization Parameters:',
        enable=enable_progress_bar,
        # Progress bar will be skipped for smaller models.
    ) as progress_bar:
      for sg_ind, subgraph in enumerate(self.float_model.subgraphs):

        graph_info = qtyping.GraphInfo(
            subgraph.tensors, self.float_model.buffers
        )
        # Add input/output operators to the subgraph.
        subgraph_operators = subgraph.operators + (
            tfl_flatbuffer_utils.get_subgraph_input_output_operators(subgraph)
        )
        for subgraph_op_id, op in enumerate(subgraph_operators):
          progress_bar.update_single_step()
          # Get the op key.
          if isinstance(op, qtyping.IOOperator):
            op_key = op.op_key
            subgraph_op_id = -1  # Virtual op, no real id.
          else:
            op_code = op_codes[op.opcodeIndex].builtinCode
            # Do not quantize unknown ops.
            if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
              op_quant_results = self._get_params_for_no_quant_op(
                  subgraph_op_id, op, subgraph.tensors
              )
              self._update_model_quant_results(op_quant_results)
              continue
            op_key = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[op_code]

            # Step1: query the quantization_recipe to get op config.
          op_scope = tfl_flatbuffer_utils.get_op_scope(op, subgraph.tensors)
          algorithm_name, op_quant_config = (
              model_recipe_manager.get_quantization_configs(op_key, op_scope)
          )

          if sg_ind in skip_subgraphs or policy.is_non_quantizable_composite_op(
              op
          ):
            algorithm_name = algorithm_manager.AlgorithmName.NO_QUANTIZE

          if algorithm_name == algorithm_manager.AlgorithmName.NO_QUANTIZE:
            side_effect_subgraphs = (
                tfl_flatbuffer_utils.get_op_side_effect_subgraphs(op)
            )
            skip_subgraphs.update(side_effect_subgraphs)

            op_quant_results = self._get_params_for_no_quant_op(
                subgraph_op_id, op, subgraph.tensors
            )

          else:
            op_info = qtyping.OpInfo(
                op, op_key, subgraph_op_id, op_quant_config
            )
            # Step2: query algorithm_manager to get/call the related function.
            materialize_func = algorithm_manager.get_quantization_func(
                algorithm_name,
                op_key,
                qtyping.QuantizeMode.MATERIALIZE,
            )
            op_quant_results = materialize_func(
                op_info=op_info,
                graph_info=graph_info,
                tensor_name_to_qsv=model_qsvs,
                tensor_quant_params_cache=self._tensor_quant_params_cache,
            )
          # Reclaim memory for op's tensors that have data.
          for tensor_idx in list(op.inputs):
            if tensor_idx != -1:
              buffer_idx = subgraph.tensors[tensor_idx].buffer
              if (
                  buffer := self.float_model.buffers[buffer_idx]
              ).data is not None:
                mmap_utils.advise_dont_need(buffer.data)

          # Step3: update the results.
          self._update_model_quant_results(op_quant_results)
    self._post_process_results()
    return self.model_quant_results

  def _check_tensor_names_are_unique(self):
    """Checks if the tensor names are unique in the model."""
    global_tensor_names = set()
    for subgraph in self.float_model.subgraphs:
      for tensor in subgraph.tensors:
        tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
        if tensor_name in global_tensor_names:
          raise ValueError(
              'Tensor name %s is not unique in the model. Please check your'
              ' model and rename the tensor as ParamsGenerator assumes tensor'
              ' names are unique.' % tensor_name
          )
        global_tensor_names.add(tensor_name)

  def _post_process_results(self) -> None:
    """Post process the quantization results.

    Raises:
      RuntimeError: If the tensors sharing the same buffer have different
      quantization settings.
    """
    self._check_and_fix_buffer_sharing()

  def _update_model_quant_results(
      self,
      op_tensor_results: list[qtyping.TensorTransformationParams],
  ) -> None:
    """Update the op quantization results to the final output.

    Args:
      op_tensor_results: Tensor level quantization params for the op.

    Raises:
      RuntimeError: If the same tensor has multiple quantization configs.
    """

    for op_tensor_result in op_tensor_results:
      tensor_name = op_tensor_result.tensor_name
      if tensor_name not in self.model_quant_results:
        self.model_quant_results[tensor_name] = copy.copy(op_tensor_result)
      else:
        tensor_params = self.model_quant_results[tensor_name]
        # Set source op.
        if op_tensor_result.producer is not None:
          # Src params must be unique (a tensor can only be produced by one op).
          if tensor_params.producer is not None:
            raise RuntimeError(
                'Tensor %s received multiple quantization parameters from the'
                ' source op, which should not happen as every tensor should'
                ' have only one source op.' % tensor_name
            )
          tensor_params.producer = op_tensor_result.producer
        # Set target op, which can be multiple (a tensor can be consumed by
        # multiple ops).
        if op_tensor_result.consumers is not None:
          if tensor_params.consumers is None:
            tensor_params.consumers = op_tensor_result.consumers[:]
          else:
            tensor_params.consumers += op_tensor_result.consumers[:]
        self.model_quant_results[tensor_name] = tensor_params

  def _get_params_for_no_quant_op(
      self,
      subgraph_op_id: int,
      op: Any,
      subgraph_tensors: list[Any],
  ) -> list[qtyping.TensorTransformationParams]:
    """Get the quantization parameters for ops require no quantization.

    Args:
      subgraph_op_id: The op id in the subgraph.
      op: The op that needs to be parsed.
      subgraph_tensors: Tensors in the subgraph.

    Returns:
      Tensor level quantization params for the op.
    """

    def no_quant_tensor_params():
      return qtyping.OpToTensorParams(
          subgraph_op_id=subgraph_op_id,
          transformations=[_QuantTrans.NO_QUANTIZE],
      )

    tensor_params = []
    for input_tensor_idx in op.inputs:
      if input_tensor_idx != -1:
        tensor = subgraph_tensors[input_tensor_idx]
        input_tensor_params = qtyping.TensorTransformationParams(
            tensor_name=tfl_flatbuffer_utils.get_tensor_name(tensor),
            consumers=[no_quant_tensor_params()],
        )
        tensor_params.append(input_tensor_params)

    for output_tensor_idx in op.outputs:
      if output_tensor_idx != -1:
        tensor = subgraph_tensors[output_tensor_idx]
        output_tensor_params = qtyping.TensorTransformationParams(
            tensor_name=tfl_flatbuffer_utils.get_tensor_name(tensor),
            producer=no_quant_tensor_params(),
        )
        tensor_params.append(output_tensor_params)
    return tensor_params

  def _mark_tensors_requiring_buffer_duplication(
      self, buffers_to_duplicate: Sequence[int]
  ) -> None:
    """Mark tensors that require buffer duplication.

    Marking a tensor means adding a DUPLICATE_BUFFER transformation as the first
    transformation to be applied for each consumer of the tensor. Need to do
    that for each consumer to preserve a zero layer and not affect the
    horizontal optimization later in the transformation instructions generator.

    Marks all tensors within each of the provided buffers as requiring buffer
    duplication, except for the last tensor. The order of tensors is assumed to
    be the same during both the marking and transformation performer steps, as
    determined by `self.buffer_to_tensors`. This allows the final tensor to
    reuse the original buffer, as it is not marked for duplication.

    Args:
      buffers_to_duplicate: Indices of the buffers to duplicate.
    """
    for buffer_idx in buffers_to_duplicate:
      for tensor in self.buffer_to_tensors[buffer_idx][:-1]:
        tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
        for consumer_params in self.model_quant_results[tensor_name].consumers:  # pyrefly: ignore[not-iterable]
          consumer_params.transformations.insert(
              0, _QuantTrans.DUPLICATE_BUFFER
          )

  def _mark_tensors_requiring_tensor_duplication(
      self, tensor_names_to_duplicate
  ) -> None:
    """Mark tensors that require tensor duplication.

    Marking a tensor means adding a DUPLICATE_TENSOR transformation as the first
    transformation to be applied for each consumer of the tensor. Need to do
    that for each consumer to preserve a zero layer and not affect the
    horizontal optimization later in the transformation instructions generator.

    Args:
      tensor_names_to_duplicate: Names of tensors to duplicate.
    """
    for tensor_name in tensor_names_to_duplicate:
      for consumer_params in self.model_quant_results[tensor_name].consumers:  # pyrefly: ignore[not-iterable]
        consumer_params.transformations.insert(0, _QuantTrans.DUPLICATE_TENSOR)

  def _check_buffer_sharing_for_tensor(self, tensor: Any) -> bool:
    """Check buffer sharing for the tensor against itself.

    Args:
      tensor: The tensor to check.

    Returns:
      Whether the tensor has compatible quantization parameters.

    Raises:
      RuntimeError: If the tensor has incompatible quantization parameters
      and the buffer is not constant.
    """
    tensor_params = self.model_quant_results.get(
        tfl_flatbuffer_utils.get_tensor_name(tensor), None
    )
    if tensor_params is None:
      return True

    if _are_tensor_consumer_params_compatible(tensor_params):
      return True
    elif _is_constant_tensor(tensor, self.float_model.buffers):
      return False
    else:
      error_msg = (
          f'The tensor {tensor.name} consumers do not have the same'
          ' quantization parameters. Please modify your quantization recipe to'
          ' make sure the two tensors have the same quantization settings.'
      )
      raise RuntimeError(error_msg)

  def _check_buffer_sharing_for_self_compatible_tensors(
      self, tensor1: Any, tensor2: Any
  ) -> bool:
    """Checks that self-compatible tensors have the same quantization params.

    Self-compatible means that all tensor's consumers have the same quantization
    parameters.

    Args:
      tensor1: The first tensor to check.
      tensor2: The second tensor to check.

    Returns:
      Whether the tensors have compatible quantization parameters.

    Raises:
      RuntimeError: If the tensors have incompatible quantization parameters
      and the buffer is not constant.
    """
    tensor1_params = self.model_quant_results.get(
        tfl_flatbuffer_utils.get_tensor_name(tensor1), None
    )
    tensor2_params = self.model_quant_results.get(
        tfl_flatbuffer_utils.get_tensor_name(tensor2), None
    )

    if tensor1_params is None or tensor2_params is None:
      return True

    if _are_self_compatible_tensors_compatible_to_each_other(
        tensor1_params, tensor2_params
    ):
      return True
    elif _is_constant_tensor(tensor1, self.float_model.buffers):
      return False
    else:
      error_msg = (
          f'The tensors {tensor1.name} and {tensor2.name} do not have'
          ' the same quantization parameters even though they share the'
          ' same buffer. Please modify your quantization recipe to make'
          ' sure the two tensors have the same quantization settings.'
      )
      raise RuntimeError(error_msg)

  def _check_and_fix_buffer_sharing(self) -> None:
    """Check and fix tensor/buffer sharing issues when possible.

    This function checks if tensors sharing the same buffer have the same
    quantization settings. If not, when it's possible, it will fix it by marking
    such tensors or buffers to be duplicated. Otherwise, it will raise an error.

    Possible cases that can be fixed by duplication:
      1. A constant tensor recieves different quantization parameters from its
      consumers. In this case, the tensor is marked for duplication.
      2. Two or more tensors share the same constant buffer and have different
      quantization parameters. In this case, the buffer is marked for
      duplication.

    Raises:
      RuntimeError: If the tensors sharing the same buffer have different
        quantization settings and it can't be resolved by duplicating the
        buffer/tensor.
    """
    buffers_to_duplicate = []
    tensor_names_to_duplicate = []
    for buffer_idx, tensors in self.buffer_to_tensors.items():
      # TODO: b/458797890 - Investigate if skipping buffer_idx == 0 is a
      # correct fix, or if it just covers up a deeper issue. This is only
      # required when statically quantizing models that have already been
      # quantized dynamically.
      if not tensors or buffer_idx == 0:
        continue
      # Check if any of the tensors needs to be duplicated.
      for tensor in tensors:
        if not self._check_buffer_sharing_for_tensor(tensor):
          tensor_names_to_duplicate.append(
              tfl_flatbuffer_utils.get_tensor_name(tensor)
          )
      # Check if the buffer needs to be duplicated.
      tensor_1 = tensors[0]
      tensor_name_1 = tfl_flatbuffer_utils.get_tensor_name(tensor_1)
      if tensor_name_1 in tensor_names_to_duplicate:
        buffers_to_duplicate.append(buffer_idx)
        continue
      for tensor_2 in tensors[1:]:
        tensor_name_2 = tfl_flatbuffer_utils.get_tensor_name(tensor_2)
        if (
            tensor_name_2 in tensor_names_to_duplicate
            or not self._check_buffer_sharing_for_self_compatible_tensors(
                tensor_1, tensor_2
            )
        ):
          buffers_to_duplicate.append(buffer_idx)
          break

    # Fix the buffer sharing issues.
    self._mark_tensors_requiring_buffer_duplication(buffers_to_duplicate)
    self._mark_tensors_requiring_tensor_duplication(tensor_names_to_duplicate)


def _are_tensor_consumer_params_compatible(
    params: qtyping.TensorTransformationParams,
) -> bool:
  """Check if all tensor's consumers have the same quantization parameters."""
  if params.consumers is None or len(params.consumers) < 2:
    return True
  consumer_1 = params.consumers[0]
  for consumer in params.consumers[1:]:
    if not _compatible_tensor_params(consumer, consumer_1):
      return False
  return True


def _are_self_compatible_tensors_compatible_to_each_other(
    params1: qtyping.TensorTransformationParams,
    params2: qtyping.TensorTransformationParams,
) -> bool:
  """Check if two self compatible tensors are compatible to each other.

  Self compatible means that all tensor's consumers have the same quantization
  parameters.

  Args:
    params1: The first tensor transformation params.
    params2: The second tensor transformation params.

  Returns:
    Whether the two tensors are compatible to each other.
  """
  # Check the producer.
  if params1.producer is None or params2.producer is None:
    if params1.producer != params2.producer:
      return False
  elif not _compatible_tensor_params(params1.producer, params2.producer):
    return False

  # Check the consumers.
  if params1.consumers is None or params2.consumers is None:
    if params1.consumers != params2.consumers:
      return False
  else:
    # Since all consumer params within each tensor are the same, it's enough to
    # check only the first consumers.
    if not _compatible_tensor_params(
        params1.consumers[0], params2.consumers[0]
    ):
      return False
  return True


def _same_tensor_params_except_id(
    params1: qtyping.OpToTensorParams,
    params2: qtyping.OpToTensorParams,
) -> bool:
  """Check if two op to tensor params are the same except for subgraph_op_id."""
  return params1.transformations == params2.transformations and (
      params1.parameters == params2.parameters
      or params1.parameters is None
      and params2.parameters is None
  )


def _compatible_tensor_params(
    params1: qtyping.OpToTensorParams,
    params2: qtyping.OpToTensorParams,
) -> bool:
  """Check if two op to tensor params are compatible."""
  float_source_transformations = [
      _QuantTrans.ADD_QUANTIZE,
      _QuantTrans.NO_QUANTIZE,
      _QuantTrans.INSERT_HADAMARD_ROTATION,
      _QuantTrans.INSERT_DECOMPOSED_HADAMARD_ROTATION,
      _QuantTrans.INSERT_MULTIPLY,
  ]
  quantized_source_transformations = [
      _QuantTrans.QUANTIZE_TENSOR,
      _QuantTrans.ADD_DEQUANTIZE,
  ]
  if _same_tensor_params_except_id(params1, params2):
    return True
  # We only need to check the first transformation because transformations are
  # applied in order, and as long as the one that's immediately after the tensor
  # is the same, it's compatible.
  if (
      params1.transformations[0] in float_source_transformations
      and params2.transformations[0] in float_source_transformations
  ):
    return True
  if (
      params1.transformations[0] in quantized_source_transformations
      and params2.transformations[0] in quantized_source_transformations
      and params1.parameters == params2.parameters
  ):
    return True
  return False


def _is_constant_tensor(tensor: Any, buffers: Sequence[Any]) -> bool:
  """Check if the tensor is a constant tensor."""
  return buffers[tensor.buffer].data is not None
