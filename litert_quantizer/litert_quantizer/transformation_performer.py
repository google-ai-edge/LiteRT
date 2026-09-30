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

"""Python manager for transformations to be applied to TFlite models."""

from collections.abc import Sequence

import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.transformations import dequant_insert
from litert_quantizer.transformations import duplicate_buffer
from litert_quantizer.transformations import duplicate_tensor
from litert_quantizer.transformations import insert_decomposed_hadamard_rotation
from litert_quantizer.transformations import insert_hadamard_rotation
from litert_quantizer.transformations import insert_multiply
from litert_quantizer.transformations import quant_insert
from litert_quantizer.transformations import quantize_tensor
from litert_quantizer.transformations import transformation_utils
from litert_quantizer.utils import progress_utils


class TransformationPerformer:
  """Wrapper class for transformations.

  all transformations supported by the LiteRT Quantizer should be registered in
  the Transformation Performer.

  transformation to be appied to a tensor in the Instruction Generator is
  specified
  as a key. Given this key, the transformation performer will return a function
  that will apply the corresponding transformation on the graph

  A transformation is defined as a Callable that takes the following parameters:
  tensor_id: A tensor id that represents the tensor to be applied
  operatorCodes: list of OperatorCodesT from the source TFlite ModelT
  buffers: list of BufferT from the source TFLite ModelT
  subgraph: the specific subgraph where the transformation should be applied
  producer: the op index for the producer of the tensor
  consumers: a list of op index representing consumers to apply the change on
  quant_param: the quantization parameters in qtyping.UniformQuantParams
  And returns a qtyping.TransformationInfo which contains the index where the
  ops are added and how many ops are added

  Additionally, op additions must be consecutive

  this class is expected to be created by the Model Modifier and nothing else

  Model modifier would pass in a dict of transformations to be applied, this
  class will apply the transformations in a pre-determined static order
  """

  def __init__(self):
    """Initializes the TransformationPerformer."""
    self._transformation_registration = {
        qtyping.QuantTransformation.ADD_DEQUANTIZE: (
            dequant_insert.insert_dequant
        ),
        qtyping.QuantTransformation.QUANTIZE_TENSOR: (
            quantize_tensor.quantize_tensor
        ),
        qtyping.QuantTransformation.EMULATED_SUBCHANNEL: (
            transformation_utils.raise_deprecated_error
        ),
        qtyping.QuantTransformation.ADD_QUANTIZE: quant_insert.insert_quant,
        qtyping.QuantTransformation.DUPLICATE_BUFFER: (
            duplicate_buffer.duplicate_buffer
        ),
        qtyping.QuantTransformation.DUPLICATE_TENSOR: (
            duplicate_tensor.duplicate_tensor
        ),
        qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION: (
            insert_hadamard_rotation.insert_hadamard_rotation
        ),
        qtyping.QuantTransformation.INSERT_DECOMPOSED_HADAMARD_ROTATION: (
            insert_decomposed_hadamard_rotation.insert_decomposed_hadamard_rotation
        ),
        qtyping.QuantTransformation.INSERT_MULTIPLY: (
            insert_multiply.insert_multiply
        ),
    }
    # transformations are seprated in two categories:
    # op_insertion_transformations are transformations that only insert ops
    # into the graph, whereas op_replacement_transformations will replace one op
    # with a pattern
    self._op_insertion_transformations = set([
        qtyping.QuantTransformation.ADD_DEQUANTIZE,
        qtyping.QuantTransformation.QUANTIZE_TENSOR,
        qtyping.QuantTransformation.ADD_QUANTIZE,
        qtyping.QuantTransformation.DUPLICATE_BUFFER,
        qtyping.QuantTransformation.DUPLICATE_TENSOR,
        qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION,
        qtyping.QuantTransformation.INSERT_DECOMPOSED_HADAMARD_ROTATION,
        qtyping.QuantTransformation.INSERT_MULTIPLY,
    ])
    self._op_replacement_transformations = set(
        [qtyping.QuantTransformation.EMULATED_SUBCHANNEL]
    )
    self._original_op_id_map = []
    self._added_op_id_map = []
    self._buffer_origin = {}

  def _create_op_id_map(self, tflite_model: qtyping.ModelT):
    """init the original op_id to modified op_id map.

    At the beginning the graph has not been updated, so op_id maps to it's
    current id.

    Args:
      tflite_model: the model we're create op_id mapping

    Returns:
      None, modifies self._original_op_id_map inplace
    """
    for subgraph in tflite_model.subgraphs:
      self._original_op_id_map.append(list(range(len(subgraph.operators))))
      self._added_op_id_map.append([])

  def _update_op_id_map(
      self, subgraph_id: int, original_op_id: int, num_ops_added: int
  ):
    """Update the mapping between the original op id and modified op ids.

    Args:
      subgraph_id: the index of subgraph that we're interested in
      original_op_id: the original id for which the first op is added
      num_ops_added: the number of ops added starting from the op id

    Returns:
      None, modify self._original_op_id_map
    """
    np_op_id_map = np.array(self._original_op_id_map[subgraph_id])
    np_op_id_map[original_op_id:] += num_ops_added
    self._original_op_id_map[subgraph_id] = np_op_id_map.tolist()

  def _update_instructions(
      self,
      prev_transformation_index: int,
      transformations: list[qtyping.TransformationInst],
      subgraph_id: int,
      trans_info: qtyping.TransformationInfo,
  ) -> None:
    """Update the instructions in-place after the graph is modified.

    After an op is inserted or a tensor is duplicated, the topology is changed
    and this may impact the following transformation to be applied. So we need
    to update instructions that have yet to be applied.

    Args:
      prev_transformation_index: The index of the last applied transformation.
      transformations: The list of transformations we're applying.
      subgraph_id: The subgraph where the provided instructions belong to.
      trans_info: Transformation info returned by a transformation.
    """
    prev_transformation = transformations[prev_transformation_index]
    is_prev_not_duplicate_tensor = (
        prev_transformation.transformation
        != qtyping.QuantTransformation.DUPLICATE_TENSOR
    )
    was_op_added = trans_info.num_ops_added > 0
    if not was_op_added and is_prev_not_duplicate_tensor:
      return

    if was_op_added:
      self._added_op_id_map[subgraph_id].append(
          trans_info.op_id + trans_info.num_ops_added - 1
      )

    for transformations_index in range(
        prev_transformation_index + 1, len(transformations)
    ):
      transformation = transformations[transformations_index]
      for consumer_index in transformation.consumers:
        # If the consumer needs to use newly added ops, then the new added op
        # index needs to be outside of the range of the orignal op ids.
        if consumer_index in prev_transformation.consumers:
          if was_op_added:
            transformation.producer = (
                len(self._original_op_id_map[subgraph_id])
                + len(self._added_op_id_map[subgraph_id])
                - 1
            )
          transformation.tensor_id = trans_info.output_tensor_id

  def _get_updated_producer_id(
      self, original_producer_id: int, subgraph_id: int
  ) -> int:
    """Update the producer of a transformation instruction."""
    if original_producer_id is None or original_producer_id < 0:
      producer = -1
    elif original_producer_id < len(self._original_op_id_map[subgraph_id]):
      producer = self._original_op_id_map[subgraph_id][original_producer_id]
    else:
      # If the producer id is not in the original op map, it's an added op,
      # go the added op map to find the producer.
      producer = self._added_op_id_map[subgraph_id][
          original_producer_id - len(self._original_op_id_map[subgraph_id])
      ]
    return producer

  def _get_updated_consumer_ids(
      self,
      original_consumer_ids: list[int],
      subgraph_id: int,
  ) -> list[int]:
    """Update the consumers of a transformation instruction."""
    consumers = []
    for original_op_id in original_consumer_ids:
      new_consumer_id = (
          -1
          if original_op_id == -1
          else self._original_op_id_map[subgraph_id][original_op_id]
      )
      consumers.append(new_consumer_id)
    return consumers

  def _apply_single_transformation(
      self,
      transformation_inst: qtyping.TensorTransformationInsts,
      transformation_index: int,
      tflite_model: qtyping.ModelT,
  ):
    """Apply a single transformation.

    Args:
      transformation_inst: a TensorTransformationInsts type that contains all
        transformations on a tensor
      transformation_index: the index of the transformation to be applied
      tflite_model: source tflite model to be updated

    Returns:
      None, update the transformation_inst & tflite_model in place
    """
    instruction = transformation_inst.instructions[transformation_index]  # pyrefly: ignore[unsupported-operation]
    producer = self._get_updated_producer_id(
        instruction.producer, transformation_inst.subgraph_id  # pyrefly: ignore[bad-argument-type]
    )
    consumers = self._get_updated_consumer_ids(
        instruction.consumers, transformation_inst.subgraph_id
    )
    trans_info = self._transformation_registration[instruction.transformation](
        transformation_utils.TransformationInput(
            instruction.tensor_id,
            tflite_model,
            tflite_model.subgraphs[transformation_inst.subgraph_id],
            producer,
            consumers,
            instruction.parameters,  # pyrefly: ignore[bad-argument-type]
            self._buffer_origin,
        )
    )
    self._update_instructions(
        transformation_index,
        transformation_inst.instructions,  # pyrefly: ignore[bad-argument-type]
        transformation_inst.subgraph_id,
        trans_info,
    )
    self._update_op_id_map(
        transformation_inst.subgraph_id,
        # The added op must be right before the most immediate consumer, unless
        # the consumer is the graph output (id=-1), then use the producer's
        # index instead.
        min(instruction.consumers)
        if min(instruction.consumers) >= 0
        else instruction.producer + 1,  # pyrefly: ignore[unsupported-operation]
        trans_info.num_ops_added,
    )

  def _apply_transformations(
      self,
      transformation_inst: qtyping.TensorTransformationInsts,
      tflite_model: qtyping.ModelT,
  ):
    """Apply all transformations for a tensor.

    transformations are separated in two types and applied separately in two
    different passes

    Args:
      transformation_inst: a TensorTransformationInsts type that contains all
        transformation on a tensor
      tflite_model: source tflite model to be updated

    Returns:
      None, update the transformation_inst & tflite_model in place
    """
    # pass 1: apply all the op insertion transformation, because op replacement
    # may remove consumer or producer of some tensors
    for index, instruction in enumerate(transformation_inst.instructions):  # pyrefly: ignore[bad-argument-type, not-iterable]
      if instruction.transformation in self._op_insertion_transformations:
        self._apply_single_transformation(
            transformation_inst, index, tflite_model
        )
    # pass 2: apply all the op replacement transformation
    for index, instruction in enumerate(transformation_inst.instructions):  # pyrefly: ignore[bad-argument-type, not-iterable]
      if instruction.transformation in self._op_replacement_transformations:
        self._apply_single_transformation(
            transformation_inst, index, tflite_model
        )

  def transform_graph(
      self,
      transformation_instructions: dict[str, qtyping.TensorTransformationInsts],
      tflite_model: qtyping.ModelT,
      tensor_processing_order: Sequence[str] | None = None,
      enable_progress_bar: bool | None = None,
  ) -> None:
    """Apply all transformations to the given tflite_model in place.

    Args:
      transformation_instructions: Mapping from tensor name to its
        transformation instructions, produced by
        transformation_instruction_generator.
      tflite_model: The tflite model to apply quantization to.
      tensor_processing_order: The order of tensors to process. If not provided,
        the order will be inferred from `transformation_instructions`.
      enable_progress_bar: Whether to enable the progress bar. By default, it is
        disabled for smaller models and enabled for larger models.
    """
    self._original_op_id_map = []
    self._added_op_id_map = []
    self._create_op_id_map(tflite_model)
    self._buffer_origin = dict()
    if tensor_processing_order is None:
      tensor_processing_order = transformation_instructions.keys()  # pyrefly: ignore[bad-assignment]
    with progress_utils.ProgressBar(
        total_steps=len(tensor_processing_order),  # pyrefly: ignore[bad-argument-type]
        description="Applying Transformations to tensors:",
        enable=enable_progress_bar,
    ) as progress_bar:
      for tensor_name in tensor_processing_order:  # pyrefly: ignore[not-iterable]
        progress_bar.update_single_step()
        self._apply_transformations(
            transformation_instructions[tensor_name], tflite_model
        )
    # Reseet the buffer origin mapping so that we don't keep the quant
    # parameters in memory longer than necessary.
    self._buffer_origin = {}
