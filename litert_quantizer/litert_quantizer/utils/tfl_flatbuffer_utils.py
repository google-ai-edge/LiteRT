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

"""flatbuffer utils for the Quantizer."""

import collections
import logging
import pathlib

import immutabledict
import numpy as np

from ai_edge_litert.tools import flatbuffer_utils
from ai_edge_litert.tools import mmap_utils
from litert_quantizer import qtyping


_TFLOpName = qtyping.TFLOperationName

Path = str | pathlib.Path

TFL_OP_NAME_TO_CODE = immutabledict.immutabledict({
    _TFLOpName.FULLY_CONNECTED: qtyping.BuiltinOperator.FULLY_CONNECTED,
    _TFLOpName.BATCH_MATMUL: qtyping.BuiltinOperator.BATCH_MATMUL,
    _TFLOpName.CONV_2D: qtyping.BuiltinOperator.CONV_2D,
    _TFLOpName.DEPTHWISE_CONV_2D: qtyping.BuiltinOperator.DEPTHWISE_CONV_2D,
    _TFLOpName.CONV_2D_TRANSPOSE: qtyping.BuiltinOperator.TRANSPOSE_CONV,
    _TFLOpName.EMBEDDING_LOOKUP: qtyping.BuiltinOperator.EMBEDDING_LOOKUP,
    _TFLOpName.SOFTMAX: qtyping.BuiltinOperator.SOFTMAX,
    _TFLOpName.AVERAGE_POOL_2D: qtyping.BuiltinOperator.AVERAGE_POOL_2D,
    _TFLOpName.RESHAPE: qtyping.BuiltinOperator.RESHAPE,
    _TFLOpName.TANH: qtyping.BuiltinOperator.TANH,
    _TFLOpName.TRANSPOSE: qtyping.BuiltinOperator.TRANSPOSE,
    _TFLOpName.GELU: qtyping.BuiltinOperator.GELU,
    _TFLOpName.ADD: qtyping.BuiltinOperator.ADD,
    _TFLOpName.SUB: qtyping.BuiltinOperator.SUB,
    _TFLOpName.MUL: qtyping.BuiltinOperator.MUL,
    _TFLOpName.MEAN: qtyping.BuiltinOperator.MEAN,
    _TFLOpName.RSQRT: qtyping.BuiltinOperator.RSQRT,
    _TFLOpName.CONCATENATION: qtyping.BuiltinOperator.CONCATENATION,
    _TFLOpName.STRIDED_SLICE: qtyping.BuiltinOperator.STRIDED_SLICE,
    _TFLOpName.SPLIT: qtyping.BuiltinOperator.SPLIT,
    _TFLOpName.LOGISTIC: qtyping.BuiltinOperator.LOGISTIC,
    _TFLOpName.SLICE: qtyping.BuiltinOperator.SLICE,
    _TFLOpName.SUM: qtyping.BuiltinOperator.SUM,
    _TFLOpName.SELECT: qtyping.BuiltinOperator.SELECT,
    _TFLOpName.SELECT_V2: qtyping.BuiltinOperator.SELECT_V2,
    _TFLOpName.STABLEHLO_COMPOSITE: qtyping.BuiltinOperator.STABLEHLO_COMPOSITE,
    _TFLOpName.DYNAMIC_UPDATE_SLICE: (
        qtyping.BuiltinOperator.DYNAMIC_UPDATE_SLICE
    ),
    _TFLOpName.PAD: qtyping.BuiltinOperator.PAD,
    _TFLOpName.SQUARED_DIFFERENCE: qtyping.BuiltinOperator.SQUARED_DIFFERENCE,
    _TFLOpName.MAX_POOL_2D: qtyping.BuiltinOperator.MAX_POOL_2D,
    _TFLOpName.RESIZE_BILINEAR: qtyping.BuiltinOperator.RESIZE_BILINEAR,
    _TFLOpName.RESIZE_NEAREST_NEIGHBOR: (
        qtyping.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR
    ),
    _TFLOpName.GATHER_ND: qtyping.BuiltinOperator.GATHER_ND,
    _TFLOpName.PACK: qtyping.BuiltinOperator.PACK,
    _TFLOpName.UNPACK: qtyping.BuiltinOperator.UNPACK,
    _TFLOpName.DIV: qtyping.BuiltinOperator.DIV,
    _TFLOpName.BROADCAST_TO: qtyping.BuiltinOperator.BROADCAST_TO,
    _TFLOpName.SQRT: qtyping.BuiltinOperator.SQRT,
    _TFLOpName.GATHER: qtyping.BuiltinOperator.GATHER,
    _TFLOpName.HARD_SWISH: qtyping.BuiltinOperator.HARD_SWISH,
    _TFLOpName.MAXIMUM: qtyping.BuiltinOperator.MAXIMUM,
    _TFLOpName.PADV2: qtyping.BuiltinOperator.PADV2,
    _TFLOpName.REDUCE_MIN: qtyping.BuiltinOperator.REDUCE_MIN,
    _TFLOpName.EQUAL: qtyping.BuiltinOperator.EQUAL,
    _TFLOpName.NOT_EQUAL: qtyping.BuiltinOperator.NOT_EQUAL,
    _TFLOpName.MIRROR_PAD: qtyping.BuiltinOperator.MIRROR_PAD,
    _TFLOpName.SPACE_TO_DEPTH: qtyping.BuiltinOperator.SPACE_TO_DEPTH,
    _TFLOpName.RELU: qtyping.BuiltinOperator.RELU,
})

TFL_OP_CODE_TO_NAME = immutabledict.immutabledict(
    dict((reversed(item) for item in TFL_OP_NAME_TO_CODE.items()))  # pyrefly: ignore[no-matching-overload]
)

# Quantized dimension for per-channel quantization.
# See https://www.tensorflow.org/lite/performance/quantization_spec.
TFL_OP_TO_WEIGHT_QUANTIZED_DIM = immutabledict.immutabledict({
    _TFLOpName.FULLY_CONNECTED: 0,
    _TFLOpName.DEPTHWISE_CONV_2D: 3,
    _TFLOpName.CONV_2D: 0,
    _TFLOpName.EMBEDDING_LOOKUP: 0,
    _TFLOpName.CONV_2D_TRANSPOSE: 0,
})

TFL_OP_TO_BLOCKWISE_WEIGHT_QUANTIZED_DIM = immutabledict.immutabledict({
    _TFLOpName.FULLY_CONNECTED: 1,
    _TFLOpName.EMBEDDING_LOOKUP: 1,
})

NUM_TFL_DATATYPES = 18
TENSOR_TYPE_TO_CODE = immutabledict.immutabledict(qtyping.TensorType.__dict__)
TENSOR_CODE_TO_TYPE = immutabledict.immutabledict(
    {v: k for k, v in qtyping.TensorType.__dict__.items()}
)

# Expose functions in litert.python.tools.flatbuffer_utils
write_model = flatbuffer_utils.write_model


def read_model(
    tflite_model: Path | qtyping.BufferType,
) -> qtyping.ModelT:
  """Read and convert the TFLite model into a flatbuffer object.

  Args:
    tflite_model: TFLite model path or bytearray.

  Raises:
    ValueError: Unsupported tflite_model type.

  Returns:
    flatbuffer_model: the flatbuffer_model.
  """
  if isinstance(tflite_model, Path):
    return flatbuffer_utils.read_model(tflite_model)
  elif isinstance(tflite_model, (bytes, bytearray, memoryview)):
    return flatbuffer_utils.read_model_from_bytearray(tflite_model)
  else:
    raise ValueError(
        "Unsupported tflite_model type: %s" % type(tflite_model).__name__
    )


def get_model_content(tflite_path: Path) -> memoryview:
  """Get the model content (read-only bytes) from the path.

  Args:
    tflite_path: Path to the .tflite.

  Returns:
    The model bytes.
  """
  return mmap_utils.get_file_contents(tflite_path)


def get_model_buffer(tflite_path: Path) -> bytearray:
  """Get a mutable model buffer from the path.

  Args:
    tflite_path: path to the .tflite.

  Returns:
    model_buffer: the model buffer.
  """
  return bytearray(mmap_utils.get_file_contents(tflite_path))


def parse_op_tensors(
    op: qtyping.OperatorT, subgraph_tensors: list[qtyping.TensorT]
) -> list[qtyping.TensorT]:
  """Parse the op tensors.

  Args:
    op: the op that need to be parsed.
    subgraph_tensors: list of tensors in the subgraph.

  Returns:
    tensors: list of tensors that are associated with the op.
  """

  return [
      subgraph_tensors[tensor_idx]
      for tensor_idx in list(op.outputs) + list(op.inputs)
      if tensor_idx != -1
  ]


def parse_fc_bmm_conv_tensors(
    op: qtyping.OperatorT,
    subgraph_tensors: list[qtyping.TensorT],
    input_index: int = 0,
    weight_index: int = 1,
    bias_index: int = 2,
    output_index: int = 0,
) -> tuple[
    qtyping.TensorT, qtyping.TensorT, qtyping.TensorT | None, qtyping.TensorT
]:
  """Parse tensors in FullyConnected, BatchMatmul, and Convolutions.

  Args:
    op: the TFLite op, must be fully_connected, batch_matmul, or convolution.
    subgraph_tensors: tensors in the subgraph.
    input_index: index for the input tensor.
    weight_index: index for the weight tensor.
    bias_index: index for the bias tensor.
    output_index: index for the output tensor.

  Returns:
    input_tensor, weight_tensor, bias_tensor, output_tensor
  """

  input_tensor = subgraph_tensors[op.inputs[input_index]]
  weight_tensor = subgraph_tensors[op.inputs[weight_index]]
  bias_tensor = None
  if bias_index < len(op.inputs) and op.inputs[bias_index] != -1:
    bias_tensor = subgraph_tensors[op.inputs[bias_index]]
  output_tensor = subgraph_tensors[op.outputs[output_index]]
  return input_tensor, weight_tensor, bias_tensor, output_tensor


def buffer_to_tensors(
    flatbuffer_model: qtyping.ModelT,
) -> dict[int, list[qtyping.TensorT]]:
  """Returns a map from buffer id to tensors that use it."""
  buffer_to_tensor_map = collections.defaultdict(list)
  for subgraph in flatbuffer_model.subgraphs:
    for tensor in subgraph.tensors:
      buffer_to_tensor_map[tensor.buffer].append(tensor)
  return buffer_to_tensor_map


def get_tensor_name(tensor: qtyping.TensorT) -> str:
  """Get the tensor name for a fb tensor.

  Args:
    tensor: tensor in flatbuffer.

  Returns:
    tensor_name: name of the buffer
  """
  return tensor.name.decode("utf-8")


def get_tensor_data(
    tensor: qtyping.TensorT, buffers: list[qtyping.BufferT]
) -> np.ndarray | None:
  """Get the tensor data.

  Args:
    tensor: tensor in flatbuffer.
    buffers: list of buffers

  Returns:
    tensor_data: data inside the tensor
  """
  tensor_buffer = buffers[tensor.buffer]
  buffer_data = tensor_buffer.data
  if buffer_data is None:
    return None
  data = np.frombuffer(
      buffer_data, dtype=TENSOR_CODE_TO_TYPE[tensor.type].lower()
  )
  if tensor.shape is not None:
    data = np.reshape(data, tensor.shape)
  return data


def has_same_quantization(
    tensor1: qtyping.TensorT, tensor2: qtyping.TensorT
) -> bool:
  """Check if two tensors have the same quantization.

  Args:
    tensor1: tensor in flatbuffer.
    tensor2: tensor in flatbuffer.

  Returns:
    True if two tensors have the same quantization.
  """

  def to_tuple(val):
    if val is None:
      val = []
    return tuple(val)

  same_type = tensor1.type == tensor2.type

  # Return True if both tensors are not quantized.
  if tensor1.quantization is None and tensor2.quantization is None:
    return True
  if tensor1.quantization.scale is None and tensor2.quantization.scale is None:
    return True

  same_scale = to_tuple(tensor1.quantization.scale) == to_tuple(
      tensor2.quantization.scale
  )
  same_zero_point = to_tuple(tensor1.quantization.zeroPoint) == to_tuple(
      tensor2.quantization.zeroPoint
  )
  same_quantized_dimension = (
      tensor1.quantization.quantizedDimension
      == tensor2.quantization.quantizedDimension
  )
  return (
      same_type and same_scale and same_zero_point and same_quantized_dimension
  )


def is_float_model(flatbuffer_model: qtyping.ModelT) -> bool:
  """Checks that the model is float and not already quantized."""
  for subgraph in flatbuffer_model.subgraphs:
    for tensor in subgraph.tensors:
      if tensor.quantization is None:
        continue
      if tensor.quantization.scale is not None:
        return False
  return True


def get_subgraph_input_output_operators(
    subgraph: qtyping.SubGraphT,
) -> list[qtyping.IOOperator]:
  """Get the input/output operators for the subgraph.

  Args:
    subgraph: The subgraph object.

  Returns:
    Input and output operators for the subgraph.
  """
  input_op = qtyping.IOOperator(
      inputs=[],
      outputs=subgraph.inputs,
      op_key=qtyping.TFLOperationName.INPUT,
  )
  output_op = qtyping.IOOperator(
      inputs=subgraph.outputs,
      outputs=[],
      op_key=qtyping.TFLOperationName.OUTPUT,
  )
  return [input_op, output_op]


def get_op_side_effect_subgraphs(
    op: qtyping.Operator | qtyping.OperatorT,
) -> list[int]:
  """Get indices of any subgraphs invoked as a side effect of the operator.

  Args:
    op: The operator object.

  Returns:
    A list of subgraph indices invoked by the operator. Empty if the operator
    does not invoke any subgraphs.
  """
  if opts := flatbuffer_utils.get_options_as(
      op, qtyping.StableHLOCompositeOptionsT
  ):
    return [opts.decompositionSubgraphIndex]
  # Can add other nested ops here (control flow ops, etc).
  return []


def get_op_name_by_index(
    flatbuffer_model: qtyping.ModelT, subgraph_id: int, op_index: int
) -> str:
  """Get the op name from the flatbuffer model."""
  op = flatbuffer_model.subgraphs[subgraph_id].operators[op_index]
  builtin_code = flatbuffer_model.operatorCodes[op.opcodeIndex].builtinCode
  return TFL_OP_CODE_TO_NAME[builtin_code]


def get_op_scope(
    op: qtyping.OperatorT,
    subgraph_tensors: list[qtyping.TensorT],
    max_length: int = 10000,
) -> str:
  """Get the op scope.

  Op scope is defined by the output tensor names (following ModelExplorer). If
  no output tensors are present, the input tensor names are used instead.

  Args:
    op: The op that needs to be parsed.
    subgraph_tensors: Tensors in the subgraph.
    max_length: The maximum length of the scope string. If the scope string is
      longer than this length, it will be truncated to this length to avoid
      overwhelming regex matching engine.

  Returns:
    Scope for the op.
  """

  def _get_valid_tensor_names(tensor_indices: list[int]) -> list[str]:
    """Gets names of tensors for valid indices."""
    names = []
    for idx in tensor_indices:
      if idx != -1:
        names.append(get_tensor_name(subgraph_tensors[idx]))
    return [name for name in names if name]

  # Op scope is determined by output tensors.
  tensor_names = _get_valid_tensor_names(op.outputs)
  # If no output tensors, use input tensors.
  if not tensor_names:
    tensor_names = _get_valid_tensor_names(op.inputs)

  scope = ";".join(tensor_names)
  # Add a trailing semicolon to help distinguish scopes with the same prefix.
  if tensor_names:
    scope += ";"
  if len(scope) > max_length:
    logging.warning(
        "Op scope is too long, truncating to %d characters. Truncated scope:"
        " %s",
        max_length,
        scope,
    )
  return scope[:max_length]
