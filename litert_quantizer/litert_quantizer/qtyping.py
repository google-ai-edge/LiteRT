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

"""Type hinting support for LiteRT Quantizer."""

import collections
from collections.abc import MutableMapping
import copy
import dataclasses
import enum
from typing import Any, Callable, Mapping, Optional, TypeAlias, Union

import immutabledict
import numpy as np

from ai_edge_litert.tools import flatbuffer_utils

QSV: TypeAlias = MutableMapping[str, Any]

# A collection of quantization configuration.
# Key: scope regex.
# Value: list of OpQuantizationRecipe in dictionary format.
ModelQuantizationRecipe = list[dict[str, Any]]

# Types imported from `schema_py_generated`.
ActivationFunctionType = flatbuffer_utils.ActivationFunctionType
BlockwiseQuantizationT = flatbuffer_utils.BlockwiseQuantizationT
Buffer = flatbuffer_utils.Buffer
BufferT: TypeAlias = (
    flatbuffer_utils.BufferT
)  # Mutable flatbuffer buffer representation
BuiltinOperator = flatbuffer_utils.BuiltinOperator
BuiltinOptions = flatbuffer_utils.BuiltinOptions
BuiltinOptions2 = flatbuffer_utils.BuiltinOptions2
FullyConnectedOptionsT = flatbuffer_utils.FullyConnectedOptionsT
MulOptions = flatbuffer_utils.schema_fb.MulOptions
MulOptionsT = flatbuffer_utils.schema_fb.MulOptionsT
Model = flatbuffer_utils.Model
ModelT: TypeAlias = (
    flatbuffer_utils.ModelT
)  # Mutable flatbuffer model representation containing subgraphs and buffers
Operator = flatbuffer_utils.Operator
OperatorCode = flatbuffer_utils.OperatorCode
OperatorCodeT = flatbuffer_utils.OperatorCodeT
OperatorT: TypeAlias = (
    flatbuffer_utils.OperatorT
)  # Mutable flatbuffer operator representation
QuantizationDetails = flatbuffer_utils.QuantizationDetails
QuantizationParametersT: TypeAlias = (
    flatbuffer_utils.QuantizationParametersT
)  # Mutable quantization parameters (scale, zero point)
StableHLOCompositeOptions = flatbuffer_utils.StableHLOCompositeOptions
StableHLOCompositeOptionsT = flatbuffer_utils.StableHLOCompositeOptionsT
SubGraph = flatbuffer_utils.SubGraph
SubGraphT: TypeAlias = (
    flatbuffer_utils.SubGraphT
)  # Mutable flatbuffer subgraph containing tensors and operators
Tensor = flatbuffer_utils.Tensor
TensorT: TypeAlias = (
    flatbuffer_utils.TensorT
)  # Mutable flatbuffer tensor representation
TensorType = flatbuffer_utils.TensorType

# Local convenience types.
Path = flatbuffer_utils.Path
BufferType = flatbuffer_utils.BufferType
Endiness = flatbuffer_utils.Endiness


class TFLOperationName(str, enum.Enum):
  """TF Lite operation names."""

  ALL_SUPPORTED = '*'
  INPUT = 'INPUT'
  OUTPUT = 'OUTPUT'
  FULLY_CONNECTED = 'FULLY_CONNECTED'
  BATCH_MATMUL = 'BATCH_MATMUL'
  DEPTHWISE_CONV_2D = 'DEPTHWISE_CONV_2D'
  CONV_2D = 'CONV_2D'
  CONV_2D_TRANSPOSE = 'CONV_2D_TRANSPOSE'
  AVERAGE_POOL_2D = 'AVERAGE_POOL_2D'
  RESHAPE = 'RESHAPE'
  CUSTOM_OP = 'CUSTOM_OP'
  EMBEDDING_LOOKUP = 'EMBEDDING_LOOKUP'
  SOFTMAX = 'SOFTMAX'
  TANH = 'TANH'
  TRANSPOSE = 'TRANSPOSE'
  GELU = 'GELU'
  ADD = 'ADD'
  SUB = 'SUB'
  MUL = 'MUL'
  MEAN = 'MEAN'
  RSQRT = 'RSQRT'
  CONCATENATION = 'CONCATENATION'
  STRIDED_SLICE = 'STRIDED_SLICE'
  SPLIT = 'SPLIT'
  LOGISTIC = 'LOGISTIC'
  SLICE = 'SLICE'
  SUM = 'SUM'
  SELECT = 'SELECT'
  SELECT_V2 = 'SELECT_V2'
  DYNAMIC_UPDATE_SLICE = 'DYNAMIC_UPDATE_SLICE'
  STABLEHLO_COMPOSITE = 'STABLEHLO_COMPOSITE'
  PAD = 'PAD'
  SQUARED_DIFFERENCE = 'SQUARED_DIFFERENCE'
  MAX_POOL_2D = 'MAX_POOL_2D'
  RESIZE_BILINEAR = 'RESIZE_BILINEAR'
  RESIZE_NEAREST_NEIGHBOR = 'RESIZE_NEAREST_NEIGHBOR'
  GATHER_ND = 'GATHER_ND'
  PACK = 'PACK'
  UNPACK = 'UNPACK'
  DIV = 'DIV'
  BROADCAST_TO = 'BROADCAST_TO'
  SQRT = 'SQRT'
  GATHER = 'GATHER'
  HARD_SWISH = 'HARD_SWISH'
  MAXIMUM = 'MAXIMUM'
  PADV2 = 'PADV2'
  REDUCE_MIN = 'REDUCE_MIN'
  EQUAL = 'EQUAL'
  NOT_EQUAL = 'NOT_EQUAL'
  MIRROR_PAD = 'MIRROR_PAD'
  SPACE_TO_DEPTH = 'SPACE_TO_DEPTH'
  RELU = 'RELU'


class QuantizeMode(enum.Enum):
  """The mode of quantization, determining which stage is executed."""

  # Gathering information from calibration data (e.g., min/max).
  CALIBRATE = 2
  # Executing the quantization process using parameters from calibration data.
  MATERIALIZE = 3


class OpExecutionMode(str, enum.Enum):
  """How to execute the op."""

  WEIGHT_ONLY = 'WEIGHT_ONLY'  # Weight-only quantization.
  DRQ = 'DRQ'  # Dynamic range quantization.
  SRQ = 'SRQ'  # Static range quantization.


class ComputePrecision(str, enum.Enum):
  """The precision of the compute operation."""

  INTEGER = 'INTEGER'
  FLOAT = 'FLOAT'


class TensorDataType(str, enum.Enum):
  INT = 'INT'
  UINT = 'UINT'
  FLOAT = 'FLOAT'


class QuantGranularity(str, enum.Enum):
  TENSORWISE = 'TENSORWISE'
  CHANNELWISE = 'CHANNELWISE'
  # Blockwise quantization with various block sizes.
  BLOCKWISE_32 = 'BLOCKWISE_32'
  BLOCKWISE_64 = 'BLOCKWISE_64'
  BLOCKWISE_128 = 'BLOCKWISE_128'
  BLOCKWISE_256 = 'BLOCKWISE_256'


class QuantTransformation(enum.Enum):
  """Operations associated with quantization for a tensor."""

  # Do nothing: float_tensor -> float_tensor.
  NO_QUANTIZE = 0
  # Add a quantize op: float_tensor -> Quantize Op -> quantized_tensor.
  ADD_QUANTIZE = 1
  # Add a dequantize op: quantized_tensor -> Dequantize Op -> float_tensor.
  ADD_DEQUANTIZE = 2
  # Quantize the float tensor: float_tensor -> quantized_tensor.
  QUANTIZE_TENSOR = 3
  # (Deprecated) Create pattern for emulated subchannel quantization,
  # only support fully connected op.
  EMULATED_SUBCHANNEL = 4
  # Duplicate the buffer.
  DUPLICATE_BUFFER = 5
  # Duplicate the tensor.
  DUPLICATE_TENSOR = 6
  # Insert the aeq.hadamard_rotation op.
  INSERT_HADAMARD_ROTATION = 7
  # Insert decomposed Hadamard rotation ops. This expresses the Hadamard
  # rotation as matrix multiplication with Hadamard matrices.
  INSERT_DECOMPOSED_HADAMARD_ROTATION = 8
  # Insert elementwise multiply op for activation scaling.
  INSERT_MULTIPLY = 9


@dataclasses.dataclass(frozen=True)
class UniformQuantParams:
  """Parameters for uniform quantization.

  Attributes:
    num_bits: Number of bits to quantize to (e.g. 8 for int8).
    quantized_dimension: The dimension to quantize.
    scale: The scale of the quantization.
    zero_point: The zero point of the quantization.
    symmetric: Whether the quantization is symmetric (force zero_point to be 0).
    quantized_data: The quantized data.
    block_size: The block size for blockwise quantization, block_size=0 meaning
      no blockwise quantization.
    hadamard: The Hadamard rotation parameters, if set. Hadamard rotation
      mitigates quantization loss caused by outliers by rotating (or
      distributing) weights and activations into a more uniform distribution.
      This is particularly useful for low bits quantization. More technical
      details can be found in algorithms/uniform_quantize/hadamard_rotation.py.
    custom_algorithm_param: Custom algorithm-specific parameters (e.g. for
      experimental algorithms like OSCAR).
    signed: Whether the quantization is signed.
  """

  class HadamardRotationParams:
    """Parameters for the Hadamard rotation.

    Attributes:
      random_binary_vector: The random binary vector for the Hadamard rotation.
        TODO(b/415392354): Randomization is an experimental feature that's
        currently not implemented yet hence this is always 1. We will add
        support or remove in the future.
      hadamard_size: The size of the Hadamard matrix.
    """

    random_binary_vector: np.ndarray
    hadamard_size: int

    def __init__(self, random_binary_vector: np.ndarray, hadamard_size: int):
      self.random_binary_vector = random_binary_vector
      self.hadamard_size = hadamard_size

    def __eq__(self, other):
      if other.__class__ is not self.__class__:
        return NotImplemented
      return id(self) == id(other) or (
          np.array_equal(self.random_binary_vector, other.random_binary_vector)
          and self.hadamard_size == other.hadamard_size
      )

  num_bits: int
  quantized_dimension: Optional[int]
  scale: np.ndarray
  zero_point: np.ndarray
  symmetric: bool = True
  quantized_data: Optional[np.ndarray] = None
  block_size: int = 0
  hadamard: Optional[HadamardRotationParams] = None
  custom_algorithm_param: Optional[dict[str, Any]] = None
  signed: bool = True

  @classmethod
  def from_tfl_tensor_details(cls, tensor_detail) -> 'UniformQuantParams':
    """Creates UniformQuantParams from TFLite tensor details.

    Args:
      tensor_detail: The tensor details from TFLite.

    Returns:
      UniformQuantParams.
    """
    quant_params = tensor_detail['quantization_parameters']
    data_type = tensor_detail['dtype']
    if data_type not in (np.int8, np.uint8, np.int16, np.int32, np.int64):
      raise ValueError(
          f'Unsupported data type: {data_type}. Supported types are np.int8,'
          ' np.uint8, np.int16, np.int32, np.int64.'
      )
    num_bits = np.iinfo(data_type).bits
    signed = np.issubdtype(data_type, np.signedinteger)
    symmetric = signed and sum(abs(quant_params['zero_points'])) == 0
    return cls(
        quantized_dimension=quant_params['quantized_dimension'],
        num_bits=num_bits,
        scale=quant_params['scales'],
        zero_point=quant_params['zero_points'],
        symmetric=symmetric,
        block_size=quant_params['block_size'],
        signed=signed,
    )

  def __eq__(self, other):
    if other.__class__ is not self.__class__:
      return NotImplemented
    return id(self) == id(other) or (
        self.num_bits == other.num_bits
        and self.quantized_dimension == other.quantized_dimension
        and _compare_array_or_none(self.scale, other.scale)
        and _compare_array_or_none(self.zero_point, other.zero_point)
        and self.symmetric == other.symmetric
        and _compare_array_or_none(self.quantized_data, other.quantized_data)
        and self.block_size == other.block_size
        and self.hadamard == other.hadamard
        and _compare_custom_algorithm_param(
            self.custom_algorithm_param, other.custom_algorithm_param
        )
        and self.signed == other.signed
    )


@dataclasses.dataclass(frozen=True)
class NonLinearQuantParams:
  """Parameters for nonlinear quantization.

  Currently only used for fp16 quantization.

  Attributes:
    num_bits: Number of bits to quantize to (e.g. 16 for fp16).
    quantized_data: The quantized data.
    data_type: The data type of the tensor.
  """

  num_bits: int
  quantized_data: Optional[np.ndarray]
  data_type: TensorDataType = TensorDataType.FLOAT

  def __eq__(self, other):
    if other.__class__ is not self.__class__:
      return NotImplemented
    return id(self) == id(other) or (
        self.num_bits == other.num_bits
        and self.data_type == other.data_type
        and _compare_array_or_none(self.quantized_data, other.quantized_data)
    )


@dataclasses.dataclass(frozen=True)
class OpToTensorParams:
  """Tensor params authored from an associated op.

  Attributes:
    subgraph_op_id: The position of the op in the subgraph.
    transformations: The transformations to be applied to the tensor.
    parameters: The quantization parameters for the tensor.
  """

  subgraph_op_id: int
  transformations: list[QuantTransformation]
  parameters: Union[None, UniformQuantParams, NonLinearQuantParams] = None


@dataclasses.dataclass
class TensorTransformationParams:
  """Transformation info for a tensor.

  Every tensor in .tflite has the following property:
   * Produced by one source op (producer), except constant tensor or model
   input.
   * Consumed by one or many destination ops (consumer), except model output.

  Because users configure quantization settings in Op level
  `OpQuantizationConfig`, each tensor will receive transformation parameters
   * from the source op
   * from the destination ops
  """

  tensor_name: str
  producer: Optional[OpToTensorParams] = None
  consumers: Optional[list[OpToTensorParams]] = None

  def __copy__(self):
    return TensorTransformationParams(
        tensor_name=self.tensor_name,
        producer=self.producer,
        consumers=self.consumers[:] if self.consumers is not None else None,
    )


@dataclasses.dataclass(frozen=True)
class TensorQuantizationConfig:
  """Quantization configuration for a tensor.

  Attributes:
    num_bits: Number of bits to quantize to (e.g. 8 for int8).
    symmetric: Whether to perform symmetric or asymmetric quantization. In the
      symmetric quantization mode, the zero point is always 0.
    granularity: Whether to perform per-tensor, per-channel or per-block
      quantization.
    dtype: The data type of the tensor.
    algorithm_key: The algorithm key to use for quantization.
    algorithm_params: Additional parameters for the quantization algorithm.
  """

  num_bits: int
  symmetric: bool = True
  granularity: QuantGranularity = QuantGranularity.TENSORWISE
  dtype: TensorDataType = TensorDataType.INT
  algorithm_params: Mapping[str, Any] = dataclasses.field(
      default_factory=immutabledict.immutabledict
  )

  def __post_init__(self):
    if not isinstance(self.algorithm_params, immutabledict.immutabledict):
      object.__setattr__(
          self,
          'algorithm_params',
          immutabledict.immutabledict(self.algorithm_params),
      )

  def to_dict(self) -> dict[str, Any]:
    """Converts ActivationQuantizationConfig to dict."""
    return dataclasses.asdict(
        self,
        dict_factory=lambda x: {  # pylint: disable=g-long-lambda
            k: (
                dict(v)
                if isinstance(v, Mapping) and not isinstance(v, dict)
                else v
            )
            for (k, v) in x
            # Skip None and empty dict values.
            if v is not None and not (isinstance(v, Mapping) and not v)
        },
    )

  @classmethod
  def from_dict(cls, params: dict[str, Any]) -> 'TensorQuantizationConfig':
    """Converts a given dict to TensorQuantizationConfig."""
    params_copy = copy.deepcopy(params)
    # Process block_size config from legacy recipe.
    params_copy = _process_block_size(params_copy)

    # Move any unknown fields to algorithm_params for backward compatibility.
    known_fields = {f.name for f in dataclasses.fields(cls)}
    algorithm_params = params_copy.pop('algorithm_params', {})
    for key in list(params_copy.keys()):
      if key not in known_fields:
        algorithm_params[key] = params_copy.pop(key)

    return cls(algorithm_params=algorithm_params, **params_copy)


def _process_block_size(params: dict[str, Any]) -> dict[str, Any]:
  """Processes block size in the params."""
  block_size = params.pop('block_size', 0)
  if block_size > 0:
    if block_size == 32:
      params['granularity'] = QuantGranularity.BLOCKWISE_32
    elif block_size == 64:
      params['granularity'] = QuantGranularity.BLOCKWISE_64
    elif block_size == 128:
      params['granularity'] = QuantGranularity.BLOCKWISE_128
    elif block_size == 256:
      params['granularity'] = QuantGranularity.BLOCKWISE_256
    else:
      raise ValueError(f'Unsupported block size: {block_size}')
  return params


@dataclasses.dataclass(frozen=True)
class OpQuantizationConfig:
  """Configuration class to control the quantization process behavior.

  Default to float activations and weights.

  Attributes:
    activation_tensor_config: The quantization configuration for activation
      tensors in the op (i.e., runtime tensors).
    weight_tensor_config: The quantization configuration for weight tensor in
      the op.
    compute_precision: The precision of the compute operation.
    explicit_dequantize: Whether to add explicit dequantize op if compute
      precision is FLOAT, but weight is quantized.
    skip_checks: Whether to skip op quantization config checks. For advanced
      users only. If set, the quantizer will ignore all op configuration checks
      and forcefully quantize this op according to the user instructions even if
      it's not supported in the TFLite runtime.
    min_weight_elements: The minimum number of elements in the weight tensor to
      be quantized.
  """

  activation_tensor_config: Optional[TensorQuantizationConfig] = None
  # Bias tensor quantization is deduced from activation/weight config.
  # e.g., int8A X int8W => int32 bias.
  weight_tensor_config: Optional[TensorQuantizationConfig] = None
  compute_precision: ComputePrecision = ComputePrecision.FLOAT
  # TODO: b/359647578 - Set default to True.
  explicit_dequantize: bool = False
  skip_checks: bool = False
  min_weight_elements: int = 0

  def __post_init__(self):
    if (
        self.activation_tensor_config is None
        or self.weight_tensor_config is None
    ):
      return
    # Make sure the setting is valid.
    if (
        self.activation_tensor_config.dtype == TensorDataType.INT
        and self.weight_tensor_config.dtype == TensorDataType.FLOAT
    ):
      raise ValueError(
          'An op can not be set to have integer activation but float weights!'
      )
    if (
        # SRQ compliance check for the config.
        self.activation_tensor_config.dtype == TensorDataType.INT
        and self.weight_tensor_config.dtype == TensorDataType.INT
        and self.compute_precision != ComputePrecision.INTEGER
    ):
      raise ValueError(
          'Op execution mode must be SRQ (static range quantization) if both'
          ' activation and weight tensors are quantized!'
      )

  def to_dict(self) -> dict[str, Any]:
    """Converts OpQuantizationConfig to dict."""
    return dataclasses.asdict(
        self,
        dict_factory=lambda x: {  # pylint: disable=g-long-lambda
            k: (
                dict(v)
                if isinstance(v, Mapping) and not isinstance(v, dict)
                else v
            )
            for (k, v) in x
            # Skip None and empty dict values.
            if v is not None and not (isinstance(v, Mapping) and not v)
        },
    )

  @classmethod
  def from_dict(cls, params: dict[str, Any]) -> 'OpQuantizationConfig':
    """Converts a given dict to OpQuantizationConfig."""
    params_copy = copy.deepcopy(params)
    params_copy['weight_tensor_config'] = TensorQuantizationConfig.from_dict(
        params_copy['weight_tensor_config']
    )
    if 'activation_tensor_config' in params_copy:
      params_copy['activation_tensor_config'] = (
          TensorQuantizationConfig.from_dict(
              params_copy['activation_tensor_config']
          )
      )
    return cls(**params_copy)


@dataclasses.dataclass(frozen=True)
class GraphInfo:
  """Aggregates graph information needed to perform quantization for an op.

  Attributes:
    subgraph_tensors: Tensors in the subgraph.
    buffers: Buffers in the subgraph.
  """

  subgraph_tensors: list[TensorT]
  buffers: list[BufferT]


@dataclasses.dataclass(frozen=True)
class OpInfo:
  """Aggregates op information needed to perform quantization for an op.

  Attributes:
    op: The op to be quantized.
    op_name: The name of the op.
    subgraph_op_index: The position of the op in the subgraph.
    op_quant_config: The quantization configuration for the op.
  """

  op: OperatorT
  op_name: TFLOperationName
  subgraph_op_index: int  # Position of the op in the subgraph.
  op_quant_config: OpQuantizationConfig


# Data classes used by model modifier.


# TODO: b/335530570 - This needs to support more than one parameters.
@dataclasses.dataclass
class TransformationInst:
  """Transformation instruction for a tensor.

  Attributes:
    transformation: The transformation to be applied to the tensor.
    tensor_id: The id of the tensor.
    producer: The id of the producer op.
    consumers: The ids of the consumer ops.
    parameters: The quantization parameters for the tensor.
  """

  transformation: QuantTransformation
  tensor_id: int
  producer: Optional[int]
  consumers: list[int]
  parameters: Union[None, UniformQuantParams, NonLinearQuantParams] = None


@dataclasses.dataclass
class TensorTransformationInsts:
  """Transformation instructions for a tensor.

  Attributes:
    tensor_name: The name of the tensor.
    subgraph_id: The id of the subgraph.
    instructions: The transformation instructions for the tensor.
  """

  tensor_name: str
  subgraph_id: int
  instructions: Optional[list[TransformationInst]]


@dataclasses.dataclass(frozen=True)
class TransformationInfo:
  """Transformation information for an op.

  Attributes:
    op_id: The id where op replacement/insertion begins.
    num_ops_added: The number of ops added during the transformation.
    output_tensor_id: The id of the output tensor.
  """

  op_id: int
  num_ops_added: int
  output_tensor_id: int


# Policy is represented as a dict to check the op quantization config.
# Normally the policy is loaded from a json file.
ConfigCheckPolicyDict = collections.OrderedDict[
    TFLOperationName, list[OpQuantizationConfig]
]


def _compare_array_or_none(
    obj1: Optional[np.ndarray], obj2: Optional[np.ndarray]
):
  """Compares two arrays or None.

  Args:
    obj1: The first object to compare.
    obj2: The second object to compare.

  Returns:
    True if both objects are None or both objects are equal.
  """
  if obj1 is None and obj2 is None:
    return True  # Both None, so they're equal.
  if obj1 is None or obj2 is None:
    return False  # Only one is None, so they're different.
  if id(obj1) == id(obj2) or obj1.data == obj2.data:
    return True
  return np.array_equal(obj1, obj2)


def _is_equal(x: Any, y: Any) -> bool:
  """Compares two values, handling numpy arrays, None, and general objects."""
  if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
    return np.array_equal(x, y)
  if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
    return False
  return x == y


def _compare_custom_algorithm_param(
    param1: Optional[dict[str, Any]], param2: Optional[dict[str, Any]]
) -> bool:
  """Compares two custom_algorithm_param dicts or None."""
  if param1 is None and param2 is None:
    return True
  if param1 is None or param2 is None:
    return False
  if param1.keys() != param2.keys():
    return False
  return all(_is_equal(val1, param2[key]) for key, val1 in param1.items())


@dataclasses.dataclass(frozen=True)
class IOOperator:
  """IOOperator class to represent the input and output for a subgraph.

  Attributes:
    inputs: The input tensor ids of the op.
    outputs: The output tensor ids of the op.
    op_key: The op key of the op (input or output).
  """

  inputs: list[int]
  outputs: list[int]
  op_key: TFLOperationName

# The function signature for `get_tensor_quant_params_fn`.
GetTensorQuantParamsFuncSignature = Callable[
    [
        OpInfo,  # op_info
        TensorQuantizationConfig,  # tensor_quant_config
        Optional[np.ndarray],  # tensor_data
        Optional[dict[str, Any]],  # tensor qsv
    ],
    UniformQuantParams,
]
