# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Python LiteRT IR backed by the C++ LiteRT IR (`LiteRtModelT`).

The model is loaded once into a C++ `LiteRtModelT` and every query and edit
operates on that in-memory IR; flatbuffers are only touched by `Model.load_*`
and `Model.serialize`.

Ownership:
  * A `Model` owns its `LiteRtModelT`.
  * `Subgraph`, `Op`, `Tensor` and `Signature` are lightweight non-owning
    handles that keep the owning model alive. Two handles to the same IR object
    compare (and hash) equal, so they can be used as dict keys.
  * Handles to IR objects removed by an edit (`remove_op`, `dce`,
    `outline_to_dispatch_op`, `prune_unreachable_subgraphs`,
    `remove_signature`, ...) raise `LiteRtError` when used.
  * `Model.serialize` consumes the model: any later use of the model or of its
    handles raises `LiteRtError("model has been serialized")`.

Graph edits keep the def-use links consistent and refuse to break the
topological order of `Subgraph.ops`. Transient states such as a tensor that is
used but not (yet) produced by any op are allowed; `Model.validate` reports
them.

Errors: precondition failures of graph edits (bad index, handle of another
model or subgraph, topological order violation, duplicate signature key...)
raise `IndexError`/`ValueError`/`TypeError`. Failures reported by the C++ IR,
stale handles and use after `serialize` raise `LiteRtError`.

Omitted optional op inputs: the C++ loader drops TFLite operand index -1 and
`LiteRtOpT` has no slot for an absent input, so `Op.inputs` never contains None
and the inputs following an omitted one shift left (TFLite inputs `[a, -1, c]`
load as `[a, c]`). `Subgraph.add_op` accepts None only as trailing inputs
(dropped).

Example:

  model = litert_model.Model.load_from_file("model.tflite")
  subgraph = model.subgraph(0)
  for op in subgraph.ops:
    print(op.name, [t.shape for t in op.inputs], op.builtin_options)
  data = model.serialize()
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import dataclasses
import enum
import math
import os
from typing import Any, TypeAlias

import flatbuffers
import numpy as np

# pylint: disable=g-import-not-at-top
try:
  from litert.python import schema_py_generated as schema_fb  # pylint: disable=g-direct-tensorflow-import
except ImportError:
  # In OSS Bazel tests, //tflite/python:schema_py provides tflite.python.
  from tflite.python import schema_py_generated as schema_fb  # pyrefly: ignore[missing-import]

if not os.path.splitext(__file__)[0].endswith(
    os.path.join("ai_edge_litert", "litert_model", "litert_model")
):
  # Part of the litert package.
  from litert.python.litert_model import _pywrap_litert_model as _pw
else:
  # Part of the ai_edge_litert wheel, which ships the extension at its root.
  from ai_edge_litert import _pywrap_litert_model as _pw  # pyrefly: ignore[missing-import]
# pylint: enable=g-import-not-at-top

# Errors. `LiteRtError` (a `RuntimeError`) carries `status` (the LiteRtStatus
# name, e.g. "kLiteRtStatusErrorInvalidArgument") and `status_code`.
LiteRtError = _pw.LiteRtError
# Raised by shape inference; `op` is the failing `Op` when known.
ShapeInferenceError = _pw.ShapeInferenceError
# Raised by `Model.validate` (and `Model.serialize`) for invalid IR.
ValidationError = _pw.ValidationError

# Custom code of dispatch ops.
DISPATCH_OP_CUSTOM_CODE: str = _pw.DISPATCH_OP_CUSTOM_CODE
# Key of the signature the C++ loader adds to models without signatures (it
# covers subgraph 0 and names I/O after the tensors).
DEFAULT_SIGNATURE_KEY: str = _pw.DEFAULT_SIGNATURE_KEY
# Metadata key of the build stamp written by `Model.set_build_stamp`.
BUILD_STAMP_METADATA_KEY: str = _pw.BUILD_STAMP_METADATA_KEY


class ElementType(enum.IntEnum):
  """Tensor element types; values mirror `LiteRtElementType`."""

  NONE = 0
  BOOL = 6
  INT2 = 20
  INT4 = 18
  INT8 = 9
  INT16 = 7
  INT32 = 2
  INT64 = 4
  UINT4 = 21
  UINT8 = 3
  UINT16 = 17
  UINT32 = 16
  UINT64 = 13
  FLOAT8_E4M3FN = 22
  FLOAT8_E5M2 = 23
  FLOAT16 = 10
  BFLOAT16 = 19
  FLOAT32 = 1
  FLOAT64 = 11
  COMPLEX64 = 8
  COMPLEX128 = 12
  TF_RESOURCE = 14
  TF_STRING = 5
  TF_VARIANT = 15

  @classmethod
  def from_numpy(cls, dtype: Any) -> ElementType:
    """Returns the element type of a numpy dtype (or anything `np.dtype` accepts).

    Raises:
      TypeError: If the dtype has no LiteRT element type.
    """
    numpy_dtype = np.dtype(dtype)
    for element_type, candidate in _NUMPY_DTYPES.items():
      if candidate == numpy_dtype:
        return element_type
    raise TypeError(f"numpy dtype {numpy_dtype} has no LiteRT element type")

  @property
  def numpy_dtype(self) -> np.dtype | None:
    """The numpy dtype, or None (sub-byte, 8-bit/bfloat16 floats, strings...)."""
    return _NUMPY_DTYPES.get(self)

  @property
  def bit_width(self) -> int:
    """Bits per element (packed for sub-byte types); 0 if not fixed."""
    return _pw.element_type_bit_width(int(self))


_NUMPY_DTYPES: dict[ElementType, np.dtype] = {
    ElementType.BOOL: np.dtype(np.bool_),
    ElementType.INT8: np.dtype(np.int8),
    ElementType.INT16: np.dtype(np.int16),
    ElementType.INT32: np.dtype(np.int32),
    ElementType.INT64: np.dtype(np.int64),
    ElementType.UINT8: np.dtype(np.uint8),
    ElementType.UINT16: np.dtype(np.uint16),
    ElementType.UINT32: np.dtype(np.uint32),
    ElementType.UINT64: np.dtype(np.uint64),
    ElementType.FLOAT16: np.dtype(np.float16),
    ElementType.FLOAT32: np.dtype(np.float32),
    ElementType.FLOAT64: np.dtype(np.float64),
    ElementType.COMPLEX64: np.dtype(np.complex64),
    ElementType.COMPLEX128: np.dtype(np.complex128),
}


class OpCode(enum.IntEnum):
  """Op codes; values mirror `LiteRtOpCode` (TFLite `BuiltinOperator` names).

  Op codes of a loaded model that LiteRT does not know yet are represented by
  pseudo members named `UNKNOWN_<value>`.
  """

  ADD = 0
  AVERAGE_POOL_2D = 1
  CONCATENATION = 2
  CONV_2D = 3
  DEPTHWISE_CONV_2D = 4
  DEPTH_TO_SPACE = 5
  DEQUANTIZE = 6
  EMBEDDING_LOOKUP = 7
  FLOOR = 8
  FULLY_CONNECTED = 9
  HASHTABLE_LOOKUP = 10
  L2_NORMALIZATION = 11
  L2_POOL_2D = 12
  LOCAL_RESPONSE_NORMALIZATION = 13
  LOGISTIC = 14
  LSH_PROJECTION = 15
  LSTM = 16
  MAX_POOL_2D = 17
  MUL = 18
  RELU = 19
  RELU_N1_TO_1 = 20
  RELU6 = 21
  RESHAPE = 22
  RESIZE_BILINEAR = 23
  RNN = 24
  SOFTMAX = 25
  SPACE_TO_DEPTH = 26
  SVDF = 27
  TANH = 28
  CONCAT_EMBEDDINGS = 29
  SKIP_GRAM = 30
  CALL = 31
  CUSTOM = 32
  EMBEDDING_LOOKUP_SPARSE = 33
  PAD = 34
  UNIDIRECTIONAL_SEQUENCE_RNN = 35
  GATHER = 36
  BATCH_TO_SPACE_ND = 37
  SPACE_TO_BATCH_ND = 38
  TRANSPOSE = 39
  MEAN = 40
  SUB = 41
  DIV = 42
  SQUEEZE = 43
  UNIDIRECTIONAL_SEQUENCE_LSTM = 44
  STRIDED_SLICE = 45
  BIDIRECTIONAL_SEQUENCE_RNN = 46
  EXP = 47
  TOPK_V2 = 48
  SPLIT = 49
  LOG_SOFTMAX = 50
  DELEGATE = 51
  BIDIRECTIONAL_SEQUENCE_LSTM = 52
  CAST = 53
  PRELU = 54
  MAXIMUM = 55
  ARG_MAX = 56
  MINIMUM = 57
  LESS = 58
  NEG = 59
  PADV2 = 60
  GREATER = 61
  GREATER_EQUAL = 62
  LESS_EQUAL = 63
  SELECT = 64
  SLICE = 65
  SIN = 66
  TRANSPOSE_CONV = 67
  SPARSE_TO_DENSE = 68
  TILE = 69
  EXPAND_DIMS = 70
  EQUAL = 71
  NOT_EQUAL = 72
  LOG = 73
  SUM = 74
  SQRT = 75
  RSQRT = 76
  SHAPE = 77
  POW = 78
  ARG_MIN = 79
  FAKE_QUANT = 80
  REDUCE_PROD = 81
  REDUCE_MAX = 82
  PACK = 83
  LOGICAL_OR = 84
  ONE_HOT = 85
  LOGICAL_AND = 86
  LOGICAL_NOT = 87
  UNPACK = 88
  REDUCE_MIN = 89
  FLOOR_DIV = 90
  REDUCE_ANY = 91
  SQUARE = 92
  ZEROS_LIKE = 93
  FILL = 94
  FLOOR_MOD = 95
  RANGE = 96
  RESIZE_NEAREST_NEIGHBOR = 97
  LEAKY_RELU = 98
  SQUARED_DIFFERENCE = 99
  MIRROR_PAD = 100
  ABS = 101
  SPLIT_V = 102
  UNIQUE = 103
  CEIL = 104
  REVERSE_V2 = 105
  ADD_N = 106
  GATHER_ND = 107
  COS = 108
  WHERE = 109
  RANK = 110
  ELU = 111
  REVERSE_SEQUENCE = 112
  MATRIX_DIAG = 113
  QUANTIZE = 114
  MATRIX_SET_DIAG = 115
  ROUND = 116
  HARD_SWISH = 117
  IF = 118
  WHILE = 119
  NON_MAX_SUPPRESSION_V4 = 120
  NON_MAX_SUPPRESSION_V5 = 121
  SCATTER_ND = 122
  SELECT_V2 = 123
  DENSIFY = 124
  SEGMENT_SUM = 125
  BATCH_MATMUL = 126
  PLACEHOLDER_FOR_GREATER_OP_CODES = 127
  CUMSUM = 128
  CALL_ONCE = 129
  BROADCAST_TO = 130
  RFFT2D = 131
  CONV_3D = 132
  IMAG = 133
  REAL = 134
  COMPLEX_ABS = 135
  HASHTABLE = 136
  HASHTABLE_FIND = 137
  HASHTABLE_IMPORT = 138
  HASHTABLE_SIZE = 139
  REDUCE_ALL = 140
  CONV_3D_TRANSPOSE = 141
  VAR_HANDLE = 142
  READ_VARIABLE = 143
  ASSIGN_VARIABLE = 144
  BROADCAST_ARGS = 145
  RANDOM_STANDARD_NORMAL = 146
  BUCKETIZE = 147
  RANDOM_UNIFORM = 148
  MULTINOMIAL = 149
  GELU = 150
  DYNAMIC_UPDATE_SLICE = 151
  RELU_0_TO_1 = 152
  UNSORTED_SEGMENT_PROD = 153
  UNSORTED_SEGMENT_MAX = 154
  UNSORTED_SEGMENT_SUM = 155
  ATAN2 = 156
  UNSORTED_SEGMENT_MIN = 157
  SIGN = 158
  BITCAST = 159
  BITWISE_XOR = 160
  RIGHT_SHIFT = 161
  STABLEHLO_LOGISTIC = 162
  STABLEHLO_ADD = 163
  STABLEHLO_DIVIDE = 164
  STABLEHLO_MULTIPLY = 165
  STABLEHLO_MAXIMUM = 166
  STABLEHLO_RESHAPE = 167
  STABLEHLO_CLAMP = 168
  STABLEHLO_CONCATENATE = 169
  STABLEHLO_BROADCAST_IN_DIM = 170
  STABLEHLO_CONVOLUTION = 171
  STABLEHLO_SLICE = 172
  STABLEHLO_CUSTOM_CALL = 173
  STABLEHLO_REDUCE = 174
  STABLEHLO_ABS = 175
  STABLEHLO_AND = 176
  STABLEHLO_COSINE = 177
  STABLEHLO_EXPONENTIAL = 178
  STABLEHLO_FLOOR = 179
  STABLEHLO_LOG = 180
  STABLEHLO_MINIMUM = 181
  STABLEHLO_NEGATE = 182
  STABLEHLO_OR = 183
  STABLEHLO_POWER = 184
  STABLEHLO_REMAINDER = 185
  STABLEHLO_RSQRT = 186
  STABLEHLO_SELECT = 187
  STABLEHLO_SUBTRACT = 188
  STABLEHLO_TANH = 189
  STABLEHLO_SCATTER = 190
  STABLEHLO_COMPARE = 191
  STABLEHLO_CONVERT = 192
  STABLEHLO_DYNAMIC_SLICE = 193
  STABLEHLO_DYNAMIC_UPDATE_SLICE = 194
  STABLEHLO_PAD = 195
  STABLEHLO_IOTA = 196
  STABLEHLO_DOT_GENERAL = 197
  STABLEHLO_REDUCE_WINDOW = 198
  STABLEHLO_SORT = 199
  STABLEHLO_WHILE = 200
  STABLEHLO_GATHER = 201
  STABLEHLO_TRANSPOSE = 202
  DILATE = 203
  STABLEHLO_RNG_BIT_GENERATOR = 204
  REDUCE_WINDOW = 205
  STABLEHLO_COMPOSITE = 206
  STABLEHLO_SHIFT_LEFT = 207
  STABLEHLO_CBRT = 208
  STABLEHLO_CASE = 209

  @classmethod
  def _missing_(cls, value: object) -> OpCode | None:
    if isinstance(value, int) and value >= 0:
      member = int.__new__(cls, value)
      setattr(member, "_name_", f"UNKNOWN_{value}")
      setattr(member, "_value_", value)
      return member
    return None


@dataclasses.dataclass(frozen=True)
class PerTensorQuantization:
  """Per-tensor affine quantization: `real = scale * (q - zero_point)`."""

  scale: float
  zero_point: int = 0


@dataclasses.dataclass(frozen=True, eq=False)
class PerChannelQuantization:
  """Per-channel affine quantization along `quantized_dimension`.

  `scales` (float32) and `zero_points` (int64) are converted to read-only 1-D
  numpy arrays.
  """

  scales: np.ndarray
  zero_points: np.ndarray
  quantized_dimension: int

  def __post_init__(self):
    scales = np.array(self.scales, dtype=np.float32).reshape(-1)
    zero_points = np.array(self.zero_points, dtype=np.int64).reshape(-1)
    scales.setflags(write=False)
    zero_points.setflags(write=False)
    object.__setattr__(self, "scales", scales)
    object.__setattr__(self, "zero_points", zero_points)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, PerChannelQuantization):
      return NotImplemented
    return (
        self.quantized_dimension == other.quantized_dimension
        and np.array_equal(self.scales, other.scales)
        and np.array_equal(self.zero_points, other.zero_points)
    )

  def __hash__(self) -> int:
    return hash((
        self.quantized_dimension,
        self.scales.tobytes(),
        self.zero_points.tobytes(),
    ))


@dataclasses.dataclass(frozen=True)
class BlockWiseQuantization:
  """Block-wise quantization with scales (and zero points) held in tensors."""

  scales: Tensor
  zero_points: Tensor | None
  block_size: int


Quantization: TypeAlias = (
    PerTensorQuantization
    | PerChannelQuantization
    | BlockWiseQuantization
    | None
)


@dataclasses.dataclass(frozen=True)
class DispatchOpOptions:
  """Options of a dispatch op of a serialized model.

  Attributes:
    bytecode_size: Size in bytes of the bytecode (asset) of the op.
    bytecode_offset: Offset of the bytecode from the start of the model buffer.
    name: Entry point of the op in the bytecode.
  """

  bytecode_size: int
  bytecode_offset: int
  name: str


# ===----------------------------------------------------------------------===#
# Builtin options <-> tflite::Operator flatbuffer round trip.
# ===----------------------------------------------------------------------===#


def _union_members(union_type: type[Any]) -> dict[str, int]:
  return {
      name: value
      for name, value in vars(union_type).items()
      if not name.startswith("_") and isinstance(value, int) and name != "NONE"
  }


_BUILTIN_OPTIONS: dict[str, int] = _union_members(schema_fb.BuiltinOptions)
_BUILTIN_OPTIONS_2: dict[str, int] = _union_members(schema_fb.BuiltinOptions2)
_BUILTIN_OPTIONS_NAMES: dict[int, str] = {
    value: name for name, value in _BUILTIN_OPTIONS.items()
}
_BUILTIN_OPTIONS_2_NAMES: dict[int, str] = {
    value: name for name, value in _BUILTIN_OPTIONS_2.items()
}
# Lower case options name -> options name, used to find the options type that
# is named after an op (e.g. CONV_2D -> Conv2DOptions).
_OPTIONS_BY_NORMALIZED_NAME: dict[str, str] = {
    name.lower(): name for name in (*_BUILTIN_OPTIONS, *_BUILTIN_OPTIONS_2)
}


def _options_union(options: object) -> tuple[int, int, str]:
  """Returns (union 1 or 2, union type tag, options name) of an `*OptionsT`."""
  class_name = type(options).__name__
  name = class_name.removesuffix("T")
  if class_name.endswith("T") and getattr(schema_fb, class_name, None) is type(
      options
  ):
    if name in _BUILTIN_OPTIONS:
      return 1, _BUILTIN_OPTIONS[name], name
    if name in _BUILTIN_OPTIONS_2:
      return 2, _BUILTIN_OPTIONS_2[name], name
  raise TypeError(
      f"{class_name} is not a builtin options type; expected a schema_fb"
      " *OptionsT object (e.g. schema_fb.Conv2DOptionsT)"
  )


def _check_options_for_op(
    code: OpCode, current_types: tuple[int, int], options: object
) -> tuple[int, int]:
  """Validates that `options` can be attached to an op with `code`."""
  if code == OpCode.CUSTOM:
    raise TypeError(
        "CUSTOM ops have no builtin options; use custom options instead"
    )
  union, type_tag, name = _options_union(options)
  current_name = None
  if current_types[0]:
    current_name = _BUILTIN_OPTIONS_NAMES.get(current_types[0])
  elif current_types[1]:
    current_name = _BUILTIN_OPTIONS_2_NAMES.get(current_types[1])
  if current_name is not None and current_name != name:
    raise TypeError(f"{code.name} op has {current_name} options, got {name}T")
  expected = _OPTIONS_BY_NORMALIZED_NAME.get(
      code.name.replace("_", "").lower() + "options"
  )
  if expected is not None and expected != name:
    raise TypeError(f"{code.name} op expects {expected}T, got {name}T")
  return union, type_tag


def _pack_options(union: int, type_tag: int, options: object | None) -> bytes:
  operator = schema_fb.OperatorT()
  if options is not None:
    if union == 1:
      operator.builtinOptionsType = type_tag
      operator.builtinOptions = options
    else:
      operator.builtinOptions2Type = type_tag
      operator.builtinOptions2 = options
  builder = flatbuffers.Builder(256)
  builder.Finish(operator.Pack(builder))
  return bytes(builder.Output())


def _unpack_options(data: bytes, types: tuple[int, int]) -> object | None:
  operator = schema_fb.OperatorT.InitFromObj(
      schema_fb.Operator.GetRootAs(data, 0)
  )
  if types[0]:
    return operator.builtinOptions
  if types[1]:
    return operator.builtinOptions2
  return None


def _wrap_failing_op(error: ShapeInferenceError) -> None:
  """Replaces the raw failing op of `error` by an `Op` named in the message."""
  if isinstance(error.op, _pw.Op):
    error.op = Op(error.op)
    error.args = (f"{error.args[0]} [{error.op.name}]",) + error.args[1:]


# ===----------------------------------------------------------------------===#
# Handles.
# ===----------------------------------------------------------------------===#


class _Handle:
  """Base of the non-owning IR handles (identity based equality)."""

  __slots__ = ("_handle",)

  def __init__(self, handle: Any):
    self._handle = handle

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, _Handle) or type(other) is not type(self):
      return NotImplemented
    return self._handle == other._handle

  def __hash__(self) -> int:
    return hash(self._handle)


class Model:
  """A LiteRT model (owner of a C++ `LiteRtModelT`).

  Create instances with `load_from_file`, `load_from_bytes` or `create`.
  """

  __slots__ = ("_handle",)

  def __init__(self, handle: _pw.Model):
    """Wraps a model handle. Use the `load_*`/`create` class methods."""
    if not isinstance(handle, _pw.Model):
      raise TypeError("use Model.load_from_file/load_from_bytes/create")
    self._handle = handle

  @classmethod
  def load_from_file(cls, path: str | os.PathLike[str]) -> Model:
    """Loads a `.tflite` file (memory mapped; assets of dispatch ops attached)."""
    return cls(_pw.Model.load_from_file(os.fspath(path)))

  @classmethod
  def load_from_bytes(cls, data: bytes | bytearray | memoryview[int]) -> Model:
    """Loads a model from a buffer (copied into a model owned buffer)."""
    return cls(_pw.Model.load_from_bytes(data))

  @classmethod
  def create(cls) -> Model:
    """Returns an empty model (no subgraphs, signatures or metadata)."""
    return cls(_pw.Model.create())

  # Subgraphs.

  @property
  def subgraphs(self) -> list[Subgraph]:
    """All subgraphs, in order."""
    return [Subgraph(h) for h in self._handle.subgraphs()]

  def subgraph(self, index: int) -> Subgraph:
    """Returns subgraph `index` (IndexError if out of range)."""
    return Subgraph(self._handle.subgraph(index))

  def add_subgraph(self, name: str = "") -> Subgraph:
    """Appends an empty subgraph."""
    return Subgraph(self._handle.add_subgraph(name))

  def prune_unreachable_subgraphs(self) -> int:
    """Removes subgraphs not reachable from any signature (or main subgraph).

    Reachability follows signatures (or subgraph 0 when the model is
    signatureless) and `STABLEHLO_COMPOSITE` `decompositionSubgraphIndex`
    references. Remaining `STABLEHLO_COMPOSITE` ops have their
    `decompositionSubgraphIndex` attributes remapped to the new subgraph
    positions. Handles to removed subgraphs, ops and tensors become invalid and
    raise `LiteRtError` when used.

    Returns:
      The number of removed subgraphs (`0` if nothing was unreachable or if
      control-flow operators such as `WHILE` or `IF` prevented pruning).
    """
    return self._handle.prune_unreachable_subgraphs()

  # Signatures.

  @property
  def signatures(self) -> list[Signature]:
    """All signatures, in order."""
    return [Signature(h) for h in self._handle.signatures()]

  def find_signature(self, key: str) -> Signature:
    """Returns the signature `key` (KeyError if missing)."""
    return Signature(self._handle.find_signature(key))

  def add_signature(
      self,
      key: str,
      subgraph: Subgraph,
      inputs: Sequence[tuple[str, Tensor]],
      outputs: Sequence[tuple[str, Tensor]],
  ) -> Signature:
    """Adds a signature (named entry point) for `subgraph`.

    If the subgraph inputs (outputs) are empty they are set from the signature
    in the given order; otherwise the signature tensors must be exactly the
    subgraph inputs (outputs), in any order. Signature names follow the
    subgraph I/O order (that is how TFLite serializes them).

    Args:
      key: Unique signature key.
      subgraph: Subgraph called by the signature.
      inputs: (name, tensor) pairs of the inputs.
      outputs: (name, tensor) pairs of the outputs.

    Returns:
      The new signature.

    Raises:
      ValueError: If the key exists or the tensors do not match the subgraph.
    """
    return Signature(
        self._handle.add_signature(
            key,
            subgraph._handle,  # pylint: disable=protected-access
            [(name, t._handle) for name, t in inputs],  # pylint: disable=protected-access
            [(name, t._handle) for name, t in outputs],  # pylint: disable=protected-access
        )
    )

  def remove_signature(self, key: str) -> None:
    """Removes the signature `key` (KeyError if missing).

    The subgraph of the signature is kept.

    Args:
      key: Key of the signature to remove.
    """
    self._handle.remove_signature(key)

  # Metadata.

  def metadata_keys(self) -> list[str]:
    """Sorted metadata keys."""
    return self._handle.metadata_keys()

  def metadata(self, key: str) -> bytes | None:
    """Returns a copy of the metadata `key`, or None."""
    return self._handle.metadata(key)

  def push_metadata(
      self, key: str, data: bytes | bytearray | memoryview[int]
  ) -> None:
    """Adds metadata (LiteRtError if `key` already exists)."""
    self._handle.push_metadata(key, data)

  # Validation and serialization.

  def validate(self) -> None:
    """Checks the structural invariants of the IR.

    Checks def-use consistency (`graph_validation`), topological order, that
    every used tensor is constant, a subgraph input or produced by an op,
    subgraph I/O, the operator code table and that signatures match their
    subgraph I/O by position. Signature names need not be unique (the
    loader's placeholder signature repeats the name of a repeated output).

    This is stricter than what serialization needs: `serialize` only checks
    what `SerializeModel` relies on, so every model the C++ loader produces
    serializes even if `validate` reports issues (e.g. a used tensor that is
    neither constant, a subgraph input nor produced by an op).

    Raises:
      ValidationError: Listing every violation.
    """
    self._handle.validate()

  def serialize(self, *, bytecode_alignment: int = 1) -> bytes:
    """Serializes the model to a `.tflite` flatbuffer and consumes it.

    Assets attached to dispatch ops are appended after the flatbuffer, each
    aligned to `bytecode_alignment`, and the dispatch ops custom options record
    their offset and size.

    Args:
      bytecode_alignment: Alignment (a power of two) of appended assets.

    Returns:
      The serialized model.

    Raises:
      ValidationError: If the IR cannot be serialized faithfully (dangling
        edges, inconsistent def-use links, bad operator code entries,
        signatures out of sync with their subgraph); the model is not consumed.
      LiteRtError: If serialization fails (the model is consumed).
    """
    return self._handle.serialize(bytecode_alignment)

  @property
  def is_serialized(self) -> bool:
    """True once `serialize` consumed the model."""
    return self._handle.is_serialized()

  # Shape inference.

  def infer_shapes(self, *, validation_only: bool = False) -> None:
    """Runs shape inference on every subgraph (`ShapeInferenceEngine`).

    Args:
      validation_only: Only check that existing shapes match the inferred ones.

    Raises:
      ShapeInferenceError: With `op` set to the failing op when known.
    """
    try:
      self._handle.infer_shapes(validation_only)
    except ShapeInferenceError as e:
      _wrap_failing_op(e)
      raise

  def specialize_subgraph(
      self, subgraph: Subgraph, input_shapes: Sequence[Sequence[int]]
  ) -> Subgraph:
    """Appends a copy of `subgraph` with static input shapes and inferred shapes.

    Args:
      subgraph: Subgraph to copy.
      input_shapes: One shape per subgraph input.

    Returns:
      The new subgraph (no signature refers to it).

    Raises:
      ShapeInferenceError: If shape inference fails (nothing is added).
    """
    try:
      return Subgraph(
          self._handle.specialize_subgraph(
              subgraph._handle,  # pylint: disable=protected-access
              [list(shape) for shape in input_shapes],
          )
      )
    except ShapeInferenceError as e:
      _wrap_failing_op(e)
      raise

  # Dispatch ops and assets (mirrors `ApplyPluginWithPartition`).

  def register_buffer(self, data: bytes | bytearray | memoryview[int]) -> int:
    """Copies `data` into a model owned buffer and returns its id.

    The copy is kept until the model is destroyed.

    Args:
      data: Raw buffer bytes to copy into the model.

    Returns:
      The buffer id.
    """
    return self._handle.register_buffer(data)

  def attach_asset(self, op: Op, buffer_id: int, name: str) -> None:
    """Attaches buffer `buffer_id` to dispatch op `op` as entry point `name`."""
    self._handle.attach_asset(op._handle, buffer_id, name)  # pylint: disable=protected-access

  def set_build_stamp(self, soc_manufacturer: str, soc_model: str) -> None:
    """Records the compiler target in the model metadata.

    Args:
      soc_manufacturer: Target SoC manufacturer identifier.
      soc_model: Target SoC model identifier.

    Raises:
      LiteRtError: If a build stamp exists or the strings are too long.
    """
    self._handle.set_build_stamp(soc_manufacturer, soc_model)

  def build_stamp(self) -> tuple[str, str] | None:
    """Returns (soc_manufacturer, soc_model) of the build stamp, or None."""
    return self._handle.build_stamp()

  def __repr__(self) -> str:
    if self.is_serialized:
      return "<litert_model.Model (serialized)>"
    return (
        f"<litert_model.Model subgraphs={self._handle.num_subgraphs()}"
        f" signatures={[s.key for s in self.signatures]}>"
    )


class Subgraph(_Handle):
  """A subgraph (function) of a model."""

  __slots__ = ()

  @property
  def index(self) -> int:
    """Position in `Model.subgraphs`."""
    return self._handle.index()

  @property
  def name(self) -> str:
    return self._handle.name()

  def set_name(self, name: str) -> None:
    self._handle.set_name(name)

  @property
  def inputs(self) -> list[Tensor]:
    return [Tensor(h) for h in self._handle.inputs()]

  @property
  def outputs(self) -> list[Tensor]:
    return [Tensor(h) for h in self._handle.outputs()]

  def set_inputs(self, inputs: Sequence[Tensor]) -> None:
    """Replaces the subgraph inputs; signatures of the subgraph follow."""
    self._handle.set_inputs([t._handle for t in inputs])  # pylint: disable=protected-access

  def set_outputs(self, outputs: Sequence[Tensor]) -> None:
    """Replaces the subgraph outputs; signatures of the subgraph follow."""
    self._handle.set_outputs([t._handle for t in outputs])  # pylint: disable=protected-access

  @property
  def tensors(self) -> list[Tensor]:
    return [Tensor(h) for h in self._handle.tensors()]

  @property
  def ops(self) -> list[Op]:
    """Ops in topological (execution) order."""
    return [Op(h) for h in self._handle.ops()]

  def add_tensor(
      self,
      *,
      shape: Sequence[int],
      element_type: ElementType,
      name: str = "",
      quantization: Quantization = None,
      weights: bytes | bytearray | memoryview[int] | np.ndarray | None = None,
  ) -> Tensor:
    """Adds a ranked tensor.

    Args:
      shape: Dimensions, -1 for dynamic ones.
      element_type: Element type.
      name: Tensor name.
      quantization: Optional quantization parameters.
      weights: Optional constant data (copied into the model).

    Returns:
      The new tensor.
    """
    tensor = Tensor(
        self._handle.add_tensor(
            list(shape), int(ElementType(element_type)), name
        )
    )
    try:
      if quantization is not None:
        tensor.set_quantization(quantization)
      if weights is not None:
        tensor.set_weights(weights)
    except Exception:
      self._handle.remove_tensor(tensor._handle)  # pylint: disable=protected-access
      raise
    return tensor

  def remove_tensor(self, tensor: Tensor) -> None:
    """Deallocates an unused tensor (no uses, producer, I/O role)."""
    self._handle.remove_tensor(tensor._handle)  # pylint: disable=protected-access

  def add_op(
      self,
      code: OpCode,
      inputs: Sequence[Tensor | None],
      outputs: Sequence[Tensor],
      *,
      index: int | None = None,
      custom_code: str | None = None,
      custom_options: bytes | None = None,
      builtin_options: object | None = None,
  ) -> Op:
    """Adds an op.

    Args:
      code: Op code.
      inputs: Input tensors. Trailing None entries (omitted optional inputs) are
        dropped; the C++ IR cannot represent omitted inputs elsewhere.
      outputs: Output tensors; they must not be produced by another op, be
        constant or be subgraph inputs.
      index: Position in `ops` (default: append). Inputs must be produced before
        it and existing users of the outputs after it.
      custom_code: Custom code (required for CUSTOM ops only).
      custom_options: Opaque custom options.
      builtin_options: A schema_fb `*OptionsT` object matching `code`.

    Returns:
      The new op.
    """
    code = OpCode(code)
    input_list = list(inputs)
    while input_list and input_list[-1] is None:
      input_list.pop()
    input_tensors = [t for t in input_list if t is not None]
    if len(input_tensors) != len(input_list):
      raise ValueError(
          "only trailing optional inputs can be omitted (None); the LiteRT IR"
          " does not represent omitted inputs in the middle of the input list"
      )
    packed_options = None
    if builtin_options is not None:
      union, type_tag = _check_options_for_op(code, (0, 0), builtin_options)
      packed_options = _pack_options(union, type_tag, builtin_options)
    op = Op(
        self._handle.add_op(
            int(code),
            [t._handle for t in input_tensors],  # pylint: disable=protected-access
            [t._handle for t in outputs],  # pylint: disable=protected-access
            index,
            custom_code,
            custom_options,
        )
    )
    if packed_options is not None:
      try:
        op._handle.set_builtin_options_bytes(packed_options)  # pylint: disable=protected-access
      except Exception:
        self._handle.remove_op(op._handle)  # pylint: disable=protected-access
        raise
    return op

  def remove_op(self, op: Op) -> None:
    """Disconnects `op` from its tensors and deallocates it.

    The tensors are kept (use `dce` to remove the dead ones). Replace the uses
    of the op outputs first (`replace_all_uses`) to keep the graph valid.

    Args:
      op: Op to remove from this subgraph.
    """
    self._handle.remove_op(op._handle)  # pylint: disable=protected-access

  def dce(self) -> bool:
    """Removes dead ops and tensors (to a fixpoint).

    Ops whose outputs all have no uses and are not subgraph outputs are
    removed (ops without outputs are kept), then tensors without producer and
    uses (subgraph I/O and block-wise quantization tensors in use are kept).

    Returns:
      Whether anything was removed.
    """
    return self._handle.dce()

  def replace_all_uses(
      self, old: Tensor, new: Tensor, *, exclude: Iterable[Op] = ()
  ) -> None:
    """Redirects every use of `old` (except by ops in `exclude`) to `new`.

    Occurrences of `old` in the subgraph outputs and in signature outputs are
    replaced too.

    Args:
      old: Tensor whose uses are replaced.
      new: Replacement tensor.
      exclude: Ops whose uses of `old` should not be replaced.
    """
    self._handle.replace_all_uses(
        old._handle,  # pylint: disable=protected-access
        new._handle,  # pylint: disable=protected-access
        [op._handle for op in exclude],  # pylint: disable=protected-access
    )

  def infer_shapes(self, *, validation_only: bool = False) -> None:
    """Runs shape inference on this subgraph (see `Model.infer_shapes`)."""
    try:
      self._handle.infer_shapes(validation_only)
    except ShapeInferenceError as e:
      _wrap_failing_op(e)
      raise

  def outline_to_dispatch_op(self) -> Op:
    """Replaces all ops by one dispatch custom op.

    The dispatch op takes the subgraph inputs and produces the subgraph
    outputs; intermediate tensors and constants left unused are removed.
    Attach the compiled bytecode with `Model.attach_asset`.

    Each subgraph output position becomes a distinct dispatch op output. The
    first occurrence of a tensor produced by an op is kept. An output that is
    not produced by an op (a constant, or a subgraph input forwarded to an
    output) and every repeated occurrence of a tensor are replaced by a fresh
    tensor (same name, shape, element type and quantization) produced by the
    dispatch op; the subgraph outputs and, by position, the outputs of the
    subgraph's signatures are remapped to it. Signature names are kept, except
    in the placeholder signature (`DEFAULT_SIGNATURE_KEY`), where a repeated
    output name becomes "<name>_<position>".
    Forwarded inputs remain subgraph (and dispatch op) inputs.

    Returns:
      The dispatch op.
    """
    return Op(self._handle.outline_to_dispatch_op())

  def __repr__(self) -> str:
    return (
        f"<litert_model.Subgraph {self.index} name={self.name!r}"
        f" ops={self._handle.num_ops()}>"
    )


class Op(_Handle):
  """An op (node) of a subgraph."""

  __slots__ = ()

  @property
  def subgraph(self) -> Subgraph:
    return Subgraph(self._handle.subgraph())

  @property
  def code(self) -> OpCode:
    return OpCode(self._handle.code())

  @property
  def name(self) -> str:
    """Op code name, e.g. "CONV_2D"."""
    return self.code.name

  @property
  def index(self) -> int:
    """Position in `Subgraph.ops`."""
    return self._handle.index()

  @property
  def custom_code(self) -> str | None:
    """Custom code of CUSTOM ops, else None."""
    return self._handle.custom_code()

  @property
  def custom_options(self) -> bytes:
    return self._handle.custom_options()

  def set_custom_options(
      self, data: bytes | bytearray | memoryview[int]
  ) -> None:
    self._handle.set_custom_options(data)

  @property
  def inputs(self) -> list[Tensor]:
    """Input tensors (never None: omitted optional inputs are not kept)."""
    return [Tensor(h) for h in self._handle.inputs()]

  @property
  def outputs(self) -> list[Tensor]:
    return [Tensor(h) for h in self._handle.outputs()]

  def set_input(self, index: int, tensor: Tensor) -> None:
    """Replaces input `index` (the tensor must be produced before this op)."""
    self._handle.set_input(index, tensor._handle)  # pylint: disable=protected-access

  def set_output(self, index: int, tensor: Tensor) -> None:
    """Replaces output `index`; the previous output loses its producer."""
    self._handle.set_output(index, tensor._handle)  # pylint: disable=protected-access

  @property
  def builtin_options(self) -> object | None:
    """A fresh schema_fb `*OptionsT` copy of the builtin options, or None.

    Covers both `BuiltinOptions` and `BuiltinOptions2` (e.g.
    `StableHLOCompositeOptionsT`). Mutating the returned object does not change
    the op; call `set_builtin_options`.
    """
    types = self._handle.builtin_options_types()
    if not any(types):
      return None
    return _unpack_options(self._handle.builtin_options_bytes(), types)

  def set_builtin_options(self, options: object | None) -> None:
    """Replaces the builtin options (None clears them).

    Args:
      options: A schema_fb `*OptionsT` object matching the op, or None.

    Raises:
      TypeError: If `options` is not a schema_fb options type or does not match
        the op code (or the type of the op's current options).
    """
    if options is None:
      self._handle.set_builtin_options_bytes(_pack_options(1, 0, None))
      return
    union, type_tag = _check_options_for_op(
        self.code, self._handle.builtin_options_types(), options
    )
    self._handle.set_builtin_options_bytes(
        _pack_options(union, type_tag, options)
    )

  @property
  def is_dispatch_op(self) -> bool:
    return self._handle.is_dispatch_op()

  def dispatch_options(self) -> DispatchOpOptions | None:
    """Returns the dispatch options of a dispatch op of a serialized model.

    Returns:
      The `DispatchOpOptions` for a serialized dispatch op, or None for other
      ops and for dispatch ops that were not serialized yet.
    """
    options = self._handle.dispatch_options()
    if options is None:
      return None
    return DispatchOpOptions(*options)

  def __repr__(self) -> str:
    return f"<litert_model.Op {self.name} index={self.index}>"


class Tensor(_Handle):
  """A tensor (value) of a subgraph."""

  __slots__ = ()

  @property
  def subgraph(self) -> Subgraph:
    return Subgraph(self._handle.subgraph())

  @property
  def index(self) -> int:
    """Position in `Subgraph.tensors`."""
    return self._handle.index()

  @property
  def name(self) -> str:
    return self._handle.name()

  def set_name(self, name: str) -> None:
    self._handle.set_name(name)

  @property
  def is_ranked(self) -> bool:
    return self._handle.is_ranked()

  @property
  def shape(self) -> tuple[int, ...] | None:
    """Dimensions (-1 for dynamic ones, () for scalars); None if unranked."""
    shape = self._handle.shape()
    return None if shape is None else tuple(shape)

  def set_shape(self, shape: Sequence[int]) -> None:
    """Sets the dimensions (keeps the element type; makes the tensor ranked)."""
    self._handle.set_shape(list(shape))

  @property
  def element_type(self) -> ElementType:
    return ElementType(self._handle.element_type())

  def set_element_type(self, element_type: ElementType) -> None:
    self._handle.set_element_type(int(ElementType(element_type)))

  @property
  def quantization(self) -> Quantization:
    """Quantization parameters of the tensor, or None if unquantized."""
    params = self._handle.quantization()
    if params is None:
      return None
    kind = params[0]
    if kind == "per_tensor":
      return PerTensorQuantization(scale=params[1], zero_point=params[2])
    if kind == "per_channel":
      return PerChannelQuantization(
          scales=np.array(params[1], dtype=np.float32),
          zero_points=np.array(params[2], dtype=np.int64),
          quantized_dimension=params[3],
      )
    return BlockWiseQuantization(
        scales=Tensor(params[1]),
        zero_points=None if params[2] is None else Tensor(params[2]),
        block_size=params[3],
    )

  def set_quantization(self, quantization: Quantization) -> None:
    """Replaces the quantization parameters (None clears them)."""
    if quantization is None:
      self._handle.clear_quantization()
    elif isinstance(quantization, PerTensorQuantization):
      self._handle.set_per_tensor_quantization(
          float(quantization.scale), int(quantization.zero_point)
      )
    elif isinstance(quantization, PerChannelQuantization):
      self._handle.set_per_channel_quantization(
          quantization.scales.tolist(),
          quantization.zero_points.tolist(),
          int(quantization.quantized_dimension),
      )
    elif isinstance(quantization, BlockWiseQuantization):
      zero_points = quantization.zero_points
      self._handle.set_block_wise_quantization(
          quantization.scales._handle,  # pylint: disable=protected-access
          None if zero_points is None else zero_points._handle,  # pylint: disable=protected-access
          int(quantization.block_size),
      )
    else:
      raise TypeError(f"unsupported quantization {type(quantization).__name__}")

  @property
  def is_constant(self) -> bool:
    return self._handle.is_constant()

  @property
  def is_subgraph_input(self) -> bool:
    return self._handle.is_subgraph_input()

  @property
  def is_subgraph_output(self) -> bool:
    return self._handle.is_subgraph_output()

  @property
  def weights_bytes(self) -> memoryview[int] | None:
    """Read-only view of the constant data, or None.

    The view is zero-copy when the data lives in the flatbuffer the model was
    loaded from or was set with `set_weights`; otherwise it is a copy.
    """
    return self._handle.weights_buffer()

  @property
  def weights(self) -> np.ndarray | None:
    """Read-only numpy view of the constant data, or None.

    Zero-copy in the same cases as `weights_bytes`. The array has the element
    type's dtype and the tensor shape. Element types without a numpy dtype
    (sub-byte, bfloat16, 8-bit floats, strings) give the raw bytes as a 1-D
    uint8 array.
    """
    buffer = self._handle.weights_buffer()
    if buffer is None:
      return None
    dtype = self.element_type.numpy_dtype
    if dtype is None or len(buffer) % dtype.itemsize:
      return np.frombuffer(buffer, dtype=np.uint8)
    array = np.frombuffer(buffer, dtype=dtype)
    shape = self.shape
    if (
        shape is not None
        and all(d >= 0 for d in shape)
        and math.prod(shape) == array.size
    ):
      array = array.reshape(shape)
    return array

  def set_weights(
      self, data: bytes | bytearray | memoryview[int] | np.ndarray
  ) -> None:
    """Replaces the constant data (copied into a model owned buffer).

    Every copy, including the ones replaced by later calls, is kept until the
    model is destroyed so that existing `weights` views stay valid.

    Args:
      data: Raw bytes, or a numpy array whose dtype matches the element type
        (raw uint8/int8 bytes for types without numpy dtype).

    Raises:
      TypeError: If the numpy dtype does not match the element type.
      ValueError: If the size does not match the tensor type, or the tensor is
        produced by an op or is a subgraph input.
    """
    if isinstance(data, np.ndarray):
      expected = self.element_type.numpy_dtype
      allowed = (
          (expected,)
          if expected is not None
          else (np.dtype(np.uint8), np.dtype(np.int8))
      )
      if data.dtype not in allowed:
        raise TypeError(
            f"weights dtype {data.dtype} does not match element type"
            f" {self.element_type.name}"
        )
      data = np.ascontiguousarray(data)
    self._handle.set_weights(data)

  @property
  def defining_op(self) -> Op | None:
    """The op producing this tensor, or None."""
    op = self._handle.defining_op()
    return None if op is None else Op(op)

  @property
  def defining_op_output_index(self) -> int:
    """Output index in `defining_op`, or -1 if there is no producer."""
    return self._handle.defining_op_output_index()

  @property
  def users(self) -> list[tuple[Op, int]]:
    """(consumer op, input index) pairs, in C++ use order."""
    return [(Op(op), index) for op, index in self._handle.users()]

  def __repr__(self) -> str:
    return (
        f"<litert_model.Tensor {self.name!r} {self.element_type.name}"
        f" {self.shape}>"
    )


class Signature(_Handle):
  """A named entry point of a model."""

  __slots__ = ()

  @property
  def key(self) -> str:
    return self._handle.key()

  @property
  def is_placeholder(self) -> bool:
    """True for the default signature the loader adds to unsigned models.

    Placeholder signatures are serialized like any other signature.
    """
    return self.key == DEFAULT_SIGNATURE_KEY

  @property
  def subgraph(self) -> Subgraph:
    return Subgraph(self._handle.subgraph())

  @property
  def input_names(self) -> list[str]:
    return self._handle.input_names()

  @property
  def output_names(self) -> list[str]:
    return self._handle.output_names()

  @property
  def inputs(self) -> list[Tensor]:
    return [Tensor(h) for h in self._handle.inputs()]

  @property
  def outputs(self) -> list[Tensor]:
    return [Tensor(h) for h in self._handle.outputs()]

  def get_input(self, name: str) -> Tensor:
    """Returns the input tensor named `name` (KeyError if missing)."""
    for input_name, tensor in zip(self.input_names, self.inputs, strict=True):
      if input_name == name:
        return tensor
    raise KeyError(name)

  def get_output(self, name: str) -> Tensor:
    """Returns the output tensor named `name` (KeyError if missing)."""
    for output_name, tensor in zip(
        self.output_names, self.outputs, strict=True
    ):
      if output_name == name:
        return tensor
    raise KeyError(name)

  def __repr__(self) -> str:
    return (
        f"<litert_model.Signature {self.key!r} inputs={self.input_names}"
        f" outputs={self.output_names}>"
    )
