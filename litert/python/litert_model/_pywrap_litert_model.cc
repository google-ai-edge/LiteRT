// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// pybind11 bindings of the C++ LiteRT IR (`LiteRtModelT`). The public Python
// API is `litert_model.py`; this module only exposes thin handles. See
// `pywrap_handles.h` for the ownership model.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/build_stamp.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/model/shape_inference.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/python/litert_model/litert_model_lib.h"
#include "litert/python/litert_model/pywrap_handles.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11

namespace py = pybind11;

namespace litert::litert_model {
namespace {

using ::litert::internal::Dims;
using ::litert::internal::FindInput;
using ::litert::internal::kLiteRtBuildStampKey;
using ::litert::internal::ShapeInferenceEngine;

// LINT.IfChange(element_types)
#define LITERT_MODEL_ELEMENT_TYPES(X)  \
  X(kLiteRtElementTypeNone)         \
  X(kLiteRtElementTypeBool)         \
  X(kLiteRtElementTypeInt2)         \
  X(kLiteRtElementTypeInt4)         \
  X(kLiteRtElementTypeInt8)         \
  X(kLiteRtElementTypeInt16)        \
  X(kLiteRtElementTypeInt32)        \
  X(kLiteRtElementTypeInt64)        \
  X(kLiteRtElementTypeUInt4)        \
  X(kLiteRtElementTypeUInt8)        \
  X(kLiteRtElementTypeUInt16)       \
  X(kLiteRtElementTypeUInt32)       \
  X(kLiteRtElementTypeUInt64)       \
  X(kLiteRtElementTypeFloat8E4M3FN) \
  X(kLiteRtElementTypeFloat8E5M2)   \
  X(kLiteRtElementTypeFloat16)      \
  X(kLiteRtElementTypeBFloat16)     \
  X(kLiteRtElementTypeFloat32)      \
  X(kLiteRtElementTypeFloat64)      \
  X(kLiteRtElementTypeComplex64)    \
  X(kLiteRtElementTypeComplex128)   \
  X(kLiteRtElementTypeTfResource)   \
  X(kLiteRtElementTypeTfString)     \
  X(kLiteRtElementTypeTfVariant)
// LINT.ThenChange(../../c/litert_model_types.h:element_types)

// LINT.IfChange(op_codes)
#define LITERT_MODEL_OP_CODES(X)                        \
  X(kLiteRtOpCodeTflAdd)                             \
  X(kLiteRtOpCodeTflAveragePool2d)                   \
  X(kLiteRtOpCodeTflConcatenation)                   \
  X(kLiteRtOpCodeTflConv2d)                          \
  X(kLiteRtOpCodeTflDepthwiseConv2d)                 \
  X(kLiteRtOpCodeTflDepthToSpace)                    \
  X(kLiteRtOpCodeTflDequantize)                      \
  X(kLiteRtOpCodeTflEmbeddingLookup)                 \
  X(kLiteRtOpCodeTflFloor)                           \
  X(kLiteRtOpCodeTflFullyConnected)                  \
  X(kLiteRtOpCodeTflHashtableLookup)                 \
  X(kLiteRtOpCodeTflL2Normalization)                 \
  X(kLiteRtOpCodeTflL2Pool2d)                        \
  X(kLiteRtOpCodeTflLocalResponseNormalization)      \
  X(kLiteRtOpCodeTflLogistic)                        \
  X(kLiteRtOpCodeTflLshProjection)                   \
  X(kLiteRtOpCodeTflLstm)                            \
  X(kLiteRtOpCodeTflMaxPool2d)                       \
  X(kLiteRtOpCodeTflMul)                             \
  X(kLiteRtOpCodeTflRelu)                            \
  X(kLiteRtOpCodeTflReluN1To1)                       \
  X(kLiteRtOpCodeTflRelu6)                           \
  X(kLiteRtOpCodeTflReshape)                         \
  X(kLiteRtOpCodeTflResizeBilinear)                  \
  X(kLiteRtOpCodeTflRnn)                             \
  X(kLiteRtOpCodeTflSoftmax)                         \
  X(kLiteRtOpCodeTflSpaceToDepth)                    \
  X(kLiteRtOpCodeTflSvdf)                            \
  X(kLiteRtOpCodeTflTanh)                            \
  X(kLiteRtOpCodeTflConcatEmbeddings)                \
  X(kLiteRtOpCodeTflSkipGram)                        \
  X(kLiteRtOpCodeTflCall)                            \
  X(kLiteRtOpCodeTflCustom)                          \
  X(kLiteRtOpCodeTflEmbeddingLookupSparse)           \
  X(kLiteRtOpCodeTflPad)                             \
  X(kLiteRtOpCodeTflUnidirectionalSequenceRnn)       \
  X(kLiteRtOpCodeTflGather)                          \
  X(kLiteRtOpCodeTflBatchToSpaceNd)                  \
  X(kLiteRtOpCodeTflSpaceToBatchNd)                  \
  X(kLiteRtOpCodeTflTranspose)                       \
  X(kLiteRtOpCodeTflMean)                            \
  X(kLiteRtOpCodeTflSub)                             \
  X(kLiteRtOpCodeTflDiv)                             \
  X(kLiteRtOpCodeTflSqueeze)                         \
  X(kLiteRtOpCodeTflUnidirectionalSequenceLstm)      \
  X(kLiteRtOpCodeTflStridedSlice)                    \
  X(kLiteRtOpCodeTflBidirectionalSequenceRnn)        \
  X(kLiteRtOpCodeTflExp)                             \
  X(kLiteRtOpCodeTflTopkV2)                          \
  X(kLiteRtOpCodeTflSplit)                           \
  X(kLiteRtOpCodeTflLogSoftmax)                      \
  X(kLiteRtOpCodeTflDelegate)                        \
  X(kLiteRtOpCodeTflBidirectionalSequenceLstm)       \
  X(kLiteRtOpCodeTflCast)                            \
  X(kLiteRtOpCodeTflPrelu)                           \
  X(kLiteRtOpCodeTflMaximum)                         \
  X(kLiteRtOpCodeTflArgMax)                          \
  X(kLiteRtOpCodeTflMinimum)                         \
  X(kLiteRtOpCodeTflLess)                            \
  X(kLiteRtOpCodeTflNeg)                             \
  X(kLiteRtOpCodeTflPadv2)                           \
  X(kLiteRtOpCodeTflGreater)                         \
  X(kLiteRtOpCodeTflGreaterEqual)                    \
  X(kLiteRtOpCodeTflLessEqual)                       \
  X(kLiteRtOpCodeTflSelect)                          \
  X(kLiteRtOpCodeTflSlice)                           \
  X(kLiteRtOpCodeTflSin)                             \
  X(kLiteRtOpCodeTflTransposeConv)                   \
  X(kLiteRtOpCodeTflSparseToDense)                   \
  X(kLiteRtOpCodeTflTile)                            \
  X(kLiteRtOpCodeTflExpandDims)                      \
  X(kLiteRtOpCodeTflEqual)                           \
  X(kLiteRtOpCodeTflNotEqual)                        \
  X(kLiteRtOpCodeTflLog)                             \
  X(kLiteRtOpCodeTflSum)                             \
  X(kLiteRtOpCodeTflSqrt)                            \
  X(kLiteRtOpCodeTflRsqrt)                           \
  X(kLiteRtOpCodeTflShape)                           \
  X(kLiteRtOpCodeTflPow)                             \
  X(kLiteRtOpCodeTflArgMin)                          \
  X(kLiteRtOpCodeTflFakeQuant)                       \
  X(kLiteRtOpCodeTflReduceProd)                      \
  X(kLiteRtOpCodeTflReduceMax)                       \
  X(kLiteRtOpCodeTflPack)                            \
  X(kLiteRtOpCodeTflLogicalOr)                       \
  X(kLiteRtOpCodeTflOneHot)                          \
  X(kLiteRtOpCodeTflLogicalAnd)                      \
  X(kLiteRtOpCodeTflLogicalNot)                      \
  X(kLiteRtOpCodeTflUnpack)                          \
  X(kLiteRtOpCodeTflReduceMin)                       \
  X(kLiteRtOpCodeTflFloorDiv)                        \
  X(kLiteRtOpCodeTflReduceAny)                       \
  X(kLiteRtOpCodeTflSquare)                          \
  X(kLiteRtOpCodeTflZerosLike)                       \
  X(kLiteRtOpCodeTflFill)                            \
  X(kLiteRtOpCodeTflFloorMod)                        \
  X(kLiteRtOpCodeTflRange)                           \
  X(kLiteRtOpCodeTflResizeNearestNeighbor)           \
  X(kLiteRtOpCodeTflLeakyRelu)                       \
  X(kLiteRtOpCodeTflSquaredDifference)               \
  X(kLiteRtOpCodeTflMirrorPad)                       \
  X(kLiteRtOpCodeTflAbs)                             \
  X(kLiteRtOpCodeTflSplitV)                          \
  X(kLiteRtOpCodeTflUnique)                          \
  X(kLiteRtOpCodeTflCeil)                            \
  X(kLiteRtOpCodeTflReverseV2)                       \
  X(kLiteRtOpCodeTflAddN)                            \
  X(kLiteRtOpCodeTflGatherNd)                        \
  X(kLiteRtOpCodeTflCos)                             \
  X(kLiteRtOpCodeTflWhere)                           \
  X(kLiteRtOpCodeTflRank)                            \
  X(kLiteRtOpCodeTflElu)                             \
  X(kLiteRtOpCodeTflReverseSequence)                 \
  X(kLiteRtOpCodeTflMatrixDiag)                      \
  X(kLiteRtOpCodeTflQuantize)                        \
  X(kLiteRtOpCodeTflMatrixSetDiag)                   \
  X(kLiteRtOpCodeTflRound)                           \
  X(kLiteRtOpCodeTflHardSwish)                       \
  X(kLiteRtOpCodeTflIf)                              \
  X(kLiteRtOpCodeTflWhile)                           \
  X(kLiteRtOpCodeTflNonMaxSuppressionV4)             \
  X(kLiteRtOpCodeTflNonMaxSuppressionV5)             \
  X(kLiteRtOpCodeTflScatterNd)                       \
  X(kLiteRtOpCodeTflSelectV2)                        \
  X(kLiteRtOpCodeTflDensify)                         \
  X(kLiteRtOpCodeTflSegmentSum)                      \
  X(kLiteRtOpCodeTflBatchMatmul)                     \
  X(kLiteRtOpCodeTflPlaceholderForGreaterOpCodeTfls) \
  X(kLiteRtOpCodeTflCumsum)                          \
  X(kLiteRtOpCodeTflCallOnce)                        \
  X(kLiteRtOpCodeTflBroadcastTo)                     \
  X(kLiteRtOpCodeTflRfft2d)                          \
  X(kLiteRtOpCodeTflConv3d)                          \
  X(kLiteRtOpCodeTflImag)                            \
  X(kLiteRtOpCodeTflReal)                            \
  X(kLiteRtOpCodeTflComplexAbs)                      \
  X(kLiteRtOpCodeTflHashtable)                       \
  X(kLiteRtOpCodeTflHashtableFind)                   \
  X(kLiteRtOpCodeTflHashtableImport)                 \
  X(kLiteRtOpCodeTflHashtableSize)                   \
  X(kLiteRtOpCodeTflReduceAll)                       \
  X(kLiteRtOpCodeTflConv3dTranspose)                 \
  X(kLiteRtOpCodeTflVarHandle)                       \
  X(kLiteRtOpCodeTflReadVariable)                    \
  X(kLiteRtOpCodeTflAssignVariable)                  \
  X(kLiteRtOpCodeTflBroadcastArgs)                   \
  X(kLiteRtOpCodeTflRandomStandardNormal)            \
  X(kLiteRtOpCodeTflBucketize)                       \
  X(kLiteRtOpCodeTflRandomUniform)                   \
  X(kLiteRtOpCodeTflMultinomial)                     \
  X(kLiteRtOpCodeTflGelu)                            \
  X(kLiteRtOpCodeTflDynamicUpdateSlice)              \
  X(kLiteRtOpCodeTflRelu0To1)                        \
  X(kLiteRtOpCodeTflUnsortedSegmentProd)             \
  X(kLiteRtOpCodeTflUnsortedSegmentMax)              \
  X(kLiteRtOpCodeTflUnsortedSegmentSum)              \
  X(kLiteRtOpCodeTflAtan2)                           \
  X(kLiteRtOpCodeTflUnsortedSegmentMin)              \
  X(kLiteRtOpCodeTflSign)                            \
  X(kLiteRtOpCodeTflBitcast)                         \
  X(kLiteRtOpCodeTflBitwiseXor)                      \
  X(kLiteRtOpCodeTflRightShift)                      \
  X(kLiteRtOpCodeShloLogistic)                       \
  X(kLiteRtOpCodeShloAdd)                            \
  X(kLiteRtOpCodeShloDivide)                         \
  X(kLiteRtOpCodeShloMultiply)                       \
  X(kLiteRtOpCodeShloMaximum)                        \
  X(kLiteRtOpCodeShloReshape)                        \
  X(kLiteRtOpCodeShloClamp)                          \
  X(kLiteRtOpCodeShloConcatenate)                    \
  X(kLiteRtOpCodeShloBroadcastInDim)                 \
  X(kLiteRtOpCodeShloConvolution)                    \
  X(kLiteRtOpCodeShloSlice)                          \
  X(kLiteRtOpCodeShloCustomCall)                     \
  X(kLiteRtOpCodeShloReduce)                         \
  X(kLiteRtOpCodeShloAbs)                            \
  X(kLiteRtOpCodeShloAnd)                            \
  X(kLiteRtOpCodeShloCosine)                         \
  X(kLiteRtOpCodeShloExponential)                    \
  X(kLiteRtOpCodeShloFloor)                          \
  X(kLiteRtOpCodeShloLog)                            \
  X(kLiteRtOpCodeShloMinimum)                        \
  X(kLiteRtOpCodeShloNegate)                         \
  X(kLiteRtOpCodeShloOr)                             \
  X(kLiteRtOpCodeShloPower)                          \
  X(kLiteRtOpCodeShloRemainder)                      \
  X(kLiteRtOpCodeShloRsqrt)                          \
  X(kLiteRtOpCodeShloSelect)                         \
  X(kLiteRtOpCodeShloSubtract)                       \
  X(kLiteRtOpCodeShloTanh)                           \
  X(kLiteRtOpCodeShloScatter)                        \
  X(kLiteRtOpCodeShloCompare)                        \
  X(kLiteRtOpCodeShloConvert)                        \
  X(kLiteRtOpCodeShloDynamicSlice)                   \
  X(kLiteRtOpCodeShloDynamicUpdateSlice)             \
  X(kLiteRtOpCodeShloPad)                            \
  X(kLiteRtOpCodeShloIota)                           \
  X(kLiteRtOpCodeShloGeneral)                        \
  X(kLiteRtOpCodeShloWindow)                         \
  X(kLiteRtOpCodeShloSort)                           \
  X(kLiteRtOpCodeShloWhile)                          \
  X(kLiteRtOpCodeShloGather)                         \
  X(kLiteRtOpCodeShloTranspose)                      \
  X(kLiteRtOpCodeTflDilate)                          \
  X(kLiteRtOpCodeShloRngBitGenerator)                \
  X(kLiteRtOpCodeTflReduceWindow)                    \
  X(kLiteRtOpCodeShloComposite)                      \
  X(kLiteRtOpCodeShloShiftLeft)                      \
  X(kLiteRtOpCodeShloCbrt)                           \
  X(kLiteRtOpCodeShloCase)
// LINT.ThenChange(../../c/litert_op_code.h:op_codes)

#define LITERT_MODEL_NAME_VALUE(name) {#name, static_cast<int>(name)},

const std::vector<std::pair<std::string, int>>& ElementTypeValues() {
  static const auto* const kValues =
      new std::vector<std::pair<std::string, int>>{
          LITERT_MODEL_ELEMENT_TYPES(LITERT_MODEL_NAME_VALUE)};
  return *kValues;
}

const std::vector<std::pair<std::string, int>>& OpCodeValues() {
  static const auto* const kValues =
      new std::vector<std::pair<std::string, int>>{
          LITERT_MODEL_OP_CODES(LITERT_MODEL_NAME_VALUE)};
  return *kValues;
}

#undef LITERT_MODEL_NAME_VALUE

bool IsKnown(const std::vector<std::pair<std::string, int>>& values,
             int value) {
  for (const auto& [unused_name, known] : values) {
    if (known == value) return true;
  }
  return false;
}

LiteRtElementType CheckedElementType(int value) {
  if (!IsKnown(ElementTypeValues(), value)) {
    throw py::value_error(absl::StrFormat("unknown element type %d", value));
  }
  return static_cast<LiteRtElementType>(value);
}

LiteRtOpCode CheckedOpCode(int value) {
  if (!IsKnown(OpCodeValues(), value)) {
    throw py::value_error(absl::StrFormat("unknown op code %d", value));
  }
  return static_cast<LiteRtOpCode>(value);
}

std::vector<int32_t> CheckedDims(const std::vector<int64_t>& shape) {
  if (shape.size() > LITERT_TENSOR_MAX_RANK) {
    throw py::value_error(absl::StrFormat("rank %d exceeds the maximum of %d",
                                          shape.size(),
                                          LITERT_TENSOR_MAX_RANK));
  }
  std::vector<int32_t> dims;
  dims.reserve(shape.size());
  for (int64_t dim : shape) {
    if (dim < -1 || dim > std::numeric_limits<int32_t>::max()) {
      throw py::value_error(absl::StrFormat(
          "invalid dimension %d (expected -1 for dynamic or a non-negative "
          "size)",
          dim));
    }
    dims.push_back(static_cast<int32_t>(dim));
  }
  return dims;
}

// Range checks are left to the IR helpers; only negative indices are rejected
// here, before the conversion to size_t.
size_t NonNegativeIndex(int64_t index, const char* what) {
  if (index < 0) {
    throw py::index_error(
        absl::StrFormat("%s index %d is negative", what, index));
  }
  return static_cast<size_t>(index);
}

// Copies a C-contiguous Python buffer into an owned buffer.
OwningBufferRef<uint8_t> CopyBuffer(const py::buffer& data) {
  Py_buffer view;
  if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_C_CONTIGUOUS) != 0) {
    throw py::error_already_set();
  }
  OwningBufferRef<uint8_t> copy(static_cast<const uint8_t*>(view.buf),
                                static_cast<size_t>(view.len));
  PyBuffer_Release(&view);
  return copy;
}

py::bytes ToBytes(BufferRef<uint8_t> buffer) {
  return py::bytes(reinterpret_cast<const char*>(buffer.Data()), buffer.Size());
}

// Graph edit preconditions (bad index, wrong subgraph, topological order...)
// are caller errors: raise IndexError/ValueError. Other failures raise
// LiteRtError.
[[noreturn]] void ThrowEditError(const Error& error) {
  switch (StatusOf(error)) {
    case kLiteRtStatusErrorIndexOOB:
      throw py::index_error(CleanErrorMessage(error.Message()));
    case kLiteRtStatusErrorInvalidArgument:
    case kLiteRtStatusErrorInvalidGraphInvariant:
    case kLiteRtStatusErrorNotFound:
    case kLiteRtStatusErrorAlreadyExists:
      throw py::value_error(CleanErrorMessage(error.Message()));
    default:
      throw IrError(StatusOf(error), error.Message());
  }
}

void CheckEdit(const Expected<void>& result) {
  if (!result) ThrowEditError(result.Error());
}

template <class T>
T EditValueOrThrow(Expected<T>&& result) {
  if (!result) ThrowEditError(result.Error());
  return std::move(*result);
}

void CheckSameModel(const State& expected, const State& actual) {
  if (expected.get() != actual.get()) {
    throw py::value_error("the handle belongs to a different model");
  }
}

void CheckSameSubgraph(const PySubgraph& subgraph, const PyTensor& tensor) {
  CheckSameModel(subgraph.state, tensor.parent.state);
  if (tensor.parent.subgraph != subgraph.subgraph) {
    throw py::value_error("the tensor belongs to a different subgraph");
  }
}

void CheckSameSubgraph(const PySubgraph& subgraph, const PyOp& op) {
  CheckSameModel(subgraph.state, op.parent.state);
  if (op.parent.subgraph != subgraph.subgraph) {
    throw py::value_error("the op belongs to a different subgraph");
  }
}

std::vector<LiteRtTensorT*> RawTensors(const PySubgraph& subgraph,
                                       const std::vector<PyTensor>& tensors) {
  std::vector<LiteRtTensorT*> raw;
  raw.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    CheckSameSubgraph(subgraph, tensor);
    raw.push_back(&tensor.Get());
  }
  return raw;
}

template <class Container>
py::list TensorList(const PySubgraph& parent, const Container& tensors) {
  py::list result;
  for (LiteRtTensorT* tensor : tensors) {
    result.append(PyTensor::Make(parent, tensor));
  }
  return result;
}

LiteRtElementType GetElementType(const LiteRtTensorT& tensor) {
  if (tensor.Type().first == kLiteRtRankedTensorType) {
    return tensor.Type().second.ranked_tensor_type.element_type;
  }
  return tensor.Type().second.unranked_tensor_type.element_type;
}

std::optional<std::vector<int32_t>> GetShape(const LiteRtTensorT& tensor) {
  if (tensor.Type().first != kLiteRtRankedTensorType) {
    return std::nullopt;
  }
  const auto& layout = tensor.Type().second.ranked_tensor_type.layout;
  return std::vector<int32_t>(layout.dimensions,
                              layout.dimensions + layout.rank);
}

// Returns the byte size of the constant data of a tensor of the given type and
// shape, or std::nullopt if it has no fixed size. Throws if it overflows.
std::optional<size_t> CheckedDenseByteSize(LiteRtElementType element_type,
                                           absl::Span<const int32_t> dims) {
  auto size = DenseByteSize(element_type, dims);
  if (!size) {
    throw py::value_error("the tensor type is too large to hold constant data");
  }
  return *size;
}

// Checks that the weights of `tensor` (if any) fit the given type and shape.
void CheckWeightsFit(const LiteRtTensorT& tensor,
                     LiteRtElementType element_type,
                     absl::Span<const int32_t> dims) {
  const size_t size = tensor.Weights().Buffer().Size();
  if (size == 0) return;
  const auto expected = CheckedDenseByteSize(element_type, dims);
  if (expected.has_value() && *expected != size) {
    throw py::value_error(absl::StrFormat(
        "the tensor holds %d bytes of constant data but the new type needs %d",
        size, *expected));
  }
}

const char* StatusName(LiteRtStatus status) {
  const char* name = LiteRtGetStatusString(status);
  return name != nullptr ? name : "kLiteRtStatusErrorUnknown";
}

[[noreturn]] void ThrowShapeInferenceError(const State& state,
                                           LiteRtStatus status,
                                           LiteRtOp failing_op) {
  std::string message =
      absl::StrFormat("shape inference failed: %s", StatusName(status));
  std::shared_ptr<PyOp> op_handle;
  if (failing_op != nullptr) {
    LiteRtModelT& model = state->Model();
    if (LiteRtSubgraphT* subgraph = FindOwningSubgraph(model, *failing_op)) {
      op_handle = std::make_shared<PyOp>(
          PyOp::Make(PySubgraph::Make(state, subgraph), failing_op));
      absl::StrAppendFormat(&message, " at op %d (op code %d) of subgraph %d",
                            ValueOrThrow(FindOpIndex(*subgraph, *failing_op)),
                            static_cast<int>(failing_op->OpCode()),
                            ValueOrThrow(FindSubgraphIndex(model, *subgraph)));
    }
  }
  IrError error(status, std::move(message), ErrorKind::kShapeInference);
  error.set_op(std::move(op_handle));
  throw error;
}

// Read-only buffer exporting a view of model owned memory. Keeps the model
// state alive while any memoryview/numpy array refers to it.
struct WeightsView {
  State state;
  const uint8_t* data;
  size_t size;
};

//===----------------------------------------------------------------------===//
// Model.
//===----------------------------------------------------------------------===//

struct PyModel {
  State state;

  explicit PyModel(std::unique_ptr<LiteRtModelT> model)
      : state(std::make_shared<ModelState>(std::move(model))) {}

  LiteRtModelT& Get() const { return state->Model(); }

  static PyModel LoadFromFile(const std::string& path) {
    return PyModel(ValueOrThrow(internal::LoadModelFromFile(path)));
  }

  static PyModel LoadFromBytes(const py::buffer& data) {
    OwningBufferRef<uint8_t> copy = CopyBuffer(data);
    // `LoadModelFromBuffer(OwningBufferRef&&)` aborts on invalid input, so
    // verify the flatbuffer first.
    if (!internal::VerifyFlatbuffer(copy.Data(), copy.Size())) {
      throw IrError(kLiteRtStatusErrorInvalidFlatbuffer,
                    "the buffer is not a valid TFLite flatbuffer");
    }
    return PyModel(
        ValueOrThrow(internal::LoadModelFromBuffer(std::move(copy))));
  }

  static PyModel Create() { return PyModel(std::make_unique<LiteRtModelT>()); }

  size_t NumSubgraphs() const { return Get().NumSubgraphs(); }

  PySubgraph Subgraph(int64_t index) const {
    LiteRtModelT& model = Get();
    if (index < 0 || static_cast<size_t>(index) >= model.NumSubgraphs()) {
      throw py::index_error(
          absl::StrFormat("subgraph index %d out of range [0, %d)", index,
                          model.NumSubgraphs()));
    }
    return PySubgraph::Make(state, model.Subgraphs()[index]);
  }

  py::list Subgraphs() const {
    py::list result;
    for (LiteRtSubgraph subgraph : Get().Subgraphs()) {
      result.append(PySubgraph::Make(state, subgraph));
    }
    return result;
  }

  PySubgraph AddSubgraph(const std::string& name) const {
    LiteRtSubgraphT& subgraph = Get().EmplaceSubgraph();
    subgraph.SetName(name);
    return PySubgraph::Make(state, &subgraph);
  }

  size_t PruneUnreachableSubgraphs() const {
    LiteRtModelT& model = Get();
    absl::flat_hash_set<const void*> before;
    for (const LiteRtSubgraphT* subgraph : model.Subgraphs()) {
      before.insert(subgraph);
      for (const LiteRtOpT* op : subgraph->Ops()) before.insert(op);
      for (const LiteRtTensorT* tensor : subgraph->Tensors()) {
        before.insert(tensor);
      }
    }
    const size_t removed =
        ValueOrThrow(litert_model::PruneUnreachableSubgraphs(model));
    if (removed > 0) {
      for (const LiteRtSubgraphT* subgraph : model.Subgraphs()) {
        before.erase(subgraph);
        for (const LiteRtOpT* op : subgraph->Ops()) before.erase(op);
        for (const LiteRtTensorT* tensor : subgraph->Tensors()) {
          before.erase(tensor);
        }
      }
      for (const void* freed : before) state->Forget(freed);
    }
    return removed;
  }

  py::list Signatures() const {
    py::list result;
    for (LiteRtSignature signature : Get().Signatures()) {
      result.append(PySignature::Make(state, signature));
    }
    return result;
  }

  PySignature FindSignature(const std::string& key) const {
    auto signature = Get().FindSignature(key);
    if (!signature) {
      throw py::key_error(key);
    }
    return PySignature::Make(state, &signature->get());
  }

  PySignature AddSignature(
      const std::string& key, const PySubgraph& subgraph,
      const std::vector<std::pair<std::string, PyTensor>>& inputs,
      const std::vector<std::pair<std::string, PyTensor>>& outputs) const {
    CheckSameModel(state, subgraph.state);
    LiteRtSubgraphT& raw_subgraph = subgraph.Get();
    auto to_named = [&subgraph](const auto& entries) {
      std::vector<NamedTensor> named;
      for (const auto& [name, tensor] : entries) {
        CheckSameSubgraph(subgraph, tensor);
        named.emplace_back(name, &tensor.Get());
      }
      return named;
    };
    const auto named_inputs = to_named(inputs);
    const auto named_outputs = to_named(outputs);
    LiteRtSignatureT* signature = EditValueOrThrow(litert_model::AddSignature(
        Get(), key, raw_subgraph, named_inputs, named_outputs));
    return PySignature::Make(state, signature);
  }

  void RemoveSignature(const std::string& key) const {
    LiteRtModelT& model = Get();
    if (!model.FindSignature(key)) {
      throw py::key_error(key);
    }
    state->Forget(EditValueOrThrow(litert_model::RemoveSignature(model, key)));
  }

  std::vector<std::string> MetadataKeys() const {
    LiteRtModelT& model = Get();
    std::vector<std::string> keys;
    for (auto it = model.MetadataBegin(); it != model.MetadataEnd(); ++it) {
      keys.push_back(it->first);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
  }

  std::optional<py::bytes> Metadata(const std::string& key) const {
    auto buffer = Get().FindMetadata(key);
    if (!buffer) {
      return std::nullopt;
    }
    return ToBytes(*buffer);
  }

  void PushMetadata(const std::string& key, const py::buffer& data) const {
    OwningBufferRef<uint8_t> copy = CopyBuffer(data);
    const LiteRtStatus status = Get().PushMetadata(key, std::move(copy));
    if (status != kLiteRtStatusOk) {
      throw IrError(status,
                    absl::StrFormat("metadata '%s' already exists", key));
    }
  }

  void Validate() const {
    ThrowIfError(ValidateModel(Get()), ErrorKind::kValidation);
  }

  py::bytes Serialize(size_t bytecode_alignment) const {
    LiteRtModelT& model = Get();
    if (bytecode_alignment == 0 ||
        (bytecode_alignment & (bytecode_alignment - 1)) != 0) {
      throw py::value_error(
          absl::StrFormat("bytecode_alignment must be a power of two, got %d",
                          bytecode_alignment));
    }
    ThrowIfError(ValidateModel(model, ValidationLevel::kSerializable),
                 ErrorKind::kValidation);
    // `SerializeModel` takes the model by rvalue and leaves it unusable.
    state->MarkSerialized();
    auto serialized =
        internal::SerializeModel(std::move(model), bytecode_alignment);
    return ToBytes(ValueOrThrow(std::move(serialized)));
  }

  bool IsSerialized() const { return state->serialized(); }

  void InferShapes(bool validation_only) const {
    ShapeInferenceEngine engine(&Get());
    LiteRtOp failing_op = nullptr;
    const LiteRtStatus status =
        engine.InferShapes(validation_only, &failing_op);
    if (status != kLiteRtStatusOk) {
      ThrowShapeInferenceError(state, status, failing_op);
    }
  }

  PySubgraph SpecializeSubgraph(
      const PySubgraph& subgraph,
      const std::vector<std::vector<int64_t>>& input_shapes) const {
    CheckSameModel(state, subgraph.state);
    LiteRtModelT& model = Get();
    LiteRtSubgraphT& source = subgraph.Get();
    if (input_shapes.size() != source.NumInputs()) {
      throw py::value_error(absl::StrFormat("expected %d input shapes, got %d",
                                            source.NumInputs(),
                                            input_shapes.size()));
    }
    std::vector<Dims> dims;
    for (const auto& shape : input_shapes) {
      dims.push_back(CheckedDims(shape));
    }
    const size_t num_subgraphs = model.NumSubgraphs();
    ShapeInferenceEngine engine(&model);
    LiteRtSubgraphT* specialized = nullptr;
    const LiteRtStatus status =
        engine.SpecializeSubgraph(&source, absl::MakeSpan(dims), &specialized);
    if (status != kLiteRtStatusOk) {
      // Drop the partially built copy; no handle refers to it yet.
      model.ResizeSubgraphsDown(num_subgraphs);
      ThrowShapeInferenceError(state, status, nullptr);
    }
    return PySubgraph::Make(state, specialized);
  }

  uint32_t RegisterBuffer(const py::buffer& data) const {
    OwningBufferRef<uint8_t> copy = CopyBuffer(data);
    // See `ModelState::KeepBuffer` for why the buffer is not registered as
    // owned.
    return Get().Buffers()->RegisterNonOwnedBuffer(
        state->KeepBuffer(std::move(copy)));
  }

  void AttachAssetToOp(const PyOp& op, uint32_t buffer_id,
                       const std::string& name) const {
    CheckSameModel(state, op.parent.state);
    CheckEdit(AttachAsset(Get(), op.Get(), buffer_id, name));
  }

  void SetBuildStamp(const std::string& soc_manufacturer,
                     const std::string& soc_model) const {
    LiteRtModelT& model = Get();
    auto stamp =
        ValueOrThrow(internal::MakeBuildStamp(soc_manufacturer, soc_model));
    const LiteRtStatus status =
        model.PushMetadata(kLiteRtBuildStampKey, std::move(stamp));
    if (status != kLiteRtStatusOk) {
      throw IrError(status, "the model already has a build stamp");
    }
  }

  std::optional<std::tuple<std::string, std::string>> BuildStamp() const {
    const auto stamp = GetBuildStamp(Get());
    if (!stamp.has_value()) {
      return std::nullopt;
    }
    return std::make_tuple(std::string(stamp->soc_manufacturer),
                           std::string(stamp->soc_model));
  }
};

//===----------------------------------------------------------------------===//
// Subgraph.
//===----------------------------------------------------------------------===//

size_t SubgraphIndex(const PySubgraph& self) {
  return ValueOrThrow(FindSubgraphIndex(self.state->Model(), self.Get()));
}

PyTensor AddTensor(const PySubgraph& self, const std::vector<int64_t>& shape,
                   int element_type, const std::string& name) {
  LiteRtSubgraphT& subgraph = self.Get();
  const auto dims = CheckedDims(shape);
  const LiteRtElementType type = CheckedElementType(element_type);
  LiteRtTensorT& tensor = subgraph.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(type, dims));
  tensor.SetName(name);
  tensor.SetTensorIndex(static_cast<uint32_t>(subgraph.Tensors().size() - 1));
  return PyTensor::Make(self, &tensor);
}

PyOp AddOpToSubgraph(const PySubgraph& self, int code,
                     const std::vector<PyTensor>& inputs,
                     const std::vector<PyTensor>& outputs,
                     std::optional<int64_t> index,
                     std::optional<std::string> custom_code,
                     std::optional<py::bytes> custom_options) {
  LiteRtSubgraphT& subgraph = self.Get();
  AddOpOptions options;
  if (index.has_value()) {
    options.index = NonNegativeIndex(*index, "op");
  }
  options.custom_code = std::move(custom_code);
  if (custom_options.has_value()) {
    const std::string bytes = *custom_options;
    options.custom_options = std::vector<uint8_t>(bytes.begin(), bytes.end());
  }
  const auto raw_inputs = RawTensors(self, inputs);
  const auto raw_outputs = RawTensors(self, outputs);
  LiteRtOpT* op =
      EditValueOrThrow(AddOp(self.state->Model(), subgraph, CheckedOpCode(code),
                             raw_inputs, raw_outputs, options));
  return PyOp::Make(self, op);
}

void RemoveOpFromSubgraph(const PySubgraph& self, const PyOp& op) {
  CheckSameSubgraph(self, op);
  LiteRtOpT& raw_op = op.Get();
  CheckEdit(RemoveOp(self.state->Model(), self.Get(), raw_op));
  self.state->Forget(&raw_op);
}

void RemoveTensorFromSubgraph(const PySubgraph& self, const PyTensor& tensor) {
  CheckSameSubgraph(self, tensor);
  LiteRtTensorT& raw_tensor = tensor.Get();
  CheckEdit(RemoveTensor(self.Get(), raw_tensor));
  self.state->Forget(&raw_tensor);
}

bool Dce(const PySubgraph& self) {
  LiteRtSubgraphT& subgraph = self.Get();
  FreedObjectTracker tracker(*self.state, subgraph);
  return EliminateDeadCode(self.state->Model(), subgraph);
}

void ReplaceAllUsesInSubgraph(const PySubgraph& self,
                              const PyTensor& old_tensor,
                              const PyTensor& new_tensor,
                              const std::vector<PyOp>& exclude) {
  CheckSameSubgraph(self, old_tensor);
  CheckSameSubgraph(self, new_tensor);
  absl::flat_hash_set<const LiteRtOpT*> excluded;
  for (const auto& op : exclude) {
    CheckSameSubgraph(self, op);
    excluded.insert(&op.Get());
  }
  CheckEdit(ReplaceAllUses(self.state->Model(), self.Get(), old_tensor.Get(),
                           new_tensor.Get(), excluded));
}

void InferSubgraphShapes(const PySubgraph& self, bool validation_only) {
  LiteRtSubgraphT& subgraph = self.Get();
  ShapeInferenceEngine engine(&self.state->Model());
  LiteRtOp failing_op = nullptr;
  const LiteRtStatus status =
      engine.InferSubgraphShapes(&subgraph, validation_only, &failing_op);
  if (status != kLiteRtStatusOk) {
    ThrowShapeInferenceError(self.state, status, failing_op);
  }
}

PyOp OutlineToDispatchOp(const PySubgraph& self) {
  LiteRtSubgraphT& subgraph = self.Get();
  FreedObjectTracker tracker(*self.state, subgraph);
  LiteRtOpT* op =
      ValueOrThrow(OutlineSubgraphToDispatchOp(self.state->Model(), subgraph));
  tracker.Commit();
  return PyOp::Make(self, op);
}

//===----------------------------------------------------------------------===//
// Op.
//===----------------------------------------------------------------------===//

std::optional<std::string> CustomCode(const PyOp& self) {
  const auto custom_code = self.Get().CustomCode();
  if (!custom_code) {
    return std::nullopt;
  }
  return std::string(*custom_code);
}

void SetInput(const PyOp& self, int64_t index, const PyTensor& tensor) {
  CheckSameSubgraph(self.parent, tensor);
  CheckEdit(SetOpInput(self.parent.Get(), self.Get(),
                       NonNegativeIndex(index, "input"), tensor.Get()));
}

void SetOutput(const PyOp& self, int64_t index, const PyTensor& tensor) {
  CheckSameSubgraph(self.parent, tensor);
  CheckEdit(SetOpOutput(self.parent.Get(), self.Get(),
                        NonNegativeIndex(index, "output"), tensor.Get()));
}

std::optional<std::tuple<size_t, size_t, std::string>> DispatchOptions(
    const PyOp& self) {
  auto options = ReadDispatchOpOptions(self.Get());
  if (!options) {
    if (StatusOf(options.Error()) == kLiteRtStatusErrorNotFound) {
      return std::nullopt;
    }
    throw IrError(StatusOf(options.Error()), options.Error().Message());
  }
  return std::make_tuple(options->bytecode_size, options->bytecode_offset,
                         options->name);
}

//===----------------------------------------------------------------------===//
// Tensor.
//===----------------------------------------------------------------------===//

void SetShape(const PyTensor& self, const std::vector<int64_t>& shape) {
  LiteRtTensorT& tensor = self.Get();
  const auto dims = CheckedDims(shape);
  const LiteRtElementType element_type = GetElementType(tensor);
  CheckWeightsFit(tensor, element_type, dims);
  tensor.SetType(MakeRankedTensorType(element_type, dims));
}

void SetElementType(const PyTensor& self, int element_type) {
  LiteRtTensorT& tensor = self.Get();
  const LiteRtElementType type = CheckedElementType(element_type);
  if (tensor.Type().first == kLiteRtRankedTensorType) {
    const auto dims = *GetShape(tensor);
    CheckWeightsFit(tensor, type, dims);
    tensor.Type().second.ranked_tensor_type.element_type = type;
  } else {
    tensor.Type().second.unranked_tensor_type.element_type = type;
  }
}

py::object GetQuantization(const PyTensor& self) {
  const LiteRtTensorT& tensor = self.Get();
  const auto& [type, detail] = tensor.Qparams();
  switch (type) {
    case kLiteRtQuantizationPerTensor:
      return py::make_tuple("per_tensor", detail.per_tensor.scale,
                            detail.per_tensor.zero_point);
    case kLiteRtQuantizationPerChannel: {
      const auto& per_channel = detail.per_channel;
      std::vector<float> scales(per_channel.scales,
                                per_channel.scales + per_channel.num_channels);
      std::vector<int64_t> zero_points(
          per_channel.zero_points,
          per_channel.zero_points + per_channel.num_channels);
      return py::make_tuple("per_channel", scales, zero_points,
                            per_channel.quantized_dimension);
    }
    case kLiteRtQuantizationBlockWise: {
      const auto& block_wise = detail.block_wise;
      py::object zero_points = py::none();
      if (block_wise.zero_points != nullptr) {
        zero_points =
            py::cast(PyTensor::Make(self.parent, block_wise.zero_points));
      }
      return py::make_tuple("block_wise",
                            PyTensor::Make(self.parent, block_wise.scales),
                            zero_points, block_wise.block_size);
    }
    default:
      return py::none();
  }
}

void ClearQuantization(const PyTensor& self) {
  self.Get().SetQarams(MakeEmptyQuantization());
}

void SetPerTensorQuantization(const PyTensor& self, float scale,
                              int64_t zero_point) {
  self.Get().SetQarams(MakePerTensorQuantization(scale, zero_point));
}

void SetPerChannelQuantization(const PyTensor& self,
                               const std::vector<float>& scales,
                               const std::vector<int64_t>& zero_points,
                               int32_t quantized_dimension) {
  LiteRtTensorT& tensor = self.Get();
  if (scales.empty() || scales.size() != zero_points.size()) {
    throw py::value_error(absl::StrFormat(
        "per-channel quantization needs as many scales as zero points (got "
        "%d and %d, at least one)",
        scales.size(), zero_points.size()));
  }
  if (quantized_dimension < 0) {
    throw py::value_error("quantized_dimension must be non-negative");
  }
  if (const auto shape = GetShape(tensor); shape.has_value()) {
    if (static_cast<size_t>(quantized_dimension) >= shape->size()) {
      throw py::value_error(
          absl::StrFormat("quantized_dimension %d is out of range for rank %d",
                          quantized_dimension, shape->size()));
    }
    const int32_t channels = (*shape)[quantized_dimension];
    if (channels >= 0 && static_cast<size_t>(channels) != scales.size()) {
      throw py::value_error(absl::StrFormat(
          "dimension %d has %d channels but %d scales were given",
          quantized_dimension, channels, scales.size()));
    }
  }
  tensor.SetQarams(MakePerChannelQuantization(scales, zero_points,
                                              quantized_dimension, tensor));
}

void SetBlockWiseQuantization(const PyTensor& self, const PyTensor& scales,
                              std::optional<PyTensor> zero_points,
                              int32_t block_size) {
  LiteRtTensorT& tensor = self.Get();
  CheckSameSubgraph(self.parent, scales);
  if (zero_points.has_value()) {
    CheckSameSubgraph(self.parent, *zero_points);
  }
  if (block_size <= 0) {
    throw py::value_error("block_size must be positive");
  }
  if (&scales.Get() == &tensor ||
      (zero_points.has_value() && &zero_points->Get() == &tensor)) {
    throw py::value_error(
        "a tensor cannot hold its own block-wise quantization parameters");
  }
  tensor.SetQarams(MakeBlockWiseQuantization(
      &scales.Get(), zero_points.has_value() ? &zero_points->Get() : nullptr,
      block_size));
}

py::object WeightsBuffer(const PyTensor& self) {
  const LiteRtTensorT& tensor = self.Get();
  const BufferRef<uint8_t> buffer = tensor.Weights().Buffer();
  if (buffer.Size() == 0) {
    return py::none();
  }
  // Only memory that never moves or dies before the model is exposed without a
  // copy: buffers kept by the state (`set_weights`) and the flatbuffer the
  // model was loaded from. Buffers owned by `BufferManager` move when it grows
  // (see `ModelState::KeepBuffer`), so they are copied.
  ModelState& state = *self.parent.state;
  bool stable = state.IsKeptBuffer(buffer.Data());
  if (!stable) {
    const auto& flatbuffer = internal::GetTflFlatbuffer(state.Model());
    if (flatbuffer.FlatbufferModelPtr() != nullptr) {
      const BufferRef<uint8_t> allocation = flatbuffer.Buf();
      const uint8_t* begin = allocation.Data();
      stable = buffer.Data() >= begin &&
               buffer.Data() + buffer.Size() <= begin + allocation.Size();
    }
  }
  if (!stable) {
    return py::memoryview(ToBytes(buffer));
  }
  return py::memoryview(
      py::cast(WeightsView{self.parent.state, buffer.Data(), buffer.Size()}));
}

void SetWeights(const PyTensor& self, const py::buffer& data) {
  LiteRtTensorT& tensor = self.Get();
  if (tensor.DefiningOp() != nullptr) {
    throw py::value_error(
        "cannot set constant data on a tensor produced by an op");
  }
  if (FindInput(self.parent.Get(), tensor)) {
    throw py::value_error("cannot set constant data on a subgraph input");
  }
  OwningBufferRef<uint8_t> copy = CopyBuffer(data);
  if (copy.Size() == 0) {
    throw py::value_error("constant data must not be empty");
  }
  if (const auto shape = GetShape(tensor); shape.has_value()) {
    const auto expected = CheckedDenseByteSize(GetElementType(tensor), *shape);
    if (expected.has_value() && *expected != copy.Size()) {
      throw py::value_error(absl::StrFormat(
          "the tensor type needs %d bytes of constant data, got %d", *expected,
          copy.Size()));
    }
  }
  // See `ModelState::KeepBuffer` for why the buffer is not registered as owned.
  SetWeightsFromUnownedBuffer(tensor.Weights(),
                              self.parent.state->KeepBuffer(std::move(copy)));
}

py::list Users(const PyTensor& self) {
  const LiteRtTensorT& tensor = self.Get();
  py::list result;
  for (size_t i = 0; i < tensor.NumUses(); ++i) {
    const auto [user, input_index] = tensor.GetUse(i);
    result.append(py::make_tuple(PyOp::Make(self.parent, user), input_index));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Python exceptions.
//===----------------------------------------------------------------------===//

PyObject* g_litert_error = nullptr;
PyObject* g_shape_inference_error = nullptr;
PyObject* g_validation_error = nullptr;

PyObject* NewException(py::module_& module, const char* name, const char* doc,
                       PyObject* base) {
  const std::string qualified =
      absl::StrFormat("%s.%s", PyModule_GetName(module.ptr()), name);
  PyObject* type =
      PyErr_NewExceptionWithDoc(qualified.c_str(), doc, base, nullptr);
  if (type == nullptr) {
    throw py::error_already_set();
  }
  // `add_object` adds a reference owned by the module; the new reference
  // returned above is kept by the caller for the lifetime of the process.
  module.add_object(name, py::handle(type));
  return type;
}

void TranslateIrError(const IrError& error) {
  PyObject* type = g_litert_error;
  if (error.kind() == ErrorKind::kShapeInference) {
    type = g_shape_inference_error;
  } else if (error.kind() == ErrorKind::kValidation) {
    type = g_validation_error;
  }
  py::object instance =
      py::reinterpret_borrow<py::object>(type)(py::str(error.what()));
  instance.attr("status") = py::str(StatusName(error.status()));
  instance.attr("status_code") = py::int_(static_cast<int>(error.status()));
  instance.attr("op") = error.op() ? py::cast(*error.op()) : py::none();
  PyErr_SetObject(type, instance.ptr());
}

}  // namespace
}  // namespace litert::litert_model

namespace {

using ::litert::litert_model::IrError;
using ::litert::litert_model::PyModel;
using ::litert::litert_model::PyOp;
using ::litert::litert_model::PySignature;
using ::litert::litert_model::PySubgraph;
using ::litert::litert_model::PyTensor;
using ::litert::litert_model::WeightsView;

template <class Handle>
size_t HashHandle(const Handle& handle, const void* object) {
  return std::hash<const void*>{}(object) ^ (handle.id * 0x9E3779B97F4A7C15ULL);
}

}  // namespace

PYBIND11_MODULE(_pywrap_litert_model, m) {
  namespace ir = ::litert::litert_model;
  m.doc() = "pybind11 bindings of the C++ LiteRT IR (LiteRtModelT).";

  ir::g_litert_error = ir::NewException(
      m, "LiteRtError", "An error reported by the LiteRT C++ IR.",
      PyExc_RuntimeError);
  ir::g_shape_inference_error = ir::NewException(
      m, "ShapeInferenceError", "Shape inference failed.", ir::g_litert_error);
  ir::g_validation_error = ir::NewException(
      m, "ValidationError", "The IR violates a structural invariant.",
      ir::g_litert_error);
  py::register_exception_translator([](std::exception_ptr exception) {
    try {
      if (exception) std::rethrow_exception(exception);
    } catch (const IrError& error) {
      ir::TranslateIrError(error);
    }
  });

  m.def("element_type_values", [] {
    py::dict result;
    for (const auto& [name, value] : ir::ElementTypeValues()) {
      result[py::str(name)] = value;
    }
    return result;
  });
  m.def("op_code_values", [] {
    py::dict result;
    for (const auto& [name, value] : ir::OpCodeValues()) {
      result[py::str(name)] = value;
    }
    return result;
  });
  m.def("element_type_bit_width", [](int element_type) {
    return ir::ElementTypeBitWidth(ir::CheckedElementType(element_type));
  });
  m.attr("DISPATCH_OP_CUSTOM_CODE") = std::string(ir::kDispatchOpCustomCode);
  m.attr("DEFAULT_SIGNATURE_KEY") = std::string(::litert::kDefaultSignatureKey);
  m.attr("BUILD_STAMP_METADATA_KEY") =
      std::string(::litert::internal::kLiteRtBuildStampKey);

  py::class_<WeightsView>(m, "_WeightsView", py::buffer_protocol())
      .def_buffer([](WeightsView& view) {
        return py::buffer_info(const_cast<uint8_t*>(view.data), sizeof(uint8_t),
                               py::format_descriptor<uint8_t>::format(), 1,
                               {static_cast<py::ssize_t>(view.size)},
                               {sizeof(uint8_t)},
                               /*readonly=*/true);
      });

  py::class_<PyModel>(m, "Model")
      .def_static("load_from_file", &PyModel::LoadFromFile, py::arg("path"))
      .def_static("load_from_bytes", &PyModel::LoadFromBytes, py::arg("data"))
      .def_static("create", &PyModel::Create)
      .def("num_subgraphs", &PyModel::NumSubgraphs)
      .def("subgraph", &PyModel::Subgraph, py::arg("index"))
      .def("subgraphs", &PyModel::Subgraphs)
      .def("add_subgraph", &PyModel::AddSubgraph, py::arg("name"))
      .def("prune_unreachable_subgraphs", &PyModel::PruneUnreachableSubgraphs)
      .def("signatures", &PyModel::Signatures)
      .def("find_signature", &PyModel::FindSignature, py::arg("key"))
      .def("add_signature", &PyModel::AddSignature, py::arg("key"),
           py::arg("subgraph"), py::arg("inputs"), py::arg("outputs"))
      .def("remove_signature", &PyModel::RemoveSignature, py::arg("key"))
      .def("metadata_keys", &PyModel::MetadataKeys)
      .def("metadata", &PyModel::Metadata, py::arg("key"))
      .def("push_metadata", &PyModel::PushMetadata, py::arg("key"),
           py::arg("data"))
      .def("validate", &PyModel::Validate)
      .def("serialize", &PyModel::Serialize, py::arg("bytecode_alignment"))
      .def("is_serialized", &PyModel::IsSerialized)
      .def("infer_shapes", &PyModel::InferShapes, py::arg("validation_only"))
      .def("specialize_subgraph", &PyModel::SpecializeSubgraph,
           py::arg("subgraph"), py::arg("input_shapes"))
      .def("register_buffer", &PyModel::RegisterBuffer, py::arg("data"))
      .def("attach_asset", &PyModel::AttachAssetToOp, py::arg("op"),
           py::arg("buffer_id"), py::arg("name"))
      .def("set_build_stamp", &PyModel::SetBuildStamp,
           py::arg("soc_manufacturer"), py::arg("soc_model"))
      .def("build_stamp", &PyModel::BuildStamp);

  py::class_<PySubgraph>(m, "Subgraph")
      .def("index", &ir::SubgraphIndex)
      .def(
          "name",
          [](const PySubgraph& self) { return std::string(self.Get().Name()); })
      .def(
          "set_name",
          [](const PySubgraph& self, const std::string& name) {
            self.Get().SetName(name);
          },
          py::arg("name"))
      .def("inputs",
           [](const PySubgraph& self) {
             return ir::TensorList(self, self.Get().Inputs());
           })
      .def("outputs",
           [](const PySubgraph& self) {
             return ir::TensorList(self, self.Get().Outputs());
           })
      .def(
          "set_inputs",
          [](const PySubgraph& self, const std::vector<PyTensor>& inputs) {
            const auto raw = ir::RawTensors(self, inputs);
            ir::CheckEdit(
                ir::SetSubgraphInputs(self.state->Model(), self.Get(), raw));
          },
          py::arg("inputs"))
      .def(
          "set_outputs",
          [](const PySubgraph& self, const std::vector<PyTensor>& outputs) {
            const auto raw = ir::RawTensors(self, outputs);
            ir::CheckEdit(
                ir::SetSubgraphOutputs(self.state->Model(), self.Get(), raw));
          },
          py::arg("outputs"))
      .def("tensors",
           [](const PySubgraph& self) {
             return ir::TensorList(self, self.Get().Tensors());
           })
      .def("num_ops",
           [](const PySubgraph& self) { return self.Get().Ops().size(); })
      .def("ops",
           [](const PySubgraph& self) {
             py::list result;
             for (LiteRtOp op : self.Get().Ops()) {
               result.append(PyOp::Make(self, op));
             }
             return result;
           })
      .def("add_tensor", &ir::AddTensor, py::arg("shape"),
           py::arg("element_type"), py::arg("name"))
      .def("add_op", &ir::AddOpToSubgraph, py::arg("code"), py::arg("inputs"),
           py::arg("outputs"), py::arg("index"), py::arg("custom_code"),
           py::arg("custom_options"))
      .def("remove_op", &ir::RemoveOpFromSubgraph, py::arg("op"))
      .def("remove_tensor", &ir::RemoveTensorFromSubgraph, py::arg("tensor"))
      .def("dce", &ir::Dce)
      .def("replace_all_uses", &ir::ReplaceAllUsesInSubgraph,
           py::arg("old_tensor"), py::arg("new_tensor"), py::arg("exclude"))
      .def("infer_shapes", &ir::InferSubgraphShapes, py::arg("validation_only"))
      .def("outline_to_dispatch_op", &ir::OutlineToDispatchOp)
      .def("__eq__",
           [](const PySubgraph& self, const PySubgraph& other) {
             return self.state == other.state &&
                    self.subgraph == other.subgraph && self.id == other.id;
           })
      .def("__hash__", [](const PySubgraph& self) {
        return HashHandle(self, self.subgraph);
      });

  py::class_<PyOp>(m, "Op")
      .def("subgraph", [](const PyOp& self) { return self.parent; })
      .def("code",
           [](const PyOp& self) {
             return static_cast<int>(self.Get().OpCode());
           })
      .def("index",
           [](const PyOp& self) {
             return ir::ValueOrThrow(
                 ir::FindOpIndex(self.parent.Get(), self.Get()));
           })
      .def("custom_code", &ir::CustomCode)
      .def("custom_options",
           [](const PyOp& self) {
             return ir::ToBytes(self.Get().CustomOptions());
           })
      .def(
          "set_custom_options",
          [](const PyOp& self, const py::buffer& data) {
            auto copy = ir::CopyBuffer(data);
            if (copy.Size() == 0) {
              self.Get().ClearCustomOptions();
            } else {
              self.Get().SetCustomOptions(std::move(copy));
            }
          },
          py::arg("data"))
      .def("inputs",
           [](const PyOp& self) {
             return ir::TensorList(self.parent, self.Get().Inputs());
           })
      .def("outputs",
           [](const PyOp& self) {
             return ir::TensorList(self.parent, self.Get().Outputs());
           })
      .def("set_input", &ir::SetInput, py::arg("index"), py::arg("tensor"))
      .def("set_output", &ir::SetOutput, py::arg("index"), py::arg("tensor"))
      .def("builtin_options_types",
           [](const PyOp& self) {
             return ir::GetBuiltinOptionsTypes(self.Get());
           })
      .def("builtin_options_bytes",
           [](const PyOp& self) {
             return ir::ToBytes(ir::PackBuiltinOptions(self.Get()));
           })
      .def(
          "set_builtin_options_bytes",
          [](const PyOp& self, const py::buffer& data) {
            auto copy = ir::CopyBuffer(data);
            ir::ThrowIfError(ir::UnpackBuiltinOptions(copy, self.Get()));
          },
          py::arg("data"))
      .def("is_dispatch_op",
           [](const PyOp& self) { return ir::IsDispatchOp(self.Get()); })
      .def("dispatch_options", &ir::DispatchOptions)
      .def("__eq__",
           [](const PyOp& self, const PyOp& other) {
             return self.parent.state == other.parent.state &&
                    self.op == other.op && self.id == other.id;
           })
      .def("__hash__",
           [](const PyOp& self) { return HashHandle(self, self.op); });

  py::class_<PyTensor>(m, "Tensor")
      .def("subgraph", [](const PyTensor& self) { return self.parent; })
      .def("index",
           [](const PyTensor& self) {
             return ir::ValueOrThrow(
                 ir::FindTensorIndex(self.parent.Get(), self.Get()));
           })
      .def("name",
           [](const PyTensor& self) { return std::string(self.Get().Name()); })
      .def(
          "set_name",
          [](const PyTensor& self, const std::string& name) {
            self.Get().SetName(name);
          },
          py::arg("name"))
      .def("is_ranked",
           [](const PyTensor& self) {
             return self.Get().Type().first == kLiteRtRankedTensorType;
           })
      .def("shape",
           [](const PyTensor& self) { return ir::GetShape(self.Get()); })
      .def("set_shape", &ir::SetShape, py::arg("shape"))
      .def("element_type",
           [](const PyTensor& self) {
             return static_cast<int>(ir::GetElementType(self.Get()));
           })
      .def("set_element_type", &ir::SetElementType, py::arg("element_type"))
      .def("quantization", &ir::GetQuantization)
      .def("clear_quantization", &ir::ClearQuantization)
      .def("set_per_tensor_quantization", &ir::SetPerTensorQuantization,
           py::arg("scale"), py::arg("zero_point"))
      .def("set_per_channel_quantization", &ir::SetPerChannelQuantization,
           py::arg("scales"), py::arg("zero_points"),
           py::arg("quantized_dimension"))
      .def("set_block_wise_quantization", &ir::SetBlockWiseQuantization,
           py::arg("scales"), py::arg("zero_points"), py::arg("block_size"))
      .def(
          "is_constant",
          [](const PyTensor& self) { return ir::IsConstantTensor(self.Get()); })
      .def("is_subgraph_input",
           [](const PyTensor& self) {
             return ::litert::internal::FindInput(self.parent.Get(), self.Get())
                 .has_value();
           })
      .def("is_subgraph_output",
           [](const PyTensor& self) {
             return ::litert::internal::FindOutput(self.parent.Get(),
                                                   self.Get())
                 .has_value();
           })
      .def("weights_buffer", &ir::WeightsBuffer)
      .def("set_weights", &ir::SetWeights, py::arg("data"))
      .def("defining_op",
           [](const PyTensor& self) -> std::optional<PyOp> {
             LiteRtOpT* op = self.Get().DefiningOp();
             if (op == nullptr) return std::nullopt;
             return PyOp::Make(self.parent, op);
           })
      .def("defining_op_output_index",
           [](const PyTensor& self) -> int64_t {
             const LiteRtTensorT& tensor = self.Get();
             if (tensor.DefiningOp() == nullptr) return -1;
             return tensor.DefiningOpOutInd();
           })
      .def("users", &ir::Users)
      .def("__eq__",
           [](const PyTensor& self, const PyTensor& other) {
             return self.parent.state == other.parent.state &&
                    self.tensor == other.tensor && self.id == other.id;
           })
      .def("__hash__",
           [](const PyTensor& self) { return HashHandle(self, self.tensor); });

  py::class_<PySignature>(m, "Signature")
      .def(
          "key",
          [](const PySignature& self) { return std::string(self.Get().Key()); })
      .def("subgraph",
           [](const PySignature& self) {
             return PySubgraph::Make(self.state, &self.Get().GetSubgraph());
           })
      .def("input_names",
           [](const PySignature& self) { return self.Get().InputNames(); })
      .def("output_names",
           [](const PySignature& self) { return self.Get().OutputNames(); })
      .def("inputs",
           [](const PySignature& self) {
             LiteRtSignatureT& signature = self.Get();
             const auto parent =
                 PySubgraph::Make(self.state, &signature.GetSubgraph());
             py::list result;
             for (size_t i = 0; i < signature.InputNames().size(); ++i) {
               result.append(
                   PyTensor::Make(parent, signature.GetInputTensor(i)));
             }
             return result;
           })
      .def("outputs",
           [](const PySignature& self) {
             LiteRtSignatureT& signature = self.Get();
             const auto parent =
                 PySubgraph::Make(self.state, &signature.GetSubgraph());
             py::list result;
             for (size_t i = 0; i < signature.OutputNames().size(); ++i) {
               result.append(
                   PyTensor::Make(parent, signature.GetOutputTensor(i)));
             }
             return result;
           })
      .def("__eq__",
           [](const PySignature& self, const PySignature& other) {
             return self.state == other.state &&
                    self.signature == other.signature && self.id == other.id;
           })
      .def("__hash__", [](const PySignature& self) {
        return HashHandle(self, self.signature);
      });
}
