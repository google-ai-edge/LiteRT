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

#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_runtime_api.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "driver_types.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/compiler/subbyte_gemv_plugin.h"
#include "litert/vendors/nvidia/compiler/tensorrt_rtx_plugin_compat.h"
#include "litert/vendors/nvidia/memory_profile.h"
#include "litert/vendors/nvidia/tensorrt_logger.h"
#include "NvInfer.h"

namespace litert::nvidia {
namespace {

using ::litert::compiler::Op;
using ::litert::compiler::Subgraph;
using ::litert::compiler::Tensor;

// The budget for a partition's intermediate buffers. Myelin compiles a whole
// partition as one fused region, so this must cover the full activation arena
// of the largest island (multi-layer prefill attention islands need >1GB); it
// is a cap on tactic choice, not an upfront allocation.
constexpr size_t kDefaultTensorRtWorkspaceBytes = 4096ULL << 20;
// Sized for LLM prefill attention at a few thousand tokens of context (score
// tensors like [1, heads, seq, seq]); larger tensors stay on the CPU.
constexpr size_t kDefaultMaxTensorRtBatchMatmulOutputElements = 1ULL << 24;
constexpr size_t kDefaultMaxTensorRtFillBytes = 16ULL << 20;
// Large enough to admit LLM vocab-projection heads (Gemma4's INT2 LM head
// packs to ~100MB); float heads beyond this stay on the CPU.
constexpr size_t kDefaultMaxTensorRtFullyConnectedWeightBytes = 256ULL << 20;
constexpr size_t kDefaultMaxTensorRtSoftmaxElements = 1ULL << 24;

template <typename T>
using TrtPtr = std::unique_ptr<T>;

size_t EnvSizeT(const char* name, size_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}

bool EnvEnabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  return std::strcmp(value, "0") != 0;
}

size_t TensorRtWorkspaceBytes() {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_WORKSPACE_MB",
                  kDefaultTensorRtWorkspaceBytes >> 20)
         << 20;
}

size_t MaxTensorRtSoftmaxElements() {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_MAX_SOFTMAX_ELEMENTS",
                  kDefaultMaxTensorRtSoftmaxElements);
}

size_t MaxTensorRtBatchMatmulOutputElements() {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_MAX_BATCH_MATMUL_OUTPUT_ELEMENTS",
                  kDefaultMaxTensorRtBatchMatmulOutputElements);
}

size_t MaxTensorRtFillBytes() {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_MAX_FILL_BYTES",
                  kDefaultMaxTensorRtFillBytes);
}

size_t MaxTensorRtFullyConnectedWeightBytes() {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_MAX_FC_WEIGHT_BYTES",
                  kDefaultMaxTensorRtFullyConnectedWeightBytes);
}

bool LogUnsupportedOpDetails() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_LOG_UNSUPPORTED_OPS");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void LogFullyConnectedReject(const Op& op, const char* reason) {
  if (!LogUnsupportedOpDetails()) {
    return;
  }
  auto summarize_tensor = [](const Tensor& tensor) {
    std::string summary =
        "tensor=" + std::to_string(tensor.TensorIndex()) +
        " type=" + std::to_string(static_cast<int>(tensor.ElementType()));
    auto ranked_type = tensor.RankedTensorType();
    if (ranked_type) {
      summary += " shape=[";
      const auto dims = ranked_type->Layout().Dimensions();
      for (int i = 0; i < dims.size(); ++i) {
        if (i > 0) {
          summary += "x";
        }
        summary += std::to_string(dims[i]);
      }
      summary += "]";
    } else {
      summary += " shape=<unranked>";
    }
    summary += tensor.HasWeights() ? " const" : " runtime";
    summary += " qtype=" + std::to_string(tensor.QTypeId());
    return summary;
  };
  std::string detail = std::string(reason) + " inputs={";
  for (int i = 0; i < op.Inputs().size(); ++i) {
    if (i > 0) {
      detail += "; ";
    }
    detail += summarize_tensor(op.Inputs()[i]);
  }
  detail += "} outputs={";
  for (int i = 0; i < op.Outputs().size(); ++i) {
    if (i > 0) {
      detail += "; ";
    }
    detail += summarize_tensor(op.Outputs()[i]);
  }
  detail += "}";
  LITERT_LOG(LITERT_INFO, "NVIDIA TensorRT-RTX FC unsupported: %s",
             detail.c_str());
}

Expected<nvinfer1::DataType> ConvertDataType(litert::ElementType type) {
  switch (type) {
    case litert::ElementType::Float32:
      return nvinfer1::DataType::kFLOAT;
    case litert::ElementType::Float16:
      return nvinfer1::DataType::kHALF;
    case litert::ElementType::BFloat16:
      return nvinfer1::DataType::kBF16;
    case litert::ElementType::Float8E4M3FN:
      return nvinfer1::DataType::kFP8;
    case litert::ElementType::Int4:
      return nvinfer1::DataType::kINT4;
    case litert::ElementType::Int8:
      return nvinfer1::DataType::kINT8;
    case litert::ElementType::Int32:
      return nvinfer1::DataType::kINT32;
    case litert::ElementType::Int64:
      return nvinfer1::DataType::kINT64;
    case litert::ElementType::Bool:
      return nvinfer1::DataType::kBOOL;
    default:
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported TensorRT tensor element type: " +
                       std::to_string(static_cast<int>(type)));
  }
}

bool IsFloatLike(litert::ElementType type) {
  return type == litert::ElementType::Float32 ||
         type == litert::ElementType::Float16 ||
         type == litert::ElementType::BFloat16;
}

bool IsTensorRtElementType(litert::ElementType type) {
  return IsFloatLike(type) || type == litert::ElementType::Float8E4M3FN ||
         type == litert::ElementType::Int4 ||
         type == litert::ElementType::Int8 ||
         type == litert::ElementType::Int32 ||
         type == litert::ElementType::Int64 ||
         type == litert::ElementType::Bool;
}

bool IsTensorRtArithmeticElementType(litert::ElementType type) {
  return IsFloatLike(type) || type == litert::ElementType::Int8 ||
         type == litert::ElementType::Int32 ||
         type == litert::ElementType::Int64;
}

Expected<nvinfer1::Dims> ConvertDims(const litert::RankedTensorType& type) {
  const auto& layout = type.Layout();
  if (layout.HasStrides()) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "TensorRT backend does not support strided LiteRT tensors");
  }
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int32_t>(layout.Rank());
  if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
    return Error(kLiteRtStatusErrorUnsupported, "Tensor rank is too large");
  }
  for (int i = 0; i < dims.nbDims; ++i) {
    const int32_t dim = layout.Dimensions()[i];
    if (dim < 0) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Dynamic shapes are not supported yet");
    }
    dims.d[i] = dim;
  }
  return dims;
}

Expected<size_t> NumElements(const litert::RankedTensorType& type) {
  return type.Layout().NumElements();
}

Expected<size_t> ElementByteSize(litert::ElementType type) {
  switch (type) {
    case litert::ElementType::Float32:
    case litert::ElementType::Int32:
      return 4;
    case litert::ElementType::Float16:
    case litert::ElementType::BFloat16:
      return 2;
    case litert::ElementType::Float8E4M3FN:
    case litert::ElementType::Int8:
    case litert::ElementType::Bool:
      return 1;
    case litert::ElementType::Int64:
      return 8;
    default:
      return Error(kLiteRtStatusErrorUnsupported,
                   "Element type does not have a supported byte width");
  }
}

Expected<bool> IsStaticRankedTensor(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
  if (type.Layout().HasStrides()) {
    return false;
  }
  for (auto dim : type.Layout().Dimensions()) {
    if (dim < 0) {
      return false;
    }
  }
  return true;
}

Expected<bool> IsStaticTensorRtTensor(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(bool static_ranked, IsStaticRankedTensor(tensor));
  if (!static_ranked) {
    return false;
  }
  return IsTensorRtElementType(tensor.ElementType());
}

Expected<bool> IsStaticFloatTensor(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(bool static_ranked, IsStaticRankedTensor(tensor));
  if (!static_ranked) {
    return false;
  }
  return IsFloatLike(tensor.ElementType());
}

bool SameElementType(const Tensor& lhs, const Tensor& rhs) {
  return lhs.ElementType() == rhs.ElementType();
}

bool HasRuntimeInput(const Op& op) {
  for (const auto& input : op.Inputs()) {
    if (!input.HasWeights()) {
      return true;
    }
  }
  return false;
}

Expected<size_t> StaticTensorByteSize(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
  if (type.ElementType() == litert::ElementType::Int4) {
    return (num_elements + 1) / 2;
  }
  LITERT_ASSIGN_OR_RETURN(size_t element_size,
                          ElementByteSize(type.ElementType()));
  return num_elements * element_size;
}

bool IsTensorRtQuantizedElementType(litert::ElementType type) {
  return type == litert::ElementType::Int4 ||
         type == litert::ElementType::Int8 ||
         type == litert::ElementType::Float8E4M3FN;
}

Expected<bool> HasSymmetricTensorRtQuantization(const Tensor& tensor,
                                                bool allow_per_channel);

bool IsSupportedFusedActivation(uint32_t activation) {
  using Opt = litert::ActivationFunctionType;
  return activation == Opt::kActivationFunctionTypeNone ||
         activation == Opt::kActivationFunctionTypeRelu ||
         activation == Opt::kActivationFunctionTypeRelu6 ||
         activation == Opt::kActivationFunctionTypeReluN1To1;
}

Expected<uint32_t> GetFusedActivation(const Op& op) {
  uint32_t activation = litert::kActivationFunctionTypeNone;
  LiteRtStatus status = kLiteRtStatusOk;
  switch (op.Code()) {
    case kLiteRtOpCodeTflAdd:
      status = LiteRtGetAddFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflMul:
      status = LiteRtGetMulFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflSub:
      status = LiteRtGetSubFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflDiv:
      status = LiteRtGetDivFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflFullyConnected:
      status =
          LiteRtGetFullyConnectedFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflConv2d:
      status = LiteRtGetConv2dFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflConcatenation:
      status =
          LiteRtGetConcatenationFusedActivationOption(op.Get(), &activation);
      break;
    default:
      return activation;
  }
  if (status != kLiteRtStatusOk) {
    return Error(status, "Failed to read fused activation option");
  }
  if (!IsSupportedFusedActivation(activation)) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Fused activation is not supported");
  }
  return activation;
}

Expected<bool> HasSupportedFusedActivation(const Op& op) {
  auto activation = GetFusedActivation(op);
  return activation.HasValue();
}

Expected<std::vector<int32_t>> ReadInt32Constant(const Tensor& tensor) {
  if (!tensor.HasWeights() ||
      tensor.ElementType() != litert::ElementType::Int32) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Expected constant int32 tensor");
  }
  LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
  auto bytes = tensor.Weights().Bytes();
  if (bytes.size() != num_elements * sizeof(int32_t)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Unexpected int32 constant size");
  }
  std::vector<int32_t> values(num_elements);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

bool IsElementwiseOp(LiteRtOpCode code) {
  return code == kLiteRtOpCodeTflAdd || code == kLiteRtOpCodeTflMul ||
         code == kLiteRtOpCodeTflSub || code == kLiteRtOpCodeTflDiv ||
         code == kLiteRtOpCodeTflMaximum;
}

Expected<bool> HasSymmetricPerTensorInt8Quantization(const Tensor& tensor) {
  if (tensor.ElementType() != litert::ElementType::Int8 ||
      tensor.QTypeId() != kLiteRtQuantizationPerTensor) {
    return false;
  }
  return HasSymmetricTensorRtQuantization(tensor,
                                          /*allow_per_channel=*/false);
}

Expected<bool> IsElementwiseSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (!HasRuntimeInput(op)) {
    return false;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticTensorRtTensor(tensor));
    if (!supported || !IsTensorRtArithmeticElementType(tensor.ElementType())) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!output_supported ||
      !IsTensorRtArithmeticElementType(op.Outputs()[0].ElementType())) {
    return false;
  }
  if (!SameElementType(op.Inputs()[0], op.Inputs()[1]) ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  if (op.Code() == kLiteRtOpCodeTflMaximum) {
    return true;
  }
  if (op.Inputs()[0].ElementType() == litert::ElementType::Int8) {
    if (EnvEnabled("LITERT_NVIDIA_TENSORRT_DISABLE_INT8_ELEMENTWISE",
                   /*default_value=*/false)) {
      return false;
    }
    // Symmetric per-tensor int8 elementwise is lowered as DQ -> op -> Q.
    for (const auto& tensor :
         {op.Inputs()[0], op.Inputs()[1], op.Outputs()[0]}) {
      LITERT_ASSIGN_OR_RETURN(bool symmetric,
                              HasSymmetricPerTensorInt8Quantization(tensor));
      if (!symmetric) {
        return false;
      }
    }
    return HasSupportedFusedActivation(op);
  }
  if (op.Inputs()[0].ElementType() == litert::ElementType::Int32 ||
      op.Inputs()[0].ElementType() == litert::ElementType::Int64) {
    // Integer index arithmetic (e.g. cache positions) with no activation.
    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    return activation == litert::kActivationFunctionTypeNone;
  }
  if (!IsFloatLike(op.Inputs()[0].ElementType())) {
    return false;
  }
  return HasSupportedFusedActivation(op);
}

Expected<bool> IsComparisonSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (!HasRuntimeInput(op)) {
    return false;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticTensorRtTensor(tensor));
    if (!supported || !IsTensorRtArithmeticElementType(tensor.ElementType())) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  return output_supported &&
         op.Outputs()[0].ElementType() == litert::ElementType::Bool &&
         SameElementType(op.Inputs()[0], op.Inputs()[1]);
}

Expected<bool> IsLogicalAndSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (!HasRuntimeInput(op)) {
    return false;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticTensorRtTensor(tensor));
    if (!supported || tensor.ElementType() != litert::ElementType::Bool) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  return output_supported &&
         op.Outputs()[0].ElementType() == litert::ElementType::Bool;
}

Expected<bool> IsUnaryActivationSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticFloatTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  return input_supported && output_supported;
}

Expected<bool> IsGeluSupported(const Op& op) {
  LITERT_ASSIGN_OR_RETURN(bool supported, IsUnaryActivationSupported(op));
  bool approximate = false;
  return supported && LiteRtGetGeluApproximateOption(op.Get(), &approximate) ==
                          kLiteRtStatusOk;
}

Expected<bool> IsSoftmaxSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  float beta = 1.0f;
  if (LiteRtGetSoftmaxBetaOption(op.Get(), &beta) != kLiteRtStatusOk ||
      beta != 1.0f) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(size_t input_elements, NumElements(input_type));
  if (input_elements > MaxTensorRtSoftmaxElements()) {
    return false;
  }
  return IsUnaryActivationSupported(op);
}

Expected<bool> IsReshapeSupported(const Op& op) {
  if (op.Inputs().size() < 1 || op.Outputs().size() != 1) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported) {
    return false;
  }
  if (!SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  // The shape operand (constant or runtime-computed) is ignored: the static
  // output shape fully determines the reshape.
  return true;
}

Expected<bool> IsCastSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  return input_supported && output_supported;
}

Expected<bool> IsTransposeSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  auto perm = ReadInt32Constant(op.Inputs()[1]);
  return perm.HasValue();
}

Expected<bool> IsSliceSupported(const Op& op) {
  if (op.Inputs().size() != 3 || op.Outputs().size() != 1) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const size_t rank = input_type.Layout().Rank();
  // The size operand is ignored: the static output shape determines it. The
  // begin operand may be a constant or a runtime int32 vector of length rank.
  auto begin = ReadInt32Constant(op.Inputs()[1]);
  if (begin) {
    return begin->size() == rank;
  }
  if (op.Inputs()[1].HasWeights() ||
      op.Inputs()[1].ElementType() != litert::ElementType::Int32) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto begin_type, op.Inputs()[1].RankedTensorType());
  if (begin_type.Layout().Rank() != 1 ||
      begin_type.Layout().Dimensions()[0] != static_cast<int32_t>(rank)) {
    return false;
  }
  // Runtime offsets lower to a gather along the sliced axis; only one axis
  // may shrink (full-extent axes must have offset zero by slice semantics).
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  int shrinking_axes = 0;
  for (size_t i = 0; i < rank; ++i) {
    if (output_type.Layout().Dimensions()[i] !=
        input_type.Layout().Dimensions()[i]) {
      ++shrinking_axes;
    }
  }
  return shrinking_axes <= 1;
}

// Composite ops this backend can lower natively (instead of letting the
// runtime inline their decompositions, whose reshape/index glue fragments
// Myelin fusion into thousands of tiny kernels per step).
std::string CompositeOpName(const Op& op) {
  const char* name = nullptr;
  if (LiteRtGetSHLOCompositeOpName(op.Get(), &name) != kLiteRtStatusOk ||
      name == nullptr) {
    return "";
  }
  return name;
}

// Default: all supported composites lower natively. Set
// LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES to "none" (or "0") to inline
// everything, or to a comma list of short names (rms_norm,cache_update,
// runtime_bmm) to enable a subset.
bool NativeCompositeEnabled(const std::string& full_name) {
  const std::string short_name =
      full_name.rfind("odml.", 0) == 0 ? full_name.substr(5) : full_name;
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES");
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "none") == 0) {
    return false;
  }
  const std::string list(value);
  size_t begin = 0;
  while (begin < list.size()) {
    size_t end = list.find(',', begin);
    if (end == std::string::npos) {
      end = list.size();
    }
    if (list.compare(begin, end - begin, short_name) == 0) {
      return true;
    }
    begin = end + 1;
  }
  return false;
}

Expected<bool> IsCompositeSupported(const Op& op) {
  const std::string name = CompositeOpName(op);
  if (name.empty() || !NativeCompositeEnabled(name)) {
    return false;
  }
  if (name == "odml.rms_norm") {
    if (op.Inputs().size() < 1 || op.Inputs().size() > 2 ||
        op.Outputs().size() != 1) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(bool x_ok, IsStaticFloatTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(bool out_ok, IsStaticFloatTensor(op.Outputs()[0]));
    if (!x_ok || !out_ok) {
      return false;
    }
    if (op.Inputs().size() == 2) {
      LITERT_ASSIGN_OR_RETURN(bool w_ok, IsStaticFloatTensor(op.Inputs()[1]));
      if (!w_ok) {
        return false;
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto x_type, op.Inputs()[0].RankedTensorType());
    return x_type.Layout().Rank() >= 1;
  }
  if (name == "odml.cache_update") {
    // (update_a f32, update_b f32, _, cache_a i8, cache_b i8, start_a i32,
    //  start_b i32) -> (cache_a', cache_b'); see the odml decomposition.
    if (op.Inputs().size() < 7 || op.Outputs().size() != 2) {
      return false;
    }
    for (int i : {0, 1}) {
      LITERT_ASSIGN_OR_RETURN(bool ok, IsStaticFloatTensor(op.Inputs()[i]));
      if (!ok) {
        return false;
      }
    }
    for (int i : {3, 4}) {
      LITERT_ASSIGN_OR_RETURN(bool ok, IsStaticTensorRtTensor(op.Inputs()[i]));
      LITERT_ASSIGN_OR_RETURN(
          bool quant, HasSymmetricPerTensorInt8Quantization(op.Inputs()[i]));
      if (!ok || !quant) {
        return false;
      }
    }
    for (int i : {5, 6}) {
      if (op.Inputs()[i].ElementType() != litert::ElementType::Int32) {
        return false;
      }
    }
    for (int i : {0, 1}) {
      LITERT_ASSIGN_OR_RETURN(
          bool quant, HasSymmetricPerTensorInt8Quantization(op.Outputs()[i]));
      if (!quant) {
        return false;
      }
    }
    return true;
  }
  if (name == "odml.runtime_bmm") {
    // (a, cache, positions) -> a @ dequantize(cache)^T over the full
    // context. The decomposition computes only a sliding window and fills
    // the remainder with a placeholder, but every non-window position is
    // either masked downstream (score path) or multiplied by a zero
    // probability (probs x transposed-values path), so the full matmul is
    // equivalent.
    if (op.Inputs().size() != 3 || op.Outputs().size() != 1) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(bool a_ok, IsStaticFloatTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(bool out_ok, IsStaticFloatTensor(op.Outputs()[0]));
    LITERT_ASSIGN_OR_RETURN(bool cache_static,
                            IsStaticTensorRtTensor(op.Inputs()[1]));
    if (!a_ok || !out_ok || !cache_static) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(
        bool cache_quant,
        HasSymmetricPerTensorInt8Quantization(op.Inputs()[1]));
    if (!cache_quant && !IsFloatLike(op.Inputs()[1].ElementType())) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto a_type, op.Inputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto c_type, op.Inputs()[1].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto o_type, op.Outputs()[0].RankedTensorType());
    const auto a_dims = a_type.Layout().Dimensions();
    const auto c_dims = c_type.Layout().Dimensions();
    const auto o_dims = o_type.Layout().Dimensions();
    const int rank = a_type.Layout().Rank();
    if (rank < 2 || c_type.Layout().Rank() != rank ||
        o_type.Layout().Rank() != rank) {
      return false;
    }
    if (a_dims[rank - 1] != c_dims[rank - 1] ||
        o_dims[rank - 2] != a_dims[rank - 2] ||
        o_dims[rank - 1] != c_dims[rank - 2]) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(size_t out_elements, NumElements(o_type));
    return out_elements <= MaxTensorRtBatchMatmulOutputElements();
  }
  return false;
}

Expected<bool> IsDynamicUpdateSliceSupported(const Op& op) {
  if (op.Inputs().size() != 3 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool operand_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool update_supported,
                          IsStaticTensorRtTensor(op.Inputs()[1]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!operand_supported || !update_supported || !output_supported ||
      !SameElementType(op.Inputs()[0], op.Inputs()[1]) ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  const auto element_type = op.Inputs()[0].ElementType();
  if (!IsFloatLike(element_type) && element_type != litert::ElementType::Int8 &&
      element_type != litert::ElementType::Int32) {
    return false;
  }
  if (op.Inputs()[2].ElementType() != litert::ElementType::Int32) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto operand_type, op.Inputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto update_type, op.Inputs()[1].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto indices_type, op.Inputs()[2].RankedTensorType());
  const int rank = operand_type.Layout().Rank();
  if (rank < 1 || rank > 4 || update_type.Layout().Rank() != rank ||
      output_type.Layout().Rank() != rank ||
      indices_type.Layout().Rank() != 1) {
    return false;
  }
  const auto operand_dims = operand_type.Layout().Dimensions();
  const auto update_dims = update_type.Layout().Dimensions();
  const auto output_dims = output_type.Layout().Dimensions();
  const auto indices_dims = indices_type.Layout().Dimensions();
  if (indices_dims[0] != rank || operand_dims != output_dims) {
    return false;
  }
  // The lowering scatters elements along a single partial axis; every other
  // axis must be written in full (its offset is then zero by construction).
  int partial_axes = 0;
  for (int i = 0; i < rank; ++i) {
    if (update_dims[i] > operand_dims[i]) {
      return false;
    }
    if (update_dims[i] < operand_dims[i]) {
      ++partial_axes;
    }
  }
  return partial_axes <= 1;
}

Expected<bool> IsBatchMatmulSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticFloatTensor(tensor));
    if (!supported) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    if (type.Layout().Rank() < 2) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  if (!output_supported) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(size_t output_elements, NumElements(output_type));
  return output_elements <= MaxTensorRtBatchMatmulOutputElements();
}

Expected<bool> IsFullyConnectedSupported(const Op& op) {
  if (op.Inputs().size() < 2 || op.Inputs().size() > 3 ||
      op.Outputs().size() != 1) {
    LogFullyConnectedReject(op, "bad input/output count");
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported) {
    LogFullyConnectedReject(op,
                            "input or output is not static TensorRT tensor");
    return false;
  }
  const bool input_float = IsFloatLike(op.Inputs()[0].ElementType());
  const bool output_float = IsFloatLike(op.Outputs()[0].ElementType());
  bool input_quantized = false;
  bool output_quantized = false;
  if (op.Inputs()[0].ElementType() == litert::ElementType::Int8) {
    LITERT_ASSIGN_OR_RETURN(input_quantized, HasSymmetricTensorRtQuantization(
                                                 op.Inputs()[0],
                                                 /*allow_per_channel=*/false));
  }
  if (op.Outputs()[0].ElementType() == litert::ElementType::Int8) {
    LITERT_ASSIGN_OR_RETURN(output_quantized, HasSymmetricTensorRtQuantization(
                                                  op.Outputs()[0],
                                                  /*allow_per_channel=*/false));
  }
  if (!((input_float && output_float) ||
        (input_quantized && output_quantized))) {
    LogFullyConnectedReject(
        op, "input/output are neither float nor symmetric int8 quantized");
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool weights_supported,
                          IsStaticRankedTensor(op.Inputs()[1]));
  if (!weights_supported) {
    LogFullyConnectedReject(op, "weights are not static ranked");
    return false;
  }
  if (op.Inputs()[1].HasWeights() &&
      op.Inputs()[1].Weights().Bytes().size() >
          MaxTensorRtFullyConnectedWeightBytes()) {
    LogFullyConnectedReject(op, "weights exceed configured byte cap");
    return false;
  }
  const bool weights_float = IsFloatLike(op.Inputs()[1].ElementType());
  // Reject FCs whose weight shape is listed in this env (comma-separated
  // "OUTxIN" entries), keeping specific problematic matmuls on the CPU.
  if (const char* disabled_shapes =
          std::getenv("LITERT_NVIDIA_TENSORRT_DISABLE_FC_SHAPES");
      disabled_shapes != nullptr && disabled_shapes[0] != '\0') {
    auto weights_type = op.Inputs()[1].RankedTensorType();
    if (weights_type && weights_type->Layout().Rank() == 2) {
      const auto dims = weights_type->Layout().Dimensions();
      const std::string shape =
          std::to_string(dims[0]) + "x" + std::to_string(dims[1]);
      const std::string list(disabled_shapes);
      size_t begin = 0;
      while (begin < list.size()) {
        size_t end = list.find(',', begin);
        if (end == std::string::npos) {
          end = list.size();
        }
        if (list.compare(begin, end - begin, shape) == 0) {
          LogFullyConnectedReject(op, "weight shape disabled via env");
          return false;
        }
        begin = end + 1;
      }
    }
  }
  // Debug switch to keep sub-byte (INT4/INT2) weight matmuls on the CPU.
  if (EnvEnabled("LITERT_NVIDIA_TENSORRT_DISABLE_SUBBYTE_WEIGHTS",
                 /*default_value=*/false) &&
      (op.Inputs()[1].ElementType() == litert::ElementType::Int4 ||
       op.Inputs()[1].ElementType() == litert::ElementType::Int2)) {
    LogFullyConnectedReject(op, "sub-byte weights disabled via env");
    return false;
  }
  bool weights_quantized = false;
  if (op.Inputs()[1].ElementType() == litert::ElementType::Int8 ||
      op.Inputs()[1].ElementType() == litert::ElementType::Int4 ||
      op.Inputs()[1].ElementType() == litert::ElementType::Int2) {
    LITERT_ASSIGN_OR_RETURN(weights_quantized, HasSymmetricTensorRtQuantization(
                                                   op.Inputs()[1],
                                                   /*allow_per_channel=*/true));
    // Sub-byte weights are repacked at engine-build time, so they must be
    // compile-time constants.
    if (op.Inputs()[1].ElementType() != litert::ElementType::Int8 &&
        !op.Inputs()[1].HasWeights()) {
      weights_quantized = false;
    }
  }
  if (!weights_float && !weights_quantized) {
    LogFullyConnectedReject(
        op, "weights are neither float nor supported quantized");
    return false;
  }
  if (op.Inputs().size() == 3) {
    LITERT_ASSIGN_OR_RETURN(bool bias_supported,
                            IsStaticFloatTensor(op.Inputs()[2]));
    if (!op.Inputs()[2].HasWeights() || !bias_supported) {
      LogFullyConnectedReject(op, "bias is not constant static float");
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto weights_type, op.Inputs()[1].RankedTensorType());
  if (input_type.Layout().Rank() < 2 || weights_type.Layout().Rank() != 2) {
    LogFullyConnectedReject(op, "input or weights rank is unsupported");
    return false;
  }
  uint32_t weights_format = litert::kFullyConnectedOptionsWeightsFormatDefault;
  if (LiteRtGetFullyConnectedWeightsFormatOption(op.Get(), &weights_format) !=
          kLiteRtStatusOk ||
      weights_format != litert::kFullyConnectedOptionsWeightsFormatDefault) {
    LogFullyConnectedReject(op, "unsupported weights format option");
    return false;
  }
  bool keep_num_dims = false;
  if (LiteRtGetFullyConnectedKeepNumDimsOption(op.Get(), &keep_num_dims) !=
      kLiteRtStatusOk) {
    LogFullyConnectedReject(op, "failed to read keep_num_dims option");
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool activation_supported,
                          HasSupportedFusedActivation(op));
  if (!activation_supported) {
    LogFullyConnectedReject(op, "unsupported fused activation");
  }
  return activation_supported;
}

Expected<bool> IsConcatenationSupported(const Op& op) {
  if (op.Inputs().empty() || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!output_supported) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  const int rank = output_type.Layout().Rank();
  for (const auto& input : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool input_supported,
                            IsStaticTensorRtTensor(input));
    if (!input_supported || !SameElementType(input, op.Outputs()[0])) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto input_type, input.RankedTensorType());
    if (input_type.Layout().Rank() != rank) {
      return false;
    }
  }
  int32_t axis = 0;
  if (LiteRtGetConcatenationAxisOption(op.Get(), &axis) != kLiteRtStatusOk) {
    return false;
  }
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return false;
  }
  return HasSupportedFusedActivation(op);
}

Expected<bool> IsSelectV2Supported(const Op& op) {
  if (op.Inputs().size() != 3 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool cond_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool then_supported,
                          IsStaticTensorRtTensor(op.Inputs()[1]));
  LITERT_ASSIGN_OR_RETURN(bool else_supported,
                          IsStaticTensorRtTensor(op.Inputs()[2]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!cond_supported || !then_supported || !else_supported ||
      !output_supported) {
    return false;
  }
  return op.Inputs()[0].ElementType() == litert::ElementType::Bool &&
         SameElementType(op.Inputs()[1], op.Inputs()[2]) &&
         SameElementType(op.Inputs()[1], op.Outputs()[0]);
}

Expected<bool> IsSumSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticFloatTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported) {
    return false;
  }
  auto axes = ReadInt32Constant(op.Inputs()[1]);
  if (!axes) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const int rank = input_type.Layout().Rank();
  if (rank <= 0 || rank >= 32) {
    return false;
  }
  for (int32_t axis : *axes) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      return false;
    }
  }
  bool keep_dims = false;
  return LiteRtGetSumKeepDimsOption(op.Get(), &keep_dims) == kLiteRtStatusOk;
}

Expected<bool> IsReduceMaxSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported ||
      !IsTensorRtArithmeticElementType(op.Inputs()[0].ElementType()) ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  auto axes = ReadInt32Constant(op.Inputs()[1]);
  if (!axes) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const int rank = input_type.Layout().Rank();
  if (rank <= 0 || rank >= 32) {
    return false;
  }
  for (int32_t axis : *axes) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      return false;
    }
  }
  bool keep_dims = false;
  return LiteRtGetReduceMaxKeepDimsOption(op.Get(), &keep_dims) ==
         kLiteRtStatusOk;
}

Expected<bool> HasSymmetricTensorRtQuantization(const Tensor& tensor,
                                                bool allow_per_channel) {
  switch (tensor.QTypeId()) {
    case kLiteRtQuantizationPerTensor: {
      auto q = tensor.PerTensorQuantization();
      return q.scale > 0.0f && q.zero_point == 0;
    }
    case kLiteRtQuantizationPerChannel: {
      if (!allow_per_channel) {
        return false;
      }
      auto q = tensor.PerChannelQuantization();
      if (q.num_channels == 0 || q.scales == nullptr ||
          q.zero_points == nullptr) {
        return false;
      }
      for (uint64_t i = 0; i < q.num_channels; ++i) {
        if (q.scales[i] <= 0.0f || q.zero_points[i] != 0) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

Expected<bool> IsQuantizeSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported ||
      !IsFloatLike(op.Inputs()[0].ElementType()) ||
      !IsTensorRtQuantizedElementType(op.Outputs()[0].ElementType())) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  if (input_type.Layout().Rank() == 0) {
    return false;
  }
  return HasSymmetricTensorRtQuantization(
      op.Outputs()[0], /*allow_per_channel=*/op.Inputs()[0].HasWeights());
}

Expected<bool> IsDequantizeSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  const Tensor input = op.Inputs()[0];
  LITERT_ASSIGN_OR_RETURN(bool input_static, IsStaticRankedTensor(input));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  // Runtime tensors must be a TensorRT-representable quantized type. INT2 is
  // additionally allowed for constants, which are repacked to INT4 weights.
  const bool input_type_ok =
      IsTensorRtQuantizedElementType(input.ElementType()) ||
      (input.HasWeights() && input.ElementType() == litert::ElementType::Int2);
  if (!input_static || !output_supported || !input_type_ok ||
      !IsFloatLike(op.Outputs()[0].ElementType())) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, input.RankedTensorType());
  if (input_type.Layout().Rank() == 0) {
    return false;
  }
  return HasSymmetricTensorRtQuantization(
      input, /*allow_per_channel=*/input.HasWeights());
}

Expected<bool> IsFillSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (!ReadInt32Constant(op.Inputs()[0])) {
    return false;
  }
  if (!op.Inputs()[1].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto value_type, op.Inputs()[1].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  LITERT_ASSIGN_OR_RETURN(size_t value_elements, NumElements(value_type));
  LITERT_ASSIGN_OR_RETURN(size_t output_bytes,
                          StaticTensorByteSize(op.Outputs()[0]));
  if (output_bytes > MaxTensorRtFillBytes()) {
    return false;
  }
  return output_supported && value_elements == 1 &&
         value_type.ElementType() == output_type.ElementType();
}

Expected<bool> IsPackSupported(const Op& op) {
  if (op.Inputs().empty() || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!output_supported) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  const int output_rank = output_type.Layout().Rank();
  if (output_rank <= 0) {
    return false;
  }
  int32_t axis = 0;
  int32_t values_count = 0;
  if (LiteRtGetPackAxisOption(op.Get(), &axis) != kLiteRtStatusOk ||
      LiteRtGetPackValuesCountOption(op.Get(), &values_count) !=
          kLiteRtStatusOk ||
      values_count != static_cast<int32_t>(op.Inputs().size())) {
    return false;
  }
  if (axis < 0) {
    axis += output_rank;
  }
  if (axis < 0 || axis >= output_rank ||
      output_type.Layout().Dimensions()[axis] != values_count) {
    return false;
  }
  for (const auto& input : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool input_supported,
                            IsStaticTensorRtTensor(input));
    if (!input_supported || !SameElementType(input, op.Outputs()[0])) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto input_type, input.RankedTensorType());
    if (input_type.Layout().Rank() + 1 != output_rank) {
      return false;
    }
  }
  return true;
}

Expected<bool> IsUnpackSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().empty()) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  if (!input_supported) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const int input_rank = input_type.Layout().Rank();
  if (input_rank <= 0) {
    return false;
  }
  int32_t axis = 0;
  int32_t num = 0;
  if (LiteRtGetUnpackAxisOption(op.Get(), &axis) != kLiteRtStatusOk ||
      LiteRtGetUnpackNumOption(op.Get(), &num) != kLiteRtStatusOk ||
      num != static_cast<int32_t>(op.Outputs().size())) {
    return false;
  }
  if (axis < 0) {
    axis += input_rank;
  }
  if (axis < 0 || axis >= input_rank ||
      input_type.Layout().Dimensions()[axis] != num) {
    return false;
  }
  for (const auto& output : op.Outputs()) {
    LITERT_ASSIGN_OR_RETURN(bool output_supported,
                            IsStaticTensorRtTensor(output));
    if (!output_supported || !SameElementType(op.Inputs()[0], output)) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto output_type, output.RankedTensorType());
    if (output_type.Layout().Rank() + 1 != input_rank) {
      return false;
    }
  }
  return true;
}

Expected<bool> IsFloorModSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticTensorRtTensor(tensor));
    if (!supported || tensor.ElementType() != litert::ElementType::Int32) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  return output_supported &&
         op.Outputs()[0].ElementType() == litert::ElementType::Int32;
}

Expected<bool> IsConv2dSupported(const Op& op) {
  if (op.Inputs().size() < 2 || op.Inputs().size() > 3 ||
      op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticFloatTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported || !op.Inputs()[1].HasWeights()) {
    return false;
  }
  if (op.Inputs().size() == 3 && !op.Inputs()[2].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto filter_type, op.Inputs()[1].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  if (input_type.Layout().Rank() != 4 || filter_type.Layout().Rank() != 4 ||
      output_type.Layout().Rank() != 4) {
    return false;
  }
  uint32_t padding = litert::kPaddingSame;
  if (LiteRtGetConv2dPaddingOption(op.Get(), &padding) != kLiteRtStatusOk ||
      padding > litert::kPaddingValid) {
    return false;
  }
  return HasSupportedFusedActivation(op);
}

std::string TensorName(const char* prefix, const Tensor& tensor, int index) {
  return std::string(prefix) + "_" + std::to_string(index) + "_tensor_" +
         std::to_string(tensor.TensorIndex());
}

bool TensorHasShape(const Tensor& tensor,
                    std::initializer_list<int32_t> expected) {
  auto type = tensor.RankedTensorType();
  if (!type) {
    return false;
  }
  const auto dimensions = type->Layout().Dimensions();
  return dimensions.size() == expected.size() &&
         std::equal(dimensions.begin(), dimensions.end(), expected.begin());
}

bool IsUnquantizedFloat32(const Tensor& tensor) {
  return tensor.ElementType() == litert::ElementType::Float32 &&
         tensor.QTypeId() == kLiteRtQuantizationNone;
}

bool IsDefinedBy(const Tensor& tensor, const Op& op) {
  const auto defining_op = tensor.DefiningOp();
  return defining_op.has_value() && defining_op->op == op.Get();
}

bool HasOnlyUse(const Tensor& tensor, const Op& op,
                LiteRtParamIndex input_index) {
  const auto uses = tensor.Uses();
  return uses.size() == 1 && uses[0].user.Get() == op.Get() &&
         uses[0].user_arg_ind == input_index;
}

std::optional<uint32_t> FindSubgraphOutputPort(const Subgraph& subgraph,
                                               const Tensor& tensor) {
  const auto outputs = subgraph.Outputs();
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].Get() == tensor.Get()) {
      return i;
    }
  }
  return std::nullopt;
}

std::optional<float> ReadFloat32Scalar(const Tensor& tensor) {
  if (!tensor.HasWeights() || !IsUnquantizedFloat32(tensor)) {
    return std::nullopt;
  }
  auto type = tensor.RankedTensorType();
  if (!type) {
    return std::nullopt;
  }
  auto num_elements = NumElements(*type);
  const auto bytes = tensor.Weights().Bytes();
  if (!num_elements || *num_elements != 1 || bytes.size() != sizeof(float)) {
    return std::nullopt;
  }
  float value = 0.0f;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

uint16_t Float32ToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

bool HasNoFusedActivation(const Op& op) {
  auto activation = GetFusedActivation(op);
  return activation.HasValue() &&
         *activation == litert::kActivationFunctionTypeNone;
}

Expected<std::optional<TensorRtLlmHeadBuildData>> MatchTensorRtLlmHead(
    const Subgraph& subgraph) {
  constexpr uint32_t kGemma4HiddenSize = 1536;
  constexpr uint32_t kGemma4VocabSize = 262144;
  constexpr float kGemma4SoftCap = 30.0f;
  constexpr size_t kTailOpCount = 4;

  const auto no_match = [] {
    return std::optional<TensorRtLlmHeadBuildData>();
  };
  const auto ops = subgraph.Ops();
  if (ops.size() < kTailOpCount) {
    return no_match();
  }
  const Op& fc = ops[ops.size() - 4];
  const Op& scale_down = ops[ops.size() - 3];
  const Op& tanh = ops[ops.size() - 2];
  const Op& scale_up = ops[ops.size() - 1];
  if (fc.Code() != kLiteRtOpCodeTflFullyConnected ||
      scale_down.Code() != kLiteRtOpCodeTflMul ||
      tanh.Code() != kLiteRtOpCodeTflTanh ||
      scale_up.Code() != kLiteRtOpCodeTflMul) {
    return no_match();
  }

  const auto fc_inputs = fc.Inputs();
  const auto fc_outputs = fc.Outputs();
  const auto scale_down_inputs = scale_down.Inputs();
  const auto scale_down_outputs = scale_down.Outputs();
  const auto tanh_inputs = tanh.Inputs();
  const auto tanh_outputs = tanh.Outputs();
  const auto scale_up_inputs = scale_up.Inputs();
  const auto scale_up_outputs = scale_up.Outputs();
  if (fc_inputs.size() != 2 || fc_outputs.size() != 1 ||
      scale_down_inputs.size() != 2 || scale_down_outputs.size() != 1 ||
      tanh_inputs.size() != 1 || tanh_outputs.size() != 1 ||
      scale_up_inputs.size() != 2 || scale_up_outputs.size() != 1) {
    return no_match();
  }

  const Tensor& hidden = fc_inputs[0];
  const Tensor& weights = fc_inputs[1];
  const Tensor& fc_output = fc_outputs[0];
  const Tensor& scale_down_output = scale_down_outputs[0];
  const Tensor& tanh_output = tanh_outputs[0];
  const Tensor& logits = scale_up_outputs[0];
  if (scale_down_inputs[0].Get() != fc_output.Get() ||
      tanh_inputs[0].Get() != scale_down_output.Get() ||
      scale_up_inputs[0].Get() != tanh_output.Get() ||
      !IsDefinedBy(fc_output, fc) ||
      !IsDefinedBy(scale_down_output, scale_down) ||
      !IsDefinedBy(tanh_output, tanh) || !IsDefinedBy(logits, scale_up) ||
      !HasOnlyUse(fc_output, scale_down, 0) ||
      !HasOnlyUse(scale_down_output, tanh, 0) ||
      !HasOnlyUse(tanh_output, scale_up, 0) || !logits.Uses().empty()) {
    return no_match();
  }

  const auto hidden_port = FindSubgraphOutputPort(subgraph, hidden);
  const auto logits_port = FindSubgraphOutputPort(subgraph, logits);
  if (!hidden_port.has_value() || !logits_port.has_value() ||
      *hidden_port == *logits_port || !hidden.DefiningOp().has_value() ||
      !HasOnlyUse(hidden, fc, 0)) {
    return no_match();
  }

  if (!IsUnquantizedFloat32(hidden) || !IsUnquantizedFloat32(fc_output) ||
      !IsUnquantizedFloat32(scale_down_output) ||
      !IsUnquantizedFloat32(tanh_output) || !IsUnquantizedFloat32(logits) ||
      !TensorHasShape(hidden,
                      {1, 1, static_cast<int32_t>(kGemma4HiddenSize)}) ||
      !TensorHasShape(weights, {static_cast<int32_t>(kGemma4VocabSize),
                                static_cast<int32_t>(kGemma4HiddenSize)}) ||
      !TensorHasShape(fc_output,
                      {1, 1, static_cast<int32_t>(kGemma4VocabSize)}) ||
      !TensorHasShape(scale_down_output,
                      {1, 1, static_cast<int32_t>(kGemma4VocabSize)}) ||
      !TensorHasShape(tanh_output,
                      {1, 1, static_cast<int32_t>(kGemma4VocabSize)}) ||
      !TensorHasShape(logits, {1, 1, static_cast<int32_t>(kGemma4VocabSize)})) {
    return no_match();
  }

  if (!weights.HasWeights() ||
      weights.ElementType() != litert::ElementType::Int2 ||
      weights.QTypeId() != kLiteRtQuantizationPerChannel ||
      !HasNoFusedActivation(fc) || !HasNoFusedActivation(scale_down) ||
      !HasNoFusedActivation(scale_up)) {
    return no_match();
  }
  uint32_t weights_format = litert::kFullyConnectedOptionsWeightsFormatDefault;
  bool keep_num_dims = false;
  if (LiteRtGetFullyConnectedWeightsFormatOption(fc.Get(), &weights_format) !=
          kLiteRtStatusOk ||
      weights_format != litert::kFullyConnectedOptionsWeightsFormatDefault ||
      LiteRtGetFullyConnectedKeepNumDimsOption(fc.Get(), &keep_num_dims) !=
          kLiteRtStatusOk ||
      !keep_num_dims) {
    return no_match();
  }

  const auto down = ReadFloat32Scalar(scale_down_inputs[1]);
  const auto cap = ReadFloat32Scalar(scale_up_inputs[1]);
  if (!down.has_value() || !cap.has_value() || *cap != kGemma4SoftCap ||
      *down != 1.0f / kGemma4SoftCap) {
    return no_match();
  }

  const auto quantization = weights.PerChannelQuantization();
  if (quantization.quantized_dimension != 0 ||
      quantization.num_channels != kGemma4VocabSize ||
      quantization.scales == nullptr || quantization.zero_points == nullptr) {
    return no_match();
  }
  for (uint64_t i = 0; i < quantization.num_channels; ++i) {
    if (!std::isfinite(quantization.scales[i]) ||
        quantization.scales[i] <= 0.0f || quantization.zero_points[i] != 0) {
      return no_match();
    }
  }

  const auto raw_weights = weights.Weights().Bytes();
  const uint64_t num_elements =
      static_cast<uint64_t>(kGemma4HiddenSize) * kGemma4VocabSize;
  if (raw_weights.size() != (num_elements + 3) / 4) {
    return no_match();
  }
  TensorRtLlmHeadBuildData result;
  result.hidden_output_port = *hidden_port;
  result.logits_output_port = *logits_port;
  result.k = kGemma4HiddenSize;
  result.n = kGemma4VocabSize;
  result.soft_cap = kGemma4SoftCap;
  result.weight_format = TensorRtLlmHeadWeightFormat::kInt2TfliteRowMajor;
  result.packed_weights = raw_weights;
  result.bf16_scales.resize(kGemma4VocabSize * sizeof(uint16_t));
  for (uint32_t i = 0; i < kGemma4VocabSize; ++i) {
    const uint16_t bits = Float32ToBf16Bits(quantization.scales[i]);
    result.bf16_scales[2 * i] = static_cast<uint8_t>(bits);
    result.bf16_scales[2 * i + 1] = static_cast<uint8_t>(bits >> 8);
  }
  return std::optional<TensorRtLlmHeadBuildData>(std::move(result));
}

std::optional<int> CurrentCudaComputeCapability() {
  int device = 0;
  cudaDeviceProp properties{};
  const cudaError_t get_device_status = cudaGetDevice(&device);
  if (get_device_status != cudaSuccess) {
    LITERT_LOG(LITERT_WARNING, "NVIDIA CUDA device query failed: %s",
               cudaGetErrorString(get_device_status));
    return std::nullopt;
  }
  const cudaError_t properties_status =
      cudaGetDeviceProperties(&properties, device);
  if (properties_status != cudaSuccess) {
    LITERT_LOG(LITERT_WARNING, "NVIDIA CUDA device properties query failed: %s",
               cudaGetErrorString(properties_status));
    return std::nullopt;
  }
  return properties.major * 10 + properties.minor;
}

bool RunningUnderWsl() {
  static const bool is_wsl = [] {
    std::FILE* f = std::fopen("/proc/version", "r");
    if (f == nullptr) {
      return false;
    }
    char buffer[512] = {};
    const size_t read = std::fread(buffer, 1, sizeof(buffer) - 1, f);
    std::fclose(f);
    (void)read;
    for (char& c : buffer) {
      c = std::tolower(static_cast<unsigned char>(c));
    }
    return std::strstr(buffer, "microsoft") != nullptr;
  }();
  return is_wsl;
}

// WSL2's stream-ordered allocator (cudaMallocAsync) pools cannot grow to the
// sizes the TensorRT builder requests for large partitions, so builder
// allocations fail with spurious OOMs while plain cudaMalloc of the same size
// succeeds. Route builder allocations through synchronous cudaMalloc there.
class SyncCudaGpuAllocator : public nvinfer1::IGpuAllocator {
 public:
  void* allocate(uint64_t const size, uint64_t const /*alignment*/,
                 nvinfer1::AllocatorFlags const /*flags*/) noexcept override {
    if (size == 0) {
      return nullptr;
    }
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
      return nullptr;
    }
    return ptr;
  }

  bool deallocate(void* const memory) noexcept override {
    return cudaFree(memory) == cudaSuccess;
  }

  void* allocateAsync(uint64_t const size, uint64_t const alignment,
                      nvinfer1::AllocatorFlags const flags,
                      cudaStream_t /*stream*/) noexcept override {
    return allocate(size, alignment, flags);
  }

  bool deallocateAsync(void* const memory,
                       cudaStream_t /*stream*/) noexcept override {
    // cudaFree synchronizes the device, so in-flight work that uses this
    // memory completes before it is released.
    return deallocate(memory);
  }
};

bool UseSyncCudaAllocator() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_SYNC_ALLOCATOR");
  if (value != nullptr && value[0] != '\0') {
    return std::strcmp(value, "0") != 0;
  }
  return RunningUnderWsl();
}

struct OwnedRefitWeight {
  std::string name;
  int32_t data_type = 0;
  uint64_t count = 0;
  size_t owned_weight_index = 0;
};

nvinfer1::Permutation MakePermutation(std::initializer_list<int32_t> values) {
  nvinfer1::Permutation permutation{};
  int index = 0;
  for (int32_t value : values) {
    permutation.order[index++] = value;
  }
  return permutation;
}

class TensorRtGraphBuilder {
 public:
  Expected<TensorRtBuildResult> Build(const Subgraph& subgraph) {
    LogMemoryProfile("compiler", "graph_build_begin");
    builder_.reset(nvinfer1::createInferBuilder(logger_));
    if (!builder_) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create TensorRT builder");
    }
    if (UseSyncCudaAllocator()) {
      builder_->setGpuAllocator(&sync_allocator_);
    }

    network_.reset(builder_->createNetworkV2(/*flags=*/0));
    if (!network_) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create TensorRT network");
    }

    config_.reset(builder_->createBuilderConfig());
    if (!config_) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create TensorRT builder config");
    }
    config_->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                TensorRtWorkspaceBytes());
    // Lower levels profile fewer tactics, which cuts engine-build time and
    // the autotuner's transient device-memory footprint on large partitions.
    const char* opt_level_env =
        std::getenv("LITERT_NVIDIA_TENSORRT_BUILDER_OPT_LEVEL");
    if (opt_level_env != nullptr && opt_level_env[0] != '\0') {
      char* end = nullptr;
      const long parsed = std::strtol(opt_level_env, &end, 10);
      if (end != opt_level_env && *end == '\0' && parsed >= 0 && parsed <= 5) {
        config_->setBuilderOptimizationLevel(static_cast<int32_t>(parsed));
      }
    }
    // TensorRT derives the tactic-DRAM budget from free device memory, which
    // is unreliable under WSL; allow pinning it explicitly.
    const size_t tactic_dram_mb =
        EnvSizeT("LITERT_NVIDIA_TENSORRT_TACTIC_DRAM_MB", 0);
    if (tactic_dram_mb > 0) {
      config_->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACTIC_DRAM,
                                  tactic_dram_mb << 20);
    }
    if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_ALLOW_TF32",
                    /*default_value=*/false)) {
      config_->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }
    if (TensorRtSharedWeightsEnabled()) {
      // Only constants explicitly registered below are refittable. Making all
      // builder-generated weights refittable produces an archive that the
      // TensorRT-RTX recorder cannot serialize for these large graphs.
      config_->setFlag(nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL);
    }
#if !defined(TRT_MAJOR_RTX)
    if (std::getenv("LITERT_NVIDIA_TENSORRT_FP16") != nullptr) {
      config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
#else
    if (std::getenv("LITERT_NVIDIA_TENSORRT_FP16") != nullptr) {
      LITERT_LOG(LITERT_WARNING,
                 "Ignoring LITERT_NVIDIA_TENSORRT_FP16 because TensorRT-RTX "
                 "uses strongly typed networks and does not support "
                 "BuilderFlag::kFP16");
    }
#endif

    LogMemoryProfile("compiler", "builder_configured");

    const auto ops = subgraph.Ops();
    std::optional<TensorRtLlmHeadBuildData> trtllm_head;
    const std::optional<int> compute_capability =
        CurrentCudaComputeCapability();
    if (compute_capability.has_value() &&
        IsInt2GemvComputeCapabilitySupported(*compute_capability)) {
      LITERT_ASSIGN_OR_RETURN(trtllm_head, MatchTensorRtLlmHead(subgraph));
      if (trtllm_head.has_value()) {
        LITERT_LOG(
            LITERT_INFO,
            "NVIDIA TensorRT-RTX external W2 head matched: hidden_port=%u "
            "logits_port=%u K=%u N=%u weights_bytes=%zu scales_bytes=%zu",
            trtllm_head->hidden_output_port, trtllm_head->logits_output_port,
            trtllm_head->k, trtllm_head->n, trtllm_head->packed_weights.size(),
            trtllm_head->bf16_scales.size());
      }
    }

    LITERT_RETURN_IF_ERROR(AddInputs(subgraph));
    const size_t num_ops_to_lower =
        ops.size() - (trtllm_head.has_value() ? 4 : 0);
    for (size_t i = 0; i < num_ops_to_lower; ++i) {
      LITERT_RETURN_IF_ERROR(LowerOp(ops[i]));
    }
    LITERT_RETURN_IF_ERROR(MarkOutputs(subgraph, trtllm_head));
    LogMemoryProfile("compiler", "graph_lowered");

    // The CUDA GEMV plugin receives packed subbyte weights through an INT8
    // constant input. Those weights cannot use the safe stripping path below,
    // so keep the decode plan portable and self-contained instead of making it
    // refittable only for its comparatively tiny scale constants.
    const bool strip_plan =
        TensorRtSharedWeightsEnabled() && !uses_cuda_subbyte_gemv_;
    if (strip_plan) {
      // Weight-stripped TensorRT-RTX plans require a GPU build targeting at
      // most one compute capability. Ordinary plans retain RTX's portable
      // multi-architecture default.
      if (!config_->setNbComputeCapabilities(1) ||
          !config_->setComputeCapability(nvinfer1::ComputeCapability::kCURRENT,
                                         0)) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to target the current GPU for a stripped plan");
      }
      config_->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
      size_t refit_weight_bytes = 0;
      for (const auto& weight : owned_refit_weights_) {
        refit_weight_bytes += owned_weights_[weight.owned_weight_index].size();
      }
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX building a stripped selectively "
                 "refittable plan with %zu weights (%zu bytes)",
                 owned_refit_weights_.size(), refit_weight_bytes);
    } else if (TensorRtSharedWeightsEnabled()) {
      config_->clearFlag(nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL);
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX keeping CUDA GEMV plugin plan "
                 "self-contained because its packed INT8 plugin weights "
                 "cannot be stripped safely by this SDK");
    }

    LogMemoryProfile("compiler", "engine_serialize_begin");
    TrtPtr<nvinfer1::IHostMemory> serialized(
        builder_->buildSerializedNetwork(*network_, *config_));
    if (!serialized) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to build serialized TensorRT engine");
    }
    LogMemoryProfile("compiler", "engine_serialize_end");
    if (EnvEnabled("LITERT_NVIDIA_TENSORRT_DUMP_ENGINE_INFO",
                   /*default_value=*/false)) {
      TrtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger_));
      if (runtime) {
        TrtPtr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
            serialized->data(), serialized->size()));
        if (engine) {
          TrtPtr<nvinfer1::IEngineInspector> inspector(
              engine->createEngineInspector());
          if (inspector) {
            const char* info = inspector->getEngineInformation(
                nvinfer1::LayerInformationFormat::kONELINE);
            LITERT_LOG(LITERT_INFO, "TensorRT engine layers:\n%s",
                       info != nullptr ? info : "(null)");
          }
        }
      }
    }

    TensorRtBuildResult result;
    result.input_names = input_names_;
    result.output_names = output_names_;
    result.trtllm_head = std::move(trtllm_head);
    result.is_stripped_plan = strip_plan;
    // TensorRT-RTX has consumed every network constant at this point. Destroy
    // the network before copying its serialized plan into the result so the
    // compiler-owned constants do not overlap that additional engine copy.
    network_.reset();
    if (strip_plan) {
      result.refit_weights.reserve(owned_refit_weights_.size());
      for (auto& weight : owned_refit_weights_) {
        result.refit_weights.push_back(
            {std::move(weight.name),
             static_cast<TensorRtWeightDataType>(weight.data_type),
             weight.count,
             std::move(owned_weights_[weight.owned_weight_index])});
      }
    }
    owned_weights_.clear();
    owned_weights_.shrink_to_fit();
    result.engine.resize(serialized->size());
    std::memcpy(result.engine.data(), serialized->data(), serialized->size());
    LogMemoryProfile("compiler", "engine_copied");
    return result;
  }

 private:
  // FP16-activation mode: activations flow through the network as FP16 real
  // values; int8 activation quantization survives only at island boundaries
  // (one dequantize per input, one quantize per output) and weights keep
  // their weight-only-quantized form inside the matmuls. This collapses the
  // per-op Cast/Mul Q-DQ chains that otherwise fragment Myelin fusion into
  // thousands of kernels per step.
  static bool Fp16ActivationsEnabled() {
    const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }

  // Limits odml.runtime_bmm to a static KV-cache prefix, then restores the
  // graph's fixed output shape where needed. This is correct only when the
  // runtime never attends beyond the configured prefix.
  static size_t RuntimeBmmContextLimit() {
    return EnvSizeT("LITERT_NVIDIA_TENSORRT_RUNTIME_BMM_CONTEXT_LIMIT",
                    /*default_value=*/0);
  }

  // Materializing dequantized weight constants removes the per-step
  // Cast/Mul/Move weight-dequantize kernels that Myelin otherwise runs on
  // every invocation (it has no fused sub-byte weight-only GEMV at M=1).
  //   kFloat: fp16/bf16 constants (2 bytes/element).
  //   kFp8:   fp8-e4m3 value constants + per-channel scale dequantize, which
  //           Myelin fuses into a single GEMV that reads 1 byte/element.
  //           int4/int2 integer values are exactly representable in e4m3.
  // kCudaGemv keeps packed INT2/INT4 weights and performs dequantization and
  // GEMV in one native CUDA plugin without materializing expanded weights.
  enum class PredequantMode { kOff, kFloat, kFp8, kCudaGemv };

  static PredequantMode PredequantizeFcWeightsMode() {
    const char* value =
        std::getenv("LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
      return PredequantMode::kOff;
    }
    if (std::strcmp(value, "fp8") == 0) {
      return PredequantMode::kFp8;
    }
    if (std::strcmp(value, "cuda_gemv") == 0) {
      return PredequantMode::kCudaGemv;
    }
    return PredequantMode::kFloat;
  }

  // bf16 keeps fp32's exponent range, sidestepping overflow in reductions
  // and exp-heavy regions at slightly coarser mantissa.
  static nvinfer1::DataType ModeFloatType() {
    const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS");
    if (value != nullptr && std::strcmp(value, "bf16") == 0) {
      return nvinfer1::DataType::kBF16;
    }
    return nvinfer1::DataType::kHALF;
  }

  const char* KeepName(std::string name) {
    names_.push_back(std::move(name));
    return names_.back().c_str();
  }

  std::string UniqueName(std::string base) {
    base += "_trt_";
    base += std::to_string(next_name_id_++);
    return base;
  }

  Expected<void> RegisterRefitWeight(nvinfer1::Weights weights,
                                     const std::string& base_name) {
    if (!TensorRtSharedWeightsEnabled() || weights.values == nullptr ||
        weights.count <= 0) {
      return {};
    }
    // TensorRT-RTX 1.5.0 build 114 fails in stdArchiveRecorder when an INT8
    // constant is marked for stripping, even though equivalent INT4 and
    // floating-point constants serialize and refit correctly. Leave INT8
    // weights embedded until the SDK recorder supports them.
    if (weights.type == nvinfer1::DataType::kINT8) {
      return {};
    }
    for (const auto& existing : owned_refit_weights_) {
      if (owned_weights_[existing.owned_weight_index].data() ==
          weights.values) {
        return {};
      }
    }
    size_t owned_weight_index = owned_weights_.size();
    for (size_t i = 0; i < owned_weights_.size(); ++i) {
      if (owned_weights_[i].data() == weights.values) {
        owned_weight_index = i;
        break;
      }
    }
    if (owned_weight_index == owned_weights_.size()) {
      return Error(kLiteRtStatusErrorCompilation,
                   "TensorRT refit weight is not compiler-owned");
    }
    std::string name = UniqueName("refit_" + base_name);
    const char* kept_name = KeepName(name);
    if (!network_->setWeightsName(weights, kept_name)) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to name TensorRT refit weight");
    }
    if (!network_->markWeightsRefittable(kept_name)) {
      LITERT_LOG(LITERT_VERBOSE,
                 "NVIDIA TensorRT-RTX weight cannot be refitted: %s",
                 kept_name);
      return {};
    }
    owned_refit_weights_.push_back(
        {std::move(name), static_cast<int32_t>(weights.type),
         static_cast<uint64_t>(weights.count), owned_weight_index});
    return {};
  }

  Expected<void> AddInputs(const Subgraph& subgraph) {
    auto inputs = subgraph.Inputs();
    input_names_.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      LITERT_ASSIGN_OR_RETURN(auto type, input.RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
      LITERT_ASSIGN_OR_RETURN(auto data_type,
                              ConvertDataType(type.ElementType()));
      std::string name = TensorName("input", input, i);
      auto* tensor = network_->addInput(KeepName(name), data_type, dims);
      if (tensor == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT input");
      }
      raw_input_map_[input.Get()] = tensor;
      nvinfer1::ITensor* value = tensor;
      if (Fp16ActivationsEnabled()) {
        LITERT_ASSIGN_OR_RETURN(bool int8_quantized,
                                HasSymmetricPerTensorInt8Quantization(input));
        if (int8_quantized) {
          LITERT_ASSIGN_OR_RETURN(
              value, AddDequantizeTensor(input, tensor, ModeFloatType()));
        } else if (data_type == nvinfer1::DataType::kFLOAT) {
          LITERT_ASSIGN_OR_RETURN(value,
                                  AddCastTensor(tensor, ModeFloatType()));
        }
      }
      tensor_map_[input.Get()] = value;
      input_names_.push_back(name);
    }
    return {};
  }

  Expected<void> MarkOutputs(
      const Subgraph& subgraph,
      const std::optional<TensorRtLlmHeadBuildData>& trtllm_head) {
    auto outputs = subgraph.Outputs();
    output_names_.reserve(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      if (trtllm_head.has_value() && i == trtllm_head->logits_output_port) {
        // LiteRT still exposes this port; NVIDIA dispatch writes it with the
        // external W2 head after the TensorRT-RTX prefix.
        output_names_.emplace_back();
        continue;
      }
      const auto& output = outputs[i];
      auto it = tensor_map_.find(output.Get());
      if (it == tensor_map_.end() || it->second == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Subgraph output was not produced by TensorRT network");
      }
      nvinfer1::ITensor* out = it->second;
      if (Fp16ActivationsEnabled()) {
        // Prefer a raw boundary-format tensor when one was recorded (cache
        // updates), otherwise convert the in-network FP16 value back to the
        // tensor's boundary format (int8 for quantized, fp32 for float).
        if (auto raw = raw_output_map_.find(output.Get());
            raw != raw_output_map_.end()) {
          out = raw->second;
        }
        LITERT_ASSIGN_OR_RETURN(auto type, output.RankedTensorType());
        LITERT_ASSIGN_OR_RETURN(auto boundary_type,
                                ConvertDataType(type.ElementType()));
        if (out->getType() != boundary_type) {
          LITERT_ASSIGN_OR_RETURN(
              bool int8_quantized,
              HasSymmetricPerTensorInt8Quantization(output));
          if (int8_quantized && out->getType() != nvinfer1::DataType::kINT8) {
            LITERT_ASSIGN_OR_RETURN(
                out, AddQuantizeTensor(out, output, nvinfer1::DataType::kINT8));
          } else if (!int8_quantized) {
            LITERT_ASSIGN_OR_RETURN(out, AddCastTensor(out, boundary_type));
          }
        }
      }
      std::string name = TensorName("output", output, i);
      out->setName(KeepName(name));
      network_->markOutput(*out);
      output_names_.push_back(name);
    }
    return {};
  }

  Expected<nvinfer1::Weights> MakeWeights(const Tensor& tensor) {
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
    if (type.ElementType() == litert::ElementType::Int2) {
      return MakeInt2WeightsAsInt4(tensor, num_elements);
    }
    auto data_type_or = ConvertDataType(type.ElementType());
    if (!data_type_or) {
      return Error(data_type_or.Error().Status(),
                   data_type_or.Error().Message() + " for constant tensor " +
                       std::to_string(tensor.TensorIndex()));
    }
    auto data_type = *data_type_or;
    size_t expected_bytes = 0;
    if (type.ElementType() == litert::ElementType::Int4) {
      expected_bytes = (num_elements + 1) / 2;
    } else {
      LITERT_ASSIGN_OR_RETURN(size_t element_size,
                              ElementByteSize(type.ElementType()));
      expected_bytes = num_elements * element_size;
    }
    const auto bytes = tensor.Weights().Bytes();
    if (bytes.size() != expected_bytes) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected TensorRT weight buffer size");
    }
    owned_weights_.emplace_back(bytes.begin(), bytes.end());
    return nvinfer1::Weights{data_type, owned_weights_.back().data(),
                             static_cast<int64_t>(num_elements)};
  }

  // TensorRT has no 2-bit type; widen TFLite INT2 constants (four values per
  // byte, little-endian bit fields) to TensorRT INT4 nibbles (two values per
  // byte, low nibble first). Values are in [-2, 1], so they fit INT4 exactly
  // and the quantization scales carry over unchanged.
  Expected<nvinfer1::Weights> MakeInt2WeightsAsInt4(const Tensor& tensor,
                                                    size_t num_elements) {
    const auto bytes = tensor.Weights().Bytes();
    if (bytes.size() != (num_elements + 3) / 4) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected TensorRT INT2 weight buffer size");
    }
    owned_weights_.emplace_back((num_elements + 1) / 2);
    auto& packed = owned_weights_.back();
    for (size_t i = 0; i < num_elements; ++i) {
      const uint8_t byte = static_cast<uint8_t>(bytes[i / 4]);
      const int shift = static_cast<int>(i % 4) * 2;
      // Sign-extend the 2-bit field to int8, then keep its low nibble.
      const int8_t value =
          static_cast<int8_t>(static_cast<uint8_t>(byte << (6 - shift))) >> 6;
      const uint8_t nibble = static_cast<uint8_t>(value) & 0x0F;
      if (i % 2 == 0) {
        packed[i / 2] = nibble;
      } else {
        packed[i / 2] |= nibble << 4;
      }
    }
    return nvinfer1::Weights{nvinfer1::DataType::kINT4, packed.data(),
                             static_cast<int64_t>(num_elements)};
  }

  Expected<nvinfer1::Weights> MakeConv2dWeightsOihw(const Tensor& tensor) {
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    if (type.ElementType() != litert::ElementType::Float32 &&
        type.ElementType() != litert::ElementType::Float16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Conv2D weights must be FP32 or FP16");
    }
    const auto dims = type.Layout().Dimensions();
    const int out_channels = dims[0];
    const int kernel_h = dims[1];
    const int kernel_w = dims[2];
    const int in_channels = dims[3];
    LITERT_ASSIGN_OR_RETURN(size_t element_size,
                            ElementByteSize(type.ElementType()));
    auto src = tensor.Weights().Bytes();
    if (src.size() != static_cast<size_t>(out_channels) * kernel_h * kernel_w *
                          in_channels * element_size) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected Conv2D filter size");
    }
    owned_weights_.emplace_back(src.size());
    auto& dst = owned_weights_.back();
    for (int o = 0; o < out_channels; ++o) {
      for (int i = 0; i < in_channels; ++i) {
        for (int h = 0; h < kernel_h; ++h) {
          for (int w = 0; w < kernel_w; ++w) {
            const size_t src_index =
                (((o * kernel_h + h) * kernel_w + w) * in_channels + i) *
                element_size;
            const size_t dst_index =
                (((o * in_channels + i) * kernel_h + h) * kernel_w + w) *
                element_size;
            std::memcpy(dst.data() + dst_index, src.data() + src_index,
                        element_size);
          }
        }
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto data_type,
                            ConvertDataType(type.ElementType()));
    return nvinfer1::Weights{
        data_type, dst.data(),
        static_cast<int64_t>(out_channels) * in_channels * kernel_h * kernel_w};
  }

  static uint16_t Fp32ToFp16Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (exponent == 128) {  // Inf/NaN.
      return sign | 0x7C00 | (mantissa != 0 ? 0x200 : 0);
    }
    if (exponent > 15) {  // Overflow -> Inf.
      return sign | 0x7C00;
    }
    if (exponent >= -14) {  // Normal half.
      // Round mantissa from 23 to 10 bits, to nearest even.
      uint32_t half = ((exponent + 15) << 10) | (mantissa >> 13);
      const uint32_t round_bits = mantissa & 0x1FFF;
      if (round_bits > 0x1000 || (round_bits == 0x1000 && (half & 1))) {
        ++half;  // Carry may bump the exponent; the encoding stays valid.
      }
      return sign | half;
    }
    if (exponent >= -24) {  // Subnormal half.
      mantissa |= 0x800000;
      const int shift = -14 - exponent;
      uint32_t half = mantissa >> (13 + shift);
      const uint32_t round_mask = (1u << (13 + shift)) - 1;
      const uint32_t round_bits = mantissa & round_mask;
      const uint32_t halfway = 1u << (12 + shift);
      if (round_bits > halfway || (round_bits == halfway && (half & 1))) {
        ++half;
      }
      return sign | half;
    }
    return sign;  // Underflow -> signed zero.
  }

  static uint16_t Fp32ToBf16Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + rounding) >> 16);
  }

  Expected<nvinfer1::ITensor*> AddFloatConstant(
      absl::Span<const float> values, nvinfer1::Dims dims,
      const std::string& name,
      nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT) {
    if (data_type == nvinfer1::DataType::kHALF) {
      owned_weights_.emplace_back(values.size() * sizeof(uint16_t));
      auto& storage = owned_weights_.back();
      auto* half_data = reinterpret_cast<uint16_t*>(storage.data());
      for (size_t i = 0; i < values.size(); ++i) {
        half_data[i] = Fp32ToFp16Bits(values[i]);
      }
    } else if (data_type == nvinfer1::DataType::kBF16) {
      // bf16 is the top 16 bits of the fp32 encoding (round-to-nearest-even).
      owned_weights_.emplace_back(values.size() * sizeof(uint16_t));
      auto& storage = owned_weights_.back();
      auto* bf16_data = reinterpret_cast<uint16_t*>(storage.data());
      for (size_t i = 0; i < values.size(); ++i) {
        bf16_data[i] = Fp32ToBf16Bits(values[i]);
      }
    } else if (data_type == nvinfer1::DataType::kFLOAT) {
      owned_weights_.emplace_back(values.size() * sizeof(float));
      auto& storage = owned_weights_.back();
      std::memcpy(storage.data(), values.data(), storage.size());
    } else {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported TensorRT float constant type");
    }
    nvinfer1::Weights weights{data_type, owned_weights_.back().data(),
                              static_cast<int64_t>(values.size())};
    auto* layer = network_->addConstant(dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT float constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(weights, name));
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    return layer->getOutput(0);
  }

  Expected<int8_t> ReadQuantizedConstantValue(const Tensor& tensor,
                                              size_t index) {
    const auto bytes = tensor.Weights().Bytes();
    switch (tensor.ElementType()) {
      case litert::ElementType::Int8:
        if (index >= bytes.size()) {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "Unexpected INT8 weight index");
        }
        return static_cast<int8_t>(bytes[index]);
      case litert::ElementType::Int4: {
        if (index / 2 >= bytes.size()) {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "Unexpected INT4 weight index");
        }
        const uint8_t byte = static_cast<uint8_t>(bytes[index / 2]);
        const uint8_t nibble =
            (index % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        return static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
      }
      case litert::ElementType::Int2: {
        if (index / 4 >= bytes.size()) {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "Unexpected INT2 weight index");
        }
        const uint8_t byte = static_cast<uint8_t>(bytes[index / 4]);
        const int shift = static_cast<int>(index % 4) * 2;
        return static_cast<int8_t>(static_cast<uint8_t>(byte << (6 - shift))) >>
               6;
      }
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported predequantized FC weight type");
    }
  }

  Expected<nvinfer1::ITensor*> AddPredequantizedFcWeights(
      const Tensor& tensor, nvinfer1::DataType data_type) {
    if (data_type != nvinfer1::DataType::kHALF &&
        data_type != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Predequantized FC weights require FP16 or BF16");
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    if (tensor.QTypeId() != kLiteRtQuantizationPerChannel) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Predequantized FC weights require per-channel scales");
    }
    auto q = tensor.PerChannelQuantization();
    if (q.num_channels == 0 || q.scales == nullptr ||
        q.zero_points == nullptr) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Invalid predequantized FC weight quantization");
    }
    const auto dims = type.Layout().Dimensions();
    const int32_t axis = q.quantized_dimension;
    if (axis < 0 || axis >= dims.size() ||
        dims[axis] != static_cast<int32_t>(q.num_channels)) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Predequantized FC weight scale axis does not match shape");
    }
    for (uint64_t i = 0; i < q.num_channels; ++i) {
      if (q.scales[i] <= 0.0f || q.zero_points[i] != 0) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Predequantized FC weights require symmetric scales");
      }
    }
    LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
    size_t inner = 1;
    for (int i = axis + 1; i < dims.size(); ++i) {
      inner *= dims[i];
    }
    owned_weights_.emplace_back(num_elements * sizeof(uint16_t));
    auto& storage = owned_weights_.back();
    auto* dst = reinterpret_cast<uint16_t*>(storage.data());
    for (size_t i = 0; i < num_elements; ++i) {
      const size_t channel = (i / inner) % q.num_channels;
      LITERT_ASSIGN_OR_RETURN(const int8_t quantized,
                              ReadQuantizedConstantValue(tensor, i));
      const float value = static_cast<float>(quantized) * q.scales[channel];
      dst[i] = data_type == nvinfer1::DataType::kHALF ? Fp32ToFp16Bits(value)
                                                      : Fp32ToBf16Bits(value);
    }
    LITERT_ASSIGN_OR_RETURN(auto trt_dims, ConvertDims(type));
    nvinfer1::Weights weights{data_type, storage.data(),
                              static_cast<int64_t>(num_elements)};
    auto* layer = network_->addConstant(trt_dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add predequantized FC weight constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        weights,
        "predequantized_fc_weights_" + std::to_string(tensor.TensorIndex())));
    layer->getOutput(0)->setName(KeepName(UniqueName(
        "predequantized_fc_weights_" + std::to_string(tensor.TensorIndex()))));
    return layer->getOutput(0);
  }

  // Encodes int4/int2 weight values as fp8-e4m3 constants (exact for
  // integers in [-8, 7]) and dequantizes with the per-channel scales.
  // Myelin fuses this fp8-weight dequantize into the GEMV, so decode reads
  // 1 byte/element with no per-step weight materialization.
  Expected<nvinfer1::ITensor*> AddFp8PredequantizedFcWeights(
      const Tensor& tensor, nvinfer1::DataType compute_type) {
    if (compute_type != nvinfer1::DataType::kHALF &&
        compute_type != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "FP8 predequantized FC weights require FP16 or BF16");
    }
    const auto element_type = tensor.ElementType();
    if (element_type != litert::ElementType::Int4 &&
        element_type != litert::ElementType::Int2) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "FP8 predequantized FC weights require int4/int2 values");
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    if (tensor.QTypeId() != kLiteRtQuantizationPerChannel) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "FP8 predequantized FC weights require per-channel scales");
    }
    auto q = tensor.PerChannelQuantization();
    if (q.num_channels == 0 || q.scales == nullptr ||
        q.zero_points == nullptr) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Invalid FP8 predequantized FC weight quantization");
    }
    LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
    // e4m3 encodings of the integers -8..7 (index = value + 8).
    static constexpr uint8_t kInt4ToE4m3[16] = {
        0xD0, 0xCE, 0xCC, 0xCA, 0xC8, 0xC4, 0xC0, 0xB8,
        0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E};
    const auto bytes = tensor.Weights().Bytes();
    owned_weights_.emplace_back(num_elements);
    auto& storage = owned_weights_.back();
    auto* dst = reinterpret_cast<uint8_t*>(storage.data());
    if (element_type == litert::ElementType::Int4) {
      if (bytes.size() * 2 < num_elements) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Unexpected INT4 weight byte count");
      }
      for (size_t i = 0; i < num_elements; ++i) {
        const uint8_t byte = static_cast<uint8_t>(bytes[i / 2]);
        const uint8_t nibble =
            (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        const int8_t value =
            static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
        dst[i] = kInt4ToE4m3[value + 8];
      }
    } else {
      if (bytes.size() * 4 < num_elements) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Unexpected INT2 weight byte count");
      }
      for (size_t i = 0; i < num_elements; ++i) {
        const uint8_t byte = static_cast<uint8_t>(bytes[i / 4]);
        const int shift = static_cast<int>(i % 4) * 2;
        const int8_t value =
            static_cast<int8_t>(static_cast<uint8_t>(byte << (6 - shift))) >> 6;
        dst[i] = kInt4ToE4m3[value + 8];
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto trt_dims, ConvertDims(type));
    nvinfer1::Weights weights{nvinfer1::DataType::kFP8, storage.data(),
                              static_cast<int64_t>(num_elements)};
    auto* constant = network_->addConstant(trt_dims, weights);
    if (constant == nullptr || constant->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add FP8 FC weight constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        weights, "fp8_fc_weights_" + std::to_string(tensor.TensorIndex())));
    constant->getOutput(0)->setName(KeepName(
        UniqueName("fp8_fc_weights_" + std::to_string(tensor.TensorIndex()))));
    int32_t axis = -1;
    LITERT_ASSIGN_OR_RETURN(
        auto* scale, AddQuantizationScaleTensor(
                         tensor, axis, QuantizationScaleType(compute_type)));
    auto* dq =
        network_->addDequantize(*constant->getOutput(0), *scale, compute_type);
    if (dq == nullptr || dq->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add FP8 FC weight dequantize layer");
    }
    if (axis >= 0) {
      dq->setAxis(axis);
    }
    return dq->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddCudaSubbyteGemv(
      const Tensor& tensor, nvinfer1::ITensor* activation,
      nvinfer1::DataType compute_type) {
    if (activation == nullptr || compute_type != nvinfer1::DataType::kBF16 ||
        activation->getType() != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires BF16 activations");
    }
    const auto element_type = tensor.ElementType();
    if (element_type != litert::ElementType::Int2 &&
        element_type != litert::ElementType::Int4) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires INT2 or INT4 weights");
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    const auto dims = type.Layout().Dimensions();
    if (dims.size() != 2 || dims[0] <= 0 || dims[1] <= 0 || dims[1] % 16 != 0) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires aligned rank-2 weights");
    }
    const auto activation_dims = activation->getDimensions();
    if (activation_dims.nbDims < 1 ||
        activation_dims.d[activation_dims.nbDims - 1] != dims[1]) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV activation shape does not match weights");
    }
    for (int i = 0; i + 1 < activation_dims.nbDims; ++i) {
      if (activation_dims.d[i] != 1) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "CUDA subbyte GEMV requires static M=1");
      }
    }
    if (tensor.QTypeId() != kLiteRtQuantizationPerChannel) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires per-channel scales");
    }
    const auto q = tensor.PerChannelQuantization();
    if (q.quantized_dimension != 0 || q.num_channels != dims[0] ||
        q.scales == nullptr || q.zero_points == nullptr) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Invalid CUDA subbyte GEMV weight quantization");
    }
    for (uint64_t i = 0; i < q.num_channels; ++i) {
      if (q.scales[i] <= 0.0f || q.zero_points[i] != 0) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "CUDA subbyte GEMV requires symmetric scales");
      }
    }
    LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
    const int32_t bit_width = element_type == litert::ElementType::Int2 ? 2 : 4;
    const size_t expected_bytes =
        (num_elements + (8 / bit_width) - 1) / (8 / bit_width);
    const auto bytes = tensor.Weights().Bytes();
    if (bytes.size() != expected_bytes) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected CUDA subbyte GEMV weight byte count");
    }

    owned_weights_.emplace_back(bytes.begin(), bytes.end());
    nvinfer1::Dims packed_dims{};
    packed_dims.nbDims = 1;
    packed_dims.d[0] = static_cast<int32_t>(expected_bytes);
    nvinfer1::Weights packed_weights{nvinfer1::DataType::kINT8,
                                     owned_weights_.back().data(),
                                     static_cast<int64_t>(expected_bytes)};
    auto* packed_constant = network_->addConstant(packed_dims, packed_weights);
    if (packed_constant == nullptr ||
        packed_constant->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add packed CUDA subbyte GEMV weights");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        packed_weights,
        "cuda_subbyte_gemv_weights_" + std::to_string(tensor.TensorIndex())));

    nvinfer1::Dims scale_dims{};
    scale_dims.nbDims = 1;
    scale_dims.d[0] = static_cast<int32_t>(q.num_channels);
    LITERT_ASSIGN_OR_RETURN(
        auto* scales,
        AddFloatConstant(
            absl::Span<const float>(q.scales, q.num_channels), scale_dims,
            "cuda_subbyte_gemv_scales_" + std::to_string(tensor.TensorIndex()),
            nvinfer1::DataType::kBF16));

    TrtPtr<nvinfer1::IPluginV3> plugin(
        CreateSubbyteGemvPlugin(bit_width, dims[0], dims[1]));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create CUDA subbyte GEMV plugin");
    }
    nvinfer1::ITensor* inputs[] = {activation, packed_constant->getOutput(0),
                                   scales};
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs, std::size(inputs), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add CUDA subbyte GEMV plugin layer");
    }
    layer->getOutput(0)->setName(KeepName(UniqueName(
        "cuda_subbyte_gemv_" + std::to_string(tensor.TensorIndex()))));
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX CUDA subbyte GEMV: tensor=%u bits=%d "
               "N=%lld K=%lld",
               tensor.TensorIndex(), bit_width, static_cast<long long>(dims[0]),
               static_cast<long long>(dims[1]));
    owned_plugins_.push_back(std::move(plugin));
    uses_cuda_subbyte_gemv_ = true;
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddInt32Constant(
      absl::Span<const int32_t> values, nvinfer1::Dims dims,
      const std::string& name) {
    owned_weights_.emplace_back(values.size() * sizeof(int32_t));
    auto& storage = owned_weights_.back();
    std::memcpy(storage.data(), values.data(), storage.size());
    nvinfer1::Weights weights{nvinfer1::DataType::kINT32, storage.data(),
                              static_cast<int64_t>(values.size())};
    auto* layer = network_->addConstant(dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT int32 constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(weights, name));
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddQuantizationScaleTensor(
      const Tensor& quantized_tensor, int32_t& axis,
      nvinfer1::DataType scale_type, float scale_multiplier = 1.0f) {
    axis = -1;
    if (quantized_tensor.QTypeId() == kLiteRtQuantizationPerTensor) {
      auto q = quantized_tensor.PerTensorQuantization();
      if (q.scale <= 0.0f || q.zero_point != 0) {
        return Error(
            kLiteRtStatusErrorUnsupported,
            "TensorRT Q/DQ requires symmetric per-tensor quantization");
      }
      nvinfer1::Dims scalar_dims{};
      scalar_dims.nbDims = 0;
      const float scale = q.scale * scale_multiplier;
      return AddFloatConstant(
          absl::MakeConstSpan(&scale, 1), scalar_dims,
          "quant_scale_tensor_" +
              std::to_string(quantized_tensor.TensorIndex()),
          scale_type);
    }

    if (quantized_tensor.QTypeId() == kLiteRtQuantizationPerChannel) {
      auto q = quantized_tensor.PerChannelQuantization();
      if (q.num_channels == 0 || q.scales == nullptr ||
          q.zero_points == nullptr) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Invalid per-channel quantization metadata");
      }
      for (uint64_t i = 0; i < q.num_channels; ++i) {
        if (q.scales[i] <= 0.0f || q.zero_points[i] != 0) {
          return Error(
              kLiteRtStatusErrorUnsupported,
              "TensorRT Q/DQ requires symmetric per-channel quantization");
        }
      }
      if (q.num_channels >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Per-channel quantization scale count is too large");
      }
      axis = q.quantized_dimension;
      nvinfer1::Dims scale_dims{};
      scale_dims.nbDims = 1;
      scale_dims.d[0] = static_cast<int32_t>(q.num_channels);
      std::vector<float> scaled;
      absl::Span<const float> scales(q.scales, q.num_channels);
      if (scale_multiplier != 1.0f) {
        scaled.assign(scales.begin(), scales.end());
        for (float& scale : scaled) {
          scale *= scale_multiplier;
        }
        scales = absl::MakeConstSpan(scaled);
      }
      return AddFloatConstant(
          scales, scale_dims,
          "quant_scale_tensor_" +
              std::to_string(quantized_tensor.TensorIndex()),
          scale_type);
    }

    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported TensorRT quantization metadata");
  }

  // TensorRT requires the Q/DQ scale tensor type to match the float side of
  // the conversion (the dequantize output or the quantize input).
  static nvinfer1::DataType QuantizationScaleType(nvinfer1::DataType type) {
    if (type == nvinfer1::DataType::kHALF ||
        type == nvinfer1::DataType::kBF16) {
      return type;
    }
    return nvinfer1::DataType::kFLOAT;
  }

  Expected<nvinfer1::ITensor*> AddDequantizeTensor(
      const Tensor& quantized_tensor, nvinfer1::ITensor* input,
      nvinfer1::DataType output_type, float scale_multiplier = 1.0f) {
    int32_t axis = -1;
    LITERT_ASSIGN_OR_RETURN(
        auto* scale, AddQuantizationScaleTensor(
                         quantized_tensor, axis,
                         QuantizationScaleType(output_type), scale_multiplier));
    auto* layer = network_->addDequantize(*input, *scale, output_type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT dequantize layer");
    }
    if (axis >= 0) {
      layer->setAxis(axis);
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddQuantizeTensor(
      nvinfer1::ITensor* input, const Tensor& quantized_tensor,
      nvinfer1::DataType output_type) {
    // TensorRT's Q/DQ graph optimizer may treat a same-scale Q->DQ pair as an
    // identity, which drops the fake-quant saturation TFLite applies. Clamp
    // to the representable range explicitly so saturation semantics survive
    // any such rewrite.
    if (output_type == nvinfer1::DataType::kINT8 &&
        quantized_tensor.QTypeId() == kLiteRtQuantizationPerTensor) {
      const float scale = quantized_tensor.PerTensorQuantization().scale;
      auto* clip_layer =
          network_->addActivation(*input, nvinfer1::ActivationType::kCLIP);
      if (clip_layer == nullptr || clip_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT quantize clamp layer");
      }
      clip_layer->setAlpha(-128.0f * scale);
      clip_layer->setBeta(127.0f * scale);
      input = clip_layer->getOutput(0);
    }
    int32_t axis = -1;
    LITERT_ASSIGN_OR_RETURN(
        auto* scale,
        AddQuantizationScaleTensor(quantized_tensor, axis,
                                   QuantizationScaleType(input->getType())));
    auto* layer = network_->addQuantize(*input, *scale, output_type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT quantize layer");
    }
    if (axis >= 0) {
      layer->setAxis(axis);
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddCastTensor(nvinfer1::ITensor* input,
                                             nvinfer1::DataType output_type) {
    if (input->getType() == output_type) {
      // Skip identity casts: an interposed cast layer can also break
      // TensorRT's dequantize->matmul weight-only-quantization fusion.
      return input;
    }
    auto* layer = network_->addCast(*input, output_type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT cast layer");
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddZeroFloatTensor(nvinfer1::Dims dims,
                                                  nvinfer1::DataType data_type,
                                                  const std::string& name) {
    if (data_type != nvinfer1::DataType::kFLOAT &&
        data_type != nvinfer1::DataType::kHALF &&
        data_type != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Zero padding supports only TensorRT float tensors");
    }
    size_t elements = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
      if (dims.d[i] < 0) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Zero padding requires static dimensions");
      }
      elements *= static_cast<size_t>(dims.d[i]);
    }
    std::vector<float> zeros(elements, 0.0f);
    return AddFloatConstant(absl::MakeConstSpan(zeros), dims, name, data_type);
  }

  Expected<nvinfer1::ITensor*> SliceTensorStatic(nvinfer1::ITensor* input,
                                                 int axis, int start_index,
                                                 int size,
                                                 const std::string& name) {
    const auto input_dims = input->getDimensions();
    if (axis < 0 || axis >= input_dims.nbDims || size < 0 || start_index < 0 ||
        start_index + size > input_dims.d[axis]) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Invalid TensorRT static slice range");
    }
    nvinfer1::Dims start{};
    nvinfer1::Dims output_dims = input_dims;
    nvinfer1::Dims stride{};
    start.nbDims = input_dims.nbDims;
    stride.nbDims = input_dims.nbDims;
    for (int i = 0; i < input_dims.nbDims; ++i) {
      start.d[i] = 0;
      stride.d[i] = 1;
    }
    start.d[axis] = start_index;
    output_dims.d[axis] = size;
    auto* layer = network_->addSlice(*input, start, output_dims, stride);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT static slice layer");
    }
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddRepeatedScalarConstant(
      const Tensor& value_tensor, const Tensor& output_tensor) {
    LITERT_ASSIGN_OR_RETURN(auto value_type, value_tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_type, output_tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(size_t value_elements, NumElements(value_type));
    if (value_elements != 1 ||
        value_type.ElementType() != output_type.ElementType()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Fill expects a scalar value matching the output type");
    }
    LITERT_ASSIGN_OR_RETURN(size_t element_size,
                            ElementByteSize(output_type.ElementType()));
    auto value_bytes = value_tensor.Weights().Bytes();
    if (value_bytes.size() != element_size) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected fill scalar byte size");
    }
    LITERT_ASSIGN_OR_RETURN(size_t output_elements, NumElements(output_type));
    owned_weights_.emplace_back(output_elements * element_size);
    auto& storage = owned_weights_.back();
    for (size_t i = 0; i < output_elements; ++i) {
      std::memcpy(storage.data() + i * element_size, value_bytes.data(),
                  element_size);
    }
    LITERT_ASSIGN_OR_RETURN(auto data_type,
                            ConvertDataType(output_type.ElementType()));
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    nvinfer1::Weights weights{data_type, storage.data(),
                              static_cast<int64_t>(output_elements)};
    auto* layer = network_->addConstant(output_dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT fill constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        weights,
        "fill_constant_tensor_" + std::to_string(output_tensor.TensorIndex())));
    layer->getOutput(0)->setName(
        KeepName(UniqueName("fill_constant_tensor_" +
                            std::to_string(output_tensor.TensorIndex()))));
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> GetTensor(const Tensor& tensor) {
    if (auto it = tensor_map_.find(tensor.Get()); it != tensor_map_.end()) {
      return it->second;
    }
    if (!tensor.HasWeights()) {
      return Error(kLiteRtStatusErrorCompilation,
                   "TensorRT tensor is not available");
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
    LITERT_ASSIGN_OR_RETURN(auto weights, MakeWeights(tensor));
    auto* layer = network_->addConstant(dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        weights, "constant_tensor_" + std::to_string(tensor.TensorIndex())));
    auto* out = layer->getOutput(0);
    out->setName(KeepName(
        UniqueName("constant_tensor_" + std::to_string(tensor.TensorIndex()))));
    if (Fp16ActivationsEnabled()) {
      // Keep per-channel (weight) constants raw for the matmul WoQ pattern;
      // fold everything else into FP16 so downstream types match.
      LITERT_ASSIGN_OR_RETURN(bool int8_per_tensor,
                              HasSymmetricPerTensorInt8Quantization(tensor));
      if (int8_per_tensor) {
        LITERT_ASSIGN_OR_RETURN(
            out, AddDequantizeTensor(tensor, out, ModeFloatType()));
      } else if (out->getType() == nvinfer1::DataType::kFLOAT) {
        LITERT_ASSIGN_OR_RETURN(out, AddCastTensor(out, ModeFloatType()));
      }
    }
    tensor_map_[tensor.Get()] = out;
    return out;
  }

  Expected<nvinfer1::ITensor*> AddFusedActivation(nvinfer1::ITensor* input,
                                                  uint32_t activation) {
    using Opt = litert::ActivationFunctionType;
    if (activation == Opt::kActivationFunctionTypeNone) {
      return input;
    }
    nvinfer1::ActivationType type = nvinfer1::ActivationType::kRELU;
    float alpha = 0.0f;
    float beta = 0.0f;
    if (activation == Opt::kActivationFunctionTypeRelu) {
      type = nvinfer1::ActivationType::kRELU;
    } else if (activation == Opt::kActivationFunctionTypeRelu6) {
      type = nvinfer1::ActivationType::kCLIP;
      alpha = 0.0f;
      beta = 6.0f;
    } else if (activation == Opt::kActivationFunctionTypeReluN1To1) {
      type = nvinfer1::ActivationType::kCLIP;
      alpha = -1.0f;
      beta = 1.0f;
    } else {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported TensorRT fused activation");
    }
    auto* layer = network_->addActivation(*input, type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT activation");
    }
    if (type == nvinfer1::ActivationType::kCLIP) {
      layer->setAlpha(alpha);
      layer->setBeta(beta);
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> ExpandRankForElementwise(
      nvinfer1::ITensor* tensor, int target_rank) {
    const int rank = tensor->getDimensions().nbDims;
    if (rank == target_rank) {
      return tensor;
    }
    if (rank > target_rank || target_rank > nvinfer1::Dims::MAX_DIMS) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "TensorRT elementwise rank expansion is not supported");
    }

    nvinfer1::Dims reshape{};
    reshape.nbDims = target_rank;
    const int rank_delta = target_rank - rank;
    for (int i = 0; i < rank_delta; ++i) {
      reshape.d[i] = 1;
    }
    const auto dims = tensor->getDimensions();
    for (int i = 0; i < rank; ++i) {
      reshape.d[rank_delta + i] = dims.d[i];
    }

    auto* layer = network_->addShuffle(*tensor);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT elementwise rank expansion");
    }
    layer->setReshapeDimensions(reshape);
    return layer->getOutput(0);
  }

  Expected<void> MatchElementwiseRanks(nvinfer1::ITensor*& lhs,
                                       nvinfer1::ITensor*& rhs) {
    const int lhs_rank = lhs->getDimensions().nbDims;
    const int rhs_rank = rhs->getDimensions().nbDims;
    const int target_rank = std::max(lhs_rank, rhs_rank);
    LITERT_ASSIGN_OR_RETURN(lhs, ExpandRankForElementwise(lhs, target_rank));
    LITERT_ASSIGN_OR_RETURN(rhs, ExpandRankForElementwise(rhs, target_rank));
    return {};
  }

  Expected<nvinfer1::ITensor*> ExpandMatrixWeightsRank(
      nvinfer1::ITensor* weights, int target_rank) {
    const auto dims = weights->getDimensions();
    if (dims.nbDims == target_rank) {
      return weights;
    }
    if (dims.nbDims != 2 || target_rank < 2 ||
        target_rank > nvinfer1::Dims::MAX_DIMS) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "TensorRT matrix weight rank expansion is not supported");
    }

    nvinfer1::Dims reshape{};
    reshape.nbDims = target_rank;
    for (int i = 0; i < target_rank - 2; ++i) {
      reshape.d[i] = 1;
    }
    reshape.d[target_rank - 2] = dims.d[0];
    reshape.d[target_rank - 1] = dims.d[1];

    auto* layer = network_->addShuffle(*weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT matrix weight rank expansion");
    }
    layer->setReshapeDimensions(reshape);
    return layer->getOutput(0);
  }

  Expected<void> SetOutputTensor(const Tensor& tensor, nvinfer1::ITensor* out) {
    // FP16-activation invariant: no FP32 value enters the tensor map, so
    // every float consumer sees a uniform type and Myelin fuses freely.
    if (Fp16ActivationsEnabled() &&
        out->getType() == nvinfer1::DataType::kFLOAT) {
      LITERT_ASSIGN_OR_RETURN(out, AddCastTensor(out, ModeFloatType()));
    }
    if (out == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "TensorRT layer produced null output");
    }
    out->setName(
        KeepName(UniqueName("tensor_" + std::to_string(tensor.TensorIndex()))));
    tensor_map_[tensor.Get()] = out;
    return {};
  }

  Expected<void> LowerElementwise(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* lhs, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* rhs, GetTensor(op.Inputs()[1]));
    nvinfer1::ElementWiseOperation operation;
    switch (op.Code()) {
      case kLiteRtOpCodeTflAdd:
        operation = nvinfer1::ElementWiseOperation::kSUM;
        break;
      case kLiteRtOpCodeTflMul:
        operation = nvinfer1::ElementWiseOperation::kPROD;
        break;
      case kLiteRtOpCodeTflSub:
        operation = nvinfer1::ElementWiseOperation::kSUB;
        break;
      case kLiteRtOpCodeTflDiv:
        operation = nvinfer1::ElementWiseOperation::kDIV;
        break;
      case kLiteRtOpCodeTflMaximum:
        operation = nvinfer1::ElementWiseOperation::kMAX;
        break;
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT elementwise op");
    }
    const bool quantized_int8 =
        !Fp16ActivationsEnabled() && op.Code() != kLiteRtOpCodeTflMaximum &&
        op.Inputs()[0].ElementType() == litert::ElementType::Int8;
    if (quantized_int8) {
      LITERT_ASSIGN_OR_RETURN(
          lhs,
          AddDequantizeTensor(op.Inputs()[0], lhs, nvinfer1::DataType::kFLOAT));
      LITERT_ASSIGN_OR_RETURN(
          rhs,
          AddDequantizeTensor(op.Inputs()[1], rhs, nvinfer1::DataType::kFLOAT));
    }
    LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(lhs, rhs));
    auto* layer = network_->addElementWise(*lhs, *rhs, operation);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT elementwise layer");
    }
    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    LITERT_ASSIGN_OR_RETURN(
        auto* out, AddFusedActivation(layer->getOutput(0), activation));
    if (quantized_int8) {
      LITERT_ASSIGN_OR_RETURN(
          out,
          AddQuantizeTensor(out, op.Outputs()[0], nvinfer1::DataType::kINT8));
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerComparison(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* lhs, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* rhs, GetTensor(op.Inputs()[1]));
    nvinfer1::ElementWiseOperation operation;
    bool invert_result = false;
    switch (op.Code()) {
      case kLiteRtOpCodeTflLess:
        operation = nvinfer1::ElementWiseOperation::kLESS;
        break;
      case kLiteRtOpCodeTflGreaterEqual:
        operation = nvinfer1::ElementWiseOperation::kLESS;
        invert_result = true;
        break;
      case kLiteRtOpCodeTflNotEqual:
        operation = nvinfer1::ElementWiseOperation::kEQUAL;
        invert_result = true;
        break;
      case kLiteRtOpCodeTflLogicalAnd:
        operation = nvinfer1::ElementWiseOperation::kAND;
        break;
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT comparison op");
    }
    LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(lhs, rhs));
    auto* layer = network_->addElementWise(*lhs, *rhs, operation);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT comparison layer");
    }
    nvinfer1::ITensor* out = layer->getOutput(0);
    if (invert_result) {
      auto* not_layer =
          network_->addUnary(*out, nvinfer1::UnaryOperation::kNOT);
      if (not_layer == nullptr || not_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT logical-not layer");
      }
      out = not_layer->getOutput(0);
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerActivation(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    uint32_t activation = litert::kActivationFunctionTypeNone;
    switch (op.Code()) {
      case kLiteRtOpCodeTflRelu:
        activation = litert::kActivationFunctionTypeRelu;
        break;
      case kLiteRtOpCodeTflRelu6:
        activation = litert::kActivationFunctionTypeRelu6;
        break;
      case kLiteRtOpCodeTflReluN1To1:
        activation = litert::kActivationFunctionTypeReluN1To1;
        break;
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT activation op");
    }
    LITERT_ASSIGN_OR_RETURN(auto* out, AddFusedActivation(input, activation));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerUnary(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    if (op.Code() == kLiteRtOpCodeTflTanh) {
      auto* layer =
          network_->addActivation(*input, nvinfer1::ActivationType::kTANH);
      if (layer == nullptr || layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT tanh layer");
      }
      return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
    }

    nvinfer1::UnaryOperation operation;
    switch (op.Code()) {
      case kLiteRtOpCodeTflSin:
        operation = nvinfer1::UnaryOperation::kSIN;
        break;
      case kLiteRtOpCodeTflCos:
        operation = nvinfer1::UnaryOperation::kCOS;
        break;
      case kLiteRtOpCodeTflRsqrt:
        operation = nvinfer1::UnaryOperation::kSQRT;
        break;
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT unary op");
    }
    auto* layer = network_->addUnary(*input, operation);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT unary layer");
    }
    nvinfer1::ITensor* out = layer->getOutput(0);
    if (op.Code() == kLiteRtOpCodeTflRsqrt) {
      auto* recip = network_->addUnary(*out, nvinfer1::UnaryOperation::kRECIP);
      if (recip == nullptr || recip->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT reciprocal layer");
      }
      out = recip->getOutput(0);
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerGelu(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    bool approximate = false;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetGeluApproximateOption(op.Get(), &approximate));
    auto* layer = network_->addActivation(
        *input, approximate ? nvinfer1::ActivationType::kGELU_TANH
                            : nvinfer1::ActivationType::kGELU_ERF);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT GELU layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerCast(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto trt_type,
                            ConvertDataType(output_type.ElementType()));
    auto* layer = network_->addCast(*input, trt_type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT cast layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerQuantize(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    if (Fp16ActivationsEnabled()) {
      // Values already flow as FP16 reals; a (re)quantize only clamps to the
      // output tensor's representable range.
      nvinfer1::ITensor* out = input;
      if (op.Outputs()[0].QTypeId() == kLiteRtQuantizationPerTensor) {
        const float scale = op.Outputs()[0].PerTensorQuantization().scale;
        auto* clip_layer =
            network_->addActivation(*input, nvinfer1::ActivationType::kCLIP);
        if (clip_layer == nullptr || clip_layer->getOutput(0) == nullptr) {
          return Error(kLiteRtStatusErrorCompilation,
                       "Failed to add TensorRT quantize clamp layer");
        }
        clip_layer->setAlpha(-128.0f * scale);
        clip_layer->setBeta(127.0f * scale);
        out = clip_layer->getOutput(0);
      }
      return SetOutputTensor(op.Outputs()[0], out);
    }
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto trt_type,
                            ConvertDataType(output_type.ElementType()));
    LITERT_ASSIGN_OR_RETURN(
        auto* out, AddQuantizeTensor(input, op.Outputs()[0], trt_type));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerDequantize(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    if (Fp16ActivationsEnabled() &&
        input->getType() != nvinfer1::DataType::kINT8 &&
        input->getType() != nvinfer1::DataType::kINT4) {
      // The value is already the dequantized FP16 real.
      return SetOutputTensor(op.Outputs()[0], input);
    }
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto trt_type,
                            ConvertDataType(output_type.ElementType()));
    if (Fp16ActivationsEnabled()) {
      trt_type = ModeFloatType();
    }
    LITERT_ASSIGN_OR_RETURN(
        auto* out, AddDequantizeTensor(op.Inputs()[0], input, trt_type));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerFill(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(
        auto* constant,
        AddRepeatedScalarConstant(op.Inputs()[1], op.Outputs()[0]));
    return SetOutputTensor(op.Outputs()[0], constant);
  }

  Expected<void> LowerSoftmax(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    auto* layer = network_->addSoftMax(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT softmax layer");
    }
    const int rank = input->getDimensions().nbDims;
    if (rank <= 0 || rank >= 32) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported softmax tensor rank");
    }
    layer->setAxes(1U << (rank - 1));
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerReshape(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT reshape layer");
    }
    layer->setReshapeDimensions(output_dims);
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerPack(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    int32_t axis = 0;
    LITERT_RETURN_IF_ERROR(LiteRtGetPackAxisOption(op.Get(), &axis));
    const int output_rank = output_dims.nbDims;
    if (axis < 0) {
      axis += output_rank;
    }

    nvinfer1::Dims expanded_input_dims = output_dims;
    expanded_input_dims.d[axis] = 1;
    std::vector<nvinfer1::ITensor*> expanded_inputs;
    expanded_inputs.reserve(op.Inputs().size());
    for (const auto& input : op.Inputs()) {
      LITERT_ASSIGN_OR_RETURN(auto* input_tensor, GetTensor(input));
      LITERT_ASSIGN_OR_RETURN(auto* expanded,
                              ReshapeTensor(input_tensor, expanded_input_dims));
      expanded_inputs.push_back(expanded);
    }

    auto* layer = network_->addConcatenation(
        expanded_inputs.data(), static_cast<int32_t>(expanded_inputs.size()));
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT pack concatenation layer");
    }
    layer->setAxis(axis);
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerSlice(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    const int rank = input_type.Layout().Rank();
    if (output_dims.nbDims != rank) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Slice rank metadata does not match input rank");
    }
    nvinfer1::Dims start{};
    nvinfer1::Dims stride{};
    start.nbDims = rank;
    stride.nbDims = rank;
    for (int i = 0; i < rank; ++i) {
      start.d[i] = 0;
      stride.d[i] = 1;
    }
    auto begin = ReadInt32Constant(op.Inputs()[1]);
    if (!begin) {
      // Runtime slice offsets lower to a gather of `begin[axis] + iota` along
      // the single shrinking axis. A slice-layer start tensor would become a
      // TensorRT shape input, which requires optimization profiles.
      const auto input_dims = input->getDimensions();
      int slice_axis = 0;
      for (int i = 0; i < rank; ++i) {
        if (output_dims.d[i] != input_dims.d[i]) {
          slice_axis = i;
          break;
        }
      }
      LITERT_ASSIGN_OR_RETURN(auto* begin_tensor, GetTensor(op.Inputs()[1]));
      LITERT_ASSIGN_OR_RETURN(
          auto* begin_scalar,
          SliceTensor1d(begin_tensor, /*start_index=*/slice_axis, /*size=*/1));
      const int32_t gather_size = output_dims.d[slice_axis];
      std::vector<int32_t> iota(gather_size);
      for (int32_t i = 0; i < gather_size; ++i) {
        iota[i] = i;
      }
      nvinfer1::Dims iota_dims{};
      iota_dims.nbDims = 1;
      iota_dims.d[0] = gather_size;
      LITERT_ASSIGN_OR_RETURN(auto* iota_tensor,
                              AddInt32Constant(iota, iota_dims, "slice_iota"));
      auto* index_layer = network_->addElementWise(
          *iota_tensor, *begin_scalar, nvinfer1::ElementWiseOperation::kSUM);
      if (index_layer == nullptr || index_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT slice index layer");
      }
      auto* gather_layer =
          network_->addGather(*input, *index_layer->getOutput(0), slice_axis);
      if (gather_layer == nullptr || gather_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT slice gather layer");
      }
      return SetOutputTensor(op.Outputs()[0], gather_layer->getOutput(0));
    }
    if (begin->size() != static_cast<size_t>(rank)) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Slice begin size does not match input rank");
    }
    for (int i = 0; i < rank; ++i) {
      start.d[i] = (*begin)[i];
    }
    auto* layer = network_->addSlice(*input, start, output_dims, stride);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT slice layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<nvinfer1::ITensor*> SliceTensor1d(nvinfer1::ITensor* input,
                                             int start_index, int size) {
    nvinfer1::Dims start{};
    nvinfer1::Dims output_dims{};
    nvinfer1::Dims stride{};
    start.nbDims = 1;
    output_dims.nbDims = 1;
    stride.nbDims = 1;
    start.d[0] = start_index;
    output_dims.d[0] = size;
    stride.d[0] = 1;
    auto* layer = network_->addSlice(*input, start, output_dims, stride);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT 1D slice layer");
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> ReshapeTensor(nvinfer1::ITensor* input,
                                             nvinfer1::Dims dims) {
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT reshape helper layer");
    }
    layer->setReshapeDimensions(dims);
    return layer->getOutput(0);
  }

  Expected<void> LowerUnpack(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto input_dims, ConvertDims(input_type));
    int32_t axis = 0;
    LITERT_RETURN_IF_ERROR(LiteRtGetUnpackAxisOption(op.Get(), &axis));
    const int input_rank = input_dims.nbDims;
    if (axis < 0) {
      axis += input_rank;
    }

    nvinfer1::Dims stride{};
    stride.nbDims = input_rank;
    for (int i = 0; i < input_rank; ++i) {
      stride.d[i] = 1;
    }

    for (int i = 0; i < op.Outputs().size(); ++i) {
      nvinfer1::Dims start{};
      nvinfer1::Dims size = input_dims;
      start.nbDims = input_rank;
      for (int dim = 0; dim < input_rank; ++dim) {
        start.d[dim] = 0;
      }
      start.d[axis] = i;
      size.d[axis] = 1;
      auto* slice = network_->addSlice(*input, start, size, stride);
      if (slice == nullptr || slice->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT unpack slice layer");
      }
      LITERT_ASSIGN_OR_RETURN(auto output_type,
                              op.Outputs()[i].RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
      LITERT_ASSIGN_OR_RETURN(auto* reshaped,
                              ReshapeTensor(slice->getOutput(0), output_dims));
      LITERT_RETURN_IF_ERROR(SetOutputTensor(op.Outputs()[i], reshaped));
    }
    return {};
  }

  Expected<void> LowerDynamicUpdateSlice(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* operand, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* update, GetTensor(op.Inputs()[1]));
    LITERT_ASSIGN_OR_RETURN(auto* start_indices, GetTensor(op.Inputs()[2]));
    if (Fp16ActivationsEnabled() &&
        op.Inputs()[0].ElementType() == litert::ElementType::Int8) {
      // Cache updates scatter into the int8 boundary tensor directly: use the
      // raw int8 network input as the operand and quantize the FP16 update
      // values, so the cache never round-trips through FP16 storage.
      if (auto raw = raw_input_map_.find(op.Inputs()[0].Get());
          raw != raw_input_map_.end()) {
        operand = raw->second;
      } else if (operand->getType() != nvinfer1::DataType::kINT8) {
        LITERT_ASSIGN_OR_RETURN(operand,
                                AddQuantizeTensor(operand, op.Inputs()[0],
                                                  nvinfer1::DataType::kINT8));
      }
      if (update->getType() != nvinfer1::DataType::kINT8) {
        LITERT_ASSIGN_OR_RETURN(update,
                                AddQuantizeTensor(update, op.Inputs()[1],
                                                  nvinfer1::DataType::kINT8));
      }
    }
    const auto operand_dims = operand->getDimensions();
    const auto update_dims = update->getDimensions();
    const int rank = operand_dims.nbDims;

    // The update writes every axis in full except (at most) one partial axis,
    // so it is a scatter of elements along that axis: for each update element
    // the target coordinate is its own coordinate plus the runtime offset.
    int partial_axis = -1;
    for (int i = 0; i < rank; ++i) {
      if (update_dims.d[i] < operand_dims.d[i]) {
        partial_axis = i;
        break;
      }
    }
    if (partial_axis < 0) {
      // Full overwrite: the result is just the update (offsets must be zero).
      // Copy through an identity layer so the update tensor keeps its name.
      auto* identity = network_->addIdentity(*update);
      if (identity == nullptr || identity->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT DynamicUpdateSlice identity");
      }
      return SetOutputTensor(op.Outputs()[0], identity->getOutput(0));
    }

    // indices[c0, .., cA, .., cn] = cA + start_indices[A], built with a
    // linspace fill so no index constant is materialized.
    LITERT_ASSIGN_OR_RETURN(
        auto* offset_1d,
        SliceTensor1d(start_indices, /*start_index=*/partial_axis,
                      /*size=*/1));
    nvinfer1::Dims scalar_dims{};
    scalar_dims.nbDims = 0;
    LITERT_ASSIGN_OR_RETURN(auto* offset_scalar,
                            ReshapeTensor(offset_1d, scalar_dims));
    std::vector<int32_t> steps(rank, 0);
    steps[partial_axis] = 1;
    nvinfer1::Dims steps_dims{};
    steps_dims.nbDims = 1;
    steps_dims.d[0] = rank;
    LITERT_ASSIGN_OR_RETURN(auto* steps_tensor,
                            AddInt32Constant(steps, steps_dims, "dus_steps"));
    auto* fill_layer =
        network_->addFill(update_dims, nvinfer1::FillOperation::kLINSPACE,
                          nvinfer1::DataType::kINT32);
    if (fill_layer == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT DynamicUpdateSlice index fill");
    }
    fill_layer->setInput(1, *offset_scalar);
    fill_layer->setInput(2, *steps_tensor);
    auto* indices = fill_layer->getOutput(0);
    if (indices == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to build TensorRT DynamicUpdateSlice indices");
    }

    auto* layer = network_->addScatter(*operand, *indices, *update,
                                       nvinfer1::ScatterMode::kELEMENT);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(
          kLiteRtStatusErrorCompilation,
          "Failed to add TensorRT scatter layer for DynamicUpdateSlice");
    }
    layer->setAxis(partial_axis);
    nvinfer1::ITensor* scatter_out = layer->getOutput(0);
    if (Fp16ActivationsEnabled() &&
        scatter_out->getType() == nvinfer1::DataType::kINT8) {
      // Keep the raw int8 result for the island boundary, and publish the
      // FP16 view for in-island consumers (attention reads the updated
      // cache); the dequantize fuses into the consuming matmul.
      raw_output_map_[op.Outputs()[0].Get()] = scatter_out;
      LITERT_ASSIGN_OR_RETURN(
          auto* value,
          AddDequantizeTensor(op.Outputs()[0], scatter_out, ModeFloatType()));
      return SetOutputTensor(op.Outputs()[0], value);
    }
    return SetOutputTensor(op.Outputs()[0], scatter_out);
  }

  Expected<void> LowerTranspose(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto perm, ReadInt32Constant(op.Inputs()[1]));
    if (perm.size() > nvinfer1::Dims::MAX_DIMS) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Transpose permutation rank is too large");
    }
    nvinfer1::Permutation trt_perm{};
    for (int i = 0; i < perm.size(); ++i) {
      trt_perm.order[i] = perm[i];
    }
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT transpose layer");
    }
    layer->setFirstTranspose(trt_perm);
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerConcatenation(const Op& op) {
    std::vector<nvinfer1::ITensor*> inputs;
    inputs.reserve(op.Inputs().size());
    for (const auto& input : op.Inputs()) {
      LITERT_ASSIGN_OR_RETURN(auto* tensor, GetTensor(input));
      inputs.push_back(tensor);
    }
    auto* layer = network_->addConcatenation(
        inputs.data(), static_cast<int32_t>(inputs.size()));
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT concatenation layer");
    }
    int32_t axis = 0;
    LITERT_RETURN_IF_ERROR(LiteRtGetConcatenationAxisOption(op.Get(), &axis));
    const int rank = inputs[0]->getDimensions().nbDims;
    if (axis < 0) {
      axis += rank;
    }
    layer->setAxis(axis);
    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    LITERT_ASSIGN_OR_RETURN(
        auto* out, AddFusedActivation(layer->getOutput(0), activation));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerBatchMatmul(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* lhs, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* rhs, GetTensor(op.Inputs()[1]));
    bool adj_x = false;
    bool adj_y = false;
    LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAdjXOption(op.Get(), &adj_x));
    LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAdjYOption(op.Get(), &adj_y));
    auto* layer = network_->addMatrixMultiply(
        *lhs,
        adj_x ? nvinfer1::MatrixOperation::kTRANSPOSE
              : nvinfer1::MatrixOperation::kNONE,
        *rhs,
        adj_y ? nvinfer1::MatrixOperation::kTRANSPOSE
              : nvinfer1::MatrixOperation::kNONE);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT batch matmul layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> FinishFullyConnected(const Op& op, nvinfer1::ITensor* out,
                                      nvinfer1::DataType compute_type) {
    if (op.Inputs().size() == 3) {
      LITERT_ASSIGN_OR_RETURN(auto* bias, GetTensor(op.Inputs()[2]));
      LITERT_ASSIGN_OR_RETURN(bias, AddCastTensor(bias, compute_type));
      LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(out, bias));
      auto* bias_layer = network_->addElementWise(
          *out, *bias, nvinfer1::ElementWiseOperation::kSUM);
      if (bias_layer == nullptr || bias_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT fully connected bias layer");
      }
      out = bias_layer->getOutput(0);
    }
    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    LITERT_ASSIGN_OR_RETURN(out, AddFusedActivation(out, activation));
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto trt_output_type,
                            ConvertDataType(output_type.ElementType()));
    if (Fp16ActivationsEnabled()) {
      // Keep the FP16 real value; boundary conversion happens at MarkOutputs.
    } else if (!IsFloatLike(op.Outputs()[0].ElementType())) {
      LITERT_ASSIGN_OR_RETURN(
          out, AddQuantizeTensor(out, op.Outputs()[0], trt_output_type));
    } else if (trt_output_type != compute_type) {
      LITERT_ASSIGN_OR_RETURN(out, AddCastTensor(out, trt_output_type));
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerFullyConnected(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    // TensorRT executes sub-byte weight-only quantization (INT4) only with
    // half-precision activations, so those matmuls compute in FP16. Wider
    // weight types keep FP32 compute.
    const bool sub_byte_weights =
        op.Inputs()[1].ElementType() == litert::ElementType::Int4 ||
        op.Inputs()[1].ElementType() == litert::ElementType::Int2;
    nvinfer1::DataType compute_type = sub_byte_weights
                                          ? nvinfer1::DataType::kHALF
                                          : nvinfer1::DataType::kFLOAT;
    float weight_scale_multiplier = 1.0f;
    if (Fp16ActivationsEnabled()) {
      // Activations already flow as FP16 reals; compute in FP16 and
      // dequantize only the weights (weight-only quantization).
      compute_type = ModeFloatType();
    }
    nvinfer1::ITensor* weights = nullptr;
    const PredequantMode predequant_mode = PredequantizeFcWeightsMode();
    // Predequantization pays off only for M=1 GEMVs (decode): Myelin has no
    // fused sub-byte weight-only GEMV, so it re-materializes dequantized
    // weights every step. At M>=128 (prefill) the one-off materialization
    // amortizes and the dense GEMM is faster than a fused fp8-weight GEMM,
    // so larger-M matmuls keep the quantized-constant path.
    bool static_m_is_one = true;
    {
      const auto in_dims = input->getDimensions();
      for (int i = 0; i + 1 < in_dims.nbDims; ++i) {
        if (in_dims.d[i] != 1) {
          static_m_is_one = false;
          break;
        }
      }
    }
    if (Fp16ActivationsEnabled() &&
        predequant_mode == PredequantMode::kCudaGemv && sub_byte_weights &&
        static_m_is_one) {
      LITERT_ASSIGN_OR_RETURN(input, AddCastTensor(input, compute_type));
      auto fused = AddCudaSubbyteGemv(op.Inputs()[1], input, compute_type);
      if (fused.HasValue()) {
        return FinishFullyConnected(op, *fused, compute_type);
      }
      LITERT_LOG(LITERT_INFO, "CUDA subbyte GEMV unavailable for tensor %u: %s",
                 op.Inputs()[1].TensorIndex(), fused.Error().Message().c_str());
    }
    if (Fp16ActivationsEnabled() && predequant_mode != PredequantMode::kOff &&
        predequant_mode != PredequantMode::kCudaGemv && static_m_is_one &&
        (op.Inputs()[1].ElementType() == litert::ElementType::Int8 ||
         op.Inputs()[1].ElementType() == litert::ElementType::Int4 ||
         op.Inputs()[1].ElementType() == litert::ElementType::Int2)) {
      // fp8 mode: sub-byte integer values are exact in e4m3 and the fp8
      // dequantize fuses into the GEMV. int8 values need >4 mantissa bits,
      // so those tensors (small kv projections) take float constants.
      Expected<nvinfer1::ITensor*> predequantized =
          (predequant_mode == PredequantMode::kFp8 && sub_byte_weights)
              ? AddFp8PredequantizedFcWeights(op.Inputs()[1], compute_type)
              : AddPredequantizedFcWeights(op.Inputs()[1], compute_type);
      if (predequantized.HasValue()) {
        weights = *predequantized;
      } else {
        LITERT_LOG(LITERT_INFO,
                   "Predequantized FC weights unavailable for tensor %u: %s",
                   op.Inputs()[1].TensorIndex(),
                   predequantized.Error().Message().c_str());
      }
    }
    if (weights == nullptr) {
      LITERT_ASSIGN_OR_RETURN(weights, GetTensor(op.Inputs()[1]));
    }
    if (Fp16ActivationsEnabled()) {
      if (weights->getType() == nvinfer1::DataType::kINT8 ||
          weights->getType() == nvinfer1::DataType::kINT4) {
        LITERT_ASSIGN_OR_RETURN(
            weights,
            AddDequantizeTensor(op.Inputs()[1], weights, compute_type));
      }
    } else if (IsFloatLike(op.Inputs()[0].ElementType())) {
      LITERT_ASSIGN_OR_RETURN(auto input_type,
                              op.Inputs()[0].RankedTensorType());
      if (!sub_byte_weights) {
        LITERT_ASSIGN_OR_RETURN(compute_type,
                                ConvertDataType(input_type.ElementType()));
      }
    } else if (sub_byte_weights ||
               EnvEnabled("LITERT_NVIDIA_TENSORRT_FOLD_INPUT_SCALE",
                          /*default_value=*/false)) {
      // TensorRT has no mixed int8-activation x int4-weight matmul pattern,
      // and folding also prevents Myelin from fusing the activation Q/DQ pair
      // with the matmul into a quantized GEMM. Feed the raw integer values
      // and fold the activation's per-tensor scale into the weight dequantize
      // scales, which is mathematically identical for symmetric quantization.
      weight_scale_multiplier = op.Inputs()[0].PerTensorQuantization().scale;
    } else {
      LITERT_ASSIGN_OR_RETURN(
          input, AddDequantizeTensor(op.Inputs()[0], input, compute_type));
    }
    if (!Fp16ActivationsEnabled() &&
        !IsFloatLike(op.Inputs()[1].ElementType())) {
      LITERT_ASSIGN_OR_RETURN(
          weights, AddDequantizeTensor(op.Inputs()[1], weights, compute_type,
                                       weight_scale_multiplier));
    }
    LITERT_ASSIGN_OR_RETURN(input, AddCastTensor(input, compute_type));
    LITERT_ASSIGN_OR_RETURN(weights, AddCastTensor(weights, compute_type));
    // Flatten the activation to [M, K] instead of expanding the weights'
    // rank: a reshape between the weight dequantize and the matmul breaks
    // TensorRT's weight-only-quantization fusion, and Myelin then
    // materializes the dequantized weights to memory on every invocation
    // (the dominant decode-step cost). Activation-side reshapes fuse freely.
    const auto input_dims = input->getDimensions();
    const auto weight_dims = weights->getDimensions();
    const bool flatten_input = input_dims.nbDims != 2;
    nvinfer1::ITensor* matmul_input = input;
    if (flatten_input) {
      int64_t m = 1;
      for (int i = 0; i + 1 < input_dims.nbDims; ++i) {
        m *= input_dims.d[i];
      }
      nvinfer1::Dims flat{};
      flat.nbDims = 2;
      flat.d[0] = m;
      flat.d[1] = input_dims.d[input_dims.nbDims - 1];
      LITERT_ASSIGN_OR_RETURN(matmul_input, ReshapeTensor(input, flat));
    }
    if (weight_dims.nbDims != 2) {
      LITERT_ASSIGN_OR_RETURN(
          weights, ExpandMatrixWeightsRank(
                       weights, matmul_input->getDimensions().nbDims));
    }
    auto* layer = network_->addMatrixMultiply(
        *matmul_input, nvinfer1::MatrixOperation::kNONE, *weights,
        nvinfer1::MatrixOperation::kTRANSPOSE);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT fully connected matmul layer");
    }
    nvinfer1::ITensor* out = layer->getOutput(0);
    if (flatten_input) {
      LITERT_ASSIGN_OR_RETURN(auto out_type,
                              op.Outputs()[0].RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto out_dims, ConvertDims(out_type));
      LITERT_ASSIGN_OR_RETURN(out, ReshapeTensor(out, out_dims));
    }
    return FinishFullyConnected(op, out, compute_type);
  }

  Expected<void> LowerSelectV2(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* condition, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* then_tensor, GetTensor(op.Inputs()[1]));
    LITERT_ASSIGN_OR_RETURN(auto* else_tensor, GetTensor(op.Inputs()[2]));
    const int target_rank = std::max({condition->getDimensions().nbDims,
                                      then_tensor->getDimensions().nbDims,
                                      else_tensor->getDimensions().nbDims});
    LITERT_ASSIGN_OR_RETURN(condition,
                            ExpandRankForElementwise(condition, target_rank));
    LITERT_ASSIGN_OR_RETURN(then_tensor,
                            ExpandRankForElementwise(then_tensor, target_rank));
    LITERT_ASSIGN_OR_RETURN(else_tensor,
                            ExpandRankForElementwise(else_tensor, target_rank));
    auto* layer = network_->addSelect(*condition, *then_tensor, *else_tensor);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT select layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerSum(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto axes, ReadInt32Constant(op.Inputs()[1]));
    // FP16 reduction sums overflow half range (e.g. RMSNorm sums of squares
    // across the hidden dim); accumulate in FP32 and convert back.
    const bool widen = Fp16ActivationsEnabled() &&
                       input->getType() != nvinfer1::DataType::kFLOAT;
    if (widen) {
      LITERT_ASSIGN_OR_RETURN(input,
                              AddCastTensor(input, nvinfer1::DataType::kFLOAT));
    }
    const int rank = input->getDimensions().nbDims;
    uint32_t axes_mask = 0;
    for (int32_t axis : axes) {
      if (axis < 0) {
        axis += rank;
      }
      axes_mask |= 1U << axis;
    }
    bool keep_dims = false;
    LITERT_RETURN_IF_ERROR(LiteRtGetSumKeepDimsOption(op.Get(), &keep_dims));
    auto* layer = network_->addReduce(*input, nvinfer1::ReduceOperation::kSUM,
                                      axes_mask, keep_dims);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT sum-reduce layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerReduceMax(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto axes, ReadInt32Constant(op.Inputs()[1]));
    const int rank = input->getDimensions().nbDims;
    uint32_t axes_mask = 0;
    for (int32_t axis : axes) {
      if (axis < 0) {
        axis += rank;
      }
      axes_mask |= 1U << axis;
    }
    bool keep_dims = false;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetReduceMaxKeepDimsOption(op.Get(), &keep_dims));
    auto* layer = network_->addReduce(*input, nvinfer1::ReduceOperation::kMAX,
                                      axes_mask, keep_dims);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT reduce-max layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerFloorMod(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* lhs, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* rhs, GetTensor(op.Inputs()[1]));
    LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(lhs, rhs));
    auto* div = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
    if (div == nullptr || div->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT floor_div layer");
    }
    auto* product = network_->addElementWise(
        *div->getOutput(0), *rhs, nvinfer1::ElementWiseOperation::kPROD);
    if (product == nullptr || product->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT floor_mod product layer");
    }
    auto* result = network_->addElementWise(
        *lhs, *product->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
    if (result == nullptr || result->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT floor_mod layer");
    }
    return SetOutputTensor(op.Outputs()[0], result->getOutput(0));
  }

  Expected<nvinfer1::ITensor*> TransposeNhcwToNchw(nvinfer1::ITensor* input) {
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT NHWC->NCHW transpose");
    }
    layer->setFirstTranspose(MakePermutation({0, 3, 1, 2}));
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> TransposeNchwToNhwc(nvinfer1::ITensor* input) {
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT NCHW->NHWC transpose");
    }
    layer->setFirstTranspose(MakePermutation({0, 2, 3, 1}));
    return layer->getOutput(0);
  }

  Expected<void> LowerConv2d(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input_nhwc, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* input_nchw, TransposeNhcwToNchw(input_nhwc));
    LITERT_ASSIGN_OR_RETURN(auto filter_type,
                            op.Inputs()[1].RankedTensorType());
    const auto filter_dims = filter_type.Layout().Dimensions();
    const int out_channels = filter_dims[0];
    const int kernel_h = filter_dims[1];
    const int kernel_w = filter_dims[2];
    LITERT_ASSIGN_OR_RETURN(auto kernel_weights,
                            MakeConv2dWeightsOihw(op.Inputs()[1]));
    nvinfer1::Weights bias_weights{};
    if (op.Inputs().size() == 3) {
      LITERT_ASSIGN_OR_RETURN(bias_weights, MakeWeights(op.Inputs()[2]));
    }

    nvinfer1::Dims kernel_dims{};
    kernel_dims.nbDims = 2;
    kernel_dims.d[0] = kernel_h;
    kernel_dims.d[1] = kernel_w;
    auto* conv = network_->addConvolutionNd(
        *input_nchw, out_channels, kernel_dims, kernel_weights, bias_weights);
    if (conv == nullptr || conv->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT Conv2D layer");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        kernel_weights,
        "conv_kernel_" + std::to_string(op.Inputs()[1].TensorIndex())));
    if (bias_weights.values != nullptr && bias_weights.count > 0) {
      LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
          bias_weights,
          "conv_bias_" + std::to_string(op.Inputs()[2].TensorIndex())));
    }
    int32_t stride_w = 1;
    int32_t stride_h = 1;
    int32_t dilation_w = 1;
    int32_t dilation_h = 1;
    uint32_t padding = litert::kPaddingSame;
    LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideWOption(op.Get(), &stride_w));
    LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideHOption(op.Get(), &stride_h));
    LITERT_RETURN_IF_ERROR(
        LiteRtGetConv2dDilationWOption(op.Get(), &dilation_w));
    LITERT_RETURN_IF_ERROR(
        LiteRtGetConv2dDilationHOption(op.Get(), &dilation_h));
    LITERT_RETURN_IF_ERROR(LiteRtGetConv2dPaddingOption(op.Get(), &padding));
    nvinfer1::Dims stride_dims{};
    stride_dims.nbDims = 2;
    stride_dims.d[0] = stride_h;
    stride_dims.d[1] = stride_w;
    conv->setStrideNd(stride_dims);
    nvinfer1::Dims dilation_dims{};
    dilation_dims.nbDims = 2;
    dilation_dims.d[0] = dilation_h;
    dilation_dims.d[1] = dilation_w;
    conv->setDilationNd(dilation_dims);
    if (padding == litert::kPaddingSame) {
      conv->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }

    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    LITERT_ASSIGN_OR_RETURN(auto* conv_output,
                            AddFusedActivation(conv->getOutput(0), activation));
    LITERT_ASSIGN_OR_RETURN(auto* output_nhwc,
                            TransposeNchwToNhwc(conv_output));
    return SetOutputTensor(op.Outputs()[0], output_nhwc);
  }

  // Scatters `update` into `operand` at the runtime offset held in
  // `start_indices` (i32[rank]); exactly one axis may be partial. Mirrors
  // LowerDynamicUpdateSlice's linspace/scatter construction.
  Expected<nvinfer1::ITensor*> BuildScatterUpdate(
      nvinfer1::ITensor* operand, nvinfer1::ITensor* update,
      nvinfer1::ITensor* start_indices) {
    const auto operand_dims = operand->getDimensions();
    const auto update_dims = update->getDimensions();
    const int rank = operand_dims.nbDims;
    int partial_axis = -1;
    for (int i = 0; i < rank; ++i) {
      if (update_dims.d[i] < operand_dims.d[i]) {
        partial_axis = i;
        break;
      }
    }
    if (partial_axis < 0) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Composite cache update expects a partial axis");
    }
    LITERT_ASSIGN_OR_RETURN(
        auto* offset_1d,
        SliceTensor1d(start_indices, /*start_index=*/partial_axis,
                      /*size=*/1));
    nvinfer1::Dims scalar_dims{};
    scalar_dims.nbDims = 0;
    LITERT_ASSIGN_OR_RETURN(auto* offset_scalar,
                            ReshapeTensor(offset_1d, scalar_dims));
    std::vector<int32_t> steps(rank, 0);
    steps[partial_axis] = 1;
    nvinfer1::Dims steps_dims{};
    steps_dims.nbDims = 1;
    steps_dims.d[0] = rank;
    LITERT_ASSIGN_OR_RETURN(
        auto* steps_tensor,
        AddInt32Constant(steps, steps_dims, "composite_dus_steps"));
    auto* fill_layer =
        network_->addFill(update_dims, nvinfer1::FillOperation::kLINSPACE,
                          nvinfer1::DataType::kINT32);
    if (fill_layer == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add composite cache-update index fill");
    }
    fill_layer->setInput(1, *offset_scalar);
    fill_layer->setInput(2, *steps_tensor);
    auto* indices = fill_layer->getOutput(0);
    if (indices == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to build composite cache-update indices");
    }
    auto* layer = network_->addScatter(*operand, *indices, *update,
                                       nvinfer1::ScatterMode::kELEMENT);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add composite cache-update scatter");
    }
    layer->setAxis(partial_axis);
    return layer->getOutput(0);
  }

  // Returns the int8 boundary form of a quantized tensor's value: the raw
  // network input when available, otherwise the current value (re)quantized.
  Expected<nvinfer1::ITensor*> GetRawInt8Tensor(const Tensor& tensor) {
    if (auto raw = raw_input_map_.find(tensor.Get());
        raw != raw_input_map_.end() &&
        raw->second->getType() == nvinfer1::DataType::kINT8) {
      return raw->second;
    }
    LITERT_ASSIGN_OR_RETURN(auto* value, GetTensor(tensor));
    if (value->getType() == nvinfer1::DataType::kINT8) {
      return value;
    }
    return AddQuantizeTensor(value, tensor, nvinfer1::DataType::kINT8);
  }

  // Publishes an int8 scatter result: raw form for the island boundary and,
  // in half-precision mode, a dequantized view for in-island consumers.
  Expected<void> SetQuantizedScatterOutput(const Tensor& tensor,
                                           nvinfer1::ITensor* scatter_out) {
    if (Fp16ActivationsEnabled()) {
      raw_output_map_[tensor.Get()] = scatter_out;
      LITERT_ASSIGN_OR_RETURN(
          auto* value,
          AddDequantizeTensor(tensor, scatter_out, ModeFloatType()));
      return SetOutputTensor(tensor, value);
    }
    return SetOutputTensor(tensor, scatter_out);
  }

  // odml.rms_norm(x[, w]) = x * rsqrt(mean(x^2, last_axis) + eps) [* w].
  // The decomposition's epsilon is 1e-6 for this model family.
  Expected<void> LowerCompositeRmsNorm(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* x, GetTensor(op.Inputs()[0]));
    const int rank = x->getDimensions().nbDims;
    auto* sq_layer =
        network_->addElementWise(*x, *x, nvinfer1::ElementWiseOperation::kPROD);
    if (sq_layer == nullptr || sq_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm square");
    }
    auto* mean_layer = network_->addReduce(
        *sq_layer->getOutput(0), nvinfer1::ReduceOperation::kAVG,
        /*reduceAxes=*/1U << (rank - 1), /*keepDims=*/true);
    if (mean_layer == nullptr || mean_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm mean");
    }
    nvinfer1::Dims one_dims{};
    one_dims.nbDims = rank;
    for (int i = 0; i < rank; ++i) {
      one_dims.d[i] = 1;
    }
    const float eps_value = 1e-6f;
    LITERT_ASSIGN_OR_RETURN(
        auto* eps, AddFloatConstant(absl::MakeConstSpan(&eps_value, 1),
                                    one_dims, "rms_norm_eps", x->getType()));
    auto* add_layer = network_->addElementWise(
        *mean_layer->getOutput(0), *eps, nvinfer1::ElementWiseOperation::kSUM);
    if (add_layer == nullptr || add_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm epsilon");
    }
    auto* sqrt_layer = network_->addUnary(*add_layer->getOutput(0),
                                          nvinfer1::UnaryOperation::kSQRT);
    if (sqrt_layer == nullptr || sqrt_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm sqrt");
    }
    auto* recip_layer = network_->addUnary(*sqrt_layer->getOutput(0),
                                           nvinfer1::UnaryOperation::kRECIP);
    if (recip_layer == nullptr || recip_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm reciprocal");
    }
    auto* norm_layer = network_->addElementWise(
        *x, *recip_layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (norm_layer == nullptr || norm_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm scale");
    }
    nvinfer1::ITensor* out = norm_layer->getOutput(0);
    if (op.Inputs().size() == 2) {
      LITERT_ASSIGN_OR_RETURN(auto* w, GetTensor(op.Inputs()[1]));
      LITERT_ASSIGN_OR_RETURN(w, AddCastTensor(w, out->getType()));
      LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(out, w));
      auto* w_layer = network_->addElementWise(
          *out, *w, nvinfer1::ElementWiseOperation::kPROD);
      if (w_layer == nullptr || w_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add rms_norm weight");
      }
      out = w_layer->getOutput(0);
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  // odml.cache_update: quantize both updates and scatter them into their
  // int8 caches (the second one through a [0,1,3,2] transpose), mirroring
  // the decomposition without its inlined index glue.
  Expected<void> LowerCompositeCacheUpdate(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* update_a, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* update_b, GetTensor(op.Inputs()[1]));
    LITERT_ASSIGN_OR_RETURN(auto* start_a, GetTensor(op.Inputs()[5]));
    LITERT_ASSIGN_OR_RETURN(auto* start_b, GetTensor(op.Inputs()[6]));
    LITERT_ASSIGN_OR_RETURN(auto* cache_a, GetRawInt8Tensor(op.Inputs()[3]));
    LITERT_ASSIGN_OR_RETURN(auto* cache_b, GetRawInt8Tensor(op.Inputs()[4]));
    LITERT_ASSIGN_OR_RETURN(auto* q_a,
                            AddQuantizeTensor(update_a, op.Outputs()[0],
                                              nvinfer1::DataType::kINT8));
    LITERT_ASSIGN_OR_RETURN(auto* q_b,
                            AddQuantizeTensor(update_b, op.Outputs()[1],
                                              nvinfer1::DataType::kINT8));
    auto* shuffle = network_->addShuffle(*q_b);
    if (shuffle == nullptr || shuffle->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add cache_update transpose");
    }
    nvinfer1::Permutation perm{};
    perm.order[0] = 0;
    perm.order[1] = 1;
    perm.order[2] = 3;
    perm.order[3] = 2;
    shuffle->setFirstTranspose(perm);
    LITERT_ASSIGN_OR_RETURN(auto* out_a,
                            BuildScatterUpdate(cache_a, q_a, start_a));
    LITERT_ASSIGN_OR_RETURN(
        auto* out_b,
        BuildScatterUpdate(cache_b, shuffle->getOutput(0), start_b));
    LITERT_RETURN_IF_ERROR(SetQuantizedScatterOutput(op.Outputs()[0], out_a));
    return SetQuantizedScatterOutput(op.Outputs()[1], out_b);
  }

  // odml.runtime_bmm(a, cache, positions) = a @ dequantize(cache)^T over the
  // full context (window-equivalent; see IsCompositeSupported).
  Expected<void> LowerCompositeRuntimeBmm(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* a, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* cache, GetTensor(op.Inputs()[1]));
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    const auto a_dims = a->getDimensions();
    const auto cache_dims = cache->getDimensions();
    nvinfer1::ITensor* padding = nullptr;
    const size_t context_limit = RuntimeBmmContextLimit();
    if (context_limit > 0 && a_dims.nbDims >= 2 &&
        cache_dims.nbDims == a_dims.nbDims &&
        output_dims.nbDims == a_dims.nbDims &&
        context_limit <= static_cast<size_t>(std::numeric_limits<int>::max())) {
      const int rank = a_dims.nbDims;
      const int limit = static_cast<int>(context_limit);
      const int a_context_axis = rank - 1;
      const int cache_context_axis = rank - 2;
      const int cache_transposed_context_axis = rank - 1;
      const bool score_bmm =
          cache_dims.d[cache_context_axis] > limit &&
          output_dims.d[rank - 1] == cache_dims.d[cache_context_axis] &&
          a_dims.d[rank - 1] == cache_dims.d[rank - 1];
      const bool value_bmm =
          cache_dims.d[cache_transposed_context_axis] > limit &&
          a_dims.d[a_context_axis] ==
              cache_dims.d[cache_transposed_context_axis] &&
          output_dims.d[rank - 1] == cache_dims.d[cache_context_axis];
      if (score_bmm) {
        LITERT_ASSIGN_OR_RETURN(
            cache, SliceTensorStatic(cache, cache_context_axis,
                                     /*start_index=*/0, limit,
                                     "runtime_bmm_score_cache_prefix"));
        nvinfer1::Dims padding_dims = output_dims;
        padding_dims.d[rank - 1] -= limit;
        if (padding_dims.d[rank - 1] > 0) {
          LITERT_ASSIGN_OR_RETURN(
              padding, AddZeroFloatTensor(padding_dims, a->getType(),
                                          "runtime_bmm_score_tail_padding"));
        }
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT runtime_bmm score prefix limit=%d "
                   "full_context=%d",
                   limit, output_dims.d[rank - 1]);
      } else if (value_bmm) {
        LITERT_ASSIGN_OR_RETURN(
            a, SliceTensorStatic(a, a_context_axis, /*start_index=*/0, limit,
                                 "runtime_bmm_value_probs_prefix"));
        LITERT_ASSIGN_OR_RETURN(
            cache, SliceTensorStatic(cache, cache_transposed_context_axis,
                                     /*start_index=*/0, limit,
                                     "runtime_bmm_value_cache_prefix"));
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT runtime_bmm value prefix limit=%d "
                   "full_context=%d",
                   limit, cache_dims.d[cache_transposed_context_axis]);
      }
    }
    if (cache->getType() == nvinfer1::DataType::kINT8) {
      LITERT_ASSIGN_OR_RETURN(
          cache, AddDequantizeTensor(op.Inputs()[1], cache, a->getType()));
    }
    LITERT_ASSIGN_OR_RETURN(cache, AddCastTensor(cache, a->getType()));
    auto* layer = network_->addMatrixMultiply(
        *a, nvinfer1::MatrixOperation::kNONE, *cache,
        nvinfer1::MatrixOperation::kTRANSPOSE);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add runtime_bmm matmul");
    }
    nvinfer1::ITensor* out = layer->getOutput(0);
    if (padding != nullptr) {
      std::array<nvinfer1::ITensor*, 2> inputs = {out, padding};
      auto* concat = network_->addConcatenation(inputs.data(), inputs.size());
      if (concat == nullptr || concat->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add runtime_bmm score padding concatenation");
      }
      concat->setAxis(output_dims.nbDims - 1);
      out = concat->getOutput(0);
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerComposite(const Op& op) {
    const std::string name = CompositeOpName(op);
    if (name == "odml.rms_norm") {
      return LowerCompositeRmsNorm(op);
    }
    if (name == "odml.cache_update") {
      return LowerCompositeCacheUpdate(op);
    }
    if (name == "odml.runtime_bmm") {
      return LowerCompositeRuntimeBmm(op);
    }
    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported composite op during lowering");
  }

  Expected<void> LowerOp(const Op& op) {
    switch (op.Code()) {
      case kLiteRtOpCodeTflAdd:
      case kLiteRtOpCodeTflMul:
      case kLiteRtOpCodeTflSub:
      case kLiteRtOpCodeTflDiv:
      case kLiteRtOpCodeTflMaximum:
        return LowerElementwise(op);
      case kLiteRtOpCodeTflLess:
      case kLiteRtOpCodeTflGreaterEqual:
      case kLiteRtOpCodeTflNotEqual:
      case kLiteRtOpCodeTflLogicalAnd:
        return LowerComparison(op);
      case kLiteRtOpCodeTflRelu:
      case kLiteRtOpCodeTflRelu6:
      case kLiteRtOpCodeTflReluN1To1:
        return LowerActivation(op);
      case kLiteRtOpCodeTflSin:
      case kLiteRtOpCodeTflCos:
      case kLiteRtOpCodeTflTanh:
      case kLiteRtOpCodeTflRsqrt:
        return LowerUnary(op);
      case kLiteRtOpCodeTflGelu:
        return LowerGelu(op);
      case kLiteRtOpCodeTflCast:
        return LowerCast(op);
      case kLiteRtOpCodeTflQuantize:
        return LowerQuantize(op);
      case kLiteRtOpCodeTflDequantize:
        return LowerDequantize(op);
      case kLiteRtOpCodeTflFill:
        return LowerFill(op);
      case kLiteRtOpCodeTflSoftmax:
        return LowerSoftmax(op);
      case kLiteRtOpCodeTflReshape:
        return LowerReshape(op);
      case kLiteRtOpCodeTflPack:
        return LowerPack(op);
      case kLiteRtOpCodeTflSlice:
        return LowerSlice(op);
      case kLiteRtOpCodeTflUnpack:
        return LowerUnpack(op);
      case kLiteRtOpCodeTflDynamicUpdateSlice:
        return LowerDynamicUpdateSlice(op);
      case kLiteRtOpCodeTflTranspose:
        return LowerTranspose(op);
      case kLiteRtOpCodeTflConcatenation:
        return LowerConcatenation(op);
      case kLiteRtOpCodeTflBatchMatmul:
        return LowerBatchMatmul(op);
      case kLiteRtOpCodeTflFullyConnected:
        return LowerFullyConnected(op);
      case kLiteRtOpCodeTflSelectV2:
        return LowerSelectV2(op);
      case kLiteRtOpCodeTflSum:
        return LowerSum(op);
      case kLiteRtOpCodeTflReduceMax:
        return LowerReduceMax(op);
      case kLiteRtOpCodeTflFloorMod:
        return LowerFloorMod(op);
      case kLiteRtOpCodeTflConv2d:
        return LowerConv2d(op);
      case kLiteRtOpCodeShloComposite:
        return LowerComposite(op);
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT op during lowering");
    }
  }

  TensorRtLogger logger_;
  SyncCudaGpuAllocator sync_allocator_;
  TrtPtr<nvinfer1::IBuilder> builder_;
  TrtPtr<nvinfer1::INetworkDefinition> network_;
  TrtPtr<nvinfer1::IBuilderConfig> config_;
  std::deque<std::string> names_;
  std::vector<std::vector<uint8_t>> owned_weights_;
  std::vector<OwnedRefitWeight> owned_refit_weights_;
  std::vector<TrtPtr<nvinfer1::IPluginV3>> owned_plugins_;
  std::unordered_map<LiteRtTensor, nvinfer1::ITensor*> tensor_map_;
  // Network inputs in their boundary (pre-FP16-conversion) form; used by
  // in-place style ops (cache updates) that operate on the raw int8 data.
  std::unordered_map<LiteRtTensor, nvinfer1::ITensor*> raw_input_map_;
  // Produced tensors that already carry the boundary format (e.g. int8 cache
  // scatter results); MarkOutputs prefers these over the FP16 view.
  std::unordered_map<LiteRtTensor, nvinfer1::ITensor*> raw_output_map_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  int next_name_id_ = 0;
  bool uses_cuda_subbyte_gemv_ = false;
};

}  // namespace

bool TensorRtSharedWeightsEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS",
                    /*default_value=*/false);
}

bool IsTensorRtOpSupported(const Op& op) {
  Expected<bool> result = false;
  switch (op.Code()) {
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflMul:
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflDiv:
    case kLiteRtOpCodeTflMaximum:
      result = IsElementwiseSupported(op);
      break;
    case kLiteRtOpCodeTflLess:
    case kLiteRtOpCodeTflGreaterEqual:
    case kLiteRtOpCodeTflNotEqual:
      result = IsComparisonSupported(op);
      break;
    case kLiteRtOpCodeTflLogicalAnd:
      result = IsLogicalAndSupported(op);
      break;
    case kLiteRtOpCodeTflRelu:
    case kLiteRtOpCodeTflRelu6:
    case kLiteRtOpCodeTflReluN1To1:
      result = IsUnaryActivationSupported(op);
      break;
    case kLiteRtOpCodeTflSin:
    case kLiteRtOpCodeTflCos:
    case kLiteRtOpCodeTflTanh:
    case kLiteRtOpCodeTflRsqrt:
      result = IsUnaryActivationSupported(op);
      break;
    case kLiteRtOpCodeTflGelu:
      result = IsGeluSupported(op);
      break;
    case kLiteRtOpCodeTflCast:
      result = IsCastSupported(op);
      break;
    case kLiteRtOpCodeTflQuantize:
      result = IsQuantizeSupported(op);
      break;
    case kLiteRtOpCodeTflDequantize:
      result = IsDequantizeSupported(op);
      break;
    case kLiteRtOpCodeTflFill:
      result = IsFillSupported(op);
      break;
    case kLiteRtOpCodeTflSoftmax:
      result = IsSoftmaxSupported(op);
      break;
    case kLiteRtOpCodeTflReshape:
      result = IsReshapeSupported(op);
      break;
    case kLiteRtOpCodeTflPack:
      result = IsPackSupported(op);
      break;
    case kLiteRtOpCodeTflSlice:
      result = IsSliceSupported(op);
      break;
    case kLiteRtOpCodeTflUnpack:
      result = IsUnpackSupported(op);
      break;
    case kLiteRtOpCodeTflDynamicUpdateSlice:
      result = IsDynamicUpdateSliceSupported(op);
      break;
    case kLiteRtOpCodeTflTranspose:
      result = IsTransposeSupported(op);
      break;
    case kLiteRtOpCodeTflConcatenation:
      result = IsConcatenationSupported(op);
      break;
    case kLiteRtOpCodeTflBatchMatmul:
      result = IsBatchMatmulSupported(op);
      break;
    case kLiteRtOpCodeTflFullyConnected:
      result = IsFullyConnectedSupported(op);
      break;
    case kLiteRtOpCodeTflSelectV2:
      result = IsSelectV2Supported(op);
      break;
    case kLiteRtOpCodeTflSum:
      result = IsSumSupported(op);
      break;
    case kLiteRtOpCodeTflReduceMax:
      result = IsReduceMaxSupported(op);
      break;
    case kLiteRtOpCodeTflFloorMod:
      result = IsFloorModSupported(op);
      break;
    case kLiteRtOpCodeTflConv2d:
      result = IsConv2dSupported(op);
      break;
    case kLiteRtOpCodeShloComposite:
      result = IsCompositeSupported(op);
      break;
    default:
      return false;
  }
  return result.HasValue() && result.Value();
}

Expected<TensorRtBuildResult> BuildTensorRtEngine(const Subgraph& subgraph) {
  auto result = [&]() {
    TensorRtGraphBuilder builder;
    return builder.Build(subgraph);
  }();
  LogMemoryProfile("compiler", "graph_builder_destroyed");
  return result;
}

}  // namespace litert::nvidia
