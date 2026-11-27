// Copyright 2026 Amlogic, Inc.
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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <dlfcn.h>

#include "absl/container/flat_hash_map.h" // from @com_google_absl
#include "absl/container/flat_hash_set.h" // from @com_google_absl
#include "absl/strings/str_format.h"      // from @com_google_absl
#include "absl/strings/string_view.h"     // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper_with_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/options/litert_amlogic_options.h"  // IWYU pragma: keep
#include "litert/compiler/cc/litert_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/amlogic/core/aml_compiler_core_loader.h"
#include "litert/vendors/amlogic/core/amlogic_dispatch_info.h"
#include "litert/vendors/amlogic/compiler/node_binary_info.h"

using LiteRtBufferId = uint32_t;
using LiteRtContextHandleIdx = uint32_t;
using WeightSharingMap =
    absl::flat_hash_map<LiteRtBufferId, LiteRtContextHandleIdx>;

namespace {

constexpr char kPluginManufacturer[] = "Amlogic";
constexpr LiteRtParamIndex kDefaultPartitionIndex = 0;

// ADLA compile targets exposed via --soc_model / apply_plugin.
// Keep this list aligned with aml_adla_get_chip_info() and
// kCompileTargetSpecs (axi_sram_size) in tflite_to_convert.cc.
constexpr std::pair<const char *, const char *> kPluginSocModels[] = {
    {"c3", "c3"},
    {"s5", "s5"},
    {"t7c", "t7c"},
    {"t3x", "t3x"},
    {"s6", "s6"},
    {"c4", "c4"},
    {"a9", "a9"},
    {"c5_0", "c5_0"},
    {"c5_1", "c5_1"},
};

constexpr auto kNumPluginSocModels =
    sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

// ADLA_LogLevel: VERBOSE=2, DEBUG=3, INFO=4, WARN=5, ERROR=6, FATAL=7.
// 0 means compiler core reads AML_DELEGATE_LOG_LEVEL itself.
// LiteRT Debug is more verbose than LiteRT Verbose, so the names do not line up.
// Public option -> ADLA_LogLevel. off is FATAL so the core does not fall
// back to the environment variable.
uint32_t AdlaLogLevelFromAmlogicOption(LrtAmlogicOptionsLogLevel level) {
  switch (level) {
    case kLrtAmlogicLogLevelDebug:
      return 3;
    case kLrtAmlogicLogLevelVerbose:
      return 2;
    case kLrtAmlogicLogLevelInfo:
      return 4;
    case kLrtAmlogicLogLevelWarn:
      return 5;
    case kLrtAmlogicLogLevelError:
      return 6;
    case kLrtAmlogicLogOff:
      return 7;
  }
  return 0;
}

uint32_t AdlaLogLevelFromLiteRt() {
  LiteRtLogSeverity severity = kLiteRtLogSeverityInfo;
  if (LiteRtGetMinLoggerSeverity(LiteRtGetDefaultLogger(), &severity) !=
      kLiteRtStatusOk) {
    return 0;
  }
  switch (severity) {
    case kLiteRtLogSeverityDebug:
      return 2;
    case kLiteRtLogSeverityVerbose:
      return 3;
    case kLiteRtLogSeverityInfo:
      return 4;
    case kLiteRtLogSeverityWarning:
      return 5;
    case kLiteRtLogSeverityError:
      return 6;
    case kLiteRtLogSeveritySilent:
      return 7;
    default:
      return 0;
  }
}

// LiteRT TFL opcodes/element types are defined to match TFLite numeric
// values (see litert_op_code.h / litert_model_types.h comments). AML IR
// stores TFLite semantics; keep the conversion explicit at the adapter.
int ToTfLiteBuiltinOp(LiteRtOpCode op_code) {
  return static_cast<int>(op_code);
}

int ToTfLiteType(litert::ElementType element_type) {
  return static_cast<int>(element_type);
}

LiteRtStatus ToLiteRtStatus(AmlCompilerStatus status) {
  switch (status) {
  case kAmlCompilerStatusOk:
    return kLiteRtStatusOk;
  case kAmlCompilerStatusErrorInvalidArgument:
    return kLiteRtStatusErrorInvalidArgument;
  case kAmlCompilerStatusErrorRuntimeFailure:
  default:
    return kLiteRtStatusErrorRuntimeFailure;
  }
}

// Reasons AML Partition will not select an op for NPU.
// Add new enumerators here and handle them in GetAmlogicPartitionRejectReason().
enum class AmlogicPartitionRejectReason {
  kNone = 0,
  kUnsupportedOpcode,      // opcode denylist, leave on CPU
  kNotInAllowlist,         // neither denylist nor ADLA allowlist
  kDynamicTensorShape,     // unranked, or any dim < 0
  kUnsupportedElementType,
  kNonConstWeight,         // conv/fc filter is not a constant
};

const char *AmlogicPartitionRejectReasonName(AmlogicPartitionRejectReason reason) {
  switch (reason) {
  case AmlogicPartitionRejectReason::kNone:
    return "none";
  case AmlogicPartitionRejectReason::kUnsupportedOpcode:
    return "unsupported_opcode";
  case AmlogicPartitionRejectReason::kNotInAllowlist:
    return "opcode_not_in_allowlist";
  case AmlogicPartitionRejectReason::kDynamicTensorShape:
    return "dynamic_tensor_shape";
  case AmlogicPartitionRejectReason::kUnsupportedElementType:
    return "unsupported_element_type";
  case AmlogicPartitionRejectReason::kNonConstWeight:
    return "non_const_weight";
  }
  return "unknown";
}

bool OpcodeInList(LiteRtOpCode code, const LiteRtOpCode *ops, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (ops[i] == code) {
      return true;
    }
  }
  return false;
}

// Opcodes AML NPU does not support; leave them on CPU.
constexpr LiteRtOpCode kAmlogicRejectedOpcodes[] = {
    kLiteRtOpCodeTflConv3d,
    kLiteRtOpCodeTflConv3dTranspose,
    // CUSTOM stays off NPU until custom code and custom options are forwarded.
    kLiteRtOpCodeTflCustom,
};

// ADLA TFLite allowlist from Amlogic_NPU_Support_Operator_List CN v0.1
// (2026-07-15): union of ADLA2/ADLA3 Hardware + Software op lists.
constexpr LiteRtOpCode kAmlogicSupportedOpcodes[] = {
    // ADLA2/ADLA3 Hardware Op List
    kLiteRtOpCodeTflAdd,
    kLiteRtOpCodeTflAveragePool2d,
    kLiteRtOpCodeTflConcatenation,
    kLiteRtOpCodeTflConv2d,
    kLiteRtOpCodeTflDepthwiseConv2d,
    kLiteRtOpCodeTflDepthToSpace,
    kLiteRtOpCodeTflEmbeddingLookup,
    kLiteRtOpCodeTflFullyConnected,
    kLiteRtOpCodeTflL2Pool2d,
    kLiteRtOpCodeTflLogistic,
    kLiteRtOpCodeTflLstm,
    kLiteRtOpCodeTflMaxPool2d,
    kLiteRtOpCodeTflMul,
    kLiteRtOpCodeTflRelu,
    kLiteRtOpCodeTflReluN1To1,
    kLiteRtOpCodeTflRelu6,
    kLiteRtOpCodeTflReshape,
    kLiteRtOpCodeTflResizeBilinear,
    kLiteRtOpCodeTflRnn,
    kLiteRtOpCodeTflSpaceToDepth,
    kLiteRtOpCodeTflTanh,
    kLiteRtOpCodeTflPad,
    kLiteRtOpCodeTflGather,
    kLiteRtOpCodeTflBatchToSpaceNd,
    kLiteRtOpCodeTflSpaceToBatchNd,
    kLiteRtOpCodeTflTranspose,
    kLiteRtOpCodeTflMean,
    kLiteRtOpCodeTflSub,
    kLiteRtOpCodeTflSqueeze,
    kLiteRtOpCodeTflUnidirectionalSequenceLstm,
    kLiteRtOpCodeTflStridedSlice,
    kLiteRtOpCodeTflSplit,
    kLiteRtOpCodeTflBidirectionalSequenceLstm,
    kLiteRtOpCodeTflPrelu,
    kLiteRtOpCodeTflMaximum,
    kLiteRtOpCodeTflMinimum,
    kLiteRtOpCodeTflLess,
    kLiteRtOpCodeTflNeg,
    kLiteRtOpCodeTflPadv2,
    kLiteRtOpCodeTflGreater,
    kLiteRtOpCodeTflGreaterEqual,
    kLiteRtOpCodeTflLessEqual,
    kLiteRtOpCodeTflSelect,
    kLiteRtOpCodeTflSlice,
    kLiteRtOpCodeTflTransposeConv,
    kLiteRtOpCodeTflTile,
    kLiteRtOpCodeTflExpandDims,
    kLiteRtOpCodeTflEqual,
    kLiteRtOpCodeTflNotEqual,
    kLiteRtOpCodeTflSum,
    kLiteRtOpCodeTflShape,
    kLiteRtOpCodeTflReduceMax,
    kLiteRtOpCodeTflPack,
    kLiteRtOpCodeTflLogicalOr,
    kLiteRtOpCodeTflLogicalAnd,
    kLiteRtOpCodeTflLogicalNot,
    kLiteRtOpCodeTflUnpack,
    kLiteRtOpCodeTflReduceMin,
    kLiteRtOpCodeTflReduceAny,
    kLiteRtOpCodeTflSquare,
    kLiteRtOpCodeTflFill,
    kLiteRtOpCodeTflResizeNearestNeighbor,
    kLiteRtOpCodeTflLeakyRelu,
    kLiteRtOpCodeTflSquaredDifference,
    kLiteRtOpCodeTflMirrorPad,
    kLiteRtOpCodeTflAbs,
    kLiteRtOpCodeTflSplitV,
    kLiteRtOpCodeTflReverseV2,
    kLiteRtOpCodeTflAddN,
    kLiteRtOpCodeTflGatherNd,
    kLiteRtOpCodeTflQuantize,
    kLiteRtOpCodeTflHardSwish,
    kLiteRtOpCodeTflSelectV2,
    kLiteRtOpCodeTflDensify,
    kLiteRtOpCodeTflBatchMatmul,
    kLiteRtOpCodeTflCumsum,
    kLiteRtOpCodeTflBroadcastTo,
    kLiteRtOpCodeTflReduceAll,
    kLiteRtOpCodeTflBroadcastArgs,
    // ADLA2 Software Op List (ADLA3 HW extras included here)
    kLiteRtOpCodeTflDequantize,
    kLiteRtOpCodeTflFloor,
    kLiteRtOpCodeTflL2Normalization,
    kLiteRtOpCodeTflLocalResponseNormalization,
    kLiteRtOpCodeTflSoftmax,
    kLiteRtOpCodeTflDiv,
    kLiteRtOpCodeTflExp,
    kLiteRtOpCodeTflTopkV2,
    kLiteRtOpCodeTflLogSoftmax,
    kLiteRtOpCodeTflCast,
    kLiteRtOpCodeTflArgMax,
    kLiteRtOpCodeTflSin,
    kLiteRtOpCodeTflLog,
    kLiteRtOpCodeTflSqrt,
    kLiteRtOpCodeTflRsqrt,
    kLiteRtOpCodeTflPow,
    kLiteRtOpCodeTflArgMin,
    kLiteRtOpCodeTflReduceProd,
    kLiteRtOpCodeTflOneHot,
    kLiteRtOpCodeTflFloorDiv,
    kLiteRtOpCodeTflFloorMod,
    kLiteRtOpCodeTflRange,
    kLiteRtOpCodeTflUnique,
    kLiteRtOpCodeTflCeil,
    kLiteRtOpCodeTflCos,
    kLiteRtOpCodeTflElu,
    kLiteRtOpCodeTflRound,
    kLiteRtOpCodeTflNonMaxSuppressionV4,
    kLiteRtOpCodeTflNonMaxSuppressionV5,
    kLiteRtOpCodeTflScatterNd,
    kLiteRtOpCodeTflMultinomial,
    kLiteRtOpCodeTflGelu,
};

bool IsAmlogicRejectedOpcode(LiteRtOpCode code) {
  return OpcodeInList(code, kAmlogicRejectedOpcodes,
                      sizeof(kAmlogicRejectedOpcodes) /
                          sizeof(kAmlogicRejectedOpcodes[0]));
}

bool IsAmlogicSupportedOpcode(LiteRtOpCode code) {
  return OpcodeInList(code, kAmlogicSupportedOpcodes,
                      sizeof(kAmlogicSupportedOpcodes) /
                          sizeof(kAmlogicSupportedOpcodes[0]));
}

// True if the tensor is unranked or has a dynamic dimension (typically -1).
// Rank-0 scalars have an empty dim list and are not treated as dynamic.
bool TensorHasDynamicDims(const litert::compiler::Tensor &tensor) {
  if (tensor.TypeId() != kLiteRtRankedTensorType) {
    return true;
  }
  auto ranked_type_or = tensor.RankedTensorType();
  if (!ranked_type_or) {
    return true;
  }
  for (const auto dim : ranked_type_or->Layout().Dimensions()) {
    if (dim < 0) {
      return true;
    }
  }
  return false;
}

template <typename TensorList>
bool AnyTensorHasDynamicDims(const TensorList &tensors) {
  for (const auto &tensor : tensors) {
    if (TensorHasDynamicDims(tensor)) {
      return true;
    }
  }
  return false;
}

// CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE_CONV and FULLY_CONNECTED share one
// filter slot (input 1). NPU compile bakes that filter with
// AmlTensorSetConstData. If it is not constant, leave the op on CPU.
bool FilterWeightIsNonConstant(const litert::compiler::Op &op) {
  switch (op.Code()) {
    case kLiteRtOpCodeTflConv2d:
    case kLiteRtOpCodeTflDepthwiseConv2d:
    case kLiteRtOpCodeTflTransposeConv:
    case kLiteRtOpCodeTflFullyConnected:
      break;
    default:
      return false;
  }
  const std::vector<litert::compiler::Tensor> inputs = op.Inputs();
  constexpr size_t kFilterInput = 1;
  if (inputs.size() <= kFilterInput) {
    return true;
  }
  return !inputs[kFilterInput].IsConstant();
}

// Element types ADLA can compile. Other types stay on CPU.
bool IsAmlogicSupportedElementType(litert::ElementType element_type) {
  switch (element_type) {
    case litert::ElementType::Bool:
    case litert::ElementType::Int8:
    case litert::ElementType::Int16:
    case litert::ElementType::Int32:
    case litert::ElementType::UInt8:
    case litert::ElementType::Float16:
    case litert::ElementType::Float32:
      return true;
    default:
      return false;
  }
}

template <typename TensorList>
bool AnyUnsupportedElementType(const TensorList &tensors) {
  for (const auto &tensor : tensors) {
    if (TensorHasDynamicDims(tensor)) {
      continue;
    }
    if (!IsAmlogicSupportedElementType(tensor.ElementType())) {
      return true;
    }
  }
  return false;
}

// Extension window for NPU-unsupported ops. First match wins; do not PushOp.
// Opcode support: denylist first, then ADLA allowlist. Unknown opcodes log
// and stay on CPU.
AmlogicPartitionRejectReason GetAmlogicPartitionRejectReason(const litert::compiler::Op &op) {
  const LiteRtOpCode opcode = op.Code();
  if (IsAmlogicRejectedOpcode(opcode)) {
    return AmlogicPartitionRejectReason::kUnsupportedOpcode;
  }
  if (!IsAmlogicSupportedOpcode(opcode)) {
    return AmlogicPartitionRejectReason::kNotInAllowlist;
  }
  if (AnyTensorHasDynamicDims(op.Inputs()) ||
      AnyTensorHasDynamicDims(op.Outputs())) {
    return AmlogicPartitionRejectReason::kDynamicTensorShape;
  }
  if (AnyUnsupportedElementType(op.Inputs()) ||
      AnyUnsupportedElementType(op.Outputs())) {
    return AmlogicPartitionRejectReason::kUnsupportedElementType;
  }
  if (FilterWeightIsNonConstant(op)) {
    return AmlogicPartitionRejectReason::kNonConstWeight;
  }
  return AmlogicPartitionRejectReason::kNone;
}

std::optional<const char *> FindSocModel(absl::string_view soc_model_name) {
  for (auto i = 0; i < kNumPluginSocModels; ++i) {
    if (soc_model_name == kPluginSocModels[i].first) {
      return kPluginSocModels[i].second;
    }
  }
  return std::nullopt;
}

LiteRtStatus CheckIr(AmlCompilerStatus status, const char *what) {
  if (status == kAmlCompilerStatusOk) {
    return kLiteRtStatusOk;
  }
  LITERT_LOG(LITERT_ERROR, "%s failed status=%d", what,
             static_cast<int>(status));
  return ToLiteRtStatus(status);
}

using AmlogicTensorMap = absl::flat_hash_map<LiteRtTensor, AmlTensor>;

LiteRtStatus FillAmlogicTensor(AmlTensor t, const litert::compiler::Tensor &tensor,
                           bool copy_const_weights, int aml_index) {
  const aml_compiler_core::FuncTable* api = aml_compiler_core::Funcs(true);
  if (api == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  LiteRtStatus st =
      CheckIr(api->AmlTensorSetName(t, std::string(tensor.Name()).c_str()),
              "AmlTensorSetName");
  if (st != kLiteRtStatusOk) {
    return st;
  }
  st =       CheckIr(api->AmlTensorSetDtype(t, ToTfLiteType(tensor.ElementType())),
               "AmlTensorSetDtype");
  if (st != kLiteRtStatusOk) {
    return st;
  }
  // Do not use LiteRT TensorIndex(): CloneTo copies decomp-local indices
  // (0/1/2) into the inlined main graph, so multiple tensors collide.
  st =       CheckIr(api->AmlTensorSetIndex(t, aml_index), "AmlTensorSetIndex");
  if (st != kLiteRtStatusOk) {
    return st;
  }
  auto ranked_type_or = tensor.RankedTensorType();
  if (ranked_type_or) {
    auto dims = ranked_type_or->Layout().Dimensions();
    std::vector<int> dim_vec(dims.begin(), dims.end());
    if (!dim_vec.empty()) {
      st =       CheckIr(api->AmlTensorSetDims(t, dim_vec.data(),
                                    static_cast<uint32_t>(dim_vec.size())),
                   "AmlTensorSetDims");
      if (st != kLiteRtStatusOk) {
        return st;
      }
    }
  }
  std::vector<int> zp;
  std::vector<float> scale;
  if (tensor.QTypeId() == kLiteRtQuantizationPerTensor) {
    LiteRtQuantizationPerTensor per_tensor = tensor.PerTensorQuantization();
    zp.push_back(static_cast<int>(per_tensor.zero_point));
    scale.push_back(per_tensor.scale);
  } else if (tensor.QTypeId() == kLiteRtQuantizationPerChannel) {
    LiteRtQuantizationPerChannel per_channel =
        tensor.PerChannelQuantization();
    for (int num = 0; num < per_channel.num_channels; num++) {
      zp.push_back(static_cast<int>(per_channel.zero_points[num]));
      scale.push_back(per_channel.scales[num]);
    }
  }
  if (!zp.empty() || !scale.empty()) {
    st =     CheckIr(api->AmlTensorSetQuant(t, zp.empty() ? nullptr : zp.data(),
                                   static_cast<uint32_t>(zp.size()),
                                   scale.empty() ? nullptr : scale.data(),
                                   static_cast<uint32_t>(scale.size())),
                 "AmlTensorSetQuant");
    if (st != kLiteRtStatusOk) {
      return st;
    }
  }
  if (copy_const_weights && !tensor.DefiningOp().has_value() &&
      tensor.HasWeights()) {
    auto bytes = tensor.Weights().Bytes();
    st = CheckIr(api->AmlTensorSetConstData(t, bytes.data(), bytes.size()),
                 "AmlTensorSetConstData");
    if (st != kLiteRtStatusOk) {
      return st;
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus EnsureTensor(AmlGraph graph, AmlogicTensorMap *tensors,
                          const litert::compiler::Tensor &tensor,
                          bool copy_const_weights, AmlTensor *out) {
  const aml_compiler_core::FuncTable* api = aml_compiler_core::Funcs(true);
  if (api == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  LiteRtTensor key = tensor.Get();
  auto it = tensors->find(key);
  if (it != tensors->end()) {
    if (copy_const_weights && !tensor.DefiningOp().has_value() &&
        tensor.HasWeights()) {
      auto bytes = tensor.Weights().Bytes();
      LiteRtStatus st =
          CheckIr(api->AmlTensorSetConstData(it->second, bytes.data(), bytes.size()),
                  "AmlTensorSetConstData");
      if (st != kLiteRtStatusOk) {
        return st;
      }
    }
    *out = it->second;
    return kLiteRtStatusOk;
  }
  AmlTensor t = nullptr;
  LiteRtStatus st =       CheckIr(api->AmlTensorCreate(graph, &t), "AmlTensorCreate");
  if (st != kLiteRtStatusOk) {
    return st;
  }
  const int aml_index = static_cast<int>(tensors->size());
  st = FillAmlogicTensor(t, tensor, copy_const_weights, aml_index);
  if (st != kLiteRtStatusOk) {
    return st;
  }
  (*tensors)[key] = t;
  *out = t;
  return kLiteRtStatusOk;
}

LiteRtStatus FillAmlogicGraphIo(const litert::compiler::Subgraph &subgraph, AmlGraph graph,
                            AmlogicTensorMap *tensors) {
  const aml_compiler_core::FuncTable* api = aml_compiler_core::Funcs(true);
  if (api == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  for (const auto &tensor : subgraph.Inputs()) {
    AmlTensor t = nullptr;
    LiteRtStatus st =
        EnsureTensor(graph, tensors, tensor, /*copy_const_weights=*/false, &t);
    if (st != kLiteRtStatusOk) {
      return st;
    }
    st =       CheckIr(api->AmlGraphAddInput(graph, t), "AmlGraphAddInput");
    if (st != kLiteRtStatusOk) {
      return st;
    }
  }
  for (const auto &tensor : subgraph.Outputs()) {
    AmlTensor t = nullptr;
    LiteRtStatus st =
        EnsureTensor(graph, tensors, tensor, /*copy_const_weights=*/false, &t);
    if (st != kLiteRtStatusOk) {
      return st;
    }
    st =       CheckIr(api->AmlGraphAddOutput(graph, t), "AmlGraphAddOutput");
    if (st != kLiteRtStatusOk) {
      return st;
    }
  }
  return kLiteRtStatusOk;
}

// DENSIFY has no AML sparsity parameters. The constant input is already the
// dense payload (see ExpandSparseConstant). Attach those bytes to the output
// tensor and do not emit a DENSIFY node, so the next op receives
// element_count * element_size (float16 = element_count * 2).
LiteRtStatus FoldDensifyConst(const litert::compiler::Op &op, AmlGraph graph,
                              AmlogicTensorMap *tensors) {
  const aml_compiler_core::FuncTable* api = aml_compiler_core::Funcs(true);
  if (api == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  const auto inputs = op.Inputs();
  const auto outputs = op.Outputs();
  if (inputs.size() != 1 || outputs.size() != 1 || !inputs[0].HasWeights()) {
    LITERT_LOG(LITERT_ERROR,
               "DENSIFY requires one constant input and one output");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto &in = inputs[0];
  const auto &out = outputs[0];
  auto out_type = out.RankedTensorType();
  if (!out_type) {
    return out_type.Error().Status();
  }
  auto dense_size = out_type->Bytes();
  if (!dense_size) {
    return dense_size.Error().Status();
  }
  const absl::Span<const uint8_t> bytes = in.Weights().Bytes();
  if (bytes.size() != *dense_size) {
    const std::string tensor_name(out.Name());
    LITERT_LOG(LITERT_ERROR,
               "DENSIFY '%s' const size %zu != dense size %zu",
               tensor_name.c_str(), bytes.size(), *dense_size);
    return kLiteRtStatusErrorInvalidArgument;
  }
  AmlTensor t = nullptr;
  LiteRtStatus st =
      EnsureTensor(graph, tensors, out, /*copy_const_weights=*/false, &t);
  if (st != kLiteRtStatusOk) {
    return st;
  }
  st = CheckIr(api->AmlTensorSetConstData(t, bytes.data(), bytes.size()),
               "AmlTensorSetConstData");
  if (st != kLiteRtStatusOk) {
    return st;
  }
  const std::string tensor_name(out.Name());
  LITERT_LOG(LITERT_INFO,
             "DENSIFY folded to const '%s': data_size=%zu",
             tensor_name.c_str(), bytes.size());
  return kLiteRtStatusOk;
}

LiteRtStatus ConvertOpToAmlogicNode(const litert::compiler::Op &op, int node_index,
                                AmlGraph graph, AmlogicTensorMap *tensors) {
  const aml_compiler_core::FuncTable* api = aml_compiler_core::Funcs(true);
  if (api == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (op.Code() == kLiteRtOpCodeTflDensify && !op.Inputs().empty() &&
      op.Inputs()[0].HasWeights()) {
    return FoldDensifyConst(op, graph, tensors);
  }
  AmlNode node = nullptr;
  LiteRtStatus st =
      CheckIr(api->AmlNodeCreate(graph, ToTfLiteBuiltinOp(op.Code()), node_index,
                            &node),
              "AmlNodeCreate");
  if (st != kLiteRtStatusOk) {
    return st;
  }

  LiteRtStatus parse_status =
      ParseOpParams(node, ToTfLiteBuiltinOp(op.Code()), op.Get());
  if (parse_status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR,
               "ParseOpParams failed for op_code=%d node_index=%d status=%d",
               static_cast<int>(op.Code()), node_index,
               static_cast<int>(parse_status));
    return parse_status;
  }

  // AmlTensorSetQuant has no axis argument. Pass the per-channel axis as an
  // int32 node param. Prefer the first weight tensor so a depthwise filter
  // (axis 3) is not overwritten by a later per-channel bias (axis 0).
  bool quant_dim_set = false;
  int32_t fallback_quant_dim = 0;
  bool have_fallback_quant_dim = false;
  for (const auto &tensor : op.Inputs()) {
    AmlTensor t = nullptr;
    st = EnsureTensor(graph, tensors, tensor, /*copy_const_weights=*/true, &t);
    if (st != kLiteRtStatusOk) {
      return st;
    }
    st =       CheckIr(api->AmlNodeAddInput(node, t), "AmlNodeAddInput");
    if (st != kLiteRtStatusOk) {
      return st;
    }
    if (tensor.QTypeId() != kLiteRtQuantizationPerChannel) {
      continue;
    }
    const int32_t quantized_dimension =
        tensor.PerChannelQuantization().quantized_dimension;
    if (tensor.HasWeights() && !quant_dim_set) {
      st = CheckIr(api->AmlNodeSetParamI(node, "quantized_dimension",
                                    static_cast<int>(quantized_dimension)),
                   "AmlNodeSetParamI quantized_dimension");
      if (st != kLiteRtStatusOk) {
        return st;
      }
      quant_dim_set = true;
    } else if (!have_fallback_quant_dim) {
      fallback_quant_dim = quantized_dimension;
      have_fallback_quant_dim = true;
    }
  }
  if (!quant_dim_set && have_fallback_quant_dim) {
    st = CheckIr(api->AmlNodeSetParamI(node, "quantized_dimension",
                                  static_cast<int>(fallback_quant_dim)),
                 "AmlNodeSetParamI quantized_dimension");
    if (st != kLiteRtStatusOk) {
      return st;
    }
  }
  for (const auto &tensor : op.Outputs()) {
    AmlTensor t = nullptr;
    st = EnsureTensor(graph, tensors, tensor, /*copy_const_weights=*/false, &t);
    if (st != kLiteRtStatusOk) {
      return st;
    }
    st =       CheckIr(api->AmlNodeAddOutput(node, t), "AmlNodeAddOutput");
    if (st != kLiteRtStatusOk) {
      return st;
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus FillAmlogicGraphNodes(const litert::compiler::Subgraph &subgraph, AmlGraph graph,
                               AmlogicTensorMap *tensors) {
  int node_index = 0;
  for (const auto &op : subgraph.Ops()) {
    LiteRtStatus status =
        ConvertOpToAmlogicNode(op, node_index++, graph, tensors);
    if (status != kLiteRtStatusOk) {
      return status;
    }
  }
  return kLiteRtStatusOk;
}

}

// Plugins can hold state.
//
// Configurations
//
struct LiteRtCompiledResultT {
  std::vector<std::vector<char>> context_bin;
  std::vector<std::string> graph_names;
  // byte_code_index[i] is the bytecode module for call i. AML emits one module.
  std::vector<size_t> byte_code_index;
};
//
// Plugin Definition
//
class LiteRtCompilerPluginT {
 public:
  LiteRtCompilerPluginT(const LiteRtCompilerContext* compiler_context,
                        LiteRtEnvironmentOptions env_options,
                        LiteRtOptions litert_options)
      : compiler_context_(compiler_context) {
    (void)env_options;
    LITERT_LOG(LITERT_DEBUG, "[AML] LiteRtCompilerPluginT init");

    // These two entry points must run inside libLiteRt.so. The plugin's own
    // copies walk std::string / std::unique_ptr built by the runtime's libc++.
    if (!litert_options || compiler_context_ == nullptr ||
        compiler_context_->get_opaque_options == nullptr ||
        compiler_context_->find_opaque_options_data == nullptr) {
      return;
    }

    LiteRtOpaqueOptions opaque_options = nullptr;
    if (compiler_context_->get_opaque_options(litert_options, &opaque_options) !=
            kLiteRtStatusOk ||
        opaque_options == nullptr) {
      return;
    }

    void* payload = nullptr;
    if (compiler_context_->find_opaque_options_data(
            opaque_options, LrtAmlogicOptionsGetIdentifier(), &payload) !=
            kLiteRtStatusOk ||
        payload == nullptr) {
      return;
    }
    if (LrtCreateAmlogicOptionsFromToml(static_cast<const char*>(payload),
                                    &aml_options_) != kLiteRtStatusOk) {
      aml_options_ = nullptr;
    }
  }

  ~LiteRtCompilerPluginT() { LrtDestroyAmlogicOptions(aml_options_); }

  std::string GetModelNameOrDefault() const {
    if (aml_options_ != nullptr) {
      const char* model_name = nullptr;
      if (LrtAmlogicOptionsGetModelName(aml_options_, &model_name) ==
              kLiteRtStatusOk &&
          model_name != nullptr && model_name[0] != '\0') {
        return model_name;
      }
    }
    // Unnamed models must not share one default, or later compiles overwrite
    // earlier ADLA files.
    static std::atomic<int> next_id{0};
    return absl::StrFormat("aml_model_%d", next_id.fetch_add(1));
  }

  uint32_t GetCompilerLogLevel() const {
    if (aml_options_ != nullptr) {
      LrtAmlogicOptionsLogLevel level;
      if (LrtAmlogicOptionsGetLogLevel(aml_options_, &level) == kLiteRtStatusOk) {
        return AdlaLogLevelFromAmlogicOption(level);
      }
    }
    return AdlaLogLevelFromLiteRt();
  }

  std::string GetModelPathOrEmpty() const {
    if (aml_options_ == nullptr) {
      return "";
    }
    const char* model_path = nullptr;
    if (LrtAmlogicOptionsGetModelPath(aml_options_, &model_path) !=
            kLiteRtStatusOk ||
        model_path == nullptr) {
      return "";
    }
    return model_path;
  }

  void RecordPartitionSubgraph(LiteRtSubgraph subgraph) {
    if (partitioned_subgraphs_.insert(subgraph).second) {
      ++partition_subgraph_count_;
    }
  }

  int GetPartitionSubgraphCount() const { return partition_subgraph_count_; }

  const LiteRtCompilerContext* ctx() const { return compiler_context_; }

 private:
  const LiteRtCompilerContext* compiler_context_ = nullptr;
  LrtAmlogicOptions aml_options_ = nullptr;
  absl::flat_hash_set<LiteRtSubgraph> partitioned_subgraphs_;
  int partition_subgraph_count_ = 0;
};
extern "C" LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion *api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

extern "C" const char *LiteRtGetCompilerPluginSocManufacturer() {
  return kPluginManufacturer;
}

extern "C" LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators *supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex *num_supported_soc_models) {
  LITERT_LOG(LITERT_DEBUG,
             "[AML] LiteRtGetNumCompilerPluginSupportedSocModels enter");
  if (!compiler_plugin || !num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = kNumPluginSocModels;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char **soc_model_name) {
  LITERT_LOG(LITERT_DEBUG,
             "[AML] LiteRtGetCompilerPluginSupportedSocModel enter");
  if (!compiler_plugin || !soc_model_name) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (soc_model_idx < 0 || soc_model_idx >= kNumPluginSocModels) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = kPluginSocModels[soc_model_idx].first;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetCompilerPluginSDKVersion(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  if (compiler_plugin == nullptr || sdk_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Probe only: do not force-load the SDK (and its libcompiler_adla.so
  // dependency) just to read a version string. Unmapped (e.g. unit tests)
  // reports the fallback below.
  const aml_compiler_core::FuncTable* api = aml_compiler_core::Funcs(false);
  const AmlCompilerCoreVersion* version =
      api != nullptr && api->AmlCompilerCoreGetVersion != nullptr
          ? api->AmlCompilerCoreGetVersion()
          : nullptr;
  if (version != nullptr && version->string != nullptr) {
    *sdk_version = version->string;
    return kLiteRtStatusOk;
  }
  static const char kAmlogicSdkVersion[] = "aml-adla";
  *sdk_version = kAmlogicSdkVersion;
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

extern "C" LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void **byte_code, size_t *byte_code_size) {
  LITERT_LOG(LITERT_DEBUG, "[AML] LiteRtGetCompiledResultByteCode enter");
  if (!compiled_result || !byte_code || !byte_code_size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (byte_code_idx >= compiled_result->context_bin.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  *byte_code = compiled_result->context_bin[byte_code_idx].data();
  *byte_code_size = compiled_result->context_bin[byte_code_idx].size();
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void **call_info, size_t *call_info_size,
    LiteRtParamIndex *byte_code_idx) {
  if (!compiled_result || !call_info || !call_info_size || !byte_code_idx) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (call_idx >= compiled_result->graph_names.size() ||
      call_idx >= compiled_result->byte_code_index.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  *call_info = compiled_result->graph_names.at(call_idx).data();
  *call_info_size = compiled_result->graph_names.at(call_idx).size();
  *byte_code_idx = compiled_result->byte_code_index[call_idx];

  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex *num_calls) {
  if (!compiled_result || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->graph_names.size();
  return kLiteRtStatusOk;
}

extern "C" void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

extern "C" LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex *num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->context_bin.size();
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  LiteRtPropagateMinLoggerSeverityWithCompilerContext(compiler_context, env);
  if (options == nullptr || env == nullptr) {
    LITERT_LOG(LITERT_WARNING,
               "AML compiler plugin created with null options, these will be "
               "defaulted.");
  }
  *compiler_plugin = new LiteRtCompilerPluginT(compiler_context, env, options);
  return kLiteRtStatusOk;
}

extern "C" void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

extern "C" LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                                      const char *soc_model,
                                                      LiteRtSubgraph subgraph,
                                                      LiteRtOpList selected_ops) {
  if (compiler_plugin == nullptr || compiler_plugin->ctx() == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  compiler_plugin->RecordPartitionSubgraph(subgraph);

  litert::compiler::Subgraph graph(compiler_plugin->ctx(), subgraph);

  LITERT_LOG(LITERT_INFO, "Starting AML Partitioning");

  for (const auto &op : graph.Ops()) {
    // odml.npu_call is owned by PartitionModel. Vendor code cannot rename it
    // without LiteRT Core, so leave every composite on the framework path.
    if (op.Is(kLiteRtOpCodeShloComposite)) {
      continue;
    }
    const AmlogicPartitionRejectReason reject = GetAmlogicPartitionRejectReason(op);
    if (reject != AmlogicPartitionRejectReason::kNone) {
      LITERT_LOG(LITERT_DEBUG,
                 "AML skip op_code=%d reason=%s (leave on CPU)",
                 static_cast<int>(op.Code()),
                 AmlogicPartitionRejectReasonName(reject));
      continue;
    }
    LiteRtPushOp(selected_ops, op.Get(), kDefaultPartitionIndex);
  }

  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char *soc_model,
    LiteRtModel partitions, LiteRtCompiledResult *compiled_result) {
  auto *plugin = compiler_plugin;
  if (plugin == nullptr || plugin->ctx() == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const std::string model_name =
      plugin ? plugin->GetModelNameOrDefault() : std::string("aml_model_0");
  const std::string model_path = plugin ? plugin->GetModelPathOrEmpty() : "";
  LITERT_LOG(LITERT_DEBUG, "model_name: %s", model_name.c_str());
  LITERT_LOG(LITERT_DEBUG, "model_path: %s", model_path.c_str());

  litert::compiler::Model model(plugin->ctx(), partitions);
  const auto num_partitions = model.NumSubgraphs();
  LITERT_LOG(LITERT_DEBUG, "partition_subgraphs=%d, compile_subgraphs=%zu",
             plugin ? plugin->GetPartitionSubgraphCount() : 0, num_partitions);

  auto result = std::make_unique<LiteRtCompiledResultT>();
  // One shared ADLA module. Every call's byte_code_index points at it.
  result->byte_code_index.assign(num_partitions, 0);

  // Online: soc_model is often unset → keep empty so FinalizeAndCompile /
  // ApplyCompileTargetConfig queries the device via aml_adla_get_chip_info().
  // Do NOT default to kPluginSocModels[0] ("c3").
  std::string compile_target;
  if (soc_model != nullptr && soc_model[0] != '\0') {
    auto resolved_target = FindSocModel(soc_model);
    if (!resolved_target.has_value()) {
      LITERT_LOG(LITERT_ERROR, "Unsupported AML soc_model: %s", soc_model);
      return kLiteRtStatusErrorInvalidArgument;
    }
    compile_target = *resolved_target;
  }
  LITERT_LOG(LITERT_INFO, "AML compile target (soc_model): %s",
             compile_target.empty() ? "(empty, device query)"
                                    : compile_target.c_str());

  const aml_compiler_core::FuncTable* api = aml_compiler_core::Funcs(true);
  if (api == nullptr) {
    LITERT_LOG(LITERT_ERROR, "AML compiler SDK (%s) unavailable",
               aml_compiler_core::kSoName);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  AmlContext ctx = api->AmlContextCreate();
  if (ctx == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }

  for (int partition_idx = 0; partition_idx < num_partitions; ++partition_idx) {
    auto subgraph_or = model.Subgraph(partition_idx);
    if (!subgraph_or) {
      api->AmlContextDestroy(ctx);
      return subgraph_or.Error().Status();
    }
    auto subgraph = std::move(*subgraph_or);
    LITERT_LOG(LITERT_DEBUG, "==============================");
    LITERT_LOG(LITERT_DEBUG, "Subgraph index: %d", partition_idx);
    std::string &entry_point_name = result->graph_names.emplace_back();
    entry_point_name = absl::StrFormat(kAmlogicPartitionCallInfoFmt, partition_idx);

    AmlGraph graph = nullptr;
    LiteRtStatus       ir_status = CheckIr(api->AmlGraphCreate(ctx, &graph), "AmlGraphCreate");
    if (ir_status == kLiteRtStatusOk) {
      ir_status = CheckIr(api->AmlGraphSetTarget(graph, compile_target.c_str()),
                          "AmlGraphSetTarget");
    }
    if (ir_status == kLiteRtStatusOk) {
      ir_status = CheckIr(api->AmlGraphSetName(graph, entry_point_name.c_str()),
                          "AmlGraphSetName");
    }
    AmlogicTensorMap tensors;
    if (ir_status == kLiteRtStatusOk) {
      ir_status = FillAmlogicGraphIo(subgraph, graph, &tensors);
    }
    if (ir_status == kLiteRtStatusOk) {
      ir_status = FillAmlogicGraphNodes(subgraph, graph, &tensors);
    }
    if (ir_status != kLiteRtStatusOk) {
      api->AmlContextDestroy(ctx);
      return ir_status;
    }

  }

  AmlCompilerExternalOptions external_options;
  api->AmlCompilerExternalOptionsInitDefaults(&external_options);
  external_options.log_level =
      plugin != nullptr ? plugin->GetCompilerLogLevel()
                        : AdlaLogLevelFromLiteRt();
  LITERT_LOG(LITERT_INFO, "AML compiler core log_level=%u",
             external_options.log_level);

  AmlCompilerCoreOptions options = {};
  api->AmlCompilerCoreOptionsInitDefaults(&options);
  options.compile_target =
      compile_target.empty() ? nullptr : compile_target.c_str();
  options.compile_subgraphs = num_partitions;
  // Always embed. Architecture is not a proxy for offline vs on-device compile.
  options.embed_adla_in_result = 1;
  options.external_options = &external_options;
  options.external_options_size = sizeof(external_options);

  AmlCompilerCoreCompiledArtifact artifact = nullptr;
  const AmlCompilerStatus compile_status = api->AmlContextCompile(
      ctx, model_name.c_str(), model_path.c_str(), &options, &artifact);
  api->AmlContextDestroy(ctx);
  if (compile_status != kAmlCompilerStatusOk || artifact == nullptr) {
    LITERT_LOG(LITERT_ERROR, "AmlContextCompile failed status=%d",
               static_cast<int>(compile_status));
    api->AmlCompilerCoreDestroyCompiledArtifact(artifact);
    return ToLiteRtStatus(compile_status != kAmlCompilerStatusOk
                              ? compile_status
                              : kAmlCompilerStatusErrorRuntimeFailure);
  }

  const void *adla_bin_ptr = nullptr;
  size_t adla_bin_size = 0;
  const char *adla_path_cstr = nullptr;
  const char *fp_cstr = nullptr;
  int cache_hit = 0;
  api->AmlCompilerCoreGetCompiledArtifactAdlaBinary(artifact, &adla_bin_ptr,
                                                    &adla_bin_size);
  api->AmlCompilerCoreGetCompiledArtifactAdlaPath(artifact, &adla_path_cstr);
  api->AmlCompilerCoreGetCompiledArtifactFingerprint(artifact, &fp_cstr);
  api->AmlCompilerCoreGetCompiledArtifactCacheHit(artifact, &cache_hit);
  LITERT_LOG(LITERT_INFO,
             "compiler core done path=%s fp=%s cache_hit=%d embed_size=%zu",
             adla_path_cstr ? adla_path_cstr : "",
             fp_cstr ? fp_cstr : "", cache_hit, adla_bin_size);

  if (adla_bin_ptr == nullptr || adla_bin_size == 0) {
    LITERT_LOG(LITERT_ERROR,
               "AmlContextCompile produced no embedded ADLA binary");
    api->AmlCompilerCoreDestroyCompiledArtifact(artifact);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // One bytecode module: the raw ADLA binary. Call info carries subgraph index.
  result->context_bin.resize(1);
  const auto *adla_bytes = static_cast<const char *>(adla_bin_ptr);
  result->context_bin[0].assign(adla_bytes, adla_bytes + adla_bin_size);
  LITERT_LOG(LITERT_INFO,
             "embedded one ADLA bytecode module size=%zu partitions=%d",
             adla_bin_size, num_partitions);

  api->AmlCompilerCoreDestroyCompiledArtifact(artifact);

  *compiled_result = result.release();
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation **transformations, LiteRtParamIndex *num_patterns) {
  if (compiler_plugin == nullptr || transformations == nullptr ||
      num_patterns == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // AML plugin currently does not register graph rewrite patterns.
  *transformations = nullptr;
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char *soc_model_name) {
  static constexpr LiteRtApiVersion kApiVersion{LITERT_API_VERSION_MAJOR,
                                                LITERT_API_VERSION_MINOR,
                                                LITERT_API_VERSION_PATCH};
  if (LiteRtCompareApiVersion(api_version, kApiVersion) > 0) {
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  }
  if (soc_model_name != nullptr && soc_model_name[0] != '\0' &&
      !FindSocModel(soc_model_name).has_value()) {
    LITERT_LOG(LITERT_ERROR, "Unsupported AML soc_model: %s", soc_model_name);
    return kLiteRtStatusErrorInvalidArgument;
  }
  (void)compiler_plugin;
  (void)env;
  (void)options;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetCompiledResultHandle(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    LiteRtJitExecutable* handle) {
  (void)compiled_result;
  (void)call_idx;
  if (handle == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *handle = nullptr;
  return kLiteRtStatusErrorUnsupported;
}

namespace {

static const LiteRtCompilerPluginInterface_V1 AmlogicCompilerPluginInterface = {
    .abi_header = {
            .struct_size = sizeof(LiteRtCompilerPluginInterface_V1),
            .major_version = 1,
            .minor_version = 0,
            .reserved = 0,
        },
    .get_compiler_plugin_version = LiteRtGetCompilerPluginVersion,
    .get_compiler_plugin_soc_manufacturer =
        LiteRtGetCompilerPluginSocManufacturer,
    .create_compiler_plugin = LiteRtCreateCompilerPlugin,
    .destroy_compiler_plugin = LiteRtDestroyCompilerPlugin,
    .get_compiler_plugin_supported_hardware =
        LiteRtGetCompilerPluginSupportedHardware,
    .get_num_compiler_plugin_supported_models =
        LiteRtGetNumCompilerPluginSupportedSocModels,
    .get_compiler_plugin_supported_soc_model =
        LiteRtGetCompilerPluginSupportedSocModel,
    .compiler_plugin_partition = LiteRtCompilerPluginPartition,
    .compiler_plugin_compile = LiteRtCompilerPluginCompile,
    .destroy_compiled_result = LiteRtDestroyCompiledResult,
    .get_compiled_result_byte_code = LiteRtGetCompiledResultByteCode,
    .get_compiled_result_num_byte_code = LiteRtCompiledResultNumByteCodeModules,
    .get_compiled_result_call_info = LiteRtGetCompiledResultCallInfo,
    .get_num_compiled_result_calls = LiteRtGetNumCompiledResultCalls,
    .register_all_transformations =
        LiteRtCompilerPluginRegisterAllTransformations,
    .get_compiler_plugin_sdk_version = LiteRtGetCompilerPluginSDKVersion,
    .get_compiled_result_handle = LiteRtGetCompiledResultHandle,
    .check_compiler_compatibility =
        LiteRtCompilerPluginCheckCompilerCompatibility,
};

}  // namespace

extern "C" LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginQueryInterface(
    LiteRtCompilerPluginInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (litert_runtime_version.major >= 1 &&
      interface_id == kLiteRtCompilerPluginInterfaceBasic) {
    *out_interface = &AmlogicCompilerPluginInterface;
    return kLiteRtStatusOk;
  }
  return kLiteRtStatusErrorUnsupported;
}
