// Copyright (C) 2023 Amlogic, Inc. All rights reserved.
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

#include <algorithm>
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
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/options/litert_aml_options.h"  // IWYU pragma: keep
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/aml/compiler/aml_dispatch_info.h"
#include "litert/vendors/aml/compiler/node_binary_info.h"
#include "compiler_core_context.h"

using LiteRtBufferId = uint32_t;
using LiteRtContextHandleIdx = uint32_t;
using WeightSharingMap =
    absl::flat_hash_map<LiteRtBufferId, LiteRtContextHandleIdx>;

namespace
{

  constexpr char kPluginManufacturer[] = "Aml";
  constexpr LiteRtParamIndex kDefaultPartitionIndex = 0;
  constexpr LiteRtParamIndex kDefaultPartitionNum = 1;
  // Multi-subgraph compile limit for testing (0 = compile all partitions).
  constexpr int kTestSubgraphCompileLimit = 0;

  static constexpr absl::string_view kEntryPointNameFmt = "aml_partition_0";

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

  /**
   * @brief Whether to embed ADLA bytes into context_bin.
   * @note ARM online: false — dispatch loads .adla from disk.
   *       Host/x86 offline apply: true — embed once into dispatch_info[0]
   *       for multi-subgraph packs (partitions 1..N-1 keep empty adla_bin).
   */
#if defined(__aarch64__) || defined(__arm__) || defined(__ARM_ARCH)
  constexpr bool kEmbedAdlaInDispatch = false;
#else
  constexpr bool kEmbedAdlaInDispatch = true;
#endif

  // LiteRT TFL opcodes/element types are defined to match TFLite numeric
  // values (see litert_op_code.h / litert_model_types.h comments). AML IR
  // stores TFLite semantics; keep the conversion explicit at the adapter.
  int ToTfLiteBuiltinOp(LiteRtOpCode op_code)
  {
    return static_cast<int>(op_code);
  }

  int ToTfLiteType(litert::ElementType element_type)
  {
    return static_cast<int>(element_type);
  }

  LiteRtStatus ToLiteRtStatus(AmlCompilerStatus status)
  {
    switch (status)
    {
    case kAmlCompilerStatusOk:
      return kLiteRtStatusOk;
    case kAmlCompilerStatusErrorInvalidArgument:
      return kLiteRtStatusErrorInvalidArgument;
    case kAmlCompilerStatusErrorRuntimeFailure:
    default:
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }

  // Ordinary composite name so PartitionModel takes the Google 2.2 inline path.
  // odml.npu_call is never inlined; strip that tag and do not PushOp.
  constexpr char kAmlInlineCompositeName[] = "odml.regular_composite";

  void RetagCompositeToForceInline(LiteRtOp op)
  {
    auto &&opts = litert::internal::TakeTflOptions2(*op);
    auto *composite_opts = opts.AsStableHLOCompositeOptions();
    if (composite_opts == nullptr)
    {
      return;
    }
    if (composite_opts->name != litert::CompositeOptions::kNpuCall)
    {
      return;
    }
    LITERT_LOG(LITERT_INFO,
               "AML retag composite '%s' as %s to force inline (decomp subgraph=%d)",
               composite_opts->name.c_str(), kAmlInlineCompositeName,
               composite_opts->decomposition_subgraph_index);
    composite_opts->name = kAmlInlineCompositeName;
  }

  // Reasons AML Partition will not select an op for NPU.
  // Add new enumerators here and handle them in GetAmlPartitionRejectReason().
  enum class AmlPartitionRejectReason {
    kNone = 0,
    kUnsupportedOpcode,      // opcode denylist, leave on CPU
    kNotInAllowlist,         // neither denylist nor ADLA allowlist
    kDynamicTensorShape,     // unranked, or any dim < 0
    kNonConstConvWeights,    // CONV_2D filter/bias must be constant
    // kScalarReshape,       // e.g. 1x1 f32 -> rank-0 scalar
  };

  const char *AmlPartitionRejectReasonName(AmlPartitionRejectReason reason) {
    switch (reason) {
    case AmlPartitionRejectReason::kNone:
      return "none";
    case AmlPartitionRejectReason::kUnsupportedOpcode:
      return "unsupported_opcode";
    case AmlPartitionRejectReason::kNotInAllowlist:
      return "opcode_not_in_allowlist";
    case AmlPartitionRejectReason::kDynamicTensorShape:
      return "dynamic_tensor_shape";
    case AmlPartitionRejectReason::kNonConstConvWeights:
      return "non_const_conv_weights";
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
  constexpr LiteRtOpCode kAmlRejectedOpcodes[] = {
      kLiteRtOpCodeTflConv3d,
      kLiteRtOpCodeTflConv3dTranspose,
  };

  // ADLA TFLite allowlist from Amlogic_NPU_Support_Operator_List CN v0.1
  // (2026-07-15): union of ADLA2/ADLA3 Hardware + Software op lists.
  constexpr LiteRtOpCode kAmlSupportedOpcodes[] = {
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
      kLiteRtOpCodeTflCustom,
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

  bool IsAmlRejectedOpcode(LiteRtOpCode code) {
    return OpcodeInList(code, kAmlRejectedOpcodes,
                        sizeof(kAmlRejectedOpcodes) /
                            sizeof(kAmlRejectedOpcodes[0]));
  }

  bool IsAmlSupportedOpcode(LiteRtOpCode code) {
    return OpcodeInList(code, kAmlSupportedOpcodes,
                        sizeof(kAmlSupportedOpcodes) /
                            sizeof(kAmlSupportedOpcodes[0]));
  }

  // True if the tensor is unranked or has a dynamic dimension (typically -1).
  // Rank-0 scalars have an empty dim list and are not treated as dynamic.
  bool TensorHasDynamicDims(const litert::Tensor &tensor) {
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

  // TFLite CONV_2D inputs: [0]=activation, [1]=filter, [2]=bias (optional).
  // AML only supports constant filter/bias (folded weights), not runtime tensors.
  bool Conv2dHasNonConstWeightOrBias(const litert::Op &op) {
    if (!op.Is(kLiteRtOpCodeTflConv2d)) {
      return false;
    }
    const auto &inputs = op.Inputs();
    if (inputs.size() < 2) {
      return true;
    }
    if (!inputs[1].IsConstant()) {
      return true;
    }
    if (inputs.size() >= 3 && !inputs[2].IsConstant()) {
      return true;
    }
    return false;
  }

  // Extension window for NPU-unsupported ops. First match wins; do not PushOp.
  // Opcode support: denylist first, then ADLA allowlist. Unknown opcodes log
  // and stay on CPU.
  AmlPartitionRejectReason GetAmlPartitionRejectReason(const litert::Op &op) {
    const LiteRtOpCode opcode = op.Code();
    if (IsAmlRejectedOpcode(opcode)) {
      return AmlPartitionRejectReason::kUnsupportedOpcode;
    }
    if (!IsAmlSupportedOpcode(opcode)) {
      LITERT_LOG(LITERT_WARNING,
                 "AML op_code=%d neither in reject list nor ADLA allowlist; "
                 "unsupported (leave on CPU)",
                 static_cast<int>(opcode));
      return AmlPartitionRejectReason::kNotInAllowlist;
    }
    if (AnyTensorHasDynamicDims(op.Inputs()) ||
        AnyTensorHasDynamicDims(op.Outputs())) {
      return AmlPartitionRejectReason::kDynamicTensorShape;
    }
    if (Conv2dHasNonConstWeightOrBias(op)) {
      return AmlPartitionRejectReason::kNonConstConvWeights;
    }
    // Future checks:
    // - scalar reshape (1x1 -> rank 0)
    // - DEPTHWISE_CONV_2D / TRANSPOSE_CONV non-const weights (same pattern)
    return AmlPartitionRejectReason::kNone;
  }

  std::optional<const char *> FindSocModel(absl::string_view soc_model_name)
  {
    for (auto i = 0; i < kNumPluginSocModels; ++i)
    {
      if (soc_model_name == kPluginSocModels[i].first)
      {
        return kPluginSocModels[i].second;
      }
    }
    return std::nullopt;
  }

  LiteRtStatus CheckIr(AmlCompilerStatus status, const char *what)
  {
    if (status == kAmlCompilerStatusOk)
    {
      return kLiteRtStatusOk;
    }
    LITERT_LOG(LITERT_ERROR, "%s failed status=%d", what,
               static_cast<int>(status));
    return ToLiteRtStatus(status);
  }

  using AmlTensorMap = absl::flat_hash_map<LiteRtTensor, AmlTensor>;

  LiteRtStatus FillAmlTensor(AmlTensor t, const litert::Tensor &tensor,
                             bool copy_const_weights, int aml_index)
  {
    LiteRtStatus st =
        CheckIr(AmlTensorSetName(t, std::string(tensor.Name()).c_str()),
                "AmlTensorSetName");
    if (st != kLiteRtStatusOk)
    {
      return st;
    }
    st = CheckIr(AmlTensorSetDtype(t, ToTfLiteType(tensor.ElementType())),
                 "AmlTensorSetDtype");
    if (st != kLiteRtStatusOk)
    {
      return st;
    }
    // Do not use LiteRT TensorIndex(): CloneTo copies decomp-local indices
    // (0/1/2) into the inlined main graph, so multiple tensors collide.
    st = CheckIr(AmlTensorSetIndex(t, aml_index), "AmlTensorSetIndex");
    if (st != kLiteRtStatusOk)
    {
      return st;
    }
    auto ranked_type_or = tensor.RankedTensorType();
    if (ranked_type_or)
    {
      auto dims = ranked_type_or->Layout().Dimensions();
      std::vector<int> dim_vec(dims.begin(), dims.end());
      if (!dim_vec.empty())
      {
        st = CheckIr(AmlTensorSetDims(t, dim_vec.data(),
                                      static_cast<uint32_t>(dim_vec.size())),
                     "AmlTensorSetDims");
        if (st != kLiteRtStatusOk)
        {
          return st;
        }
      }
    }
    std::vector<int> zp;
    std::vector<float> scale;
    if (tensor.QTypeId() == kLiteRtQuantizationPerTensor)
    {
      LiteRtQuantizationPerTensor per_tensor = tensor.PerTensorQuantization();
      zp.push_back(static_cast<int>(per_tensor.zero_point));
      scale.push_back(per_tensor.scale);
    }
    else if (tensor.QTypeId() == kLiteRtQuantizationPerChannel)
    {
      LiteRtQuantizationPerChannel per_channel =
          tensor.PerChannelQuantization();
      for (int num = 0; num < per_channel.num_channels; num++)
      {
        zp.push_back(static_cast<int>(per_channel.zero_points[num]));
        scale.push_back(per_channel.scales[num]);
      }
    }
    if (!zp.empty() || !scale.empty())
    {
      st = CheckIr(AmlTensorSetQuant(t, zp.empty() ? nullptr : zp.data(),
                                     static_cast<uint32_t>(zp.size()),
                                     scale.empty() ? nullptr : scale.data(),
                                     static_cast<uint32_t>(scale.size())),
                   "AmlTensorSetQuant");
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
    }
    if (copy_const_weights && !tensor.DefiningOp().has_value() &&
        tensor.HasWeights())
    {
      auto bytes = tensor.Weights().Bytes();
      st = CheckIr(AmlTensorSetConstData(t, bytes.data(), bytes.size()),
                   "AmlTensorSetConstData");
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus EnsureTensor(AmlGraph graph, AmlTensorMap *tensors,
                            const litert::Tensor &tensor,
                            bool copy_const_weights, AmlTensor *out)
  {
    LiteRtTensor key = tensor.Get();
    auto it = tensors->find(key);
    if (it != tensors->end())
    {
      if (copy_const_weights && !tensor.DefiningOp().has_value() &&
          tensor.HasWeights())
      {
        auto bytes = tensor.Weights().Bytes();
        LiteRtStatus st =
            CheckIr(AmlTensorSetConstData(it->second, bytes.data(), bytes.size()),
                    "AmlTensorSetConstData");
        if (st != kLiteRtStatusOk)
        {
          return st;
        }
      }
      *out = it->second;
      return kLiteRtStatusOk;
    }
    AmlTensor t = nullptr;
    LiteRtStatus st = CheckIr(AmlTensorCreate(graph, &t), "AmlTensorCreate");
    if (st != kLiteRtStatusOk)
    {
      return st;
    }
    const int aml_index = static_cast<int>(tensors->size());
    st = FillAmlTensor(t, tensor, copy_const_weights, aml_index);
    if (st != kLiteRtStatusOk)
    {
      return st;
    }
    (*tensors)[key] = t;
    *out = t;
    return kLiteRtStatusOk;
  }

  LiteRtStatus FillAmlGraphIo(const litert::Subgraph &subgraph, AmlGraph graph,
                              AmlTensorMap *tensors)
  {
    for (const auto &tensor : subgraph.Inputs())
    {
      AmlTensor t = nullptr;
      LiteRtStatus st =
          EnsureTensor(graph, tensors, tensor, /*copy_const_weights=*/false, &t);
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
      st = CheckIr(AmlGraphAddInput(graph, t), "AmlGraphAddInput");
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
    }
    for (const auto &tensor : subgraph.Outputs())
    {
      AmlTensor t = nullptr;
      LiteRtStatus st =
          EnsureTensor(graph, tensors, tensor, /*copy_const_weights=*/false, &t);
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
      st = CheckIr(AmlGraphAddOutput(graph, t), "AmlGraphAddOutput");
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus ConvertOpToAmlNode(const litert::Op &op, int node_index,
                                  AmlGraph graph, AmlTensorMap *tensors)
  {
    AmlNode node = nullptr;
    LiteRtStatus st =
        CheckIr(AmlNodeCreate(graph, ToTfLiteBuiltinOp(op.Code()), node_index,
                              &node),
                "AmlNodeCreate");
    if (st != kLiteRtStatusOk)
    {
      return st;
    }

    LiteRtStatus parse_status =
        ParseOpParams(node, ToTfLiteBuiltinOp(op.Code()), op.Get());
    if (parse_status != kLiteRtStatusOk)
    {
      LITERT_LOG(LITERT_ERROR,
                 "ParseOpParams failed for op_code=%d node_index=%d status=%d",
                 static_cast<int>(op.Code()), node_index,
                 static_cast<int>(parse_status));
      return parse_status;
    }

    for (const auto &tensor : op.Inputs())
    {
      AmlTensor t = nullptr;
      st = EnsureTensor(graph, tensors, tensor, /*copy_const_weights=*/true, &t);
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
      st = CheckIr(AmlNodeAddInput(node, t), "AmlNodeAddInput");
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
    }
    for (const auto &tensor : op.Outputs())
    {
      AmlTensor t = nullptr;
      st = EnsureTensor(graph, tensors, tensor, /*copy_const_weights=*/false, &t);
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
      st = CheckIr(AmlNodeAddOutput(node, t), "AmlNodeAddOutput");
      if (st != kLiteRtStatusOk)
      {
        return st;
      }
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus FillAmlGraphNodes(const litert::Subgraph &subgraph, AmlGraph graph,
                                 AmlTensorMap *tensors)
  {
    int node_index = 0;
    for (const auto &op : subgraph.Ops())
    {
      LiteRtStatus status =
          ConvertOpToAmlNode(op, node_index++, graph, tensors);
      if (status != kLiteRtStatusOk)
      {
        return status;
      }
    }
    return kLiteRtStatusOk;
  }

  /**
   * @brief Build runtime dispatch metadata for one compiled partition.
   * @param subgraph LiteRT partition subgraph.
   * @param partition_idx Subgraph / partition index (@c subgraph_idx).
   * @param model_name Model artifact base name (shared-load key with path).
   * @param model_path Output directory for ADLA assets.
   * @param entry_point_name Graph entry point name (e.g. aml_partition_N).
   * @return AML_Dispatch_Info (adla_bin filled later for partition 0 only).
   * @note Does not embed ADLA here; packing after compile assigns adla_bin.
   */
  AML_Dispatch_Info MakeDispatchInfo(const litert::Subgraph &subgraph,
                                     int partition_idx,
                                     const std::string &model_name,
                                     const std::string &model_path,
                                     const std::string &entry_point_name)
  {
    AML_Dispatch_Info dispatch_info;
    dispatch_info.model_path = model_path;
    dispatch_info.model_names = model_name;
    dispatch_info.graph_names = entry_point_name;
    dispatch_info.subgraph_idx = partition_idx;
    for (const auto &tensor : subgraph.Inputs())
    {
      dispatch_info.graph_inputs.push_back(std::string(tensor.Name()));
    }
    for (const auto &tensor : subgraph.Outputs())
    {
      dispatch_info.graph_outputs.push_back(std::string(tensor.Name()));
    }
    return dispatch_info;
  }

}

// Plugins can hold state.
//
// Configurations
//
struct LiteRtCompiledResultT
{
  std::vector<std::vector<char>> context_bin;
  std::vector<std::string> graph_names;
  // byte_code_index[i] is the index of the byte code in context_bin that
  // corresponds to the i-th call.
  std::vector<size_t> byte_code_index;
};
//
// Plugin Definition
//
class LiteRtCompilerPluginT {
 public:
  LiteRtCompilerPluginT(LiteRtEnvironmentOptions env_options,
                        LiteRtOptions litert_options) {
    (void)env_options;
    LITERT_LOG(LITERT_DEBUG, "[AML] LiteRtCompilerPluginT init");

    if (!litert_options) {
      return;
    }

    LiteRtOpaqueOptions opaque_options = nullptr;
    if (LiteRtGetOpaqueOptions(litert_options, &opaque_options) !=
        kLiteRtStatusOk) {
      return;
    }

    if (LiteRtAmlOptionsGet(opaque_options, &aml_options_) != kLiteRtStatusOk) {
      aml_options_ = nullptr;
    }
  }

  std::string GetModelNameOrDefault() const {
    if (aml_options_ == nullptr) {
      return "test";
    }
    const char* model_name = nullptr;
    if (LiteRtAmlOptionsGetModelName(aml_options_, &model_name) !=
            kLiteRtStatusOk ||
        model_name == nullptr || model_name[0] == '\0') {
      return "test";
    }
    return model_name;
  }

  std::string GetModelPathOrEmpty() const {
    if (aml_options_ == nullptr) {
      return "";
    }
    const char* model_path = nullptr;
    if (LiteRtAmlOptionsGetModelPath(aml_options_, &model_path) !=
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

 private:
  LiteRtAmlOptions aml_options_ = nullptr;
  absl::flat_hash_set<LiteRtSubgraph> partitioned_subgraphs_;
  int partition_subgraph_count_ = 0;
};
extern "C" LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion *api_version)
{
  if (api_version == nullptr)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

extern "C" const char *LiteRtGetCompilerPluginSocManufacturer()
{
  return kPluginManufacturer;
}

extern "C" LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators *supported_hardware)
{
  if (!compiler_plugin || !supported_hardware)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex *num_supported_soc_models)
{
  LITERT_LOG(LITERT_DEBUG,
             "[AML] LiteRtGetNumCompilerPluginSupportedSocModels enter");
  if (!compiler_plugin || !num_supported_soc_models)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = kNumPluginSocModels;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char **soc_model_name)
{
  LITERT_LOG(LITERT_DEBUG,
             "[AML] LiteRtGetCompilerPluginSupportedSocModel enter");
  if (!compiler_plugin || !soc_model_name)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (soc_model_idx < 0 || soc_model_idx >= kNumPluginSocModels)
  {
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
  static const char kAmlSdkVersion[] = "aml-adla";
  *sdk_version = kAmlSdkVersion;
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

extern "C" LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void **byte_code, size_t *byte_code_size)
{
  LITERT_LOG(LITERT_DEBUG, "[AML] LiteRtGetCompiledResultByteCode enter");
  if (!compiled_result || !byte_code || !byte_code_size)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *byte_code = compiled_result->context_bin[byte_code_idx].data();
  *byte_code_size = compiled_result->context_bin[byte_code_idx].size();
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void **call_info, size_t *call_info_size,
    LiteRtParamIndex *byte_code_idx)
{
  if (!compiled_result || !call_info || !call_info_size)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  else if (call_idx >= compiled_result->graph_names.size())
  {
    return kLiteRtStatusErrorIndexOOB;
  }

  *call_info = compiled_result->graph_names.at(call_idx).data();
  *call_info_size = compiled_result->graph_names.at(call_idx).size();
  *byte_code_idx = compiled_result->byte_code_index[call_idx];

  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex *num_calls)
{
  if (!compiled_result || !num_calls)
  {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->graph_names.size();
  return kLiteRtStatusOk;
}

extern "C" void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result)
{
  delete compiled_result;
}

extern "C" LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex *num_byte_code)
{
  // *num_byte_code = 1;
  *num_byte_code = compiled_result->context_bin.size();
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtCreateCompilerPlugin(
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  if (options == nullptr || env == nullptr) {
    LITERT_LOG(LITERT_WARNING,
               "AML compiler plugin created with null options, these will be "
               "defaulted.");
  }
  *compiler_plugin = new LiteRtCompilerPluginT(env, options);
  return kLiteRtStatusOk;
}

extern "C" void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin)
{
  delete compiler_plugin;
}

extern "C" LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                                      const char *soc_model,
                                                      LiteRtSubgraph subgraph,
                                                      LiteRtOpList selected_ops)
{
  if (compiler_plugin != nullptr) {
    compiler_plugin->RecordPartitionSubgraph(subgraph);
  }

  ::litert::Subgraph graph(subgraph);

  LITERT_LOG(LITERT_INFO, "Starting AML Partitioning");

  for (const auto &op : graph.Ops())
  {
    if (op.Is(kLiteRtOpCodeShloComposite)) {
      RetagCompositeToForceInline(op.Get());
      continue;
    }
    const AmlPartitionRejectReason reject = GetAmlPartitionRejectReason(op);
    if (reject != AmlPartitionRejectReason::kNone) {
      LITERT_LOG(LITERT_INFO,
                 "AML skip op_code=%d reason=%s (leave on CPU)",
                 static_cast<int>(op.Code()),
                 AmlPartitionRejectReasonName(reject));
      continue;
    }
    LiteRtPushOp(selected_ops, op.Get(), kDefaultPartitionIndex);
  }

  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char *soc_model,
    LiteRtModel partitions, LiteRtCompiledResult *compiled_result)
{
  auto *plugin = compiler_plugin;
  const std::string model_name = plugin ? plugin->GetModelNameOrDefault() : "test";
  const std::string model_path = plugin ? plugin->GetModelPathOrEmpty() : "";
  LITERT_LOG(LITERT_DEBUG, "model_name: %s", model_name.c_str());
  LITERT_LOG(LITERT_DEBUG, "model_path: %s", model_path.c_str());

  auto model = litert::ExtendedModel::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();
  const int compile_subgraphs =
      (kTestSubgraphCompileLimit > 0)
          ? std::min(static_cast<int>(num_partitions), kTestSubgraphCompileLimit)
          : static_cast<int>(num_partitions);
  if (kTestSubgraphCompileLimit > 0 &&
      compile_subgraphs < static_cast<int>(num_partitions))
  {
    LITERT_LOG(LITERT_DEBUG,
               "partition_subgraphs=%d, compile_subgraphs=%zu, compile_limit=%d",
               plugin ? plugin->GetPartitionSubgraphCount() : 0, num_partitions,
               compile_subgraphs);
  }
  else
  {
    LITERT_LOG(LITERT_DEBUG,
               "partition_subgraphs=%d, compile_subgraphs=%zu",
               plugin ? plugin->GetPartitionSubgraphCount() : 0, num_partitions);
  }

  auto result = std::make_unique<LiteRtCompiledResultT>();
  result->context_bin.resize(num_partitions);
  result->byte_code_index.resize(num_partitions);

  // Online: soc_model is often unset → keep empty so FinalizeAndCompile /
  // ApplyCompileTargetConfig queries the device via aml_adla_get_chip_info().
  // Do NOT default to kPluginSocModels[0] ("c3").
  std::string compile_target;
  if (soc_model != nullptr && soc_model[0] != '\0')
  {
    auto resolved_target = FindSocModel(soc_model);
    if (!resolved_target.has_value())
    {
      LITERT_LOG(LITERT_ERROR, "Unsupported AML soc_model: %s", soc_model);
      return kLiteRtStatusErrorInvalidArgument;
    }
    compile_target = *resolved_target;
  }
  LITERT_LOG(LITERT_INFO, "AML compile target (soc_model): %s",
             compile_target.empty() ? "(empty, device query)"
                                    : compile_target.c_str());

  std::vector<AML_Dispatch_Info> dispatch_infos;
  dispatch_infos.reserve(num_partitions);

  AmlContext ctx = AmlContextCreate();
  if (ctx == nullptr)
  {
    return kLiteRtStatusErrorRuntimeFailure;
  }

  for (int partition_idx = 0; partition_idx < num_partitions; ++partition_idx)
  {
    auto subgraph_or = model.Subgraph(partition_idx); // 得到 Expected<Subgraph>
    auto &subgraph = *subgraph_or;
    LITERT_LOG(LITERT_DEBUG, "==============================");
    LITERT_LOG(LITERT_DEBUG, "Subgraph index: %d", partition_idx);
    std::string &entry_point_name = result->graph_names.emplace_back();
    entry_point_name = absl::StrFormat("aml_partition_%d", partition_idx);

    result->byte_code_index[partition_idx] = partition_idx;

    AmlGraph graph = nullptr;
    LiteRtStatus ir_status = CheckIr(AmlGraphCreate(ctx, &graph), "AmlGraphCreate");
    if (ir_status == kLiteRtStatusOk)
    {
      ir_status = CheckIr(AmlGraphSetTarget(graph, compile_target.c_str()),
                          "AmlGraphSetTarget");
    }
    if (ir_status == kLiteRtStatusOk)
    {
      ir_status = CheckIr(AmlGraphSetName(graph, entry_point_name.c_str()),
                          "AmlGraphSetName");
    }
    AmlTensorMap tensors;
    if (ir_status == kLiteRtStatusOk)
    {
      ir_status = FillAmlGraphIo(subgraph, graph, &tensors);
    }
    if (ir_status == kLiteRtStatusOk)
    {
      ir_status = FillAmlGraphNodes(subgraph, graph, &tensors);
    }
    if (ir_status != kLiteRtStatusOk)
    {
      AmlContextDestroy(ctx);
      return ir_status;
    }

    dispatch_infos.push_back(MakeDispatchInfo(subgraph, partition_idx, model_name,
                                              model_path, entry_point_name));
  }

  AmlCompilerCoreOptions options = {};
  AmlCompilerCoreOptionsInitDefaults(&options);
  options.compile_target =
      compile_target.empty() ? nullptr : compile_target.c_str();
  options.compile_subgraphs = static_cast<size_t>(compile_subgraphs);
  options.embed_adla_in_result = kEmbedAdlaInDispatch ? 1 : 0;

  AmlCompilerCoreCompiledArtifact artifact = nullptr;
  const AmlCompilerStatus compile_status = AmlContextCompile(
      ctx, model_name.c_str(), model_path.c_str(), &options, &artifact);
  AmlContextDestroy(ctx);
  if (compile_status != kAmlCompilerStatusOk || artifact == nullptr)
  {
    LITERT_LOG(LITERT_ERROR, "AmlContextCompile failed status=%d",
               static_cast<int>(compile_status));
    AmlCompilerCoreDestroyCompiledArtifact(artifact);
    return ToLiteRtStatus(compile_status != kAmlCompilerStatusOk
                              ? compile_status
                              : kAmlCompilerStatusErrorRuntimeFailure);
  }

  const void *adla_bin_ptr = nullptr;
  size_t adla_bin_size = 0;
  const char *adla_path_cstr = nullptr;
  const char *fp_cstr = nullptr;
  int cache_hit = 0;
  AmlCompilerCoreGetCompiledArtifactAdlaBinary(artifact, &adla_bin_ptr,
                                               &adla_bin_size);
  AmlCompilerCoreGetCompiledArtifactAdlaPath(artifact, &adla_path_cstr);
  AmlCompilerCoreGetCompiledArtifactFingerprint(artifact, &fp_cstr);
  AmlCompilerCoreGetCompiledArtifactCacheHit(artifact, &cache_hit);
  LITERT_LOG(LITERT_INFO,
             "compiler core done path=%s fp=%s cache_hit=%d embed_size=%zu",
             adla_path_cstr ? adla_path_cstr : "",
             fp_cstr ? fp_cstr : "", cache_hit, adla_bin_size);

  std::vector<uint8_t> adla_bin;
  if (adla_bin_ptr != nullptr && adla_bin_size > 0)
  {
    const auto *bytes = static_cast<const uint8_t *>(adla_bin_ptr);
    adla_bin.assign(bytes, bytes + adla_bin_size);
  }

  /**
   * Offline multi-subgraph packing:
   *   - One compiled ADLA for all partitions.
   *   - Embed adla_bin only into dispatch_info[0] when kEmbedAdlaInDispatch.
   *   - Partitions 1..N-1 keep metadata (subgraph_idx / IO) with empty adla_bin.
   * Runtime: load once from [0] (or model_path), Acquire by key, BindSubgraph.
   */
  for (int partition_idx = 0; partition_idx < num_partitions; ++partition_idx)
  {
    AML_Dispatch_Info &dispatch_info = dispatch_infos[partition_idx];
    if (kEmbedAdlaInDispatch && partition_idx == 0 && !adla_bin.empty())
    {
      dispatch_info.adla_bin = adla_bin;
      dispatch_info.adla_bin_size = adla_bin_size;
    }
    else
    {
      dispatch_info.adla_bin.clear();
      dispatch_info.adla_bin_size = 0;
    }

    const std::string serialized_dispatch_info =
        SerializeDispatchInfo(dispatch_info);
    result->context_bin[partition_idx].assign(
        reinterpret_cast<const char *>(serialized_dispatch_info.data()),
        reinterpret_cast<const char *>(serialized_dispatch_info.data() +
                                       serialized_dispatch_info.size()));
  }
  if (kEmbedAdlaInDispatch && num_partitions > 1)
  {
    LITERT_LOG(LITERT_INFO,
               "offline multi-subgraph: embedded adla_bin only in "
               "dispatch_info[0] (size=%zu), partitions=%d",
               adla_bin_size, num_partitions);
  }

  AmlCompilerCoreDestroyCompiledArtifact(artifact);

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
  return kLiteRtStatusOk;
}
