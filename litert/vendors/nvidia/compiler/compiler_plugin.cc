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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper_with_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_c_types_printing.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/plugin/algo.h"
#include "litert/core/model/model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"
#include "litert/vendors/nvidia/memory_profile.h"
#include "NvInferVersion.h"

namespace {

constexpr char kPluginManufacturer[] = "NVIDIA";
constexpr char kPluginSocModel[] = "tensorrt-rtx";

enum class PartitionPolicy {
  kSafe,
  kGemma4,
  kCompute,
  kAll,
};

std::string PartitionName(int index) {
  return "tensorrt_partition_" + std::to_string(index);
}

std::string TensorSummary(const litert::compiler::Tensor& tensor) {
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
}

std::string OpSummary(const litert::compiler::Op& op) {
  std::string summary =
      std::string(litert::GetOpCodeStringView(op.Code())) + " inputs={";
  for (int i = 0; i < op.Inputs().size(); ++i) {
    if (i > 0) {
      summary += "; ";
    }
    summary += TensorSummary(op.Inputs()[i]);
  }
  summary += "} outputs={";
  for (int i = 0; i < op.Outputs().size(); ++i) {
    if (i > 0) {
      summary += "; ";
    }
    summary += TensorSummary(op.Outputs()[i]);
  }
  summary += "}";
  return summary;
}

bool LogPartitionOps() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_LOG_PARTITION_OPS");
  return value != nullptr && value[0] != '\0';
}

bool LogUnsupportedOps() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_LOG_UNSUPPORTED_OPS");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

PartitionPolicy GetPartitionPolicy() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_PARTITION_POLICY");
  if (value == nullptr || value[0] == '\0') {
    return PartitionPolicy::kSafe;
  }
  const std::string policy(value);
  if (policy == "all") {
    return PartitionPolicy::kAll;
  }
  if (policy == "gemma4") {
    return PartitionPolicy::kGemma4;
  }
  if (policy == "compute") {
    return PartitionPolicy::kCompute;
  }
  return PartitionPolicy::kSafe;
}

const char* PartitionPolicyName(PartitionPolicy policy) {
  switch (policy) {
    case PartitionPolicy::kAll:
      return "all";
    case PartitionPolicy::kCompute:
      return "compute";
    case PartitionPolicy::kGemma4:
      return "gemma4";
    case PartitionPolicy::kSafe:
      return "safe";
  }
}

bool IsComputeHeavyOp(const litert::compiler::Op& op) {
  switch (op.Code()) {
    case kLiteRtOpCodeTflBatchMatmul:
    case kLiteRtOpCodeTflConv2d:
    case kLiteRtOpCodeTflFullyConnected:
    case kLiteRtOpCodeTflSoftmax:
      return true;
    default:
      return false;
  }
}

bool IsSafeGemma4Op(const litert::compiler::Op& op) {
  switch (op.Code()) {
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflMul:
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflDiv:
    case kLiteRtOpCodeTflMaximum:
    case kLiteRtOpCodeTflRsqrt:
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflSum:
      return true;
    default:
      return false;
  }
}

bool IsValidatedGemma4TensorRtOp(const litert::compiler::Op& op) {
  // The full Gemma4 transformer stack. Delegating whole layers as few large
  // islands is required for output quality: every extra island boundary is a
  // CPU/TensorRT requantization seam, and past ~two dozen seams per token the
  // accumulated drift corrupts greedy decoding.
  switch (op.Code()) {
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflBatchMatmul:
    case kLiteRtOpCodeTflCast:
    case kLiteRtOpCodeTflConcatenation:
    case kLiteRtOpCodeTflCos:
    case kLiteRtOpCodeTflDequantize:
    case kLiteRtOpCodeTflDynamicUpdateSlice:
    case kLiteRtOpCodeTflFill:
    case kLiteRtOpCodeTflFullyConnected:
    case kLiteRtOpCodeTflGelu:
    case kLiteRtOpCodeTflGreaterEqual:
    case kLiteRtOpCodeTflLess:
    case kLiteRtOpCodeTflLogicalAnd:
    case kLiteRtOpCodeTflMaximum:
    case kLiteRtOpCodeTflMul:
    case kLiteRtOpCodeTflNotEqual:
    case kLiteRtOpCodeTflPack:
    case kLiteRtOpCodeTflQuantize:
    case kLiteRtOpCodeTflReduceMax:
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflRsqrt:
    case kLiteRtOpCodeTflSelectV2:
    case kLiteRtOpCodeShloComposite:
    case kLiteRtOpCodeTflSin:
    case kLiteRtOpCodeTflSlice:
    case kLiteRtOpCodeTflSoftmax:
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflSum:
    case kLiteRtOpCodeTflTanh:
    case kLiteRtOpCodeTflTranspose:
    case kLiteRtOpCodeTflUnpack:
      return true;
    default:
      return false;
  }
}

bool MatchesOpNameToken(const litert::compiler::Op& op,
                        const std::string& token) {
  if (token.empty()) {
    return false;
  }
  const std::string op_name(litert::GetOpCodeStringView(op.Code()));
  if (token == op_name) {
    return true;
  }
  const size_t dot = op_name.find('.');
  return dot != std::string::npos && token == op_name.substr(dot + 1);
}

bool EnvOpListContains(const char* env_name, const litert::compiler::Op& op) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  const std::string list(value);
  size_t begin = 0;
  while (begin < list.size()) {
    size_t end = list.find(',', begin);
    if (end == std::string::npos) {
      end = list.size();
    }
    if (MatchesOpNameToken(op, list.substr(begin, end - begin))) {
      return true;
    }
    begin = end + 1;
  }
  return false;
}

// True if `name` appears in the comma-separated list env var `env_name`.
bool EnvNameListContains(const char* env_name, const char* name) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0' || name == nullptr) {
    return false;
  }
  const std::string list(value);
  const std::string target(name);
  size_t begin = 0;
  while (begin < list.size()) {
    size_t end = list.find(',', begin);
    if (end == std::string::npos) {
      end = list.size();
    }
    if (list.compare(begin, end - begin, target) == 0) {
      return true;
    }
    begin = end + 1;
  }
  return false;
}

bool EnvEnableOpListIsSet() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_ENABLE_OPS");
  return value != nullptr && value[0] != '\0';
}

bool IsPolicyAllowedOp(const litert::compiler::Op& op, PartitionPolicy policy) {
  if (EnvEnableOpListIsSet()) {
    return EnvOpListContains("LITERT_NVIDIA_TENSORRT_ENABLE_OPS", op);
  }
  switch (policy) {
    case PartitionPolicy::kAll:
      return true;
    case PartitionPolicy::kCompute:
      return IsComputeHeavyOp(op);
    case PartitionPolicy::kGemma4:
      return IsValidatedGemma4TensorRtOp(op);
    case PartitionPolicy::kSafe:
      return IsSafeGemma4Op(op);
  }
}

bool IsEnvDisabledOp(const litert::compiler::Op& op) {
  return EnvOpListContains("LITERT_NVIDIA_TENSORRT_DISABLE_OPS", op);
}

size_t EnvSizeT(const char* name, size_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const uint64_t parsed = std::strtoul(value, &end, 10);
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

size_t EnvSizeTAllowZero(const char* name, size_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const uint64_t parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0') {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}

size_t MinPartitionOps(PartitionPolicy policy) {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_MIN_PARTITION_OPS", 8);
}

size_t MinPartitionOutputBytes(PartitionPolicy policy) {
  const size_t safe_default = 1ULL << 20;
  const size_t broad_default = 16ULL << 10;
  return EnvSizeT(
      "LITERT_NVIDIA_TENSORRT_MIN_PARTITION_OUTPUT_BYTES",
      policy == PartitionPolicy::kSafe ? safe_default : broad_default);
}

size_t MaxPartitions(PartitionPolicy policy) {
  const size_t default_value =
      policy == PartitionPolicy::kSafe ? 8 : std::numeric_limits<size_t>::max();
  return EnvSizeTAllowZero("LITERT_NVIDIA_TENSORRT_MAX_PARTITIONS",
                           default_value);
}

size_t MaxSelectedOpsPerSubgraph(PartitionPolicy policy) {
  return EnvSizeTAllowZero("LITERT_NVIDIA_TENSORRT_MAX_SELECTED_OPS",
                           /*default_value=*/0);
}

size_t ElementByteSize(litert::ElementType type) {
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
    case litert::ElementType::Int4:
      return 1;
    case litert::ElementType::Int64:
      return 8;
    default:
      return 0;
  }
}

size_t TensorByteSize(const litert::compiler::Tensor& tensor) {
  auto ranked_type = tensor.RankedTensorType();
  if (!ranked_type) {
    return 0;
  }
  auto num_elements = ranked_type->Layout().NumElements();
  if (!num_elements) {
    return 0;
  }
  if (ranked_type->ElementType() == litert::ElementType::Int4) {
    return (*num_elements + 1) / 2;
  }
  return *num_elements * ElementByteSize(ranked_type->ElementType());
}

size_t PartitionOutputBytes(const LiteRtCompilerContext* ctx,
                            const std::vector<LiteRtOp>& partition) {
  size_t bytes = 0;
  for (auto* raw_op : partition) {
    litert::compiler::Op op(ctx, raw_op);
    for (const auto& output : op.Outputs()) {
      bytes += TensorByteSize(output);
    }
  }
  return bytes;
}

bool ContainsComputeDenseOp(const LiteRtCompilerContext* ctx,
                            const std::vector<LiteRtOp>& partition) {
  for (auto* raw_op : partition) {
    switch (litert::compiler::Op(ctx, raw_op).Code()) {
      case kLiteRtOpCodeTflBatchMatmul:
      case kLiteRtOpCodeTflConv2d:
      case kLiteRtOpCodeTflFullyConnected:
        return true;
      default:
        break;
    }
  }
  return false;
}

bool ContainsReductionOp(const LiteRtCompilerContext* ctx,
                         const std::vector<LiteRtOp>& partition) {
  for (auto* raw_op : partition) {
    switch (litert::compiler::Op(ctx, raw_op).Code()) {
      case kLiteRtOpCodeTflSoftmax:
      case kLiteRtOpCodeTflSum:
      case kLiteRtOpCodeTflReduceMax:
        return true;
      default:
        break;
    }
  }
  return false;
}

bool ShouldKeepPartition(const LiteRtCompilerContext* ctx,
                         const std::vector<LiteRtOp>& partition,
                         PartitionPolicy policy, size_t output_bytes) {
  if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_FILTER_SMALL_PARTITIONS",
                  /*default_value=*/true)) {
    return true;
  }
  if (ContainsComputeDenseOp(ctx, partition)) {
    return true;
  }
  if (ContainsReductionOp(ctx, partition) &&
      output_bytes >= MinPartitionOutputBytes(policy)) {
    return true;
  }
  return partition.size() >= MinPartitionOps(policy) &&
         output_bytes >= MinPartitionOutputBytes(policy);
}

std::vector<LiteRtOpWithPartitionIndex> FilterSmallPartitions(
    const LiteRtCompilerContext* ctx,
    const std::vector<LiteRtOpWithPartitionIndex>& selected_ops,
    LiteRtSubgraph subgraph, PartitionPolicy policy) {
  auto islands = litert::internal::GroupPartitionsV2(selected_ops, subgraph);
  struct IslandInfo {
    size_t index;
    size_t ops;
    size_t output_bytes;
  };
  std::vector<IslandInfo> keep_candidates;
  std::unordered_set<LiteRtOp> kept_ops;
  size_t dropped_islands = 0;
  size_t dropped_ops = 0;
  for (size_t i = 0; i < islands.size(); ++i) {
    const auto& island = islands[i];
    const size_t output_bytes = PartitionOutputBytes(ctx, island);
    if (ShouldKeepPartition(ctx, island, policy, output_bytes)) {
      keep_candidates.push_back({i, island.size(), output_bytes});
    } else {
      ++dropped_islands;
      dropped_ops += island.size();
    }
  }

  const size_t max_partitions = MaxPartitions(policy);
  size_t capped_islands = 0;
  size_t capped_ops = 0;
  std::unordered_set<size_t> kept_island_indexes;
  if (max_partitions != 0 && keep_candidates.size() > max_partitions) {
    std::vector<IslandInfo> ranked = keep_candidates;
    std::sort(ranked.begin(), ranked.end(),
              [](const IslandInfo& lhs, const IslandInfo& rhs) {
                if (lhs.output_bytes != rhs.output_bytes) {
                  return lhs.output_bytes > rhs.output_bytes;
                }
                if (lhs.ops != rhs.ops) {
                  return lhs.ops > rhs.ops;
                }
                return lhs.index < rhs.index;
              });
    for (size_t i = 0; i < max_partitions; ++i) {
      kept_island_indexes.insert(ranked[i].index);
    }
    for (size_t i = max_partitions; i < ranked.size(); ++i) {
      ++capped_islands;
      capped_ops += ranked[i].ops;
    }
  } else {
    for (const auto& candidate : keep_candidates) {
      kept_island_indexes.insert(candidate.index);
    }
  }

  for (const auto& index : kept_island_indexes) {
    const auto& island = islands[index];
    kept_ops.insert(island.begin(), island.end());
  }

  std::vector<LiteRtOpWithPartitionIndex> filtered;
  filtered.reserve(kept_ops.size());
  for (const auto& selected_op : selected_ops) {
    if (kept_ops.find(selected_op.first) != kept_ops.end()) {
      filtered.push_back(selected_op);
    }
  }
  LITERT_LOG(LITERT_INFO,
             "NVIDIA TensorRT-RTX partition filter: candidate_islands=%zu "
             "candidate_ops=%zu kept_islands=%zu kept_ops=%zu "
             "dropped_islands=%zu dropped_ops=%zu capped_islands=%zu "
             "capped_ops=%zu min_ops=%zu min_output_bytes=%zu "
             "max_partitions=%zu",
             islands.size(), selected_ops.size(), kept_island_indexes.size(),
             filtered.size(), dropped_islands, dropped_ops, capped_islands,
             capped_ops, MinPartitionOps(policy),
             MinPartitionOutputBytes(policy), max_partitions);
  return filtered;
}

}  // namespace

struct LiteRtCompiledResultT {
  std::vector<std::vector<uint8_t>> bytecodes;
  std::vector<std::string> call_infos;
};

struct LiteRtCompilerPluginT {
  explicit LiteRtCompilerPluginT(const LiteRtCompilerContext* ctx) : ctx(ctx) {}
  const LiteRtCompilerContext* ctx = nullptr;
};

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return kPluginManufacturer;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = static_cast<LiteRtHwAccelerators>(
      kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorNpu);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (!compiler_plugin || !num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = 1;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (!compiler_plugin || !soc_model_name || soc_model_idx != 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = kPluginSocModel;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSDKVersion(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  if (!compiler_plugin || !sdk_version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  static const auto* const version = new std::string(
      "TensorRT-RTX via NvInfer headers " + std::to_string(NV_TENSORRT_MAJOR) +
      "." + std::to_string(NV_TENSORRT_MINOR) + "." +
      std::to_string(NV_TENSORRT_PATCH));
  *sdk_version = version->c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->bytecodes.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result || !byte_code || !byte_code_size ||
      byte_code_idx >= compiled_result->bytecodes.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& bytecode = compiled_result->bytecodes[byte_code_idx];
  *byte_code = bytecode.data();
  *byte_code_size = bytecode.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (!compiled_result || !call_info || !call_info_size || !byte_code_idx) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (call_idx >= compiled_result->call_infos.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  const auto& info = compiled_result->call_infos[call_idx];
  *call_info = info.data();
  *call_info_size = info.size();
  *byte_code_idx = call_idx;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->call_infos.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  if (compiler_context == nullptr || compiler_plugin == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtPropagateMinLoggerSeverityWithCompilerContext(compiler_context, env);
  *compiler_plugin = new LiteRtCompilerPluginT(compiler_context);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  if (!compiler_plugin || !subgraph || !selected_ops) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  litert::compiler::Subgraph graph(compiler_plugin->ctx, subgraph);
  // Leave named subgraphs on CPU (comma list). Each delegated signature's
  // engines embed their own weight copies, so skipping rarely-used signatures
  // (e.g. "verify", "prefill_1024") cuts device memory roughly per signature.
  {
    const std::string subgraph_name(subgraph->Name());
    if (!subgraph_name.empty() &&
        EnvNameListContains("LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS",
                            subgraph_name.c_str())) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX skipping subgraph `%s` "
                 "(LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS)",
                 subgraph_name.c_str());
      return kLiteRtStatusOk;
    }
  }
  // Debug bounds to delegate only subgraphs within an op-count range (e.g. to
  // isolate decode-vs-prefill behavior); 0 disables each bound.
  const size_t min_subgraph_ops =
      EnvSizeTAllowZero("LITERT_NVIDIA_TENSORRT_MIN_SUBGRAPH_OPS", 0);
  const size_t max_subgraph_ops =
      EnvSizeTAllowZero("LITERT_NVIDIA_TENSORRT_MAX_SUBGRAPH_OPS", 0);
  const size_t num_graph_ops = graph.Ops().size();
  if ((min_subgraph_ops != 0 && num_graph_ops < min_subgraph_ops) ||
      (max_subgraph_ops != 0 && num_graph_ops > max_subgraph_ops)) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX skipping subgraph with %zu ops "
               "(bounds min=%zu max=%zu)",
               num_graph_ops, min_subgraph_ops, max_subgraph_ops);
    return kLiteRtStatusOk;
  }
  const PartitionPolicy policy = GetPartitionPolicy();
  int total_ops = 0;
  int selected_op_count = 0;
  int policy_rejected_op_count = 0;
  int cap_rejected_op_count = 0;
  std::map<LiteRtOpCode, int> unsupported_counts;
  std::map<LiteRtOpCode, int> policy_rejected_counts;
  std::map<LiteRtOpCode, int> cap_rejected_counts;
  std::vector<LiteRtOpWithPartitionIndex> candidate_ops;
  const size_t max_selected_ops = MaxSelectedOpsPerSubgraph(policy);
  for (const auto& op : graph.Ops()) {
    ++total_ops;
    const bool supported = litert::nvidia::IsTensorRtOpSupported(op);
    const bool policy_allowed = IsPolicyAllowedOp(op, policy);
    const bool env_disabled = IsEnvDisabledOp(op);
    if (supported && policy_allowed && !env_disabled) {
      if (max_selected_ops == 0 || candidate_ops.size() < max_selected_ops) {
        candidate_ops.push_back({op.Get(), 0});
        ++selected_op_count;
      } else {
        ++cap_rejected_op_count;
        ++cap_rejected_counts[op.Code()];
      }
    } else if (!supported &&
               (policy == PartitionPolicy::kAll || policy_allowed)) {
      ++unsupported_counts[op.Code()];
      if (LogUnsupportedOps()) {
        const std::string op_summary = OpSummary(op);
        LITERT_LOG(LITERT_INFO, "NVIDIA TensorRT-RTX unsupported op detail: %s",
                   op_summary.c_str());
      }
    } else {
      ++policy_rejected_op_count;
      ++policy_rejected_counts[op.Code()];
    }
  }
  LITERT_LOG(LITERT_INFO,
             "NVIDIA TensorRT-RTX partition scan: soc=%s total_ops=%d "
             "selected_ops=%d unsupported_kinds=%zu policy=%s "
             "policy_rejected_ops=%d policy_rejected_kinds=%zu "
             "cap_rejected_ops=%d cap_rejected_kinds=%zu max_selected_ops=%zu",
             soc_model ? soc_model : "(null)", total_ops, selected_op_count,
             unsupported_counts.size(), PartitionPolicyName(policy),
             policy_rejected_op_count, policy_rejected_counts.size(),
             cap_rejected_op_count, cap_rejected_counts.size(),
             max_selected_ops);
  for (const auto& [code, count] : unsupported_counts) {
    LITERT_LOG(LITERT_INFO, "NVIDIA TensorRT-RTX unsupported op: %s count=%d",
               std::string(litert::GetOpCodeStringView(code)).c_str(), count);
  }
  for (const auto& [code, count] : policy_rejected_counts) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX policy skipped op: %s count=%d",
               std::string(litert::GetOpCodeStringView(code)).c_str(), count);
  }
  for (const auto& [code, count] : cap_rejected_counts) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX selected-op cap skipped op: %s count=%d",
               std::string(litert::GetOpCodeStringView(code)).c_str(), count);
  }
  auto filtered_ops = FilterSmallPartitions(compiler_plugin->ctx, candidate_ops,
                                            subgraph, policy);
  for (const auto& op : filtered_ops) {
    LITERT_RETURN_IF_ERROR(
        compiler_plugin->ctx->push_op(selected_ops, op.first, op.second));
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  if (!compiler_plugin || !partitions || !compiled_result) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  litert::compiler::Model model(compiler_plugin->ctx, partitions);
  auto result = std::make_unique<LiteRtCompiledResultT>();
  const auto num_partitions = model.NumSubgraphs();
  result->bytecodes.reserve(num_partitions);
  result->call_infos.reserve(num_partitions);
  size_t total_bytecode_bytes = 0;
  litert::nvidia::LogMemoryProfile("compiler", "compile_begin", soc_model);

  for (LiteRtParamIndex i = 0; i < num_partitions; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto subgraph, model.Subgraph(i));
    const auto function_name = PartitionName(i);
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX compiling partition %d/%d with %d ops",
               static_cast<int>(i + 1), static_cast<int>(num_partitions),
               static_cast<int>(subgraph.Ops().size()));
    if (LogPartitionOps()) {
      for (const auto& op : subgraph.Ops()) {
        const std::string op_summary = OpSummary(op);
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT-RTX partition %d op detail: %s",
                   static_cast<int>(i + 1), op_summary.c_str());
      }
    }
    litert::nvidia::LogMemoryProfile("compiler", "partition_build_begin",
                                     function_name.c_str());
    auto engine_or = litert::nvidia::BuildTensorRtEngine(subgraph);
    if (!engine_or) {
      std::string op_codes;
      for (const auto& op : subgraph.Ops()) {
        if (!op_codes.empty()) {
          op_codes += ", ";
        }
        op_codes += std::string(litert::GetOpCodeStringView(op.Code()));
      }
      LITERT_LOG(LITERT_ERROR,
                 "NVIDIA TensorRT-RTX failed to compile partition %d/%d "
                 "with %d ops: %s. Ops: [%s]",
                 static_cast<int>(i + 1), static_cast<int>(num_partitions),
                 static_cast<int>(subgraph.Ops().size()),
                 engine_or.Error().Message().c_str(), op_codes.c_str());
      for (const auto& op : subgraph.Ops()) {
        const std::string op_summary = OpSummary(op);
        LITERT_LOG(LITERT_ERROR, "NVIDIA TensorRT-RTX failed op detail: %s",
                   op_summary.c_str());
      }
      return engine_or.Error().Status();
    }
    auto engine = std::move(*engine_or);
    litert::nvidia::LogMemoryProfile("compiler", "partition_build_end",
                                     function_name.c_str());
    litert::nvidia::TensorRtLlmHead trtllm_head;
    const litert::nvidia::TensorRtLlmHead* trtllm_head_ptr = nullptr;
    if (engine.trtllm_head.has_value()) {
      trtllm_head.hidden_output_port = engine.trtllm_head->hidden_output_port;
      trtllm_head.logits_output_port = engine.trtllm_head->logits_output_port;
      trtllm_head.k = engine.trtllm_head->k;
      trtllm_head.n = engine.trtllm_head->n;
      trtllm_head.soft_cap = engine.trtllm_head->soft_cap;
      trtllm_head.weight_format = engine.trtllm_head->weight_format;
      trtllm_head.packed_weights = engine.trtllm_head->packed_weights.data();
      trtllm_head.packed_weights_size =
          engine.trtllm_head->packed_weights.size();
      trtllm_head.bf16_scales = engine.trtllm_head->bf16_scales.data();
      trtllm_head.bf16_scales_size = engine.trtllm_head->bf16_scales.size();
      trtllm_head_ptr = &trtllm_head;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto packed,
        litert::nvidia::PackTensorRtBytecode(
            function_name, engine.input_names, engine.output_names,
            engine.engine.data(), engine.engine.size(), trtllm_head_ptr));
    total_bytecode_bytes += packed.size();
    LITERT_LOG(
        LITERT_INFO,
        "NVIDIA TensorRT-RTX compiled partition %d/%d: "
        "engine_bytes=%zu bytecode_bytes=%zu cumulative_bytecode_bytes=%zu",
        static_cast<int>(i + 1), static_cast<int>(num_partitions),
        engine.engine.size(), packed.size(), total_bytecode_bytes);
    result->call_infos.push_back(function_name);
    result->bytecodes.push_back(std::move(packed));
    litert::nvidia::LogMemoryProfile("compiler", "partition_retained",
                                     function_name.c_str());
  }

  litert::nvidia::LogMemoryProfile("compiler", "compile_end", soc_model);
  *compiled_result = result.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  if (!transformations || !num_patterns) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *transformations = nullptr;
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  static constexpr LiteRtApiVersion kApiVersion{LITERT_API_VERSION_MAJOR,
                                                LITERT_API_VERSION_MINOR,
                                                LITERT_API_VERSION_PATCH};
  if (LiteRtCompareApiVersion(api_version, kApiVersion) > 0) {
    LITERT_LOG(LITERT_ERROR,
               "LiteRT caller API version is newer than NVIDIA compiler "
               "plugin API version");
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  }
  if (soc_model_name != nullptr &&
      std::string(soc_model_name) != kPluginSocModel) {
    LITERT_LOG(LITERT_WARNING,
               "NVIDIA TensorRT-RTX compiler accepts RTX CUDA targets; "
               "requested SoC model was %s",
               soc_model_name);
  }
  return kLiteRtStatusOk;
}
