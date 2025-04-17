
// Copyright 2024 Google LLC.
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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/options/litert_qualcomm_options.h"  // IWYU pragma: keep
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/options_helper.h"
#include "litert/vendors/qualcomm/compiler/qnn_compose_graph.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "third_party/qairt/latest/include/QNN/QnnLog.h"

using ::litert::qnn::QnnManager;
using LiteRtBufferId = uint32_t;
using LiteRtContextHandleIdx = uint32_t;
using WeightSharingMap =
    absl::flat_hash_map<LiteRtBufferId, LiteRtContextHandleIdx>;

//
// Configurations
//

namespace {

constexpr char kPluginManufacturer[] = "Qualcomm";
constexpr LiteRtParamIndex kDefaultPartitionIndex = 0;
constexpr LiteRtParamIndex kDefaultPartitionNum = 1;

static constexpr absl::string_view kEntryPointNameFmt = "qnn_partition_%d";

std::optional<::qnn::SocInfo> FindSocModel(absl::string_view soc_model_name) {
  std::optional<::qnn::SocInfo> soc_model;
  for (auto i = 0; i < ::qnn::kNumSocInfos; ++i) {
    if (soc_model_name == ::qnn::kSocInfos[i].soc_name) {
      soc_model = ::qnn::kSocInfos[i];
      break;
    }
  }
  return soc_model;
}


}  // namespace

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
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (!compiler_plugin || !num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = ::qnn::kNumSocInfos;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (!compiler_plugin || !soc_model_name) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (soc_model_idx < 0 || soc_model_idx >= ::qnn::kNumSocInfos) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = ::qnn::kSocInfos[soc_model_idx].soc_name;
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

struct LiteRtCompiledResultT {
  std::vector<std::vector<char>> context_bin;
  std::vector<std::string> graph_names;
  // byte_code_index[i] is the index of the byte code in context_bin that
  // corresponds to the i-th call.
  std::vector<size_t> byte_code_index;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result || !byte_code || !byte_code_size) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *byte_code = compiled_result->context_bin[byte_code_idx].data();
  *byte_code_size = compiled_result->context_bin[byte_code_idx].size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (!compiled_result || !call_info || !call_info_size) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (call_idx >= compiled_result->graph_names.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  *call_info = compiled_result->graph_names.at(call_idx).data();
  *call_info_size = compiled_result->graph_names.at(call_idx).size();
  *byte_code_idx = compiled_result->byte_code_index[call_idx];

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->graph_names.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  *num_byte_code = compiled_result->context_bin.size();
  return kLiteRtStatusOk;
}

//
// Plugin Definition
//

// Plugins can hold state.
class LiteRtCompilerPluginT {
 public:
  LiteRtCompilerPluginT(LiteRtEnvironmentOptions env, LiteRtOptions options)
      : options_({env, options}) {}

  litert::OptionsHelper& Options() { return options_; }

  QnnLog_Level_t LogLevel() {
    auto qnn_opts = options_.FindOptions<::litert::qualcomm::QualcommOptions>();
    if (!qnn_opts) {
      return QNN_LOG_LEVEL_INFO;
    }
    auto log_level_opt = qnn_opts->GetLogLevel();
    switch (log_level_opt) {
      case kLiteRtQualcommLogOff:
        return QNN_LOG_LEVEL_ERROR;
      case kLiteRtQualcommLogLevelError:
        return QNN_LOG_LEVEL_ERROR;
      case kLiteRtQualcommLogLevelWarn:
        return QNN_LOG_LEVEL_WARN;
      case kLiteRtQualcommLogLevelInfo:
        return QNN_LOG_LEVEL_INFO;
      case kLiteRtQualcommLogLevelVerbose:
        return QNN_LOG_LEVEL_VERBOSE;
      case kLiteRtQualcommLogLevelDebug:
        return QNN_LOG_LEVEL_DEBUG;
    }
  }

  bool IsWeightSharingSupported(::qnn::DspArch dsp_arch) {
#ifdef __ANDROID__
    return false;
#else
    auto qnn_opts = options_.FindOptions<::litert::qualcomm::QualcommOptions>();
    if (!qnn_opts) {
      return true;
    }
    return qnn_opts->GetEnableWeightSharing() &&
           dsp_arch >= ::qnn::DspArch::V73;
#endif
  }

 private:
  litert::OptionsHelper options_;
};

LiteRtStatus LiteRtCompilerPluginSetFlags(LiteRtCompilerPlugin compiler_plugin,
                                          LiteRtParamIndex num_flags,
                                          const char** keys,
                                          const char** values) {
  // This is in the process of deprecation.
  LITERT_LOG(LITERT_WARNING, "LiteRtCompilerPluginSetFlags is deprecated");
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin,
                                        LiteRtEnvironmentOptions env,
                                        LiteRtOptions options) {
  if (options == nullptr || env == nullptr) {
    LITERT_LOG(LITERT_WARNING,
               "QNN compiler plugin created with null options, these will be "
               "defaulted.");
  }
  auto* plugin = new LiteRtCompilerPluginT(env, options);
  *compiler_plugin = plugin;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::Subgraph graph(subgraph);

  auto backend_configs = QnnManager::DefaultBackendConfigs();
  auto qnn_manager =
      QnnManager::Create(backend_configs, std::nullopt,
                         soc_model ? FindSocModel(soc_model) : std::nullopt,
                         compiler_plugin->LogLevel());
  if (!qnn_manager) {
    LITERT_LOG(LITERT_ERROR, "%s", qnn_manager.Error().Message().data());
    return qnn_manager.Error().Status();
  }
  LITERT_LOG(LITERT_INFO, "%s", "QNN manager created");

  for (const auto& op : graph.Ops()) {
    // default constructed, won't add tensor to QNN
    ::qnn::TensorPool tensor_pool;
    std::vector<::qnn::TensorWrapperRef> input_tensors;
    for (const auto& input : op.Inputs()) {
      ::qnn::TensorWrapper* res{nullptr};
      LITERT_RETURN_IF_ERROR(
          litert::qnn::ConvertTensor(input, tensor_pool, res));
      input_tensors.emplace_back(*res);
    }

    std::vector<::qnn::TensorWrapperRef> output_tensors;
    for (const auto& output : op.Outputs()) {
      ::qnn::TensorWrapper* res{nullptr};
      LITERT_RETURN_IF_ERROR(
          litert::qnn::ConvertTensor(output, tensor_pool, res));
      output_tensors.emplace_back(*res);
    }

    std::vector<::qnn::OpWrapper> op_wrappers;
    LITERT_RETURN_IF_ERROR(litert::qnn::ConvertOp(
        op, tensor_pool, input_tensors, output_tensors, op_wrappers));
    tensor_pool.ForEach([](::qnn::TensorWrapper& tensor_wrapper) {
      // TODO(chunhsue): Use compile interface to get useQInt16AsQUint16.
      constexpr bool useQInt16AsQUint16 = true;
      if constexpr (useQInt16AsQUint16) {
        tensor_wrapper.ConvertQint16ToQuint16();
      }
    });
    // Empty op_wrappers means the op is not supported by QNN.
    if (op_wrappers.empty()) {
      continue;
    }
    if (std::all_of(
            op_wrappers.begin(), op_wrappers.end(),
            [&qnn_manager](::qnn::OpWrapper& op_wrapper) -> bool {
              return kLiteRtStatusOk ==
                     (*qnn_manager)->ValidateOp(op_wrapper.GetOpConfig());
            })) {
      LITERT_RETURN_IF_ERROR(
          // Use default partition index if vendor doesn't support multiple
          // partitions.
          LiteRtPushOp(selected_ops, op.Get(), kDefaultPartitionIndex));
    }
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  auto model = litert::Model::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();

  LITERT_LOG(LITERT_INFO,
             "Starting QNN Compilation for %d subgraphs, soc_model=%s",
             num_partitions, soc_model);

  auto opt_soc_model = soc_model ? FindSocModel(soc_model) : std::nullopt;
  if (opt_soc_model) {
    LITERT_LOG(LITERT_INFO, "Compiling QNN SoC model: %s", soc_model);
  } else if (soc_model) {
    LITERT_LOG(LITERT_ERROR, "Unexpected SoC model: %s", soc_model);
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto result = std::make_unique<LiteRtCompiledResultT>();
  // Prepare one context binary per partition, since each partition is a
  // separate subgraph that maps to a single Dispatch Op in the compiled the
  // model.
  result->context_bin.resize(num_partitions);
  result->byte_code_index.resize(num_partitions);

  // Initialize SDK and load qnn shared libraries.
  LITERT_LOG(LITERT_INFO, "%s", "Creating QNN manager");
  auto backend_configs = QnnManager::DefaultBackendConfigs();
  auto qnn_manager =
      QnnManager::Create(backend_configs, /*shared_library_dir=*/std::nullopt,
                         opt_soc_model, compiler_plugin->LogLevel());
  if (!qnn_manager) {
    LITERT_LOG(LITERT_ERROR, "%s", qnn_manager.Error().Message().c_str());
    return qnn_manager.Error().Status();
  }
  LITERT_LOG(LITERT_INFO, "%s", "QNN manager created");

  // Map of LiteRt buffer id to context handle index.
  // This map memerizes the last context handle index of a weight was registered
  // in.
  WeightSharingMap weight_sharing_map;
  LiteRtContextHandleIdx next_context_handle_idx = 0;

  std::vector<QnnManager::ContextHandle> context_handles;

  // Compile each partition (subgraph) individually.
  for (int partition_idx = 0; partition_idx < num_partitions; ++partition_idx) {
    LiteRtContextHandleIdx context_handle_idx = next_context_handle_idx;
    uint64_t largest_weight_size = 0;
    // Check all weights in this subgraph, see if any of them were previously
    // seen and added to existing qnn context, use the largest weight size to
    // determine which context to use.
    for (const auto& op : model.Subgraph(partition_idx)->Ops()) {
      for (const auto& input : op.Inputs()) {
        if (input.IsConstant()) {
          auto buffer_id = input.Weights().BufferId();
          auto it = weight_sharing_map.find(buffer_id);
          if (it != weight_sharing_map.end()) {
            if (input.Weights().Bytes().size() >= largest_weight_size) {
              context_handle_idx = it->second;
              largest_weight_size = input.Weights().Bytes().size();
            }
          }
        }
      }
    }
    // If we didn't find a existing context handle for this subgraph, create a
    // new one.
    if (context_handle_idx == next_context_handle_idx) {
      // Initialize context.
      LITERT_LOG(LITERT_INFO, "%s", "Creating context handle");
      // We enable weight sharing by default, this could lead to issue when
      // support legacy SoC.
      // TODO(jiunkaiy): use option to control weight sharing.
      auto context_configs = QnnManager::WeightSharingContextConfigs();
      // Disable weight sharing if we have only one partition or SoC doesn't
      // support weight sharing.
      if (num_partitions == kDefaultPartitionNum ||
          !compiler_plugin->IsWeightSharingSupported(
              opt_soc_model.value().dsp_arch)) {
        context_configs = QnnManager::DefaultContextConfigs();
      }
      auto context_handle =
          (*qnn_manager)->CreateContextHandle(context_configs);
      if (!context_handle) {
        LITERT_LOG(LITERT_ERROR, "%s",
                   context_handle.Error().Message().c_str());
        return context_handle.Error().Status();
      }
      context_handles.push_back(std::move(context_handle.Value()));
      LITERT_LOG(LITERT_INFO, "%s", "Context handle created");
      ++next_context_handle_idx;
    }
    // Set context handle index for all weight buffers in this subgraph.
    for (const auto& op : model.Subgraph(partition_idx)->Ops()) {
      for (const auto& input : op.Inputs()) {
        if (input.IsConstant()) {
          auto buffer_id = input.Weights().BufferId();
          weight_sharing_map[buffer_id] = context_handle_idx;
        }
      }
    }

    // Compose graphs.
    LITERT_LOG(LITERT_INFO, "%s", "Composing graph");
    std::string& entry_point_name = result->graph_names.emplace_back();
    result->byte_code_index[partition_idx] = context_handle_idx;
    entry_point_name = absl::StrFormat(kEntryPointNameFmt, partition_idx);
    LITERT_LOG(LITERT_INFO, "Entry point name: %s", entry_point_name.c_str());
    LiteRtSubgraph partition = model.Subgraph(partition_idx)->Get();
    LITERT_RETURN_IF_ERROR(litert::qnn::ComposeGraph(
        **qnn_manager, context_handles[context_handle_idx].get(), partition,
        entry_point_name));
    LITERT_LOG(LITERT_INFO, "%s", "Graph composed");
  }

  // Generate context binary.
  result->context_bin.resize(next_context_handle_idx);
  for (int i = 0; i < next_context_handle_idx; ++i) {
    LITERT_LOG(LITERT_INFO, "%s", "Generating context binary");
    LITERT_RETURN_IF_ERROR((*qnn_manager)
                               ->GenerateContextBinary(context_handles[i].get(),
                                                       result->context_bin[i]));
    LITERT_LOG(LITERT_INFO, "Context binary %d generated", i);
  }
  *compiled_result = result.release();

  return kLiteRtStatusOk;
}
