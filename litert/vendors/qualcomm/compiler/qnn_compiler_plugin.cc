
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
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "QnnCommon.h"  // from @qairt
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper_with_compiler_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/internal/litert_context_wrapper.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_options_wrapper.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/compiler/qnn_compose_graph.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

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

bool IsWeightSharingSupported(::qnn::DspArch dsp_arch) {
#if defined(__x86_64__) || defined(_M_X64)
  return dsp_arch >= ::qnn::DspArch::V73;
#else
  return false;
#endif
}

LiteRtStatus MoveSchematic(absl::string_view graph_name,
                           absl::string_view dest_dir_str) {
  if (dest_dir_str.empty()) return kLiteRtStatusOk;
  std::error_code ec;
  std::filesystem::path dest_dir = std::string(dest_dir_str);
  std::filesystem::path schematic_file_name(
      absl::StrCat(graph_name, "_schematic.bin"));
  std::filesystem::path src_path =
      std::filesystem::current_path(ec) / schematic_file_name;
  if (ec) {
    LITERT_LOG(LITERT_ERROR, "Failed to get current CWD for schematic move: %s",
               ec.message().c_str());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (!std::filesystem::exists(src_path, ec) || ec) {
    LITERT_LOG(LITERT_ERROR, "QNN schematic file does not exist at %s: %s",
               src_path.c_str(), ec ? ec.message().c_str() : "Not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  std::filesystem::create_directories(dest_dir, ec);
  if (ec) {
    LITERT_LOG(LITERT_ERROR, "Failed to create directory %s: %s",
               dest_dir.c_str(), ec.message().c_str());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  std::filesystem::path dest_path = dest_dir / schematic_file_name;
  if (std::filesystem::exists(dest_path, ec) && !ec) {
    if (std::filesystem::equivalent(src_path, dest_path, ec)) {
      // Source and destination are the same file, no need to copy.
      return kLiteRtStatusOk;
    }
    if (ec) {
      // Ignore error from equivalent if it's just because file doesn't exist,
      // but here we already checked exists(dest_path), so it shouldn't fail
      // unless there is a permissions issue.
      LITERT_LOG(LITERT_ERROR, "Failed to check if paths are equivalent: %s",
                 ec.message().c_str());
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }
  std::filesystem::copy_file(src_path, dest_path,
                             std::filesystem::copy_options::overwrite_existing,
                             ec);
  if (ec) {
    LITERT_LOG(LITERT_ERROR, "Failed to copy schematic file from %s to %s: %s",
               src_path.c_str(), dest_path.c_str(), ec.message().c_str());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  std::filesystem::remove(src_path, ec);
  if (ec) {
    LITERT_LOG(LITERT_ERROR, "Failed to remove source schematic file %s: %s",
               src_path.c_str(), ec.message().c_str());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

// Compile-time custom-op packages always target the CPU backend; this is
// the value passed to QNN's RegisterOpPackage for compilation.
constexpr char kCustomOpPackageCompileTarget[] = "CPU";

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
  // Hold the context handles for Just-In-Time if enabled.
  std::vector<QnnManager::ContextHandle> context_handles;
  // Hold the QnnJitGraph for each subgraph for Just-In-Time.
  std::vector<std::unique_ptr<litert::qnn::QnnJitGraph>> jit_graphs;
};

LiteRtStatus LiteRtGetCompiledResultHandle(LiteRtCompiledResult compiled_result,
                                           LiteRtParamIndex call_idx,
                                           LiteRtJitExecutable* handle) {
  if (!compiled_result || !handle) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (compiled_result->jit_graphs.empty()) {
    *handle = nullptr;
    return kLiteRtStatusOk;
  }
  if (call_idx >= compiled_result->jit_graphs.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *handle = reinterpret_cast<LiteRtJitExecutable>(
      compiled_result->jit_graphs[call_idx].get());
  return kLiteRtStatusOk;
}

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
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = !compiled_result->jit_graphs.empty()
                       ? compiled_result->jit_graphs.size()
                       : compiled_result->context_bin.size();
  return kLiteRtStatusOk;
}

//
// Plugin Definition
//

// Plugins can hold state.
class LiteRtCompilerPluginT {
 public:
  LiteRtCompilerPluginT(const LiteRtCompilerContext* ctx,
                        LiteRtEnvironmentOptions env_options,
                        LiteRtOptions litert_options)
      : ctx_(ctx) {
    if (litert_options) {
      opts_ = litert::internal::OptionsWrapper(
          litert::internal::ContextWrapper(ctx), litert_options,
          litert::OwnHandle::kNo);
      if (opts_) {
        auto opq = opts_->GetOpaqueOptions();
        if (opq) {
          auto target_opq_status =
              opq->FindOpaqueOptions(LrtQualcommOptionsGetIdentifier());
          if (target_opq_status) {
            const char* options_data =
                static_cast<const char*>(target_opq_status.Value());
            LrtQualcommOptions options_handle = nullptr;
            if (LrtCreateQualcommOptionsFromToml(
                    options_data, &options_handle) == kLiteRtStatusOk) {
              qualcomm_options_ =
                  litert::qualcomm::QualcommOptions(options_handle);
              InitQnnOptions(qnn_options_, qualcomm_options_.Value());
            }
          }
        }
      }
    }

    if (env_options && ctx) {
      LiteRtAny compiler_plugin_lib_dir_any;
      auto status = ctx->get_environment_options_value(
          env_options, kLiteRtEnvOptionTagCompilerPluginLibraryDir,
          &compiler_plugin_lib_dir_any);
      if (status == kLiteRtStatusOk && compiler_plugin_lib_dir_any.str_value) {
        shared_library_dir_ =
            std::string(compiler_plugin_lib_dir_any.str_value);
      }
    }
  }

  const ::qnn::Options& Options() const { return qnn_options_; }

  LiteRtStatus initQnnManager(std::unique_ptr<QnnManager> qnn_manager) {
    if (const auto& custom_op_package = qnn_options_.GetCustomOpPackage();
        !custom_op_package.name.empty()) {
      LITERT_RETURN_IF_ERROR(qnn_manager->RegisterOpPackage(
          custom_op_package.compile_package_path,
          custom_op_package.interface_provider, kCustomOpPackageCompileTarget));
    }
    qnn_manager_ = std::move(qnn_manager);
    return kLiteRtStatusOk;
  }

  QnnManager* QNN() { return qnn_manager_.get(); }

  const LiteRtCompilerContext* ctx() const { return ctx_; }

  const std::optional<std::string>& shared_library_dir() const {
    return shared_library_dir_;
  }

 private:
  const LiteRtCompilerContext* ctx_;
  litert::Expected<litert::internal::OptionsWrapper> opts_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null options");
  litert::Expected<litert::qualcomm::QualcommOptions> qualcomm_options_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null Qualcomm options");
  ::qnn::Options qnn_options_{};
  QnnManager::Ptr qnn_manager_ = nullptr;
  std::optional<std::string> shared_library_dir_;
};

LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  if (compiler_context == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Compiler context is null");
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (options == nullptr || env == nullptr) {
    LITERT_LOG(LITERT_WARNING,
               "QNN compiler plugin created with null options, these will be "
               "defaulted.");
  }
  // Propagate the min logger severity from the environment.
  if (env != nullptr) {
    LiteRtPropagateMinLoggerSeverityWithCompilerContext(compiler_context, env);
  }

  auto* plugin = new LiteRtCompilerPluginT(compiler_context, env, options);
  *compiler_plugin = plugin;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtGetCompilerPluginSDKVersion(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  if (!compiler_plugin || !sdk_version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  QnnManager* qnn_manager = compiler_plugin->QNN();
  if (!qnn_manager) {
    std::optional<::qnn::SocInfo> soc_info = std::nullopt;
#if defined(__x86_64__) || defined(_M_X64)
    soc_info = qnn::FindSocModel("SM8750");
#endif
    auto qnn_manager_or =
        QnnManager::Create(compiler_plugin->Options(), std::nullopt, soc_info);
    if (!qnn_manager_or) {
      LITERT_LOG(LITERT_ERROR, "Failed to create QNN manager: %s",
                 qnn_manager_or.Error().Message().data());
      return qnn_manager_or.Error().Status();
    }
    LITERT_RETURN_IF_ERROR(
        compiler_plugin->initQnnManager(std::move(*qnn_manager_or)));
    qnn_manager = compiler_plugin->QNN();
  }

  const char* build_id = nullptr;
  if (qnn_manager->Api() == nullptr) {
    LITERT_LOG(LITERT_ERROR, "QNN API not resolved");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (qnn_manager->Api()->backendGetBuildId(&build_id) != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get QNN backend build ID");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  *sdk_version = build_id;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::compiler::Subgraph graph(compiler_plugin->ctx(), subgraph);
  QnnManager* qnn_manager = compiler_plugin->QNN();
  auto opt_soc_model = soc_model ? qnn::FindSocModel(soc_model) : std::nullopt;
  bool soc_model_mismatch = false;
  if (qnn_manager && opt_soc_model.has_value()) {
    soc_model_mismatch =
        (qnn_manager->GetSocInfo().soc_model != opt_soc_model->soc_model);
  }
  if (!qnn_manager || soc_model_mismatch) {
    if (soc_model_mismatch) {
      LITERT_LOG(LITERT_INFO,
                 "Recreating QNN manager due to SoC mismatch: current %s, "
                 "target %s",
                 qnn_manager->GetSocInfo().soc_name, opt_soc_model->soc_name);
    }
    auto qnn_manager_or = QnnManager::Create(
        compiler_plugin->Options(), compiler_plugin->shared_library_dir(),
        opt_soc_model);
    if (!qnn_manager_or) {
      LITERT_LOG(LITERT_ERROR, "%s", qnn_manager_or.Error().Message().data());
      return qnn_manager_or.Error().Status();
    }
    LITERT_LOG(LITERT_INFO, "%s", "QNN manager created");
    LITERT_RETURN_IF_ERROR(
        compiler_plugin->initQnnManager(std::move(*qnn_manager_or)));
    qnn_manager = compiler_plugin->QNN();
  }

  const auto ops = graph.Ops();
  for (size_t op_index = 0; op_index < ops.size(); ++op_index) {
    const auto& op = ops[op_index];
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
        compiler_plugin->Options(), op, tensor_pool, input_tensors,
        output_tensors, op_wrappers, op_index, qnn_manager->GetSdkVersion()));

    // Empty op_wrappers means the op is not supported by QNN.
    if (op_wrappers.empty()) {
      continue;
    }

    // Validate all OPs by QNN.
    if (std::all_of(op_wrappers.begin(), op_wrappers.end(),
                    [&qnn_manager](::qnn::OpWrapper& op_wrapper) -> bool {
                      return kLiteRtStatusOk ==
                             qnn_manager->ValidateOp(op_wrapper);
                    })) {
      LITERT_RETURN_IF_ERROR(
          // Use default partition index if vendor doesn't support multiple
          // partitions.
          compiler_plugin->ctx()->push_op(selected_ops, op.Get(),
                                          kDefaultPartitionIndex));
    }
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  litert::compiler::Model model(compiler_plugin->ctx(), partitions);
  auto num_partitions = model.NumSubgraphs();

  LITERT_LOG(LITERT_INFO,
             "Starting QNN Compilation for %d subgraphs, soc_model=%s",
             num_partitions, soc_model);

  auto opt_soc_model = soc_model ? qnn::FindSocModel(soc_model) : std::nullopt;
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
  QnnManager* qnn_manager = compiler_plugin->QNN();
  auto options = compiler_plugin->Options();
  if (!options.GetSchematicDir().empty()) {
    LITERT_LOG(LITERT_INFO,
               "Schematic directory is set. Enabling optrace profiling. "
               "(Original profiling level: %d)",
               static_cast<int>(options.GetProfiling()));
    options.SetProfiling(::qnn::Profiling::kOptrace);
  }
  if (!options.GetSaverOutputDir().empty()) {
    LITERT_LOG(
        LITERT_WARNING,
        "Overriding graph IO tensor mem type to Raw because Saver is enabled.");
    options.SetGraphIOTensorMemType(::qnn::GraphIOTensorMemType::kRaw);
  }
  const bool ir_backend_override =
      !options.GetDlcDir().empty() &&
      options.GetBackendType() != ::qnn::BackendType::kIrBackend;
  if (ir_backend_override) {
    LITERT_LOG(LITERT_WARNING,
               "Overriding backend type to IR Backend because DLC dir is set.");
    options.SetBackendType(::qnn::BackendType::kIrBackend);
  }

  if (options.GetBackendType() == ::qnn::BackendType::kIrBackend) {
    std::string dlc_dir(options.GetDlcDir());
    if (!dlc_dir.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(dlc_dir, ec);
      if (ec) {
        LITERT_LOG(LITERT_ERROR, "Failed to create DLC directory %s: %s",
                   dlc_dir.c_str(), ec.message().c_str());
        return kLiteRtStatusErrorRuntimeFailure;
      }
    }
  }

  bool soc_model_mismatch = false;
  if (qnn_manager && opt_soc_model.has_value()) {
    soc_model_mismatch =
        (qnn_manager->GetSocInfo().soc_model != opt_soc_model->soc_model);
  }

  if (!qnn_manager || ir_backend_override || soc_model_mismatch) {
    if (soc_model_mismatch) {
      LITERT_LOG(LITERT_INFO,
                 "Recreating QNN manager due to SoC mismatch: current %s, "
                 "target %s",
                 qnn_manager->GetSocInfo().soc_name, opt_soc_model->soc_name);
    }
    // Initialize SDK and load qnn shared libraries.
    LITERT_LOG(LITERT_INFO, "%s", "Creating QNN manager");
    auto qnn_manager_or = QnnManager::Create(
        options, compiler_plugin->shared_library_dir(), opt_soc_model);
    if (!qnn_manager_or) {
      LITERT_LOG(LITERT_ERROR, "%s", qnn_manager_or.Error().Message().data());
      return qnn_manager_or.Error().Status();
    }
    LITERT_LOG(LITERT_INFO, "%s", "QNN manager created");
    LITERT_RETURN_IF_ERROR(
        compiler_plugin->initQnnManager(std::move(*qnn_manager_or)));
    qnn_manager = compiler_plugin->QNN();
  }

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
    LITERT_ASSIGN_OR_RETURN(auto subgraph, model.Subgraph(partition_idx));
    for (const auto& op : subgraph.Ops()) {
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
      auto context_configs = QnnManager::DefaultContextConfigs();
      if (options.GetEnableWeightSharing()) {
        switch (options.GetBackendType()) {
          case ::qnn::BackendType::kHtpBackend: {
            // Only enable weight sharing if we have multiple partitions and
            // the current SoC support weight sharing feature.
            bool enable_weight_sharing =
                num_partitions != kDefaultPartitionNum &&
                IsWeightSharingSupported(opt_soc_model.value().dsp_arch);
            if (enable_weight_sharing) {
              context_configs = QnnManager::WeightSharingContextConfigs();
              LITERT_LOG(LITERT_INFO, "Enable weight sharing feature");
            } else {
              LITERT_LOG(LITERT_WARNING,
                         "Disable weight sharing feature. Only support with "
                         "multiple partitions and dsp_arch >= v73");
            }
            break;
          }
          default: {
            LITERT_LOG(LITERT_ERROR,
                       "Weight sharing is only supported in HTP backend.");
            return kLiteRtStatusErrorInvalidArgument;
          }
        }
      } else if (options.GetBackendType() == ::qnn::BackendType::kGpuBackend) {
        if (options.GetGpuPerformanceMode() !=
            ::qnn::GpuPerformanceMode::kDefault) {
          context_configs = QnnManager::GpuPerformanceContextConfigs(
              options.GetGpuPerformanceMode());
          LITERT_LOG(LITERT_INFO, "Enable GPU performance mode: %d",
                     static_cast<int>(options.GetGpuPerformanceMode()));
        }
      }
      auto context_handle = qnn_manager->CreateContextHandle(
          context_configs, options.GetProfiling());
      if (!context_handle) {
        LITERT_LOG(LITERT_ERROR, "%s", context_handle.Error().Message().data());
        return context_handle.Error().Status();
      }
      context_handles.push_back(std::move(context_handle.Value()));
      LITERT_LOG(LITERT_INFO, "%s", "Context handle created");
      ++next_context_handle_idx;
    }
    // Set context handle index for all weight buffers in this subgraph.
    LITERT_ASSIGN_OR_RETURN(auto partition, model.Subgraph(partition_idx));
    for (const auto& op : partition.Ops()) {
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

    std::vector<::qnn::TensorWrapper> inputs;
    std::vector<::qnn::TensorWrapper> outputs;

    LITERT_RETURN_IF_ERROR(litert::qnn::ComposeGraph(
        compiler_plugin->ctx(), *qnn_manager,
        context_handles[context_handle_idx].Get(),
        context_handles[context_handle_idx].get_profile_handle(),
        partition.Get(), entry_point_name, options, &inputs, &outputs));
    LITERT_LOG(LITERT_INFO, "%s", "Graph composed");

    if (!options.GetSchematicDir().empty()) {
      LITERT_RETURN_IF_ERROR(
          MoveSchematic(entry_point_name, options.GetSchematicDir()));
    }

    if (options.GetEnableJustInTime()) {
      auto jit_graph = std::make_unique<litert::qnn::QnnJitGraph>();
      jit_graph->context_handle = context_handles[context_handle_idx].Get();
      jit_graph->graph_name = entry_point_name;
      jit_graph->inputs = std::move(inputs);
      jit_graph->outputs = std::move(outputs);
      result->jit_graphs.push_back(std::move(jit_graph));
    }
  }

  if (!options.GetSaverOutputDir().empty()) {
    LITERT_LOG(LITERT_WARNING,
               "Since Saver is enabled, functional context binaries are "
               "excluded from the compiled TFLite.");
    result->context_bin.resize(next_context_handle_idx);
    *compiled_result = result.release();
    return kLiteRtStatusOk;
  }

  if (options.GetBackendType() == ::qnn::BackendType::kIrBackend) {
    LITERT_LOG(LITERT_WARNING,
               "Since IR backend is enabled, functional context binaries are "
               "excluded from the compiled TFLite.");
    result->context_bin.resize(next_context_handle_idx);
    *compiled_result = result.release();
    return kLiteRtStatusOk;
  } else if (options.GetEnableJustInTime()) {
    LITERT_LOG(LITERT_INFO,
               "Just-In-Time enabled. Skipping context binary generation.");
    result->context_handles = std::move(context_handles);
  } else {
    // Generate context binary.
    result->context_bin.resize(next_context_handle_idx);
    for (int i = 0; i < next_context_handle_idx; ++i) {
      LITERT_LOG(LITERT_INFO, "%s", "Generating context binary");
      LITERT_RETURN_IF_ERROR(qnn_manager->GenerateContextBinary(
          context_handles[i].Get(), result->context_bin[i]));
      LITERT_LOG(LITERT_INFO, "Context binary %d generated", i);
    }
  }
  *compiled_result = result.release();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  // TODO(jiunkaiy): Check if the QAIRT SDK version meets the minimum required
  // version.

  // Check LiteRt API version for backward compatibility.
  static constexpr LiteRtApiVersion kApiVersion{LITERT_API_VERSION_MAJOR,
                                                LITERT_API_VERSION_MINOR,
                                                LITERT_API_VERSION_PATCH};
  if (LiteRtCompareApiVersion(api_version, kApiVersion) > 0) {
    LITERT_LOG(
        LITERT_ERROR,
        "Incompatible compiler version. Found LiteRT API version %d.%d.%d, "
        "but version <= %d.%d.%d is required.",
        api_version.major, api_version.minor, api_version.patch,
        kApiVersion.major, kApiVersion.minor, kApiVersion.patch);
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  } else if (LiteRtCompareApiVersion(api_version, kApiVersion) < 0) {
    LITERT_LOG(LITERT_WARNING,
               "LiteRT API version (%d.%d.%d) is older than the "
               "compiler plugin version (%d.%d.%d). An update is recommended.",
               api_version.major, api_version.minor, api_version.patch,
               kApiVersion.major, kApiVersion.minor, kApiVersion.patch);
  }

  // Check if the SoC model is supported.
  if (!soc_model_name) {
    LITERT_LOG(LITERT_WARNING, "SoC model name is not specified.");
  } else if (!::qnn::FindSocModel(soc_model_name).has_value()) {
    LITERT_LOG(LITERT_ERROR, "Unsupported SoC model: %s", soc_model_name);
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  }

  return kLiteRtStatusOk;
}

namespace {

static LiteRtCompilerPluginInterface_V0_1 QnnCompilerPluginInterface = {
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
    .get_compiler_plugin_sdk_version = LiteRtGetCompilerPluginSDKVersion,
    .compiler_plugin_partition = LiteRtCompilerPluginPartition,
    .compiler_plugin_compile = LiteRtCompilerPluginCompile,
    .destroy_compiled_result = LiteRtDestroyCompiledResult,
    .get_compiled_result_byte_code = LiteRtGetCompiledResultByteCode,
    .get_compiled_result_handle = LiteRtGetCompiledResultHandle,
    .get_compiled_result_num_byte_code = LiteRtCompiledResultNumByteCodeModules,
    .get_compiled_result_call_info = LiteRtGetCompiledResultCallInfo,
    .get_num_compiled_result_calls = LiteRtGetNumCompiledResultCalls,
    .register_all_transformations =
        LiteRtCompilerPluginRegisterAllTransformations,
    .check_compiler_compatibility =
        LiteRtCompilerPluginCheckCompilerCompatibility,
};

}  // namespace

extern "C" LiteRtStatus LiteRtCompilerPluginQueryInterface(
    LiteRtCompilerPluginInterfaceId interface_id,
    LiteRtApiVersion requested_version, void** out_interface) {
  if (requested_version.major == 0 && requested_version.minor == 1) {
    if (interface_id == kLiteRtCompilerPluginInterfaceBasic) {
      *out_interface = &QnnCompilerPluginInterface;
      return kLiteRtStatusOk;
    }
  }
  return kLiteRtStatusErrorUnsupported;
}
