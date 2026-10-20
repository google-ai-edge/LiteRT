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

#include "litert/compiler/plugin/compiler_plugin.h"

#include <stdlib.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/plugin/algo.h"
#include "litert/compiler/plugin/litert_compiler_options.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dynamic_loading.h"
#include "litert/core/environment.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/model/subgraph_refs.h"
#include "litert/core/options.h"
#include "litert/core/util/perfetto_profiling.h"
#include "litert/core/version.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"

namespace litert::internal {

//
// CompiledResult
//

Expected<BufferRef<uint8_t>> CompiledResult::ByteCode(
    LiteRtParamIndex byte_code_idx) const {
  if (!LITERT_ABI_HAS_API(parent_v1_, /*req_major=*/1,
                          get_compiled_result_byte_code)) {
    return litert::Error(kLiteRtStatusErrorUnsupported);
  }
  const void* data = nullptr;
  size_t size = 0;
  LITERT_RETURN_IF_ERROR(parent_v1_->get_compiled_result_byte_code(
      compiled_result_handle_, byte_code_idx, &data, &size));
  return BufferRef(data, size);
}

Expected<LiteRtParamIndex> CompiledResult::NumByteCodeModules() const {
  if (compiled_result_handle_ == nullptr) {
    return 0;
  }
  if (!LITERT_ABI_HAS_API(parent_v1_, /*req_major=*/1,
                          get_compiled_result_num_byte_code)) {
    return litert::Error(kLiteRtStatusErrorUnsupported);
  }
  LiteRtParamIndex byte_code_idx = 0;
  LITERT_RETURN_IF_ERROR(parent_v1_->get_compiled_result_num_byte_code(
      compiled_result_handle_, &byte_code_idx));
  return byte_code_idx;
}

Expected<LiteRtParamIndex> CompiledResult::NumCalls() const {
  if (compiled_result_handle_ == nullptr) {
    return 0;
  }
  if (!LITERT_ABI_HAS_API(parent_v1_, /*req_major=*/1,
                          get_num_compiled_result_calls)) {
    return litert::Error(kLiteRtStatusErrorUnsupported);
  }
  LiteRtParamIndex num_calls = 0;
  LITERT_RETURN_IF_ERROR(parent_v1_->get_num_compiled_result_calls(
      compiled_result_handle_, &num_calls));
  return num_calls;
}

Expected<CallInformation> CompiledResult::CallInfo(
    LiteRtParamIndex call_idx) const {
  if (!LITERT_ABI_HAS_API(parent_v1_, /*req_major=*/1,
                          get_compiled_result_call_info)) {
    return litert::Error(kLiteRtStatusErrorUnsupported);
  }
  const void* data = nullptr;
  size_t size = 0;
  LiteRtParamIndex byte_code_idx = 0;

  LITERT_RETURN_IF_ERROR(parent_v1_->get_compiled_result_call_info(
      compiled_result_handle_, call_idx, &data, &size, &byte_code_idx));

  absl::string_view call_info_str(reinterpret_cast<const char*>(data), size);
  return ::litert::internal::CallInformation(call_info_str, byte_code_idx);
}

Expected<LiteRtJitExecutable> CompiledResult::GetHandle(
    LiteRtParamIndex byte_code_idx) const {
  if (LITERT_ABI_HAS_API(parent_v1_, /*req_major=*/1,
                         get_compiled_result_handle)) {
    LiteRtJitExecutable handle = nullptr;
    LITERT_RETURN_IF_ERROR(parent_v1_->get_compiled_result_handle(
        compiled_result_handle_, byte_code_idx, &handle));
    return handle;
  }
  return litert::Error(kLiteRtStatusErrorUnsupported);
}

CompiledResult::~CompiledResult() {
  if (compiled_result_handle_ != nullptr &&
      LITERT_ABI_HAS_API(parent_v1_, /*req_major=*/1,
                         destroy_compiled_result)) {
    parent_v1_->destroy_compiled_result(compiled_result_handle_);
  }
}

CompiledResult::CompiledResult(CompiledResult&& other) noexcept
    : parent_v1_(other.parent_v1_),
      compiled_result_handle_(other.compiled_result_handle_) {
  other.parent_v1_ = nullptr;
  other.compiled_result_handle_ = nullptr;
}

CompiledResult& CompiledResult::operator=(CompiledResult&& other) noexcept {
  if (this != &other) {
    if (compiled_result_handle_ != nullptr &&
        LITERT_ABI_HAS_API(parent_v1_, /*req_major=*/1,
                           destroy_compiled_result)) {
      parent_v1_->destroy_compiled_result(compiled_result_handle_);
    }
    parent_v1_ = other.parent_v1_;
    other.parent_v1_ = nullptr;

    compiled_result_handle_ = other.compiled_result_handle_;
    other.compiled_result_handle_ = nullptr;
  }
  return *this;
}

//
// CompilerPlugin
//

namespace {

LiteRtStatus ResolvePluginApi(SharedLibrary& lib,
                              const LiteRtCompilerPluginInterface_V1*& result,
                              LiteRtApiVersion& negotiated_version) {
  auto query_interface_res =
      lib.LookupSymbol<LiteRtCompilerPluginQueryInterfaceT>(
          kLiteRtCompilerPluginQueryInterface.data());
  if (!query_interface_res) {
    if (lib.LookupSymbol<void*>("LiteRtCreateCompilerPlugin")) {
      LITERT_LOG(LITERT_WARNING,
                 "Vendor library exports legacy symbol "
                 "'LiteRtCreateCompilerPlugin' instead of "
                 "'LiteRtCompilerPluginQueryInterface'. Please recompile "
                 "vendor library against the versioned LiteRT Compiler Plugin "
                 "C ABI.");
    }
    return query_interface_res.Error().Status();
  }
  auto query_interface = *query_interface_res;

  LiteRtApiVersion runtime_ver = {LITERT_COMPILER_PLUGIN_ABI_VERSION_MAJOR,
                                  LITERT_COMPILER_PLUGIN_ABI_VERSION_MINOR,
                                  LITERT_COMPILER_PLUGIN_ABI_VERSION_PATCH};
  LiteRtInterface basic_interface = nullptr;

  LiteRtStatus status = litert::internal::NegotiateInterface(
      query_interface, kLiteRtCompilerPluginInterfaceBasic, runtime_ver,
      /*expected_abi_major=*/LITERT_COMPILER_PLUGIN_ABI_VERSION_MAJOR,
      &basic_interface, &negotiated_version);
  if (status != kLiteRtStatusOk || basic_interface == nullptr) {
    LITERT_LOG(LITERT_ERROR,
               "Failed to negotiate compiler plugin interface version "
               "(expected major version %d)",
               LITERT_COMPILER_PLUGIN_ABI_VERSION_MAJOR);
    return kLiteRtStatusErrorWrongVersion;
  }

  result = reinterpret_cast<const LiteRtCompilerPluginInterface_V1*>(
      basic_interface);
  return kLiteRtStatusOk;
}

Expected<std::vector<std::string>> GetSocModels(
    const LiteRtCompilerPluginInterface_V1* api,
    LiteRtCompilerPlugin plugin_handle) {
  std::vector<std::string> soc_models;
  if (!LITERT_ABI_HAS_API(api, /*req_major=*/1,
                          get_num_compiler_plugin_supported_models) ||
      !LITERT_ABI_HAS_API(api, /*req_major=*/1,
                          get_compiler_plugin_supported_soc_model)) {
    return soc_models;
  }

  LiteRtParamIndex num_models = 0;
  LITERT_RETURN_IF_ERROR(api->get_num_compiler_plugin_supported_models(
      plugin_handle, &num_models));

  for (LiteRtParamIndex i = 0; i < num_models; ++i) {
    const char* model = nullptr;
    if (api->get_compiler_plugin_supported_soc_model(
            plugin_handle, i, &model) != kLiteRtStatusOk ||
        model == nullptr) {
      continue;
    }
    soc_models.push_back(std::string(model));
  }

  return soc_models;
}

// Sort plugins so that we first apply those supporting NPU, then those
// supporting GPU, and finally those supporting CPU.
void SortPlugins(std::vector<CompilerPlugin>& compiler_plugins) {
  std::sort(compiler_plugins.begin(), compiler_plugins.end(),
            [](auto& x, auto& y) {
              auto x_supported_hardware = x.SupportedHardware();
              auto y_supported_hardware = y.SupportedHardware();
              if (x_supported_hardware && y_supported_hardware) {
                bool x_npu = (*x_supported_hardware & kLiteRtHwAcceleratorNpu);
                bool x_gpu = (*x_supported_hardware & kLiteRtHwAcceleratorGpu);
                bool x_cpu = (*x_supported_hardware & kLiteRtHwAcceleratorCpu);
                bool y_npu = (*y_supported_hardware & kLiteRtHwAcceleratorNpu);
                bool y_gpu = (*y_supported_hardware & kLiteRtHwAcceleratorGpu);
                bool y_cpu = (*y_supported_hardware & kLiteRtHwAcceleratorCpu);
                int x_score = 100 * x_npu + 10 * x_gpu + x_cpu;
                int y_score = 100 * y_npu + 10 * y_gpu + y_cpu;
                return x_score < y_score;
              }
              return true;
            });
}

}  // namespace

Expected<CompilerPlugin> CompilerPlugin::LoadPlugin(
    const absl::string_view lib_path, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  CompilerPlugin plugin;
  LITERT_LOG(LITERT_INFO, "Loading plugin at: %s", lib_path.data());

#ifdef __ANDROID__
  // Unloading the library on android can lead to crashes.
  auto flags = RtldFlags::Lazy().Local().NoDelete();
#else
  auto flags = RtldFlags::Now().Local().NoDelete();
#endif

  LITERT_ASSIGN_OR_RETURN(plugin.lib_, SharedLibrary::Load(lib_path, flags));
  LITERT_LOG(LITERT_INFO, "Loaded plugin at: %s", lib_path.data());

  LITERT_RETURN_IF_ERROR(ResolvePluginApi(plugin.lib_, plugin.plugin_api_v1_,
                                          plugin.negotiated_version_));
  LITERT_LOG(LITERT_INFO, "Resolved plugin api at: %s", lib_path.data());

  plugin.env_ = env;
  if (!LITERT_ABI_HAS_API(plugin.plugin_api_v1_, /*req_major=*/1,
                          create_compiler_plugin) ||
      !LITERT_ABI_HAS_API(plugin.plugin_api_v1_, /*req_major=*/1,
                          destroy_compiler_plugin)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Plugin missing required entry points");
  }
  LITERT_RETURN_IF_ERROR(plugin.plugin_api_v1_->create_compiler_plugin(
      LrtGetCompilerContext(), &plugin.plugin_handle_, env, options));
  LITERT_LOG(LITERT_INFO, "Initialize plugin at: %s", lib_path.data());

  // This should never change throughout the lifetime of the compiler
  // plugin so save to avoid recalling.
  auto soc_models = GetSocModels(plugin.plugin_api_v1_, plugin.plugin_handle_);
  if (!soc_models) {
    return soc_models.Error();
  }
  plugin.soc_models_ = *soc_models;
  plugin.options_ = options;

  return plugin;
}

Expected<std::vector<CompilerPlugin>> CompilerPlugin::LoadPlugins(
    absl::Span<const absl::string_view> lib_search_paths,
    LiteRtEnvironmentOptions env, LiteRtOptions options) {
  std::vector<std::string> plugin_lib_paths;
  for (auto search_path : lib_search_paths) {
    // Skip paths that are not valid.
    if (Exists(search_path)) {
      LITERT_RETURN_IF_ERROR(
          FindLiteRtCompilerPluginSharedLibs(search_path, plugin_lib_paths));
    }
  }

  std::vector<CompilerPlugin> loaded_plugins;
  loaded_plugins.reserve(lib_search_paths.size());

  for (const auto& lib_path : plugin_lib_paths) {
    LITERT_LOG(LITERT_INFO, "Attempting to load plugin at: %s",
               lib_path.c_str());
    auto plugin = LoadPlugin(lib_path, env, options);
    if (!plugin.HasValue()) {
      LITERT_LOG(LITERT_WARNING, "Failed to load plugin at: %s with error: %s",
                 lib_path.c_str(), plugin.Error().Message().c_str());
      continue;
    }
    loaded_plugins.push_back(std::move(plugin.Value()));
  }

  // Sort plugins.
  SortPlugins(loaded_plugins);

  return loaded_plugins;
}

CompilerPlugin::CompilerPlugin(CompilerPlugin&& other)
    : soc_models_(std::move(other.soc_models_)),
      lib_(std::move(other.lib_)),
      options_(other.options_),
      env_(std::move(other.env_)),
      plugin_api_v1_(other.plugin_api_v1_),
      plugin_handle_(std::move(other.plugin_handle_)),
      transformations_(std::move(other.transformations_)),
      max_transformation_iterations_(other.max_transformation_iterations_),
      negotiated_version_(other.negotiated_version_) {
  other.soc_models_ = {};
  other.plugin_api_v1_ = nullptr;
  other.plugin_handle_ = nullptr;
  other.options_ = nullptr;
  other.env_ = nullptr;
  other.transformations_ = {};
  other.max_transformation_iterations_ = 100;
  other.negotiated_version_ = {0, 0, 0};
}

CompilerPlugin& CompilerPlugin::operator=(CompilerPlugin&& other) {
  if (this != &other) {
    std::swap(soc_models_, other.soc_models_);
    std::swap(lib_, other.lib_);
    std::swap(env_, other.env_);
    std::swap(plugin_api_v1_, other.plugin_api_v1_);
    std::swap(plugin_handle_, other.plugin_handle_);
    std::swap(options_, other.options_);
    std::swap(transformations_, other.transformations_);
    std::swap(max_transformation_iterations_,
              other.max_transformation_iterations_);
    std::swap(negotiated_version_, other.negotiated_version_);
  }
  return *this;
}

CompilerPlugin::~CompilerPlugin() {
  if (plugin_handle_ != nullptr && plugin_api_v1_ != nullptr &&
      LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                         destroy_compiler_plugin)) {
    plugin_api_v1_->destroy_compiler_plugin(plugin_handle_);
  }
}

std::string CompilerPlugin::DebugString() const {
  std::string version_str = "?";
  if (auto version = ApiVersion(); version) {
    version_str = absl::StrFormat("%d.%d.%d", version->major, version->minor,
                                  version->patch);
  }
  return absl::StrFormat("%s compiler plugin (ver %s)", SocManufacturer(),
                         version_str);
}

Expected<LiteRtApiVersion> CompilerPlugin::ApiVersion() const {
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          get_compiler_plugin_version)) {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "get_compiler_plugin_version not supported");
  }
  LiteRtApiVersion api_version;
  LITERT_RETURN_IF_ERROR(
      plugin_api_v1_->get_compiler_plugin_version(&api_version));
  return api_version;
}

Expected<LiteRtHwAccelerators> CompilerPlugin::SupportedHardware() const {
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          get_compiler_plugin_supported_hardware)) {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "get_compiler_plugin_supported_hardware not supported");
  }
  LiteRtHwAccelerators supported_hardware;
  LITERT_RETURN_IF_ERROR(plugin_api_v1_->get_compiler_plugin_supported_hardware(
      plugin_handle_, &supported_hardware));
  return supported_hardware;
}

Expected<std::string> CompilerPlugin::SdkVersion() const {
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          get_compiler_plugin_sdk_version)) {
    return std::string("");
  }
  const char* sdk_version = nullptr;
  LITERT_RETURN_IF_ERROR(plugin_api_v1_->get_compiler_plugin_sdk_version(
      plugin_handle_, &sdk_version));
  if (sdk_version == nullptr) {
    return std::string("");
  }
  return std::string(sdk_version);
}

uint64_t CompilerPlugin::Capabilities() const {
  // The capability query was added in compiler plugin ABI 1.1. Plugins built
  // against an older interface table (or leaving the member null) are treated
  // as having no optional capabilities.
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          get_compiler_plugin_capabilities)) {
    return kLiteRtCompilerPluginCapabilityNone;
  }
  uint64_t capabilities = kLiteRtCompilerPluginCapabilityNone;
  if (auto status = plugin_api_v1_->get_compiler_plugin_capabilities(
          plugin_handle_, &capabilities);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "Failed to query compiler plugin capabilities (status %d); "
               "assuming none.",
               status);
    return kLiteRtCompilerPluginCapabilityNone;
  }
  return capabilities;
}

Expected<void> CompilerPlugin::RegisterAllTransformations() {
  // Clear previously registered transformations in case this CompilerPlugin
  // instance is reused across multiple models (e.g. in ATS test suites).
  transformations_.clear();
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          register_all_transformations)) {
    return {};
  }
  LiteRtParamIndex num_patterns = 0;
  LiteRtTransformation* transformations = nullptr;

  LITERT_RETURN_IF_ERROR(plugin_api_v1_->register_all_transformations(
      plugin_handle_, &transformations, &num_patterns));
  if (num_patterns > 0) {
    if (transformations == nullptr) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Transformations array pointer is null.");
    }
    for (LiteRtParamIndex i = 0; i < num_patterns; ++i) {
      if (transformations[i].pattern == nullptr) {
        return Unexpected(
            kLiteRtStatusInvalidTransformation,
            absl::StrFormat(
                "Transformation %d has a invalid pattern function.", i));
      }
      if (transformations[i].name == nullptr) {
        return Unexpected(
            kLiteRtStatusInvalidTransformation,
            absl::StrFormat(
                "Transformation %d has a invalid transformation name.", i));
      }
      transformations_.push_back(transformations[i]);
    }
  }

  // Sort transformations by benefit.
  std::sort(transformations_.begin(), transformations_.end(),
            [](const LiteRtTransformation& a, const LiteRtTransformation& b) {
              return a.benefit > b.benefit;
            });
  return {};
}

Expected<void> CompilerPlugin::GreedyPatternMatchAndRewrite(
    LiteRtModelT& model) {
  LITERT_LOG(LITERT_DEBUG, "GreedyPatternMatchAndRewrite, total patterns: %d",
             transformations_.size());
  for (auto& subgraph : model.Subgraphs()) {
    bool subgraph_modified = true;
    size_t iterations = 0;
    while (subgraph_modified) {
      subgraph_modified = false;
      LITERT_LOG(LITERT_DEBUG, "Iteration %zu", iterations);
      if (iterations++ >= max_transformation_iterations_) {
        LITERT_LOG(LITERT_WARNING,
                   "Hit max transformation iterations limit (%zu)",
                   max_transformation_iterations_);
        break;
      }
      std::queue<LiteRtOp> worklist;
      for (const auto& op : subgraph->Ops()) {
        worklist.push(op);
      }
      LITERT_LOG(LITERT_DEBUG, "Worklist size: %lu", worklist.size());
      while (!worklist.empty()) {
        LiteRtOp op = worklist.front();
        worklist.pop();

        // Check if the op is still in the subgraph.
        if (!absl::c_linear_search(subgraph->Ops(), op)) {
          continue;
        }
        LITERT_LOG(LITERT_DEBUG, "Matching pattern for op: %d", op->OpCode());
        for (const auto& transformation : transformations_) {
          LiteRtBuilderT builder;
          LITERT_LOG(LITERT_DEBUG, "Matching pattern '%s'",
                     transformation.name);
          // Call the function pointer.
          if (transformation.pattern(LrtGetCompilerContext(), &builder, op) ==
              kLiteRtStatusOk) {
            LITERT_LOG(LITERT_DEBUG, "Matched pattern '%s'",
                       transformation.name);

            builder.ApplyChanges(subgraph);
            subgraph_modified = true;
            // Break from the inner transformation loop since the graph changed.
            break;
          }
        }
        if (subgraph_modified) {
          // Restart the scan for this subgraph as it has been modified.
          LITERT_LOG(LITERT_DEBUG, "Restarting scan");
          break;
        }
      }
    }
  }
  return {};
}

Expected<std::vector<LiteRtOpWithPartitionIndex>> CompilerPlugin::Partition(
    LiteRtSubgraph subgraph, absl::string_view soc_model) {
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          compiler_plugin_partition)) {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "compiler_plugin_partition not supported");
  }
  LiteRtOpListT ops;
  const char* soc_model_str = !soc_model.empty() ? soc_model.data() : nullptr;
  LITERT_PERFETTO_TRACE_EVENT("CompilerPlugin Partition");
  LITERT_RETURN_IF_ERROR(plugin_api_v1_->compiler_plugin_partition(
      plugin_handle_, soc_model_str, subgraph, &ops));
  return ops.Values();
}

Expected<CompiledResult> CompilerPlugin::Compile(LiteRtModel partitions,
                                                 absl::string_view soc_model) {
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          compiler_plugin_compile)) {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "compiler_plugin_compile not supported");
  }
  CompiledResult result = MakeResult();
  // Skip compilation if the model is empty.
  if (partitions->Subgraphs().empty()) {
    LITERT_LOG(LITERT_INFO,
               "CompilerPlugin::Compile: Skipped for non partitioned model.");
    return result;
  }
  // If the user has passed an soc_model, then we use it; otherwise we let the
  // backend pick the appropriate one by passing nullptr as soc_model. This is
  // important for on-device compilation, where the backend must determine the
  // SoC model based on the user device.
  const char* soc_model_str = !soc_model.empty() ? soc_model.data() : nullptr;
  LITERT_PERFETTO_TRACE_EVENT("CompilerPlugin Compile");
  LITERT_RETURN_IF_ERROR(plugin_api_v1_->compiler_plugin_compile(
      plugin_handle_, soc_model_str, partitions,
      &result.compiled_result_handle_));
  return result;
}

namespace {

LiteRtStatus PartitionSubgraph(
    std::vector<LiteRtOpWithPartitionIndex> selected_ops,
    LiteRtSubgraphT& subgraph, std::vector<LiteRtOp>& res_ops,
    std::vector<LiteRtSubgraph>& res_subgraphs, LiteRtModelT& model,
    const LiteRtCompilerOptionsPartitionStrategy& partition_strategy_option,
    size_t max_partitions = 0) {
  // Pick partition strategy based on compiler options.
  std::vector<std::vector<LiteRtOp>> (*partition_strategy_func)(
      const std::vector<LiteRtOpWithPartitionIndex>&, LiteRtSubgraph) =
      GroupPartitions;
  if (partition_strategy_option ==
      kLiteRtCompilerOptionsPartitionStrategyDefault) {
    partition_strategy_func = GroupPartitionsV2;
  }
  LITERT_LOG(LITERT_INFO, "Partition strategy: %d", partition_strategy_option);

  // Group selected ops into connected islands.
  auto islands = partition_strategy_func(selected_ops, &subgraph);
  if (islands.empty()) {
    return kLiteRtStatusOk;
  }

  if (max_partitions > 0 && islands.size() > max_partitions) {
    LITERT_LOG(LITERT_ERROR,
               "Partitioning yielded %lu NPU islands, which exceeds the "
               "maximum allowed (%lu).",
               islands.size(), max_partitions);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // For each connected island, slice into new subgraph and replace use with
  // single dispatch op.
  for (auto& island : islands) {
    auto& new_subgraph = model.EmplaceSubgraph();
    LITERT_ASSIGN_OR_RETURN(auto* dispatch_op,
                            OutlinePartition(subgraph, &new_subgraph, island));
    res_ops.push_back(dispatch_op);
    res_subgraphs.push_back(&new_subgraph);
  }

  return kLiteRtStatusOk;
}

bool IsCpuCall(LiteRtOp op) {
  auto info = GetOptionsAs<CompositeOptions>(op);
  return info && info->name == CompositeOptions::kCpuCall;
}

// Removes selections that must not be honored: odml.cpu_call composites, and
// region-bearing ops if the plugin cannot compile region subgraphs.
void FilterSelectedOps(std::vector<LiteRtOpWithPartitionIndex>& selected_ops,
                       bool supports_region_subgraphs) {
  size_t num_ignored_region_ops = 0;
  selected_ops.erase(
      std::remove_if(selected_ops.begin(), selected_ops.end(),
                     [&](const LiteRtOpWithPartitionIndex& selected) {
                       if (IsCpuCall(selected.first)) {
                         return true;
                       }
                       if (!supports_region_subgraphs &&
                           HasRegionRefs(*selected.first)) {
                         ++num_ignored_region_ops;
                         return true;
                       }
                       return false;
                     }),
      selected_ops.end());
  if (num_ignored_region_ops > 0) {
    LITERT_LOG(LITERT_WARNING,
               "Ignoring the selection of %zu region-bearing op(s): the "
               "compiler plugin does not report "
               "kLiteRtCompilerPluginCapabilityRegionSubgraphs.",
               num_ignored_region_ops);
  }
}

absl::flat_hash_map<LiteRtSubgraph, uint32_t> IndexSubgraphs(
    const LiteRtModelT& model) {
  absl::flat_hash_map<LiteRtSubgraph, uint32_t> index_of;
  index_of.reserve(model.NumSubgraphs());
  for (uint32_t i = 0; i < model.NumSubgraphs(); ++i) {
    index_of.emplace(model.Subgraphs()[i], i);
  }
  return index_of;
}

absl::flat_hash_set<LiteRtSubgraph> SignatureSubgraphs(
    const LiteRtModelT& model) {
  absl::flat_hash_set<LiteRtSubgraph> result;
  for (LiteRtSignature signature : model.Signatures()) {
    result.insert(&signature->GetSubgraph());
  }
  return result;
}

// Points every region reference to `from` held by ops in `subgraphs` at `to`.
void RedirectRegionRefs(LiteRtModelT& model,
                        absl::Span<const uint32_t> subgraphs, int32_t from,
                        int32_t to) {
  for (const uint32_t sg : subgraphs) {
    for (LiteRtOp op : model.Subgraph(sg).Ops()) {
      if (!HasSubgraphRefs(*op)) {
        continue;
      }
      UpdateSubgraphRefs(*op, [from, to](int32_t& ind, SubgraphRefKind kind) {
        if (ind == from && IsRegionRefKind(kind)) {
          ind = to;
        }
      });
    }
  }
}

// Appends a clone of subgraph `sg` to `model` and points the region references
// held by ops in `callers` at the clone. Returns the index of the clone.
Expected<uint32_t> CloneAndRedirect(LiteRtModelT& model, uint32_t sg,
                                    absl::Span<const uint32_t> callers) {
  LITERT_ASSIGN_OR_RETURN(const uint32_t clone,
                          CloneSubgraph(model, model.Subgraph(sg)));
  RedirectRegionRefs(model, callers, static_cast<int32_t>(sg),
                     static_cast<int32_t>(clone));
  return clone;
}

// Bookkeeping for the subgraphs of the input model while they are visited.
struct VisitState {
  explicit VisitState(size_t num_subgraphs)
      : input_num_sgs(num_subgraphs),
        visited(num_subgraphs, false),
        cpu_resident(num_subgraphs, false) {}

  // Number of subgraphs of the input model. Subgraphs created during
  // partitioning (outlined partitions and clones) are appended after them.
  size_t input_num_sgs;
  // Composite decompositions, which are handled separately.
  absl::flat_hash_set<uint32_t> decomp_subgraphs;
  // Subgraphs exposed through a signature, which can be called directly.
  absl::flat_hash_set<LiteRtSubgraph> signature_subgraphs;
  std::vector<bool> visited;
  // Visited subgraphs that remain in the model (as opposed to being owned by
  // an offloaded op).
  std::vector<bool> cpu_resident;
};

// Returns true if the region subgraph `sg` of the input model should be
// partitioned on its own. This is only the case for control flow bodies that
// stay callable from CPU, i.e. that are referenced by a control flow op
// staying on CPU or exposed through a signature. Computation regions are never
// partitioned on their own (see PartitionModel).
//
// If a body to be partitioned is also referenced by offloaded ops (outlined
// partitions, or region subgraphs owned by them), those ops are first
// redirected to a pristine clone, so that they don't end up sharing a
// subgraph that contains dispatch ops.
Expected<bool> PrepareRegionSubgraphForPartitioning(LiteRtModelT& model,
                                                    uint32_t sg,
                                                    const VisitState& state) {
  // Earlier visits may have outlined or inlined callers of `sg`, so the call
  // graph is rebuilt for every region subgraph. This is O(#regions * #ops).
  LITERT_ASSIGN_OR_RETURN(auto graph, SubgraphCallGraph::Build(model));
  bool has_computation_caller = false;
  bool callable_from_cpu =
      state.signature_subgraphs.contains(model.Subgraphs()[sg]);
  std::vector<uint32_t> offloaded_callers;
  for (const auto& edge : graph.Callers(sg)) {
    if (!IsRegionRefKind(edge.kind) ||
        state.decomp_subgraphs.contains(edge.caller)) {
      continue;
    }
    if (edge.kind == SubgraphRefKind::kComputation) {
      has_computation_caller = true;
    }
    const bool caller_offloaded =
        edge.caller >= state.input_num_sgs ||
        (state.visited[edge.caller] && !state.cpu_resident[edge.caller]);
    if (caller_offloaded) {
      offloaded_callers.push_back(edge.caller);
    } else if (edge.kind == SubgraphRefKind::kControlFlow &&
               state.cpu_resident[edge.caller]) {
      callable_from_cpu = true;
    }
  }
  if (has_computation_caller || !callable_from_cpu) {
    return false;
  }
  if (!offloaded_callers.empty()) {
    LITERT_ASSIGN_OR_RETURN(const uint32_t clone,
                            CloneAndRedirect(model, sg, offloaded_callers));
    LITERT_LOG(LITERT_INFO,
               "Cloned control flow subgraph<%u> as subgraph<%u> for "
               "offloaded callers before partitioning it for CPU callers.",
               sg, clone);
  }
  return true;
}

// Selects ops of the candidate subgraph `sg` with `partition_fn`, inlines the
// unselected composites and outlines the selected ops into new partition
// subgraphs appended to `model`.
Expected<void> PartitionCandidateSubgraph(
    PartitionFn partition_fn, const PartitionModelOptions& options,
    LiteRtModelT& model, uint32_t sg, std::vector<LiteRtOp>& dispatch_ops,
    std::vector<LiteRtSubgraph>& entry_subgraphs,
    std::vector<size_t>& selected_composite_subgraph_indexes) {
  auto* subgraph = model.Subgraphs()[sg];
  auto selected_ops = partition_fn(subgraph);
  // TODO ensure selected ops don't contain npu_calls.
  if (!selected_ops) {
    return selected_ops.Error();
  }

  // Shield odml.cpu_call from being selected by the plugin, as well as
  // region-bearing ops if the plugin can't compile region subgraphs.
  FilterSelectedOps(*selected_ops, options.supports_region_subgraphs);

  // Find all composite ops (except npu_calls and cpu_calls) that are not
  // selected.
  std::vector<LiteRtOp> ops_to_inline;
  for (auto& op : subgraph->Ops()) {
    auto info = GetOptionsAs<CompositeOptions>(op);
    if (!info) {
      continue;
    }
    if (info->name == CompositeOptions::kNpuCall ||
        info->name == CompositeOptions::kCpuCall) {
      continue;
    }
    auto is_composite_selected = std::any_of(
        selected_ops->begin(), selected_ops->end(),
        [&op](const auto& selected_op) { return selected_op.first == op; });
    if (!is_composite_selected) {
      ops_to_inline.push_back(op);
    }
  }

  // Inline all composite ops that are not selected.
  for (auto& op : ops_to_inline) {
    auto info = GetOptionsAs<CompositeOptions>(op);
    auto status = InlineSubgraph(model, *op, model.Subgraphs()[info->subgraph]);
    if (status) {
      // Now the decomposition subgraph is inlined into the main subgraph,
      // therefore its also "selected".
      selected_composite_subgraph_indexes.push_back(info->subgraph);
    }
  }
  if (!ops_to_inline.empty()) {
    LITERT_LOG(LITERT_INFO, "Inlined %zu composite ops into subgraph<%u>",
               ops_to_inline.size(), sg);
    // Re-do partitioning only if inlining happened.
    selected_ops->clear();
    selected_ops = partition_fn(subgraph);
    if (!selected_ops) {
      return selected_ops.Error();
    }

    // Re-shield odml.cpu_call and unsupported region-bearing ops.
    FilterSelectedOps(*selected_ops, options.supports_region_subgraphs);
  }

  // Record all decomposition subgraph indexes, where its compositie op will
  // be compiled without relying on the decomposition body.
  for (auto& op : *selected_ops) {
    auto info = GetOptionsAs<CompositeOptions>(op.first);
    if (!info) {
      continue;
    }
    if (info->name == CompositeOptions::kNpuCall) {
      continue;
    }
    selected_composite_subgraph_indexes.push_back(info->subgraph);
  }
  const size_t num_selected_ops = selected_ops->size();
  const size_t num_ops = subgraph->Ops().size();

  const size_t num_partitions_before = dispatch_ops.size();
  LITERT_RETURN_IF_ERROR(PartitionSubgraph(
      std::move(*selected_ops), *subgraph, dispatch_ops, entry_subgraphs, model,
      options.partition_strategy, options.max_partitions));
  LITERT_LOG(LITERT_INFO,
             "Partitioned subgraph<%u>, selected %zu "
             "ops, from a total of %zu ops. resulted in %zu partitions.",
             sg, num_selected_ops, num_ops,
             dispatch_ops.size() - num_partitions_before);
  return {};
}

// Returns true if the decomposition of any odml.npu_call composite
// (transitively) contains a region-bearing op.
bool NpuCallsContainRegions(const LiteRtModelT& model,
                            const SubgraphCallGraph& graph) {
  std::vector<uint32_t> pending;
  ForEachIr(model, [&](LiteRtOp op) {
    auto info = GetOptionsAs<CompositeOptions>(op);
    if (info && info->name == CompositeOptions::kNpuCall &&
        info->subgraph >= 0) {
      pending.push_back(static_cast<uint32_t>(info->subgraph));
    }
  });
  absl::flat_hash_set<uint32_t> seen(pending.begin(), pending.end());
  while (!pending.empty()) {
    const uint32_t sg = pending.back();
    pending.pop_back();
    for (const auto& edge : graph.Callees(sg)) {
      if (IsRegionRefKind(edge.kind)) {
        return true;
      }
      if (seen.insert(edge.callee).second) {
        pending.push_back(edge.callee);
      }
    }
  }
  return false;
}

// Returns the region subgraphs (transitively) referenced by `entries`,
// excluding the entries themselves, in breadth-first order.
std::vector<uint32_t> CollectRegionClosure(const SubgraphCallGraph& graph,
                                           absl::Span<const uint32_t> entries) {
  absl::flat_hash_set<uint32_t> visited(entries.begin(), entries.end());
  std::vector<uint32_t> closure;
  std::queue<uint32_t> pending;
  for (const uint32_t entry : entries) {
    pending.push(entry);
  }
  while (!pending.empty()) {
    const uint32_t sg = pending.front();
    pending.pop();
    for (const auto& edge : graph.Callees(sg)) {
      if (!IsRegionRefKind(edge.kind)) {
        continue;
      }
      if (visited.insert(edge.callee).second) {
        closure.push_back(edge.callee);
        pending.push(edge.callee);
      }
    }
  }
  return closure;
}

// Computes the region closure of `entries` and makes sure every subgraph in it
// is referenced only from within the entries and the closure: subgraphs also
// referenced by ops staying in the model, or exposed through a signature, are
// replaced by clones. Returns the closure as indices into `model`.
Expected<std::vector<uint32_t>> OwnRegionClosure(
    LiteRtModelT& model, absl::Span<const LiteRtSubgraph> entries) {
  const auto signature_subgraphs = SignatureSubgraphs(model);
  // Each round replaces at least one shared subgraph of the closure by a clone
  // whose callers are all owned. Cloning a subgraph can only make the
  // subgraphs it references shared, i.e. sharing moves one level down the
  // (acyclic) reference graph per round, so the loop terminates after at most
  // depth-of-the-reference-graph rounds.
  while (true) {
    LITERT_ASSIGN_OR_RETURN(auto graph, SubgraphCallGraph::Build(model));
    const auto index_of = IndexSubgraphs(model);
    std::vector<uint32_t> entry_inds;
    entry_inds.reserve(entries.size());
    for (LiteRtSubgraph entry : entries) {
      const auto it = index_of.find(entry);
      ABSL_DCHECK(it != index_of.end());
      entry_inds.push_back(it->second);
    }
    auto closure = CollectRegionClosure(graph, entry_inds);

    std::vector<uint32_t> owned(entry_inds);
    owned.insert(owned.end(), closure.begin(), closure.end());
    const absl::flat_hash_set<uint32_t> owned_set(owned.begin(), owned.end());

    std::vector<uint32_t> shared;
    for (const uint32_t sg : closure) {
      bool is_shared = signature_subgraphs.contains(model.Subgraphs()[sg]);
      for (const auto& edge : graph.Callers(sg)) {
        is_shared |= !owned_set.contains(edge.caller);
      }
      if (is_shared) {
        shared.push_back(sg);
      }
    }
    if (shared.empty()) {
      return closure;
    }
    for (const uint32_t sg : shared) {
      LITERT_ASSIGN_OR_RETURN(const uint32_t clone,
                              CloneAndRedirect(model, sg, owned));
      LITERT_LOG(LITERT_INFO,
                 "Cloned shared region subgraph<%u> as subgraph<%u> for "
                 "offloaded ops.",
                 sg, clone);
    }
  }
}

// Gives every private region subgraph of `partitions` a unique, non-empty
// name. Backends importing the partitions model (e.g. through
// tflite::FlatBufferToMlir) derive symbol names from subgraph names.
void NameRegionSubgraphs(LiteRtModelT& partitions) {
  absl::flat_hash_set<std::string> names;
  for (LiteRtSubgraph sg : partitions.Subgraphs()) {
    if (!sg->Name().empty()) {
      names.insert(std::string(sg->Name()));
    }
  }
  absl::flat_hash_set<std::string> seen;
  for (size_t i = partitions.NumEntrySubgraphs(); i < partitions.NumSubgraphs();
       ++i) {
    LiteRtSubgraphT& sg = partitions.Subgraph(i);
    std::string name(sg.Name());
    if (!name.empty() && seen.insert(name).second) {
      continue;
    }
    size_t suffix = i;
    do {
      name = absl::StrCat("litert_region_", suffix++);
    } while (names.contains(name));
    names.insert(name);
    seen.insert(name);
    sg.SetName(std::move(name));
  }
}

// Removes the given region subgraphs from `model` once nothing references them
// anymore (e.g. the bodies of a composite whose decomposition was dropped).
void RemoveOrphanedRegionSubgraphs(
    LiteRtModelT& model, const absl::flat_hash_set<LiteRtSubgraph>& regions) {
  const auto signature_subgraphs = SignatureSubgraphs(model);
  // Every round removes at least one subgraph, so this terminates. Removing a
  // subgraph can orphan the subgraphs it references, hence the loop.
  while (true) {
    auto graph = SubgraphCallGraph::Build(model);
    if (!graph) {
      LITERT_LOG(LITERT_WARNING, "Skipping region cleanup: %s",
                 graph.Error().Message().c_str());
      return;
    }
    std::vector<size_t> orphans;
    for (uint32_t i = 0; i < model.NumSubgraphs(); ++i) {
      LiteRtSubgraph sg = model.Subgraphs()[i];
      if (regions.contains(sg) && !graph->IsReferenced(i) &&
          !signature_subgraphs.contains(sg)) {
        orphans.push_back(i);
      }
    }
    if (orphans.empty()) {
      return;
    }
    LITERT_LOG(LITERT_INFO, "Removing %zu unreferenced region subgraph(s).",
               orphans.size());
    model.Yank(std::move(orphans));
  }
}

// Moves `entry_subgraphs` (in order) and the region subgraphs they reference
// out of `model` into a new partitions model, cleans up region subgraphs of
// `model` that became unreferenced and validates both models.
Expected<LiteRtModelT> BuildPartitionsModel(
    LiteRtModelT& model, absl::Span<const LiteRtSubgraph> entry_subgraphs,
    const absl::flat_hash_set<LiteRtSubgraph>& region_subgraphs,
    bool supports_region_subgraphs) {
  // Region subgraphs referenced by the subgraphs to be compiled travel with
  // them as private subgraphs.
  LITERT_ASSIGN_OR_RETURN(auto region_closure,
                          OwnRegionClosure(model, entry_subgraphs));
  if (!region_closure.empty() && !supports_region_subgraphs) {
    // Region-bearing ops are never outlined for such plugins and npu_calls
    // containing regions are rejected upfront, so this is unexpected.
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Region subgraphs would be passed to a compiler plugin that "
                 "does not support them.");
  }

  // Entry partitions come first, in the order of their dispatch ops, followed
  // by the private region subgraphs.
  const auto index_of = IndexSubgraphs(model);
  std::vector<size_t> to_compile;
  to_compile.reserve(entry_subgraphs.size() + region_closure.size());
  for (LiteRtSubgraph entry : entry_subgraphs) {
    const auto it = index_of.find(entry);
    ABSL_DCHECK(it != index_of.end());
    to_compile.push_back(it->second);
  }
  to_compile.insert(to_compile.end(), region_closure.begin(),
                    region_closure.end());

  auto partitions = model.Yank(std::move(to_compile));
  partitions.SetNumEntrySubgraphs(entry_subgraphs.size());
  NameRegionSubgraphs(partitions);

  RemoveOrphanedRegionSubgraphs(model, region_subgraphs);

  if (auto status = ValidateSubgraphRefs(model); !status) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 absl::StrCat("Partitioning left an invalid subgraph "
                              "reference in the model: ",
                              status.Error().Message()));
  }
  if (auto status = ValidateSubgraphRefs(partitions); !status) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 absl::StrCat("Partitioning left an invalid subgraph "
                              "reference in the partitions model: ",
                              status.Error().Message()));
  }
  if (partitions.NumSubgraphs() > partitions.NumEntrySubgraphs()) {
    LITERT_LOG(LITERT_INFO,
               "Partitions model has %zu entry subgraph(s) and %zu private "
               "region subgraph(s).",
               partitions.NumEntrySubgraphs(),
               partitions.NumSubgraphs() - partitions.NumEntrySubgraphs());
  }
  return partitions;
}

}  // namespace

Expected<PartitionResult> PartitionModel(
    CompilerPlugin& compiler_plugin, LiteRtModelT& model,
    absl::string_view soc_model,
    const absl::flat_hash_set<uint32_t>& subgraphs_to_partition) {
  PartitionModelOptions options;
  options.supports_region_subgraphs = compiler_plugin.SupportsRegionSubgraphs();
  // Get partition strategy from compiler options.
  if (auto compiler_options = compiler_plugin.CompilerOptions();
      compiler_options.HasValue()) {
    options.partition_strategy = compiler_options->partition_strategy;
    options.max_partitions = compiler_options->max_partitions;
  }
  auto partition_fn = [&compiler_plugin, soc_model](LiteRtSubgraph subgraph) {
    return compiler_plugin.Partition(subgraph, soc_model);
  };
  return PartitionModel(partition_fn, options, model, subgraphs_to_partition);
}

Expected<PartitionResult> PartitionModel(
    PartitionFn partition_fn, const PartitionModelOptions& options,
    LiteRtModelT& model,
    const absl::flat_hash_set<uint32_t>& subgraphs_to_partition) {
  // This algorithm decides the subgraphs to be partitioned by the plugin. This
  // is a trivial process with the exception of composite ops and their
  // decomposition subgraphs. Currently, we deploy the most naive approach to
  // handling composite ops.
  //
  // There are a couple cases to consider:
  // 1. The composite op is an "odml.npu_call", in which case it represents a
  // partition which was explicitly requested by the model author.
  //
  // In this case, the the composite itself is always selected, regardless of
  // whether the plugin selects it. Its subgraph is not passed to the partition
  // function and it is passed in its entirety to the compilation function.
  //
  // More advanced behavior could include:
  // * Ensuring the plugin can compile the entire partition, and inlining it if
  // not.
  //
  // 2. The composite op is an "odml.cpu_call", in which case it represents an
  // explicit request to leave a portion of the model on the CPU.
  //
  // In this case, the composite itself is NEVER selected for partitioning. Its
  // decomposition subgraph is inlined back into the main subgraph AFTER the
  // partitioning phase is complete. This ensures that the operations within
  // the decomposition are not available for acceleration.
  //
  // 3. Standard non npu_call or cpu_call composite ops.
  //
  // 3.1 Composite op is supported by the plugin. Since a plugin is capable of
  // directly compile the composite op, the composite op will be selected
  // during partitioning. Therefore, all ops in the decomposition subgraph will
  // be ignored, furthermore, the decomposition subgraph will be removed from
  // the model.
  //
  // 3.2 Composite op is not supported by the plugin. In this case, the
  // composite op will not be selected during partitioning. To give best effort
  // compilation, we inline the decomposition subgraph into the main subgraph,
  // that is, replace the composite op with the ops in the decomposition
  // subgraph. After re-partitioning, there are chances that the decomposition
  // Ops are now selected by the plugin.
  //
  // More advanced behavior could include:
  // * Allowing the plugin to compile the decomposition subgraph in the case
  // it cannot lower the composite directly. Potentially inline in this case
  // contingent on the availability of a suitable CPU kernel for the composite
  // op.
  //
  // ASSUMPTIONS:
  // * npu_call ops ARE NOT nested within decompositions of other npu_call ops.
  // * Standard composite ops ARE allowed to be nested within decompositions of
  // npu_call ops.
  // * No two npu_call ops share the same subgraph.
  // * Any composite op in a decomposition subgraph will be treated as a regular
  // op and inlined back into the main subgraph.
  //
  // REGIONS
  //
  // Region-bearing ops (e.g. STABLEHLO_REDUCE_WINDOW, STABLEHLO_SORT, WHILE,
  // IF) reference other subgraphs by index, see core/model/subgraph_refs.h.
  // These subgraphs are part of the semantics of the referencing op:
  //
  // 1. Computation regions (reduction bodies, comparators, scatter updates) are
  // never partitioned on their own. They are scalar functions applied per
  // element, and the TFLite CPU kernels (e.g. stablehlo_reduce_window,
  // stablehlo_scatter) do not invoke them: they require a single builtin op in
  // the body and pattern-match its op code. Replacing the body with a
  // DISPATCH_OP would make the CPU kernel fail.
  //
  // 2. Control flow bodies (WHILE/IF/CASE/CALL_ONCE) are partitioned on their
  // own while they stay callable from CPU (an op referencing them stays on CPU,
  // or they are exposed through a signature), which is the legacy behavior.
  //
  // 3. Region-bearing ops may only be offloaded by plugins reporting
  // kLiteRtCompilerPluginCapabilityRegionSubgraphs. Their region subgraphs are
  // moved into the partitions model after the entry partitions (one per
  // dispatch op), with all subgraph indices remapped, and are cloned first if
  // they are still used outside of the offloaded ops. For other plugins,
  // selections of region-bearing ops are ignored, and odml.npu_call
  // composites containing region-bearing ops are rejected before the model is
  // modified.
  //
  // 4. Region subgraphs left without any reference are removed from the model.

  VisitState state(model.NumSubgraphs());
  std::vector<size_t> selected_composite_subgraph_indexes;

  // Find decomposition subgraphs and track all cpu_call ops and their
  // decomposition subgraphs for late inlining.
  std::vector<std::pair<LiteRtOp, uint32_t>> cpu_calls;
  ForEachIr(&model, [&](LiteRtOp op) {
    auto info = GetOptionsAs<CompositeOptions>(op);
    if (!info) {
      return;
    }
    state.decomp_subgraphs.insert(info->subgraph);
    if (info->name == CompositeOptions::kCpuCall) {
      cpu_calls.push_back({op, info->subgraph});
    }
  });

  // Find region subgraphs of the input model. Subgraphs are visited callers
  // first so that it is known whether the ops referencing a region stay on CPU
  // by the time the region is visited.
  LITERT_ASSIGN_OR_RETURN(auto input_graph, SubgraphCallGraph::Build(model));
  LITERT_ASSIGN_OR_RETURN(auto visit_order, input_graph.TopologicalOrder());
  if (!options.supports_region_subgraphs &&
      NpuCallsContainRegions(model, input_graph)) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "An odml.npu_call contains region-bearing ops, but the "
                 "compiler plugin does not report "
                 "kLiteRtCompilerPluginCapabilityRegionSubgraphs.");
  }
  absl::flat_hash_set<LiteRtSubgraph> region_subgraphs;
  for (uint32_t i = 0; i < state.input_num_sgs; ++i) {
    if (input_graph.IsReferencedAs(i, SubgraphRefKind::kComputation) ||
        input_graph.IsReferencedAs(i, SubgraphRefKind::kControlFlow)) {
      region_subgraphs.insert(model.Subgraphs()[i]);
    }
  }
  state.signature_subgraphs = SignatureSubgraphs(model);

  // Build partition result via calling plugin on non-decomposition subgraphs.
  std::vector<LiteRtOp> dispatch_ops;
  // Subgraph to be compiled for each dispatch op.
  std::vector<LiteRtSubgraph> entry_subgraphs;
  for (const uint32_t i : visit_order) {
    if (state.decomp_subgraphs.contains(i)) {
      continue;
    }
    state.visited[i] = true;
    if (region_subgraphs.contains(model.Subgraphs()[i])) {
      LITERT_ASSIGN_OR_RETURN(
          const bool partition_on_its_own,
          PrepareRegionSubgraphForPartitioning(model, i, state));
      if (!partition_on_its_own) {
        LITERT_LOG(LITERT_INFO,
                   "Subgraph<%u> is a region of another op, not partitioning "
                   "it on its own.",
                   i);
        continue;
      }
    }
    state.cpu_resident[i] = true;
    if (!subgraphs_to_partition.empty() &&
        !subgraphs_to_partition.contains(i)) {
      continue;
    }
    LITERT_RETURN_IF_ERROR(PartitionCandidateSubgraph(
        partition_fn, options, model, i, dispatch_ops, entry_subgraphs,
        selected_composite_subgraph_indexes));
  }

  // Late inline all odml.cpu_call ops.
  for (auto& [op, sg_idx] : cpu_calls) {
    auto status = InlineSubgraph(model, *op, model.Subgraphs()[sg_idx]);
    if (status) {
      selected_composite_subgraph_indexes.push_back(sg_idx);
    }
  }

  ABSL_DCHECK_EQ(dispatch_ops.size(), entry_subgraphs.size());

  // Remove all decomposition subgraphs from the model. Composite ops still
  // referring to them (i.e. offloaded composites) are left without a
  // decomposition. Forget them as region candidates first, since their
  // storage is released.
  absl::c_sort(selected_composite_subgraph_indexes);
  selected_composite_subgraph_indexes.erase(
      std::unique(selected_composite_subgraph_indexes.begin(),
                  selected_composite_subgraph_indexes.end()),
      selected_composite_subgraph_indexes.end());
  for (const size_t ind : selected_composite_subgraph_indexes) {
    region_subgraphs.erase(model.Subgraphs()[ind]);
  }
  model.Yank(std::move(selected_composite_subgraph_indexes));

  // Collect all npu_call ops, and their decomposition subgraphs indexes.
  // Note:  we do this after partitioning and removing decomposition subgraphs,
  // so subgraph indexes of npu_calls are also updated.
  std::vector<CompositeOptions> npu_calls;
  ForEachIr(&model, [&](LiteRtOp op) {
    auto info = GetOptionsAs<CompositeOptions>(op);
    if (!info) {
      return;
    }
    if (info->name == CompositeOptions::kNpuCall) {
      npu_calls.push_back(std::move(*info));
    }
  });

  // Add collect all the subgraphs to be compiled. These are the bodies of
  // outlined partitions or npu_calls.
  for (auto& npu_call : npu_calls) {
    auto* op = npu_call.op;
    entry_subgraphs.push_back(model.Subgraphs()[npu_call.subgraph]);
    MakeDispatchOp(*op);
    dispatch_ops.push_back(op);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto partitions,
      BuildPartitionsModel(model, entry_subgraphs, region_subgraphs,
                           options.supports_region_subgraphs));
  return PartitionResult{std::move(dispatch_ops), std::move(partitions)};
}

Expected<PartitionResult> PartitionModelDirect(
    std::vector<LiteRtOpWithPartitionIndex> selected_ops, LiteRtModelT& model) {
  if (model.Subgraphs().size() != 1) {
    // Only single subgraphs supported for direct partitioning.
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }
  // Accumulate partition results for each subgraph in model.
  auto* subgraph = model.Subgraphs().front();
  std::vector<LiteRtOp> dispatch_ops;
  std::vector<LiteRtSubgraph> entry_subgraphs;
  LITERT_RETURN_IF_ERROR(PartitionSubgraph(
      std::move(selected_ops), *subgraph, dispatch_ops, entry_subgraphs, model,
      kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected));
  ABSL_DCHECK_EQ(dispatch_ops.size(), model.NumSubgraphs() - 1);

  std::vector<size_t> decomps_to_compile;
  for (auto i = 1; i < model.NumSubgraphs(); ++i) {
    // Outlined subgraphs are pushed to the back of the model.
    decomps_to_compile.push_back(i);
  }
  auto new_model = model.Yank(std::move(decomps_to_compile));

  return PartitionResult{std::move(dispatch_ops), std::move(new_model)};
}

Expected<void> ApplyPluginWithPartition(CompilerPlugin& compiler_plugin,
                                        LiteRtModelT& model,
                                        PartitionResult partitions,
                                        absl::string_view soc_model,
                                        ApplyPluginsResult& result) {
  auto& dispatch_ops = partitions.first;
  auto& sliced_model = partitions.second;

  // The leading entry subgraphs of the partitions model correspond 1:1 to the
  // dispatch ops; any trailing subgraphs are private region subgraphs.
  if (sliced_model.NumEntrySubgraphs() != dispatch_ops.size()) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Partitions model has %d entry subgraphs but "
                        "there are %d dispatch ops",
                        sliced_model.NumEntrySubgraphs(), dispatch_ops.size()));
  }

  // Pass sliced subgraphs to plugin for compilation.
  auto compiled_result = compiler_plugin.Compile(&sliced_model, soc_model);
  if (!compiled_result) {
    return compiled_result.Error();
  }

  CompiledResult& stored_result = *compiled_result;
  bool has_jit_handles = false;

  // Register byte code buffers as external buffers. Map the byte code indices
  // to the registered buffer ids.
  auto num_byte_code = stored_result.NumByteCodeModules();
  if (!num_byte_code) {
    return num_byte_code.Error();
  }

  std::vector<LiteRtParamIndex> byte_code_idx_to_buf_id(*num_byte_code);

  for (auto i = 0; i < *num_byte_code; ++i) {
    // Check if the compiler yielded a JIT handle
    auto handle_or_error = stored_result.GetHandle(i);
    LiteRtJitExecutableConst exec_handle = nullptr;
    if (handle_or_error.HasValue()) {
      exec_handle = handle_or_error.Value();
    } else if (handle_or_error.Error().Status() !=
               kLiteRtStatusErrorUnsupported) {
      LITERT_LOG(LITERT_WARNING, "Failed to get JIT handle %d: %s", i,
                 handle_or_error.Error().Message().c_str());
    }

    if (exec_handle != nullptr) {
      has_jit_handles = true;
      // If we have a JIT handle, we don't need to register the bytecode buffer.
      // We register an empty buffer to keep indices consistent and satisfy
      // the model's asset attachment requirements.
      OwningBufferRef<uint8_t> empty_byte_code;
      const auto buf_id =
          model.Buffers()->RegisterOwnedBuffer(std::move(empty_byte_code));
      byte_code_idx_to_buf_id[i] = buf_id;
    } else {
      auto byte_code = stored_result.ByteCode(i);
      if (!byte_code) {
        return byte_code.Error();
      }

      OwningBufferRef<uint8_t> owned_byte_code(byte_code->Data(),
                                               byte_code->Size());
      const auto buf_id =
          model.Buffers()->RegisterOwnedBuffer(std::move(owned_byte_code));
      byte_code_idx_to_buf_id[i] = buf_id;
    }
  }

  // Register byte code buffers and add edges from dispatch ops to them.
  for (auto i = 0; i < dispatch_ops.size(); ++i) {
    auto* dispatch_op = dispatch_ops.at(i);

    auto call_info = stored_result.CallInfo(i);
    if (!call_info) {
      return call_info.Error();
    }
    auto [name, byte_code_idx] = *call_info;
    const auto buf_id = byte_code_idx_to_buf_id[byte_code_idx];

    model.AttachAssetToOp(dispatch_op, buf_id, std::string(name));

    // Store the JIT handle if we got one.
    auto handle_or_error = stored_result.GetHandle(byte_code_idx);
    if (handle_or_error.HasValue() && handle_or_error.Value() != nullptr) {
      result.jit_executable_handles[name] = handle_or_error.Value();
    }
  }

  // Bytecode modules have been copied into model-owned buffers and no longer
  // depend on the plugin result. Retain only results that own JIT handles.
  if (has_jit_handles) {
    result.compiled_results.push_back(std::move(*compiled_result));
  }

  // Tag the model with make/model from the plugin.
  auto build_stamp =
      MakeBuildStamp(compiler_plugin.SocManufacturer(), soc_model);
  if (!build_stamp) {
    return build_stamp.Error();
  }

  if (auto status =
          model.PushMetadata(kLiteRtBuildStampKey, std::move(*build_stamp));
      status != kLiteRtStatusOk) {
    return Error(status);
  }

  return {};
}

namespace {

// Groups the input shape overrides carried by `compiler_options` by signature
// key and applies them (plus shape inference) to the corresponding subgraphs.
// Overrides targeting signatures excluded by `selected_signature_keys` (which
// may have been pruned from `model` prior to compilation) are skipped.
// Signatures are processed in first-seen order so error messages are stable.
Expected<bool> ApplyCompilerOptionsInputShapes(
    const LiteRtCompilerOptionsT& compiler_options,
    absl::Span<const std::string> selected_signature_keys,
    LiteRtModelT& model) {
  if (!compiler_options.HasInputShapes()) {
    return false;
  }

  struct PerSignatureShapes {
    std::vector<Dims> positional_inputs;
    std::vector<std::pair<std::string, Dims>> tensor_inputs;
    std::vector<std::pair<std::string, Dims>> signature_inputs;
  };
  std::vector<std::pair<std::string, PerSignatureShapes>> by_signature;
  auto resolve_sig_key =
      [&](absl::string_view raw_sig_key) -> absl::string_view {
    return (raw_sig_key.empty() && !model.Signatures().empty())
               ? model.Signatures()[0]->Key()
               : raw_sig_key;
  };
  auto is_signature_selected = [&](absl::string_view sig_key) -> bool {
    if (selected_signature_keys.empty() || model.Signatures().empty()) {
      return true;
    }
    return absl::c_linear_search(selected_signature_keys, sig_key);
  };
  auto bucket_for = [&](absl::string_view raw_sig_key) -> PerSignatureShapes* {
    absl::string_view sig_key = resolve_sig_key(raw_sig_key);
    if (!is_signature_selected(sig_key)) {
      LITERT_LOG(LITERT_INFO,
                 "Skipping input shape override for unselected signature '%s'.",
                 std::string(sig_key).c_str());
      return nullptr;
    }
    for (auto& [key, bucket] : by_signature) {
      if (key == sig_key) return &bucket;
    }
    return &by_signature
                .emplace_back(std::string(sig_key), PerSignatureShapes{})
                .second;
  };

  for (const auto& e : compiler_options.positional_input_shapes) {
    if (auto* bucket = bucket_for(e.signature_key); bucket != nullptr) {
      bucket->positional_inputs.push_back(e.dims);
    }
  }
  for (const auto& e : compiler_options.tensor_input_shapes) {
    if (auto* bucket = bucket_for(e.signature_key); bucket != nullptr) {
      bucket->tensor_inputs.emplace_back(e.name, e.dims);
    }
  }
  for (const auto& e : compiler_options.signature_input_shapes) {
    if (auto* bucket = bucket_for(e.signature_key); bucket != nullptr) {
      bucket->signature_inputs.emplace_back(e.name, e.dims);
    }
  }

  if (by_signature.empty()) {
    return false;
  }

  std::vector<std::pair<LiteRtTensor, TensorType>> saved_types;
  if (by_signature.size() > 1) {
    for (auto* sg : model.Subgraphs()) {
      if (sg == nullptr) continue;
      for (auto* t : sg->Tensors()) {
        if (t != nullptr) {
          saved_types.emplace_back(t, t->Type());
        }
      }
    }
  }

  ShapeInferenceEngine engine(&model);
  for (const auto& [sig_key, bucket] : by_signature) {
    auto res = engine.ApplyInputShapes(
        sig_key, absl::MakeConstSpan(bucket.positional_inputs),
        absl::MakeConstSpan(bucket.tensor_inputs),
        absl::MakeConstSpan(bucket.signature_inputs));
    if (!res) {
      for (const auto& [t, type] : saved_types) {
        t->SetType(type);
      }
      return res.Error();
    }
    LITERT_LOG(LITERT_INFO, "Applied input shapes for signature '%s'.",
               sig_key.empty() ? "<default>" : sig_key.c_str());
  }
  return true;
}

Expected<void> ApplySinglePlugin(
    CompilerPlugin& compiler_plugin, LiteRtModelT& model,
    absl::string_view soc_model,
    const absl::flat_hash_set<uint32_t>& subgraphs_to_partition,
    ApplyPluginsResult& result) {
  if (auto sdk_version = compiler_plugin.SdkVersion(); sdk_version.HasValue()) {
    if (!sdk_version->empty()) {
      LITERT_LOG(LITERT_INFO, "Applying plugin with SDK version: %s",
                 sdk_version->c_str());
    } else {
      LITERT_LOG(
          LITERT_WARNING,
          "Applying plugin with unknown SDK version, because the "
          "targeted SoC Manufacturer does not report its SDK version. If you "
          "are seeing this message, please be aware that on-device compilation "
          "caching won't be updated if using a different version of the SDK.");
    }
  } else {
    LITERT_LOG(LITERT_WARNING, "Failed to get plugin SDK version: %s",
               sdk_version.Error().Message().c_str());
  }

  // Check compiler compatibility.
  const auto compatibility =
      compiler_plugin.CheckCompilerCompatibility(soc_model);
  if (!compatibility) {
    LITERT_LOG(LITERT_ERROR, "%s", compatibility.Error().Message().c_str());
    return compatibility.Error();
  }

  // Compiler Plugin: Transformation, apply transformations to model.
  auto status = TransformModel(compiler_plugin, model, soc_model);
  if (!status) {
    return status.Error();
  }

  // Compiler Plugin: Partitioning, collect partitions to pass to compilation.
  auto partitions =
      PartitionModel(compiler_plugin, model, soc_model, subgraphs_to_partition);
  if (!partitions) {
    return partitions.Error();
  }

  // Compiler Plugin: Compilation, compile partitions and apply to model.
  return ApplyPluginWithPartition(compiler_plugin, model,
                                  std::move(*partitions), soc_model, result);
}

}  // namespace

Expected<bool> ApplyCompilerOptionsInputShapes(
    const CompilerPlugin& compiler_plugin, LiteRtModelT& model) {
  auto compiler_options = compiler_plugin.CompilerOptions();
  if (!compiler_options.HasValue()) {
    if (compiler_options.Error().Status() == kLiteRtStatusErrorNotFound) {
      return false;
    }
    return compiler_options.Error();
  }
  return ApplyCompilerOptionsInputShapes(
      compiler_options.Value(), compiler_plugin.SelectedSignatureKeys(), model);
}

Expected<void> TransformModel(CompilerPlugin& compiler_plugin,
                              LiteRtModelT& model,
                              absl::string_view soc_model) {
  auto status = compiler_plugin.RegisterAllTransformations();
  if (!status) {
    return status;
  }
  LITERT_LOG(LITERT_INFO, "Registered %d transformations.",
             compiler_plugin.GetNumTransformations());

  status = compiler_plugin.GreedyPatternMatchAndRewrite(model);

  if (!status) {
    return status;
  }
  LITERT_LOG(LITERT_INFO, "Applied transformations.");
  return {};
}

Expected<void> ApplyPlugin(
    CompilerPlugin& compiler_plugin, LiteRtModelT& model,
    absl::string_view soc_model,
    const absl::flat_hash_set<uint32_t>& subgraphs_to_partition,
    ApplyPluginsResult& result) {
  LITERT_RETURN_IF_ERROR(
      ApplyCompilerOptionsInputShapes(compiler_plugin, model));
  return ApplySinglePlugin(compiler_plugin, model, soc_model,
                           subgraphs_to_partition, result);
}

Expected<ApplyPluginsResult> ApplyPlugins(
    LiteRtEnvironment environment, LiteRtOptions options, LiteRtModel model,
    LiteRtHwAcceleratorSet selected_hw_accelerators, bool* mutated) {
  auto option =
      environment->GetOption(kLiteRtEnvOptionTagCompilerPluginLibraryDir);
  if (!option.has_value() || option->type != kLiteRtAnyTypeString) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Compiler plugin is not configured");
  }
  std::string compiler_plugin_lib_path = option->str_value;

  const std::array<const absl::string_view, 1>
      compiler_plugin_lib_search_paths = {compiler_plugin_lib_path};

  auto compiler_plugins = litert::internal::CompilerPlugin::LoadPlugins(
      compiler_plugin_lib_search_paths, &environment->GetOptions(), options);
  if (!compiler_plugins) {
    return compiler_plugins.Error();
  }
  LITERT_ASSIGN_OR_RETURN(auto result,
                          ApplyPlugins(model, selected_hw_accelerators,
                                       compiler_plugins.Value(), mutated));
  result.compiler_plugins = std::move(*compiler_plugins);
  return result;
}

Expected<ApplyPluginsResult> ApplyPlugins(
    LiteRtModel model, LiteRtHwAcceleratorSet selected_hw_accelerators,
    std::vector<CompilerPlugin>& compiler_plugins, bool* mutated) {
  if (compiler_plugins.empty()) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "No compiler plugin found");
  }

  bool has_matching_plugin = false;
  for (auto& compiler_plugin : compiler_plugins) {
    auto supported_hw = compiler_plugin.SupportedHardware();
    if (supported_hw && (*supported_hw & selected_hw_accelerators)) {
      has_matching_plugin = true;
      break;
    }
  }
  if (has_matching_plugin) {
    LITERT_ASSIGN_OR_RETURN(
        bool shapes_applied,
        ApplyCompilerOptionsInputShapes(compiler_plugins.front(), *model));
    if (shapes_applied && mutated != nullptr) {
      *mutated = true;
    }
  }

  std::vector<std::string> success_messages;
  std::vector<std::string> error_messages;

  ApplyPluginsResult result;
  result.num_applied_plugins = 0;
  for (auto& compiler_plugin : compiler_plugins) {
    auto plugin_name = compiler_plugin.DebugString();

    auto plugin_supported_hardware = compiler_plugin.SupportedHardware();
    if (!plugin_supported_hardware) {
      error_messages.push_back(absl::StrCat(
          plugin_name, " ", plugin_supported_hardware.Error().Message()));
      continue;
    }

    if (*plugin_supported_hardware & selected_hw_accelerators) {
      auto status = ApplySinglePlugin(compiler_plugin, *model, "", {}, result);

      if (!status) {
        error_messages.push_back(
            absl::StrCat(plugin_name, " ", status.Error().Message()));
        continue;
      }

      if (mutated != nullptr) {
        *mutated = true;
      }

      success_messages.push_back(absl::StrCat(plugin_name));
      result.num_applied_plugins++;
    }
  }

  result.success_message = absl::StrJoin(success_messages, ", ");
  result.error_message = absl::StrJoin(error_messages, ", ");

  return result;
}

Expected<LiteRtCompilerOptionsT> CompilerPlugin::CompilerOptions() const {
  LiteRtCompilerOptionsT result;
  if (options_ == nullptr) {
    return litert::Error(kLiteRtStatusErrorNotFound,
                         "No compiler options found");
  }
  LiteRtOpaqueOptions opaque_options;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptions(options_, &opaque_options));
  if (opaque_options == nullptr) {
    return result;
  }

  // Parse compiler options from the opaque options.
  void* data;
  if (auto status = LiteRtFindOpaqueOptionsData(
          opaque_options, LiteRtCompilerOptionsT::Identifier(), &data);
      status != kLiteRtStatusOk) {
    if (status == kLiteRtStatusErrorNotFound) {
      return result;
    }
    return Unexpected(status);
  }
  std::string options_str(static_cast<const char*>(data));
  if (auto status = litert::internal::ParseLiteRtCompilerOptions(
          options_str.data(), options_str.size(), &result);
      status != kLiteRtStatusOk) {
    return Unexpected(status);
  }
  return result;
}

absl::Span<const std::string> CompilerPlugin::SelectedSignatureKeys() const {
  if (options_ == nullptr) {
    return {};
  }
  return options_->selected_signature_keys;
}

Expected<CompilerPlugin> CompilerPlugin::FindPlugin(
    absl::string_view soc_manufacturer,
    absl::Span<const absl::string_view> lib_search_paths,
    LiteRtEnvironmentOptions env, LiteRtOptions options) {
  LITERT_ASSIGN_OR_RETURN(auto plugins, CompilerPlugin::LoadPlugins(
                                            lib_search_paths, env, options));
  LITERT_LOG(LITERT_INFO, "Found %d plugin(s)", plugins.size());
  if (plugins.size() == 1 && soc_manufacturer.empty()) {
    // Don't check soc string if there is only one plugin, for compatibility
    // with tooling and upstreams.
    return std::move(plugins.front());
  }
  for (auto& plugin : plugins) {
    if (plugin.SocManufacturer() == soc_manufacturer) {
      return std::move(plugin);
    }
  }
  return Error(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("No compiler plugin found for soc manufacturer %s",
                      soc_manufacturer));
}

Expected<bool> CompilerPlugin::CheckCompilerCompatibility(
    absl::string_view soc_model) {
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          check_compiler_compatibility)) {
    return true;
  }
  const char* soc_model_name = nullptr;
  if (!soc_model.empty()) {
    soc_model_name = soc_model.data();
  }
  LiteRtApiVersion host_version{LITERT_API_VERSION_MAJOR,
                                LITERT_API_VERSION_MINOR,
                                LITERT_API_VERSION_PATCH};
  LITERT_RETURN_IF_ERROR(plugin_api_v1_->check_compiler_compatibility(
      host_version, plugin_handle_, env_, options_, soc_model_name));
  return true;
}
}  // namespace litert::internal
