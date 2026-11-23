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
#include <numeric>
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
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_op_code.h"
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
#include "litert/core/model/subgraph_call_graph.h"
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

// Returns a null-terminated pointer for `value` (backed by `storage`), or
// nullptr when `value` is empty.
const char* OptionalCStr(absl::string_view value, std::string& storage) {
  if (value.empty()) {
    return nullptr;
  }
  storage.assign(value.data(), value.size());
  return storage.c_str();
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
    version_str = absl::StrCat(version->major, ".", version->minor, ".",
                               version->patch);
  }
  return absl::StrCat(SocManufacturer(), " compiler plugin (ver ", version_str,
                      ")");
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

bool CompilerPlugin::AcceptsHelperSubgraphs() const {
  return LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                            compiler_plugin_compile_v2);
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
            absl::StrCat("Transformation ", i,
                         " has a invalid pattern function."));
      }
      if (transformations[i].name == nullptr) {
        return Unexpected(
            kLiteRtStatusInvalidTransformation,
            absl::StrCat("Transformation ", i,
                         " has a invalid transformation name."));
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
    LiteRtModelT& model, LiteRtParamIndex subgraph_index,
    absl::string_view soc_model) {
  if (subgraph_index >= model.NumSubgraphs()) {
    return Unexpected(kLiteRtStatusErrorIndexOOB,
                      "Subgraph index out of range for Partition");
  }
  if (LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                         compiler_plugin_partition_v2)) {
    LiteRtOpListT ops;
    std::string soc_model_storage;
    const char* soc_model_str = OptionalCStr(soc_model, soc_model_storage);
    LITERT_PERFETTO_TRACE_EVENT("CompilerPlugin Partition");
    LITERT_RETURN_IF_ERROR(plugin_api_v1_->compiler_plugin_partition_v2(
        plugin_handle_, soc_model_str, &model, subgraph_index, &ops));
    return ops.Values();
  }
  return Partition(&model.Subgraph(subgraph_index), soc_model);
}

Expected<std::vector<LiteRtOpWithPartitionIndex>> CompilerPlugin::Partition(
    LiteRtSubgraph subgraph, absl::string_view soc_model) {
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          compiler_plugin_partition)) {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "compiler_plugin_partition not supported");
  }
  LiteRtOpListT ops;
  std::string soc_model_storage;
  const char* soc_model_str = OptionalCStr(soc_model, soc_model_storage);
  LITERT_PERFETTO_TRACE_EVENT("CompilerPlugin Partition");
  LITERT_RETURN_IF_ERROR(plugin_api_v1_->compiler_plugin_partition(
      plugin_handle_, soc_model_str, subgraph, &ops));
  return ops.Values();
}

Expected<CompiledResult> CompilerPlugin::Compile(
    LiteRtModel partitions, LiteRtParamIndex num_entry_subgraphs,
    absl::string_view soc_model) {
  const size_t num_subgraphs = partitions->NumSubgraphs();
  if (num_entry_subgraphs > num_subgraphs) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "num_entry_subgraphs exceeds number of subgraphs");
  }
  CompiledResult result = MakeResult();
  if (num_entry_subgraphs == 0) {
    LITERT_LOG(LITERT_INFO,
               "CompilerPlugin::Compile: Skipped for non partitioned model.");
    return result;
  }
  // If the user has passed an soc_model, then we use it; otherwise we let the
  // backend pick the appropriate one by passing nullptr as soc_model. This is
  // important for on-device compilation, where the backend must determine the
  // SoC model based on the user device.
  std::string soc_model_storage;
  const char* soc_model_str = OptionalCStr(soc_model, soc_model_storage);
  if (AcceptsHelperSubgraphs()) {
    LITERT_PERFETTO_TRACE_EVENT("CompilerPlugin Compile");
    LITERT_RETURN_IF_ERROR(plugin_api_v1_->compiler_plugin_compile_v2(
        plugin_handle_, soc_model_str, partitions, num_entry_subgraphs,
        &result.compiled_result_handle_));
    return result;
  }
  if (!LITERT_ABI_HAS_API(plugin_api_v1_, /*req_major=*/1,
                          compiler_plugin_compile)) {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "compiler_plugin_compile not supported");
  }
  // Adapt `partitions` for a v1 plugin: clear composite decompositions in the
  // entry subgraphs and discard all helper subgraphs. Validate first so that
  // nothing is mutated on failure.
  for (LiteRtParamIndex i = 0; i < num_entry_subgraphs; ++i) {
    for (const LiteRtOpT* op : partitions->Subgraph(i).Ops()) {
      if (CompositeDecompositionIndex(*op).has_value()) {
        continue;
      }
      for (const int32_t ref : SubgraphRefs(*op)) {
        if (static_cast<size_t>(ref) >= num_entry_subgraphs) {
          return Unexpected(kLiteRtStatusErrorInvalidGraphInvariant,
                            "v1 compiler plugins cannot take ops that "
                            "reference helper subgraphs");
        }
      }
    }
  }
  for (LiteRtParamIndex i = 0; i < num_entry_subgraphs; ++i) {
    for (LiteRtOp op : partitions->Subgraph(i).Ops()) {
      ClearCompositeDecomposition(*op);
    }
  }
  if (num_entry_subgraphs < num_subgraphs) {
    std::vector<size_t> helper_indices(num_subgraphs - num_entry_subgraphs);
    std::iota(helper_indices.begin(), helper_indices.end(),
              num_entry_subgraphs);
    LITERT_RETURN_IF_ERROR(
        partitions->DiscardSubgraphs(std::move(helper_indices)));
  }
  LITERT_PERFETTO_TRACE_EVENT("CompilerPlugin Compile");
  LITERT_RETURN_IF_ERROR(plugin_api_v1_->compiler_plugin_compile(
      plugin_handle_, soc_model_str, partitions,
      &result.compiled_result_handle_));
  return result;
}

Expected<CompiledResult> CompilerPlugin::Compile(LiteRtModel partitions,
                                                 absl::string_view soc_model) {
  return Compile(partitions, partitions->NumSubgraphs(), soc_model);
}

namespace {

LiteRtStatus PartitionSubgraph(
    std::vector<LiteRtOpWithPartitionIndex> selected_ops,
    LiteRtSubgraphT& subgraph, std::vector<LiteRtOp>& res_ops,
    LiteRtModelT& model,
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
  }

  return kLiteRtStatusOk;
}

// A set of subgraphs, keyed by pointer (stable while subgraphs are added or
// reordered).
using SubgraphSet = absl::flat_hash_set<LiteRtSubgraph>;

// True for control flow ops, whose CPU kernels invoke each referenced subgraph
// as a whole graph, so ops inside may be offloaded. Other region ops (REDUCE,
// SORT, ...) interpret their subgraphs, which must stay unchanged.
bool IsControlFlowOp(LiteRtOpCode code) {
  switch (code) {
    case kLiteRtOpCodeTflWhile:
    case kLiteRtOpCodeTflIf:
    case kLiteRtOpCodeTflCallOnce:
    case kLiteRtOpCodeShloWhile:
    case kLiteRtOpCodeShloCase:
      return true;
    default:
      return false;
  }
}

void MarkModified(bool* model_modified) {
  if (model_modified != nullptr) {
    *model_modified = true;
  }
}

enum class CompositeKind { kNone, kNpuCall, kCpuCall, kStandard };

CompositeKind GetCompositeKind(LiteRtOp op) {
  auto info = GetOptionsAs<CompositeOptions>(op);
  if (!info) {
    return CompositeKind::kNone;
  }
  if (info->name == CompositeOptions::kNpuCall) {
    return CompositeKind::kNpuCall;
  }
  if (info->name == CompositeOptions::kCpuCall) {
    return CompositeKind::kCpuCall;
  }
  return CompositeKind::kStandard;
}

int32_t IndexOf(const LiteRtModelT& model, const LiteRtSubgraphT* subgraph) {
  const auto subgraphs = model.Subgraphs();
  const auto it = absl::c_find(subgraphs, subgraph);
  ABSL_DCHECK(it != subgraphs.end());
  return static_cast<int32_t>(it - subgraphs.begin());
}

std::string CloneName(const LiteRtSubgraphT& origin, int32_t origin_index,
                      absl::string_view role, int32_t clone_index) {
  const std::string base = origin.Name().empty()
                               ? absl::StrCat("subgraph_", origin_index)
                               : std::string(origin.Name());
  return absl::StrCat(base, "_", role, "_clone", clone_index);
}

// Adds the subgraphs at `refs`, and every subgraph they transitively
// reference, to `pinned`.
void PinClosure(const LiteRtModelT& model, absl::Span<const int32_t> refs,
                SubgraphSet& pinned) {
  std::vector<int32_t> pending(refs.begin(), refs.end());
  while (!pending.empty()) {
    LiteRtSubgraph subgraph = model.Subgraphs()[pending.back()];
    pending.pop_back();
    if (!pinned.insert(subgraph).second) {
      continue;
    }
    for (const LiteRtOpT* op : subgraph->Ops()) {
      for (const int32_t ref : SubgraphRefs(*op)) {
        pending.push_back(ref);
      }
    }
  }
}

// Fails if `roots`, or a subgraph that ships with them, has an op the plugin
// must not receive: a region op (plugin without `compiler_plugin_compile_v2`)
// or an odml.cpu_call that cannot be inlined. Follows cpu_call bodies and, when
// `accepts_helper_subgraphs` is true, every subgraph reference.
Expected<void> CheckShippedSubgraphs(const LiteRtModelT& model,
                                     absl::Span<const LiteRtSubgraph> roots,
                                     bool accepts_helper_subgraphs) {
  SubgraphSet visited;
  std::vector<LiteRtSubgraph> pending(roots.begin(), roots.end());
  while (!pending.empty()) {
    LiteRtSubgraph subgraph = pending.back();
    pending.pop_back();
    if (!visited.insert(subgraph).second) {
      continue;
    }
    for (LiteRtOp op : subgraph->Ops()) {
      if (!accepts_helper_subgraphs && IsRegionOp(*op)) {
        return Unexpected(
            kLiteRtStatusErrorUnsupported,
            absl::StrCat("A subgraph passed to the compiler plugin contains an "
                         "op (code ",
                         op->OpCode(),
                         ") that references other subgraphs, but the plugin "
                         "does not implement compiler_plugin_compile_v2"));
      }
      if (GetCompositeKind(op) == CompositeKind::kCpuCall) {
        LiteRtSubgraph body = CompositeDecomposition(model, *op);
        ABSL_DCHECK(body != nullptr);  // Validated before the walk.
        if (auto inlinable = CheckInlinable(*op, *body); !inlinable) {
          return Unexpected(
              kLiteRtStatusErrorUnsupported,
              absl::StrCat("An odml.cpu_call in a subgraph passed to the "
                           "compiler plugin cannot be inlined: ",
                           inlinable.Error().Message()));
        }
        pending.push_back(body);
        continue;
      }
      if (accepts_helper_subgraphs) {
        for (const int32_t ref : SubgraphRefs(*op)) {
          pending.push_back(model.Subgraphs()[ref]);
        }
      }
    }
  }
  return {};
}

// Validates that every selected op belongs to `subgraph`, then removes (and
// logs) selections the partitioner handles itself (npu_call and cpu_call
// composites), ops without tensors, region ops when the plugin does not
// implement `compiler_plugin_compile_v2`, and (for v2 plugins) ops whose
// shipped subgraph closure contains an uninlinable odml.cpu_call.
Expected<void> FilterSelection(
    const LiteRtModelT& model, const LiteRtSubgraphT& subgraph,
    std::vector<LiteRtOpWithPartitionIndex>& selected,
    bool accepts_helper_subgraphs, int32_t subgraph_index) {
  const absl::flat_hash_set<LiteRtOp> subgraph_ops(subgraph.Ops().begin(),
                                                   subgraph.Ops().end());
  for (const auto& [op, partition_index] : selected) {
    if (!subgraph_ops.contains(op)) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrCat("Compiler plugin selected an op that does not belong to "
                       "subgraph<",
                       subgraph_index, ">"));
    }
  }
  std::vector<std::string> dropped_op_codes;
  size_t num_dropped_tensorless = 0;
  selected.erase(
      std::remove_if(selected.begin(), selected.end(),
                     [&](const LiteRtOpWithPartitionIndex& selection) {
                       LiteRtOp op = selection.first;
                       const auto kind = GetCompositeKind(op);
                       if (kind == CompositeKind::kNpuCall ||
                           kind == CompositeKind::kCpuCall) {
                         return true;
                       }
                       if (!accepts_helper_subgraphs && IsRegionOp(*op)) {
                         dropped_op_codes.push_back(absl::StrCat(op->OpCode()));
                         return true;
                       }
                       // An op without tensors (e.g. a CALL_ONCE) cannot be
                       // connected to a dispatch op: as a partition of its
                       // own, the dispatch op would have no edges and would
                       // be removed by DCE.
                       if (op->Inputs().empty() && op->Outputs().empty()) {
                         num_dropped_tensorless++;
                         return true;
                       }
                       // Only v2 plugins ship referenced helper subgraphs.
                       if (accepts_helper_subgraphs) {
                         const auto refs = SubgraphRefs(*op);
                         if (!refs.empty()) {
                           std::vector<LiteRtSubgraph> callees;
                           callees.reserve(refs.size());
                           for (const int32_t ref : refs) {
                             callees.push_back(model.Subgraphs()[ref]);
                           }
                           if (auto status = CheckShippedSubgraphs(
                                   model, callees,
                                   /*accepts_helper_subgraphs=*/true);
                               !status) {
                             LITERT_LOG(LITERT_WARNING,
                                        "Keeping op (code %d) in subgraph<%d> "
                                        "on CPU: %s",
                                        op->OpCode(), subgraph_index,
                                        status.Error().Message().c_str());
                             return true;
                           }
                         }
                       }
                       return false;
                     }),
      selected.end());
  if (num_dropped_tensorless > 0) {
    LITERT_LOG(LITERT_WARNING,
               "Plugin selected %zu op(s) without inputs or outputs in "
               "subgraph<%d>. Keeping them on CPU.",
               num_dropped_tensorless, subgraph_index);
  }
  if (!dropped_op_codes.empty()) {
    LITERT_LOG(LITERT_WARNING,
               "Plugin selected %zu op(s) in subgraph<%d> that reference other "
               "subgraphs (op codes: %s) but does not implement "
               "compiler_plugin_compile_v2. Keeping them on CPU, which splits "
               "the surrounding partitions.",
               dropped_op_codes.size(), subgraph_index,
               absl::StrJoin(dropped_op_codes, ", ").c_str());
  }
  return {};
}

// `subgraph` must be partitioned for its CPU callers (signatures, the main
// subgraph and the ops in `cpu_callers`) but must also stay unchanged for its
// other callers. Gives those other callers a copy, so `subgraph` itself can be
// partitioned in place.
Expected<void> SplitPinnedSubgraph(LiteRtModelT& model,
                                   LiteRtSubgraphT& subgraph,
                                   absl::Span<const LiteRtOp> cpu_callers,
                                   SubgraphSet& pinned) {
  const int32_t index = IndexOf(model, &subgraph);
  const int32_t clone_index = static_cast<int32_t>(model.NumSubgraphs());
  LITERT_ASSIGN_OR_RETURN(
      LiteRtSubgraph clone,
      CloneSubgraph(model, subgraph,
                    CloneName(subgraph, index, "pinned", clone_index)));
  const absl::flat_hash_set<LiteRtOp> keep(cpu_callers.begin(),
                                           cpu_callers.end());
  for (LiteRtSubgraph caller : model.Subgraphs()) {
    for (LiteRtOp op : caller->Ops()) {
      if (!keep.contains(op)) {
        RetargetSubgraphRefs(*op, index, clone_index);
      }
    }
  }
  // The callees of `subgraph` stay pinned: the clone still references them.
  pinned.erase(&subgraph);
  pinned.insert(clone);
  return {};
}

struct PartitionWalkState {
  bool accepts_helper_subgraphs = false;
  bool* model_modified = nullptr;
  LiteRtCompilerOptionsPartitionStrategy strategy =
      kLiteRtCompilerOptionsPartitionStrategyDefault;
  size_t max_partitions = 0;
  SubgraphSet pinned;
  std::vector<LiteRtOp> dispatch_ops;
  std::vector<LiteRtSubgraph> outlined;
  // Subgraphs passed to the plugin's partition function, in walk order.
  std::vector<LiteRtSubgraph> processed;
};

// Runs the plugin on `subgraph` and outlines the selected ops.
Expected<void> PartitionOneSubgraph(CompilerPlugin& compiler_plugin,
                                    LiteRtModelT& model,
                                    absl::string_view soc_model,
                                    LiteRtSubgraphT& subgraph,
                                    int32_t subgraph_index,
                                    PartitionWalkState& state) {
  ABSL_DCHECK_EQ(subgraph_index, IndexOf(model, &subgraph));
  LITERT_ASSIGN_OR_RETURN(
      auto selected,
      compiler_plugin.Partition(model, subgraph_index, soc_model));
  // Filtered before inlining too: a bad selection fails before any mutation,
  // and dropped standard composites are inlined.
  LITERT_RETURN_IF_ERROR(FilterSelection(model, subgraph, selected,
                                         state.accepts_helper_subgraphs,
                                         subgraph_index));

  // Inline standard composites the plugin did not select, so that the plugin
  // can pick ops from their decompositions instead.
  std::vector<LiteRtOp> ops_to_inline;
  for (LiteRtOp op : subgraph.Ops()) {
    if (GetCompositeKind(op) != CompositeKind::kStandard) {
      continue;
    }
    const bool is_selected = absl::c_any_of(
        selected,
        [op](const auto& selection) { return selection.first == op; });
    if (!is_selected) {
      ops_to_inline.push_back(op);
    }
  }
  size_t num_inlined = 0;
  for (LiteRtOp op : ops_to_inline) {
    LiteRtSubgraph decomposition = CompositeDecomposition(model, *op);
    if (decomposition == nullptr) {
      continue;
    }
    MarkModified(state.model_modified);
    if (auto status = InlineSubgraph(model, *op, decomposition); !status) {
      const auto info = GetOptionsAs<CompositeOptions>(op);
      LITERT_LOG(LITERT_ERROR,
                 "Failed to inline composite op '%s' (subgraph %d): %s",
                 std::string(info->name).c_str(),
                 IndexOf(model, decomposition),
                 status.Error().Message().c_str());
      continue;
    }
    ++num_inlined;
  }
  if (num_inlined > 0) {
    LITERT_LOG(LITERT_INFO, "Inlined %zu composite ops into subgraph<%d>",
               num_inlined, subgraph_index);
    LITERT_ASSIGN_OR_RETURN(
        selected, compiler_plugin.Partition(model, subgraph_index, soc_model));
    LITERT_RETURN_IF_ERROR(FilterSelection(model, subgraph, selected,
                                           state.accepts_helper_subgraphs,
                                           subgraph_index));
  }

  // For v2 plugins, pin the closure of every selected op's subgraph references
  // (region ops and selected standard composites) before outlining moves them.
  // For v1 plugins, region ops were dropped above and composite decompositions
  // are cleared before packaging rather than shipped as helpers.
  if (state.accepts_helper_subgraphs) {
    for (const auto& [op, partition_index] : selected) {
      const auto refs = SubgraphRefs(*op);
      if (!refs.empty()) {
        PinClosure(model, refs, state.pinned);
      }
    }
  }

  const size_t num_selected_ops = selected.size();
  const size_t num_ops = subgraph.Ops().size();
  const size_t num_dispatch_ops = state.dispatch_ops.size();
  const size_t num_subgraphs = model.NumSubgraphs();
  if (!selected.empty()) {
    MarkModified(state.model_modified);
  }
  LITERT_RETURN_IF_ERROR(
      PartitionSubgraph(std::move(selected), subgraph, state.dispatch_ops,
                        model, state.strategy, state.max_partitions));
  for (size_t i = num_subgraphs; i < model.NumSubgraphs(); ++i) {
    state.outlined.push_back(model.Subgraphs()[i]);
  }
  LITERT_LOG(LITERT_INFO,
             "Partitioned subgraph<%d>, selected %zu ops, from a total of %zu "
             "ops. resulted in %zu partitions.",
             subgraph_index, num_selected_ops, num_ops,
             state.dispatch_ops.size() - num_dispatch_ops);
  return {};
}

std::vector<int32_t> IndicesOf(const LiteRtModelT& model,
                               absl::Span<const LiteRtSubgraph> subgraphs) {
  std::vector<int32_t> indices;
  indices.reserve(subgraphs.size());
  for (LiteRtSubgraph subgraph : subgraphs) {
    indices.push_back(IndexOf(model, subgraph));
  }
  return indices;
}

// Inlines every odml.cpu_call in `host`, including cpu_calls that inlining
// brings into `host` from nested bodies. No other subgraph is modified (the
// bodies are copied, not changed). A cpu_call that cannot be inlined is an
// error if `must_succeed`; otherwise it stays a composite, which the CPU runs
// by executing its body.
Expected<void> InlineCpuCallsInto(LiteRtModelT& model, LiteRtSubgraphT& host,
                                  bool must_succeed,
                                  bool* model_modified = nullptr) {
  absl::flat_hash_set<LiteRtOp> kept;
  while (true) {
    LiteRtOp cpu_call = nullptr;
    for (LiteRtOp op : host.Ops()) {
      if (GetCompositeKind(op) == CompositeKind::kCpuCall &&
          !kept.contains(op)) {
        cpu_call = op;
        break;
      }
    }
    if (cpu_call == nullptr) {
      return {};
    }
    LiteRtSubgraph body = CompositeDecomposition(model, *cpu_call);
    ABSL_DCHECK(body != nullptr);  // Validated before the walk.
    MarkModified(model_modified);
    auto status = InlineSubgraph(model, *cpu_call, body);
    if (status) {
      continue;
    }
    if (must_succeed) {
      return Unexpected(
          kLiteRtStatusErrorInvalidGraphInvariant,
          absl::StrCat("Cannot inline an odml.cpu_call in a subgraph passed "
                       "to the compiler plugin: ",
                       status.Error().Message()));
    }
    LITERT_LOG(LITERT_WARNING, "Keeping an odml.cpu_call on CPU: %s",
               status.Error().Message().c_str());
    kept.insert(cpu_call);
  }
}

// True if an op other than `except`, in a subgraph whose index satisfies
// `in_scope`, references subgraph `index`.
template <typename InScope>
bool IsReferenced(const LiteRtModelT& model, int32_t index, InScope in_scope,
                  const LiteRtOpT* except = nullptr) {
  for (size_t i = 0; i < model.NumSubgraphs(); ++i) {
    if (!in_scope(i)) {
      continue;
    }
    for (const LiteRtOpT* op : model.Subgraph(i).Ops()) {
      if (op != except && absl::c_linear_search(SubgraphRefs(*op), index)) {
        return true;
      }
    }
  }
  return false;
}

// Returns the subgraph the npu_call `op` compiles. Entry subgraphs are moved
// into the partitions model and may be modified there, so each npu_call gets
// a decomposition no other live op or signature uses; a shared one is cloned
// for `op`. (An earlier npu_call never keeps a decomposition `op` also uses.)
Expected<LiteRtSubgraph> ExclusiveNpuCallDecomposition(
    LiteRtModelT& model, LiteRtOpT& op,
    const absl::flat_hash_set<int32_t>& live, const SubgraphSet& cpu_roots) {
  LiteRtSubgraph decomposition = CompositeDecomposition(model, op);
  ABSL_DCHECK(decomposition != nullptr);  // Validated before the walk.
  const int32_t index = IndexOf(model, decomposition);
  const auto is_live = [&](size_t i) {
    return live.contains(static_cast<int32_t>(i));
  };
  if (!cpu_roots.contains(decomposition) &&
      !IsReferenced(model, index, is_live, &op)) {
    return decomposition;
  }
  const int32_t clone_index = static_cast<int32_t>(model.NumSubgraphs());
  LITERT_LOG(LITERT_WARNING,
             "odml.npu_call decomposition subgraph<%d> is also used elsewhere; "
             "compiling a copy (its byte code is duplicated)",
             index);
  LITERT_ASSIGN_OR_RETURN(
      LiteRtSubgraph clone,
      CloneSubgraph(model, *decomposition,
                    CloneName(*decomposition, index, "npu_call", clone_index)));
  RetargetSubgraphRefs(op, index, clone_index);
  return clone;
}

// Gives every helper subgraph in `partitions` a non-empty name and makes all
// non-empty names unique. Region references are resolved by subgraph name when
// the model is imported to MLIR, so a helper with an empty or duplicate name
// breaks them. Entry subgraphs (`[0, num_entry_subgraphs)`) are only referenced
// by their dispatch ops and keep their names, so an empty entry name still
// imports as `fn_<index>`.
void NameSubgraphs(LiteRtModelT& partitions, size_t num_entry_subgraphs) {
  absl::flat_hash_set<std::string> used;
  for (size_t i = 0; i < partitions.NumSubgraphs(); ++i) {
    auto& subgraph = partitions.Subgraph(i);
    std::string name(subgraph.Name());
    if (name.empty()) {
      if (i < num_entry_subgraphs) {
        continue;
      }
      name = absl::StrCat("region_", i);
    }
    std::string unique_name = name;
    for (int suffix = 1; !used.insert(unique_name).second; ++suffix) {
      unique_name = absl::StrCat(name, "_", suffix);
    }
    subgraph.SetName(std::move(unique_name));
  }
}

// Clears the decompositions of the standard and npu_call composites in
// `subgraphs`, so that a v1 plugin compiles them as opaque ops and receives no
// helper subgraphs. cpu_call bodies are kept: they are inlined.
void ClearDecompositionsForV1(absl::Span<const LiteRtSubgraph> subgraphs) {
  for (LiteRtSubgraph subgraph : subgraphs) {
    for (LiteRtOp op : subgraph->Ops()) {
      const auto kind = GetCompositeKind(op);
      if (kind == CompositeKind::kStandard || kind == CompositeKind::kNpuCall) {
        ClearCompositeDecomposition(*op);
      }
    }
  }
}

// Moves `entries` and the subgraphs they transitively reference into a new
// model, laid out as `[entries..., helpers...]`. Referenced subgraphs that are
// still used on CPU are cloned. Model mutations here only happen when
// `entries` is non-empty, which means the model was already marked modified.
Expected<LiteRtModelT> PackagePartitions(
    LiteRtModelT& model, const std::vector<LiteRtSubgraph>& cpu_roots,
    const SubgraphSet& cpu_root_set, const std::vector<LiteRtSubgraph>& entries,
    bool accepts_helper_subgraphs) {
  if (!accepts_helper_subgraphs) {
    ClearDecompositionsForV1(entries);
  }
  // Drop subgraphs that neither run on CPU nor are needed by the entries, so
  // that every remaining caller outside the bundle is live.
  std::vector<LiteRtSubgraph> roots = cpu_roots;
  roots.insert(roots.end(), entries.begin(), entries.end());
  LITERT_RETURN_IF_ERROR(PruneUnreachableSubgraphs(model, roots));

  LITERT_ASSIGN_OR_RETURN(auto graph, BuildSubgraphCallGraph(model));
  const auto closure = SubgraphClosure(graph, IndicesOf(model, entries));
  SubgraphSet bundle(entries.begin(), entries.end());
  std::vector<LiteRtSubgraph> helpers;
  for (const int32_t index : TopologicalOrder(graph)) {
    LiteRtSubgraph subgraph = model.Subgraphs()[index];
    if (closure.contains(index) && !bundle.contains(subgraph)) {
      helpers.push_back(subgraph);
    }
  }
  bundle.insert(helpers.begin(), helpers.end());

  // Visit callers before callees: cloning a helper leaves the original on the
  // CPU side, which makes it an outside caller of the helper's callees.
  for (auto& helper : helpers) {
    const int32_t index = IndexOf(model, helper);
    const auto outside = [&](size_t i) {
      return !bundle.contains(model.Subgraphs()[i]);
    };
    if (!cpu_root_set.contains(helper) &&
        !IsReferenced(model, index, outside)) {
      continue;
    }
    const int32_t clone_index = static_cast<int32_t>(model.NumSubgraphs());
    LITERT_ASSIGN_OR_RETURN(
        LiteRtSubgraph clone,
        CloneSubgraph(model, *helper,
                      CloneName(*helper, index, "npu", clone_index)));
    for (LiteRtSubgraph subgraph : bundle) {
      for (LiteRtOp op : subgraph->Ops()) {
        RetargetSubgraphRefs(*op, index, clone_index);
      }
    }
    bundle.erase(helper);
    bundle.insert(clone);
    helper = clone;
  }

  std::vector<LiteRtSubgraph> packaged = entries;
  packaged.insert(packaged.end(), helpers.begin(), helpers.end());
  const std::vector<int32_t> indices = IndicesOf(model, packaged);
  LITERT_ASSIGN_OR_RETURN(
      auto partitions,
      model.Yank(std::vector<size_t>(indices.begin(), indices.end())));

  // Stage 2: Inline all odml.cpu_call ops inside the NPU partitions model
  // (entries and helpers) in one pass, then prune the inlined cpu_call bodies.
  // Every cpu_call here was validated up front or during selection filtering,
  // so inlining must succeed.
  if (!helpers.empty()) {
    for (LiteRtSubgraph subgraph : partitions.Subgraphs()) {
      LITERT_RETURN_IF_ERROR(
          InlineCpuCallsInto(partitions, *subgraph, /*must_succeed=*/true));
    }
    const auto partition_entries =
        partitions.Subgraphs().subspan(0, entries.size());
    if (!accepts_helper_subgraphs) {
      // Inlined cpu_call bodies may have brought in more composites.
      ClearDecompositionsForV1(partition_entries);
    }
    LITERT_RETURN_IF_ERROR(PruneUnreachableSubgraphs(
        partitions, std::vector<LiteRtSubgraph>(partition_entries.begin(),
                                                partition_entries.end())));
  }
  ABSL_DCHECK(accepts_helper_subgraphs ||
              partitions.NumSubgraphs() == entries.size());
  if (partitions.NumSubgraphs() > entries.size()) {
    NameSubgraphs(partitions, entries.size());
  }
  return partitions;
}

// The CPU roots of the model, in order, and as a set.
struct CpuRoots {
  std::vector<LiteRtSubgraph> ordered;
  SubgraphSet set;

  void Add(LiteRtSubgraph subgraph) {
    if (set.insert(subgraph).second) {
      ordered.push_back(subgraph);
    }
  }
};

// CPU roots: signatures, the main subgraph, and any other subgraph without
// callers that is not a composite decomposition.
CpuRoots FindCpuRoots(const LiteRtModelT& model, const SubgraphCallGraph& graph,
                      const std::vector<LiteRtSubgraph>& initial_subgraphs) {
  std::vector<int> num_callers(initial_subgraphs.size(), 0);
  for (const auto& callees : graph.callees) {
    for (const int32_t callee : callees) {
      ++num_callers[callee];
    }
  }
  CpuRoots roots;
  for (LiteRtSignature signature : model.Signatures()) {
    roots.Add(&signature->GetSubgraph());
  }
  if (!initial_subgraphs.empty()) {
    roots.Add(initial_subgraphs[LiteRtModelT::kMainSubgraphIndex]);
  }
  for (size_t i = 0; i < initial_subgraphs.size(); ++i) {
    // A composite decomposition always has its composite as a caller.
    if (num_callers[i] == 0) {
      roots.Add(initial_subgraphs[i]);
    }
  }
  return roots;
}

// Pins the subgraphs that must never be partitioned, wherever they are used:
// the bodies of npu_call and cpu_call composites, which are compiled or inlined
// as a whole, and the subgraphs of ops that interpret them (REDUCE, SORT, ...),
// which need them unchanged whether they end up offloaded or on CPU. Fails if
// an npu_call or cpu_call has no body. Returns the npu_call bodies.
Expected<std::vector<LiteRtSubgraph>> PinUnpartitionableSubgraphs(
    const LiteRtModelT& model,
    const std::vector<LiteRtSubgraph>& initial_subgraphs, SubgraphSet& pinned) {
  std::vector<LiteRtSubgraph> npu_call_bodies;
  for (LiteRtSubgraph subgraph : initial_subgraphs) {
    for (LiteRtOp op : subgraph->Ops()) {
      const auto kind = GetCompositeKind(op);
      if (kind == CompositeKind::kNpuCall || kind == CompositeKind::kCpuCall) {
        LiteRtSubgraph body = CompositeDecomposition(model, *op);
        if (body == nullptr) {
          return Unexpected(kLiteRtStatusErrorInvalidArgument,
                            "odml.npu_call or odml.cpu_call has no "
                            "decomposition subgraph");
        }
        if (kind == CompositeKind::kNpuCall) {
          npu_call_bodies.push_back(body);
        }
        PinClosure(model, {IndexOf(model, body)}, pinned);
      } else if (IsRegionOp(*op) && !IsControlFlowOp(op->OpCode())) {
        PinClosure(model, SubgraphRefs(*op), pinned);
      }
    }
  }
  return npu_call_bodies;
}

// CPU roots split by whether they start the walk.
struct WalkRoots {
  SubgraphSet walked;
  SubgraphSet excluded;
};

// Only CPU roots selected by `subgraphs_to_partition` (all of them if it is
// empty) start the walk. The others, and everything they reference, are
// pinned and excluded from walking. Indices that are not CPU roots are ignored
// with a warning.
WalkRoots SelectWalkRoots(
    const LiteRtModelT& model, const CpuRoots& cpu_roots,
    const std::vector<LiteRtSubgraph>& initial_subgraphs,
    const absl::flat_hash_set<uint32_t>& subgraphs_to_partition,
    SubgraphSet& pinned) {
  WalkRoots roots;
  for (LiteRtSubgraph subgraph : cpu_roots.ordered) {
    const int32_t index = IndexOf(model, subgraph);
    if (subgraphs_to_partition.empty() ||
        subgraphs_to_partition.contains(static_cast<uint32_t>(index))) {
      roots.walked.insert(subgraph);
    } else {
      roots.excluded.insert(subgraph);
      PinClosure(model, {index}, pinned);
    }
  }
  for (const uint32_t index : subgraphs_to_partition) {
    if (index >= initial_subgraphs.size() ||
        !cpu_roots.set.contains(initial_subgraphs[index])) {
      LITERT_LOG(LITERT_WARNING,
                 "Ignoring subgraph<%u> in the subgraphs to partition: it is "
                 "not a CPU root (a signature subgraph, the main subgraph or "
                 "an unreferenced subgraph).",
                 index);
    }
  }
  return roots;
}

// Visits the subgraphs callers first from the walked roots, passes each to the
// plugin and outlines the selected ops. The bodies of control flow left on CPU
// become roots in turn (unless they are excluded CPU roots); atomic region
// bodies were already pinned up front by `PinUnpartitionableSubgraphs`.
Expected<void> WalkAndPartition(
    CompilerPlugin& compiler_plugin, LiteRtModelT& model,
    absl::string_view soc_model, const SubgraphCallGraph& graph,
    const std::vector<LiteRtSubgraph>& initial_subgraphs, WalkRoots roots,
    PartitionWalkState& state) {
  // CPU control-flow ops that made each subgraph a root.
  absl::flat_hash_map<LiteRtSubgraph, std::vector<LiteRtOp>> root_callers;
  for (const int32_t index : TopologicalOrder(graph)) {
    LiteRtSubgraph subgraph = initial_subgraphs[index];
    if (!roots.walked.contains(subgraph)) {
      continue;
    }
    if (state.pinned.contains(subgraph)) {
      // Partitioned in place for its CPU callers (its signature, if it is a
      // CPU root, and `root_callers`); every other caller gets an unchanged
      // copy.
      MarkModified(state.model_modified);
      LITERT_RETURN_IF_ERROR(SplitPinnedSubgraph(
          model, *subgraph, root_callers[subgraph], state.pinned));
    }
    LITERT_RETURN_IF_ERROR(PartitionOneSubgraph(
        compiler_plugin, model, soc_model, *subgraph, index, state));
    state.processed.push_back(subgraph);
    // Ops left in the subgraph run on CPU.
    for (LiteRtOp op : subgraph->Ops()) {
      if (!IsRegionOp(*op) || !IsControlFlowOp(op->OpCode())) {
        continue;
      }
      for (const int32_t ref : SubgraphRefs(*op)) {
        LiteRtSubgraph callee = model.Subgraphs()[ref];
        if (!roots.excluded.contains(callee)) {
          roots.walked.insert(callee);
          root_callers[callee].push_back(op);
        }
      }
    }
  }
  return {};
}

// Turns each npu_call in a walked subgraph into a dispatch op compiling its
// decomposition, which becomes an entry after the outlined partitions.
// npu_calls elsewhere stay composites. Returns all entries, in dispatch op
// order.
Expected<std::vector<LiteRtSubgraph>> PromoteNpuCalls(
    LiteRtModelT& model, const CpuRoots& cpu_roots, PartitionWalkState& state) {
  std::vector<LiteRtSubgraph> entries = state.outlined;
  // Only references from live subgraphs make a decomposition shared; cpu_call
  // bodies inlined before this and decompositions cleared or inlined during
  // the walk are dead. Processed subgraphs are reachable from the CPU roots.
  LITERT_ASSIGN_OR_RETURN(auto live_graph, BuildSubgraphCallGraph(model));
  std::vector<int32_t> live_roots = IndicesOf(model, cpu_roots.ordered);
  const auto outlined_indices = IndicesOf(model, state.outlined);
  live_roots.insert(live_roots.end(), outlined_indices.begin(),
                    outlined_indices.end());
  const auto live = SubgraphClosure(live_graph, live_roots);
  for (LiteRtSubgraph subgraph : state.processed) {
    for (LiteRtOp op : subgraph->Ops()) {
      if (GetCompositeKind(op) != CompositeKind::kNpuCall) {
        continue;
      }
      MarkModified(state.model_modified);
      LITERT_ASSIGN_OR_RETURN(
          LiteRtSubgraph decomposition,
          ExclusiveNpuCallDecomposition(model, *op, live, cpu_roots.set));
      entries.push_back(decomposition);
      MakeDispatchOp(*op);
      state.dispatch_ops.push_back(op);
    }
  }
  return entries;
}

}  // namespace

Expected<PartitionResult> PartitionModel(
    CompilerPlugin& compiler_plugin, LiteRtModelT& model,
    absl::string_view soc_model,
    const absl::flat_hash_set<uint32_t>& subgraphs_to_partition,
    bool* model_modified) {
  // Visits subgraphs callers first from the CPU roots, partitions them with
  // the plugin and packages the partitions for compilation. Control-flow
  // bodies left on CPU are partitioned too; other region bodies, the bodies of
  // offloaded ops and of npu_call/cpu_call composites are pinned (a subgraph
  // that is both is cloned). Offloaded ops take their closure along as helper
  // subgraphs, which only compile-v2 plugins receive. An npu_call in a walked
  // subgraph is offloaded with its body as an entry; a cpu_call is inlined; a
  // standard composite is compiled as-is if selected, else inlined.

  // Fails without mutating the model if the subgraph references are invalid.
  LITERT_ASSIGN_OR_RETURN(const auto graph, BuildSubgraphCallGraph(model));
  const std::vector<LiteRtSubgraph> initial_subgraphs(model.Subgraphs().begin(),
                                                      model.Subgraphs().end());

  PartitionWalkState state;
  state.accepts_helper_subgraphs = compiler_plugin.AcceptsHelperSubgraphs();
  state.model_modified = model_modified;
  if (auto compiler_options = compiler_plugin.CompilerOptions();
      compiler_options.HasValue()) {
    state.strategy = compiler_options->partition_strategy;
    state.max_partitions = compiler_options->max_partitions;
  }

  LITERT_ASSIGN_OR_RETURN(
      const auto npu_call_bodies,
      PinUnpartitionableSubgraphs(model, initial_subgraphs, state.pinned));
  // Every npu_call body is checked before anything is mutated, including those
  // that end up not being offloaded: it must not need helpers the plugin
  // cannot take, and its cpu_calls must be inlinable.
  LITERT_RETURN_IF_ERROR(CheckShippedSubgraphs(model, npu_call_bodies,
                                               state.accepts_helper_subgraphs));

  const CpuRoots cpu_roots = FindCpuRoots(model, graph, initial_subgraphs);
  WalkRoots walk_roots = SelectWalkRoots(model, cpu_roots, initial_subgraphs,
                                         subgraphs_to_partition, state.pinned);
  LITERT_RETURN_IF_ERROR(WalkAndPartition(compiler_plugin, model, soc_model,
                                          graph, initial_subgraphs,
                                          std::move(walk_roots), state));

  // Stage 1 cpu_call inlining, now that the plugin has seen every walked
  // subgraph. A cpu_call that cannot be inlined stays a composite.
  for (LiteRtSubgraph subgraph : state.processed) {
    LITERT_RETURN_IF_ERROR(InlineCpuCallsInto(
        model, *subgraph, /*must_succeed=*/false, model_modified));
  }

  LITERT_ASSIGN_OR_RETURN(std::vector<LiteRtSubgraph> entries,
                          PromoteNpuCalls(model, cpu_roots, state));
  ABSL_DCHECK_EQ(state.dispatch_ops.size(), entries.size());

  LITERT_ASSIGN_OR_RETURN(
      auto partitions,
      PackagePartitions(model, cpu_roots.ordered, cpu_roots.set, entries,
                        state.accepts_helper_subgraphs));
  return PartitionResult{std::move(state.dispatch_ops), std::move(partitions)};
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
  LITERT_RETURN_IF_ERROR(PartitionSubgraph(
      std::move(selected_ops), *subgraph, dispatch_ops, model,
      kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected));
  ABSL_DCHECK_EQ(dispatch_ops.size(), model.NumSubgraphs() - 1);

  std::vector<size_t> decomps_to_compile;
  for (auto i = 1; i < model.NumSubgraphs(); ++i) {
    // Outlined subgraphs are pushed to the back of the model.
    decomps_to_compile.push_back(i);
  }
  LITERT_ASSIGN_OR_RETURN(auto new_model,
                          model.Yank(std::move(decomps_to_compile)));

  return PartitionResult{std::move(dispatch_ops), std::move(new_model)};
}

Expected<void> ApplyPluginWithPartition(CompilerPlugin& compiler_plugin,
                                        LiteRtModelT& model,
                                        PartitionResult partitions,
                                        absl::string_view soc_model,
                                        ApplyPluginsResult& result) {
  auto& dispatch_ops = partitions.first;
  auto& sliced_model = partitions.second;

  // Pass sliced subgraphs to plugin for compilation.
  auto compiled_result =
      compiler_plugin.Compile(&sliced_model, dispatch_ops.size(), soc_model);
  if (!compiled_result) {
    return compiled_result.Error();
  }

  CompiledResult& stored_result = *compiled_result;
  bool has_jit_handles = false;

  // The plugin must emit exactly one call per entry subgraph; helper subgraphs
  // after the entries get no call of their own.
  if (auto num_calls = stored_result.NumCalls();
      num_calls.HasValue() && *num_calls != dispatch_ops.size()) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 absl::StrCat("Plugin emitted ", *num_calls, " calls for ",
                              dispatch_ops.size(), " entry subgraphs"));
  }

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

Expected<void> TransformModelImpl(CompilerPlugin& compiler_plugin,
                                  LiteRtModelT& model,
                                  absl::string_view soc_model,
                                  bool* model_modified) {
  auto status = compiler_plugin.RegisterAllTransformations();
  if (!status) {
    return status;
  }
  LITERT_LOG(LITERT_INFO, "Registered %d transformations.",
             compiler_plugin.GetNumTransformations());
  if (compiler_plugin.GetNumTransformations() == 0) {
    return {};
  }
  if (model_modified != nullptr) {
    *model_modified = true;
  }
  status = compiler_plugin.GreedyPatternMatchAndRewrite(model);
  if (!status) {
    return status;
  }
  LITERT_LOG(LITERT_INFO, "Applied transformations.");
  return {};
}

// Applies `compiler_plugin` to `model`. If it fails, `*model_modified` (if
// given) tells whether `model` may already have been changed: transformation,
// partitioning and compilation modify it in place and are not undone.
Expected<void> ApplySinglePlugin(
    CompilerPlugin& compiler_plugin, LiteRtModelT& model,
    absl::string_view soc_model,
    const absl::flat_hash_set<uint32_t>& subgraphs_to_partition,
    ApplyPluginsResult& result, bool* model_modified = nullptr) {
  if (model_modified != nullptr) {
    *model_modified = false;
  }
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
  auto status =
      TransformModelImpl(compiler_plugin, model, soc_model, model_modified);
  if (!status) {
    return status.Error();
  }

  // Compiler Plugin: Partitioning, collect partitions to pass to compilation.
  auto partitions =
      PartitionModel(compiler_plugin, model, soc_model, subgraphs_to_partition,
                     model_modified);
  if (!partitions) {
    return partitions.Error();
  }

  // Compiler Plugin: Compilation, compile partitions and apply to model.
  MarkModified(model_modified);
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
  return TransformModelImpl(compiler_plugin, model, soc_model,
                            /*model_modified=*/nullptr);
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
      bool model_modified = false;
      auto status = ApplySinglePlugin(compiler_plugin, *model, "", {}, result,
                                      &model_modified);

      if (!status) {
        error_messages.push_back(
            absl::StrCat(plugin_name, " ", status.Error().Message()));
        if (model_modified) {
          // The model may be half partitioned (e.g. dispatch ops without
          // byte code). Applying further plugins to it is pointless; the
          // caller must not use it.
          result.model_invalidated = true;
          break;
        }
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
      absl::StrCat("No compiler plugin found for soc manufacturer ",
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
