// Copyright 2025 Google LLC.
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

#include "litert/core/cache/compilation_cache.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <ios>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif  // __ANDROID__

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/cache/hash_util.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/options.h"
#include "litert/core/util/flatbuffer_tools.h"

namespace litert::internal {

namespace {

std::string GetCachedModelFilePath(absl::string_view cache_root_path,
                                   uint64_t model_hash) {
  return litert::internal::Join(
      {cache_root_path, absl::StrCat(model_hash, ".tflite")});
}

Expected<std::vector<litert::internal::CompilationCache::CompilerPluginInfo>>
GetPluginInfo(
    const std::vector<litert::internal::CompilerPlugin>& compiler_plugins) {
  std::vector<litert::internal::CompilationCache::CompilerPluginInfo>
      cache_info(compiler_plugins.size());
  for (int i = 0; i < compiler_plugins.size(); ++i) {
    const auto& plugin = compiler_plugins[i];
    LITERT_ASSIGN_OR_RETURN(LiteRtApiVersion api_version, plugin.ApiVersion());
    LITERT_ASSIGN_OR_RETURN(LiteRtHwAcceleratorSet hw_accelerators,
                            plugin.SupportedHardware());
    absl::string_view manufacturer = plugin.SocManufacturer();
    cache_info[i] = {
        .api_version = api_version,
        .hw_accelerators = (LiteRtHwAccelerators)hw_accelerators,
        .manufacturer = manufacturer,
    };
  }
  return cache_info;
}

uint64_t GetHash(const LiteRtOptionsT& options) {
  uint64_t seed = 0;
  HashCombine(
      seed, options.hardware_accelerators,
      options.version.major  // Minor updates should not invalid the cache.
  );

  for (LiteRtOpaqueOptions it = options.options; it;) {
    uint64_t opaque_hash = 0;
    // It's fine if an opaque option doesn't implement hashing; we skip it.
    if (LiteRtGetOpaqueOptionsHash(it, &opaque_hash) == kLiteRtStatusOk) {
      HashCombine(seed, opaque_hash);
    }
    if (LiteRtGetNextOpaqueOptions(&it) != kLiteRtStatusOk) break;
  }

  return seed;
}

uint64_t GetHash(const LiteRtApiVersion& api_version) {
  std::string_view api_version_str(reinterpret_cast<const char*>(&api_version),
                                   sizeof(api_version));
  std::hash<std::string_view> hasher;
  return hasher(api_version_str);
}

uint64_t GetHash(
    const CompilationCache::CompilerPluginInfo& compiler_plugin_info) {
  uint64_t ans = GetHash(compiler_plugin_info.api_version);
  HashCombine(ans, compiler_plugin_info.hw_accelerators);
  HashCombine(ans, compiler_plugin_info.manufacturer);
  return ans;
}

Expected<uint64_t> GetHash(const LiteRtModelT& model) {
  const ::litert::internal::FlatbufferWrapper& tfl_wrapper =
      litert::internal::GetTflFlatbuffer(model);
  const litert::BufferRef<uint8_t>& tfl_buf = tfl_wrapper.Buf();
  if (tfl_buf.Data() == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Model buffer is null");
  }

  std::string_view tfl_buf_str(reinterpret_cast<const char*>(tfl_buf.StrData()),
                               tfl_buf.Size());
  std::hash<std::string_view> hasher;
  const uint64_t model_hash = hasher(tfl_buf_str);
  return model_hash;
}
}  // namespace

Expected<CompilationCache> CompilationCache::Create(
    absl::string_view cache_root_path) {
  if (!Exists(cache_root_path)) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Cache root path does not exist");
  }
  return CompilationCache(cache_root_path);
}

Expected<uint64_t> CompilationCache::GetModelHash(
    const LiteRtModelT& model, const LiteRtOptionsT& options,
    const CompilerPluginInfo& compiler_plugin_info) {
  std::vector<CompilerPluginInfo> compiler_plugin_infos{compiler_plugin_info};
  return GetModelHash(model, options, compiler_plugin_infos);
}

Expected<uint64_t> CompilationCache::GetModelHash(
    const LiteRtModelT& model, const LiteRtOptionsT& options,
    const std::vector<CompilerPluginInfo>& compiler_plugin_infos) {
  uint64_t combined_hash = 0;
  for (const auto& compiler_plugin_info : compiler_plugin_infos) {
    uint64_t compiler_plugin_hash = GetHash(compiler_plugin_info);
    HashCombine(combined_hash, compiler_plugin_hash);
  }

#ifdef __ANDROID__
  // The compiler plugin of some vendors has dependencies on files (e.g. shared
  // libraries that are opened by their compiler plugin implementation) that
  // are part of the Android image.  Therefore if the Android build fingerprint
  // changes, the cached model might become invalid.
  char build_fingerprint[PROP_VALUE_MAX];
  if (__system_property_get("ro.build.fingerprint", build_fingerprint)) {
    HashCombine(combined_hash, std::string(build_fingerprint));
  }
#endif  // __ANDROID__

  LITERT_ASSIGN_OR_RETURN(uint64_t model_hash, GetHash(model));
  uint64_t options_hash = GetHash(options);

  HashCombine(combined_hash, model_hash, options_hash);
  return combined_hash;
}

litert::Expected<uint64_t> CompilationCache::TryGetModelHash(
    LiteRtModelT& model, LiteRtOptions options,
    Expected<std::vector<litert::internal::CompilerPlugin>>& compiler_plugins) {
  if (!compiler_plugins) {
    return compiler_plugins.Error();
  }
  // Compute the hash, so we can check the cache.
  LITERT_ASSIGN_OR_RETURN(
      std::vector<litert::internal::CompilationCache::CompilerPluginInfo>
          compiler_plugin_infos,
      GetPluginInfo(compiler_plugins.Value()));
  return litert::internal::CompilationCache::GetModelHash(
      model, *options, compiler_plugin_infos);
}

Expected<void> CompilationCache::SaveModel(const LiteRtModelT& model,
                                           uint64_t model_hash) {
  const ::litert::internal::FlatbufferWrapper& tfl_wrapper =
      litert::internal::GetTflFlatbuffer(model);
  const litert::BufferRef<uint8_t>& tfl_buf = tfl_wrapper.Buf();
  return SaveModel(tfl_buf, model_hash);
}

Expected<void> CompilationCache::SaveModel(
    const litert::BufferRef<uint8_t>& model_buffer, uint64_t model_hash) {
  const std::string cached_model_file_path =
      GetCachedModelFilePath(cache_root_path_, model_hash);
  std::ofstream output_file(cached_model_file_path,
                            std::ios::out | std::ios::binary);
  if (!output_file.is_open()) {
    LITERT_LOG(LITERT_ERROR, "Failed to open cache file for writing: %s",
               cached_model_file_path.c_str());
    return Unexpected(kLiteRtStatusErrorFileIO,
                      "Failed to open cache file for writing");
  }

  size_t data_size = model_buffer.Size();
  const char* data = reinterpret_cast<const char*>(model_buffer.Data());
  output_file.write(data, data_size);

  if (!output_file.good()) {
    LITERT_LOG(LITERT_ERROR, "Failed to write all data to cache file: %s",
               cached_model_file_path.c_str());
    return Unexpected(kLiteRtStatusErrorFileIO,
                      "Failed to write all data to cache file");
  }
  return Expected<void>();
}

Expected<std::optional<LiteRtModelT::Ptr>> CompilationCache::TryLoadModel(
    uint64_t model_hash) {
  std::string expected_model_file_path =
      GetCachedModelFilePath(cache_root_path_, model_hash);
  if (!Exists(expected_model_file_path)) {
    return Expected<std::optional<LiteRtModelT::Ptr>>(std::nullopt);
  }

  LITERT_ASSIGN_OR_RETURN(
      LiteRtModelT::Ptr cached_model,
      litert::internal::LoadModelFromFile(expected_model_file_path));
  return std::make_optional(std::move(cached_model));
}

CompilationCache::CompilationCache(absl::string_view cache_root_path)
    : cache_root_path_(cache_root_path) {}

}  // namespace litert::internal
