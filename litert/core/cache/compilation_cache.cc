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

#include <algorithm>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
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
                                   CompilationCache::CacheKey cache_key,
                                   absl::string_view model_name) {
  std::string model_dir = model_name.empty() ? "mem" : std::string(model_name);
  return litert::internal::Join(
      {cache_root_path, model_dir, absl::StrCat(cache_key.content_hash),
       absl::StrCat(cache_key.config_hash, ".tflite")});
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
    LITERT_ASSIGN_OR_RETURN(std::string sdk_version, plugin.SdkVersion());
    cache_info[i] = {
        .api_version = api_version,
        .hw_accelerators = (LiteRtHwAccelerators)hw_accelerators,
        .manufacturer = manufacturer,
        .sdk_version = std::move(sdk_version),
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
  HashCombine(ans, compiler_plugin_info.sdk_version);
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

Expected<CompilationCache::CacheKey> CompilationCache::GetModelHash(
    const LiteRtModelT& model, const LiteRtOptionsT& options,
    const CompilerPluginInfo& compiler_plugin_info) {
  std::vector<CompilerPluginInfo> compiler_plugin_infos{compiler_plugin_info};
  return GetModelHash(model, options, compiler_plugin_infos);
}

Expected<CompilationCache::CacheKey> CompilationCache::GetModelHash(
    const LiteRtModelT& model, const LiteRtOptionsT& options,
    const std::vector<CompilerPluginInfo>& compiler_plugin_infos) {
  uint64_t config_hash = 0;
  for (const auto& compiler_plugin_info : compiler_plugin_infos) {
    HashCombine(config_hash, GetHash(compiler_plugin_info));
  }

#ifdef __ANDROID__
  char build_fingerprint[PROP_VALUE_MAX];
  if (__system_property_get("ro.build.fingerprint", build_fingerprint)) {
    HashCombine(config_hash, std::string(build_fingerprint));
  }
#endif  // __ANDROID__

  LITERT_ASSIGN_OR_RETURN(uint64_t content_hash, GetHash(model));
  HashCombine(config_hash, GetHash(options));

  return CacheKey{.content_hash = content_hash, .config_hash = config_hash};
}

litert::Expected<CompilationCache::CacheKey> CompilationCache::TryGetModelHash(
    LiteRtModelT& model, LiteRtOptions options,
    Expected<std::vector<litert::internal::CompilerPlugin>>& compiler_plugins) {
  if (!compiler_plugins) {
    return compiler_plugins.Error();
  }
  LITERT_ASSIGN_OR_RETURN(
      std::vector<litert::internal::CompilationCache::CompilerPluginInfo>
          compiler_plugin_infos,
      GetPluginInfo(compiler_plugins.Value()));
  return litert::internal::CompilationCache::GetModelHash(
      model, *options, compiler_plugin_infos);
}

Expected<void> CompilationCache::SaveModel(const LiteRtModelT& model,
                                           CacheKey cache_key,
                                           absl::string_view model_name) {
  const ::litert::internal::FlatbufferWrapper& tfl_wrapper =
      litert::internal::GetTflFlatbuffer(model);
  const litert::BufferRef<uint8_t>& tfl_buf = tfl_wrapper.Buf();
  return SaveModel(tfl_buf, cache_key, model_name);
}

Expected<void> CompilationCache::SaveModel(
    const litert::BufferRef<uint8_t>& model_buffer, CacheKey cache_key,
    absl::string_view model_name) {
  const std::string cached_model_file_path =
      GetCachedModelFilePath(cache_root_path_, cache_key, model_name);

  // Create directories if needed!
  LITERT_ASSIGN_OR_RETURN(std::string parent_dir,
                          Parent(cached_model_file_path));
  LITERT_RETURN_IF_ERROR(MkDir(parent_dir));

  {
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
  }

  LITERT_ASSIGN_OR_RETURN(auto inventory, BuildInventory());

  // Case 1 Cleanup: Limit configurations per model content.
  std::vector<CacheEntry> same_model_content_entries;
  std::string current_model_id =
      model_name.empty() ? "mem" : std::string(model_name);

  for (const auto& entry : inventory) {
    if (entry.model_id == current_model_id &&
        entry.content_hash == cache_key.content_hash) {
      same_model_content_entries.push_back(entry);
    }
  }

  if (same_model_content_entries.size() > max_configs_per_model_) {
    std::sort(same_model_content_entries.begin(),
              same_model_content_entries.end(),
              [](const CacheEntry& a, const CacheEntry& b) {
                return a.last_modified < b.last_modified;
              });

    size_t num_to_remove =
        same_model_content_entries.size() - max_configs_per_model_;
    for (size_t i = 0; i < num_to_remove; ++i) {
      LITERT_RETURN_IF_ERROR(RemoveFile(same_model_content_entries[i].path));
    }
  }

  // Case 2 Cleanup: Remove old content directories for named models.
  if (!model_name.empty()) {
    for (const auto& entry : inventory) {
      if (entry.model_id == model_name &&
          entry.content_hash != cache_key.content_hash) {
        std::string dir_path = litert::internal::Join(
            {cache_root_path_, model_name, absl::StrCat(entry.content_hash)});
        LITERT_RETURN_IF_ERROR(RmDir(dir_path));
      }
    }
  }

  // Case 3 Cleanup: Global LRU eviction.
  if (max_total_size_ > 0) {
    size_t total_size = 0;
    for (const auto& entry : inventory) {
      total_size += entry.size;
    }

    if (total_size > max_total_size_) {
      std::sort(inventory.begin(), inventory.end(),
                [](const CacheEntry& a, const CacheEntry& b) {
                  return a.last_modified < b.last_modified;
                });

      for (const auto& entry : inventory) {
        if (total_size <= max_total_size_) {
          break;
        }
        LITERT_RETURN_IF_ERROR(RemoveFile(entry.path));
        total_size -= entry.size;
      }
    }
  }

  return Expected<void>();
}

Expected<std::optional<LiteRtModelT::Ptr>> CompilationCache::TryLoadModel(
    CacheKey cache_key, absl::string_view model_name) {
  std::string expected_model_file_path =
      GetCachedModelFilePath(cache_root_path_, cache_key, model_name);
  if (!Exists(expected_model_file_path)) {
    if (!model_name.empty()) {
      std::string fallback_path =
          GetCachedModelFilePath(cache_root_path_, cache_key, "");
      if (Exists(fallback_path)) {
        expected_model_file_path = fallback_path;
      } else {
        return Expected<std::optional<LiteRtModelT::Ptr>>(std::nullopt);
      }
    } else {
      return Expected<std::optional<LiteRtModelT::Ptr>>(std::nullopt);
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      LiteRtModelT::Ptr cached_model,
      litert::internal::LoadModelFromFile(expected_model_file_path));
  return std::make_optional(std::move(cached_model));
}

CompilationCache::CompilationCache(absl::string_view cache_root_path)
    : cache_root_path_(cache_root_path) {}

Expected<std::vector<CompilationCache::CacheEntry>>
CompilationCache::BuildInventory() const {
  LITERT_ASSIGN_OR_RETURN(auto maybe_files, RecursiveListDir(cache_root_path_));

  std::vector<CacheEntry> inventory;
  for (const auto& file_path : maybe_files) {
    absl::string_view rel_path = file_path;
    if (absl::StartsWith(rel_path, cache_root_path_)) {
      rel_path = rel_path.substr(cache_root_path_.size());
      if (absl::StartsWith(rel_path, "/")) {
        rel_path = rel_path.substr(1);
      }
    }

    std::vector<absl::string_view> parts = absl::StrSplit(rel_path, '/');
    if (parts.size() != 3) {
      continue;
    }

    absl::string_view model_id = parts[0];
    absl::string_view content_hash_str = parts[1];
    absl::string_view config_hash_str = parts[2];

    size_t last_dot = config_hash_str.find_last_of('.');
    if (last_dot != absl::string_view::npos) {
      config_hash_str = config_hash_str.substr(0, last_dot);
    }

    uint64_t content_hash = 0;
    uint64_t config_hash = 0;

    if (!absl::SimpleAtoi(content_hash_str, &content_hash) ||
        !absl::SimpleAtoi(config_hash_str, &config_hash)) {
      continue;
    }

    LITERT_ASSIGN_OR_RETURN(size_t size, Size(file_path));
    LITERT_ASSIGN_OR_RETURN(auto mtime, GetLastWriteTime(file_path));

    inventory.push_back({
        .path = file_path,
        .size = size,
        .last_modified = mtime,
        .model_id = std::string(model_id),
        .content_hash = content_hash,
        .config_hash = config_hash,
    });
  }

  return inventory;
}

}  // namespace litert::internal
