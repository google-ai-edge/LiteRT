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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CORE_CACHE_COMPILATION_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CORE_CACHE_COMPILATION_CACHE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/model/model.h"
#include "litert/core/options.h"

namespace litert::internal {

class CompilationCache {
 public:
  // Subset of compiler plugin information relevant to generate the hash.
  struct CompilerPluginInfo {
    LiteRtApiVersion api_version;
    LiteRtHwAccelerators hw_accelerators;
    std::string_view manufacturer;
  };

  // Creates a compilation cache instance that uses the provided
  // 'cache_root_path' as the filesystem location to store and load models.
  // Returns an error if the cache path does not exist in the filesystem.
  static Expected<CompilationCache> Create(absl::string_view cache_root_path);

  // Returns the hash associated with the provided 'model'. The hash is
  // computed as the combined 'std::hash' of the following properties:
  // - the serialized model buffer
  // - the options used to compile the model
  // - the compiler plugin information
  // - TODO(b/414861277): Take runtime shapes into account
  // - TODO(b/414861277): Take opaque vendor options into account
  static Expected<uint64_t> GetModelHash(
      const LiteRtModelT& model, const LiteRtOptionsT& options,
      const CompilerPluginInfo& compiler_plugin_info);
  static Expected<uint64_t> GetModelHash(
      const LiteRtModelT& model, const LiteRtOptionsT& options,
      const std::vector<CompilerPluginInfo>& compiler_plugin_infos);
  // Returns the hash of the model, with respect to the given options and
  // compiler plugins.
  static litert::Expected<uint64_t> TryGetModelHash(
      LiteRtModelT& model, LiteRtOptions options,
      litert::Expected<std::vector<litert::internal::CompilerPlugin>>&
          compiler_plugins);

  // Saves the provided 'model' in the cache, associated with the 'model_hash'.
  // The overload taking a 'model_buffer' assumes the caller already
  // has obtained the serialized representation of the LiteRtModelT.
  Expected<void> SaveModel(const LiteRtModelT& model, uint64_t model_hash);
  Expected<void> SaveModel(const litert::BufferRef<uint8_t>& model_buffer,
                           uint64_t model_hash);

  // Tries to load a model associated with the 'model_hash' from the cache.
  //
  // - Returns an empty optional if no such model can be found, i.e. a cache
  //   miss occured.
  // - Returns an optional of value 'LiteRtModelT::Ptr' if a cache hit occured.
  // - Returns a failure status if an error occurred trying to load the model.
  Expected<std::optional<LiteRtModelT::Ptr>> TryLoadModel(uint64_t model_hash);

 private:
  // Creates a compilation cache instance that uses the provided
  // 'cache_root_path' as the filesystem location to store and load models.
  explicit CompilationCache(absl::string_view cache_root_path);

  // The cache root path.
  std::string cache_root_path_;
};

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CORE_CACHE_COMPILATION_CACHE_H_
