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

#ifndef ODML_LITERT_LITERT_VENDORS_CC_LITERT_COMPILER_PLUGIN_H_
#define ODML_LITERT_LITERT_VENDORS_CC_LITERT_COMPILER_PLUGIN_H_

#include <memory>

#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_compiler_plugin.h"

namespace litert {

// Deleter for incomplete compiler plugin type.
struct LiteRtCompilerPluginDeleter {
  void operator()(LiteRtCompilerPlugin plugin) {
    if (plugin != nullptr) {
      LiteRtDestroyCompilerPlugin(plugin);
    }
  }
};

// Smart pointer wrapper for incomplete plugin type.
using PluginPtr =
    std::unique_ptr<LiteRtCompilerPluginT, LiteRtCompilerPluginDeleter>;

// Initialize a plugin via c-api and wrap result in smart pointer.
inline PluginPtr CreatePlugin(LiteRtEnvironmentOptions env = nullptr,
                              LiteRtOptions options = nullptr) {
  LiteRtCompilerPlugin plugin;
  LITERT_CHECK_STATUS_OK(LiteRtCreateCompilerPlugin(&plugin, env, options));
  return PluginPtr(plugin);
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_VENDORS_CC_LITERT_COMPILER_PLUGIN_H_
