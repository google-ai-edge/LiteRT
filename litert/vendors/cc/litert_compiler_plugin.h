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

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"

namespace litert {

class StaticallyLinkedPlugin {
 public:
  static Expected<StaticallyLinkedPlugin> Create(
      const LiteRtCompilerContext* compiler_context = nullptr,
      LiteRtEnvironmentOptions env = nullptr, LiteRtOptions options = nullptr) {
    LiteRtCompilerPluginInterface_V0_1* api = nullptr;
    LiteRtApiVersion try_version = {0, 1, 0};
    LiteRtStatus status = LiteRtCompilerPluginQueryInterface(
        kLiteRtCompilerPluginInterfaceBasic, try_version,
        reinterpret_cast<void**>(&api));
    if (status != kLiteRtStatusOk || api == nullptr) {
      return Unexpected(status != kLiteRtStatusOk
                            ? status
                            : kLiteRtStatusErrorRuntimeFailure);
    }

    LiteRtCompilerPlugin plugin = nullptr;
    status =
        api->create_compiler_plugin(compiler_context, &plugin, env, options);
    if (status != kLiteRtStatusOk) {
      return Unexpected(status);
    }

    return StaticallyLinkedPlugin(api, plugin);
  }

  ~StaticallyLinkedPlugin() {
    if (plugin_ != nullptr) {
      api_->destroy_compiler_plugin(plugin_);
    }
  }

  StaticallyLinkedPlugin(StaticallyLinkedPlugin&& other) {
    api_ = other.api_;
    plugin_ = other.plugin_;
    other.plugin_ = nullptr;
  }

  StaticallyLinkedPlugin& operator=(StaticallyLinkedPlugin&& other) {
    if (plugin_ != nullptr) {
      api_->destroy_compiler_plugin(plugin_);
    }
    api_ = other.api_;
    plugin_ = other.plugin_;
    other.plugin_ = nullptr;
    return *this;
  }

  LiteRtCompilerPlugin Get() const { return plugin_; }
  const LiteRtCompilerPluginInterface_V0_1* Api() const { return api_; }

 private:
  StaticallyLinkedPlugin(LiteRtCompilerPluginInterface_V0_1* api,
                         LiteRtCompilerPlugin plugin)
      : api_(api), plugin_(plugin) {}

  LiteRtCompilerPluginInterface_V0_1* api_;
  LiteRtCompilerPlugin plugin_;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_VENDORS_CC_LITERT_COMPILER_PLUGIN_H_
