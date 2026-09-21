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

#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/version.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"

namespace litert {

class StaticallyLinkedPlugin {
 public:
  static Expected<StaticallyLinkedPlugin> Create(
      const LiteRtCompilerContext* compiler_context = nullptr,
      LiteRtEnvironmentOptions env = nullptr, LiteRtOptions options = nullptr) {
    LiteRtInterface raw_api = nullptr;
    LiteRtApiVersion runtime_version = {
        LITERT_COMPILER_PLUGIN_ABI_VERSION_MAJOR,
        LITERT_COMPILER_PLUGIN_ABI_VERSION_MINOR,
        LITERT_COMPILER_PLUGIN_ABI_VERSION_PATCH};
    LiteRtStatus status = internal::NegotiateInterface(
        LiteRtCompilerPluginQueryInterface, kLiteRtCompilerPluginInterfaceBasic,
        runtime_version,
        /*expected_abi_major=*/LITERT_COMPILER_PLUGIN_ABI_VERSION_MAJOR,
        &raw_api);
    if (status != kLiteRtStatusOk || raw_api == nullptr) {
      return Unexpected(
          status != kLiteRtStatusOk ? status : kLiteRtStatusErrorWrongVersion);
    }
    const auto* api =
        reinterpret_cast<const LiteRtCompilerPluginInterface_V1*>(raw_api);
    if (!LITERT_ABI_HAS_API(api, 1, create_compiler_plugin) ||
        !LITERT_ABI_HAS_API(api, 1, destroy_compiler_plugin)) {
      return Unexpected(kLiteRtStatusErrorWrongVersion);
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
    if (plugin_ != nullptr && api_ != nullptr &&
        LITERT_ABI_HAS_API(api_, 1, destroy_compiler_plugin)) {
      api_->destroy_compiler_plugin(plugin_);
    }
  }

  StaticallyLinkedPlugin(const StaticallyLinkedPlugin&) = delete;
  StaticallyLinkedPlugin& operator=(const StaticallyLinkedPlugin&) = delete;

  StaticallyLinkedPlugin(StaticallyLinkedPlugin&& other) noexcept {
    api_ = other.api_;
    plugin_ = other.plugin_;
    other.plugin_ = nullptr;
    other.api_ = nullptr;
  }

  StaticallyLinkedPlugin& operator=(StaticallyLinkedPlugin&& other) noexcept {
    if (this != &other) {
      if (plugin_ != nullptr && api_ != nullptr &&
          LITERT_ABI_HAS_API(api_, 1, destroy_compiler_plugin)) {
        api_->destroy_compiler_plugin(plugin_);
      }
      api_ = other.api_;
      plugin_ = other.plugin_;
      other.plugin_ = nullptr;
      other.api_ = nullptr;
    }
    return *this;
  }

  LiteRtCompilerPlugin Get() const { return plugin_; }
  const LiteRtCompilerPluginInterface_V1* Api() const { return api_; }

 private:
  StaticallyLinkedPlugin(const LiteRtCompilerPluginInterface_V1* api,
                         LiteRtCompilerPlugin plugin)
      : api_(api), plugin_(plugin) {}

  const LiteRtCompilerPluginInterface_V1* api_ = nullptr;
  LiteRtCompilerPlugin plugin_ = nullptr;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_VENDORS_CC_LITERT_COMPILER_PLUGIN_H_
