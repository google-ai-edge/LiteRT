// Copyright 2026 Amlogic, Inc.
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

#include <gtest/gtest.h>

#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"

namespace {

const LiteRtCompilerPluginInterface_V1* PluginApi() {
  LiteRtInterface raw = nullptr;
  const LiteRtApiVersion runtime_version = {1, 0, 0};
  EXPECT_EQ(LiteRtCompilerPluginQueryInterface(
                kLiteRtCompilerPluginInterfaceBasic, runtime_version, &raw),
            kLiteRtStatusOk);
  return reinterpret_cast<const LiteRtCompilerPluginInterface_V1*>(raw);
}

TEST(AmlogicCompilerPluginTest, QueryInterfaceAndManufacturer) {
  const LiteRtCompilerPluginInterface_V1* api = PluginApi();
  ASSERT_NE(api, nullptr);
  EXPECT_TRUE(LITERT_ABI_IS_COMPATIBLE(api, 1, 0));
  EXPECT_STREQ(api->get_compiler_plugin_soc_manufacturer(), "Amlogic");
}

TEST(AmlogicCompilerPluginTest, SocModels) {
  const LiteRtCompilerPluginInterface_V1* api = PluginApi();
  ASSERT_NE(api, nullptr);

  LiteRtCompilerPlugin plugin = nullptr;
  ASSERT_EQ(api->create_compiler_plugin(nullptr, &plugin, nullptr, nullptr),
            kLiteRtStatusOk);
  ASSERT_NE(plugin, nullptr);

  LiteRtParamIndex num_models = 0;
  EXPECT_EQ(api->get_num_compiler_plugin_supported_models(nullptr, &num_models),
            kLiteRtStatusErrorInvalidArgument);
  ASSERT_EQ(api->get_num_compiler_plugin_supported_models(plugin, &num_models),
            kLiteRtStatusOk);
  EXPECT_EQ(num_models, 9);

  const char* soc_model = nullptr;
  ASSERT_EQ(api->get_compiler_plugin_supported_soc_model(plugin, 0, &soc_model),
            kLiteRtStatusOk);
  EXPECT_STREQ(soc_model, "c3");
  EXPECT_EQ(api->get_compiler_plugin_supported_soc_model(plugin, num_models,
                                                         &soc_model),
            kLiteRtStatusErrorInvalidArgument);

  const char* sdk_version = nullptr;
  ASSERT_EQ(api->get_compiler_plugin_sdk_version(plugin, &sdk_version),
            kLiteRtStatusOk);
  ASSERT_NE(sdk_version, nullptr);
  EXPECT_NE(sdk_version[0], '\0');

  EXPECT_EQ(api->check_compiler_compatibility(
                {LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
                 LITERT_API_VERSION_PATCH},
                plugin, nullptr, nullptr, "not_a_chip"),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->check_compiler_compatibility(
                {LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
                 LITERT_API_VERSION_PATCH},
                plugin, nullptr, nullptr, "c3"),
            kLiteRtStatusOk);

  LiteRtParamIndex num_byte_code = 0;
  EXPECT_EQ(api->get_compiled_result_num_byte_code(nullptr, &num_byte_code),
            kLiteRtStatusErrorInvalidArgument);

  api->destroy_compiler_plugin(plugin);
}

}  // namespace
