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

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/filesystem.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"

namespace litert::internal {

class CompilerPluginFriend {
 public:
  static Expected<CompilerPlugin> LoadPlugin(
      absl::string_view lib_path, LiteRtEnvironmentOptions env = nullptr,
      LiteRtOptions options = nullptr) {
    return CompilerPlugin::LoadPlugin(lib_path, env, options);
  }
};

namespace {

using ::litert::internal::CompilerPlugin;
using ::litert::internal::Join;
using ::litert::testing::GetLiteRtPath;

static constexpr absl::string_view kPluginDir =
    "vendors/examples/compatibility_test";

std::string GetPluginPath(absl::string_view so_filename) {
  return Join({GetLiteRtPath(kPluginDir), so_filename});
}

// -----------------------------------------------------------------------------
// Scenario 1: Older Vendor and New LiteRT
// -----------------------------------------------------------------------------

// 1a: Older vendor returns supported interface (V1.0 with truncated
// struct_size).
// Verifies runtime negotiates V1.0, detects newer APIs as absent via
// LITERT_ABI_HAS_API, and successfully invokes supported partition and compile.
TEST(CompilerPluginCompatibilityTest,
     Scenario1a_OlderVendor_SupportedInterface) {
  std::string path = GetPluginPath("libLiteRtCompilerPlugin_OlderSupported.so");
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, CompilerPluginFriend::LoadPlugin(path, /*env=*/nullptr,
                                                    /*options=*/nullptr));

  // Negotiated version is V1.0.
  EXPECT_EQ(plugin.NegotiatedVersion().major, 1);
  EXPECT_EQ(plugin.NegotiatedVersion().minor, 0);

  // Supported API returns correct metadata.
  EXPECT_EQ(plugin.SocManufacturer(), "OlderSupportedManufacturer");

  // Optional API beyond struct_size is safely detected as absent:
  // SdkVersion returns empty string when absent.
  auto sdk_res = plugin.SdkVersion();
  ASSERT_TRUE(sdk_res.HasValue());
  EXPECT_EQ(*sdk_res, "");

  // Invoke real partition on a model graph.
  auto model = testing::LoadTestFileModel("mul_simple.tflite");
  auto subgraph = model.MainSubgraph();
  LITERT_ASSERT_OK_AND_ASSIGN(auto ops, plugin.Partition(subgraph->Get()));
  EXPECT_EQ(ops.size(), 2);

  // Invoke real compile.
  auto& model_ref = *model.Get();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              plugin.Compile(&model_ref, "MockSocModel"));
  auto bytecode = result.ByteCode();
  ASSERT_TRUE(bytecode.HasValue());
  EXPECT_GT(bytecode->Size(), 0);

  // GetHandle is beyond struct_size in older plugin; returns Unsupported.
  auto handle_res = result.GetHandle(0);
  EXPECT_FALSE(handle_res.HasValue());
  EXPECT_EQ(handle_res.Error().Status(), kLiteRtStatusErrorUnsupported);
}

// 1b: Older vendor returns non-supported interface (query returns unsupported).
TEST(CompilerPluginCompatibilityTest,
     Scenario1b_OlderVendor_UnsupportedInterface) {
  std::string path =
      GetPluginPath("libLiteRtCompilerPlugin_OlderUnsupported.so");
  auto plugin_res = CompilerPluginFriend::LoadPlugin(path, /*env=*/nullptr,
                                                     /*options=*/nullptr);
  EXPECT_FALSE(plugin_res.HasValue());
  EXPECT_EQ(plugin_res.Error().Status(), kLiteRtStatusErrorWrongVersion);
}

// -----------------------------------------------------------------------------
// Scenario 2: Newer Vendor and Old LiteRT
// -----------------------------------------------------------------------------

// 2a: Newer vendor returns supported interface (V1.2 with appended extension).
// Verifies runtime negotiates V1.2, safely ignores appended methods, and
// successfully invokes partition and compile.
TEST(CompilerPluginCompatibilityTest,
     Scenario2a_NewerVendor_SupportedInterface) {
  std::string path = GetPluginPath("libLiteRtCompilerPlugin_NewerSupported.so");
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, CompilerPluginFriend::LoadPlugin(path, /*env=*/nullptr,
                                                    /*options=*/nullptr));

  // Negotiated version is V1.2.
  EXPECT_EQ(plugin.NegotiatedVersion().major, 1);
  EXPECT_EQ(plugin.NegotiatedVersion().minor, 2);

  EXPECT_EQ(plugin.SocManufacturer(), "NewerSupportedManufacturer");

  // SdkVersion is present and succeeds.
  LITERT_ASSERT_OK_AND_ASSIGN(auto sdk, plugin.SdkVersion());
  EXPECT_EQ(sdk, "1.0.0-mock");

  // Partition and compile succeed.
  auto model = testing::LoadTestFileModel("mul_simple.tflite");
  auto subgraph = model.MainSubgraph();
  LITERT_ASSERT_OK_AND_ASSIGN(auto ops, plugin.Partition(subgraph->Get()));
  EXPECT_EQ(ops.size(), 2);

  auto& model_ref = *model.Get();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              plugin.Compile(&model_ref, "MockSocModel"));
  auto bytecode = result.ByteCode();
  ASSERT_TRUE(bytecode.HasValue());
  EXPECT_GT(bytecode->Size(), 0);
}

// 2b: Newer vendor returns non-supported interface (future V2-only vendor).
TEST(CompilerPluginCompatibilityTest,
     Scenario2b_NewerVendor_UnsupportedInterface) {
  std::string path =
      GetPluginPath("libLiteRtCompilerPlugin_NewerUnsupported.so");
  auto plugin_res = CompilerPluginFriend::LoadPlugin(path, /*env=*/nullptr,
                                                     /*options=*/nullptr);
  EXPECT_FALSE(plugin_res.HasValue());
  EXPECT_EQ(plugin_res.Error().Status(), kLiteRtStatusErrorWrongVersion);
}

// -----------------------------------------------------------------------------
// Scenario 3: Same Version Vendor and LiteRT
// -----------------------------------------------------------------------------

// 3a: Same version vendor returns supported interface (V1.1 full matching).
TEST(CompilerPluginCompatibilityTest,
     Scenario3a_SameVersion_SupportedInterface) {
  std::string path = GetPluginPath("libLiteRtCompilerPlugin_SameSupported.so");
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin, CompilerPluginFriend::LoadPlugin(path, /*env=*/nullptr,
                                                    /*options=*/nullptr));

  EXPECT_EQ(plugin.NegotiatedVersion().major, 1);
  EXPECT_EQ(plugin.NegotiatedVersion().minor, 1);
  EXPECT_EQ(plugin.SocManufacturer(), "SameSupportedManufacturer");

  LITERT_ASSERT_OK_AND_ASSIGN(auto sdk, plugin.SdkVersion());
  EXPECT_EQ(sdk, "1.0.0-mock");

  auto model = testing::LoadTestFileModel("mul_simple.tflite");
  auto subgraph = model.MainSubgraph();
  LITERT_ASSERT_OK_AND_ASSIGN(auto ops, plugin.Partition(subgraph->Get()));
  EXPECT_EQ(ops.size(), 2);

  auto& model_ref = *model.Get();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              plugin.Compile(&model_ref, "MockSocModel"));
  auto bytecode = result.ByteCode();
  ASSERT_TRUE(bytecode.HasValue());
  EXPECT_GT(bytecode->Size(), 0);
}

// 3b: Vendor returns non-supported interface (returns OK but null interface).
TEST(CompilerPluginCompatibilityTest,
     Scenario3b_SameVersion_NullInterfaceRejected) {
  std::string path = GetPluginPath("libLiteRtCompilerPlugin_NullInterface.so");
  auto plugin_res = CompilerPluginFriend::LoadPlugin(path, /*env=*/nullptr,
                                                     /*options=*/nullptr);
  EXPECT_FALSE(plugin_res.HasValue());
  EXPECT_EQ(plugin_res.Error().Status(), kLiteRtStatusErrorWrongVersion);
}

// -----------------------------------------------------------------------------
// Scenario 4: Completely Incompatible Vendor and LiteRT
// -----------------------------------------------------------------------------

// 4a: Incompatible major version (major_version = 99).
TEST(CompilerPluginCompatibilityTest,
     Scenario4a_IncompatibleMajorVersionRejected) {
  std::string path = GetPluginPath("libLiteRtCompilerPlugin_Incompatible.so");
  auto plugin_res = CompilerPluginFriend::LoadPlugin(path, /*env=*/nullptr,
                                                     /*options=*/nullptr);
  EXPECT_FALSE(plugin_res.HasValue());
  EXPECT_EQ(plugin_res.Error().Status(), kLiteRtStatusErrorWrongVersion);
}

// 4b: Corrupted header (struct_size = 4 < sizeof(LiteRtAbiHeader)).
TEST(CompilerPluginCompatibilityTest, Scenario4b_CorruptedHeaderRejected) {
  std::string path =
      GetPluginPath("libLiteRtCompilerPlugin_CorruptedHeader.so");
  auto plugin_res = CompilerPluginFriend::LoadPlugin(path, /*env=*/nullptr,
                                                     /*options=*/nullptr);
  EXPECT_FALSE(plugin_res.HasValue());
  EXPECT_EQ(plugin_res.Error().Status(), kLiteRtStatusErrorWrongVersion);
}

}  // namespace
}  // namespace litert::internal
