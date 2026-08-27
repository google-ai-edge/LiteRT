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

#include "litert/cc/litert_environment.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_runtime_builtin.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

namespace litert {
namespace {

TEST(EnvironmentTest, Default) {
  auto env = litert::Environment::Create({});
  EXPECT_TRUE(env);
}

TEST(EnvironmentTest, SupportsFP16) {
  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);
  // Default environment shouldn't have FP16 support set, so it should be false.
  EXPECT_FALSE(env->SupportsFP16());
}

TEST(EnvironmentTest, GetAvailableAccelerators) {
  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);
  LITERT_ASSERT_OK_AND_ASSIGN(auto accelerators,
                              env->GetAvailableAccelerators());
  ASSERT_FALSE(accelerators.empty());
  EXPECT_EQ(accelerators[0], HwAccelerators::kCpu);
}

TEST(EnvironmentTest, HasRuntimeProxy) {
  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);
  EXPECT_NE(env->GetHolder().runtime, nullptr);
}

TEST(EnvironmentTest, CreateWithSystemRuntime) {
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kSystemRuntimeHandle,
          litert::LiteRtVariant(
              reinterpret_cast<int64_t>(GetLiteRtRuntimeBuiltin())),
      },
  };
  auto env = litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
  EXPECT_TRUE(env);
}

TEST(EnvironmentTest, CreateWithSystemGpuAcceleratorHandle) {
  static const LiteRtAcceleratorDef kDummyGpuAcceleratorDef = {
      .abi_header =
          {
              .struct_size = sizeof(LiteRtAcceleratorDef),
              .major_version = 1,
              .minor_version = 0,
              .reserved = 0,
          },
      .get_name =
          [](LiteRtAcceleratorConst, const char** name) {
            *name = "DummyGpu";
            return kLiteRtStatusOk;
          },
      .get_version =
          [](LiteRtAcceleratorConst, LiteRtApiVersion* version) {
            version->major = 1;
            version->minor = 0;
            version->patch = 0;
            return kLiteRtStatusOk;
          },
      .get_hardware_support =
          [](LiteRtAcceleratorConst, LiteRtHwAcceleratorSet* hardware) {
            *hardware = kLiteRtHwAcceleratorGpu;
            return kLiteRtStatusOk;
          },
      .is_tflite_delegate_responsible_for_jit_compilation =
          [](LiteRtAcceleratorConst, bool* does_jit) {
            *does_jit = false;
            return kLiteRtStatusOk;
          },
      .create_delegate = [](LiteRtRuntimeContext*, LiteRtEnvironment,
                            LiteRtAcceleratorConst, LiteRtOptions,
                            LiteRtDelegateWrapper*) { return kLiteRtStatusOk; },
  };

  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kSystemGpuAcceleratorHandle,
          litert::LiteRtVariant(
              reinterpret_cast<const void*>(&kDummyGpuAcceleratorDef)),
      },
  };
  auto env = litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
  ASSERT_TRUE(env);
  LITERT_ASSERT_OK_AND_ASSIGN(auto accelerators,
                              env->GetAvailableAccelerators());
  ASSERT_FALSE(accelerators.empty());
  EXPECT_EQ(accelerators[0], HwAccelerators::kGpu);
}

TEST(EnvironmentTest, Options) {
  constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  auto env = litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
  EXPECT_TRUE(env);
}

TEST(EnvironmentTest, CompiledModelBasic) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create CompiledModel.
  auto compiled_model = CompiledModel::Create(
      env, testing::GetTestFilePath(kModelFileName), HwAccelerators::kCpu);
  EXPECT_TRUE(compiled_model);
}

TEST(EnvironmentTest, StringLifeCycle) {
  std::string dispatch_library_dir = "/data/local/tmp";
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          absl::string_view(dispatch_library_dir),
      },
  };

  auto env = litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));

  EXPECT_TRUE(env);

  // Change the string value but the environment should still have a copy.
  dispatch_library_dir = "";

  // Create CompiledModel.
  auto compiled_model = CompiledModel::Create(
      *env, testing::GetTestFilePath(kModelFileName), HwAccelerators::kCpu);
  EXPECT_TRUE(compiled_model);
}

}  // namespace
}  // namespace litert
