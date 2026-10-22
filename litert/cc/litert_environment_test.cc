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

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_runtime_builtin.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_runtime_proxy.h"
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
              reinterpret_cast<const void*>(GetLiteRtRuntimeBuiltin())),
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env, litert::Environment::Create(litert::EnvironmentOptions(
                           absl::MakeConstSpan(environment_options))));
  EXPECT_EQ(env.GetHolder().runtime->GetAbiVersion(),
            internal::RuntimeAbiVersion(1, 0, 0));
  auto accelerators = env.GetAvailableAccelerators();
  EXPECT_FALSE(accelerators);
  EXPECT_EQ(accelerators.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(EnvironmentTest, RuntimeAbiVersionParseAndCompare) {
  using internal::RuntimeAbiVersion;

  EXPECT_EQ(RuntimeAbiVersion::Parse("1.0.0"), RuntimeAbiVersion(1, 0, 0));
  EXPECT_EQ(RuntimeAbiVersion::Parse("1.12.34"), RuntimeAbiVersion(1, 12, 34));
  EXPECT_EQ(RuntimeAbiVersion::Parse("0.9.0"), RuntimeAbiVersion(0, 9, 0));
  EXPECT_EQ(RuntimeAbiVersion::Parse("2.0.0"), RuntimeAbiVersion(2, 0, 0));

  EXPECT_FALSE(RuntimeAbiVersion::Parse("").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1.0").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1..0").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1.0.").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1.0.0.0").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1.0.0-alpha").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1.0.0 ").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("01.0.0").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1.01.0").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1.0.01").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("-1.0.0").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("1.a.0").has_value());
  EXPECT_FALSE(RuntimeAbiVersion::Parse("999999999999.0.0").has_value());

  EXPECT_TRUE(RuntimeAbiVersion(1, 0, 0).IsCompatibleWithClient());
  EXPECT_TRUE(RuntimeAbiVersion(1, 99, 99).IsCompatibleWithClient());
  EXPECT_FALSE(RuntimeAbiVersion(0, 9, 0).IsCompatibleWithClient());
  EXPECT_FALSE(RuntimeAbiVersion(2, 0, 0).IsCompatibleWithClient());

  EXPECT_EQ(RuntimeAbiVersion(1, 2, 0), RuntimeAbiVersion(1, 2, 0));
  EXPECT_NE(RuntimeAbiVersion(1, 2, 0), RuntimeAbiVersion(1, 2, 1));
  EXPECT_LT(RuntimeAbiVersion(1, 0, 9), RuntimeAbiVersion(1, 1, 0));
  EXPECT_LT(RuntimeAbiVersion(1, 1, 0), RuntimeAbiVersion(1, 1, 1));
  EXPECT_LT(RuntimeAbiVersion(1, 9, 9), RuntimeAbiVersion(2, 0, 0));
  EXPECT_GT(RuntimeAbiVersion(1, 2, 0), RuntimeAbiVersion(1, 1, 9));
  EXPECT_LE(RuntimeAbiVersion(1, 2, 0), RuntimeAbiVersion(1, 2, 0));
  EXPECT_LE(RuntimeAbiVersion(1, 1, 0), RuntimeAbiVersion(1, 2, 0));
  EXPECT_GE(RuntimeAbiVersion(1, 2, 0), RuntimeAbiVersion(1, 2, 0));
  EXPECT_GE(RuntimeAbiVersion(1, 2, 0), RuntimeAbiVersion(1, 1, 0));
}

TEST(EnvironmentTest, RuntimeProxyAbiVersion1_0_0) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment base_env, Environment::Create({}));
  internal::RuntimeProxy proxy(GetLiteRtRuntimeBuiltin(),
                               internal::RuntimeAbiVersion(1, 0, 0));
  Environment env = Environment::WrapCObject(
      internal::EnvironmentHolder{&proxy, base_env.GetHolder().handle},
      OwnHandle::kNo);

  // 1.1.0 API should return kErrorUnsupported on 1.0.0 runtime.
  auto accelerators = env.GetAvailableAccelerators();
  EXPECT_FALSE(accelerators);
  EXPECT_EQ(accelerators.Error().Status(), kLiteRtStatusErrorUnsupported);

  // 1.0.0 API (CompiledModel::Create) should succeed.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, testing::GetTestFilePath(kModelFileName),
                            HwAccelerators::kCpu));

  // 1.2.0 API (IsNonCpuFullyAccelerated) should return kErrorUnsupported on
  // 1.0.0.
  auto is_accelerated = compiled_model.IsNonCpuFullyAccelerated();
  EXPECT_FALSE(is_accelerated);
  EXPECT_EQ(is_accelerated.Error().Status(), kLiteRtStatusErrorUnsupported);

  // 1.2.0 API (GetDelegationMetrics) should return kErrorUnsupported on 1.0.0.
  auto metrics = compiled_model.GetDelegationMetrics();
  EXPECT_FALSE(metrics);
  EXPECT_EQ(metrics.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(EnvironmentTest, RuntimeProxyAbiVersion1_1_0) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment base_env, Environment::Create({}));
  internal::RuntimeProxy proxy(GetLiteRtRuntimeBuiltin(),
                               internal::RuntimeAbiVersion(1, 1, 0));
  Environment env = Environment::WrapCObject(
      internal::EnvironmentHolder{&proxy, base_env.GetHolder().handle},
      OwnHandle::kNo);

  // 1.1.0 API should succeed on 1.1.0 runtime.
  LITERT_ASSERT_OK_AND_ASSIGN(auto accelerators,
                              env.GetAvailableAccelerators());
  EXPECT_FALSE(accelerators.empty());

  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, testing::GetTestFilePath(kModelFileName),
                            HwAccelerators::kCpu));

  // 1.2.0 API (IsNonCpuFullyAccelerated) should return kErrorUnsupported on
  // 1.1.0.
  auto is_accelerated = compiled_model.IsNonCpuFullyAccelerated();
  EXPECT_FALSE(is_accelerated);
  EXPECT_EQ(is_accelerated.Error().Status(), kLiteRtStatusErrorUnsupported);

  // 1.2.0 API (GetDelegationMetrics) should return kErrorUnsupported on 1.1.0.
  auto metrics = compiled_model.GetDelegationMetrics();
  EXPECT_FALSE(metrics);
  EXPECT_EQ(metrics.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(EnvironmentTest, RuntimeProxyAbiVersion1_2_0) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment base_env, Environment::Create({}));
  internal::RuntimeProxy proxy(GetLiteRtRuntimeBuiltin(),
                               internal::RuntimeAbiVersion(1, 2, 0));
  Environment env = Environment::WrapCObject(
      internal::EnvironmentHolder{&proxy, base_env.GetHolder().handle},
      OwnHandle::kNo);

  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, testing::GetTestFilePath(kModelFileName),
                            HwAccelerators::kCpu));

  // 1.2.0 APIs (IsNonCpuFullyAccelerated, GetDelegationMetrics) should succeed
  // on 1.2.0 runtime.
  LITERT_ASSERT_OK_AND_ASSIGN(bool is_accelerated,
                              compiled_model.IsNonCpuFullyAccelerated());
  EXPECT_FALSE(is_accelerated);
  LITERT_ASSERT_OK_AND_ASSIGN(auto metrics,
                              compiled_model.GetDelegationMetrics());
  EXPECT_GE(metrics.total_node_count, 0);
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
