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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_runtime_builtin.h"
#include "litert/c/internal/litert_runtime_c_api.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
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
  ASSERT_NE(env->GetHolder().runtime, nullptr);
  EXPECT_EQ(env->GetHolder().runtime->GetAbiVersion(),
            internal::kBuiltinRuntimeAbiVersion);

  const char* abi_version = nullptr;
  ASSERT_EQ(env->GetHolder().runtime->GetRuntimeAbiVersion(&abi_version),
            kLiteRtStatusOk);
  ASSERT_NE(abi_version, nullptr);
  EXPECT_STREQ(abi_version, LITERT_RUNTIME_ABI_VERSION);
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
            internal::kBuiltinRuntimeAbiVersion);
  LITERT_ASSERT_OK_AND_ASSIGN(EnvironmentOptions retrieved_options,
                              env.GetOptions());
  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtVariant abi_option,
      retrieved_options.GetOption(
          litert::EnvironmentOptions::Tag::kRuntimeAbiVersion));
  EXPECT_STREQ(std::get<const char*>(abi_option), LITERT_RUNTIME_ABI_VERSION);
  LITERT_ASSERT_OK_AND_ASSIGN(auto accelerators,
                              env.GetAvailableAccelerators());
  EXPECT_FALSE(accelerators.empty());

  const char* abi_version = nullptr;
  ASSERT_EQ(env.GetHolder().runtime->GetRuntimeAbiVersion(&abi_version),
            kLiteRtStatusOk);
  ASSERT_NE(abi_version, nullptr);
  EXPECT_STREQ(abi_version, LITERT_RUNTIME_ABI_VERSION);

  // Verify that a system runtime reporting 1.1.0 via environment options is
  // resolved to 1.1.0 without invoking post-1.0.0 C API struct slots.
  LiteRtRuntimeCApiStruct runtime_1_1_0 = *GetLiteRtRuntimeBuiltin();
  runtime_1_1_0.litert_get_runtime_abi_version = nullptr;
  runtime_1_1_0.litert_get_environment_options_value =
      [](LiteRtEnvironmentOptions options, LiteRtEnvOptionTag tag,
         LiteRtAny* value) {
        if (tag == kLiteRtEnvOptionTagRuntimeAbiVersion) {
          value->type = kLiteRtAnyTypeString;
          value->str_value = "1.1.0";
          return kLiteRtStatusOk;
        }
        return GetLiteRtRuntimeBuiltin()->litert_get_environment_options_value(
            options, tag, value);
      };
  const std::vector<litert::EnvironmentOptions::Option> options_1_1_0 = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kSystemRuntimeHandle,
          litert::LiteRtVariant(reinterpret_cast<const void*>(&runtime_1_1_0)),
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env_1_1_0,
      litert::Environment::Create(
          litert::EnvironmentOptions(absl::MakeConstSpan(options_1_1_0))));
  EXPECT_EQ(env_1_1_0.GetHolder().runtime->GetAbiVersion(),
            internal::RuntimeAbiVersion(1, 1, 0));
  LITERT_ASSERT_OK_AND_ASSIGN(auto accelerators_1_1_0,
                              env_1_1_0.GetAvailableAccelerators());
  EXPECT_FALSE(accelerators_1_1_0.empty());
  EXPECT_EQ(env_1_1_0.GetHolder().runtime->GetRuntimeAbiVersion(&abi_version),
            kLiteRtStatusErrorUnsupported);

  // Verify that WrapCObject with an external runtime also resolves ABI version
  // from environment options.
  Environment wrapped_env_1_1_0 = Environment::WrapCObject(
      &runtime_1_1_0, env_1_1_0.GetHolder().handle, OwnHandle::kNo);
  EXPECT_EQ(wrapped_env_1_1_0.GetHolder().runtime->GetAbiVersion(),
            internal::RuntimeAbiVersion(1, 1, 0));

  // Verify that an older 1.0.0 system runtime (which returns NotFound for
  // kLiteRtEnvOptionTagRuntimeAbiVersion and lacks post-1.0.0 struct slots)
  // safely remains at 1.0.0 without dereferencing post-1.0.0 slots.
  LiteRtRuntimeCApiStruct runtime_1_0_0 = *GetLiteRtRuntimeBuiltin();
  runtime_1_0_0.litert_get_num_accelerators = nullptr;
  runtime_1_0_0.litert_get_accelerator = nullptr;
  runtime_1_0_0.litert_get_accelerator_hardware_support = nullptr;
  runtime_1_0_0.litert_get_runtime_abi_version = nullptr;
  runtime_1_0_0.litert_get_environment_options_value =
      [](LiteRtEnvironmentOptions options, LiteRtEnvOptionTag tag,
         LiteRtAny* value) {
        if (tag == kLiteRtEnvOptionTagRuntimeAbiVersion) {
          return kLiteRtStatusErrorNotFound;
        }
        return GetLiteRtRuntimeBuiltin()->litert_get_environment_options_value(
            options, tag, value);
      };
  const std::vector<litert::EnvironmentOptions::Option> options_1_0_0 = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kSystemRuntimeHandle,
          litert::LiteRtVariant(reinterpret_cast<const void*>(&runtime_1_0_0)),
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env_1_0_0,
      litert::Environment::Create(
          litert::EnvironmentOptions(absl::MakeConstSpan(options_1_0_0))));
  EXPECT_EQ(env_1_0_0.GetHolder().runtime->GetAbiVersion(),
            internal::RuntimeAbiVersion(1, 0, 0));
  auto unsupported_accelerators = env_1_0_0.GetAvailableAccelerators();
  EXPECT_FALSE(unsupported_accelerators);
  EXPECT_EQ(unsupported_accelerators.Error().Status(),
            kLiteRtStatusErrorUnsupported);

  // Verify that an incompatible major version in environment options falls back
  // to 1.0.0.
  LiteRtRuntimeCApiStruct runtime_incompatible = *GetLiteRtRuntimeBuiltin();
  runtime_incompatible.litert_get_runtime_abi_version = nullptr;
  runtime_incompatible.litert_get_environment_options_value =
      [](LiteRtEnvironmentOptions options, LiteRtEnvOptionTag tag,
         LiteRtAny* value) {
        if (tag == kLiteRtEnvOptionTagRuntimeAbiVersion) {
          value->type = kLiteRtAnyTypeString;
          value->str_value = "2.0.0";
          return kLiteRtStatusOk;
        }
        return GetLiteRtRuntimeBuiltin()->litert_get_environment_options_value(
            options, tag, value);
      };
  const std::vector<litert::EnvironmentOptions::Option> options_incompatible = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kSystemRuntimeHandle,
          litert::LiteRtVariant(
              reinterpret_cast<const void*>(&runtime_incompatible)),
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env_incompatible,
      litert::Environment::Create(litert::EnvironmentOptions(
          absl::MakeConstSpan(options_incompatible))));
  EXPECT_EQ(env_incompatible.GetHolder().runtime->GetAbiVersion(),
            internal::RuntimeAbiVersion(1, 0, 0));
}

TEST(EnvironmentTest, GetRuntimeAbiVersionCApi) {
  EXPECT_EQ(LiteRtGetRuntimeAbiVersion(nullptr),
            kLiteRtStatusErrorInvalidArgument);

  const char* abi_version = nullptr;
  ASSERT_EQ(LiteRtGetRuntimeAbiVersion(&abi_version), kLiteRtStatusOk);
  ASSERT_NE(abi_version, nullptr);
  EXPECT_STREQ(abi_version, LITERT_RUNTIME_ABI_VERSION);
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

  // 1.4.0 API (GetRuntimeAbiVersion) should return kErrorUnsupported on 1.0.0.
  const char* abi_version = nullptr;
  EXPECT_EQ(proxy.GetRuntimeAbiVersion(&abi_version),
            kLiteRtStatusErrorUnsupported);
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

  // 1.4.0 API (GetRuntimeAbiVersion) should return kErrorUnsupported on 1.1.0.
  const char* abi_version = nullptr;
  EXPECT_EQ(proxy.GetRuntimeAbiVersion(&abi_version),
            kLiteRtStatusErrorUnsupported);
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

  // 1.4.0 API (GetRuntimeAbiVersion) should return kErrorUnsupported on 1.2.0.
  const char* abi_version = nullptr;
  EXPECT_EQ(proxy.GetRuntimeAbiVersion(&abi_version),
            kLiteRtStatusErrorUnsupported);
}

TEST(EnvironmentTest, RuntimeProxyAbiVersion1_5_0) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment base_env, Environment::Create({}));
  internal::RuntimeProxy proxy(GetLiteRtRuntimeBuiltin(),
                               internal::RuntimeAbiVersion(1, 5, 0));
  Environment env = Environment::WrapCObject(
      internal::EnvironmentHolder{&proxy, base_env.GetHolder().handle},
      OwnHandle::kNo);

  const char* abi_version = nullptr;
  ASSERT_EQ(proxy.GetRuntimeAbiVersion(&abi_version), kLiteRtStatusOk);
  ASSERT_NE(abi_version, nullptr);
  EXPECT_STREQ(abi_version, LITERT_RUNTIME_ABI_VERSION);

  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, testing::GetTestFilePath(kModelFileName),
                            HwAccelerators::kCpu));

  // 1.6.0 API (LastInferenceDurationNanoseconds) should return
  // kErrorUnsupported on 1.5.0.
  auto duration = compiled_model.LastInferenceDurationNanoseconds();
  EXPECT_FALSE(duration);
  EXPECT_EQ(duration.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(EnvironmentTest, RuntimeProxyAbiVersion1_6_0) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment base_env, Environment::Create({}));
  internal::RuntimeProxy proxy(GetLiteRtRuntimeBuiltin(),
                               internal::RuntimeAbiVersion(1, 6, 0));
  Environment env = Environment::WrapCObject(
      internal::EnvironmentHolder{&proxy, base_env.GetHolder().handle},
      OwnHandle::kNo);

  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, testing::GetTestFilePath(kModelFileName),
                            HwAccelerators::kCpu));

  // 1.6.0 API (LastInferenceDurationNanoseconds) should succeed on 1.6.0
  // runtime.
  LITERT_ASSERT_OK_AND_ASSIGN(
      int64_t duration, compiled_model.LastInferenceDurationNanoseconds());
  EXPECT_EQ(duration, -1);
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

TEST(EnvironmentTest, CreateWithUnsupportedSystemGpuAcceleratorHandle) {
  static const LiteRtAcceleratorDef kUnsupportedGpuAcceleratorDef = {
      .abi_header =
          {
              .struct_size = sizeof(LiteRtAcceleratorDef),
              .major_version = 1,
              .minor_version = 0,
              .reserved = 0,
          },
      .get_name =
          [](LiteRtAcceleratorConst, const char** name) {
            *name = "UnsupportedGpu";
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
            *hardware = kLiteRtHwAcceleratorNone;
            return kLiteRtStatusErrorUnsupported;
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
              reinterpret_cast<const void*>(&kUnsupportedGpuAcceleratorDef)),
      },
  };
  auto env = litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
  ASSERT_TRUE(env);
  LITERT_ASSERT_OK_AND_ASSIGN(auto accelerators,
                              env->GetAvailableAccelerators());
  EXPECT_THAT(accelerators,
              ::testing::Not(::testing::Contains(HwAccelerators::kGpu)));
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
