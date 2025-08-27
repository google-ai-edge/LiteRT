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

#include "litert/runtime/gpu_environment.h"

#include <any>
#include <array>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "litert/c/litert_any.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_any.h"
#include "litert/core/environment.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

TEST(EnvironmentSingletonTest, OpenClEnvironment) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif

  if (!ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }

  ml_drift::cl::Environment env;
  ASSERT_OK(ml_drift::cl::CreateEnvironment(&env));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny context_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(env.context().context()))));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny queue_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(env.queue()->queue()))));

  const std::array<LiteRtEnvOption, 2> environment_options = {
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagOpenClContext,
          /*.value=*/context_id,
      },
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagOpenClCommandQueue,
          /*.value=*/queue_id,
      },
  };
  auto litert_envt = LiteRtEnvironmentT::CreateWithOptions(environment_options);
  ASSERT_TRUE(litert_envt);
  auto singleton_env =
      litert::internal::GpuEnvironment::Create(litert_envt->get());
  ASSERT_TRUE(singleton_env);
  EXPECT_EQ((*singleton_env)->GetContext()->context(), env.context().context());
  EXPECT_EQ((*singleton_env)->GetCommandQueue()->queue(), env.queue()->queue());
}

}  // namespace
}  // namespace litert
