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

#include <cstddef>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace {

using ::testing::ElementsAre;

TEST(JitCompilation, JitBypassesCaching) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          litert_libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerCacheDir,
          // Use a dummy cache dir to force the caching logic to trigger.
          "/tmp/dummy_cache_dir_for_test",
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  std::string path = litert::testing::GetTestFilePath("one_mul.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(env, path, litert::HwAccelerators::kNpu));

  // Verify it actually succeeded and loaded.
  auto num_signatures = compiled_model.GetNumSignatures();
  ASSERT_EQ(num_signatures, 1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));
  EXPECT_EQ(input_buffers.size(), 2);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));
  EXPECT_EQ(output_buffers.size(), 1);

  LITERT_ASSERT_OK(input_buffers[0].Write<float>({2.0f, 2.0f, 2.0f, 2.0f}));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));

  // Execute model.
  const size_t signature_index = 0;
  compiled_model.Run(/*signature_index=*/signature_index, input_buffers,
                     output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], litert::TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output, ElementsAre(2.0f, 4.0f, 6.0f, 8.0f));
  }
}

}  // namespace
