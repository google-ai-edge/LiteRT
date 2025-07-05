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

#include <array>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

constexpr absl::string_view kCompilerPluginLibSearchPath = "/data/local/tmp";
constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

using testing::FloatNear;
using testing::Pointwise;

TEST(JitCompilation, Qualcomm) {
  const std::array environment_options = {
      litert::Environment::Option{
          /*.tag=*/litert::Environment::OptionTag::CompilerPluginLibraryDir,
          /*.value=*/kCompilerPluginLibSearchPath,
      },
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(auto environment,
                              litert::Environment::Create(environment_options));

  auto model_path = litert::testing::GetTestFilePath(kModelFileName);
  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              litert::Model::CreateFromFile(model_path));

  auto num_signatures = model.GetNumSignatures();
  ASSERT_EQ(num_signatures, 1);

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              litert::CompiledModel::Create(
                                  environment, model, kLiteRtHwAcceleratorNpu));

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());
  EXPECT_EQ(input_buffers.size(), 2);

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());
  EXPECT_EQ(output_buffers.size(), 1);

  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], litert::TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}
