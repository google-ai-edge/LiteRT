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

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_profiler_event.h"
#include "litert/cc/internal/litert_compiled_model_next.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_profiler.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

Expected<Options> CreateGpuOptions(bool external_tensors_mode) {
  LITERT_ASSIGN_OR_RETURN(litert::Options options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kGpu);
  LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
  LITERT_RETURN_IF_ERROR(
      gpu_options.EnableExternalTensorsMode(external_tensors_mode));
  return std::move(options);
}

class CompiledModelGpuTest : public ::testing::TestWithParam<bool> {};

TEST_P(CompiledModelGpuTest, WithProfiler) {
  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateGpuOptions(/*no_immutable_external_tensors_mode=*/true));
  LITERT_ASSIGN_OR_ABORT(auto& runtime_options, options.GetRuntimeOptions());
  runtime_options.SetEnableProfiling(/*enabled=*/true);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModelNext::Create(*env, testing::GetTestFilePath(kModelFileName),
                                options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto profiler, compiled_model.GetProfiler());
  ASSERT_TRUE(profiler.StartProfiling());

  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_TRUE(input_buffers[0].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check the profiler.
  ASSERT_TRUE(profiler.StopProfiling());
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_events, profiler.GetNumEvents());
  ASSERT_GT(num_events, 2);
  LITERT_ASSERT_OK_AND_ASSIGN(auto events, profiler.GetEvents());
  ASSERT_EQ(events[0].event_source, ProfiledEventSource::LITERT);

  // Check model output.
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsOpenClMemory());
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_buffers[0], TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

INSTANTIATE_TEST_SUITE_P(CompiledModelGpuTest, CompiledModelGpuTest,
                         ::testing::Values(false, true));


}  // namespace
}  // namespace litert
