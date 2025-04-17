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

#include <cstdint>
#include <cstring>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_event_type.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

void BasicTest() {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(*env, model, kLiteRtHwAcceleratorGpu));

  auto signatures = model.GetSignatures().Value();
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers, compiled_model.CreateOutputBuffers(signature_index));

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
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
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsOpenClMemory());
  {
    auto lock_and_addr =
        litert::TensorBufferScopedLock::Create<const float>(output_buffers[0]);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModelGpuTest, Basic) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  BasicTest();
}

TEST(CompiledModelGpuTest, Basic2nd) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  // Run the test twice to verify that the CL environment is shared between
  // instances.
  BasicTest();
}

TEST(CompiledModelGpuTest, Async) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(*env, model, kLiteRtHwAcceleratorGpu));

  auto signatures = model.GetSignatures().Value();
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_event,
                              Event::CreateManaged(LiteRtEventTypeOpenCl));
  // Copy of the event to trigger the signal since the ownership of the
  // input_event is transferred to the input_buffers[0].
  LiteRtEvent litert_input_event = input_event.Get();

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_TRUE(input_buffers[0].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Bind the input event to the input buffers.
  // Note: The task should be done after the input buffers are filled.
  // Otherwise the input_buffers[0].Write<> will be blocked by the associated
  // event.
  input_buffers[0].SetEvent(std::move(input_event));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers, compiled_model.CreateOutputBuffers(signature_index));

  // Execute model asynchronously.
  bool async_execution_mode = true;
  compiled_model.RunAsync(signature_index, input_buffers, output_buffers,
                          async_execution_mode);

  // Signal the input event to resume the async execution.
  LiteRtEventSignal(litert_input_event);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsOpenClMemory());
  {
    auto lock_and_addr =
        litert::TensorBufferScopedLock::Create<const float>(output_buffers[0]);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModelGpuTest, PartialDelegation) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  constexpr const char *kModelPartilaFileName =
      "simple_cast_and_add_op.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(
          auto model, Model::CreateFromFile(
                          testing::GetTestFilePath(kModelPartilaFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LiteRtHwAcceleratorSet accelerator_flags =
      kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu;
  auto compilation_options = Options::Create();
  compilation_options->SetHardwareAccelerators(accelerator_flags);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(*env, model, *compilation_options));

  auto signatures = model.GetSignatures().Value();
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers, compiled_model.CreateOutputBuffers(signature_index));

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 3);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_EQ(input_names.at(2), "arg2");
  EXPECT_TRUE(input_buffers[0].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));
  int64_t arg2_data[1] = {1};
  ASSERT_TRUE(input_buffers[2].Write<int64_t>(
      absl::MakeConstSpan(arg2_data, 1)));

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add1");
  EXPECT_TRUE(output_buffers[0].IsOpenClMemory());
  {
    auto lock_and_addr =
        litert::TensorBufferScopedLock::Create<const float>(output_buffers[0]);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    float expected_output[2] = {12.0f, 23.0f};
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << expected_output[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), expected_output));
  }
}
}  // namespace
}  // namespace litert
