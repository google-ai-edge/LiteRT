// Copyright 2026 Google LLC.
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
#include <dlfcn.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_compiled_model_next.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_dispatch.h"

namespace litert {
namespace {

class DispatchDelegateAsyncTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto mock_lib_dir =
        litert::testing::GetLiteRtPath("runtime/dispatch");
    const auto vendor_lib_dir =
        litert::testing::GetLiteRtPath("vendors/examples");

    const std::vector<litert::EnvironmentOptions::Option> environment_options =
        {
            litert::EnvironmentOptions::Option{
                litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
                mock_lib_dir,
            },
            litert::EnvironmentOptions::Option{
                litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
                vendor_lib_dir,
            },
        };
    auto env_res = litert::Environment::Create(
        litert::EnvironmentOptions(absl::MakeConstSpan(environment_options)));
    ASSERT_TRUE(env_res.HasValue());
    env_ = std::make_unique<Environment>(std::move(*env_res));

    std::string mock_lib_path =
        mock_lib_dir + "/libLiteRtDispatch_Mock_Async.so";
    mock_lib_handle_ = dlopen(mock_lib_path.c_str(), RTLD_NOW);
    ASSERT_NE(mock_lib_handle_, nullptr);

    mock_set_env_ = (void (*)(LiteRtEnvironment))dlsym(
        mock_lib_handle_, "MockDispatchSetEnvironment");
    mock_signal_next_job_ =
        (void (*)())dlsym(mock_lib_handle_, "MockDispatchSignalNextJob");
    mock_is_unregistered_ = (bool (*)(LiteRtTensorBufferHandle))dlsym(
        mock_lib_handle_, "MockDispatchIsBufferUnregistered");
    mock_get_handle_ = (LiteRtTensorBufferHandle (*)(LiteRtTensorBuffer))dlsym(
        mock_lib_handle_, "MockDispatchGetHandle");

    ASSERT_NE(mock_set_env_, nullptr);
    ASSERT_NE(mock_signal_next_job_, nullptr);
    ASSERT_NE(mock_is_unregistered_, nullptr);
    ASSERT_NE(mock_get_handle_, nullptr);

    mock_set_env_(env_->GetHolder().handle);
  }

  void TearDown() override {
    if (mock_lib_handle_) {
      dlclose(mock_lib_handle_);
    }
  }

  void SignalNextJob() { mock_signal_next_job_(); }
  bool IsBufferUnregistered(LiteRtTensorBufferHandle handle) const {
    return mock_is_unregistered_(handle);
  }
  LiteRtTensorBufferHandle GetHandle(LiteRtTensorBuffer buffer) const {
    return mock_get_handle_(buffer);
  }

  void* mock_lib_handle_ = nullptr;
  void (*mock_set_env_)(LiteRtEnvironment) = nullptr;
  void (*mock_signal_next_job_)() = nullptr;
  bool (*mock_is_unregistered_)(LiteRtTensorBufferHandle) = nullptr;
  LiteRtTensorBufferHandle (*mock_get_handle_)(LiteRtTensorBuffer) = nullptr;

  std::unique_ptr<Environment> env_;
};

TEST_F(DispatchDelegateAsyncTest,
       SwapInputTensorBufferAsyncDeferredUnregister) {
  // 1. Setup Environment (done in fixture)
  auto& env = *env_;

  // 2. Load Model
  std::string model_path = litert::testing::GetTestFilePath("one_mul.tflite");

  // 3. Create CompiledModel
  LITERT_ASSERT_OK_AND_ASSIGN(auto compilation_options, Options::Create());
  LITERT_ASSERT_OK(compilation_options.SetHardwareAccelerators(
      litert::HwAccelerators::kNpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModelNext::Create(env, model_path, compilation_options));

  // 4. Prepare for Execution
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  ASSERT_THAT(input_names, ::testing::Not(::testing::IsEmpty()));

  std::vector<TensorBuffer> inputs;
  for (auto input_name : input_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffer,
                                compiled_model.CreateInputBuffer(input_name));
    inputs.push_back(std::move(input_buffer));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  std::vector<TensorBuffer> outputs;
  for (auto output_name : output_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffer,
                                compiled_model.CreateOutputBuffer(output_name));
    outputs.push_back(std::move(output_buffer));
  }

  // 5. RunAsync (First Invocation)
  bool async = true;
  LITERT_ASSERT_OK(compiled_model.RunAsync(inputs, outputs, async, nullptr));

  // Verify that the output buffer has an event attached.
  EXPECT_TRUE(outputs[0].HasEvent());

  // Get handle of the first input buffer to track it
  LiteRtTensorBufferHandle input_buffer_handle_run1 =
      GetHandle(inputs[0].Get());

  ASSERT_NE(input_buffer_handle_run1, 0);

  // 6. Swap Input Buffer and Output Buffer for the second invocation
  LITERT_ASSERT_OK_AND_ASSIGN(auto new_input_buffer,
                              compiled_model.CreateInputBuffer(input_names[0]));

  inputs[0] = std::move(new_input_buffer);

  std::vector<TensorBuffer> outputs2;
  for (auto output_name : output_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffer,
                                compiled_model.CreateOutputBuffer(output_name));
    outputs2.push_back(std::move(output_buffer));
  }

  // 7. RunAsync (Second Invocation)
  LITERT_ASSERT_OK(compiled_model.RunAsync(inputs, outputs2, async, nullptr));

  // Verify buffer was NOT unregistered yet because job 1 is not signaled.
  EXPECT_FALSE(IsBufferUnregistered(input_buffer_handle_run1));

  // 8. Signal job completion
  SignalNextJob();

  // 9. RunAsync (Third Invocation)
  std::vector<TensorBuffer> outputs3;
  for (auto output_name : output_names) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffer,
                                compiled_model.CreateOutputBuffer(output_name));
    outputs3.push_back(std::move(output_buffer));
  }

  LITERT_ASSERT_OK(compiled_model.RunAsync(inputs, outputs3, async, nullptr));

  // Verify buffer IS unregistered now
  EXPECT_TRUE(IsBufferUnregistered(input_buffer_handle_run1));
}

}  // namespace
}  // namespace litert
