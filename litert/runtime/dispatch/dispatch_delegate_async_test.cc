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
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_dispatch.h"

#if !defined(LITERT_WINDOWS_OS) && !defined(__APPLE__)
#include <dlfcn.h>
#endif  // !defined(LITERT_WINDOWS_OS) && !defined(__APPLE__)

namespace litert {
namespace {

class DispatchDelegateAsyncTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if defined(LITERT_WINDOWS_OS) || defined(__APPLE__)
    GTEST_SKIP() << "Mock async dispatch test helpers use dlopen/dlsym and "
                    "Linux sync fence FDs.";
#else
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
        mock_lib_handle_, "LiteRtDispatch_MockDispatchSetEnvironment");
    mock_signal_next_job_ = (void (*)())dlsym(
        mock_lib_handle_, "LiteRtDispatch_MockDispatchSignalNextJob");
    mock_is_unregistered_ = (bool (*)(LiteRtTensorBufferHandle))dlsym(
        mock_lib_handle_, "LiteRtDispatch_MockDispatchIsBufferUnregistered");
    mock_get_handle_ = (LiteRtTensorBufferHandle (*)(LiteRtTensorBuffer))dlsym(
        mock_lib_handle_, "LiteRtDispatch_MockDispatchGetHandle");
    mock_num_registrations_ = (int (*)())dlsym(
        mock_lib_handle_, "LiteRtDispatch_MockDispatchNumRegistrations");
    mock_num_unregistrations_ = (int (*)())dlsym(
        mock_lib_handle_, "LiteRtDispatch_MockDispatchNumUnregistrations");
    mock_num_detaches_ = (int (*)())dlsym(
        mock_lib_handle_, "LiteRtDispatch_MockDispatchNumDetaches");

    ASSERT_NE(mock_set_env_, nullptr);
    ASSERT_NE(mock_signal_next_job_, nullptr);
    ASSERT_NE(mock_is_unregistered_, nullptr);
    ASSERT_NE(mock_get_handle_, nullptr);
    ASSERT_NE(mock_num_registrations_, nullptr);
    ASSERT_NE(mock_num_unregistrations_, nullptr);
    ASSERT_NE(mock_num_detaches_, nullptr);

    mock_set_env_(env_->GetHolder().handle);
#endif  // defined(LITERT_WINDOWS_OS) || defined(__APPLE__)
  }

  void TearDown() override {
#if !defined(LITERT_WINDOWS_OS) && !defined(__APPLE__)
    if (mock_lib_handle_) {
      dlclose(mock_lib_handle_);
    }
#endif  // !defined(LITERT_WINDOWS_OS) && !defined(__APPLE__)
  }

  void SignalNextJob() { mock_signal_next_job_(); }
  bool IsBufferUnregistered(LiteRtTensorBufferHandle handle) const {
    return mock_is_unregistered_(handle);
  }
  LiteRtTensorBufferHandle GetHandle(LiteRtTensorBuffer buffer) const {
    return mock_get_handle_(buffer);
  }
  // Numbers of Dispatch API (un)registrations so far in this process.
  int NumRegistrations() const { return mock_num_registrations_(); }
  int NumUnregistrations() const { return mock_num_unregistrations_(); }
  // Number of Dispatch API detaches so far in this process.
  int NumDetaches() const { return mock_num_detaches_(); }

  // Compiles one_mul.tflite (output = input0 * input1) for the mock NPU.
  Expected<CompiledModel> CreateOneMulModel() {
    LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
    LITERT_RETURN_IF_ERROR(compilation_options.SetHardwareAccelerators(
        litert::HwAccelerators::kNpu));
    return CompiledModel::Create(
        *env_, litert::testing::GetTestFilePath("one_mul.tflite"),
        compilation_options);
  }

  void* mock_lib_handle_ = nullptr;
  void (*mock_set_env_)(LiteRtEnvironment) = nullptr;
  void (*mock_signal_next_job_)() = nullptr;
  bool (*mock_is_unregistered_)(LiteRtTensorBufferHandle) = nullptr;
  LiteRtTensorBufferHandle (*mock_get_handle_)(LiteRtTensorBuffer) = nullptr;
  int (*mock_num_registrations_)() = nullptr;
  int (*mock_num_unregistrations_)() = nullptr;
  int (*mock_num_detaches_)() = nullptr;

  std::unique_ptr<Environment> env_;
};

TEST_F(DispatchDelegateAsyncTest,
       SwapInputTensorBufferAsyncDeferredUnregister) {
#if defined(LITERT_WINDOWS_OS) || defined(__APPLE__)
  GTEST_SKIP() << "Mock async dispatch test helpers use dlopen/dlsym and Linux "
                  "sync fence FDs.";
#else
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
      CompiledModel::Create(env, model_path, compilation_options));

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
  const int num_detaches = NumDetaches();
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
  // It is not detached, since attaching its replacement detached it.
  EXPECT_EQ(NumDetaches(), num_detaches);
#endif  // defined(LITERT_WINDOWS_OS) || defined(__APPLE__)
}

constexpr size_t kOneMulNumElements = 2 * 2;

struct OneMulBuffers {
  std::vector<TensorBuffer> inputs;
  std::vector<TensorBuffer> outputs;
};

Expected<OneMulBuffers> CreateOneMulBuffers(
    const CompiledModel& compiled_model) {
  OneMulBuffers buffers;
  LITERT_ASSIGN_OR_RETURN(auto input_names,
                          compiled_model.GetSignatureInputNames());
  for (auto input_name : input_names) {
    LITERT_ASSIGN_OR_RETURN(auto input,
                            compiled_model.CreateInputBuffer(input_name));
    buffers.inputs.push_back(std::move(input));
  }
  LITERT_ASSIGN_OR_RETURN(auto output_names,
                          compiled_model.GetSignatureOutputNames());
  for (auto output_name : output_names) {
    LITERT_ASSIGN_OR_RETURN(auto output,
                            compiled_model.CreateOutputBuffer(output_name));
    buffers.outputs.push_back(std::move(output));
  }
  return buffers;
}

Expected<void> Fill(TensorBuffer& buffer, float value) {
  const std::vector<float> data(kOneMulNumElements, value);
  return buffer.Write(absl::MakeConstSpan(data));
}

Expected<std::vector<float>> ReadAll(TensorBuffer& buffer) {
  std::vector<float> data(kOneMulNumElements);
  LITERT_RETURN_IF_ERROR(buffer.Read(absl::MakeSpan(data)));
  return data;
}

TEST_F(DispatchDelegateAsyncTest, RegisteredPoolBuffersAreRegisteredOnce) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model, CreateOneMulModel());
  LITERT_ASSERT_OK_AND_ASSIGN(auto buffers,
                              CreateOneMulBuffers(compiled_model));
  auto& [inputs, outputs] = buffers;
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  LITERT_ASSERT_OK(Fill(inputs[1], 3.0f));

  const int num_registrations = NumRegistrations();
  const int num_unregistrations = NumUnregistrations();
  constexpr int kPoolSize = 3;
  std::vector<TensorBuffer> pool;
  std::vector<RegisteredBuffer> registrations;
  for (int i = 0; i < kPoolSize; ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto buffer, compiled_model.CreateInputBuffer(input_names[0]));
    LITERT_ASSERT_OK(Fill(buffer, i + 1.0f));
    LITERT_ASSERT_OK_AND_ASSIGN(auto registration,
                                compiled_model.RegisterBuffer(buffer));
    pool.push_back(std::move(buffer));
    registrations.push_back(std::move(registration));
  }
  EXPECT_EQ(NumRegistrations(), num_registrations + kPoolSize);

  // Cycle the pool on the first input, like a buffer queue.
  constexpr int kNumRuns = 6;
  for (int i = 0; i < kNumRuns; ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(inputs[0], pool[i % kPoolSize].Duplicate());
    LITERT_ASSERT_OK(compiled_model.Run(inputs, outputs));
    LITERT_ASSERT_OK_AND_ASSIGN(auto output, ReadAll(outputs[0]));
    const float expected = (i % kPoolSize + 1.0f) * 3.0f;
    EXPECT_THAT(output, ::testing::Each(::testing::FloatEq(expected)));
  }

  // The pool, the second input and the output are each registered once.
  EXPECT_EQ(NumRegistrations(), num_registrations + kPoolSize + 2);
  EXPECT_EQ(NumUnregistrations(), num_unregistrations);
}

TEST_F(DispatchDelegateAsyncTest, RegisteredStateBuffersAreRegisteredOnce) {
  std::optional<CompiledModel> compiled_model;
  LITERT_ASSERT_OK_AND_ASSIGN(compiled_model, CreateOneMulModel());
  LITERT_ASSERT_OK_AND_ASSIGN(auto buffers,
                              CreateOneMulBuffers(*compiled_model));
  auto& [inputs, outputs] = buffers;
  LITERT_ASSERT_OK(Fill(inputs[0], 1.0f));
  LITERT_ASSERT_OK(Fill(inputs[1], 2.0f));

  const int num_registrations = NumRegistrations();
  const int num_unregistrations = NumUnregistrations();
  const int num_detaches = NumDetaches();
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_registration,
                                compiled_model->RegisterBuffer(inputs[0]));
    LITERT_ASSERT_OK_AND_ASSIGN(auto output_registration,
                                compiled_model->RegisterBuffer(outputs[0]));

    // Swap the first input and the output after each run, like recurrent state.
    constexpr int kNumRuns = 6;
    for (int i = 0; i < kNumRuns; ++i) {
      LITERT_ASSERT_OK(compiled_model->Run(inputs, outputs));
      std::swap(inputs[0], outputs[0]);
    }

    // The state buffers and the second input are each registered once, and
    // attaching them replaces the swapped ones without detaching.
    EXPECT_EQ(NumRegistrations(), num_registrations + 3);
    EXPECT_EQ(NumUnregistrations(), num_unregistrations);
    EXPECT_EQ(NumDetaches(), num_detaches);

    // 1 * 2^kNumRuns
    LITERT_ASSERT_OK_AND_ASSIGN(auto state, ReadAll(inputs[0]));
    EXPECT_THAT(state, ::testing::Each(::testing::FloatEq(64.0f)));
  }

  // Released while bound, the state buffers stay registered...
  EXPECT_EQ(NumUnregistrations(), num_unregistrations);

  // ... until the compiled model unregisters each buffer once.
  compiled_model.reset();
  EXPECT_EQ(NumUnregistrations(), num_unregistrations + 3);
}

TEST_F(DispatchDelegateAsyncTest, RegistrationsAreCounted) {
  std::optional<CompiledModel> compiled_model;
  LITERT_ASSERT_OK_AND_ASSIGN(compiled_model, CreateOneMulModel());
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model->GetSignatureInputNames());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buffer, compiled_model->CreateInputBuffer(input_names[0]));

  const int num_registrations = NumRegistrations();
  const int num_unregistrations = NumUnregistrations();
  std::optional<RegisteredBuffer> registration;
  LITERT_ASSERT_OK_AND_ASSIGN(registration,
                              compiled_model->RegisterBuffer(buffer));
  EXPECT_EQ(NumRegistrations(), num_registrations + 1);
  {
    // Registering it again does not register it again.
    LITERT_ASSERT_OK_AND_ASSIGN(auto other_registration,
                                compiled_model->RegisterBuffer(buffer));
    EXPECT_EQ(NumRegistrations(), num_registrations + 1);
  }

  // It stays registered while a registration is alive, including a moved one.
  std::optional<RegisteredBuffer> moved_registration(std::move(*registration));
  registration.reset();
  EXPECT_EQ(NumUnregistrations(), num_unregistrations);

  // The last release unregisters it right away.
  moved_registration.reset();
  EXPECT_EQ(NumUnregistrations(), num_unregistrations + 1);

  // The C API leaves releasing to the caller, so the compiled model ends it.
  LITERT_ASSERT_OK(LiteRtCompiledModelRegisterTensorBuffer(
      compiled_model->Get(), buffer.Get()));
  compiled_model.reset();
  EXPECT_EQ(NumUnregistrations(), num_unregistrations + 2);
}

TEST_F(DispatchDelegateAsyncTest, ReleasedBuffersStayRegisteredWhileInUse) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model, CreateOneMulModel());
  LITERT_ASSERT_OK_AND_ASSIGN(auto buffers,
                              CreateOneMulBuffers(compiled_model));
  auto& [inputs, outputs] = buffers;
  std::optional<RegisteredBuffer> bound_registration;
  std::optional<RegisteredBuffer> deferred_registration;
  LITERT_ASSERT_OK_AND_ASSIGN(bound_registration,
                              compiled_model.RegisterBuffer(inputs[0]));
  LITERT_ASSERT_OK_AND_ASSIGN(deferred_registration,
                              compiled_model.RegisterBuffer(inputs[1]));
  LITERT_ASSERT_OK(compiled_model.Run(inputs, outputs));

  // Released while bound, the first input stays registered.
  const int num_unregistrations = NumUnregistrations();
  bound_registration.reset();
  LITERT_ASSERT_OK(compiled_model.Run(inputs, outputs));
  EXPECT_EQ(NumUnregistrations(), num_unregistrations);

  // Released while deferred by the run that replaces it, the second input
  // stays registered.
  LITERT_ASSERT_OK_AND_ASSIGN(auto new_buffers,
                              CreateOneMulBuffers(compiled_model));
  std::swap(inputs, new_buffers.inputs);
  LITERT_ASSERT_OK(compiled_model.Run(inputs, outputs));
  deferred_registration.reset();
  EXPECT_EQ(NumUnregistrations(), num_unregistrations);

  // The next run unregisters both replaced inputs once.
  LITERT_ASSERT_OK(compiled_model.Run(inputs, outputs));
  EXPECT_EQ(NumUnregistrations(), num_unregistrations + 2);
}

}  // namespace
}  // namespace litert
