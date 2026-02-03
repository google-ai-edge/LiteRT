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

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_platform_support.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include "tflite/delegates/gpu/gl/egl_environment.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::litert::IsOkAndHolds;

constexpr absl::string_view kDispatchLibraryDir =
    "vendors/google_tensor/dispatch";

constexpr absl::string_view kPrecompiledTfliteFile =
    "simple_model_npu_google_tensor_precompiled.tflite";

TEST(CompiledModelTest, RunWithGoogleTensorModel) {
  if (!HasAhwbSupport()) {
    GTEST_SKIP()
        << "The rest of this test is specific to Android devices with a "
           "GoogleTensor eTPU";
  }

  // Environment setup.
  const std::string dispatch_library_dir =
      testing::GetLiteRtPath(kDispatchLibraryDir);
  absl::string_view dispatch_library_dir_view(dispatch_library_dir);
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          dispatch_library_dir_view,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              litert::Environment::Create(environment_options));

  std::string model_file_path =
      testing::GetTestFilePath(kPrecompiledTfliteFile);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model_file_path, HwAccelerators::kNpu));
  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());

  ASSERT_THAT(input_buffers, SizeIs(2));
  ASSERT_THAT(output_buffers, SizeIs(1));

  // Confirm input and output buffers are AHWB.
  EXPECT_THAT(input_buffers[0].BufferType(),
              IsOkAndHolds(TensorBufferType::kAhwb));
  EXPECT_THAT(input_buffers[1].BufferType(),
              IsOkAndHolds(TensorBufferType::kAhwb));
  EXPECT_THAT(output_buffers[0].BufferType(),
              IsOkAndHolds(TensorBufferType::kAhwb));

  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Run compiled model.
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModel, RunAsyncWithGoogleTensorModel) {
  if (!HasAhwbSupport()) {
    GTEST_SKIP()
        << "The rest of this test is specific to Android devices with a "
           "GoogleTensor eTPU";
  }

  // Environment setup.
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              litert::Environment::Create(environment_options));

  std::string model_file_path =
      testing::GetTestFilePath(kPrecompiledTfliteFile);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model_file_path, HwAccelerators::kNpu));
  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());

  ASSERT_THAT(input_buffers, SizeIs(2));
  ASSERT_THAT(output_buffers, SizeIs(1));

  // Confirm input and output buffers are AHWB.
  EXPECT_THAT(input_buffers[0].BufferType(),
              IsOkAndHolds(TensorBufferType::kAhwb));
  EXPECT_THAT(input_buffers[1].BufferType(),
              IsOkAndHolds(TensorBufferType::kAhwb));
  EXPECT_THAT(output_buffers[0].BufferType(),
              IsOkAndHolds(TensorBufferType::kAhwb));

  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Run compiled model.
  bool async;
  compiled_model.RunAsync(input_buffers, output_buffers, async);
  // Since output buffers have events, async should be true.
  ASSERT_TRUE(async);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

#if LITERT_HAS_OPENGL_SUPPORT
void FillGlBuffer1(LiteRtGLuint id, size_t size) {
  std::string shader_source = R"( #version 310 es
    precision highp float;
    layout(local_size_x = 1, local_size_y = 1) in;
    layout(std430, binding = 0) buffer Output {float elements[];} output_data;
    void main() {
      uint v = gl_GlobalInvocationID.x * 2u;
      output_data.elements[v] = float(v + 1u) / 1.0;
      output_data.elements[v + 1u] = float(v + 2u) / 1.0;
    })";
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  const GLchar* sources[] = {shader_source.c_str()};
  glShaderSource(shader, 1, sources, nullptr);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glCompileShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  GLuint to_buffer_program = glCreateProgram();
  glAttachShader(to_buffer_program, shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glLinkProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, id);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glUseProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDispatchCompute(size / 2, 1, 1);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
}

void FillGlBuffer2(LiteRtGLuint id, size_t size) {
  std::string shader_source = R"( #version 310 es
    precision highp float;
    layout(local_size_x = 1, local_size_y = 1) in;
    layout(std430, binding = 0) buffer Output {float elements[];} output_data;
    void main() {
      uint v = gl_GlobalInvocationID.x * 2u;
      output_data.elements[v] = float(v + 1u) / 0.1;
      output_data.elements[v + 1u] = float(v + 2u) / 0.1;
    })";
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  const GLchar* sources[] = {shader_source.c_str()};
  glShaderSource(shader, 1, sources, nullptr);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glCompileShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  GLuint to_buffer_program = glCreateProgram();
  glAttachShader(to_buffer_program, shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glLinkProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, id);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glUseProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDispatchCompute(size / 2, 1, 1);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
}

TEST(CompiledModel, RunAsyncWithGoogleTensorModelUseAhwbGlInterop) {
  if (!HasAhwbSupport() || !HasOpenGlSupport()) {
    GTEST_SKIP()
        << "The rest of this test is specific to Android devices with a "
           "GoogleTensor eTPU";
  }

  std::unique_ptr<tflite::gpu::gl::EglEnvironment> gl_env;
  ASSERT_OK(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&gl_env));
  // Environment setup.
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
      litert::Environment::Option{
          litert::Environment::OptionTag::EglDisplay,
          reinterpret_cast<int64_t>(gl_env->display()),
      },
      litert::Environment::Option{
          litert::Environment::OptionTag::EglContext,
          reinterpret_cast<int64_t>(gl_env->context().context()),
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              litert::Environment::Create(environment_options));

  std::string model_file_path =
      testing::GetTestFilePath(kPrecompiledTfliteFile);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model_file_path, HwAccelerators::kNpu));
  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());

  ASSERT_THAT(input_buffers, SizeIs(2));
  ASSERT_THAT(output_buffers, SizeIs(1));

  // Confirm input and output buffers are AHWB.
  EXPECT_THAT(input_buffers[0].BufferType(),
              IsOkAndHolds(TensorBufferType::kAhwb));
  EXPECT_THAT(input_buffers[1].BufferType(),
              IsOkAndHolds(TensorBufferType::kAhwb));
  EXPECT_THAT(output_buffers[0].BufferType(),
              IsOkAndHolds(TensorBufferType::kAhwb));
  // Write to input buffers on GPU.
  LITERT_ASSERT_OK_AND_ASSIGN(auto gl_buffer_1, input_buffers[0].GetGlBuffer());
  FillGlBuffer1(gl_buffer_1.id, 2);
  LITERT_ASSERT_OK_AND_ASSIGN(auto gl_buffer_2, input_buffers[1].GetGlBuffer());
  FillGlBuffer2(gl_buffer_2.id, 2);

  // Create EGL sync and fence before AHWB read.
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event egl_sync_event,
      Event::CreateManaged(env, Event::Type::kEglNativeSyncFence));

  // EGL does not support querying the sync fd from the EGL sync event. So we
  // dup the fd from the EGL sync event and use it to create a new event.
  LITERT_ASSERT_OK_AND_ASSIGN(int egl_sync_fd, egl_sync_event.DupFd());

  // Create two events from the same sync fd. One event will own the fd and the
  // other event will not. Ownership is required to ensure that the fd is
  // closed.
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event_1,
      Event::CreateFromSyncFenceFd(env, egl_sync_fd, /*owns_fd=*/true));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event_2,
      Event::CreateFromSyncFenceFd(env, egl_sync_fd, /*owns_fd=*/false));

  // Set event so that AHWB read is blocked by GPU write.
  input_buffers[0].SetEvent(std::move(event_1));
  input_buffers[1].SetEvent(std::move(event_2));

  // Run compiled model asynchronously.
  bool async;
  compiled_model.RunAsync(input_buffers, output_buffers, async);
  // Since output buffers have events, async should be true.
  ASSERT_TRUE(async);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}
#endif  // LITERT_HAS_OPENGL_SUPPORT

}  // namespace
}  // namespace litert
