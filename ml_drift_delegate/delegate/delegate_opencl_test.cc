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

#include "ml_drift_delegate/delegate/delegate_opencl.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/testdata/simple_add.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend_opencl.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"

namespace litert::ml_drift {
namespace {

TEST(DelegateOpenClTest, FdCachingWorksWithoutSerializationDir) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif

  std::string cache_path = testing::TempDir() + "/opencl_prog.cache";
  int fd = open(cache_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  ASSERT_GE(fd, 0);

  absl::Cleanup cleanup = [&]() {
    if (fd >= 0) close(fd);
    std::remove(cache_path.c_str());
  };

  const FileToc* fp = simple_add_create();
  auto model = tflite::FlatBufferModel::BuildFromBuffer(fp->data, fp->size);
  ASSERT_TRUE(model);
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(
      tflite::InterpreterBuilder(
          *model, tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (environment) LiteRtDestroyEnvironment(environment);
  };

  auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
      -> litert::internal::TfLiteTensorIdentifier {
    auto tensor_id = litert::internal::GetTensorIdentifier(
        *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
    if (!tensor_id) {
      return {-1, -1};
    }
    return *tensor_id;
  };
  LiteRtExternalLiteRtBufferContextT buffer_context(environment, get_tensor_id);
  interpreter->SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  auto options = MlDriftClDelegateDefaultOptionsPtr();
  options->model_token = "test_token";
  options->serialization_dir = nullptr;  // Ensure null
  options->program_cache_fd = fd;
  options->serialize_program_cache = true;
  options->runtime_context = LrtGetRuntimeContext();

  auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
  ASSERT_NE(delegate, nullptr);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  // Verify original FD is still open
  EXPECT_GE(fcntl(fd, F_GETFD), 0);

  struct stat st;
  ASSERT_EQ(fstat(fd, &st), 0);
  EXPECT_GT(st.st_size, 0);
}

TEST(DelegateOpenClTest, HostIOBufferZeroSizeError) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  GpuBackendOpenCl backend(&env);
  backend.SetForceHostPtrIOBuffer(true);

  auto zero_res = backend.CreateInternalHostBuffer(
      ::ml_drift::DataType::kFloat32, /*size=*/0, /*input=*/true);
  EXPECT_EQ(zero_res.status().code(), absl::StatusCode::kInvalidArgument);

  auto valid_res = backend.CreateInternalHostBuffer(
      ::ml_drift::DataType::kFloat32, /*size=*/64, /*input=*/true);
  ASSERT_TRUE(valid_res.ok());
  auto buf = *std::move(valid_res);
  EXPECT_TRUE(buf->IsHostPtr());
  EXPECT_NE(buf->GetHostPtr(), nullptr);
  EXPECT_TRUE(buf->OwnsHostPtr());
  EXPECT_EQ(buf->GetSize(), 64);
}

TEST(DelegateOpenClTest, HostIOBufferSyncAndDataTransfer) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  GpuBackendOpenCl backend(&env);
  backend.SetForceHostPtrIOBuffer(true);

  auto res = backend.CreateInternalHostBuffer(::ml_drift::DataType::kFloat32,
                                              /*size=*/64, /*input=*/true);
  ASSERT_TRUE(res.ok());
  auto buf = *std::move(res);

  std::vector<uint8_t> write_data(64);
  for (size_t i = 0; i < 64; ++i) {
    write_data[i] = static_cast<uint8_t>(i + 1);
  }
  EXPECT_TRUE(buf->Write(write_data).ok());

  std::vector<uint8_t> read_data(64, 0);
  EXPECT_TRUE(buf->Read(absl::MakeSpan(read_data)).ok());
  EXPECT_EQ(write_data, read_data);

  uint8_t* host_bytes = static_cast<uint8_t*>(buf->GetHostPtr());
  ASSERT_NE(host_bytes, nullptr);
  host_bytes[0] = 42;
  host_bytes[1] = 99;
  EXPECT_TRUE(buf->SyncToDevice().ok());
  EXPECT_TRUE(buf->SyncToHost().ok());
  EXPECT_EQ(host_bytes[0], 42);
  EXPECT_EQ(host_bytes[1], 99);
}

TEST(DelegateOpenClTest, WrapExternalHostBufferAndDeviceBuffer) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  GpuBackendOpenCl backend(&env);
  backend.SetForceHostPtrIOBuffer(true);

  const size_t alignment = backend.GetHostPtrAlignment();
  ASSERT_GT(alignment, 0);
  void* host_ptr = nullptr;
  ASSERT_EQ(posix_memalign(&host_ptr, alignment, 64), 0);
  absl::Cleanup free_ptr = [host_ptr]() { free(host_ptr); };

  auto null_res = backend.WrapExternalHostBuffer(::ml_drift::DataType::kFloat32,
                                                 64, /*input=*/true, nullptr);
  EXPECT_EQ(null_res.status().code(), absl::StatusCode::kInvalidArgument);

  auto zero_wrap = backend.WrapExternalHostBuffer(
      ::ml_drift::DataType::kFloat32, 0, /*input=*/true, host_ptr);
  EXPECT_EQ(zero_wrap.status().code(), absl::StatusCode::kInvalidArgument);

  auto wrap_res = backend.WrapExternalHostBuffer(::ml_drift::DataType::kFloat32,
                                                 64, /*input=*/true, host_ptr);
  ASSERT_TRUE(wrap_res.ok());
  auto wrap_buf = *std::move(wrap_res);
  EXPECT_TRUE(wrap_buf->IsHostPtr());
  EXPECT_FALSE(wrap_buf->OwnsHostPtr());
  EXPECT_EQ(wrap_buf->GetHostPtr(), host_ptr);
  EXPECT_EQ(wrap_buf->GetSize(), 64);
  EXPECT_TRUE(wrap_buf->SyncToDevice().ok());
  EXPECT_TRUE(wrap_buf->SyncToHost().ok());

  auto dev_res = backend.CreateIOBufferWithSize(::ml_drift::DataType::kFloat32,
                                                /*size=*/64, /*input=*/true);
  ASSERT_TRUE(dev_res.ok());
  auto dev_buf = *std::move(dev_res);
  EXPECT_FALSE(dev_buf->IsHostPtr());
  EXPECT_EQ(dev_buf->GetHostPtr(), nullptr);
  EXPECT_TRUE(dev_buf->SyncToDevice().ok());
  EXPECT_TRUE(dev_buf->SyncToHost().ok());
}

}  // namespace
}  // namespace litert::ml_drift
