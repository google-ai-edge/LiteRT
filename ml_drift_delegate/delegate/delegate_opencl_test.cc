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

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "testing/base/public/gunit.h"
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "third_party/odml/infra/ml_drift_delegate/testdata/simple_add.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
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

}  // namespace
}  // namespace litert::ml_drift
