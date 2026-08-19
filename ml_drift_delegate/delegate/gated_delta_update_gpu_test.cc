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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/internal/litert_tensor_buffer_registry.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
#include "ml_drift_delegate/delegate/buffer_handler_opencl.h"
#include "ml_drift_delegate/delegate/delegate_opencl.h"
#include "ml_drift_delegate/delegate/gated_delta_update_model_data.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"

namespace litert::ml_drift {
namespace {

void FillRandom(TfLiteTensor* tensor, float min_val = 0.0f,
                float max_val = 1.0f) {
  float* data = reinterpret_cast<float*>(tensor->data.raw);
  int num_elements = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    num_elements *= tensor->dims->data[i];
  }
  for (int i = 0; i < num_elements; ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    data[i] = min_val + r * (max_val - min_val);
  }
}

void CompareTensors(const TfLiteTensor* t1, const TfLiteTensor* t2,
                    float abs_tolerance, float rel_tolerance = 1e-4) {
  ASSERT_EQ(t1->type, t2->type);
  ASSERT_EQ(t1->dims->size, t2->dims->size);
  int num_elements = 1;
  for (int i = 0; i < t1->dims->size; ++i) {
    ASSERT_EQ(t1->dims->data[i], t2->dims->data[i]);
    num_elements *= t1->dims->data[i];
  }
  const float* d1 = reinterpret_cast<const float*>(t1->data.raw);
  const float* d2 = reinterpret_cast<const float*>(t2->data.raw);
  int mismatches = 0;
  for (int i = 0; i < num_elements; ++i) {
    float diff = std::abs(d1[i] - d2[i]);
    float max_val = std::max(std::abs(d1[i]), std::abs(d2[i]));
    if (diff > abs_tolerance &&
        (max_val == 0.0f || (diff / max_val) > rel_tolerance)) {
      if (mismatches < 10) {
        EXPECT_NEAR(d1[i], d2[i], abs_tolerance)
            << "Mismatch at index " << i
            << " (rel_diff=" << (max_val > 0 ? diff / max_val : 0) << ")";
      }
      mismatches++;
    }
  }
  if (mismatches > 0) {
    FAIL() << "Total mismatches: " << mismatches << " / " << num_elements;
  }
}

TEST(GatedDeltaUpdateGpuTest, CompilesAndAllocates) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif

  const FileToc* fp = gated_delta_update_model_data_create();
  auto model = tflite::FlatBufferModel::BuildFromBuffer(fp->data, fp->size);
  ASSERT_NE(model, nullptr);

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);

  ASSERT_EQ(LiteRtRegisterTensorBufferHandlers(
                environment, kLiteRtTensorBufferTypeOpenClBufferPacked,
                LiteRtCreateOpenClMemory, LiteRtDestroyOpenClMemory,
                LiteRtLockOpenClMemory, LiteRtUnlockOpenClMemory,
                LiteRtClearOpenClMemory, LiteRtImportOpenClMemory,
                kLiteRtEnvOptionTagOpenClContext,
                kLiteRtEnvOptionTagOpenClCommandQueue),
            kLiteRtStatusOk);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(
      "gated_delta_update",
      litert_torch::gdn_kernels::GetGatedDeltaUpdateRegistration());

  {
    std::unique_ptr<tflite::Interpreter> interpreter;
    ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
              kTfLiteOk);
    ASSERT_NE(interpreter, nullptr);

    auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
        -> litert::internal::TfLiteTensorIdentifier {
      auto tensor_id = litert::internal::GetTensorIdentifier(
          *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
      if (!tensor_id) {
        return {-1, -1};
      }
      return *tensor_id;
    };
    LiteRtExternalLiteRtBufferContextT buffer_context(environment,
                                                      get_tensor_id);
    interpreter->SetExternalContext(kTfLiteLiteRtBufferContext,
                                    &buffer_context);

    auto options = MlDriftClDelegateDefaultOptionsPtr();
    options->model_token = "test_token";
    options->serialization_dir = nullptr;
    options->runtime_context = LrtGetRuntimeContext();

    auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
    ASSERT_NE(delegate, nullptr);

    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    // Create CPU interpreter for reference.
    std::unique_ptr<tflite::Interpreter> cpu_interpreter;
    ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&cpu_interpreter),
              kTfLiteOk);
    ASSERT_NE(cpu_interpreter, nullptr);
    ASSERT_EQ(cpu_interpreter->AllocateTensors(), kTfLiteOk);

    // Fill inputs with same random data.
    std::srand(0);
    for (int i = 0; i < 6; ++i) {
      TfLiteTensor* cpu_tensor = cpu_interpreter->tensor(i);
      TfLiteTensor* gpu_tensor = interpreter->tensor(i);
      if (i == 4) {
        // g_t should be negative to decay the state
        FillRandom(cpu_tensor, -1.0f, -0.1f);
      } else {
        FillRandom(cpu_tensor, -0.5f, 0.5f);
      }
      std::memcpy(gpu_tensor->data.raw, cpu_tensor->data.raw,
                  cpu_tensor->bytes);
    }

    // Run CPU
    ASSERT_EQ(cpu_interpreter->Invoke(), kTfLiteOk);

    // Run GPU
    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

    // Compare outputs.
    // Output 0 (index 6) and Output 1 (index 7)
    float tolerance = 1e-2;
    CompareTensors(cpu_interpreter->tensor(6), interpreter->tensor(6),
                   tolerance);
    CompareTensors(cpu_interpreter->tensor(7), interpreter->tensor(7),
                   tolerance);

    std::cerr << "=== Resetting interpreter ===" << std::endl;
    interpreter.reset();
    cpu_interpreter.reset();
  }

  std::cerr << "=== Destroying environment ===" << std::endl;
  LiteRtDestroyEnvironment(environment);
  std::cerr << "=== Test finished ===" << std::endl;
}

}  // namespace
}  // namespace litert::ml_drift
