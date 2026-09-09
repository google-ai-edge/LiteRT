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
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
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

TEST(GatedDeltaUpdateBenchmark, BenchmarkOp) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU benchmark is not supported in MSAN/TSAN";
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

    // Qwen 3.5 0.8B representative dimensions: H=16, D_k=128, D_v=128, L=128
    int B = 1;
    int H = 16;
    int L = 128;
    int D_k = 128;
    int D_v = 128;

    ASSERT_EQ(interpreter->ResizeInputTensor(0, {B, H, L, D_k}), kTfLiteOk);
    ASSERT_EQ(interpreter->ResizeInputTensor(1, {B, H, L, D_k}), kTfLiteOk);
    ASSERT_EQ(interpreter->ResizeInputTensor(2, {B, H, L, D_v}), kTfLiteOk);
    ASSERT_EQ(interpreter->ResizeInputTensor(3, {B, H, L}), kTfLiteOk);
    ASSERT_EQ(interpreter->ResizeInputTensor(4, {B, H, L}), kTfLiteOk);
    ASSERT_EQ(interpreter->ResizeInputTensor(5, {B, H, D_k, D_v}), kTfLiteOk);

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
    options->model_token = "benchmark_token";
    options->serialization_dir = nullptr;
    options->runtime_context = LrtGetRuntimeContext();

    auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
    ASSERT_NE(delegate, nullptr);

    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    // Print shapes
    std::cerr
        << "\n================ GATED DELTA UPDATE BENCHMARK ================\n";
    for (int i = 0; i < 6; ++i) {
      TfLiteTensor* t = interpreter->tensor(i);
      std::cerr << "Input " << i << " (" << (t->name ? t->name : "unnamed")
                << "): [";
      for (int d = 0; d < t->dims->size; ++d) {
        std::cerr << t->dims->data[d] << (d + 1 < t->dims->size ? ", " : "");
      }
      std::cerr << "]\n";
    }
    for (int i = 6; i < 8; ++i) {
      TfLiteTensor* t = interpreter->tensor(i);
      std::cerr << "Output " << (i - 6) << " ("
                << (t->name ? t->name : "unnamed") << "): [";
      for (int d = 0; d < t->dims->size; ++d) {
        std::cerr << t->dims->data[d] << (d + 1 < t->dims->size ? ", " : "");
      }
      std::cerr << "]\n";
    }

    // Populate inputs
    std::srand(42);
    for (int i = 0; i < 6; ++i) {
      TfLiteTensor* gpu_tensor = interpreter->tensor(i);
      if (i == 4) {
        FillRandom(gpu_tensor, -1.0f, -0.1f);
      } else {
        FillRandom(gpu_tensor, -0.5f, 0.5f);
      }
    }

    // Warmup GPU
    constexpr int kWarmupIters = 10;
    for (int i = 0; i < kWarmupIters; ++i) {
      ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
    }

    // Benchmark GPU
    constexpr int kBenchIters = 100;
    std::vector<double> latencies_us;
    latencies_us.reserve(kBenchIters);

    for (int i = 0; i < kBenchIters; ++i) {
      absl::Time t0 = absl::Now();
      ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
      absl::Time t1 = absl::Now();
      latencies_us.push_back(absl::ToDoubleMicroseconds(t1 - t0));
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double min_us = latencies_us.front();
    double max_us = latencies_us.back();
    double sum_us =
        std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double avg_us = sum_us / kBenchIters;
    double p50_us = latencies_us[kBenchIters / 2];
    double p90_us = latencies_us[static_cast<size_t>(kBenchIters * 0.9)];

    std::cerr << std::fixed << std::setprecision(3);
    std::cerr << "\n--- GPU Latency (" << kBenchIters << " iterations) ---\n";
    std::cerr << "  Min:    " << min_us << " us (" << min_us / 1000.0
              << " ms)\n";
    std::cerr << "  Avg:    " << avg_us << " us (" << avg_us / 1000.0
              << " ms)\n";
    std::cerr << "  Median: " << p50_us << " us (" << p50_us / 1000.0
              << " ms)\n";
    std::cerr << "  P90:    " << p90_us << " us (" << p90_us / 1000.0
              << " ms)\n";
    std::cerr << "  Max:    " << max_us << " us (" << max_us / 1000.0
              << " ms)\n";
    std::cerr
        << "==============================================================\n\n";

    interpreter.reset();
  }
  LiteRtDestroyEnvironment(environment);
}

}  // namespace
}  // namespace litert::ml_drift
