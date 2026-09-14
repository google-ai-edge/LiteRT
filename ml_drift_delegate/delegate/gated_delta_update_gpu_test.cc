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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
#include "ml_drift_delegate/delegate/delegate_opencl.h"
#include "ml_drift_delegate/delegate/gated_delta_update_test_util.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"

namespace litert::ml_drift {
namespace {

void FillConstant(TfLiteTensor* tensor, float val) {
  float* data = reinterpret_cast<float*>(tensor->data.raw);
  int num_elements = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    num_elements *= tensor->dims->data[i];
  }
  for (int i = 0; i < num_elements; ++i) {
    data[i] = val;
  }
}

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

// Golden reference mathematical implementation of recurrent gated delta rule:
//   1. S'_t = S_{t-1} * exp(g_t)
//   2. kv_mem = (S'_t)^T * k_t
//   3. delta = (v_t - kv_mem) * beta_t
//   4. S_t = S'_t + k_t * delta^T
//   5. y_t = (S_t)^T * q_t
void ComputeGoldenRecurrentGatedDelta(
    const float* q, const float* k, const float* v, const float* beta,
    const float* g, const float* initial_state, float* golden_out,
    float* golden_final_state, int B, int H, int N, int D_k, int D_v) {
  int state_elements = B * H * D_k * D_v;
  std::memcpy(golden_final_state, initial_state,
              state_elements * sizeof(float));

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const int bh = b * H + h;
      float* S = golden_final_state + bh * D_k * D_v;

      for (int t = 0; t < N; ++t) {
        const float* q_t = q + (bh * N + t) * D_k;
        const float* k_t = k + (bh * N + t) * D_k;
        const float* v_t = v + (bh * N + t) * D_v;
        const float beta_t = beta[bh * N + t];
        const float g_decay = std::exp(g[bh * N + t]);
        float* out_t = golden_out + (bh * N + t) * D_v;

        // 1. Decay state S = S * exp(g)
        for (int i = 0; i < D_k * D_v; ++i) {
          S[i] *= g_decay;
        }

        // 2. kv_mem[j] = sum_i (S[i, j] * k_t[i])
        std::vector<float> kv_mem(D_v, 0.0f);
        for (int j = 0; j < D_v; ++j) {
          for (int i = 0; i < D_k; ++i) {
            kv_mem[j] += S[i * D_v + j] * k_t[i];
          }
        }

        // 3. delta[j] = (v_t[j] - kv_mem[j]) * beta_t
        std::vector<float> delta(D_v, 0.0f);
        for (int j = 0; j < D_v; ++j) {
          delta[j] = (v_t[j] - kv_mem[j]) * beta_t;
        }

        // 4. Update state: S[i, j] += k_t[i] * delta[j]
        for (int i = 0; i < D_k; ++i) {
          for (int j = 0; j < D_v; ++j) {
            S[i * D_v + j] += k_t[i] * delta[j];
          }
        }

        // 5. Read out: y_t[j] = sum_i (S[i, j] * q_t[i])
        for (int j = 0; j < D_v; ++j) {
          float sum = 0.0f;
          for (int i = 0; i < D_k; ++i) {
            sum += S[i * D_v + j] * q_t[i];
          }
          out_t[j] = sum;
        }
      }
    }
  }
}

void RunGatedDeltaUpdateTest(int B, int H, int N, int D_k, int D_v,
                             bool zero_initial_state = false,
                             float beta_fixed = -1.0f, float g_fixed = 100.0f,
                             float tolerance = 1e-2) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif

  auto model_buf = CreateGatedDeltaUpdateModelBuffer(B, H, N, D_k, D_v);
  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buf.data()), model_buf.size());
  ASSERT_NE(model, nullptr);

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);


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

    // Create CPU interpreter for reference verification.
    std::unique_ptr<tflite::Interpreter> cpu_interpreter;
    ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&cpu_interpreter),
              kTfLiteOk);
    ASSERT_NE(cpu_interpreter, nullptr);
    ASSERT_EQ(cpu_interpreter->AllocateTensors(), kTfLiteOk);

    // Populate inputs.
    std::srand(0);
    // 0: q [B, H, N, D_k]
    FillRandom(cpu_interpreter->tensor(0), -0.5f, 0.5f);
    // 1: k [B, H, N, D_k]
    FillRandom(cpu_interpreter->tensor(1), -0.5f, 0.5f);
    // 2: v [B, H, N, D_v]
    FillRandom(cpu_interpreter->tensor(2), -0.5f, 0.5f);

    // 3: beta [B, H, N]
    if (beta_fixed >= 0.0f) {
      FillConstant(cpu_interpreter->tensor(3), beta_fixed);
    } else {
      FillRandom(cpu_interpreter->tensor(3), 0.1f, 0.9f);
    }

    // 4: g [B, H, N]
    if (g_fixed <= 0.0f) {
      FillConstant(cpu_interpreter->tensor(4), g_fixed);
    } else {
      FillRandom(cpu_interpreter->tensor(4), -1.0f, -0.1f);
    }

    // 5: initial_state [B, H, D_k, D_v]
    if (zero_initial_state) {
      FillConstant(cpu_interpreter->tensor(5), 0.0f);
    } else {
      FillRandom(cpu_interpreter->tensor(5), -0.5f, 0.5f);
    }

    // Copy to GPU interpreter inputs.
    for (int i = 0; i < 6; ++i) {
      TfLiteTensor* cpu_t = cpu_interpreter->tensor(i);
      TfLiteTensor* gpu_t = interpreter->tensor(i);
      std::memcpy(gpu_t->data.raw, cpu_t->data.raw, cpu_t->bytes);
    }

    // Verify golden math matches CPU reference kernel.
    std::vector<float> golden_out(B * H * N * D_v);
    std::vector<float> golden_final_state(B * H * D_k * D_v);
    ComputeGoldenRecurrentGatedDelta(
        reinterpret_cast<const float*>(cpu_interpreter->tensor(0)->data.raw),
        reinterpret_cast<const float*>(cpu_interpreter->tensor(1)->data.raw),
        reinterpret_cast<const float*>(cpu_interpreter->tensor(2)->data.raw),
        reinterpret_cast<const float*>(cpu_interpreter->tensor(3)->data.raw),
        reinterpret_cast<const float*>(cpu_interpreter->tensor(4)->data.raw),
        reinterpret_cast<const float*>(cpu_interpreter->tensor(5)->data.raw),
        golden_out.data(), golden_final_state.data(), B, H, N, D_k, D_v);

    ASSERT_EQ(cpu_interpreter->Invoke(), kTfLiteOk);

    // Compare CPU reference against golden math.
    const float* cpu_out =
        reinterpret_cast<const float*>(cpu_interpreter->tensor(6)->data.raw);
    for (size_t i = 0; i < golden_out.size(); ++i) {
      EXPECT_NEAR(cpu_out[i], golden_out[i], 1e-4f);
    }

    // Execute GPU delegate.
    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

    // Verify GPU delegate matches CPU reference and golden output.
    CompareTensors(cpu_interpreter->tensor(6), interpreter->tensor(6),
                   tolerance);
    CompareTensors(cpu_interpreter->tensor(7), interpreter->tensor(7),
                   tolerance);

    interpreter.reset();
    cpu_interpreter.reset();
  }

  LiteRtDestroyEnvironment(environment);
}

// ---------------------------------------------------------------------------
// Tests for GatedDeltaUpdate GPU Delegate across various shapes & corner cases
// ---------------------------------------------------------------------------

// 1. Single-token autoregressive decode step (N = 1).
TEST(GatedDeltaUpdateGpuTest, SingleTokenDecode) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/1, /*D_k=*/16, /*D_v=*/16);
}

// 2. Multi-token prefill sequence (N > 1).
TEST(GatedDeltaUpdateGpuTest, MultiTokenPrefill) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/8, /*D_k=*/16, /*D_v=*/16);
}

// 3. Cold start: initial recurrent state is all zeros.
TEST(GatedDeltaUpdateGpuTest, ZeroInitialStateColdStart) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/4, /*D_k=*/16, /*D_v=*/16,
                          /*zero_initial_state=*/true);
}

// 4. Zero delta update: beta = 0 (no new memory written, state only decays).
TEST(GatedDeltaUpdateGpuTest, ZeroDeltaUpdateBetaZero) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/4, /*D_k=*/16, /*D_v=*/16,
                          /*zero_initial_state=*/false, /*beta_fixed=*/0.0f);
}

// 5. Zero decay: g = 0 (decay = exp(0) = 1.0, pure accumulation).
TEST(GatedDeltaUpdateGpuTest, ZeroDecayGZero) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/4, /*D_k=*/16, /*D_v=*/16,
                          /*zero_initial_state=*/false, /*beta_fixed=*/-1.0f,
                          /*g_fixed=*/0.0f);
}

// 6. Strong decay: g = -20 (decay -> 0, state is completely flushed).
TEST(GatedDeltaUpdateGpuTest, StrongDecayGFlush) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/4, /*D_k=*/16, /*D_v=*/16,
                          /*zero_initial_state=*/false, /*beta_fixed=*/-1.0f,
                          /*g_fixed=*/-20.0f);
}

// 7. Multi-batch and multi-head scaling.
TEST(GatedDeltaUpdateGpuTest, MultiBatchMultiHead) {
  RunGatedDeltaUpdateTest(/*B=*/2, /*H=*/4, /*N=*/4, /*D_k=*/16, /*D_v=*/16);
}

// 8. Head dimension 32 (8 k_slices).
TEST(GatedDeltaUpdateGpuTest, HeadDimension32) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/2, /*N=*/2, /*D_k=*/32, /*D_v=*/32);
}

// 9. Head dimension 64 (16 k_slices).
TEST(GatedDeltaUpdateGpuTest, HeadDimension64) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/2, /*D_k=*/64, /*D_v=*/64);
}

// 10. Qwen 3.5 representative head dimension 128 (32 k_slices).
TEST(GatedDeltaUpdateGpuTest, HeadDimension128) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/2, /*D_k=*/128, /*D_v=*/128);
}

// 11. Asymmetric key/value head dimensions (D_k != D_v).
TEST(GatedDeltaUpdateGpuTest, AsymmetricHeadDimensions) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/2, /*D_k=*/32, /*D_v=*/64);
}

// 12. Explicit test for head dimension 16 (4 k_slices, smallest supported
// power of 2).
TEST(GatedDeltaUpdateGpuTest, HeadDimension16) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/2, /*N=*/2, /*D_k=*/16, /*D_v=*/16);
}

void ExpectGatedDeltaUpdateRejected(int B, int H, int N, int D_k, int D_v) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif

  auto model_buf = CreateGatedDeltaUpdateModelBuffer(B, H, N, D_k, D_v);
  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buf.data()), model_buf.size());
  ASSERT_NE(model, nullptr);

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  auto options = MlDriftClDelegateDefaultOptionsPtr();
  options->model_token = "rejection_test_token";
  options->serialization_dir = nullptr;
  options->runtime_context = LrtGetRuntimeContext();

  auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
  ASSERT_NE(delegate, nullptr);

  EXPECT_NE(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
            kTfLiteOk);

  LiteRtDestroyEnvironment(environment);
}

// 13. Odd head dimension 19 (not a multiple of 4 or power of 2) must be
// rejected.
TEST(GatedDeltaUpdateGpuTest, RejectOddHeadDimension19) {
  ExpectGatedDeltaUpdateRejected(/*B=*/1, /*H=*/1, /*N=*/1, /*D_k=*/19,
                                 /*D_v=*/19);
}

// 14. Large power-of-2 head dimension 256 (64 k_slices, exceeding 32-lane SIMD
// width). Exercises shared-memory reduction path across all backends.
TEST(GatedDeltaUpdateGpuTest, HeadDimension256) {
  RunGatedDeltaUpdateTest(/*B=*/1, /*H=*/1, /*N=*/2, /*D_k=*/256, /*D_v=*/256);
}

// 15. Non-power-of-2 head dimension 384 (256 + 128 = 384, 96 k_slices).
// Verifies that non-power-of-2 dimensions are rejected to guarantee high
// performance and exact power-of-2 reduction without runtime folding overhead.
TEST(GatedDeltaUpdateGpuTest, RejectNonPowerOfTwoHeadDimension384) {
  ExpectGatedDeltaUpdateRejected(/*B=*/1, /*H=*/1, /*N=*/1, /*D_k=*/384,
                                 /*D_v=*/384);
}

}  // namespace
}  // namespace litert::ml_drift
