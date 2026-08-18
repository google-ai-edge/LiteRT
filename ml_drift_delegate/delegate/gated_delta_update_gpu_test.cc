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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/core/options.h"
#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.h"
#include "ml_drift_delegate/delegate/gated_delta_update_test_util.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

std::vector<float> GenerateRandom(size_t n, float min_val = 0.0f,
                                  float max_val = 1.0f) {
  std::vector<float> data(n);
  for (size_t i = 0; i < n; ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    data[i] = min_val + r * (max_val - min_val);
  }
  return data;
}

std::vector<float> GenerateConstant(size_t n, float val) {
  return std::vector<float>(n, val);
}

void CompareBuffers(absl::Span<const float> actual,
                    absl::Span<const float> expected, float abs_tolerance,
                    float rel_tolerance = 1e-4) {
  ASSERT_EQ(actual.size(), expected.size());
  int mismatches = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    float diff = std::abs(actual[i] - expected[i]);
    float max_val = std::max(std::abs(actual[i]), std::abs(expected[i]));
    if (diff > abs_tolerance &&
        (max_val == 0.0f || (diff / max_val) > rel_tolerance)) {
      if (mismatches < 10) {
        EXPECT_NEAR(actual[i], expected[i], abs_tolerance)
            << "Mismatch at index " << i
            << " (rel_diff=" << (max_val > 0 ? diff / max_val : 0) << ")";
      }
      mismatches++;
    }
  }
  if (mismatches > 0) {
    FAIL() << "Total mismatches: " << mismatches << " / " << actual.size();
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

Expected<litert::Options> CreateGpuOptions() {
  LITERT_ASSIGN_OR_RETURN(litert::Options options, litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
  LITERT_RETURN_IF_ERROR(gpu_options.EnableExternalTensorsMode(false));

  static TfLiteRegistration reg =
      *litert_torch::gdn_kernels::GetGatedDeltaUpdateRegistration();
  reg.custom_name = "gated_delta_update";
  options.AddBuildAction([&](internal::RuntimeProxy*, LiteRtOptions opts) {
    auto* opts_impl = reinterpret_cast<LiteRtOptionsT*>(opts);
    opts_impl->custom_tflite_op_registrations.push_back(&reg);
    return kLiteRtStatusOk;
  });
  return std::move(options);
}

void RunGatedDeltaUpdateTest(int B, int H, int N, int D_k, int D_v,
                             bool zero_initial_state = false,
                             float beta_fixed = -1.0f, float g_fixed = 100.0f,
                             float tolerance = 1e-2) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif

  auto model_buf = CreateGatedDeltaUpdateModelBuffer(B, H, N, D_k, D_v);

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  auto options = CreateGpuOptions();
  ASSERT_TRUE(options);

  auto compiled_model = CompiledModel::Create(
      *env,
      litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
      *options);
  ASSERT_TRUE(compiled_model);

  auto input_buffers = compiled_model->CreateInputBuffers();
  ASSERT_TRUE(input_buffers);
  auto output_buffers = compiled_model->CreateOutputBuffers();
  ASSERT_TRUE(output_buffers);

  ASSERT_EQ(input_buffers->size(), 6);
  ASSERT_EQ(output_buffers->size(), 2);

  // Populate inputs.
  std::srand(0);
  auto q_data = GenerateRandom(B * H * N * D_k, -0.5f, 0.5f);
  auto k_data = GenerateRandom(B * H * N * D_k, -0.5f, 0.5f);
  auto v_data = GenerateRandom(B * H * N * D_v, -0.5f, 0.5f);
  auto beta_data = (beta_fixed >= 0.0f)
                       ? GenerateConstant(B * H * N, beta_fixed)
                       : GenerateRandom(B * H * N, 0.1f, 0.9f);
  auto g_data = (g_fixed <= 0.0f) ? GenerateConstant(B * H * N, g_fixed)
                                  : GenerateRandom(B * H * N, -1.0f, -0.1f);
  auto state_data = zero_initial_state
                        ? GenerateConstant(B * H * D_k * D_v, 0.0f)
                        : GenerateRandom(B * H * D_k * D_v, -0.5f, 0.5f);

  ASSERT_TRUE((*input_buffers)[0].Write<float>(absl::MakeConstSpan(q_data)));
  ASSERT_TRUE((*input_buffers)[1].Write<float>(absl::MakeConstSpan(k_data)));
  ASSERT_TRUE((*input_buffers)[2].Write<float>(absl::MakeConstSpan(v_data)));
  ASSERT_TRUE(
      (*input_buffers)[3].Write<float>(absl::MakeConstSpan(beta_data)));
  ASSERT_TRUE((*input_buffers)[4].Write<float>(absl::MakeConstSpan(g_data)));
  ASSERT_TRUE(
      (*input_buffers)[5].Write<float>(absl::MakeConstSpan(state_data)));

  // Golden reference computation.
  std::vector<float> golden_out(B * H * N * D_v);
  std::vector<float> golden_final_state(B * H * D_k * D_v);
  ComputeGoldenRecurrentGatedDelta(
      q_data.data(), k_data.data(), v_data.data(), beta_data.data(),
      g_data.data(), state_data.data(), golden_out.data(),
      golden_final_state.data(), B, H, N, D_k, D_v);

  // Execute GPU compiled model.
  ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));

  // Verify GPU outputs match golden reference.
  std::vector<float> actual_out(B * H * N * D_v);
  std::vector<float> actual_final_state(B * H * D_k * D_v);
  ASSERT_TRUE((*output_buffers)[0].Read<float>(absl::MakeSpan(actual_out)));
  ASSERT_TRUE(
      (*output_buffers)[1].Read<float>(absl::MakeSpan(actual_final_state)));

  CompareBuffers(actual_out, golden_out, tolerance);
  CompareBuffers(actual_final_state, golden_final_state, tolerance);
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
  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  auto options = CreateGpuOptions();
  ASSERT_TRUE(options);

  auto compiled_model = CompiledModel::Create(
      *env,
      litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
      *options);
  EXPECT_FALSE(compiled_model);
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

// 16. Multi-batch test with large head dimension 256 (exercising shared-memory
// reduction path with B > 1).
TEST(GatedDeltaUpdateGpuTest, MultiBatchHeadDimension256) {
  RunGatedDeltaUpdateTest(/*B=*/2, /*H=*/2, /*N=*/2, /*D_k=*/256, /*D_v=*/256);
}

}  // namespace
}  // namespace litert::ml_drift

