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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/core/options.h"
#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.h"
#include "ml_drift_delegate/delegate/gated_delta_update_test_util.h"

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

TEST(GatedDeltaUpdateBenchmark, BenchmarkOp) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU benchmark is not supported in MSAN/TSAN";
#endif

  // Qwen 3.5 0.8B representative dimensions: H=16, D_k=128, D_v=128, L=128
  int B = 1;
  int H = 16;
  int L = 128;
  int D_k = 128;
  int D_v = 128;

  std::vector<uint8_t> model_buffer =
      CreateGatedDeltaUpdateModelBuffer(B, H, L, D_k, D_v);

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  auto options = CreateGpuOptions();
  ASSERT_TRUE(options);
  auto compiled_model = CompiledModel::Create(
      *env,
      litert::BufferRef<uint8_t>(model_buffer.data(), model_buffer.size()),
      *options);
  ASSERT_TRUE(compiled_model);

  auto input_buffers = compiled_model->CreateInputBuffers();
  ASSERT_TRUE(input_buffers);
  auto output_buffers = compiled_model->CreateOutputBuffers();
  ASSERT_TRUE(output_buffers);

  ASSERT_EQ(input_buffers->size(), 6);
  ASSERT_EQ(output_buffers->size(), 2);

  // Populate inputs
  std::srand(42);
  auto q_data = GenerateRandom(B * H * L * D_k, -0.5f, 0.5f);
  auto k_data = GenerateRandom(B * H * L * D_k, -0.5f, 0.5f);
  auto v_data = GenerateRandom(B * H * L * D_v, -0.5f, 0.5f);
  auto beta_data = GenerateRandom(B * H * L, 0.1f, 0.9f);
  auto g_data = GenerateRandom(B * H * L, -1.0f, -0.1f);
  auto state_data = GenerateRandom(B * H * D_k * D_v, -0.5f, 0.5f);

  ASSERT_TRUE((*input_buffers)[0].Write<float>(absl::MakeConstSpan(q_data)));
  ASSERT_TRUE((*input_buffers)[1].Write<float>(absl::MakeConstSpan(k_data)));
  ASSERT_TRUE((*input_buffers)[2].Write<float>(absl::MakeConstSpan(v_data)));
  ASSERT_TRUE(
      (*input_buffers)[3].Write<float>(absl::MakeConstSpan(beta_data)));
  ASSERT_TRUE((*input_buffers)[4].Write<float>(absl::MakeConstSpan(g_data)));
  ASSERT_TRUE(
      (*input_buffers)[5].Write<float>(absl::MakeConstSpan(state_data)));

  std::cerr
      << "\n================ GATED DELTA UPDATE BENCHMARK ================\n";
  std::cerr << "Batch: " << B << ", Heads: " << H << ", SeqLen: " << L
            << ", D_k: " << D_k << ", D_v: " << D_v << "\n";

  // Warmup GPU
  constexpr int kWarmupIters = 10;
  for (int i = 0; i < kWarmupIters; ++i) {
    ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));
  }

  // Benchmark GPU
  constexpr int kBenchIters = 100;
  std::vector<double> latencies_us;
  latencies_us.reserve(kBenchIters);

  for (int i = 0; i < kBenchIters; ++i) {
    absl::Time t0 = absl::Now();
    ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));
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
  std::cerr << "  Min:    " << min_us << " us (" << min_us / 1000.0 << " ms)\n";
  std::cerr << "  Avg:    " << avg_us << " us (" << avg_us / 1000.0 << " ms)\n";
  std::cerr << "  Median: " << p50_us << " us (" << p50_us / 1000.0 << " ms)\n";
  std::cerr << "  P90:    " << p90_us << " us (" << p90_us / 1000.0 << " ms)\n";
  std::cerr << "  Max:    " << max_us << " us (" << max_us / 1000.0 << " ms)\n";
  std::cerr
      << "==============================================================\n\n";
}

}  // namespace
}  // namespace litert::ml_drift
