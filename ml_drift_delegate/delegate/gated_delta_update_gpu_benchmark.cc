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
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
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
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

void GenerateRandom(TensorBuffer& buffer, float min_val = 0.0f,
                    float max_val = 1.0f) {
  auto host_mem = buffer.Lock(TensorBuffer::LockMode::kWrite);
  ASSERT_TRUE(host_mem);
  auto packed_size = buffer.PackedSize();
  ASSERT_TRUE(packed_size);
  size_t num_elements = *packed_size / sizeof(float);
  float* ptr = static_cast<float*>(*host_mem);
  for (size_t i = 0; i < num_elements; ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    ptr[i] = min_val + r * (max_val - min_val);
  }
  ASSERT_TRUE(buffer.Unlock());
}

Expected<litert::Options> CreateGpuOptions() {
  LITERT_ASSIGN_OR_RETURN(litert::Options options, litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  LITERT_ASSIGN_OR_RETURN(auto& gpu_opts, options.GetGpuOptions());
  LITERT_RETURN_IF_ERROR(gpu_opts.EnableBenchmarkMode(true));

  static TfLiteRegistration reg =
      *litert_torch::gdn_kernels::GetGatedDeltaUpdateRegistration();
  reg.custom_name = "gated_delta_update";
  options.AddBuildAction([&](auto*, LiteRtOptions opts) {
    auto* opts_impl = reinterpret_cast<LiteRtOptionsT*>(opts);
    opts_impl->custom_tflite_op_registrations.push_back(&reg);
    return kLiteRtStatusOk;
  });
  return std::move(options);
}

void RunBenchmarkCase(const std::string& name, int B, int H_k, int H_v, int L,
                      int D_k, int D_v, int valid_L = -1) {
  const bool has_valid_len = (valid_L >= 0);
  std::vector<uint8_t> model_buffer = CreateGatedDeltaUpdateModelBuffer(
      B, H_v, L, D_k, D_v, /*mode=*/0, H_k, /*state_dtype=*/"float32",
      has_valid_len);

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

  std::srand(42);
  GenerateRandom((*input_buffers)[0], -0.5f, 0.5f);
  GenerateRandom((*input_buffers)[1], -0.5f, 0.5f);
  GenerateRandom((*input_buffers)[2], -0.5f, 0.5f);
  GenerateRandom((*input_buffers)[3], 0.1f, 0.9f);
  GenerateRandom((*input_buffers)[4], -1.0f, -0.1f);
  GenerateRandom((*input_buffers)[5], -0.5f, 0.5f);

  if (has_valid_len) {
    std::vector<int32_t> valid_len_vec(B, valid_L);
    ASSERT_TRUE(
        (*input_buffers)[6].Write<int32_t>(absl::MakeConstSpan(valid_len_vec)));
  }

  constexpr int kWarmupIters = 10;
  for (int i = 0; i < kWarmupIters; ++i) {
    ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));
  }

  constexpr int kBenchIters = 50;
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
  double p50_us = latencies_us[kBenchIters / 2];
  double avg_us =
      std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
      kBenchIters;
  std::cerr << std::fixed << std::setprecision(2) << "  [" << name
            << "] L=" << L
            << (valid_L >= 0 ? " (valid=" + std::to_string(valid_L) + ")" : "")
            << " | Min: " << min_us << " us | P50: " << p50_us
            << " us | Avg: " << avg_us
            << " us | 48-layer P50: " << (p50_us * 48.0 / 1000.0) << " ms\n";
}

TEST(GatedDeltaUpdateBenchmark, BenchmarkQwen27B) {
  std::cerr << "\n=== Qwen3.8-27B GDN Kernel Micro-Benchmark (H_k=16, H_v=48, "
               "D=128) ===\n";
  RunBenchmarkCase("Decode L=1", 1, 16, 48, 1, 128, 128, -1);

  std::cerr << "--- Fixed L=128 ---\n";
  RunBenchmarkCase("Baseline (no valid_len)", 1, 16, 48, 128, 128, 128, -1);
  for (int valid_L : {16, 32, 64, 128}) {
    RunBenchmarkCase("Explicit valid_len", 1, 16, 48, 128, 128, 128, valid_L);
  }

  std::cerr << "--- Fixed L=256 ---\n";
  RunBenchmarkCase("Baseline (no valid_len)", 1, 16, 48, 256, 128, 128, -1);
  for (int valid_L : {16, 32, 64, 128, 256}) {
    RunBenchmarkCase("Explicit valid_len", 1, 16, 48, 256, 128, 128, valid_L);
  }

  std::cerr << "--- Fixed L=512 ---\n";
  RunBenchmarkCase("Baseline (no valid_len)", 1, 16, 48, 512, 128, 128, -1);
  for (int valid_L : {128, 256, 512}) {
    RunBenchmarkCase("Explicit valid_len", 1, 16, 48, 512, 128, 128, valid_L);
  }

  std::cerr << "--- Fixed L=1024 ---\n";
  RunBenchmarkCase("Baseline (no valid_len)", 1, 16, 48, 1024, 128, 128, -1);
  for (int valid_L : {128, 256, 512, 1024}) {
    RunBenchmarkCase("Explicit valid_len", 1, 16, 48, 1024, 128, 128, valid_L);
  }
  std::cerr << "==============================================================="
               "=======\n\n";
}

}  // namespace
}  // namespace litert::ml_drift
