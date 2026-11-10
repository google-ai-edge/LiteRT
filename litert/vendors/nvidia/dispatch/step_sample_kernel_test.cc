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

// The device sampling of a step ahead against LiteRT-LM's CPU sampler
// (runtime/components/sampling_cpu_util.cc, TopKTopPSampling on the
// candidates in its order), transcribed here with the draw given.

#include "litert/vendors/nvidia/dispatch/step_sample_kernel.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"

namespace {

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation AllocateCuda(size_t bytes) {
  void* pointer = nullptr;
  if (cudaMalloc(&pointer, bytes) != cudaSuccess) {
    ADD_FAILURE() << "cudaMalloc(" << bytes << ") failed";
  }
  return CudaAllocation(pointer);
}

// TopKTopPSampling for one row of k candidates (the vocabulary is the
// candidates), with `uniform` for its draw: the compact index it picks, and
// how far the draw lies from the nearest boundary between two candidates,
// relative to the cumulative probability (where a float32 ulp could move the
// pick).
struct HostPick {
  int index = 0;
  double margin = 1.0;
};
HostPick HostSample(const std::vector<float>& values, float p,
                    float temperature, double uniform) {
  const int k = static_cast<int>(values.size());
  const float max_logit = *std::max_element(values.begin(), values.end());
  float sum_of_exps = 0.0;
  const float current_temp =
      std::max(temperature, std::numeric_limits<float>::epsilon());
  std::vector<float> probabilities(k);
  for (int i = 0; i < k; ++i) {
    probabilities[i] = std::exp((values[i] - max_logit) / current_temp);
    sum_of_exps += probabilities[i];
  }
  if (sum_of_exps <= std::numeric_limits<float>::epsilon()) {
    float uniform_prob = 1.0 / static_cast<float>(k);
    std::fill(probabilities.begin(), probabilities.end(), uniform_prob);
  } else if (std::isinf(sum_of_exps)) {
    const int max_logit_idx = static_cast<int>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
    std::fill(probabilities.begin(), probabilities.end(), 0.0f);
    probabilities[max_logit_idx] = 1.0f;
  } else {
    float inv_sum = 1.0 / sum_of_exps;
    for (int i = 0; i < k; ++i) probabilities[i] *= inv_sum;
  }
  std::vector<int> index_of_topk(k);
  for (int i = 0; i < k; ++i) index_of_topk[i] = i;
  std::stable_sort(
      index_of_topk.begin(), index_of_topk.end(),
      [&](int i1, int i2) { return probabilities[i1] > probabilities[i2]; });
  double cumulative_prob = 0.0;
  int final_sample_size = 0;
  for (int i = 0; i < k; ++i) {
    cumulative_prob += probabilities[index_of_topk[i]];
    final_sample_size = i + 1;
    if (cumulative_prob >= p) break;
  }
  HostPick pick;
  if (cumulative_prob <= std::numeric_limits<double>::epsilon()) {
    pick.index = index_of_topk[0];
    return pick;
  }
  const double random_sample = (uniform * (cumulative_prob - 0.0)) + 0.0;
  double current_cumulative = 0.0;
  pick.index = 0;  // the CPU's initial value when no candidate is reached
  bool picked = false;
  for (int i = 0; i < final_sample_size; ++i) {
    current_cumulative += probabilities[index_of_topk[i]];
    pick.margin =
        std::min(pick.margin, std::fabs(random_sample - current_cumulative) /
                                  cumulative_prob);
    if (!picked && random_sample <= current_cumulative) {
      pick.index = index_of_topk[i];
      picked = true;
    }
  }
  return pick;
}

class StepSampleKernelTest : public testing::Test {
 protected:
  // Runs the kernel over `values` (descending) and returns the compact index.
  int DeviceSample(const std::vector<float>& values, float p, float temperature,
                   double uniform, int32_t flag = 0) {
    const int k = static_cast<int>(values.size());
    std::vector<int32_t> ids(k);
    for (int i = 0; i < k; ++i) ids[i] = 1000 + i;
    auto device_ids = AllocateCuda(k * sizeof(int32_t));
    auto device_values = AllocateCuda(k * sizeof(float));
    auto device_flags = AllocateCuda(sizeof(int32_t));
    auto device_token = AllocateCuda(sizeof(int32_t));
    EXPECT_EQ(cudaMemcpy(device_ids.get(), ids.data(), k * sizeof(int32_t),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(device_values.get(), values.data(), k * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(device_flags.get(), &flag, sizeof(flag),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(
        LiteRtNvidiaLaunchStepSample(
            static_cast<const int32_t*>(device_ids.get()),
            static_cast<const float*>(device_values.get()),
            static_cast<const int32_t*>(device_flags.get()), k, p, temperature,
            uniform, static_cast<int32_t*>(device_token.get()), nullptr),
        cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    int32_t token = -1;
    EXPECT_EQ(cudaMemcpy(&token, device_token.get(), sizeof(token),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return token - 1000;
  }
};

TEST_F(StepSampleKernelTest, PicksAsTheHostDoes) {
  uint32_t state = 2024u;
  const auto next = [&state] {
    state = state * 1664525u + 1013904223u;
    return (state >> 8) / 16777216.0;  // [0, 1)
  };
  int trials = 0;
  int mismatches = 0;
  for (const int k : {2, 8, 64, 256}) {
    for (const float p : {1.0f, 0.95f, 0.5f}) {
      for (const float temperature : {1.0f, 0.7f, 1.5f}) {
        for (int trial = 0; trial < 40; ++trial) {
          // Logits descending, a sharper or flatter top, on the fp16 grid (as
          // a float16 head leaves them: many ties).
          std::vector<float> values(k);
          const float spread = 0.5f + 6.0f * static_cast<float>(next());
          for (int i = 0; i < k; ++i) {
            const float value = 20.0f -
                                spread * std::log1p(static_cast<float>(i)) -
                                static_cast<float>(next()) * 0.3f;
            values[i] = std::round(value * 16.0f) / 16.0f;
          }
          std::sort(values.begin(), values.end(), std::greater<float>());
          const double uniform = next();
          const HostPick host = HostSample(values, p, temperature, uniform);
          const int device = DeviceSample(values, p, temperature, uniform);
          ++trials;
          if (device != host.index) {
            ++mismatches;
            // Only a draw at a boundary may fall on the other side.
            EXPECT_LT(host.margin, 1e-5)
                << "k " << k << " p " << p << " temperature " << temperature
                << " uniform " << uniform << ": host " << host.index
                << " device " << device;
          }
        }
      }
    }
  }
  EXPECT_LE(mismatches * 1000, trials);  // under 0.1%
}

TEST_F(StepSampleKernelTest, EdgesAsTheHost) {
  // A flagged row (a NaN among the logits) gives the first candidate.
  EXPECT_EQ(DeviceSample({3.0f, 2.0f, 1.0f}, 1.0f, 1.0f, 0.99, /*flag=*/1), 0);
  // p of 0 keeps the first candidate alone.
  EXPECT_EQ(DeviceSample({3.0f, 2.0f, 1.0f}, 0.0f, 1.0f, 0.99), 0);
  // A temperature of 0 (clamped to epsilon) makes the top certain: the
  // other exponents underflow to 0.
  EXPECT_EQ(DeviceSample({3.0f, 2.0f, 1.0f}, 1.0f, 0.0f, 0.99), 0);
  // Equal logits: uniform probabilities, the draw picks by thirds, in the
  // candidates' order (the CPU's stable sort keeps it).
  EXPECT_EQ(DeviceSample({1.0f, 1.0f, 1.0f}, 1.0f, 1.0f, 0.5), 1);
  EXPECT_EQ(DeviceSample({1.0f, 1.0f, 1.0f}, 1.0f, 1.0f, 0.9), 2);
  // Ties among the top, as fp16 logits give them: the pick keeps the order.
  EXPECT_EQ(DeviceSample({2.0f, 2.0f, 2.0f, 1.0f, 1.0f}, 1.0f, 1.0f, 0.1), 0);
  EXPECT_EQ(DeviceSample({2.0f, 2.0f, 2.0f, 1.0f, 1.0f}, 1.0f, 1.0f, 0.5), 1);
  EXPECT_EQ(DeviceSample({2.0f, 2.0f, 2.0f, 1.0f, 1.0f}, 1.0f, 1.0f, 0.8), 2);
  // Invalid parameters are refused.
  auto device = AllocateCuda(16);
  EXPECT_EQ(LiteRtNvidiaLaunchStepSample(
                static_cast<const int32_t*>(device.get()),
                static_cast<const float*>(device.get()), nullptr, 0, 1.0f, 1.0f,
                0.5, static_cast<int32_t*>(device.get()), nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchStepSample(
                static_cast<const int32_t*>(device.get()),
                static_cast<const float*>(device.get()), nullptr, 2, 1.0f, 1.0f,
                1.0, static_cast<int32_t*>(device.get()), nullptr),
            cudaErrorInvalidValue);
}

}  // namespace
