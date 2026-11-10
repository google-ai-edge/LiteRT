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

#include "litert/vendors/nvidia/dispatch/step_sample_kernel.h"

#include <cfloat>
#include <cstdint>

#include "cuda_runtime.h"

namespace {

constexpr int kMaxK = 1024;

// One thread: the CPU sampler's arithmetic on k values, in its order.
__global__ void StepSampleKernel(const int32_t* ids, const float* values,
                                 const int32_t* flags, int k, float p,
                                 float temperature, double uniform,
                                 int32_t* token) {
  __shared__ float probabilities[kMaxK];
  if (threadIdx.x != 0) return;
  if (flags != nullptr && flags[0] != 0) {
    *token = ids[0];
    return;
  }
  const float max_logit = values[0];
  const float current_temp = fmaxf(temperature, FLT_EPSILON);
  float sum_of_exps = 0.0f;
  for (int i = 0; i < k; ++i) {
    probabilities[i] = expf((values[i] - max_logit) / current_temp);
    sum_of_exps += probabilities[i];
  }
  if (sum_of_exps <= FLT_EPSILON) {
    const float uniform_prob = 1.0 / static_cast<float>(k);
    for (int i = 0; i < k; ++i) probabilities[i] = uniform_prob;
  } else if (isinf(sum_of_exps)) {
    for (int i = 0; i < k; ++i) probabilities[i] = 0.0f;
    probabilities[0] = 1.0f;
  } else {
    const float inv_sum = 1.0 / sum_of_exps;
    for (int i = 0; i < k; ++i) probabilities[i] *= inv_sum;
  }
  // The candidates come in the order the CPU sorts them by probability
  // (their logits descend, so their probabilities do; equal ones keep their
  // order, as the CPU's stable sort does).
  double cumulative_prob = 0.0;
  int final_sample_size = 0;
  for (int i = 0; i < k; ++i) {
    cumulative_prob += probabilities[i];
    final_sample_size = i + 1;
    if (cumulative_prob >= p) break;
  }
  if (cumulative_prob <= DBL_EPSILON) {
    *token = ids[0];
    return;
  }
  const double random_sample = (uniform * (cumulative_prob - 0.0)) + 0.0;
  double current_cumulative = 0.0;
  int picked = -1;
  for (int i = 0; i < final_sample_size; ++i) {
    current_cumulative += probabilities[i];
    if (random_sample <= current_cumulative) {
      picked = i;
      break;
    }
  }
  // When rounding keeps the running sum below the draw, the CPU's result
  // stays at its initial value: the first candidate.
  *token = ids[picked < 0 ? 0 : picked];
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchStepSample(
    const int32_t* ids, const float* values, const int32_t* flags, int32_t k,
    float p, float temperature, double uniform, int32_t* token,
    cudaStream_t stream) {
  if (ids == nullptr || values == nullptr || token == nullptr || k <= 0 ||
      k > kMaxK || !(p >= 0.0f && p <= 1.0f) || !(temperature >= 0.0f) ||
      !(uniform >= 0.0 && uniform < 1.0)) {
    return cudaErrorInvalidValue;
  }
  StepSampleKernel<<<1, 32, 0, stream>>>(ids, values, flags, k, p, temperature,
                                         uniform, token);
  return cudaPeekAtLastError();
}
