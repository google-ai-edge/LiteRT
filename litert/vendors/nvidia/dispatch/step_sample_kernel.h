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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_STEP_SAMPLE_KERNEL_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_STEP_SAMPLE_KERNEL_H_

#include <cstdint>

#include "cuda_runtime_api.h"

// Samples one token from a row's k candidates as LiteRT-LM's CPU sampler
// does (runtime/components/sampling_cpu_util.cc, TopKTopPSampling, over the
// candidates in its order: logit descending, then index ascending):
// probabilities exp((v - v[0]) / max(temperature, FLT_EPSILON)) in float32
// summed in order, the zero-sum and infinite-sum fallbacks, each divided by
// the sum through a float64 reciprocal; the smallest prefix whose float64
// cumulative probability reaches p; the draw uniform * cumulative in
// float64; the first candidate whose running float64 sum reaches the draw.
// A row flagged by the candidate selection (a NaN) gives its first
// candidate. `token` gets the candidate's id, on the device.
extern "C" cudaError_t LiteRtNvidiaLaunchStepSample(
    const int32_t* ids, const float* values, const int32_t* flags, int32_t k,
    float p, float temperature, double uniform, int32_t* token,
    cudaStream_t stream);

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_STEP_SAMPLE_KERNEL_H_
