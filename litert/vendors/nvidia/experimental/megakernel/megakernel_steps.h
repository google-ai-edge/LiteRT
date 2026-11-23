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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_STEPS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_STEPS_H_

#include <cstdint>

#include "cuda_runtime_api.h"

// The steps of a Gemma 4 mixture-of-experts decode step that the engine's
// graph runs as TensorRT layers between the CUDA kernels (the embedding
// cast, the cache writes, the norm and residual after the attention, the
// final norm and the soft cap), for the megakernel plugin's step interpreted
// on the host (megakernel_plugin.h). Activations are BF16; each step follows
// the graph's layers, rounding to BF16 where the graph holds a BF16 tensor
// (constants included).

// out[i] = BF16(in[i]).
extern "C" cudaError_t LiteRtNvidiaMegakernelToBf16(const float* in, void* out,
                                                    int32_t count,
                                                    cudaStream_t stream);

// The qkv rows of a layer whose values are its keys ([q_width + k_width]) as
// the qkv of one with values ([q_width + 2 k_width]): the keys twice.
extern "C" cudaError_t LiteRtNvidiaMegakernelRepeatKeys(const void* qkv,
                                                        void* out,
                                                        int32_t q_width,
                                                        int32_t k_width,
                                                        cudaStream_t stream);

// The token's key and value rows [heads, depth] (BF16) into the FP16 caches
// [heads, len, depth] at row position mod len (ring) or the position clamped
// to the cache.
extern "C" cudaError_t LiteRtNvidiaMegakernelWriteCacheRows(
    const void* k, const void* v, void* k_cache, void* v_cache,
    const int32_t* position, int32_t heads, int32_t len, int32_t depth,
    int32_t ring, cudaStream_t stream);

// h = x + rms_norm(o) * scale over `dim` values (BF16 in and out), the norm
// as the graph's layers compute it: o / sqrt(mean(o * o) + eps) * scale.
extern "C" cudaError_t LiteRtNvidiaMegakernelNormResidual(
    const void* o, const void* x, const float* scale, float eps, void* h,
    int32_t dim, cudaStream_t stream);

// out = FP32(BF16(rms_norm(x) * scale)) over `dim` values.
extern "C" cudaError_t LiteRtNvidiaMegakernelNormToFloat(const void* x,
                                                         const float* scale,
                                                         float eps, float* out,
                                                         int32_t dim,
                                                         cudaStream_t stream);

// out = FP32(tanh(z * scale_in) * scale_out), BF16 at every step between.
extern "C" cudaError_t LiteRtNvidiaMegakernelSoftCap(const void* z,
                                                     float scale_in,
                                                     float scale_out,
                                                     float* out, int32_t count,
                                                     cudaStream_t stream);

// The rows a decode step wrote into its layers' FP16 caches, read back for
// the engine's native in-place cache writes: per layer the row its token
// went to and the key and value rows there.
constexpr int kLiteRtNvidiaMegakernelRowsMaxLayers = 64;

struct LiteRtNvidiaMegakernelRowsLayer {
  // Caches [heads, len, depth]; rows out [heads, depth].
  const void* k_cache;
  const void* v_cache;
  void* k_row;
  void* v_row;
  int32_t heads;
  int32_t len;
  int32_t depth;
  int32_t ring;
};

struct LiteRtNvidiaMegakernelRowsParams {
  int32_t layers;
  const int32_t* position;  // [1]
  int32_t* rows;            // [layers]
  LiteRtNvidiaMegakernelRowsLayer layer[kLiteRtNvidiaMegakernelRowsMaxLayers];
};

extern "C" cudaError_t LiteRtNvidiaLaunchMegakernelRows(
    const LiteRtNvidiaMegakernelRowsParams* params, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_STEPS_H_
