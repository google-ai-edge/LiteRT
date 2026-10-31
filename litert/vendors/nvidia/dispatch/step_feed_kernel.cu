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

#include "litert/vendors/nvidia/dispatch/step_feed_kernel.h"

#include <cstdint>

#include "cuda_fp16.h"
#include "cuda_runtime.h"

namespace {

constexpr int kThreads = 256;

// Element `index` of a packed table of `bits` bits per element, the low bits
// first, two's complement.
__device__ __forceinline__ int Unpack(const uint8_t* table, size_t index,
                                      int bits) {
  if (bits == 4) {
    const int nibble = (table[index / 2] >> ((index & 1) * 4)) & 15;
    return (nibble ^ 8) - 8;
  }
  const int pair = (table[index / 4] >> ((index & 3) * 2)) & 3;
  return (pair ^ 2) - 2;
}

// The embedding of the token and the local mask of the next position, over the
// position as it is before the scalars advance; the block that finishes last
// (every other block has read the position by then) advances the scalars.
__global__ void StepFeedKernel(LiteRtNvidiaStepFeedParams p) {
  __shared__ bool last;
  const int32_t token = p.token[0];
  const int32_t position = p.positions[0];
  const int i = blockIdx.x * kThreads + threadIdx.x;
  const bool in_range = token >= 0 && token < p.vocab;
  if (i < p.dim) {
    float value = 0.0f;
    if (in_range) {
      const int q = Unpack(p.table, static_cast<size_t>(token) * p.dim + i,
                           p.bits);
      float scale;
      if (p.block_scales != nullptr) {
        const size_t block = static_cast<size_t>(token) * (p.dim / p.scale_block) +
                             i / p.scale_block;
        scale = __half2float(__ushort_as_half(p.block_scales[block]));
      } else {
        scale = p.row_scales[token];
      }
      // Two roundings, as the CPU embedder's dequantize then multiply.
      value = __fmul_rn(static_cast<float>(q), scale);
      value = __fmul_rn(value, p.multiplier);
    }
    p.embeddings[i] = value;
    if (p.embeddings_mirror != nullptr) p.embeddings_mirror[i] = value;
  }
  if (p.per_layer != nullptr && i < p.per_layer_count * p.per_layer_width) {
    float value = 0.0f;
    if (in_range) {
      const int layer = i / p.per_layer_width;
      const int column = i - layer * p.per_layer_width;
      const size_t index =
          (static_cast<size_t>(layer) * p.vocab + token) * p.per_layer_width +
          column;
      const int q = Unpack(p.per_layer_table, index, p.per_layer_bits);
      value = __fmul_rn(static_cast<float>(q),
                        p.per_layer_scales[static_cast<size_t>(layer) * p.vocab +
                                           token]);
      value = __fmul_rn(value, p.per_layer_multiplier);
    }
    p.per_layer[i] = value;
    if (p.per_layer_mirror != nullptr) p.per_layer_mirror[i] = value;
  }
  if (p.local_mask != nullptr && i < p.ring) {
    const int next = position + 1;
    const int diff = next - i;
    const int distance = ((diff % p.ring) + p.ring) % p.ring;
    const uint8_t readable = distance <= p.window - 1 && distance <= next &&
                                     distance <= p.ring - 1
                                 ? 1
                                 : 0;
    p.local_mask[i] = readable;
    if (p.local_mask_mirror != nullptr) p.local_mask_mirror[i] = readable;
  }
  if (i == 0 && p.token_mirror != nullptr) p.token_mirror[0] = token;
  // The scalars: the position, the cache parameter, the global mask's new
  // key, by the last block to finish.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int done = atomicAdd(p.blocks_done, 1u);
    last = done + 1 == gridDim.x;
  }
  __syncthreads();
  if (!last || threadIdx.x != 0) return;
  *p.blocks_done = 0;
  const int32_t next = position + 1;
  p.positions[0] = next;
  if (p.positions_mirror != nullptr) p.positions_mirror[0] = next;
  if (p.param != nullptr) {
    p.param[0] = next;
    p.param[1] = next + 1;
    p.param[2] = next + 1;
    if (p.param_mirror != nullptr) {
      for (int k = 0; k < p.param_count; ++k) p.param_mirror[k] = p.param[k];
    }
  }
  if (p.global_mask != nullptr && next >= 0 && next < p.global_keys) {
    p.global_mask[next] = 1;
  }
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchStepFeed(
    const LiteRtNvidiaStepFeedParams* params, cudaStream_t stream) {
  if (params == nullptr || params->token == nullptr ||
      params->table == nullptr ||
      (params->row_scales == nullptr) == (params->block_scales == nullptr) ||
      (params->block_scales != nullptr &&
       (params->scale_block <= 0 || params->dim % params->scale_block != 0)) ||
      params->embeddings == nullptr || params->positions == nullptr ||
      params->blocks_done == nullptr ||
      (params->bits != 4 && params->bits != 2) ||
      params->vocab <= 0 || params->dim <= 0 ||
      (params->dim * params->bits) % 8 != 0 ||
      (params->per_layer != nullptr &&
       (params->per_layer_table == nullptr ||
        params->per_layer_scales == nullptr || params->per_layer_count <= 0 ||
        params->per_layer_width <= 0 ||
        (params->per_layer_bits != 4 && params->per_layer_bits != 2) ||
        (params->per_layer_width * params->per_layer_bits) % 8 != 0)) ||
      (params->local_mask != nullptr &&
       (params->ring <= 0 || params->window <= 0 ||
        params->window > params->ring)) ||
      (params->global_mask != nullptr && params->global_keys <= 0) ||
      (params->param_mirror != nullptr && params->param_count < 3)) {
    return cudaErrorInvalidValue;
  }
  int span = params->dim;
  if (params->local_mask != nullptr && params->ring > span) span = params->ring;
  if (params->per_layer != nullptr &&
      params->per_layer_count * params->per_layer_width > span) {
    span = params->per_layer_count * params->per_layer_width;
  }
  const int blocks = (span + kThreads - 1) / kThreads;
  StepFeedKernel<<<blocks, kThreads, 0, stream>>>(*params);
  return cudaPeekAtLastError();
}
