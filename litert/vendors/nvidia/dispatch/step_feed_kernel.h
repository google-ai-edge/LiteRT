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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_STEP_FEED_KERNEL_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_STEP_FEED_KERNEL_H_

#include <cstdint>

#include "cuda_runtime_api.h"

// A decode step feeds itself (bytecode.h, TensorRtStepFeed): from the greedy
// token of a step, the next step's inputs are written in place on the device,
// and copied to host-mapped mirrors the dispatch verifies the host's own
// inputs against.
//
// The embedding is the token's row of the INT4 table (the low nibble first,
// two's complement) times its scale, then times the multiplier, each product
// rounded to nearest in float32: the arithmetic of LiteRT-LM's CPU embedder
// (tfl.embedding_lookup, then tfl.mul). The scale is the row's (one FP32 per
// row) or the block's (one FP16 per `scale_block` columns of the row, as the
// blockwise lookup converts it to float32).
//
// The position `p` is read from `positions` before anything is written. The
// local mask of position p + 1 is LiteRT-LM's ring-buffer decode mask: key
// column c is readable iff distance = (p + 1 - c) mod ring is at most
// window - 1, at most p + 1 and at most ring - 1. Then positions becomes
// p + 1, the cache parameter {p + 1, p + 2, p + 2} (its other values stay),
// and the global causal mask gains key p + 1.
struct LiteRtNvidiaStepFeedParams {
  const int32_t* token;     // [1] the greedy token, on the device
  const uint8_t* table;     // INT4 [vocab, dim]
  const float* row_scales;  // [vocab], or null with block scales
  // FP16 bits [vocab, dim / scale_block], or null with row scales.
  const uint16_t* block_scales;
  int32_t scale_block;
  float multiplier;
  int32_t vocab;
  int32_t dim;
  float* embeddings;    // [dim]
  int32_t* positions;   // [1]
  int32_t* param;       // [>= 3], or null
  uint8_t* local_mask;  // [ring], or null
  int32_t ring;
  int32_t window;
  uint8_t* global_mask;  // [global_keys], or null
  int32_t global_keys;
  // A device counter the launch owns, zero before the first launch: the
  // blocks count themselves done and the last one advances the scalars.
  unsigned int* blocks_done;
  // Host-mapped mirrors of what was written (null: not mirrored).
  int32_t* token_mirror;      // [1]
  float* embeddings_mirror;   // [dim]
  int32_t* positions_mirror;  // [1]
  int32_t* param_mirror;      // [param_count]
  int32_t param_count;
  uint8_t* local_mask_mirror;  // [ring]
};

// Enqueues the feed on `stream`, one launch. An out-of-range token (negative
// or at least vocab) gives a zero embedding and is mirrored as is. Exactly one
// of row_scales and block_scales is given; dim is a multiple of scale_block.
extern "C" cudaError_t LiteRtNvidiaLaunchStepFeed(
    const LiteRtNvidiaStepFeedParams* params, cudaStream_t stream);

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_STEP_FEED_KERNEL_H_
