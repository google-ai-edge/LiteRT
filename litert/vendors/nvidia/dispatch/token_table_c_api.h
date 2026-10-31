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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TOKEN_TABLE_C_API_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TOKEN_TABLE_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// The text embedder's token table, for a decode step that feeds itself
// (bytecode.h, TensorRtStepFeed) when the step's bytecode carries no table of
// its own. The host's embedding of a token is the token's INT4 row times its
// scale, then times the multiplier, each product rounded to float32.
typedef struct {
  uint32_t vocab;
  uint32_t dim;
  // INT4 row-major [vocab, dim], the low nibble first: vocab * dim / 2 bytes,
  // in memory (table), or in a file (table null): read from the open
  // descriptor table_fd at table_offset during the call.
  const void* table;
  size_t table_bytes;
  int32_t table_fd;
  uint64_t table_offset;
  // With scale_block 0, FP32 [vocab]: one scale per row. Otherwise FP16
  // [vocab, dim / scale_block]: one per block of scale_block columns.
  const void* scales;
  size_t scales_bytes;
  uint32_t scale_block;
  float multiplier;
} LiteRtDispatchNvidiaTokenTable;

// Registers a token table for the decode steps that await one of its vocab
// and dim. The table and the scales are copied to the device before the call
// returns; the caller's memory and descriptor are not read afterwards, and
// the device copy lives as long as the process. Returns
// kLiteRtStatusErrorNotFound when no step awaits such a table (nothing is
// copied), kLiteRtStatusErrorInvalidArgument for a malformed record,
// kLiteRtStatusErrorUnsupported before the dispatch is initialized.
LiteRtStatus LiteRtDispatchNvidiaRegisterTokenTable(
    const LiteRtDispatchNvidiaTokenTable* table);

// The local attention's sliding window (keys a position reads), for the
// masks the decode steps ahead fill, when the model's bytecode does not say:
// told before the first decode step, the feed's first mask past the window
// is right; otherwise the feed learns it from the host's masks, one step
// late. Process-wide; 0 forgets it.
LiteRtStatus LiteRtDispatchNvidiaStepFeedSetLocalWindow(uint32_t window);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TOKEN_TABLE_C_API_H_
