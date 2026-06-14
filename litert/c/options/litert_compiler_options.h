// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_COMPILER_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_COMPILER_OPTIONS_H_

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LrtCompilerOptions LrtCompilerOptions;

// Partition strategy for the compiler.
//
// Default:
//   Default partition strategy is to partition the graph based on the naive
//   cutting algorithm.
//
// Weakly connected (experimental feature):
//   Partitions the graph into weakly connected components, this may result in
//   incorrect partition where there exists a path from a selected to another,
//   where there is an unselected node in between. This feature is currently in
//   experimental stage.
//
// Transformer block (experimental feature):
//   Detects transformer-block structure (attention, feed-forward, and
//   normalization clusters) among the ops the vendor plugin selected, and
//   groups each detected block into a single partition before requesting
//   compilation from the IHV delegate. This keeps a whole block self-contained
//   so the delegate can schedule it as one unit instead of many small islands.
//   Falls back to the default strategy when no block is found.
//
// Transformer layer cut (experimental feature):
//   Like Transformer block, but instead of one partition per transformer layer,
//   groups CONSECUTIVE layers into a single partition, breaking only at the
//   layer indices supplied via LrtSetCompilerOptionsTransformerLayerCuts. This
//   lets a caller carve the decoder stack into a chosen number of multi-layer
//   chunks (e.g. cuts {16,32} on a 48-layer model -> three 16-layer
//   partitions). With no cuts set, behaves like Transformer block.
//
//   Cuts are supplied as a per-signature spec string: a ';'-separated list of
//   "signature_key=cuts" groups, where `cuts` is a ','-separated list of layer
//   indices. A bare list (or empty key) is the default applied to signatures
//   without an explicit group. Examples:
//     "16"                          -> all signatures cut at layer 16
//     "prefill_128=16,32;decode=8"  -> per-signature cuts
//     "=16;decode=8"                -> decode at 8, all others at 16
typedef enum LiteRtCompilerOptionsPartitionStrategy {
  kLiteRtCompilerOptionsPartitionStrategyDefault = 0,
  kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected = 1,
  kLiteRtCompilerOptionsPartitionStrategyTransformerBlock = 2,
  kLiteRtCompilerOptionsPartitionStrategyTransformerLayerCut = 3,
} LiteRtCompilerOptionsPartitionStrategy;

// Creates a compiler options object.
// The caller is responsible for freeing the returned options using
// `LrtDestroyCompilerOptions`.
LiteRtStatus LrtCreateCompilerOptions(LrtCompilerOptions** options);

// Destroys a compiler options object.
void LrtDestroyCompilerOptions(LrtCompilerOptions* options);

// Serializes compiler options and returns the components needed to create
// opaque options. The caller is responsible for passing these to
// `LiteRtCreateOpaqueOptions`.
LiteRtStatus LrtGetOpaqueCompilerOptionsData(const LrtCompilerOptions* options,
                                             const char** identifier,
                                             void** payload,
                                             void (**payload_deleter)(void*));

// Gets the identifier for Compiler options stored in opaque options.
const char* LrtGetCompilerOptionsIdentifier();

// Sets the partition strategy for the compiler.
LiteRtStatus LrtSetCompilerOptionsPartitionStrategy(
    LrtCompilerOptions* options,
    LiteRtCompilerOptionsPartitionStrategy partition_strategy);

// Gets the partition strategy from compiler options.
// Returns kLiteRtStatusErrorNotFound if not set.
LiteRtStatus LrtGetCompilerOptionsPartitionStrategy(
    const LrtCompilerOptions* options,
    LiteRtCompilerOptionsPartitionStrategy* partition_strategy);

// Sets the transformer layer cut spec used by the TransformerLayerCut partition
// strategy. `spec` is a null-terminated per-signature spec string (see the enum
// comment above for the grammar). The string is copied.
LiteRtStatus LrtSetCompilerOptionsTransformerLayerCuts(LrtCompilerOptions* options,
                                                       const char* spec);

// Gets the transformer layer cut spec. Returns a pointer to the internally held
// string (valid until the options are modified or destroyed) via `*spec`.
// Returns kLiteRtStatusErrorNotFound if no spec is set.
LiteRtStatus LrtGetCompilerOptionsTransformerLayerCuts(
    const LrtCompilerOptions* options, const char** spec);

// Sets the dummy option for testing.
LiteRtStatus LrtSetCompilerOptionsDummyOption(LrtCompilerOptions* options,
                                              bool dummy_option);

// Gets the dummy option from compiler options.
LiteRtStatus LrtGetCompilerOptionsDummyOption(const LrtCompilerOptions* options,
                                              bool* dummy_option);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_COMPILER_OPTIONS_H_
