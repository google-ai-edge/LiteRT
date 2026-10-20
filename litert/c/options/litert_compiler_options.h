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
typedef enum LiteRtCompilerOptionsPartitionStrategy {
  kLiteRtCompilerOptionsPartitionStrategyDefault = 0,
  kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected = 1,
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

// Sets the dummy option for testing.
LiteRtStatus LrtSetCompilerOptionsDummyOption(LrtCompilerOptions* options,
                                              bool dummy_option);

// Gets the dummy option from compiler options.
LiteRtStatus LrtGetCompilerOptionsDummyOption(const LrtCompilerOptions* options,
                                              bool* dummy_option);

// Sets the maximum number of partitions allowed.
// A value of 0 means unlimited.
LiteRtStatus LrtSetCompilerOptionsMaxPartitions(LrtCompilerOptions* options,
                                                size_t max_partitions);

// Gets the maximum number of partitions allowed.
LiteRtStatus LrtGetCompilerOptionsMaxPartitions(
    const LrtCompilerOptions* options, size_t* max_partitions);

// Appends a positional input shape override for `signature_key`'s subgraph.
// Pass NULL or "" for `signature_key` to target the default (first) signature
// or main subgraph. Call once per subgraph input for `signature_key`, in
// subgraph input order (the number of calls for a signature must equal its
// input count). `rank` may be 0 for a 0-D scalar input (in which case `dims`
// may be NULL). Each element of `dims` must be positive (> 0) or `-1` (to keep
// that dimension dynamic during shape refinement). For refinement to produce a
// fully static subgraph, all specified dimensions must be positive (> 0) and
// any non-overridden subgraph inputs must already have static shapes.
// Note: Input shape overrides only take effect when a compiler plugin is
// applied (NPU JIT or `apply_plugin`); with a CPU/GPU-only accelerator set
// they are ignored.
LiteRtStatus LrtAddCompilerOptionsPositionalInputShape(
    LrtCompilerOptions* options, const char* signature_key, const int32_t* dims,
    size_t rank);

// Appends an input shape override by tensor name within `signature_key`'s
// subgraph. Pass NULL or "" for `signature_key` to target the default (first)
// signature or main subgraph. If called multiple times for the same
// `(signature_key, tensor_name)`, the last call wins. `rank` may be 0 for a 0-D
// scalar input (in which case `dims` may be NULL). Each element of `dims` must
// be positive (> 0) or `-1` (to keep that dimension dynamic during shape
// refinement). Any non-overridden subgraph inputs must already have static
// shapes for refinement to produce a fully static subgraph.
// Note: Input shape overrides only take effect when a compiler plugin is
// applied (NPU JIT or `apply_plugin`); with a CPU/GPU-only accelerator set
// they are ignored.
LiteRtStatus LrtAddCompilerOptionsTensorInputShape(LrtCompilerOptions* options,
                                                   const char* signature_key,
                                                   const char* tensor_name,
                                                   const int32_t* dims,
                                                   size_t rank);

// Appends an input shape override by signature key and signature input name.
// Pass NULL or "" for `signature_key` to target the default (first) signature.
// If called multiple times for the same `(signature_key, input_name)`, the last
// call wins. `rank` may be 0 for a 0-D scalar input (in which case `dims` may
// be NULL). Each element of `dims` must be positive (> 0) or `-1` (to keep
// that dimension dynamic during shape refinement). Any non-overridden subgraph
// inputs must already have static shapes for refinement to produce a fully
// static subgraph.
// Note: Input shape overrides only take effect when a compiler plugin is
// applied (NPU JIT or `apply_plugin`); with a CPU/GPU-only accelerator set
// they are ignored.
LiteRtStatus LrtAddCompilerOptionsSignatureInputShape(
    LrtCompilerOptions* options, const char* signature_key,
    const char* input_name, const int32_t* dims, size_t rank);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_COMPILER_OPTIONS_H_
