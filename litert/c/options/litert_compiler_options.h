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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_COMPILER_OPTIONS_H_
