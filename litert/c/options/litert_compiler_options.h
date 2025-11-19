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

#include <stdint.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"

#ifdef __cplusplus
extern "C" {
#endif

LITERT_DEFINE_HANDLE(LiteRtCompilerOptions);

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

// Creates an opaque options object holding Compiler options.
LiteRtStatus LiteRtCreateCompilerOptions(LiteRtOpaqueOptions* options);

// Gets the underlying Compiler options from an opaque options handle.
LiteRtStatus LiteRtFindCompilerOptions(LiteRtOpaqueOptions opaque_options,
                                       LiteRtCompilerOptions* compiler_options);

// Gets the identifier for Compiler options stored in opaque options.
const char* LiteRtGetCompilerOptionsIdentifier();

// Sets the partition strategy for the compiler.
LiteRtStatus LiteRtSetCompilerOptionsPartitionStrategy(
    LiteRtCompilerOptions options,
    LiteRtCompilerOptionsPartitionStrategy partition_strategy);

LiteRtStatus LiteRtGetCompilerOptionsPartitionStrategy(
    LiteRtCompilerOptionsConst options,
    LiteRtCompilerOptionsPartitionStrategy* partition_strategy);

// Dummy options for testing.
LiteRtStatus LiteRtSetDummyCompilerOptions(LiteRtCompilerOptions options,
                                           bool dummy_option);
LiteRtStatus LiteRtGetDummyCompilerOptions(LiteRtCompilerOptionsConst options,
                                           bool* dummy_option);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_COMPILER_OPTIONS_H_
