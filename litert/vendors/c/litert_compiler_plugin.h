// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_H_
#define ODML_LITERT_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_H_

#include <stddef.h>

#include "litert/c/litert_builder.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtCompilerPlugin);

// Artifact produced from compiling a selected partition of ops.
LITERT_DEFINE_HANDLE(LiteRtCompiledResult);

// Struct to hold information about a transformation. Append only.
typedef struct {
  LiteRtPatternFn pattern;  // The function pointer of the pattern.
  const char* name;         // The name of the transformation.
  size_t benefit;  // All added transformations will be sorted by benefit
                   // in descending order.
} LiteRtTransformation;

//
// Plugin
//

// APIs are now resolved via LiteRtCompilerPluginQueryInterface defined in
// litert_compiler_plugin_api.h.

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_H_
