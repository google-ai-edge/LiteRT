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

#ifndef ODML_LITERT_LITERT_C_LITERT_OP_RESOLVER_H_
#define ODML_LITERT_LITERT_C_LITERT_OP_RESOLVER_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct TfLiteOperator;

// Callback function to resolve a builtin operator.
// The returned `TfLiteOperator` (if non-null) is owned by the caller and must
// outlive the `CompiledModel`(s) created with the associated options.
typedef const struct TfLiteOperator* (*LiteRtFindBuiltinOpFunc)(void* user_data,
                                                                int op,
                                                                int version);

// Callback function to resolve a custom operator.
// The returned `TfLiteOperator` (if non-null) is owned by the caller and must
// outlive the `CompiledModel`(s) created with the associated options.
typedef const struct TfLiteOperator* (*LiteRtFindCustomOpFunc)(
    void* user_data, const char* custom_op, int version);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_OP_RESOLVER_H_
