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

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct TfLiteRegistration;

// Callback function to resolve a builtin operator.
// Note: Because TfLiteRegistration layout depends on the TFLite version, the
// client and runtime must be built against matching TFLite versions
// (see b/580021727).
typedef const struct TfLiteRegistration* (*LiteRtFindBuiltinOpFunc)(
    void* user_data, int op, int version);

// Callback function to resolve a custom operator.
// Note: Because TfLiteRegistration layout depends on the TFLite version, the
// client and runtime must be built against matching TFLite versions
// (see b/580021727).
typedef const struct TfLiteRegistration* (*LiteRtFindCustomOpFunc)(
    void* user_data, const char* custom_op, int version);

/// C ABI interface representing a TFLite OpResolver.
///
/// @note This concrete type is part of the public API and is ABI stable.
typedef struct LiteRtOpResolverCApi {
  LiteRtFindBuiltinOpFunc find_builtin_op;
  LiteRtFindCustomOpFunc find_custom_op;
  void* user_data;
} LiteRtOpResolverCApi;

#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && \
    __SIZEOF_POINTER__ == 8
static_assert(sizeof(LiteRtOpResolverCApi) == 24,
              "LiteRtOpResolverCApi size mismatch");
static_assert(offsetof(LiteRtOpResolverCApi, find_builtin_op) == 0,
              "LiteRtOpResolverCApi find_builtin_op offset mismatch");
static_assert(offsetof(LiteRtOpResolverCApi, find_custom_op) == 8,
              "LiteRtOpResolverCApi find_custom_op offset mismatch");
static_assert(offsetof(LiteRtOpResolverCApi, user_data) == 16,
              "LiteRtOpResolverCApi user_data offset mismatch");
#endif  // __cplusplus

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_OP_RESOLVER_H_
