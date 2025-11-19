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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_STUB_OP_RESOLVER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_STUB_OP_RESOLVER_H_

#include <cstddef>

#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/mutable_op_resolver.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {

// A stub op resolver for use when building without built-in kernels.
// This resolver registers all builtin operations with minimal stub
// implementations that allow models to pass validation. The actual computation
// will be handled by LiteRT's accelerator system (NPU, GPU, CPU) through
// delegates.
class StubOpResolver : public tflite::MutableOpResolver {
 public:
  StubOpResolver() {
    // Register all built-in operations with stub implementations.
    // This comprehensive list ensures any model can pass validation.
    // The stub implementations will never actually execute - accelerators
    // will handle the operations through their respective delegates.

    // Register all builtin ops with a wide version range for compatibility
    for (int op = 0; op <= tflite::BuiltinOperator_MAX; ++op) {
      // Skip invalid enum values
      if (op == tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
        continue;
      }

      // Register with version range 1-127 to support all possible versions
      AddBuiltin(static_cast<tflite::BuiltinOperator>(op),
                 GetStubRegistration(),
                 /* min_version */ 1,
                 /* max_version */ 127);
    }
  }

  ~StubOpResolver() override = default;

 private:
  static void* StubInit(TfLiteContext* context, const char* buffer,
                        size_t length) {
    return nullptr;
  }

  static void StubFree(TfLiteContext* context, void* buffer) {}

  static TfLiteStatus StubPrepare(TfLiteContext* context, TfLiteNode* node) {
    // This should never be called as accelerators will handle the operations
    context->ReportError(
        context,
        "Stub operation invoked. This model requires accelerator support.");
    return kTfLiteError;
  }

  static TfLiteStatus StubEval(TfLiteContext* context, TfLiteNode* node) {
    // This should never be called as accelerators will handle the operations
    context->ReportError(
        context,
        "Stub operation invoked. This model requires accelerator support.");
    return kTfLiteError;
  }

  static TfLiteRegistration* GetStubRegistration() {
    static TfLiteRegistration registration = {
        .init = StubInit,
        .free = StubFree,
        .prepare = StubPrepare,
        .invoke = StubEval,
    };
    return &registration;
  }
};

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_STUB_OP_RESOLVER_H_
