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

#include <memory>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/op_resolver.h"
#include "tflite/mutable_op_resolver.h"

namespace litert::internal {

// Generated from the models by litert_selected_op_resolver().
void RegisterSelectedOps(tflite::MutableOpResolver* resolver);

Expected<std::unique_ptr<tflite::MutableOpResolver>> CreateOpResolver(
    bool use_reference_kernels) {
  if (use_reference_kernels) {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "The selected op resolver does not contain reference "
                      "kernels. Use the builtin CPU kernel mode.");
  }
  auto resolver = std::make_unique<tflite::MutableOpResolver>();
  RegisterSelectedOps(resolver.get());
  return resolver;
}

}  // namespace litert::internal
