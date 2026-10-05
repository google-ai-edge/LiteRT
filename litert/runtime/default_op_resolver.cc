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

#include "litert/cc/litert_expected.h"
#include "litert/runtime/op_resolver.h"
#include "tflite/kernels/register.h"
#include "tflite/kernels/register_ref.h"
#include "tflite/mutable_op_resolver.h"

namespace litert::internal {

Expected<std::unique_ptr<tflite::MutableOpResolver>> CreateOpResolver(
    bool use_reference_kernels) {
  if (use_reference_kernels) {
    return std::unique_ptr<tflite::MutableOpResolver>(
        std::make_unique<tflite::ops::builtin::BuiltinRefOpResolver>());
  }
  return std::unique_ptr<tflite::MutableOpResolver>(
      std::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>());
}

}  // namespace litert::internal
