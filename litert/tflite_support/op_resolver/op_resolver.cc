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

#include "litert/tflite_support/op_resolver/op_resolver.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_resolver.h"
#include "litert/cc/internal/litert_runtime_proxy.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "tflite/c/common.h"
#include "tflite/mutable_op_resolver.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace tflite_support {

namespace {

const TfLiteRegistration* FindBuiltinOpCallback(void* user_data, int op,
                                                int version) {
  auto* resolver = static_cast<const tflite::MutableOpResolver*>(user_data);
  return resolver->FindOp(static_cast<tflite::BuiltinOperator>(op), version);
}

const TfLiteRegistration* FindCustomOpCallback(void* user_data,
                                               const char* custom_op,
                                               int version) {
  auto* resolver = static_cast<const tflite::MutableOpResolver*>(user_data);
  return resolver->FindOp(custom_op, version);
}

}  // namespace

Expected<void> SetOpResolver(Options& options,
                             const tflite::MutableOpResolver* resolver) {
  if (resolver == nullptr) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "OpResolver must not be null.");
  }
  return options.AddBuildAction([resolver](internal::RuntimeProxy* runtime,
                                           LiteRtOptions litert_options) {
    LiteRtOpResolverCApi c_api = {
        /*find_builtin_op=*/FindBuiltinOpCallback,
        /*find_custom_op=*/FindCustomOpCallback,
        /*user_data=*/
        const_cast<void*>(static_cast<const void*>(resolver)),
    };
    return runtime->SetOpResolverOption(litert_options, &c_api);
  });
}

}  // namespace tflite_support
}  // namespace litert
