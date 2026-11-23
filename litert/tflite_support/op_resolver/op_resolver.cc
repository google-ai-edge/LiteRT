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

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_runtime_proxy.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/core/options.h"
#include "tflite/c/common.h"
#include "tflite/core/api/op_resolver.h"
#include "tflite/mutable_op_resolver.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace tflite_support {

namespace {

const TfLiteOperator* FindBuiltinOpCallback(void* user_data, int op,
                                            int version) {
  const auto* resolver = static_cast<const tflite::OpResolver*>(user_data);
  const TfLiteRegistration* reg =
      resolver->FindOp(static_cast<tflite::BuiltinOperator>(op), version);
  if (reg == nullptr) {
    return nullptr;
  }
  if (reg->registration_external == nullptr) {
    LITERT_LOG(LITERT_ERROR,
               "OpResolver returned TfLiteRegistration without "
               "registration_external for builtin op %d (version %d); "
               "ignoring non-ABI-stable registration.",
               op, version);
    return nullptr;
  }
  return reg->registration_external;
}

const TfLiteOperator* FindCustomOpCallback(void* user_data,
                                           const char* custom_op, int version) {
  const auto* resolver = static_cast<const tflite::OpResolver*>(user_data);
  const TfLiteRegistration* reg = resolver->FindOp(custom_op, version);
  if (reg == nullptr) {
    return nullptr;
  }
  if (reg->registration_external == nullptr) {
    LITERT_LOG(LITERT_ERROR,
               "OpResolver returned TfLiteRegistration without "
               "registration_external for custom op '%s' (version %d); "
               "ignoring non-ABI-stable registration.",
               custom_op != nullptr ? custom_op : "<null>", version);
    return nullptr;
  }
  return reg->registration_external;
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
    if (!runtime->IsBuiltinRuntime()) {
      LITERT_LOG(
          LITERT_ERROR,
          "SetOpResolver with MutableOpResolver is only supported in static "
          "runtime builds. Use SetOpaqueOpResolver across dynamic runtime "
          "boundaries.");
      return kLiteRtStatusErrorUnsupported;
    }
    litert_options->op_resolver = resolver;
    return kLiteRtStatusOk;
  });
}

Expected<void> SetOpaqueOpResolver(Options& options,
                                   const tflite::OpResolver& op_resolver) {
  void* user_data = const_cast<void*>(static_cast<const void*>(&op_resolver));
  return options.AddBuildAction([user_data](internal::RuntimeProxy* runtime,
                                            LiteRtOptions litert_options) {
    return runtime->SetOpResolverOption(litert_options, FindBuiltinOpCallback,
                                        FindCustomOpCallback, user_data);
  });
}

}  // namespace tflite_support
}  // namespace litert
