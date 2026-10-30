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
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/core/options.h"

namespace litert {
namespace internal {
class RuntimeProxy;
}  // namespace internal

namespace tflite_support {

Expected<void> SetOpResolver(Options& options,
                             const tflite::MutableOpResolver* resolver) {
  if (resolver == nullptr) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "OpResolver must not be null.");
  }
  return options.AddBuildAction([resolver](internal::RuntimeProxy* /*runtime*/,
                                           LiteRtOptions litert_options) {
    auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(litert_options);
    options_impl->op_resolver = resolver;
    return kLiteRtStatusOk;
  });
}

}  // namespace tflite_support
}  // namespace litert
