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

#include "litert/cc/litert_options_experimental.h"

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/core/options.h"

namespace litert {
namespace internal {

class RuntimeProxy;

Expected<void> AddCustomOp(Options& options, const TfLiteRegistration* reg) {
  return options.AddBuildAction(
      [reg](internal::RuntimeProxy* /*runtime*/, LiteRtOptions litert_options) {
        auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(litert_options);
        options_impl->custom_tflite_op_registrations.push_back(reg);
        return kLiteRtStatusOk;
      });
}

Expected<void> AddCustomOp(Options& options, const TfLiteOperator* op) {
  return options.AddBuildAction(
      [op](internal::RuntimeProxy* /*runtime*/, LiteRtOptions litert_options) {
        auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(litert_options);
        options_impl->custom_tflite_op_operators.push_back(op);
        return kLiteRtStatusOk;
      });
}

}  // namespace internal
}  // namespace litert
