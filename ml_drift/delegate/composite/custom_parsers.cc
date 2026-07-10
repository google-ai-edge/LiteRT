// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "third_party/odml/litert/ml_drift/delegate/composite/custom_parsers.h"

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/delegate/composite/add_values_to_cache_parser.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/moe_experts_parser.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/runtime_batched_matmul_parser.h"
#include "third_party/odml/litert/ml_drift/tflite/operation_parser.h"
#include "third_party/odml/litert/ml_drift/tflite/unimplemented_operation_parser.h"

namespace litert::ml_drift {

std::unique_ptr<TFLiteOperationParser> CustomOperationParserFactory::Create(
    std::string_view op_name) {
  if (op_name == "odml.cache_update") {
    return std::make_unique<AddValuesToCacheOperationParser>();
  }
  if (op_name == "odml.runtime_bmm") {
    return std::make_unique<RuntimeBatchedMatMulOperationParser>();
  }
  if (op_name == "moe") {
    return std::make_unique<MoeExpertsOperationParser>();
  }
  return std::make_unique<UnimplementedOperationParser>(op_name);
}

bool CustomOperationParserFactory::SupportsIntegerTypes(
    std::string_view op_name) {
  return op_name == "odml.cache_update" || op_name == "odml.runtime_bmm" ||
         op_name == "moe";
}

bool CustomOperationParserFactory::SupportsBoolTypes(std::string_view op_name) {
  return false;
}

}  // namespace litert::ml_drift
