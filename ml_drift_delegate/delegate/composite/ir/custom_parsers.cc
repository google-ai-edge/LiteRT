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

#include "ml_drift_delegate/delegate/composite/ir/custom_parsers.h"

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift_delegate/delegate/composite/ir/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/ir/moe_experts_parser.h"
#include "ml_drift_delegate/delegate/composite/ir/runtime_batched_matmul_parser.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"

namespace litert::ml_drift::ir {

CustomIrOpMap GetCustomIrParsers() {
  CustomIrOpMap parsers;

  parsers["odml.cache_update"] = GetAddValuesToCacheParser();
  parsers["moe"] = GetMoeExpertsParser();
  parsers["odml.runtime_bmm"] = GetRuntimeBatchedMatMulParser();

  return parsers;
}

}  // namespace litert::ml_drift::ir
