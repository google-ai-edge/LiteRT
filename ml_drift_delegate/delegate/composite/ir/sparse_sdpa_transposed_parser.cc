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

#include "ml_drift_delegate/delegate/composite/ir/sparse_sdpa_transposed_parser.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/sparse_sdpa_transposed_parser.h"
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

absl::Status SparseSdpaTransposedIsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  return ::litert::ml_drift::SparseSdpaTransposedOperationParser().IsSupported(
      context, tflite_node, registration);
}

void SparseSdpaTransposedConvert(
    const TfLiteContext& context, const TfLiteNode& tflite_node,
    const TfLiteRegistration& /*registration*/,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = kSparseSdpaTransposedType;

  AddNodeInputs(context, tflite_node, tensor_map, /*layout=*/{}, ir_model, op);
  for (int i = 0; i < tflite_node.outputs->size; ++i) {
    ir_model.SetProducer(tensor_map[tflite_node.outputs->data[i]], op->id);
  }

  ::litert::ml_drift::SparseSdpaTransposedAttributes attr =
      ::litert::ml_drift::ParseSparseSdpaTransposedAttributes(tflite_node)
          .value_or(::litert::ml_drift::SparseSdpaTransposedAttributes{});
  if (op->inputs.size() > 1) {
    ::ml_drift::ir::IrOp* key_producer = ir_model.FindProducer(op->inputs[1]);
    if (key_producer != nullptr &&
        key_producer->name == kAddValuesToCacheType) {
      attr.from_cache_update = true;
    }
  }
  op->attr = std::move(attr);
}

}  // namespace

CustomIrOpParser GetSparseSdpaTransposedParser() {
  return {
      .is_supported = SparseSdpaTransposedIsSupported,
      .convert = SparseSdpaTransposedConvert,
  };
}

}  // namespace litert::ml_drift::ir
