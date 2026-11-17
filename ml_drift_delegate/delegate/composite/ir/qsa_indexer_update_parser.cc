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

#include "ml_drift_delegate/delegate/composite/ir/qsa_indexer_update_parser.h"

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/qsa_indexer_update_parser.h"
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {
namespace {

absl::Status QsaIndexerUpdateIsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  return ::litert::ml_drift::QsaIndexerUpdateOperationParser().IsSupported(
      context, tflite_node, registration);
}

void QsaIndexerUpdateConvert(
    const TfLiteContext& context, const TfLiteNode& tflite_node,
    const TfLiteRegistration& /*registration*/,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = kQsaIndexerUpdateType;

  for (int i = 0; i < tflite_node.inputs->size; ++i) {
    // `ring_k` and `block_k_pool` are omitted on GPU: the kernel reads and
    // updates outputs 1 (`new_ring_k`) and 2 (`new_block_k_pool`) in-place on
    // GPU, matching `odml.short_conv_step`.
    if (i == ::litert::ml_drift::kQsaIndexerRingKInput ||
        i == ::litert::ml_drift::kQsaIndexerBlockKPoolInput) {
      continue;
    }
    const int input_id = tflite_node.inputs->data[i];
    if (input_id == kTfLiteOptionalTensor) continue;
    const ::ml_drift::ir::IrTensorId ir_tensor_id = tensor_map[input_id];
    if (tflite::IsConstantTensor(context.tensors + input_id) &&
        !ir_model.tensor(ir_tensor_id)->buffer_source.is_shared) {
      ::ml_drift::ir::IrTensor* const_tensor =
          AddConstInput(context, input_id, ir_model, /*layout=*/{});
      ir_model.AddConsumer(const_tensor->id, op->id);
    } else {
      ir_model.AddConsumer(ir_tensor_id, op->id);
    }
  }
  for (int i = 0; i < tflite_node.outputs->size; ++i) {
    ir_model.SetProducer(tensor_map[tflite_node.outputs->data[i]], op->id);
  }
  op->attr = ::litert::ml_drift::ParseQsaIndexerUpdateAttributes(tflite_node)
                 .value_or(::litert::ml_drift::QsaIndexerUpdateAttributes{});
}

}  // namespace

CustomIrOpParser GetQsaIndexerUpdateParser() {
  return {
      .is_supported = QsaIndexerUpdateIsSupported,
      .convert = QsaIndexerUpdateConvert,
  };
}

}  // namespace litert::ml_drift::ir
