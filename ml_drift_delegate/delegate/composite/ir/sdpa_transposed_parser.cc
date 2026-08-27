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

#include "ml_drift_delegate/delegate/composite/ir/sdpa_transposed_parser.h"

#include <any>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {
namespace {

constexpr int kActiveTokensAlignedIndex = 2;

absl::Status SdpaTransposedIsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* /*registration*/) {
  int num_runtime_inputs = 0;
  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    if (tflite_node->inputs->data[i] != kTfLiteOptionalTensor &&
        !::tflite::IsConstantTensor(
            &context->tensors[tflite_node->inputs->data[i]])) {
      num_runtime_inputs++;
    }
  }

  if (num_runtime_inputs < 3 || num_runtime_inputs > 5) {
    return absl::UnavailableError(
        "SDPA transposed expects between 3 and 5 inputs.");
  }

  if (tflite_node->outputs->size != 1) {
    return absl::InvalidArgumentError("SDPA transposed expects 1 output.");
  }

  return absl::OkStatus();
}

::ml_drift::ir::IrTensorId NewTensorToMergeBatch(
    ::ml_drift::ir::IrModel* ir_model, ::ml_drift::ir::IrTensorId tensor_id) {
  const ::ml_drift::ir::IrTensor* tensor = ir_model->tensor(tensor_id);
  ::ml_drift::BHWC shape = tensor->desc.GetBHWCShape();
  ::ml_drift::BHWC new_shape(1, shape.b * shape.h, shape.w, shape.c);
  ::ml_drift::ir::IrTensor* new_tensor =
      ir_model->add_tensor(tensor->desc.GetDataType(), new_shape);
  if (tensor->quant_params.has_value()) {
    new_tensor->quant_params = tensor->quant_params.value();
  }
  return new_tensor->id;
}

void Reshape(::ml_drift::ir::IrModel* ir_model,
             ::ml_drift::ir::IrTensorId before,
             ::ml_drift::ir::IrTensorId after) {
  ::ml_drift::ir::IrOp* reshape = ir_model->add_op();
  reshape->name = ToString(::ml_drift::OperationType::RESHAPE);
  ::ml_drift::ReshapeAttributes reshape_attr;
  reshape_attr.new_shape = ir_model->tensor(after)->desc.GetBHWCShape();
  reshape->attr = reshape_attr;
  ir_model->AddConsumer(before, reshape->id);
  ir_model->SetProducer(after, reshape->id);
}

::ml_drift::ir::IrTensorId ReshapeToMergeBatch(
    ::ml_drift::ir::IrModel* ir_model, ::ml_drift::ir::IrTensorId tensor_id) {
  ::ml_drift::ir::IrTensorId new_tensor_id =
      NewTensorToMergeBatch(ir_model, tensor_id);
  Reshape(ir_model, tensor_id, new_tensor_id);
  return new_tensor_id;
}

void SdpaTransposedConvert(
    const TfLiteContext& /*context*/, const TfLiteNode& tflite_node,
    const TfLiteRegistration& /*registration*/,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrTensorId input0 = tensor_map[tflite_node.inputs->data[0]];
  ::ml_drift::ir::IrTensorId input1 = tensor_map[tflite_node.inputs->data[1]];
  ::ml_drift::ir::IrTensorId input2 = tensor_map[tflite_node.inputs->data[2]];
  ::ml_drift::ir::IrTensorId input3 =
      (tflite_node.inputs->size > 3 &&
       tflite_node.inputs->data[3] != kTfLiteOptionalTensor)
          ? tensor_map[tflite_node.inputs->data[3]]
          : -1;
  ::ml_drift::ir::IrTensorId input4 =
      (tflite_node.inputs->size > 4 &&
       tflite_node.inputs->data[4] != kTfLiteOptionalTensor)
          ? tensor_map[tflite_node.inputs->data[4]]
          : -1;

  ::ml_drift::ir::IrTensorId output = tensor_map[tflite_node.outputs->data[0]];

  const auto q_shape = ir_model.tensor(input0)->desc.GetBHWCShape();
  const bool model_batch = q_shape.b != 1;
  ::ml_drift::ir::IrTensorId q_val = input0;
  ::ml_drift::ir::IrTensorId k_val = input1;
  ::ml_drift::ir::IrTensorId v_val = input2;
  ::ml_drift::ir::IrTensorId result_val = output;
  // Reshape to merge batch dimension as the kernel expects 1xBxMxN shape.
  if (model_batch) {
    q_val = ReshapeToMergeBatch(&ir_model, input0);
    k_val = ReshapeToMergeBatch(&ir_model, input1);
    v_val = ReshapeToMergeBatch(&ir_model, input2);
    result_val = NewTensorToMergeBatch(&ir_model, output);
  }

  ::litert::ml_drift::SdpaTransposedAttributes attr;
  attr.runtime_check.src_end_ch_index = kActiveTokensAlignedIndex;
  attr.is_prefill = (q_shape.w > 2 || model_batch);

  const ::ml_drift::BHWC right_shape_k =
      ir_model.tensor(k_val)->desc.GetBHWCShape();
  attr.bmm1_weights.weights_shape =
      ::ml_drift::OHWI(right_shape_k.w, right_shape_k.h, 1, right_shape_k.c);
  // TODO: b/403337563 - Support quantized weights.
  attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ir_model.tensor(k_val)->desc.GetDataType(),
      attr.bmm1_weights.weights_shape);
  attr.bmm1_weights.desc.layout =
      ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;

  const ::ml_drift::BHWC right_shape_v =
      ir_model.tensor(v_val)->desc.GetBHWCShape();
  attr.bmm2_weights.weights_shape =
      ::ml_drift::OHWI(right_shape_v.w, right_shape_v.h, 1, right_shape_v.c);
  // TODO: b/403337563 - Support quantized weights.
  attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ir_model.tensor(v_val)->desc.GetDataType(),
      attr.bmm2_weights.weights_shape);

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node.builtin_data);
  if (params && params->attributes) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(params->attributes, params->attributes_size)
            .AsMap();
    if (!flexbuffer_map["softcap"].IsNull()) {
      attr.softcap = flexbuffer_map["softcap"].AsFloat();
    }
  }

  ::ml_drift::ir::IrOp* sdpa = ir_model.add_op();
  sdpa->attr = std::move(attr);
  sdpa->name = "sdpa_transposed";
  ir_model.AddConsumer(q_val, sdpa->id);
  ir_model.AddConsumer(k_val, sdpa->id);
  ir_model.AddConsumer(v_val, sdpa->id);
  if (input3 != -1) {
    ir_model.AddConsumer(input3, sdpa->id);
  }
  if (input4 != -1) {
    ir_model.AddConsumer(input4, sdpa->id);
  }
  ir_model.SetProducer(result_val, sdpa->id);

  if (model_batch) {
    Reshape(&ir_model, result_val, output);
  }
}

}  // namespace

CustomIrOpParser GetSdpaTransposedParser() {
  return {
      .is_supported = SdpaTransposedIsSupported,
      .convert = SdpaTransposedConvert,
  };
}

}  // namespace litert::ml_drift::ir
