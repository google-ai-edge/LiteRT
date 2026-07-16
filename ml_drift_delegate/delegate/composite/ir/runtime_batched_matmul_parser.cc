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

#include "ml_drift_delegate/delegate/composite/ir/runtime_batched_matmul_parser.h"

#include <any>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

namespace {

constexpr int kActiveTokensAlignedIndex = 2;

absl::Status RuntimeBatchedMatMulIsSupported(
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

  if (num_runtime_inputs != 3) {
    return absl::UnavailableError("Runtime BatchedMatMul expects 3 inputs.");
  }

  if (tflite_node->outputs->size != 1) {
    return absl::InvalidArgumentError(
        "Runtime BatchedMatMul expects 1 output.");
  }

  const TfLiteTensor* input0 = &context->tensors[tflite_node->inputs->data[0]];
  const ::ml_drift::BHWC input0_shape =
      ::litert::ml_drift::ExtractTensorShape(input0);

  const TfLiteTensor* input1 = &context->tensors[tflite_node->inputs->data[1]];
  const ::ml_drift::BHWC input1_shape =
      ::litert::ml_drift::ExtractTensorShape(input1);

  if (input0_shape.c != input1_shape.c) {
    return absl::UnavailableError("Input tensors' channels must match.");
  }

  const TfLiteTensor* param_tensor =
      &context->tensors[tflite_node->inputs->data[2]];
  const ::ml_drift::BHWC param_tensor_shape =
      ::litert::ml_drift::ExtractTensorShape(param_tensor);
  if (param_tensor_shape != ::ml_drift::BHWC(1, 1, 1, 7)) {
    return absl::UnavailableError(
        "Param tensor shape expected to be [1, 1, 1, 7].");
  }

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
  if (!params) {
    return absl::InvalidArgumentError(
        "Runtime BatchedMatMul is missing params.");
  }
  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(params->attributes, params->attributes_size).AsMap();
  if (flexbuffer_map["is_global"].IsNull()) {
    return absl::InvalidArgumentError(
        "Runtime BatchedMatMul is missing is_global.");
  }
  if (flexbuffer_map["is_src"].IsNull()) {
    return absl::InvalidArgumentError(
        "Runtime BatchedMatMul is missing is_src.");
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

void RuntimeBatchedMatMulConvert(
    const TfLiteContext& context, const TfLiteNode& tflite_node,
    const TfLiteRegistration& /*registration*/,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrTensorId input0 = tensor_map[tflite_node.inputs->data[0]];
  ::ml_drift::ir::IrTensorId input1 = tensor_map[tflite_node.inputs->data[1]];
  ::ml_drift::ir::IrTensorId input2 = tensor_map[tflite_node.inputs->data[2]];

  ::ml_drift::ir::IrTensorId output = tensor_map[tflite_node.outputs->data[0]];
  // MLDrift supports batched matmul with single batch.
  // Model can have model batch in addition to matmul batch. In this case we
  // reshape inputs/outputs to have single batch in MLDrift. For example
  // 2x4x128x32 (2 is model batch, 4 is matmul batch) will be reshaped to
  // 1x8x128x32.

  const bool model_batch = ir_model.tensor(input0)->desc.GetBHWCShape().b != 1;
  ::ml_drift::ir::IrTensorId left_value = input0;
  ::ml_drift::ir::IrTensorId right_value = input1;
  ::ml_drift::ir::IrTensorId result_value = output;
  if (model_batch) {
    // Reshape left, B0xB1xMxK -> 1xBxMxK, B = B0 * B1
    left_value = ReshapeToMergeBatch(&ir_model, input0);
    // Reshape right, B0xB1xKxN -> 1xBxKxN, B = B0 * B1
    right_value = ReshapeToMergeBatch(&ir_model, input1);
    result_value = NewTensorToMergeBatch(&ir_model, output);
  }

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node.builtin_data);
  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(params->attributes, params->attributes_size).AsMap();
  bool is_src = flexbuffer_map["is_src"].AsBool();
  bool is_quantized = !flexbuffer_map["scale"].IsNull();
  bool rhs_cache_update = flexbuffer_map["rhs_cache_update"].AsBool();

  // Use FullyConnected for external weights if RHS is produced by
  // AddValuesToCache op. Also supports quantized case, which still expects that
  // the KV cache was produced / modified by AddValuesToCache op, but doesn't
  // need the op to be in the graph. For cases without int8 and not from
  // AddValuesToCache, use BatchedMatMul kernel.
  ::ml_drift::ir::IrOp* rhs_producer = ir_model.FindProducer(input1);
  if (rhs_cache_update || is_quantized ||
      (rhs_producer && rhs_producer->name == "add_values_to_cache")) {
    ::ml_drift::BHWC right_shape =
        ir_model.tensor(right_value)->desc.GetBHWCShape();
    ::litert::ml_drift::RuntimeBatchedMatMulAttributes attr;
    ::litert::ml_drift::ExternalWeightsAttributes& external_weights =
        attr.external_weights.emplace();
    external_weights.weights_shape =
        ::ml_drift::OHWI(right_shape.w, right_shape.h, 1, right_shape.c);
    // TODO: b/404330171 - As FC doesn't support integer weights, we use FP16
    // here. The precision information must be passed from the delegate.
    external_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
        ::ml_drift::DataType::FLOAT16, external_weights.weights_shape);
    RuntimeCheckParams& runtime_check = attr.runtime_check;
    if (is_src) {
      runtime_check.src_end_ch_index = kActiveTokensAlignedIndex;
    } else {
      external_weights.desc.layout =
          ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
      runtime_check.dst_end_ch_index = kActiveTokensAlignedIndex;
    }

    // int8 case
    const TfLiteTensor* rhs_tensor =
        &context.tensors[tflite_node.inputs->data[1]];
    if (rhs_tensor->type == kTfLiteInt8) {
      external_weights.desc.type = ::ml_drift::DataType::UINT8;
      float scale = 1.0f;
      int channel_count = 1;

      if (rhs_producer && rhs_producer->name == "add_values_to_cache") {
        const auto& add_values_to_cache_attr = std::any_cast<
            const ::litert::ml_drift::AddValuesToCacheAttributes&>(
            rhs_producer->attr);
        if (is_src) {
          scale = add_values_to_cache_attr.scale_v.value();
          channel_count = add_values_to_cache_attr.head_size;
        } else {
          scale = add_values_to_cache_attr.scale_k.value();
          channel_count = add_values_to_cache_attr.cache_size;
        }
      } else {
        // TODO(b/482104479): allow native int8 tensors in MLDrift. See
        // AddValuesToCacheOperationParser for more details.
        const ::ml_drift::ir::IrTensor* right_tensor_obj =
            ir_model.tensor(right_value);
        if (right_tensor_obj->quant_params.has_value()) {
          ir_model.ResetQuantParams(right_value);
        }
        ::ml_drift::TensorDescriptor new_desc = right_tensor_obj->desc;
        new_desc.SetDataType(::ml_drift::DataType::UINT8);
        ::ml_drift::ir::IrTensor* new_tensor = ir_model.add_tensor(new_desc);
        right_value = new_tensor->id;

        scale = flexbuffer_map["scale"].AsFloat();
        channel_count = rhs_tensor->dims->data[2];
      }
      ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::LINEAR>,
                         ::ml_drift::DataType::FLOAT32>
          scale_tensor;
      scale_tensor.shape = ::ml_drift::Linear(channel_count);
      scale_tensor.data = std::vector<float>(channel_count, scale);
      attr.scale = std::move(scale_tensor);
    }

    ::ml_drift::ir::IrOp* fc = ir_model.add_op();
    fc->attr = std::move(attr);
    fc->name = "runtime_batched_matmul";
    ir_model.AddConsumer(left_value, fc->id);
    ir_model.AddConsumer(right_value, fc->id);
    ir_model.AddConsumer(input2, fc->id);  // runtime_param
    ir_model.SetProducer(result_value, fc->id);
  } else {
    ::litert::ml_drift::RuntimeBatchedMatMulAttributes attr;
    attr.transpose_left = false;
    attr.transpose_right = true;
    if (is_src) {
      attr.runtime_check.src_end_ch_index = kActiveTokensAlignedIndex;
    } else {
      attr.runtime_check.dst_end_ch_index = kActiveTokensAlignedIndex;
    }
    ::ml_drift::ir::IrOp* bmm = ir_model.add_op();
    bmm->attr = std::move(attr);
    bmm->name = "runtime_batched_matmul";
    ir_model.AddConsumer(left_value, bmm->id);
    ir_model.AddConsumer(right_value, bmm->id);
    ir_model.AddConsumer(input2, bmm->id);  // runtime_param
    ir_model.SetProducer(result_value, bmm->id);
  }

  if (model_batch) {
    // Reshape result, 1xBxMxN -> B0xB1xMxN, B = B0 * B1
    Reshape(&ir_model, result_value, output);
  }
}

}  // namespace

CustomIrOpParser GetRuntimeBatchedMatMulParser() {
  return {
      .is_supported = RuntimeBatchedMatMulIsSupported,
      .convert = RuntimeBatchedMatMulConvert,
  };
}

}  // namespace litert::ml_drift::ir
