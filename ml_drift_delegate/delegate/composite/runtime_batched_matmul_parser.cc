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

#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"

#include <any>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr int kActiveTokensAlignedIndex = 2;

::ml_drift::Value* NewValueToMergeBatch(::ml_drift::GraphFloat32* graph,
                                        ::ml_drift::Value* val) {
  auto* new_val = graph->NewValue();
  new_val->tensor.type = val->tensor.type;
  new_val->tensor.shape =
      ::ml_drift::BHWC(1, val->tensor.shape.b * val->tensor.shape.h,
                       val->tensor.shape.w, val->tensor.shape.c);
  return new_val;
}

void Reshape(::ml_drift::GraphFloat32* graph, ::ml_drift::Value* before,
             ::ml_drift::Value* after) {
  ::ml_drift::Node* reshape = graph->NewNode();
  reshape->operation.type = ToString(::ml_drift::OperationType::RESHAPE);
  ::ml_drift::ReshapeAttributes reshape_attr;
  reshape_attr.new_shape = after->tensor.shape;
  reshape->operation.attributes = reshape_attr;
  graph->AddConsumer(reshape->id, before->id);
  graph->SetProducer(reshape->id, after->id);
}

::ml_drift::Value* ReshapeToMergeBatch(::ml_drift::GraphFloat32* graph,
                                       ::ml_drift::Value* val) {
  auto* new_val = NewValueToMergeBatch(graph, val);
  Reshape(graph, val, new_val);
  return new_val;
}

}  // namespace

absl::Status RuntimeBatchedMatMulOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (GetNumberOfRuntimeInputsForNode(context, tflite_node) != 3) {
    return absl::UnavailableError("Runtime BatchedMatMul expects 3 inputs.");
  }

  RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 0));
  RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 1));
  RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, 2));
  RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

  const TfLiteTensor* input0 = nullptr;
  RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &input0));
  const ::ml_drift::BHWC input0_shape = ExtractTensorShape(input0);

  const TfLiteTensor* input1 = nullptr;
  RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 1, &input1));
  const ::ml_drift::BHWC input1_shape = ExtractTensorShape(input1);

  // We expect the BMM to be transpose_y = true, so the channels must match.
  if (input0_shape.c != input1_shape.c) {
    return absl::UnavailableError("Input tensors' channels must match.");
  }

  const TfLiteTensor* param_tensor = nullptr;
  RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 2, &param_tensor));
  const ::ml_drift::BHWC param_tensor_shape = ExtractTensorShape(param_tensor);
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

void RuntimeBatchedMatMulOperationParser::Parse(const TfLiteNode* tflite_node,
                                                const TfLiteRegistration*,
                                                ::ml_drift::GraphFloat32* graph,
                                                ObjectReader* reader) {
  ::ml_drift::Value* input0 = reader->ReadValue(0);
  ::ml_drift::Value* input1 = reader->ReadValue(1);
  ::ml_drift::Value* input2 = reader->ReadValue(2);

  ::ml_drift::Value* output =
      reader->ReadValueByTensorIdx(tflite_node->outputs->data[0]);
  // MLDrift supports batched matmul with single batch.
  // Model can have model batch in addition to matmul batch. In this case we
  // reshape inputs/outputs to have single batch in MLDrift. For example
  // 2x4x128x32 (2 is model batch, 4 is matmul batch) will be reshaped to
  // 1x8x128x32.
  const bool model_batch = input0->tensor.shape.b != 1;
  ::ml_drift::Value* left_value = input0;
  ::ml_drift::Value* right_value = input1;
  ::ml_drift::Value* result_value = output;
  if (model_batch) {
    // Reshape left, B0xB1xMxK -> 1xBxMxK, B = B0 * B1
    left_value = ReshapeToMergeBatch(graph, input0);
    // Reshape right, B0xB1xKxN -> 1xBxKxN, B = B0 * B1
    right_value = ReshapeToMergeBatch(graph, input1);
    result_value = NewValueToMergeBatch(graph, output);
  }

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node->builtin_data);
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
  ::ml_drift::Node* rhs_producer = graph->FindProducer(input1->id);
  if (rhs_cache_update || is_quantized ||
      (rhs_producer && rhs_producer->operation.type == kAddValuesToCacheType)) {
    ::ml_drift::BHWC right_shape = right_value->tensor.shape;
    RuntimeBatchedMatMulAttributes attr;
    auto& external_weights = attr.external_weights.emplace();
    external_weights.weights_shape =
        ::ml_drift::OHWI(right_shape.w, right_shape.h, 1, right_shape.c);
    // TODO: b/404330171 - As FC doesn't support integer weights, we use FP16
    // here. The precision information must be passed from the delegate.
    external_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
        ::ml_drift::DataType::FLOAT16, external_weights.weights_shape);
    auto& runtime_check = attr.runtime_check;
    if (is_src) {
      runtime_check.src_end_ch_index = kActiveTokensAlignedIndex;
    } else {
      external_weights.desc.layout =
          ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
      runtime_check.dst_end_ch_index = kActiveTokensAlignedIndex;
    }

    // int8 case
    auto rhs_tensor = reader->GetInputTensor(1);
    if (rhs_tensor->type == kTfLiteInt8) {
      external_weights.desc.type = ::ml_drift::DataType::UINT8;
      float scale = 1.0f;
      int channel_count = 1;

      if (rhs_producer &&
          rhs_producer->operation.type == kAddValuesToCacheType) {
        const auto& add_values_to_cache_attr =
            std::any_cast<const AddValuesToCacheAttributes&>(
                rhs_producer->operation.attributes);
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
        right_value->tensor.ref = tflite_node->inputs->data[1];
        if (right_value->quant_params.has_value()) {
          right_value->quant_params.reset();
        }
        right_value->tensor.type = ::ml_drift::DataType::UINT8;

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

    ::ml_drift::Node* fc = graph->NewNode();
    fc->operation.attributes = std::move(attr);
    fc->operation.type = kRuntimeBatchedMatMulType;
    graph->AddConsumer(fc->id, left_value->id);
    graph->AddConsumer(fc->id, right_value->id);
    graph->AddConsumer(fc->id, input2->id);  // runtime_param
    graph->SetProducer(fc->id, result_value->id);
  } else {  // Use BatchedMatMul kernel.
    RuntimeBatchedMatMulAttributes attr;
    attr.transpose_left = false;
    attr.transpose_right = true;
    if (is_src) {
      attr.runtime_check.src_end_ch_index = kActiveTokensAlignedIndex;
    } else {
      attr.runtime_check.dst_end_ch_index = kActiveTokensAlignedIndex;
    }
    ::ml_drift::Node* bmm = graph->NewNode();
    bmm->operation.attributes = std::move(attr);
    bmm->operation.type = kRuntimeBatchedMatMulType;
    graph->AddConsumer(bmm->id, left_value->id);
    graph->AddConsumer(bmm->id, right_value->id);
    graph->AddConsumer(bmm->id, input2->id);  // runtime_param
    graph->SetProducer(bmm->id, result_value->id);
  }

  if (model_batch) {
    // Reshape result, 1xBxMxN -> B0xB1xMxN, B = B0 * B1
    Reshape(graph, result_value, output);
  }
}

}  // namespace litert::ml_drift
