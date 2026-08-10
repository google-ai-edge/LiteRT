// Copyright 2026 Google LLC.
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

#include "ml_drift_delegate/tflite/convert/convert_embedding_lookup.h"

#include <cstddef>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

namespace {

// Helper to convert int32 zero point to float zero point for
// EmbeddingLookupAttributes.
::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>
ConvertZeroPoint(
    const ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>&
        int_zp) {
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32> float_zp;
  float_zp.shape = int_zp.shape;
  float_zp.data.resize(int_zp.data.size());
  for (size_t i = 0; i < int_zp.data.size(); ++i) {
    float_zp.data[i] = static_cast<float>(int_zp.data[i]);
  }
  return float_zp;
}

}  // namespace

void ConvertEmbeddingLookup(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& options, ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::EmbeddingLookupAttributes attr;
  const int weights_id = node.inputs->data[1];
  const TfLiteTensor* weights_tensor = context.tensors + weights_id;
  const ::ml_drift::ir::IrTensorId ir_weights_id = tensor_map[weights_id];
  bool weights_are_shared = false;

  if (tflite::IsConstantTensor(weights_tensor) &&
      !ir_model.tensor(ir_weights_id)->buffer_source.is_shared) {
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32> tmp_zp;
    if (weights_tensor->type == kTfLiteInt2) {
      ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::UINT8>
          weights_hw;
      PopulateTensor(weights_tensor, weights_id, &weights_hw,
                     PopulateTensorFlags::kExtraBytes,
                     options.enable_spanned_weights, &attr.weights_scale,
                     &tmp_zp);
      attr.weights_type =
          ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt2;
      attr.original_weights_shape =
          ::ml_drift::OHWI(weights_hw.shape.h, 1, 1, weights_hw.shape.w);
      auto& weights_uint8 = attr.weights.emplace<
          ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::UINT8>>();
      weights_uint8.shape = attr.original_weights_shape;
      weights_uint8.data = std::move(weights_hw.data);
      weights_uint8.spanned_data = weights_hw.spanned_data;
    } else if (weights_tensor->type == kTfLiteInt4) {
      ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::UINT8>
          weights_hw;
      PopulateTensor(weights_tensor, weights_id, &weights_hw,
                     PopulateTensorFlags::kExtraBytes,
                     options.enable_spanned_weights, &attr.weights_scale,
                     &tmp_zp);
      attr.weights_type =
          ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt4;
      attr.original_weights_shape =
          ::ml_drift::OHWI(weights_hw.shape.h, 1, 1, weights_hw.shape.w);
      auto& weights_uint8 = attr.weights.emplace<
          ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::UINT8>>();
      weights_uint8.shape = attr.original_weights_shape;
      weights_uint8.data = std::move(weights_hw.data);
      weights_uint8.spanned_data = weights_hw.spanned_data;
    } else if (weights_tensor->type == kTfLiteInt8) {
      ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::INT8> weights_hw;
      PopulateTensor(weights_tensor, weights_id, &weights_hw,
                     PopulateTensorFlags::kExtraBytes,
                     options.enable_spanned_weights, &attr.weights_scale,
                     &tmp_zp);
      attr.weights_type =
          ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt8;
      attr.original_weights_shape =
          ::ml_drift::OHWI(weights_hw.shape.h, 1, 1, weights_hw.shape.w);
      auto& weights_int8 = attr.weights.emplace<
          ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8>>();
      weights_int8.shape = attr.original_weights_shape;
      weights_int8.data = std::move(weights_hw.data);
      weights_int8.spanned_data = weights_hw.spanned_data;
    } else if (weights_tensor->type == kTfLiteFloat32) {
      ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32>
          weights_hw;
      PopulateTensor(weights_tensor, weights_id, &weights_hw,
                     PopulateTensorFlags::kExtraBytes,
                     options.enable_spanned_weights);
      attr.weights_type =
          ::ml_drift::EmbeddingLookupAttributes::WeightsType::kFloat32;
      attr.original_weights_shape =
          ::ml_drift::OHWI(weights_hw.shape.h, 1, 1, weights_hw.shape.w);
      auto& weights_f32 = attr.weights.emplace<::ml_drift::Tensor<
          ::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
      weights_f32.shape = attr.original_weights_shape;
      weights_f32.data = std::move(weights_hw.data);
      weights_f32.spanned_data = weights_hw.spanned_data;
    } else {
      ABSL_LOG(FATAL) << "Unsupported weights type for EmbeddingLookup: "
                      << weights_tensor->type;
    }
    if (tmp_zp.shape.DimensionsProduct() > 0) {
      attr.weights_zero_point = ConvertZeroPoint(tmp_zp);
    }
  } else if (ir_model.tensor(ir_weights_id)->buffer_source.is_shared) {
    // Shared (external runtime) weights: mirror GraphFloat32's shared branch.
    // The weights are passed as a runtime input, so only their type and shapes
    // are recorded here (no data); the shared-memory manager materializes the
    // buffer. Blockwise additionally splits the input dim into blocks.
    const int rows = weights_tensor->dims->data[0];
    const int cols = weights_tensor->dims->data[1];
    if (weights_tensor->type == kTfLiteInt8) {
      attr.weights_type =
          ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt8;
    } else if (weights_tensor->type == kTfLiteInt4) {
      attr.weights_type =
          ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt4;
    } else if (weights_tensor->type == kTfLiteInt2) {
      attr.weights_type =
          ::ml_drift::EmbeddingLookupAttributes::WeightsType::kInt2;
    } else {
      ABSL_LOG(FATAL) << "EMBEDDING_LOOKUP: Unsupported external weights type: "
                      << weights_tensor->type;
    }
    attr.original_weights_shape = ::ml_drift::OHWI(rows, 1, 1, cols);
    attr.scale_zp_shape = ::ml_drift::OHWI(rows, 1, 1, 1);
    if (weights_tensor->quantization.type == kTfLiteBlockwiseQuantization) {
      const auto* qparams =
          reinterpret_cast<const TfLiteBlockwiseQuantization*>(
              weights_tensor->quantization.params);
      attr.scale_zp_shape.i = cols / qparams->blocksize;
    }
    weights_are_shared = true;
  } else {
    ABSL_LOG(WARNING)
        << "EmbeddingLookup weights are not constant, conversion might be "
           "incomplete.";
  }

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::EMBEDDING_LOOKUP);
  op->attr = std::move(attr);
  ir_model.AddConsumer(tensor_map[node.inputs->data[0]], op->id);
  // Shared weights are consumed as a runtime input (added after the lookup ids
  // so the op's input ordering matches GraphFloat32).
  if (weights_are_shared) {
    ir_model.AddConsumer(ir_weights_id, op->id);
  }

  // Set output producer
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], op->id);
}

}  // namespace litert::ml_drift::ir
