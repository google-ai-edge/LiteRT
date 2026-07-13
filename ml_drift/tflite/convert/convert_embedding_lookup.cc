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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_embedding_lookup.h"

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
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
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

  if (tflite::IsConstantTensor(weights_tensor)) {
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
  } else {
    ABSL_LOG(WARNING)
        << "EmbeddingLookup weights are not constant, conversion might be "
           "incomplete.";
  }

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::EMBEDDING_LOOKUP);
  op->attr = std::move(attr);
  ir_model.AddConsumer(tensor_map[node.inputs->data[0]], op->id);

  // Set output producer
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], op->id);
}

}  // namespace litert::ml_drift::ir
