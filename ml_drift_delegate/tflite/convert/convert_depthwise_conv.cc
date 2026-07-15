// Copyright 2025 Google LLC.
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

#include "ml_drift_delegate/tflite/convert/convert_depthwise_conv.h"

#include <algorithm>
#include <utility>
#include <variant>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {
namespace {

// TFLite CPU stores weights as:
//   [1, kernel_height, kernel_width, input_depth * depth_multiplier]
// TFLite GPU stores weights as:
//   [depth_multiplier, kernel_height, kernel_width, input_depth]
void TransposeWeights(const TfLiteTensor* input, const TfLiteTensor* filter,
                      const TfLiteTensor* output, int depth_multiplier,
                      ::ml_drift::DepthwiseConvolution2DAttributes* attr) {
  const int input_depth = input->dims->data[3];
  const int filter_height = filter->dims->data[1];
  const int filter_width = filter->dims->data[2];
  const int kernel_spatial_size = filter_height * filter_width;

  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32> weights;
  const auto& src_weights = GetFloatWeights(*attr);
  weights.id = src_weights.id;
  weights.shape = ::ml_drift::OHWI(depth_multiplier, filter_height,
                                   filter_width, input_depth);
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  float* dst = weights.data.data();
  const float* src = src_weights.data.data();

  const int src_outer_stride = input_depth * depth_multiplier;
  const int dst_m_stride = kernel_spatial_size * input_depth;
  for (int m = 0; m < depth_multiplier; ++m) {
    for (int s = 0; s < kernel_spatial_size; ++s) {
      const float* current_src = src + s * src_outer_stride + m;
      float* current_dst = dst + m * dst_m_stride + s * input_depth;
      for (int i = 0; i < input_depth; ++i) {
        current_dst[i] = current_src[i * depth_multiplier];
      }
    }
  }

  attr->weights.emplace<
      ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>(
      std::move(weights));
}
}  // namespace

void ConvertDepthwiseConv(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& options, ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* dw_conv_op = ir_model.add_op();
  dw_conv_op->name = ToString(::ml_drift::OperationType::DEPTHWISE_CONVOLUTION);

  ::ml_drift::DepthwiseConvolution2DAttributes attr;
  auto& weights = attr.weights.emplace<
      ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();

  const int input_id = node.inputs->data[0];
  ir_model.AddConsumer(tensor_map[input_id], dw_conv_op->id);

  const TfLiteTensor* weights_tensor = context.tensors + node.inputs->data[1];
  if (tflite::IsConstantTensor(weights_tensor)) {
    PopulateTensor(weights_tensor, node.inputs->data[1], &weights,
                   PopulateTensorFlags::kExtraBytes,
                   options.enable_spanned_weights);
  } else {
    const ::ml_drift::ir::IrTensorId weights_id =
        tensor_map[node.inputs->data[1]];
    ir_model.AddConsumer(weights_id, dw_conv_op->id);
    const ::ml_drift::BHWC weights_shape =
        ir_model.tensor(weights_id)->desc.GetBHWCShape();
    weights.shape = ::ml_drift::OHWI(weights_shape.b, weights_shape.h,
                                     weights_shape.w, weights_shape.c);
  }

  const bool has_bias =
      node.inputs->size > 2 && node.inputs->data[2] != kTfLiteOptionalTensor;
  if (has_bias) {
    const TfLiteTensor* bias_tensor = context.tensors + node.inputs->data[2];
    PopulateTensor(bias_tensor, node.inputs->data[2], &attr.bias,
                   PopulateTensorFlags::kNoExtraBytes,
                   options.enable_spanned_weights);
  }

  const auto* params =
      static_cast<const TfLiteDepthwiseConvParams*>(node.builtin_data);
  attr.strides = ToHW(params->stride_height, params->stride_width);
  attr.dilations = ::ml_drift::HW(std::max(1, params->dilation_height_factor),
                                  std::max(1, params->dilation_width_factor));

  UpdatePadding(params->padding,
                ir_model.tensor(tensor_map[input_id])->desc.GetBHWDCShape(),
                &attr);
  HandleFusedActivation(params->activation, ir_model, dw_conv_op, tensor_map,
                        node.outputs->data[0]);

  const int depth_multiplier = params->depth_multiplier;
  if (depth_multiplier != 1 && tflite::IsConstantTensor(weights_tensor)) {
    const TfLiteTensor* input_tensor = context.tensors + input_id;
    const TfLiteTensor* output_tensor = context.tensors + node.outputs->data[0];
    TransposeWeights(input_tensor, weights_tensor, output_tensor,
                     depth_multiplier, &attr);
  }
  dw_conv_op->attr = std::move(attr);
}

}  // namespace litert::ml_drift::ir
