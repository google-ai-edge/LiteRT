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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_transpose_conv.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

void ConvertTransposeConv(const TfLiteContext& context, const TfLiteNode& node,
                          const TfLiteRegistration& registration,
                          ::ml_drift::ir::TensorMap& tensor_map,
                          const IrModelBuilderOptions& options,
                          ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* ir_op = ir_model.add_op();
  ir_op->name = ToString(::ml_drift::OperationType::CONVOLUTION_TRANSPOSED);

  const bool builtin_op =
      registration.builtin_code == kTfLiteBuiltinTransposeConv;
  const int input_id = builtin_op ? node.inputs->data[2] : node.inputs->data[0];
  const int weights_id = node.inputs->data[1];
  std::optional<int> bias_id = std::nullopt;
  if (node.inputs->size == 4) {  // must be builtin op
    bias_id = node.inputs->data[3];
  } else if (!builtin_op && node.inputs->size == 3) {
    bias_id = node.inputs->data[2];
  }

  ir_model.AddConsumer(tensor_map[input_id], ir_op->id);

  ::ml_drift::ConvolutionTransposedAttributes attr;

  if (::tflite::IsConstantTensor(&context.tensors[weights_id])) {
    PopulateTensor(&context.tensors[weights_id], weights_id, &attr.weights,
                   PopulateTensorFlags::kExtraBytes,
                   options.enable_spanned_weights);
  } else {
    // Create dummy weights to bypass checks if dynamic
    ir_model.AddConsumer(tensor_map[weights_id], ir_op->id);
    const auto* dims = context.tensors[weights_id].dims;
    std::vector<int32_t> dims_vec(dims->data, dims->data + dims->size);
    auto weights_shape = GetRightAlignedBHWC(dims_vec, 4);
    attr.weights.shape = ::ml_drift::OHWI(weights_shape.b, weights_shape.h,
                                          weights_shape.w, weights_shape.c);
  }

  if (bias_id.has_value() &&
      ::tflite::IsConstantTensor(&context.tensors[bias_id.value()])) {
    PopulateTensor(&context.tensors[bias_id.value()], bias_id.value(),
                   &attr.bias, PopulateTensorFlags::kNoExtraBytes,
                   options.enable_spanned_weights);
  }

  TfLitePadding padding = kTfLitePaddingUnknown;
  if (builtin_op) {
    const auto* params =
        static_cast<const TfLiteTransposeConvParams*>(node.builtin_data);
    attr.stride =
        params ? ::ml_drift::HW(params->stride_height, params->stride_width)
               : ::ml_drift::HW(1, 1);
    if (params) padding = params->padding;
  } else {
    const auto* params =
        static_cast<const TfLiteTransposeConvParams*>(node.custom_initial_data);
    attr.stride =
        params ? ::ml_drift::HW(params->stride_height, params->stride_width)
               : ::ml_drift::HW(1, 1);
    if (params) padding = params->padding;
  }

  UpdatePadding(padding,
                ir_model.tensor(tensor_map[input_id])->desc.GetBHWDCShape(),
                &attr);

  ir_op->attr = std::move(attr);
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], ir_op->id);
}

}  // namespace litert::ml_drift::ir
