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

#include "ml_drift_delegate/tflite/convert/convert_pooling2d.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

bool IsGlobalPooling(const ::ml_drift::Pooling2DAttributes& attr,
                     const ::ml_drift::BHWDC& src_shape,
                     const ::ml_drift::BHWDC& dst_shape) {
  return dst_shape.w == 1 && dst_shape.h == 1 && attr.kernel.w == src_shape.w &&
         attr.kernel.h == src_shape.h && attr.padding.appended.w == 0 &&
         attr.padding.appended.h == 0 && attr.padding.prepended.w == 0 &&
         attr.padding.prepended.h == 0;
}

bool IsGlobalAveragePooling(const ::ml_drift::Pooling2DAttributes& attr,
                            const ::ml_drift::BHWDC& src_shape,
                            const ::ml_drift::BHWDC& dst_shape) {
  return attr.type == ::ml_drift::PoolingType::AVERAGE &&
         attr.output_indices == false &&
         IsGlobalPooling(attr, src_shape, dst_shape);
}

bool TryConvertGlobalAveragePoolingToMean(
    const TfLiteContext& context, const TfLiteNode& node,
    const ::ml_drift::Pooling2DAttributes& attr, const TfLitePoolParams* params,
    ::ml_drift::ir::TensorMap& tensor_map, ::ml_drift::ir::IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const ::ml_drift::ir::IrTensor* input_tensor =
      ir_model.tensor(tensor_map[input_id]);
  const ::ml_drift::BHWDC output_shape =
      ir_model.tensor(tensor_map[node.outputs->data[0]])->desc.GetBHWDCShape();

  if (!IsGlobalAveragePooling(attr, input_tensor->desc.GetBHWDCShape(),
                              output_shape)) {
    return false;
  }

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::MEAN);

  ::ml_drift::ReduceAttributes mean_attr;
  mean_attr.dims = {::ml_drift::Axis::WIDTH, ::ml_drift::Axis::HEIGHT};

  ir_model.AddConsumer(tensor_map[input_id], op->id);
  const int output_id = node.outputs->data[0];
  HandleFusedActivation(params->activation, ir_model, op, tensor_map,
                        output_id);
  op->attr = std::move(mean_attr);
  return true;
}

}  // namespace

void ConvertPooling2d(const TfLiteContext& context, const TfLiteNode& node,
                      const TfLiteRegistration& registration,
                      ::ml_drift::ir::TensorMap& tensor_map,
                      ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::PoolingType pooling_type = ::ml_drift::PoolingType::UNDEFINED;
  if (registration.builtin_code == kTfLiteBuiltinAveragePool2d) {
    pooling_type = ::ml_drift::PoolingType::AVERAGE;
  } else if (registration.builtin_code == kTfLiteBuiltinMaxPool2d ||
             registration.builtin_code == kTfLiteBuiltinCustom) {
    pooling_type = ::ml_drift::PoolingType::MAX;
  } else {
    ABSL_LOG(FATAL) << "Unsupported pooling type: "
                    << registration.builtin_code;
  }

  const TfLitePoolParams* params = nullptr;
  if (node.custom_initial_data) {
    params = static_cast<const TfLitePoolParams*>(node.custom_initial_data);
  } else {
    params = static_cast<const TfLitePoolParams*>(node.builtin_data);
  }

  ::ml_drift::Pooling2DAttributes attr;
  attr.type = pooling_type;
  attr.kernel = ToHW(params->filter_height, params->filter_width);
  attr.strides = ToHW(params->stride_height, params->stride_width);

  const int input_id = node.inputs->data[0];
  const ::ml_drift::ir::IrTensor* input_tensor =
      ir_model.tensor(tensor_map[input_id]);
  UpdatePadding(params->padding, input_tensor->desc.GetBHWDCShape(), &attr);

  attr.output_indices = node.outputs->size == 2;

  if (TryConvertGlobalAveragePoolingToMean(context, node, attr, params,
                                           tensor_map, ir_model)) {
    return;
  }

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::POOLING_2D);

  ir_model.AddConsumer(tensor_map[input_id], op->id);
  const int output_id = node.outputs->data[0];
  HandleFusedActivation(params->activation, ir_model, op, tensor_map,
                        output_id);

  if (attr.output_indices) {
    ir_model.SetProducer(tensor_map[node.outputs->data[1]], op->id);
  }

  op->attr = std::move(attr);
}

}  // namespace litert::ml_drift::ir
