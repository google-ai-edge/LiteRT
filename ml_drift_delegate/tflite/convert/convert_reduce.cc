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

#include "ml_drift_delegate/tflite/convert/convert_reduce.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

namespace {

::ml_drift::OperationType GetReduceOperationType(int builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinMean:
      return ::ml_drift::OperationType::MEAN;
    case kTfLiteBuiltinReduceAll:
      return ::ml_drift::OperationType::REDUCE_ALL;
    case kTfLiteBuiltinReduceAny:
      return ::ml_drift::OperationType::REDUCE_ANY;
    case kTfLiteBuiltinReduceMax:
      return ::ml_drift::OperationType::REDUCE_MAXIMUM;
    case kTfLiteBuiltinReduceMin:
      return ::ml_drift::OperationType::REDUCE_MINIMUM;
    case kTfLiteBuiltinReduceProd:
      return ::ml_drift::OperationType::REDUCE_PRODUCT;
    case kTfLiteBuiltinSum:
      return ::ml_drift::OperationType::REDUCE_SUM;
    default:
      ABSL_LOG(FATAL) << "Unsupported reduce op builtin code: " << builtin_code;
  }
}

}  // namespace

void ConvertReduce(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const int axes_id = node.inputs->data[1];
  const int output_id = node.outputs->data[0];

  const TfLiteTensor& input_tensor = context.tensors[input_id];
  const TfLiteTensor& axes_tensor = context.tensors[axes_id];
  const TfLiteTensor& output_tensor = context.tensors[output_id];

  const auto* params =
      static_cast<const TfLiteReducerParams*>(node.builtin_data);
  const bool keep_dims = params->keep_dims;

  ::ml_drift::ReduceAttributes attr;
  // Extract reduction axes.
  const int num_axes = tflite::NumElements(&axes_tensor);
  const int* axes_data = tflite::GetTensorData<int32_t>(&axes_tensor);
  for (int i = 0; i < num_axes; ++i) {
    attr.dims.insert(ExtractAxisFromIndex(input_tensor, axes_data[i]));
  }

  ::ml_drift::ir::IrOp* reduce_op = ir_model.add_op();
  reduce_op->name = ToString(GetReduceOperationType(registration.builtin_code));
  reduce_op->attr = attr;

  ir_model.AddConsumer(tensor_map[input_id], reduce_op->id);

  if (keep_dims) {
    ir_model.SetProducer(tensor_map[output_id], reduce_op->id);
  } else {
    // If keep_dims is false, we need to add an explicit RESHAPE node
    // because ml_drift's REDUCE operations often preserve the rank
    // (with dimensions of size 1) depending on the backend.
    const ::ml_drift::TensorDescriptor& input_desc =
        ir_model.tensor(tensor_map[input_id])->desc;
    const ::ml_drift::DataType dtype = input_desc.GetDataType();
    ::ml_drift::BHWDC reduce_shape = input_desc.GetBHWDCShape();
    // Update reduce_output shape to have 1s in reduced dimensions.
    for (const auto& axis : attr.dims) {
      if (axis == ::ml_drift::Axis::BATCH) reduce_shape.b = 1;
      if (axis == ::ml_drift::Axis::HEIGHT) reduce_shape.h = 1;
      if (axis == ::ml_drift::Axis::WIDTH) reduce_shape.w = 1;
      if (axis == ::ml_drift::Axis::DEPTH) reduce_shape.d = 1;
      if (axis == ::ml_drift::Axis::CHANNELS) reduce_shape.c = 1;
    }
    ::ml_drift::ir::IrTensor* reduce_output =
        ir_model.add_tensor(dtype, reduce_shape);
    ir_model.SetProducer(reduce_output->id, reduce_op->id);

    ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
    reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
    if (output_tensor.dims->size <= 4) {
      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape.CopyAllDefinedAxis(
          ExtractTensorShape(output_tensor.dims));
      reshape_op->attr = std::move(reshape_attr);
    } else {
      ::ml_drift::Reshape3DAttributes reshape_attr;
      reshape_attr.new_shape.CopyAllDefinedAxis(
          ExtractTensorShape(output_tensor.dims));
      reshape_op->attr = std::move(reshape_attr);
    }

    ir_model.AddConsumer(reduce_output->id, reshape_op->id);
    ir_model.SetProducer(tensor_map[output_id], reshape_op->id);
  }
}

}  // namespace litert::ml_drift::ir
