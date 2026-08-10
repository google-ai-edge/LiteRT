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

#include "ml_drift_delegate/tflite/convert/convert_batch_matmul.h"

#include <utility>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

void ConvertBatchMatMul(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const int input0_id = node.inputs->data[0];
  const int input1_id = node.inputs->data[1];
  const int output_id = node.outputs->data[0];

  const TfLiteTensor& input1_tensor = context.tensors[input1_id];

  const TfLiteBatchMatMulParams* params =
      reinterpret_cast<const TfLiteBatchMatMulParams*>(node.builtin_data);

  // Case 1: 1 runtime input + 2D constant weights -> FULLY_CONNECTED
  if (tflite::IsConstantTensor(&input1_tensor) &&
      input1_tensor.dims->size == 2 &&
      !ir_model.tensor(tensor_map[input1_id])->buffer_source.is_shared) {
    ::ml_drift::ir::IrOp* fc_op = ir_model.add_op();
    fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED);

    ir_model.AddConsumer(tensor_map[input0_id], fc_op->id);
    ir_model.SetProducer(tensor_map[output_id], fc_op->id);

    ::ml_drift::Tensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32> weights;
    PopulateTensor(&input1_tensor, input1_id, &weights,
                   PopulateTensorFlags::kExtraBytes);
    ::ml_drift::FullyConnectedAttributes attr;
    const int h = weights.shape.h;
    const int w = weights.shape.w;
    attr.weights.data.resize(w * h + XNN_EXTRA_BYTES / sizeof(float));

    for (int i = 0; i < w; ++i) {
      for (int j = 0; j < h; ++j) {
        attr.weights.data[i * h + j] = weights.data[j * w + i];
      }
    }

    attr.weights.id = weights.id;
    attr.weights.shape = ::ml_drift::OHWI(w, 1, 1, h);
    fc_op->attr = std::move(attr);
    return;
  }

  // Case 2: Standard BATCHED_MATMUL
  ::ml_drift::ir::IrTensorId left_id = tensor_map[input0_id];
  ::ml_drift::ir::IrTensorId right_id = tensor_map[input1_id];
  ::ml_drift::ir::IrTensorId result_id = tensor_map[output_id];

  if (tflite::IsConstantTensor(&input1_tensor)) {
    ::ml_drift::ir::IrTensor* const_tensor =
        AddConstInput(context, input1_id, ir_model, {});
    right_id = const_tensor->id;
  }

  const ::ml_drift::ir::IrTensor* input0_desc = ir_model.tensor(left_id);
  const ::ml_drift::ir::IrTensor* input1_desc = ir_model.tensor(right_id);
  const ::ml_drift::ir::IrTensor* output_desc = ir_model.tensor(result_id);

  ::ml_drift::BHWDC l_shape = input0_desc->desc.GetBHWDCShape();
  ::ml_drift::BHWDC r_shape = input1_desc->desc.GetBHWDCShape();
  ::ml_drift::BHWDC out_shape = output_desc->desc.GetBHWDCShape();

  // MLDrift supports batched matmul with single batch.
  // Model can have model batch in addition to matmul batch. In this case we
  // reshape inputs/outputs to have single batch in MLDrift. For example
  // 2x4x128x32 (2 is model batch, 4 is matmul batch) will be reshaped to
  // 1x8x128x32.
  // If shape is 2d(MxN) MLDrift treats it as Mx1x1xN. In this case we need
  // to make reshape to get 1x1xMxN.
  bool left_is_5d =
      input0_desc->desc.GetLayout() == ::ml_drift::Layout::BHWDC ||
      l_shape.d > 1;
  if (left_is_5d || l_shape.b != 1) {
    ::ml_drift::ir::IrOp* reshape_left = ir_model.add_op();
    reshape_left->name = ToString(::ml_drift::OperationType::RESHAPE);
    ::ml_drift::ReshapeAttributes reshape_attr;
    if (left_is_5d) {
      reshape_attr.new_shape = ::ml_drift::BHWC(
          1, l_shape.b * l_shape.h * l_shape.w, l_shape.d, l_shape.c);
    } else {
      if (l_shape.h == 1 && l_shape.w == 1) {
        reshape_attr.new_shape = ::ml_drift::BHWC(1, 1, l_shape.b, l_shape.c);
      } else {
        reshape_attr.new_shape =
            ::ml_drift::BHWC(1, l_shape.b * l_shape.h, l_shape.w, l_shape.c);
      }
    }
    reshape_left->attr = reshape_attr;

    ir_model.AddConsumer(left_id, reshape_left->id);
    ::ml_drift::ir::IrTensor* left_tensor = ir_model.add_tensor(
        input0_desc->desc.GetDataType(), reshape_attr.new_shape);
    ir_model.SetProducer(left_tensor->id, reshape_left->id);
    left_id = left_tensor->id;
  }

  bool right_is_5d =
      input1_desc->desc.GetLayout() == ::ml_drift::Layout::BHWDC ||
      r_shape.d > 1;
  if (right_is_5d || r_shape.b != 1) {
    ::ml_drift::ir::IrOp* reshape_right = ir_model.add_op();
    reshape_right->name = ToString(::ml_drift::OperationType::RESHAPE);
    ::ml_drift::ReshapeAttributes reshape_attr;
    if (right_is_5d) {
      reshape_attr.new_shape = ::ml_drift::BHWC(
          1, r_shape.b * r_shape.h * r_shape.w, r_shape.d, r_shape.c);
    } else {
      if (r_shape.h == 1 && r_shape.w == 1) {
        reshape_attr.new_shape = ::ml_drift::BHWC(1, 1, r_shape.b, r_shape.c);
      } else {
        reshape_attr.new_shape =
            ::ml_drift::BHWC(1, r_shape.b * r_shape.h, r_shape.w, r_shape.c);
      }
    }
    reshape_right->attr = reshape_attr;

    ir_model.AddConsumer(right_id, reshape_right->id);
    ::ml_drift::ir::IrTensor* right_tensor = ir_model.add_tensor(
        input1_desc->desc.GetDataType(), reshape_attr.new_shape);
    ir_model.SetProducer(right_tensor->id, reshape_right->id);
    right_id = right_tensor->id;
  }

  bool out_is_5d = output_desc->desc.GetLayout() == ::ml_drift::Layout::BHWDC ||
                   out_shape.d > 1;
  if (out_is_5d || out_shape.b != 1) {
    ::ml_drift::BHWC flat_out_shape;
    if (out_is_5d) {
      flat_out_shape = ::ml_drift::BHWC(
          1, out_shape.b * out_shape.h * out_shape.w, out_shape.d, out_shape.c);
    } else {
      if (out_shape.h == 1 && out_shape.w == 1) {
        flat_out_shape = ::ml_drift::BHWC(1, 1, out_shape.b, out_shape.c);
      } else {
        flat_out_shape = ::ml_drift::BHWC(1, out_shape.b * out_shape.h,
                                          out_shape.w, out_shape.c);
      }
    }
    ::ml_drift::ir::IrTensor* result_tensor =
        ir_model.add_tensor(output_desc->desc.GetDataType(), flat_out_shape);
    result_id = result_tensor->id;
  }

  ::ml_drift::ir::IrOp* bmm_op = ir_model.add_op();
  bmm_op->name = ToString(::ml_drift::OperationType::BATCHED_MATMUL);

  ir_model.AddConsumer(left_id, bmm_op->id);
  ir_model.AddConsumer(right_id, bmm_op->id);
  ir_model.SetProducer(result_id, bmm_op->id);

  ::ml_drift::BatchedMatMulAttributes attr;
  attr.transpose_left = params ? params->adj_x : false;
  attr.transpose_right = params ? params->adj_y : false;

  bmm_op->attr = std::move(attr);

  if (out_is_5d || out_shape.b != 1) {
    ::ml_drift::ir::IrOp* reshape_result = ir_model.add_op();
    reshape_result->name = ToString(::ml_drift::OperationType::RESHAPE);
    if (out_is_5d) {
      ::ml_drift::Reshape3DAttributes reshape_attr;
      reshape_attr.new_shape = ::ml_drift::BHWDC(
          out_shape.b, out_shape.h, out_shape.w, out_shape.d, out_shape.c);
      reshape_result->attr = reshape_attr;
    } else {
      ::ml_drift::ReshapeAttributes reshape_attr;
      reshape_attr.new_shape =
          ::ml_drift::BHWC(out_shape.b, out_shape.h, out_shape.w, out_shape.c);
      reshape_result->attr = reshape_attr;
    }

    ir_model.AddConsumer(result_id, reshape_result->id);
    ir_model.SetProducer(tensor_map[output_id], reshape_result->id);
  }
}

}  // namespace litert::ml_drift::ir
