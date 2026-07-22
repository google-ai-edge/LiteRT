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

#include "ml_drift_delegate/tflite/convert/convert_aux.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {
namespace {
::ml_drift::BHWC GetShape(const ::ml_drift::BHWC& shape,
                          const SizedLayout& layout, int num_dims) {
  if ((num_dims == 0 || num_dims == 1) &&
      layout.layout_1d == ::ml_drift::Layout::SCALAR) {
    return ::ml_drift::BHWC(1, 1, 1, shape.b);
  } else if (num_dims == 2 && layout.layout_2d == ::ml_drift::Layout::HW) {
    return ::ml_drift::BHWC(1, 1, shape.b, shape.c);
  } else if (num_dims == 3 && layout.layout_3d == ::ml_drift::Layout::HWC) {
    return ::ml_drift::BHWC(1, shape.b, shape.w, shape.c);
  } else {
    return shape;
  }
}

// Helper to copy tensor data and set value/attribute fields.
template <typename TensorType>
::ml_drift::ir::IrTensor* SetValueAndAttrFromTfLiteTensor(
    const TfLiteContext& context, int tensor_id, const SizedLayout& layout,
    ::ml_drift::ir::IrModel& ir_model,
    ::ml_drift::ConstTensorAttributes& attr) {
  TensorType t;
  const TfLiteTensor* tfl_tensor = context.tensors + tensor_id;
  PopulateTensor<TensorType>(tfl_tensor, tensor_id, &t,
                             PopulateTensorFlags::kNoExtraBytes);
  const ::ml_drift::BHWC shape =
      GetShape(t.shape, layout, tfl_tensor->dims->size);
  ::ml_drift::ir::IrTensor* tensor = ir_model.add_tensor(
      t.kType, ::ml_drift::BHWDC(shape.b, shape.h, shape.w, 1, shape.c));
  attr.tensor = std::move(t);
  return tensor;
}
}  // namespace

namespace convert_aux_internal {

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::Scalar* shape) {
  for (int i = 0; i < dims->size; ++i) ABSL_QCHECK_EQ(dims->data[i], 1);
  shape->v = 1;
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::Linear* shape) {
  ABSL_QCHECK(IsLinearConvertible(dims));
  shape->v = dims->data[dims->size - 1];
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::HWC* shape) {
  if (dims->size == 3) {
    shape->h = dims->data[0];
    shape->w = dims->data[1];
    shape->c = dims->data[2];
    return;
  }
  if (dims->size == 4) {
    ABSL_QCHECK_EQ(dims->data[0], 1);
    shape->h = dims->data[1];
    shape->w = dims->data[2];
    shape->c = dims->data[3];
    return;
  }
  ABSL_LOG(FATAL) << "Expected 3D or 4D (1xHxWxC) tensor for HWC";
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::HW* shape) {
  ABSL_QCHECK_EQ(dims->size, 2);
  shape->h = dims->data[0];
  shape->w = dims->data[1];
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::OHWI* shape) {
  ABSL_QCHECK_EQ(dims->size, 4);
  shape->o = dims->data[0];
  shape->h = dims->data[1];
  shape->w = dims->data[2];
  shape->i = dims->data[3];
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::BHWC* shape) {
  shape->CopyAllDefinedAxis(ExtractTensorShape(dims));
}

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::BHWDC* shape) {
  shape->CopyAllDefinedAxis(ExtractTensorShape(dims));
}

template <>
void CopyData<float>(const TfLiteTensor& src, float* dst) {
  const TfLiteType dtype = src.type;
  if (dtype == kTfLiteFloat32 ||  //
      dtype == kTfLiteFloat16 ||  //
      dtype == kTfLiteInt4 ||     //
      dtype == kTfLiteInt8 ||     //
      dtype == kTfLiteUInt8 ||    //
      dtype == kTfLiteInt32) {
    CopyFloat32Data(&src, dst);
    return;
  }
  ABSL_LOG(FATAL) << absl::StrCat(tflite::GetTensorDebugString(&src),
                                  " has unsupported dtype.");
}

}  // namespace convert_aux_internal

void HandleFusedActivation(
    TfLiteFusedActivation fused_activation,
    ::ml_drift::ir::IrModel& ir_model, ::ml_drift::ir::IrOp* op,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    int output_id) {
  if (fused_activation == kTfLiteActNone) {
    ir_model.SetProducer(tensor_map[output_id], op->id);
    return;
  }

  const ::ml_drift::TensorDescriptor& output_desc =
      ir_model.tensor(tensor_map[output_id])->desc;
  ::ml_drift::ir::IrTensor* activation_input = ir_model.add_tensor(output_desc);
  ir_model.SetProducer(activation_input->id, op->id);
  ::ml_drift::ir::IrOp* activation_op = ir_model.add_op();
  ir_model.AddConsumer(activation_input->id, activation_op->id);
  ir_model.SetProducer(tensor_map[output_id], activation_op->id);

  switch (fused_activation) {
    case kTfLiteActRelu:
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6: {
      ::ml_drift::ReLUAttributes attr;
      attr.activation_max =
          fused_activation == kTfLiteActRelu
              ? 0.0f
              : (fused_activation == kTfLiteActReluN1To1 ? 1.0f : 6.0f);
      attr.activation_min =
          fused_activation == kTfLiteActReluN1To1 ? -1.0f : 0.0f;
      activation_op->name = ToString(::ml_drift::OperationType::RELU);
      activation_op->attr = attr;
      return;
    }
    case kTfLiteActTanh: {
      activation_op->name = ToString(::ml_drift::OperationType::TANH);
      return;
    }
    case kTfLiteActSigmoid: {
      activation_op->name = ToString(::ml_drift::OperationType::SIGMOID);
      return;
    }
    case kTfLiteActSignBit: {
      activation_op->name = ToString(::ml_drift::OperationType::SIGN);
      return;
    }
    case kTfLiteActNone:
      return;
  }
}

bool MarkSharedBias(::ml_drift::ir::IrTensorId bias_id,
                    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrTensor* bias = ir_model.GetMutableTensor(bias_id);
  if (bias == nullptr || !bias->buffer_source.is_shared) {
    return false;
  }
  // Shared bias tensors are passed as runtime inputs and must be materialized
  // with LINEAR layout (parity with GraphFloat32).
  bias->buffer_source.force_linear_layout = true;
  // ExtractTensorShape() places a 1-D bias's length in the batch dim; move it
  // to channels so the shared-memory manager's LINEAR reshape to (1,1,1,c)
  // preserves the channels (parity with GraphFloat32).
  const ::ml_drift::BHWC shape = bias->desc.GetBHWCShape();
  if (shape.b != 1 && shape.c == 1) {
    bias->desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, shape.b));
  }
  return true;
}

::ml_drift::ir::IrTensor* AddConstInput(const TfLiteContext& context,
                                        int tensor_id,
                                        ::ml_drift::ir::IrModel& ir_model,
                                        const SizedLayout& layout) {
  const TfLiteTensor* tfl_tensor = context.tensors + tensor_id;
  ABSL_CHECK(
      tfl_tensor &&
      (tfl_tensor->type == kTfLiteFloat32 ||
       tfl_tensor->type == kTfLiteFloat16 || tfl_tensor->type == kTfLiteInt8 ||
       tfl_tensor->type == kTfLiteUInt8 || tfl_tensor->type == kTfLiteInt4 ||
       tfl_tensor->type == kTfLiteInt2 || tfl_tensor->type == kTfLiteBool ||
       tfl_tensor->type == kTfLiteInt32));
  ::ml_drift::ir::IrOp* node = ir_model.add_op();
  node->name = ToString(::ml_drift::OperationType::CONSTANT);
  ::ml_drift::ir::IrTensor* tensor;
  ::ml_drift::ConstTensorAttributes attr;
  if (tfl_tensor->type == kTfLiteFloat16) {
    tensor = SetValueAndAttrFromTfLiteTensor<::ml_drift::TensorFloat16>(
        context, tensor_id, layout, ir_model, attr);
  } else if (tfl_tensor->type == kTfLiteFloat32 ||
             tfl_tensor->type == kTfLiteInt8 ||
             tfl_tensor->type == kTfLiteUInt8 ||
             tfl_tensor->type == kTfLiteInt4 ||
             tfl_tensor->type == kTfLiteInt2) {
    // Note: kTfLiteInt8, kTfLiteUInt8, kTfLiteInt4, kTfLiteInt2 are currently
    // read as TensorFloat32.
    tensor = SetValueAndAttrFromTfLiteTensor<::ml_drift::TensorFloat32>(
        context, tensor_id, layout, ir_model, attr);
  } else if (tfl_tensor->type == kTfLiteBool) {
    tensor = SetValueAndAttrFromTfLiteTensor<::ml_drift::TensorBool>(
        context, tensor_id, layout, ir_model, attr);
  } else if (tfl_tensor->type == kTfLiteInt32) {
    tensor = SetValueAndAttrFromTfLiteTensor<::ml_drift::TensorInt32>(
        context, tensor_id, layout, ir_model, attr);
  } else {
    ABSL_LOG(FATAL) << "Unsupported dtype: " << tfl_tensor->type;
  }
  ir_model.SetProducer(tensor->id, node->id);
  node->attr = std::move(attr);
  return tensor;
}

::ml_drift::Axis ExtractAxisFromIndex(const TfLiteTensor& tflite_tensor,
                                      int index) {
  const TfLiteIntArray* dims = tflite_tensor.dims;
  index = ResolveNegativeIndex(index, tflite_tensor.dims->size);
  std::vector<::ml_drift::Axis> index_to_axis;
  if (dims->size == 1) {
    index_to_axis = {::ml_drift::Axis::BATCH};
  } else if (dims->size == 2) {
    index_to_axis = {::ml_drift::Axis::BATCH, ::ml_drift::Axis::CHANNELS};
  } else if (dims->size == 3) {
    index_to_axis = {::ml_drift::Axis::BATCH, ::ml_drift::Axis::WIDTH,
                     ::ml_drift::Axis::CHANNELS};
  } else if (dims->size == 4) {
    index_to_axis = {::ml_drift::Axis::BATCH, ::ml_drift::Axis::HEIGHT,
                     ::ml_drift::Axis::WIDTH, ::ml_drift::Axis::CHANNELS};
  } else {
    index_to_axis = {::ml_drift::Axis::BATCH, ::ml_drift::Axis::HEIGHT,
                     ::ml_drift::Axis::WIDTH, ::ml_drift::Axis::DEPTH,
                     ::ml_drift::Axis::CHANNELS};
  }
  return index_to_axis[index];
}

::ml_drift::BHWC GetRightAlignedBHWC(const std::vector<int32_t>& values,
                                     int32_t start_val) {
  const int size = values.size();
  if (size == 0) {
    return ::ml_drift::BHWC(start_val, start_val, start_val, start_val);
  } else if (size == 1) {
    return ::ml_drift::BHWC(start_val, start_val, start_val, values[0]);
  } else if (size == 2) {
    return ::ml_drift::BHWC(start_val, start_val, values[0], values[1]);
  } else if (size == 3) {
    return ::ml_drift::BHWC(start_val, values[0], values[1], values[2]);
  } else {
    // Drop the outermost dimension if size >= 4.
    const int offset = size - 4;
    return ::ml_drift::BHWC(values[offset], values[offset + 1],
                            values[offset + 2], values[offset + 3]);
  }
}

::ml_drift::BHWDC GetRightAlignedBHWDC(const std::vector<int32_t>& values,
                                       int32_t start_val) {
  const int size = values.size();
  if (size == 0) {
    return ::ml_drift::BHWDC(start_val, start_val, start_val, start_val,
                             start_val);
  } else if (size == 1) {
    return ::ml_drift::BHWDC(start_val, start_val, start_val, start_val,
                             values[0]);
  } else if (size == 2) {
    return ::ml_drift::BHWDC(start_val, start_val, start_val, values[0],
                             values[1]);
  } else if (size == 3) {
    return ::ml_drift::BHWDC(start_val, start_val, values[0], values[1],
                             values[2]);
  } else if (size == 4) {
    return ::ml_drift::BHWDC(start_val, values[0], values[1], values[2],
                             values[3]);
  } else {
    return ::ml_drift::BHWDC(values[0], values[1], values[2], values[3],
                             values[4]);
  }
}

}  // namespace litert::ml_drift::ir
