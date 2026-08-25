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

#include "fp16.h"  // from @FP16
#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
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

// Returns the scale/zero-point shape for a quantized fully-connected weights
// tensor of the given OHWI `weights_shape`. Scale is per-output-channel by
// default; blockwise quantization additionally splits the input dimension into
// blocks: i / blocksize.
::ml_drift::OHWI QuantizedFullyConnectedScaleShape(
    const TfLiteTensor& weights_tensor, const ::ml_drift::OHWI& weights_shape) {
  ::ml_drift::OHWI scale_shape(weights_shape.o, 1, 1, 1);
  if (weights_tensor.quantization.type == kTfLiteBlockwiseQuantization) {
    const auto* qparams = reinterpret_cast<const TfLiteBlockwiseQuantization*>(
        weights_tensor.quantization.params);
    scale_shape.i = weights_shape.i / qparams->blocksize;
  }
  return scale_shape;
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

bool ConfigSharedQuantizedFullyConnected(
    const TfLiteTensor& weights_tensor, const ::ml_drift::OHWI& weights_shape,
    ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> bias,
    ::ml_drift::ir::IrOp* fc_op) {
  // Scale is per-output-channel by default; blockwise quantization also splits
  // the input dimension into blocks (parity with GraphFloat32).
  const ::ml_drift::OHWI scale_shape =
      QuantizedFullyConnectedScaleShape(weights_tensor, weights_shape);

  const bool has_quant =
      weights_tensor.quantization.type == kTfLiteAffineQuantization;
  const auto* qparams = has_quant
                            ? reinterpret_cast<const TfLiteAffineQuantization*>(
                                  weights_tensor.quantization.params)
                            : nullptr;

  auto populate_quant_attrs = [&](auto& attr) {
    attr.scale.shape = scale_shape;
    const int num_scales =
        scale_shape.o * scale_shape.h * scale_shape.w * scale_shape.i;
    if (qparams && qparams->scale && qparams->scale->size > 0) {
      attr.scale.data = std::vector<float>(
          qparams->scale->data, qparams->scale->data + qparams->scale->size);
      if (attr.scale.data.size() == 1 && num_scales > 1) {
        attr.scale.data.resize(num_scales, attr.scale.data[0]);
      }
    }
    attr.zero_point.shape = scale_shape;
    if (qparams && qparams->zero_point && qparams->zero_point->size > 0) {
      attr.zero_point.data = std::vector<int32_t>(
          qparams->zero_point->data,
          qparams->zero_point->data + qparams->zero_point->size);
      if (attr.zero_point.data.size() == 1 && num_scales > 1) {
        attr.zero_point.data.resize(num_scales, attr.zero_point.data[0]);
      }
    } else {
      attr.zero_point.data.resize(num_scales, 0);
    }
  };

  switch (weights_tensor.type) {
    case kTfLiteInt8: {
      fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT8);
      ::ml_drift::FullyConnectedInt8Attributes attr;
      attr.weights.shape = weights_shape;
      attr.scale.shape = scale_shape;
      populate_quant_attrs(attr);
      attr.bias = std::move(bias);
      fc_op->attr = std::move(attr);
      return true;
    }
    case kTfLiteInt4: {
      fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT4);
      ::ml_drift::FullyConnectedInt4Attributes attr;
      ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT4> weights;
      weights.shape = weights_shape;
      attr.weights = std::move(weights);
      attr.scale.shape = scale_shape;
      populate_quant_attrs(attr);
      attr.bias = std::move(bias);
      fc_op->attr = std::move(attr);
      return true;
    }
    case kTfLiteInt2: {
      fc_op->name = ToString(::ml_drift::OperationType::FULLY_CONNECTED_INT2);
      ::ml_drift::FullyConnectedInt2Attributes attr;
      ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT2> weights;
      weights.shape = weights_shape;
      attr.weights = std::move(weights);
      attr.scale.shape = scale_shape;
      populate_quant_attrs(attr);
      attr.bias = std::move(bias);
      fc_op->attr = std::move(attr);
      return true;
    }
    default:
      return false;
  }
}

void PopulateBlockwiseQuantizedFullyConnected(
    const TfLiteContext& context, const TfLiteTensor& weights_tensor,
    int weights_id, const TfLiteTensor* bias_tensor, int bias_id,
    bool bias_is_const, bool enable_spanned_weights,
    ::ml_drift::FullyConnectedInt4Attributes& attr) {
  const bool copy_weights = !enable_spanned_weights;
  const int output_channels = weights_tensor.dims->data[0];
  const int input_channels = weights_tensor.dims->data[1];
  const int num_elements = output_channels * input_channels;
  const ::ml_drift::OHWI weights_shape(output_channels, 1, 1, input_channels);

  // Weights: OHWI(o, 1, 1, i), int4 unpacked into int8.
  ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT8> weights;
  weights.id = weights_id;
  weights.shape = weights_shape;
  if (copy_weights) {
    weights.data.resize(num_elements + XNN_EXTRA_BYTES);
    ::ml_drift::UnpackDenseInt4IntoInt8(
        reinterpret_cast<const int8_t*>(weights_tensor.data.raw_const),
        num_elements, weights.data.data());
  } else {
    weights.spanned_data =
        absl::MakeSpan(reinterpret_cast<int8_t*>(weights_tensor.data.raw),
                       weights_tensor.bytes);
  }
  attr.weights = std::move(weights);

  const auto* qparams = reinterpret_cast<const TfLiteBlockwiseQuantization*>(
      weights_tensor.quantization.params);

  // Scale is stored in a separate tensor referenced by index.
  const TfLiteTensor& scale = context.tensors[qparams->scale];
  if (copy_weights) {
    if (scale.type == kTfLiteFloat32) {
      attr.scale.data.assign(scale.data.f,
                             scale.data.f + scale.bytes / sizeof(float));
    } else if (scale.type == kTfLiteFloat16) {
      const auto* scale_f16 =
          reinterpret_cast<const TfLiteFloat16*>(scale.data.f16);
      const int scale_size = scale.bytes / sizeof(TfLiteFloat16);
      attr.scale.data.reserve(scale_size);
      for (int s = 0; s < scale_size; ++s) {
        attr.scale.data.push_back(fp16_ieee_to_fp32_value(scale_f16[s].data));
      }
    } else {
      ABSL_LOG(FATAL) << "Unimplemented blockwise scale dtype: " << scale.type;
    }
  } else {
    ABSL_QCHECK_EQ(scale.type, kTfLiteFloat32)
        << "Zero-copy blockwise scale must be float32.";
    attr.scale.spanned_data =
        absl::MakeSpan(scale.data.f, scale.bytes / sizeof(float));
  }
  attr.scale.shape =
      QuantizedFullyConnectedScaleShape(weights_tensor, weights_shape);

  // Zero-point is optional and, when present, stored in a separate tensor.
  if (qparams->zero_point >= 0) {
    const TfLiteTensor& zero_point = context.tensors[qparams->zero_point];
    if (copy_weights) {
      if (zero_point.type == kTfLiteInt32) {
        attr.zero_point.data.assign(
            zero_point.data.i32,
            zero_point.data.i32 + zero_point.bytes / sizeof(int32_t));
      } else if (zero_point.type == kTfLiteInt64) {
        const auto* zp64 =
            reinterpret_cast<const int64_t*>(zero_point.data.i64);
        attr.zero_point.data.assign(zp64,
                                    zp64 + zero_point.bytes / sizeof(int64_t));
      } else {
        ABSL_LOG(FATAL) << "Unimplemented blockwise zero_point dtype: "
                        << zero_point.type;
      }
    } else {
      ABSL_QCHECK_EQ(zero_point.type, kTfLiteInt32)
          << "Zero-copy blockwise zero_point must be int32.";
      attr.zero_point.spanned_data = absl::MakeSpan(
          zero_point.data.i32, zero_point.bytes / sizeof(int32_t));
    }
    attr.zero_point.shape = attr.scale.shape;
  }

  if (bias_is_const) {
    PopulateTensor(bias_tensor, bias_id, &attr.bias,
                   PopulateTensorFlags::kNoExtraBytes, enable_spanned_weights);
  }
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
