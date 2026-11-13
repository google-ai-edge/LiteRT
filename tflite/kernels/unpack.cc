/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdint.h>

#include <limits>

#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/tensor.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unpack {
namespace {

constexpr int kInputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteUnpackParams* data =
      reinterpret_cast<TfLiteUnpackParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), data->num);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  int input_elements;
  TF_LITE_ENSURE_OK(context, CheckedNumElements(input, input_elements));
  TF_LITE_ENSURE(context, input_elements > 0);
  int axis = data->axis;
  TF_LITE_ENSURE(context, axis >= INT16_MIN && axis <= INT16_MAX);
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < NumDimensions(input));
  TF_LITE_ENSURE(context, axis <= INT16_MAX);
  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteFloat16 && input->type != kTfLiteBFloat16 &&
#if defined(TFLITE_ENABLE_EXTRA_REFERENCE_KERNELS)
      input->type != kTfLiteFloat8E4M3FN && input->type != kTfLiteFloat8E5M2 &&
#endif
      input->type != kTfLiteUInt8 && input->type != kTfLiteInt8 &&
      input->type != kTfLiteInt16 && input->type != kTfLiteBool) {
    TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by unpack.",
                       TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  const TfLiteIntArray* input_shape = input->dims;
  // Num should be equal to the shape[axis].
  TF_LITE_ENSURE(context, data->num > 0);
  TF_LITE_ENSURE_EQ(context, data->num, input_shape->data[axis]);
  for (int i = 0; i < data->num; ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);
    // Guarantee input/output quantization params match as we do not support
    // rescaling of unpacked quantized tensors.
    TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                      output->params.zero_point);
    TF_LITE_ENSURE_EQ(context, input->params.scale, output->params.scale);
  }

  // Resize outputs. rank will be R - 1.
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(NumDimensions(input) - 1);
  int o = 0;
  for (int index = 0; index < NumDimensions(input); ++index) {
    if (index != axis) {
      output_shape->data[o++] = input_shape->data[index];
    }
  }

  for (int i = 0; i < data->num; ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    TfLiteIntArray* copied_output_shape = TfLiteIntArrayCopy(output_shape);
    const TfLiteStatus resize_status =
        context->ResizeTensor(context, output, copied_output_shape);
    if (resize_status != kTfLiteOk) {
      TfLiteIntArrayFree(output_shape);
      return resize_status;
    }
  }

  TfLiteIntArrayFree(output_shape);
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus UnpackImpl(TfLiteContext* context, TfLiteNode* node,
                       const TfLiteTensor* input, int output_count, int axis,
                       int input_elements) {
  TF_LITE_ENSURE(context, static_cast<size_t>(input_elements) <=
                              std::numeric_limits<size_t>::max() / sizeof(T));
  const size_t input_bytes = static_cast<size_t>(input_elements) * sizeof(T);
  TF_LITE_ENSURE(context, input->bytes >= input_bytes);
  TF_LITE_ENSURE(context, GetTensorData<T>(input) != nullptr);
  const size_t output_bytes = input_bytes / output_count;
  for (int i = 0; i < output_count; ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    TF_LITE_ENSURE(context, output->bytes >= output_bytes);
    TF_LITE_ENSURE(context, GetTensorData<T>(output) != nullptr);
  }
  tflite::UnpackParams op_params;
  op_params.axis = static_cast<int16_t>(axis);
  op_params.num_split = output_count;
  VectorOfTensors<T> all_outputs(*context, *node->outputs);
  reference_ops::Unpack<T>(op_params, GetTensorShape(input),
                           GetTensorData<T>(input), **all_outputs.shapes(),
                           all_outputs.data());
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteUnpackParams* data =
      reinterpret_cast<TfLiteUnpackParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE(context, data->num > 0);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), data->num);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  int input_elements;
  TF_LITE_ENSURE_OK(context, CheckedNumElements(input, input_elements));
  TF_LITE_ENSURE(context, input_elements > 0);
  int axis = data->axis;
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, axis >= 0 && axis < NumDimensions(input));
  TF_LITE_ENSURE(context, axis <= INT16_MAX);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input, axis), data->num);

  const int expected_output_elements = input_elements / data->num;
  for (int i = 0; i < data->num; ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);
    int output_elements;
    TF_LITE_ENSURE_OK(context, CheckedNumElements(output, output_elements));
    TF_LITE_ENSURE_EQ(context, output_elements, expected_output_elements);
  }
  switch (TfLiteTypeGetSizeBits(input->type)) {
    case 8: {
      TF_LITE_ENSURE_OK(context, UnpackImpl<uint8_t>(context, node, input,
                                                   data->num, axis, input_elements));
      break;
    }
    case 16: {
      TF_LITE_ENSURE_OK(context, UnpackImpl<uint16_t>(context, node, input,
                                                    data->num, axis, input_elements));
      break;
    }
    case 32: {
      TF_LITE_ENSURE_OK(context, UnpackImpl<uint32_t>(context, node, input,
                                                    data->num, axis, input_elements));
      break;
    }
    case 64: {
      TF_LITE_ENSURE_OK(context, UnpackImpl<uint64_t>(context, node, input,
                                                    data->num, axis, input_elements));
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by unpack.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}
}  // namespace
}  // namespace unpack

TfLiteRegistration* Register_UNPACK() {
  static TfLiteRegistration r = {nullptr, nullptr, unpack::Prepare,
                                 unpack::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
