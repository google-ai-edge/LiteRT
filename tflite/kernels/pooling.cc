/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tflite/kernels/internal/optimized/integer_ops/pooling.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/reference/integer_ops/pooling.h"
#include "tflite/kernels/internal/reference/pooling.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/tensor.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/kernels/padding.h"
#include "tflite/kernels/uint16_asym_wrapper.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pooling {

// This file has two implementation of each pooling op.
enum KernelType {
  kReference,
  kGenericOptimized,
};

enum PoolType {
  kAverage,
  kMax,
  kL2,
};

struct OpData {
  TfLitePaddingValues padding;
};

void CopyUInt16ToShiftedInt16(const TfLiteTensor* input,
                              std::vector<int16_t>* shifted) {
  const int flat_size = GetTensorShape(input).FlatSize();
  shifted->resize(flat_size);
  const uint16_t* input_data = GetTensorData<uint16_t>(input);
  for (int i = 0; i < flat_size; ++i) {
    (*shifted)[i] =
        static_cast<int16_t>(static_cast<int32_t>(input_data[i]) - 32768);
  }
}

void CopyShiftedInt16ToUInt16(const std::vector<int16_t>& shifted,
                              TfLiteTensor* output) {
  uint16_t* output_data = GetTensorData<uint16_t>(output);
  for (int i = 0; i < static_cast<int>(shifted.size()); ++i) {
    const int32_t raw = static_cast<int32_t>(shifted[i]) + 32768;
    output_data[i] = static_cast<uint16_t>(
        std::min<int32_t>(65535, std::max<int32_t>(0, raw)));
  }
}

bool AveragePoolUInt16Affine(const TfLitePoolParams* params,
                             const OpData* data,
                             const TfLiteTensor* input,
                             TfLiteTensor* output,
                             int32_t activation_min,
                             int32_t activation_max) {
  // Rebase uint16 to signed-view int16 (raw ^ 0x8000), reuse the same int16
  // AveragePool kernel used for int16, then rebase back. Assumes same-scale
  // (input_scale == output_scale, input_zp == output_zp), matching the
  // int16 kernel which accumulates raw values without rescaling.
  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape output_shape = GetTensorShape(output);
  const int in_size = input_shape.FlatSize();
  const int out_size = output_shape.FlatSize();
  const uint16_t* input_raw = GetTensorData<uint16_t>(input);
  uint16_t* output_raw = GetTensorData<uint16_t>(output);
  std::vector<int16_t> rebased_in(in_size);
  std::vector<int16_t> rebased_out(out_size);
  for (int i = 0; i < in_size; ++i) {
    rebased_in[i] = static_cast<int16_t>(input_raw[i] ^ 0x8000);
  }
  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min - 32768;
  op_params.quantized_activation_max = activation_max - 32768;
  if (!reference_integer_ops::AveragePool(op_params, input_shape,
                                          rebased_in.data(), output_shape,
                                          rebased_out.data())) {
    return false;
  }
  for (int i = 0; i < out_size; ++i) {
    output_raw[i] = static_cast<uint16_t>(rebased_out[i]) ^ 0x8000;
  }
  return true;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

template <PoolType pool_type>
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int out_width, out_height;

  // Prevent division by 0 in optimized pooling implementations
  TF_LITE_ENSURE(context, params->stride_height > 0);
  TF_LITE_ENSURE(context, params->stride_width > 0);

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      params->filter_height, params->filter_width, padding, &out_height,
      &out_width);

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
      input->type == kTfLiteInt16 || input->type == kTfLiteUInt16) {
    if (pool_type == kAverage || pool_type == kMax) {
      TFLITE_DCHECK_LE(std::abs(input->params.scale - output->params.scale),
                       1.0e-6);
      TFLITE_DCHECK_EQ(input->params.zero_point, output->params.zero_point);
    }
    if (pool_type == kL2) {
      // We currently don't have a quantized implementation of L2Pool for
      // int8/uint8/int16; only float32 and uint16 (via dequant->float->requant)
      // are supported.
      TF_LITE_ENSURE(context, input->type == kTfLiteFloat32 ||
                                  input->type == kTfLiteUInt16);
    }
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus AverageEvalFloat(TfLiteContext* context, TfLiteNode* node,
                              TfLitePoolParams* params, OpData* data,
                              const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_AVERAGE_POOL(type)                                            \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.float_activation_min = activation_min;                            \
  op_params.float_activation_max = activation_max;                            \
  TF_LITE_ENSURE(context, type::AveragePool(op_params, GetTensorShape(input), \
                                            GetTensorData<float>(input),      \
                                            GetTensorShape(output),           \
                                            GetTensorData<float>(output)))
  if (kernel_type == kReference) {
    TF_LITE_AVERAGE_POOL(reference_ops);
  } else {
    TF_LITE_AVERAGE_POOL(optimized_ops);
  }
#undef TF_LITE_AVERAGE_POOL
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus AverageEvalQuantizedUint8(TfLiteContext* context, TfLiteNode* node,
                                       TfLitePoolParams* params, OpData* data,
                                       const TfLiteTensor* input,
                                       TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
      context, params->activation, output, &activation_min, &activation_max));
#define TF_LITE_AVERAGE_POOL(type)                                            \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.quantized_activation_min = activation_min;                        \
  op_params.quantized_activation_max = activation_max;                        \
  TF_LITE_ENSURE(context, type::AveragePool(op_params, GetTensorShape(input), \
                                            GetTensorData<uint8_t>(input),    \
                                            GetTensorShape(output),           \
                                            GetTensorData<uint8_t>(output)))
  if (kernel_type == kReference) {
    TF_LITE_AVERAGE_POOL(reference_ops);
  } else {
    TF_LITE_AVERAGE_POOL(optimized_ops);
  }
#undef TF_LITE_AVERAGE_POOL
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus AverageEvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                                      TfLitePoolParams* params, OpData* data,
                                      const TfLiteTensor* input,
                                      TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;

  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);
#define TF_LITE_AVERAGE_POOL(type)                                            \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.quantized_activation_min = activation_min;                        \
  op_params.quantized_activation_max = activation_max;                        \
  TF_LITE_ENSURE(context, type::AveragePool(op_params, GetTensorShape(input), \
                                            GetTensorData<int8_t>(input),     \
                                            GetTensorShape(output),           \
                                            GetTensorData<int8_t>(output)))
  if (kernel_type == kReference) {
    TF_LITE_AVERAGE_POOL(reference_integer_ops);
  } else {
    TF_LITE_AVERAGE_POOL(optimized_integer_ops);
  }
#undef TF_LITE_AVERAGE_POOL
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus AverageEvalQuantizedInt16(TfLiteContext* context, TfLiteNode* node,
                                       TfLitePoolParams* params, OpData* data,
                                       const TfLiteTensor* input,
                                       TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  CalculateActivationRangeQuantized(context, params->activation, output,
                                    &activation_min, &activation_max);
  if (input->type == kTfLiteUInt16) {
    TF_LITE_ENSURE(context, AveragePoolUInt16Affine(
                                params, data, input, output, activation_min,
                                activation_max));
    return kTfLiteOk;
  }
#define TF_LITE_AVERAGE_POOL(type)                                            \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.quantized_activation_min = activation_min;                        \
  op_params.quantized_activation_max = activation_max;                        \
  TF_LITE_ENSURE(context, type::AveragePool(op_params, GetTensorShape(input), \
                                            GetTensorData<int16_t>(input),    \
                                            GetTensorShape(output),           \
                                            GetTensorData<int16_t>(output)))
  TF_LITE_AVERAGE_POOL(reference_integer_ops);
#undef TF_LITE_AVERAGE_POOL
  return kTfLiteOk;
}

template <KernelType kernel_type>
void MaxEvalFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLitePoolParams* params, OpData* data,
                  const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_MAX_POOL(type)                                                 \
  tflite::PoolParams op_params;                                                \
  op_params.stride_height = params->stride_height;                             \
  op_params.stride_width = params->stride_width;                               \
  op_params.filter_height = params->filter_height;                             \
  op_params.filter_width = params->filter_width;                               \
  op_params.padding_values.height = data->padding.height;                      \
  op_params.padding_values.width = data->padding.width;                        \
  op_params.float_activation_min = activation_min;                             \
  op_params.float_activation_max = activation_max;                             \
  type::MaxPool(op_params, GetTensorShape(input), GetTensorData<float>(input), \
                GetTensorShape(output), GetTensorData<float>(output))
  if (kernel_type == kReference) {
    TF_LITE_MAX_POOL(reference_ops);
  } else {
    TF_LITE_MAX_POOL(optimized_ops);
  }
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void MaxEvalQuantizedUInt8(TfLiteContext* context, TfLiteNode* node,
                           TfLitePoolParams* params, OpData* data,
                           const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);
#define TF_LITE_MAX_POOL(type)                                         \
  tflite::PoolParams op_params;                                        \
  op_params.stride_height = params->stride_height;                     \
  op_params.stride_width = params->stride_width;                       \
  op_params.filter_height = params->filter_height;                     \
  op_params.filter_width = params->filter_width;                       \
  op_params.padding_values.height = data->padding.height;              \
  op_params.padding_values.width = data->padding.width;                \
  op_params.quantized_activation_min = activation_min;                 \
  op_params.quantized_activation_max = activation_max;                 \
  type::MaxPool(op_params, GetTensorShape(input),                      \
                GetTensorData<uint8_t>(input), GetTensorShape(output), \
                GetTensorData<uint8_t>(output))
  if (kernel_type == kReference) {
    TF_LITE_MAX_POOL(reference_ops);
  } else {
    TF_LITE_MAX_POOL(optimized_ops);
  }
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void MaxEvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                          TfLitePoolParams* params, OpData* data,
                          const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);
#define TF_LITE_MAX_POOL(type)                                        \
  tflite::PoolParams op_params;                                       \
  op_params.stride_height = params->stride_height;                    \
  op_params.stride_width = params->stride_width;                      \
  op_params.filter_height = params->filter_height;                    \
  op_params.filter_width = params->filter_width;                      \
  op_params.padding_values.height = data->padding.height;             \
  op_params.padding_values.width = data->padding.width;               \
  op_params.quantized_activation_min = activation_min;                \
  op_params.quantized_activation_max = activation_max;                \
  type::MaxPool(op_params, GetTensorShape(input),                     \
                GetTensorData<int8_t>(input), GetTensorShape(output), \
                GetTensorData<int8_t>(output))
  if (kernel_type == kReference) {
    TF_LITE_MAX_POOL(reference_integer_ops);
  } else {
    TF_LITE_MAX_POOL(optimized_integer_ops);
  }
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void MaxEvalQuantizedInt16(TfLiteContext* context, TfLiteNode* node,
                           TfLitePoolParams* params, OpData* data,
                           const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  CalculateActivationRangeQuantized(context, params->activation, output,
                                    &activation_min, &activation_max);
  if (input->type == kTfLiteUInt16) {
    std::vector<int16_t> shifted_input;
    std::vector<int16_t> shifted_output(GetTensorShape(output).FlatSize());
    CopyUInt16ToShiftedInt16(input, &shifted_input);
    tflite::PoolParams op_params;
    op_params.stride_height = params->stride_height;
    op_params.stride_width = params->stride_width;
    op_params.filter_height = params->filter_height;
    op_params.filter_width = params->filter_width;
    op_params.padding_values.height = data->padding.height;
    op_params.padding_values.width = data->padding.width;
    op_params.quantized_activation_min = activation_min - 32768;
    op_params.quantized_activation_max = activation_max - 32768;
    reference_integer_ops::MaxPool(
        op_params, GetTensorShape(input), shifted_input.data(),
        GetTensorShape(output), shifted_output.data());
    CopyShiftedInt16ToUInt16(shifted_output, output);
    return;
  }
#define TF_LITE_MAX_POOL(type)                                         \
  tflite::PoolParams op_params;                                        \
  op_params.stride_height = params->stride_height;                     \
  op_params.stride_width = params->stride_width;                       \
  op_params.filter_height = params->filter_height;                     \
  op_params.filter_width = params->filter_width;                       \
  op_params.padding_values.height = data->padding.height;              \
  op_params.padding_values.width = data->padding.width;                \
  op_params.quantized_activation_min = activation_min;                 \
  op_params.quantized_activation_max = activation_max;                 \
  type::MaxPool(op_params, GetTensorShape(input),                      \
                GetTensorData<int16_t>(input), GetTensorShape(output), \
                GetTensorData<int16_t>(output))
  TF_LITE_MAX_POOL(reference_integer_ops);
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void L2EvalFloat(TfLiteContext* context, TfLiteNode* node,
                 TfLitePoolParams* params, OpData* data,
                 const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_L2_POOL(type)                                                 \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.float_activation_min = activation_min;                            \
  op_params.float_activation_max = activation_max;                            \
  type::L2Pool(op_params, GetTensorShape(input), GetTensorData<float>(input), \
               GetTensorShape(output), GetTensorData<float>(output))
  if (kernel_type == kReference) {
    TF_LITE_L2_POOL(reference_ops);
  } else {
    TF_LITE_L2_POOL(optimized_ops);
  }
#undef TF_LITE_L2_POOL
}

#undef TF_LITE_KERNEL_TYPE_DISPATCH

template <KernelType kernel_type>
TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return AverageEvalFloat<kernel_type>(context, node, params, data, input,
                                           output);
    case kTfLiteUInt8:
      return AverageEvalQuantizedUint8<kernel_type>(context, node, params, data,
                                                    input, output);
    case kTfLiteInt8:
      return AverageEvalQuantizedInt8<kernel_type>(context, node, params, data,
                                                   input, output);
    case kTfLiteInt16:  // fallthrough
    case kTfLiteUInt16:  // uint16 shares int16's 16-bit average kernel path
      return AverageEvalQuantizedInt16<kernel_type>(context, node, params, data,
                                                    input, output);
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      MaxEvalFloat<kernel_type>(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
      MaxEvalQuantizedUInt8<kernel_type>(context, node, params, data, input,
                                         output);
      break;
    case kTfLiteInt8:
      MaxEvalQuantizedInt8<kernel_type>(context, node, params, data, input,
                                        output);
      break;
    case kTfLiteInt16:  // fallthrough
    case kTfLiteUInt16:  // uint16 shares int16's 16-bit max-pool kernel path
      MaxEvalQuantizedInt16<kernel_type>(context, node, params, data, input,
                                         output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus L2Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      L2EvalFloat<kernel_type>(context, node, params, data, input, output);
      break;
    case kTfLiteUInt16: {
      // Dequantize -> float L2 pool -> requantize.
      std::vector<float> in_f;
      uint16_asym::DequantizeUInt16(input, &in_f);
      std::vector<float> out_f(GetTensorShape(output).FlatSize());
      float activation_min, activation_max;
      CalculateActivationRange(params->activation, &activation_min,
                               &activation_max);
      tflite::PoolParams op_params;
      op_params.stride_height = params->stride_height;
      op_params.stride_width = params->stride_width;
      op_params.filter_height = params->filter_height;
      op_params.filter_width = params->filter_width;
      op_params.padding_values.height = data->padding.height;
      op_params.padding_values.width = data->padding.width;
      op_params.float_activation_min = activation_min;
      op_params.float_activation_max = activation_max;
      reference_ops::L2Pool(op_params, GetTensorShape(input), in_f.data(),
                            GetTensorShape(output), out_f.data());
      uint16_asym::RequantizeToUInt16(out_f, output);
      break;
    }
    case kTfLiteUInt8:
    // We don't have a quantized implementation, so just fall through to the
    // 'default' case.
    default:
      TF_LITE_KERNEL_LOG(context, "Type %d not currently supported.",
                         input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace pooling

TfLiteRegistration* Register_AVERAGE_POOL_REF() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kAverage>,
                                 pooling::AverageEval<pooling::kReference>};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_REF() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kMax>,
                                 pooling::MaxEval<pooling::kReference>};
  return &r;
}

TfLiteRegistration* Register_L2_POOL_REF() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kL2>,
                                 pooling::L2Eval<pooling::kReference>};
  return &r;
}

TfLiteRegistration* Register_AVERAGE_POOL_GENERIC_OPT() {
  static TfLiteRegistration r = {
      pooling::Init, pooling::Free, pooling::GenericPrepare<pooling::kAverage>,
      pooling::AverageEval<pooling::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_GENERIC_OPT() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kMax>,
                                 pooling::MaxEval<pooling::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_L2_POOL_GENERIC_OPT() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kL2>,
                                 pooling::L2Eval<pooling::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_AVERAGE_POOL_2D() {
  return Register_AVERAGE_POOL_GENERIC_OPT();
}

TfLiteRegistration* Register_MAX_POOL_2D() {
  return Register_MAX_POOL_GENERIC_OPT();
}

TfLiteRegistration* Register_L2_POOL_2D() {
  return Register_L2_POOL_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
