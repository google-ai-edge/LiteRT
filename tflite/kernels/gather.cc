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
#include <stdint.h>

#include <limits>

#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace gather {
constexpr int kInputTensor = 0;
constexpr int kInputPositions = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  // Indicates that 'Eval' is a noop as the output as written during 'Prepare'.
  bool noop;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

bool CheckedDimProduct(const TfLiteTensor* tensor, int begin, int end,
                       int64_t* product) {
  if (tensor == nullptr || tensor->dims == nullptr || product == nullptr ||
      begin < 0 || end < begin || end > tensor->dims->size) {
    return false;
  }
  int64_t count = 1;
  for (int i = begin; i < end; ++i) {
    const int dim = tensor->dims->data[i];
    if (dim < 0) {
      return false;
    }
    if (dim > 0 && count > std::numeric_limits<int>::max() / dim) {
      return false;
    }
    count *= dim;
  }
  *product = count;
  return true;
}

bool CheckedNumElements(const TfLiteTensor* tensor, int64_t* num_elements) {
  if (tensor == nullptr || tensor->dims == nullptr) {
    return false;
  }
  return CheckedDimProduct(tensor, 0, tensor->dims->size, num_elements);
}

bool CheckedGatherComponentSizes(const TfLiteTensor* input,
                                 const TfLiteTensor* positions, int axis,
                                 int batch_dims) {
  int64_t component_size = 0;
  return CheckedDimProduct(input, 0, batch_dims, &component_size) &&
         CheckedDimProduct(input, batch_dims, axis, &component_size) &&
         CheckedDimProduct(input, axis + 1, input->dims->size,
                           &component_size) &&
         CheckedDimProduct(positions, batch_dims, positions->dims->size,
                           &component_size);
}

TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  data->noop = false;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* positions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputPositions, &positions));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (positions->type) {
    case kTfLiteInt64:
    case kTfLiteInt32:
    case kTfLiteInt16:
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Positions of type '%s' are not supported by gather.",
                         TfLiteTypeGetName(positions->type));
      return kTfLiteError;
  }

  // Assign to output the input type.
  output->type = input->type;

  // Check conditions for different types.
  switch (input->type) {
    case kTfLiteFloat32:
    case kTfLiteFloat16:
    case kTfLiteBFloat16:
#if defined(TFLITE_ENABLE_EXTRA_REFERENCE_KERNELS)
    case kTfLiteFloat8E4M3FN:
    case kTfLiteFloat8E5M2:
#endif
    case kTfLiteUInt8:
    case kTfLiteInt4:
    case kTfLiteInt8:
    case kTfLiteInt16:
    case kTfLiteInt64:
    case kTfLiteInt32:
    case kTfLiteBool:
      break;
    case kTfLiteString: {
      // Only 1D input is supported.
      TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  int axis = params->axis;
  TF_LITE_ENSURE(context, axis >= INT16_MIN && axis <= INT16_MAX);
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < NumDimensions(input));

  int batch_dims = params->batch_dims;
  TF_LITE_ENSURE(context, batch_dims >= INT16_MIN && batch_dims <= INT16_MAX);
  // batch_dims should be in range: [-rank(positions), rank(positions)].
  // Negative batch_dims is added with rank of positions.
  if (batch_dims < 0) {
    batch_dims += NumDimensions(positions);
  }
  TF_LITE_ENSURE(context, batch_dims <= axis);
  TF_LITE_ENSURE(context, 0 <= batch_dims && batch_dims < NumDimensions(input));
  TF_LITE_ENSURE(context, batch_dims <= NumDimensions(positions));
  for (int i = 0; i < batch_dims; ++i) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[i], positions->dims->data[i]);
  }

  int64_t input_elements = 0;
  int64_t positions_elements = 0;
  TF_LITE_ENSURE(context, CheckedNumElements(input, &input_elements));
  TF_LITE_ENSURE(context, CheckedNumElements(positions, &positions_elements));
  TF_LITE_ENSURE(
      context,
      CheckedGatherComponentSizes(input, positions, axis, batch_dims));

  const int num_dimensions =
      NumDimensions(input) + NumDimensions(positions) - 1 - batch_dims;
  TF_LITE_ENSURE(context, num_dimensions >= 0);
  int64_t expected_output_elements = 1;
  for (int i = 0; i < axis; ++i) {
    const int dim = input->dims->data[i];
    TF_LITE_ENSURE(context,
                   dim == 0 || expected_output_elements <=
                                   std::numeric_limits<int>::max() / dim);
    expected_output_elements *= dim;
  }
  for (int i = batch_dims; i < positions->dims->size; ++i) {
    const int dim = positions->dims->data[i];
    TF_LITE_ENSURE(context,
                   dim == 0 || expected_output_elements <=
                                   std::numeric_limits<int>::max() / dim);
    expected_output_elements *= dim;
  }
  for (int i = axis + 1; i < input->dims->size; ++i) {
    const int dim = input->dims->data[i];
    TF_LITE_ENSURE(context,
                   dim == 0 || expected_output_elements <=
                                   std::numeric_limits<int>::max() / dim);
    expected_output_elements *= dim;
  }

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  int output_index = 0;
  for (int i = 0; i < axis; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  for (int i = batch_dims; i < positions->dims->size; ++i) {
    output_shape->data[output_index++] = positions->dims->data[i];
  }
  for (int i = axis + 1; i < input->dims->size; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  if (IsConstantOrPersistentTensor(input) &&
      IsConstantOrPersistentTensor(positions)) {
    SetTensorToPersistentRo(output);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output, output_shape));
    TF_LITE_ENSURE_OK(context, EvalImpl(context, node));
    data->noop = true;
    return kTfLiteOk;
  }
  return context->ResizeTensor(context, output, output_shape);
}

template <typename InputT, typename PositionsT>
TfLiteStatus Gather(TfLiteContext* context, const TfLiteGatherParams& params,
                    const TfLiteTensor* input, const TfLiteTensor* positions,
                    TfLiteTensor* output) {
  int64_t input_elements = 0;
  int64_t positions_elements = 0;
  int64_t output_elements = 0;
  TF_LITE_ENSURE(context, CheckedNumElements(input, &input_elements));
  TF_LITE_ENSURE(context, CheckedNumElements(positions, &positions_elements));
  TF_LITE_ENSURE(context, CheckedNumElements(output, &output_elements));
  const size_t num_indices = positions->bytes / sizeof(PositionsT);
  TF_LITE_ENSURE(context,
                 num_indices >= static_cast<size_t>(positions_elements));
  if (input->type == kTfLiteInt4) {
    TF_LITE_ENSURE(context,
                   input->bytes >= (static_cast<size_t>(input_elements) + 1) / 2);
    TF_LITE_ENSURE(
        context,
        output->bytes >= (static_cast<size_t>(output_elements) + 1) / 2);
  } else {
    TF_LITE_ENSURE(context, input->bytes / sizeof(InputT) >=
                                static_cast<size_t>(input_elements));
    TF_LITE_ENSURE(context, output->bytes / sizeof(InputT) >=
                                static_cast<size_t>(output_elements));
  }
  const PositionsT* indexes = GetTensorData<PositionsT>(positions);
  if (positions_elements > 0) {
    TF_LITE_ENSURE(context, indexes != nullptr);
  }
  if (input_elements > 0) {
    TF_LITE_ENSURE(context, GetTensorData<InputT>(input) != nullptr);
  }
  if (output_elements > 0) {
    TF_LITE_ENSURE(context, GetTensorData<InputT>(output) != nullptr);
  }
  bool indices_has_only_positive_elements = true;
  for (int64_t i = 0; i < positions_elements; i++) {
    if (indexes[i] < 0) {
      indices_has_only_positive_elements = false;
      break;
    }
  }
  TF_LITE_ENSURE(context, indices_has_only_positive_elements);

  TF_LITE_ENSURE(context, params.axis >= INT16_MIN && params.axis <= INT16_MAX);
  TF_LITE_ENSURE(context, params.batch_dims >= INT16_MIN &&
                              params.batch_dims <= INT16_MAX);
  tflite::GatherParams op_params;
  op_params.axis = static_cast<int16_t>(params.axis);
  op_params.batch_dims = static_cast<int16_t>(params.batch_dims);
  return optimized_ops::Gather(
      op_params, GetTensorShape(input), GetTensorData<InputT>(input),
      GetTensorShape(positions), GetTensorData<PositionsT>(positions),
      GetTensorShape(output), GetTensorData<InputT>(output),
      (input->type == kTfLiteInt4));
}

template <typename PositionT>
TfLiteStatus GatherStrings(TfLiteContext* context, const TfLiteTensor* input,
                           const TfLiteTensor* positions,
                           TfLiteTensor* output) {
  DynamicBuffer buffer;

  int64_t positions_elements = 0;
  TF_LITE_ENSURE(context, CheckedNumElements(positions, &positions_elements));
  const size_t num_indices = positions->bytes / sizeof(PositionT);
  TF_LITE_ENSURE(
      context, positions_elements >= 0 &&
                   static_cast<uint64_t>(positions_elements) <=
                       static_cast<uint64_t>(num_indices));
  const PositionT* coords_data = GetTensorData<PositionT>(positions);
  TF_LITE_ENSURE(context, input->data.raw != nullptr);
  if (positions_elements > 0) {
    TF_LITE_ENSURE(context, coords_data != nullptr);
  }
  if (output->bytes > 0) {
    TF_LITE_ENSURE(context, output->data.raw != nullptr);
  }

  bool indices_has_only_positive_elements = true;
  for (int64_t i = 0; i < positions_elements; ++i) {
    if (coords_data[i] < 0) {
      indices_has_only_positive_elements = false;
      break;
    }
  }
  TF_LITE_ENSURE(context, indices_has_only_positive_elements);

  TF_LITE_ENSURE(context, input->bytes >= sizeof(int32_t));
  const int num_strings = GetStringCount(input);
  TF_LITE_ENSURE(context, num_strings >= 0);
  TF_LITE_ENSURE(context, input->bytes / sizeof(int32_t) >=
                              static_cast<size_t>(num_strings) + 2);

  for (int64_t i = 0; i < positions_elements; ++i) {
    const PositionT pos = coords_data[i];
    TF_LITE_ENSURE(context, pos < num_strings);
    const int32_t* offsets = reinterpret_cast<const int32_t*>(input->data.raw);
    const int32_t start_offset = offsets[pos + 1];
    const int32_t end_offset = offsets[pos + 2];
    TF_LITE_ENSURE(context, start_offset >= 0);
    TF_LITE_ENSURE(context, end_offset >= start_offset);
    TF_LITE_ENSURE(context, end_offset <= input->bytes);
    const auto string_ref = GetString(input, pos);
    buffer.AddString(string_ref.str, string_ref.len);
  }
  buffer.WriteToTensor(output, /*new_shape=*/nullptr);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* data = reinterpret_cast<OpData*>(node->user_data);
  if (data->noop) {
    return kTfLiteOk;
  } else {
    return EvalImpl(context, node);
  }
}

template <class PosT>
TfLiteStatus DispatchEvalInputType(TfLiteContext* const context,
                                   const TfLiteGatherParams* const params,
                                   const TfLiteTensor* const input,
                                   const TfLiteTensor* const positions,
                                   TfLiteTensor* const output) {
  if (input->type == kTfLiteString) {
    return GatherStrings<PosT>(context, input, positions, output);
  }
  if (input->type == kTfLiteInt4) {
    return Gather<int8_t, PosT>(context, *params, input, positions, output);
  }

  switch (TfLiteTypeGetSizeBits(input->type)) {
    case 8:
      return Gather<int8_t, PosT>(context, *params, input, positions, output);
    case 16:
      return Gather<int16_t, PosT>(context, *params, input, positions, output);
    case 32:
      return Gather<int32_t, PosT>(context, *params, input, positions, output);
    case 64:
      return Gather<int64_t, PosT>(context, *params, input, positions, output);
    default:
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus DispatchEvalPositionType(TfLiteContext* const context,
                                      const TfLiteGatherParams* const params,
                                      const TfLiteTensor* const input,
                                      const TfLiteTensor* const positions,
                                      TfLiteTensor* const output) {
  switch (positions->type) {
    case kTfLiteInt16:
      return DispatchEvalInputType<int16_t>(context, params, input, positions,
                                            output);
    case kTfLiteInt32:
      return DispatchEvalInputType<int32_t>(context, params, input, positions,
                                            output);
    case kTfLiteInt64:
      return DispatchEvalInputType<int64_t>(context, params, input, positions,
                                            output);
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Positions of type '%s' are not supported by gather.",
                         TfLiteTypeGetName(positions->type));
      return kTfLiteError;
  }
}

TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* positions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputPositions, &positions));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);

  int axis = params->axis;
  TF_LITE_ENSURE(context, axis >= INT16_MIN && axis <= INT16_MAX);
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < NumDimensions(input));

  int batch_dims = params->batch_dims;
  TF_LITE_ENSURE(context, batch_dims >= INT16_MIN && batch_dims <= INT16_MAX);
  if (batch_dims < 0) {
    batch_dims += NumDimensions(positions);
  }
  TF_LITE_ENSURE(context, batch_dims <= axis);
  TF_LITE_ENSURE(context, 0 <= batch_dims && batch_dims < NumDimensions(input));
  TF_LITE_ENSURE(context, batch_dims <= NumDimensions(positions));

  int64_t input_elements = 0;
  int64_t positions_elements = 0;
  TF_LITE_ENSURE(context, CheckedNumElements(input, &input_elements));
  TF_LITE_ENSURE(context, CheckedNumElements(positions, &positions_elements));
  TF_LITE_ENSURE(
      context,
      CheckedGatherComponentSizes(input, positions, axis, batch_dims));
  for (int i = 0; i < batch_dims; ++i) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[i], positions->dims->data[i]);
  }

  if (input->type == kTfLiteString) {
    TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
    TF_LITE_ENSURE_EQ(context, axis, 0);
    TF_LITE_ENSURE_EQ(context, batch_dims, 0);
  } else {
    int64_t output_elements = 0;
    TF_LITE_ENSURE(context, CheckedNumElements(output, &output_elements));
    const int expected_output_dims =
        NumDimensions(input) + NumDimensions(positions) - 1 - batch_dims;
    TF_LITE_ENSURE_EQ(context, NumDimensions(output), expected_output_dims);
    int output_index = 0;
    int64_t expected_output_elements = 1;
    for (int i = 0; i < axis; ++i) {
      const int dim = input->dims->data[i];
      TF_LITE_ENSURE_EQ(context, output->dims->data[output_index++], dim);
      TF_LITE_ENSURE(context,
                     dim == 0 || expected_output_elements <=
                                     std::numeric_limits<int>::max() / dim);
      expected_output_elements *= dim;
    }
    for (int i = batch_dims; i < positions->dims->size; ++i) {
      const int dim = positions->dims->data[i];
      TF_LITE_ENSURE_EQ(context, output->dims->data[output_index++], dim);
      TF_LITE_ENSURE(context,
                     dim == 0 || expected_output_elements <=
                                     std::numeric_limits<int>::max() / dim);
      expected_output_elements *= dim;
    }
    for (int i = axis + 1; i < input->dims->size; ++i) {
      const int dim = input->dims->data[i];
      TF_LITE_ENSURE_EQ(context, output->dims->data[output_index++], dim);
      TF_LITE_ENSURE(context,
                     dim == 0 || expected_output_elements <=
                                     std::numeric_limits<int>::max() / dim);
      expected_output_elements *= dim;
    }
    TF_LITE_ENSURE_EQ(context, output_elements, expected_output_elements);
  }

  TfLiteStatus status =
      DispatchEvalPositionType(context, params, input, positions, output);
  if (status != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "gather index out of bounds");
  }
  return status;
}
}  // namespace gather

TfLiteRegistration* Register_GATHER() {
  static TfLiteRegistration r = {gather::Init, gather::Free, gather::Prepare,
                                 gather::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
