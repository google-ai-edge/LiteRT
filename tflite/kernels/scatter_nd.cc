/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/tensor.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/kernels/uint16_asym_wrapper.h"

#include <cmath>
#include <vector>

namespace tflite {
namespace ops {
namespace builtin {
namespace scatter_nd {
constexpr int kIndices = 0;
constexpr int kUpdates = 1;
constexpr int kShape = 2;
constexpr int kOutputTensor = 0;

template <typename IndicesT>
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* shape,
                                TfLiteTensor* output) {
  const int shape_rank = SizeOfDimension(shape, 0);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(shape_rank);
  const auto* shape_data = GetTensorData<IndicesT>(shape);

  for (int i = 0; i < shape_rank; i++) {
    output_shape->data[i] = shape_data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}

template <typename IndicesT>
TfLiteStatus CheckShapes(TfLiteContext* context, const RuntimeShape& indices,
                         const RuntimeShape& updates,
                         const RuntimeShape& shape_shape,
                         const IndicesT* shape_data) {
  TF_LITE_ENSURE(context, (indices.DimensionsCount() >= 1) &&
                              (updates.DimensionsCount() >= 1) &&
                              (shape_shape.DimensionsCount() == 1));

  const int outer_dims = indices.DimensionsCount() - 1;
  for (int i = 0; i < outer_dims; ++i) {
    TF_LITE_ENSURE_EQ(context, indices.Dims(i), updates.Dims(i));
  }

  const int ix = indices.Dims(outer_dims);
  TF_LITE_ENSURE_EQ(context, updates.DimensionsCount() - outer_dims,
                    shape_shape.Dims(0) - ix);
  for (int i = 0; i + outer_dims < updates.DimensionsCount(); ++i) {
    TF_LITE_ENSURE_EQ(context, updates.Dims(i + outer_dims),
                      shape_data[ix + i]);
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kIndices, &indices));
  const TfLiteTensor* updates;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kUpdates, &updates));
  const TfLiteTensor* shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kShape, &shape));

  switch (updates->type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteBool:
    case kTfLiteInt8:
    case kTfLiteInt64:
    case kTfLiteInt32:
    case kTfLiteUInt16:
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Updates of type '%s' are not supported by scatter_nd.",
          TfLiteTypeGetName(updates->type));
      return kTfLiteError;
  }
  if (indices->type != shape->type) {
    TF_LITE_KERNEL_LOG(context, "Indices and shape must have the same type.");
    return kTfLiteError;
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = updates->type;

  if (IsConstantOrPersistentTensor(shape)) {
    switch (indices->type) {
      case kTfLiteInt32:
        TF_LITE_ENSURE_OK(
            context,
            CheckShapes<int32_t>(context, GetTensorShape(indices),
                                 GetTensorShape(updates), GetTensorShape(shape),
                                 GetTensorData<int32_t>(shape)));
        return ResizeOutputTensor<int32_t>(context, shape, output);
      default:
        TF_LITE_KERNEL_LOG(
            context, "Indices of type '%s' are not supported by scatter_nd.",
            TfLiteTypeGetName(indices->type));
        return kTfLiteError;
    }
  } else {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
}

template <typename IndicesT, typename UpdatesT>
TfLiteStatus ScatterNd(const TfLiteTensor* indices, const TfLiteTensor* updates,
                       TfLiteTensor* output) {
  return reference_ops::ScatterNd(
      GetTensorShape(indices), GetTensorData<IndicesT>(indices),
      GetTensorShape(updates), GetTensorData<UpdatesT>(updates),
      GetTensorShape(output), GetTensorData<UpdatesT>(output));
}

// Reference ScatterNd is invalid for asymmetric uint16 (raw-zero != real-zero,
// and += mixes scales). Scatter in float.
template <typename IndicesT>
TfLiteStatus ScatterUInt16Asym(TfLiteContext* context,
                                const TfLiteTensor* indices,
                                const TfLiteTensor* updates,
                                TfLiteTensor* output) {
  std::vector<float> updates_f;
  uint16_asym::DequantizeUInt16(updates, &updates_f);
  const int out_n = GetTensorShape(output).FlatSize();
  std::vector<float> out_f(out_n, 0.0f);

  const RuntimeShape indices_shape = GetTensorShape(indices);
  const RuntimeShape updates_shape = GetTensorShape(updates);
  const RuntimeShape output_shape = GetTensorShape(output);
  const int outer_dims = indices_shape.DimensionsCount() - 1;
  const int indices_nd = indices_shape.Dims(outer_dims);
  int n_slices = 1;
  for (int i = 0; i < outer_dims; ++i) n_slices *= indices_shape.Dims(i);
  int slice_size = 1;
  for (int i = outer_dims; i < updates_shape.DimensionsCount(); ++i) {
    slice_size *= updates_shape.Dims(i);
  }
  std::vector<int> dims_to_count(indices_nd, 0);
  int remain = out_n;
  for (int i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain / output_shape.Dims(i);
    remain = dims_to_count[i];
  }
  const IndicesT* idx_data = GetTensorData<IndicesT>(indices);
  for (int i = 0; i < n_slices; ++i) {
    int to_pos = 0;
    for (int j = 0; j < indices_nd; ++j) {
      IndicesT idx = idx_data[i * indices_nd + j];
      to_pos += static_cast<int>(idx) * dims_to_count[j];
    }
    if (to_pos < 0 || to_pos + slice_size > out_n) return kTfLiteError;
    for (int j = 0; j < slice_size; ++j) {
      out_f[to_pos + j] += updates_f[i * slice_size + j];
    }
  }
  uint16_asym::RequantizeToUInt16(out_f, output);
  return kTfLiteOk;
}

template <typename IndicesT>
TfLiteStatus EvalScatterNd(TfLiteContext* context, const TfLiteTensor* indices,
                           const TfLiteTensor* updates,
                           const TfLiteTensor* shape, TfLiteTensor* output) {
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(
        context, CheckShapes<IndicesT>(
                     context, GetTensorShape(indices), GetTensorShape(updates),
                     GetTensorShape(shape), GetTensorData<IndicesT>(shape)));
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor<IndicesT>(context, shape, output));
  }

  TfLiteStatus status = kTfLiteError;
  switch (updates->type) {
    case kTfLiteFloat32:
      status = ScatterNd<IndicesT, float>(indices, updates, output);
      break;
    case kTfLiteUInt8:
      status = ScatterNd<IndicesT, uint8_t>(indices, updates, output);
      break;
    case kTfLiteBool:
      status = ScatterNd<IndicesT, bool>(indices, updates, output);
      break;
    case kTfLiteInt8:
      status = ScatterNd<IndicesT, int8_t>(indices, updates, output);
      break;
    case kTfLiteInt32:
      status = ScatterNd<IndicesT, int32_t>(indices, updates, output);
      break;
    case kTfLiteInt64:
      status = ScatterNd<IndicesT, int64_t>(indices, updates, output);
      break;
    case kTfLiteUInt16:
      status = ScatterUInt16Asym<IndicesT>(context, indices, updates, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Updates of type '%s' are not supported by scatter_nd.",
          TfLiteTypeGetName(updates->type));
      return kTfLiteError;
  }
  if (status != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "scatter_nd index out of bounds");
  }
  return status;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kIndices, &indices));
  const TfLiteTensor* updates;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kUpdates, &updates));
  const TfLiteTensor* shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kShape, &shape));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (indices->type) {
    case kTfLiteInt32:
      return EvalScatterNd<int32_t>(context, indices, updates, shape, output);
    default:
      TF_LITE_KERNEL_LOG(
          context, "Indices of type '%s' are not supported by scatter_nd.",
          TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }
}

}  // namespace scatter_nd

TfLiteRegistration* Register_SCATTER_ND() {
  static TfLiteRegistration r = {/*init*/ nullptr, /*free*/ nullptr,
                                 scatter_nd::Prepare, scatter_nd::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
