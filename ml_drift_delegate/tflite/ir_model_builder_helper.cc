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

#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/kernels/internal/portable_tensor_utils.h"
#include "tflite/kernels/kernel_util.h"

// Forward declaration to avoid full dependency on pthreadpool
struct pthreadpool;
typedef struct pthreadpool* pthreadpool_t;

extern "C" {
// ml_drift:internal-replace-begin(EAP version is pointing to the old XNNPACK)
enum xnn_status xnn_run_unary_elementwise_nc(
    // create parameters
    enum xnn_unary_operator op_type, enum xnn_datatype input_datatype,
    enum xnn_datatype output_datatype, const union xnn_unary_params* params,
    const struct xnn_quantization_params* input_quantization,
    const struct xnn_quantization_params* output_quantization, uint32_t flags,
    // reshape parameters
    size_t batch_size, size_t channels, size_t input_stride,
    size_t output_stride, pthreadpool_t threadpool,
    // setup parameters
    const void* input, void* output);
/* ml_drift:external-replace-with
enum xnn_status xnn_run_convert_nc_f16_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const void* input,
    float* output,
    uint32_t flags,
    pthreadpool_t threadpool);
*/
}  // extern "C"

namespace litert::ml_drift::ir {

::ml_drift::Axis GetConcatAxis(
    const std::vector<::ml_drift::BHWDC>& input_shapes,
    const ::ml_drift::BHWDC& output_shape) {
  if (input_shapes[0].h != output_shape.h) {
    return ::ml_drift::Axis::HEIGHT;
  }
  if (input_shapes[0].w != output_shape.w) {
    return ::ml_drift::Axis::WIDTH;
  }
  if (input_shapes[0].d != output_shape.d) {
    return ::ml_drift::Axis::DEPTH;
  }
  if (input_shapes[0].c != output_shape.c) {
    return ::ml_drift::Axis::CHANNELS;
  }
  return ::ml_drift::Axis::BATCH;
}

void ResolveNegativeIndices(const TfLiteIntArray& input_dims,
                            std::vector<int>& tensor) {
  for (int i = 0; i < input_dims.size; ++i) {
    tensor[i] = ResolveNegativeIndex(tensor[i], input_dims.data[i]);
  }
}

void UpdateWithMask(int begin_mask, int end_mask,
                    const TfLiteIntArray& input_dims, std::vector<int>& starts,
                    std::vector<int>& ends) {
  for (int i = 0; i < input_dims.size; ++i) {
    if ((begin_mask >> i) & 1) starts[i] = 0;
    if ((end_mask >> i) & 1) ends[i] = input_dims.data[i];
  }
}

::ml_drift::BHWDC ExtractTensorShape(const TfLiteIntArray* dims) {
  const int size = dims->size;
  if (size == 0) {
    return ::ml_drift::BHWDC(1, 1, 1, 1, 1);
  } else if (size == 1) {
    return ::ml_drift::BHWDC(dims->data[0], 1, 1, 1, 1);
  } else if (size == 2) {
    return ::ml_drift::BHWDC(dims->data[0], 1, 1, 1, dims->data[1]);
  } else if (size == 3) {
    return ::ml_drift::BHWDC(dims->data[0], 1, dims->data[1], 1, dims->data[2]);
  } else if (size == 4) {
    return ::ml_drift::BHWDC(dims->data[0], dims->data[1], dims->data[2], 1,
                             dims->data[3]);
  } else {
    return ::ml_drift::BHWDC(dims->data[0], dims->data[1], dims->data[2],
                             dims->data[3], dims->data[4]);
  }
}

// Scan dimensions from right to left and return false if there is a mismatch
// and the mismatch isn't 1.
bool IsBroadcastable(const TfLiteIntArray* dims1, const TfLiteIntArray* dims2) {
  int idx1 = dims1->size - 1;
  int idx2 = dims2->size - 1;
  for (int i = std::max(idx1, idx2); i >= 0; --i) {
    const int dim1 = idx1 < 0 ? 1 : dims1->data[idx1];
    const int dim2 = idx2 < 0 ? 1 : dims2->data[idx2];
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) return false;
    --idx1;
    --idx2;
  }
  return true;
}

void ConvertFloat16ToFloat32(size_t num_elements, const uint16_t* src,
                             float* dst) {
  // TODO(b/490171548): Move this to weights manager.
  // ml_drift:internal-replace-begin(EAP version is pointing to the old XNNPACK)
  xnn_run_unary_elementwise_nc(
      xnn_unary_convert, xnn_datatype_fp16, xnn_datatype_fp32,
      /*params=*/nullptr, /*input_quantization=*/nullptr,
      /*output_quantization=*/nullptr,
      /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*batch_size=*/num_elements,
      /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1,
      /*threadpool=*/nullptr, /*input=*/src, /*output=*/dst);
  /* ml_drift:external-replace-with
  xnn_run_convert_nc_f16_f32(
      1, 1, 1, num_elements, src, dst,
      XNN_FLAG_DONT_SPIN_WORKERS, nullptr);
  */
}

void CopyFloat32Data(const TfLiteTensor* tfl_tensor, float* dst) {
  const TfLiteType dtype = tfl_tensor->type;
  if (dtype == kTfLiteFloat32) {
    std::memcpy(dst, tfl_tensor->data.f, tfl_tensor->bytes);
  } else if (dtype == kTfLiteFloat16) {
    ConvertFloat16ToFloat32(
        tflite::NumElements(tfl_tensor),
        reinterpret_cast<uint16_t const*>(tfl_tensor->data.f16), dst);
  } else if (dtype == kTfLiteInt4) {
    // Unpack the int4 data into int8 data and then dequantize it.
    // The temporary `bytes_unpacked` may have one more byte if the
    // number of elements is odd but the dequantized `dst` will have the
    // correct number of elements by DequantizeConstantTensor().
    const size_t bytes_unpacked = tfl_tensor->bytes * 2;
    auto unpacked_input_data = std::make_unique<int8_t[]>(bytes_unpacked);
    tflite::tensor_utils::UnpackPackedIntToInt8(tfl_tensor->data.int8,
                                                bytes_unpacked, /*bit_width=*/4,
                                                unpacked_input_data.get());
    const int8_t* input_data = unpacked_input_data.get();
    DequantizeConstantTensor(*tfl_tensor, input_data, dst);
  } else if (dtype == kTfLiteInt8) {
    DequantizeConstantTensor(*tfl_tensor, tfl_tensor->data.int8, dst);
  } else if (dtype == kTfLiteUInt8) {
    DequantizeConstantTensor(*tfl_tensor, tfl_tensor->data.uint8, dst);
  } else if (dtype == kTfLiteInt32) {
    DequantizeConstantTensor(*tfl_tensor, tfl_tensor->data.i32, dst);
  }
}

void PopulateQuantParams(const TfLiteTensor& tensor,
                         ::ml_drift::ir::IrQuantParams* quant_params) {
  const TfLiteQuantization& quant = tensor.quantization;
  ABSL_QCHECK_EQ(quant.type, TfLiteQuantizationType::kTfLiteAffineQuantization);
  const TfLiteAffineQuantization* params =
      static_cast<const TfLiteAffineQuantization*>(quant.params);
  ABSL_QCHECK_EQ(params->scale->size, 1);
  const float scale = params->scale->data[0];
  const float zero_point = static_cast<float>(params->zero_point->data[0]);

  float qmin_value = 0;
  float qmax_value = 0;
  if (tensor.type == kTfLiteUInt8) {
    qmin_value = static_cast<float>(std::numeric_limits<uint8_t>::min());
    qmax_value = static_cast<float>(std::numeric_limits<uint8_t>::max());
  } else if (tensor.type == kTfLiteInt8) {
    qmin_value = static_cast<float>(std::numeric_limits<int8_t>::min());
    qmax_value = static_cast<float>(std::numeric_limits<int8_t>::max());
  } else if (tensor.type == kTfLiteUInt4) {
    qmin_value = 0.0f;
    qmax_value = 15.0f;
  } else if (tensor.type == kTfLiteInt4) {
    qmin_value = -8.0f;
    qmax_value = 7.0f;
  } else if (tensor.type == kTfLiteInt2) {
    qmin_value = -2.0f;
    qmax_value = 1.0f;
  } else {
    ABSL_LOG(FATAL) << absl::StrCat("Type invalid for quantized tensor: ",
                                    tensor.name ? tensor.name : "unknown");
  }
  quant_params->min = scale * (static_cast<float>(qmin_value) - zero_point);
  quant_params->max = scale * (static_cast<float>(qmax_value) - zero_point);
  quant_params->scale = scale;
}

bool IsLinearConvertible(const TfLiteIntArray* dims) {
  if (dims->size <= 0) return false;
  for (int i = 0; i < dims->size - 1; ++i) {
    if (dims->data[i] != 1) return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
