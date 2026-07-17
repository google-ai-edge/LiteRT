// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "third_party/FP16/include/fp16/fp16.h"
#include "ml_drift/common/kernels/conv_apple_mpp.h"  // from @ml_drift
#include "ml_drift/common/kernels/conv_generic.h"  // from @ml_drift
#include "ml_drift/common/kernels/conv_wave_matrix.h"  // from @ml_drift
#include "ml_drift/common/kernels/conv_wave_matrix_mali.h"  // from @ml_drift
#include "ml_drift/common/kernels/conv_wave_memory.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#define F_OK 0
#else  // defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif  // defined(_WIN32)

#if defined(__APPLE__)
#include <Availability.h>
#endif

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
// clang-format off
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
// clang-format on
#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/portable_tensor_utils.h"
#include "tflite/kernels/kernel_util.h"

namespace ml_drift {
namespace {

using ::litert::ml_drift::ReleaseDataCallback;
using ::litert::ml_drift::UnownedDataTensorDescriptor;

void UnpackInt4IntoInt8(const int8_t* src, const size_t src_size,
                        ml_drift::Tensor<OHWI, DataType::INT8>* dst) {
  constexpr int uint4_bits = 4;
  constexpr int kUint4ValuesPerUint8Value =
      std::numeric_limits<uint8_t>::digits / uint4_bits;
  const size_t bytes_unpacked = src_size * kUint4ValuesPerUint8Value;
  auto unpacked_input_data = std::make_unique<int8_t[]>(bytes_unpacked);
  tflite::tensor_utils::UnpackPackedIntToInt8(
      src, bytes_unpacked, /*bit_width=*/4, dst->data.data());
}

void UnpackInt2ToInt8(const int8_t* src, const size_t src_size,
                      ml_drift::Tensor<OHWI, DataType::INT8>* dst) {
  constexpr int uint2_bits = 2;
  constexpr int kUint2ValuesPerUint8Value =
      std::numeric_limits<uint8_t>::digits / uint2_bits;
  const size_t elements_unpacked = src_size * kUint2ValuesPerUint8Value;
  tflite::tensor_utils::UnpackPackedIntToInt8(
      src, elements_unpacked, /*bit_width=*/2, dst->data.data());
}

void MadviseTensor(TfLiteTensor& tensor) {
// Do not enable madvise on Web or Windows because it has not been tested.
#if !defined(__EMSCRIPTEN__)
  void* original_ptr = tensor.data.data;
  size_t space = tensor.bytes;
  ::ml_drift::MadviseData(original_ptr, space);
#endif  // !__EMSCRIPTEN__
}

// Checks the input shapes and gpu info if the convolution int8 is supported.
bool IsConvInt8Supported(const CreateGpuModelInfo& create_info,
                         const GpuInfo& gpu_info, const GraphFloat32& graph,
                         const Node* fc_node) {
  if (create_info.hints.Check(ModelHints::kDisallow8bitConvs)) {
    return false;
  }

  Value* src_tensor = graph.FindInputs(fc_node->id)[0];
  // TODO(sulemanshahid): Add support for int4 src quantization.
  bool supports_conv_apple_mpp = SupportsConvAppleMPP(gpu_info);

  return supports_conv_apple_mpp ||
         SupportsConvWaveMatrixMaliInt8(gpu_info, src_tensor->tensor.shape) ||
         SupportsConvWaveMemoryInt8(gpu_info) ||
         SupportsConvGenericInt8(gpu_info) ||
         SupportsConvWaveMatrixInt8(gpu_info,
                                    OHWI(32, 1, 1, src_tensor->tensor.shape.c));
}

// Check if the convolution int8 kernel is recommended for the computation of
// fully connected op.
bool IsConvInt8KernelRecommendedForFullyConnectedOp(
    const CreateGpuModelInfo& create_info, const GpuInfo& gpu_info,
    const GraphFloat32& graph, const Node* fc_node) {
  Value* src_tensor = graph.FindInputs(fc_node->id)[0];
  const int total_spatial_size = src_tensor->tensor.shape.b *
                                 src_tensor->tensor.shape.h *
                                 src_tensor->tensor.shape.w;

  // In case of spatial size is small, 1x1 FullyConnected kernel is more
  // performant than Convolution kernel.
  return total_spatial_size >
         GetRecommendedMaxTotalSpatialSize(gpu_info, create_info.precision);
}

// We read weight scales as fp16, so if our hardware flushes denormals to zero
// we need to rewrite the weight scales to be the smallest normal value if
// they are denormal in f16.
// If this is OpenCL, check if this is required. For all other APIs, we
// will always rewrite the scales when in FP16.
void RewriteDenormalScales(const GpuInfo& gpu_info,
                           const CreateGpuModelInfo& create_info, float* scales,
                           int size) {
  if (scales == nullptr || size <= 0) {
    return;
  }
  if ((!gpu_info.IsApiOpenCl() ||
       gpu_info.opencl_info.is_fp16_ftz_hardware_forced) &&
      create_info.precision != CalculationsPrecision::F32) {
    const float min_normal_fp16 = 6.1035e-5f;
    for (int i = 0; i < size; ++i) {
      const float val = scales[i];
      // Check if magnitude is too small for fp16, BUT is not exactly zero
      if (std::abs(val) < min_normal_fp16 && val != 0.0f) {
        // Apply the sign of the original 'val' to our minimum normal
        // threshold
        scales[i] = std::copysign(min_normal_fp16, val);
      }
    }
  }
}

}  // namespace

void MadviseData(void* ptr, size_t space) {
// Do not enable madvise on Web because it has not been tested.
#if defined(__EMSCRIPTEN__)
  return;
#endif  // defined(__EMSCRIPTEN__)

#if defined(_WIN32)
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  DWORD alignment = sysInfo.dwPageSize;
  // If the space is less than 1MB, skip madvise because the madvise call
  // has a small latency overhead. We must balance lower memory usage with
  // faster loading.
  // TODO: b/510899567 - This threshold should be re-evaluated on Windows.
  size_t one_megabyte = 1024 * 1024;
  if (space < one_megabyte) {
    return;
  }
#else   // defined(_WIN32)
  size_t alignment = getpagesize();
#endif  // defined(_WIN32)

  size_t minimum_space = alignment;
  void* aligned_ptr = std::align(alignment, minimum_space, ptr, space);
  if (aligned_ptr) {
    // Reduce the space to the nearest alignment to avoid swapping out
    // memory that is still in use.
    space = space / alignment * alignment;
    // Tell the kernel that we do not expect to access this memory in the
    // near future. This allows the kernel to free resources associated
    // with this memory. This is NOT the same as unmapping the memory.
    // If we do attempt to access this later, the memory will be
    // repopulated from the original file on disk which would add latency.
#if defined(_WIN32)
    // The Windows equivalent is VirtualUnlock.
    VirtualUnlock(ptr, space);
#else   // defined(_WIN32)
    // WARNING: MADV_DONTNEED will zero out anonymous memory on Linux.
    // Ensure that the memory passed here is file-backed (e.g., mmaped model
    // file) and not needed again by CPU, otherwise it might cause silent
    // corruption or crashes.
    madvise(aligned_ptr, space,
            MADV_DONTNEED);  // NOLINT because sys/mman.h is already included.
#endif  // defined(_WIN32)
  }
}

TensorDescriptor SharedMemoryManager::GetInt8TensorDesc(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    const OHWI& shape, const int8_t* data, bool is_weight_sum_i_required,
    Tensor<Linear, DataType::INT32>* weights_sum_i) {
  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = shape;
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  weights_i8.data.assign(data, data + weights_i8.shape.DimensionsProduct());

  if (is_weight_sum_i_required && weights_sum_i != nullptr) {
    *weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  }

  WeightsDescription weights_desc = GetFullyConnectedInt8WeightsDesc(
      gpu_info, shape,
      create_info.hints.Check(ModelHints::kPreferTextureWeights));

  return GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);
}

TensorDescriptor SharedMemoryManager::GetInt4TensorDesc(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    const OHWI& shape, const int8_t* data, size_t bytes,
    bool is_weight_sum_i_required,
    Tensor<Linear, DataType::INT32>* weights_sum_i,
    bool experimental_int4_unpacking) {
  if (experimental_int4_unpacking) {
    auto status =
        GetInt4TensorDescExperimental(gpu_info, create_info, shape, data, bytes,
                                      is_weight_sum_i_required, weights_sum_i);
    // If the experimental int4 unpacking is not successful, fall back to the
    // default int4 unpacking. This can happen if the weights are not in a
    // supported layout.
    if (status.ok()) {
      return status.value();
    }
  }
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = shape;
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());

  UnpackInt4IntoInt8(data, bytes, &weights_i8);

  if (is_weight_sum_i_required && weights_sum_i != nullptr) {
    *weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  }

  WeightsDescription weights_desc = GetFullyConnectedInt4WeightsDesc(
      gpu_info, shape,
      create_info.hints.Check(ModelHints::kPreferTextureWeights));

  return GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);
}

TensorDescriptor SharedMemoryManager::GetInt2TensorDesc(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    const OHWI& shape, const int8_t* data, size_t bytes,
    bool is_weight_sum_i_required,
    Tensor<Linear, DataType::INT32>* weights_sum_i,
    bool experimental_int2_unpacking) {
  if (experimental_int2_unpacking) {
    auto status =
        GetInt2TensorDescExperimental(gpu_info, create_info, shape, data, bytes,
                                      is_weight_sum_i_required, weights_sum_i);
    // If the experimental int2 unpacking is not successful, fall back to the
    // default int2 unpacking. This can happen if the weights are not in a
    // supported layout.
    if (status.ok()) {
      return status.value();
    }
  }
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = shape;
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());

  UnpackInt2ToInt8(data, bytes, &weights_i8);

  if (is_weight_sum_i_required && weights_sum_i != nullptr) {
    *weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  }

  WeightsDescription weights_desc = GetFullyConnectedInt2WeightsDesc(
      gpu_info, shape,
      create_info.hints.Check(ModelHints::kPreferTextureWeights));

  return ml_drift::GetTensorDescriptorForWeightsLayout(weights_i8,
                                                       weights_desc);
}

absl::StatusOr<TensorDescriptor>
SharedMemoryManager::GetInt4TensorDescExperimental(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    const OHWI& shape, const int8_t* data, size_t bytes,
    bool is_weight_sum_i_required,
    Tensor<Linear, DataType::INT32>* weights_sum_i) {
  ml_drift::Tensor<OHWI, DataType::UINT8> weights_u8;
  weights_u8.shape = shape;
  // Remove const qualifier because ml_drift::Tensor requires the spanned_data
  // to be mutable. However, this function does not actually modify the data.
  weights_u8.spanned_data = absl::MakeSpan(
      reinterpret_cast<uint8_t*>(const_cast<int8_t*>(data)), bytes);

  WeightsDescription weights_desc = GetFullyConnectedInt4WeightsDesc(
      gpu_info, shape,
      create_info.hints.Check(ModelHints::kPreferTextureWeights));

  int total_elements_count =
      GetTotalElementsCountForLayout(weights_desc, shape);
  int int4_per_element = 2;
  std::vector<uint8_t> rearranged_weights(total_elements_count /
                                          int4_per_element);

  weights_sum_i->shape = Linear(shape.o);
  weights_sum_i->data.resize(shape.o);

  ABSL_RETURN_IF_ERROR(RearrangeWeightsUInt4Packed(
      weights_u8, weights_desc, absl::MakeSpan(rearranged_weights),
      absl::MakeSpan(weights_sum_i->data),
      /*pad_value=*/8u, /*swap_dims=*/false));

  TensorDescriptor weights_u8_td;
  if (weights_desc.IsLinearLayout()) {
    weights_u8_td = TensorDescriptor(weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_u8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, rearranged_weights.size()));
  } else {
    weights_u8_td = TensorDescriptor(DataType::UINT16,
                                     TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(weights_desc, shape);
    constexpr int uint4_bits = 4;
    constexpr int kUint4ElementsPerUint16Texel =
        std::numeric_limits<uint16_t>::digits / uint4_bits;
    tex_size.x /= kUint4ElementsPerUint16Texel;
    weights_u8_td.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  }
  weights_u8_td.SetData(std::move(rearranged_weights));
  return weights_u8_td;
}

absl::StatusOr<TensorDescriptor>
SharedMemoryManager::GetInt2TensorDescExperimental(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    const OHWI& shape, const int8_t* data, size_t bytes,
    bool is_weight_sum_i_required,
    Tensor<Linear, DataType::INT32>* weights_sum_i) {
  ml_drift::Tensor<OHWI, DataType::UINT8> weights_u8;
  weights_u8.shape = shape;
  // Remove const qualifier because ml_drift::Tensor requires the spanned_data
  // to be mutable. However, this function does not actually modify the data.
  weights_u8.spanned_data = absl::MakeSpan(
      reinterpret_cast<uint8_t*>(const_cast<int8_t*>(data)), bytes);

  WeightsDescription weights_desc = GetFullyConnectedInt2WeightsDesc(
      gpu_info, shape,
      create_info.hints.Check(ModelHints::kPreferTextureWeights));

  int total_elements_count =
      GetTotalElementsCountForLayout(weights_desc, shape);
  int int2_per_element = 4;
  std::vector<uint8_t> rearranged_weights(total_elements_count /
                                          int2_per_element);

  weights_sum_i->shape = Linear(shape.o);
  weights_sum_i->data.resize(shape.o);

  ABSL_RETURN_IF_ERROR(RearrangeWeightsUInt2Packed(
      weights_u8, weights_desc, absl::MakeSpan(rearranged_weights),
      absl::MakeSpan(weights_sum_i->data),
      /*pad_value=*/2u, /*swap_dims=*/false));

  TensorDescriptor weights_u8_td;
  if (weights_desc.IsLinearLayout()) {
    weights_u8_td = TensorDescriptor(weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_u8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, rearranged_weights.size()));
  } else {
    weights_u8_td = TensorDescriptor(DataType::UINT8,
                                     TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(weights_desc, shape);
    constexpr int uint2_bits = 2;
    constexpr int kUint2ElementsPerUint8Texel =
        std::numeric_limits<uint16_t>::digits / uint2_bits;
    tex_size.x /= kUint2ElementsPerUint8Texel;
    weights_u8_td.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  }
  weights_u8_td.SetData(std::move(rearranged_weights));
  return weights_u8_td;
}

absl::Status SharedMemoryManager::CalculateWeightsSumI(
    const TfLiteTensor& tflite_tensor, Value* shared_const_tensor,
    Tensor<Linear, DataType::INT32>* weights_sum_i) {
  BHWC& bhwc_shape = shared_const_tensor->tensor.shape;
  OHWI shape(bhwc_shape.b, bhwc_shape.h, bhwc_shape.w, bhwc_shape.c);
  const int8_t* data = tflite_tensor.data.int8;

  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = shape;
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());

  if (tflite_tensor.type == TfLiteType::kTfLiteInt8) {
    weights_i8.data.assign(data, data + weights_i8.data.size());
  } else if (tflite_tensor.type == TfLiteType::kTfLiteInt4) {
    size_t bytes = tflite_tensor.bytes;
    constexpr int kUint4Bits = 4;
    constexpr int kUint4ValuesPerUint8Value =
        std::numeric_limits<uint8_t>::digits / kUint4Bits;
    const size_t bytes_unpacked = bytes * kUint4ValuesPerUint8Value;
    weights_i8.data.resize(bytes_unpacked);
    tflite::tensor_utils::UnpackPackedIntToInt8(
        data, bytes_unpacked, /*bit_width=*/4, weights_i8.data.data());
  } else {
    return absl::InternalError(
        "Only quantized int8 and int4 FC weights are supported for src "
        "quantization.");
  }

  if (weights_sum_i != nullptr) {
    *weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  } else {
    return absl::InternalError("weights_sum_i should not be nullptr.");
  }

  return absl::OkStatus();
}

SharedMemoryManager::SharedMemoryManager(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    GraphFloat32& graph, CreateTensorFunc create_tensor_func,
    TfLiteContext* context,
    ValueIdToSharedTensorMap& buffer_id_to_spatial_tensor,
    ValueIdToSharedTensorMap& quant_param_id_to_spatial_tensor,
    bool has_prepacked_tflite_tensors,
    SerializationWeightCache* serialization_cache,
    bool madvise_original_tensors, bool experimental_int4_unpacking,
    bool experimental_int2_unpacking,
    CreateTensorFromDeviceBufferFunc create_tensor_from_device_buffer_func,
    MaybeBindTensorDataFunc maybe_bind_tensor_data_func,
    PackingLookupFunc packing_lookup_func,
    MaybeGetExternalBufferIdFunc maybe_get_external_buffer_id_func,
    DiscardTensorDataFunc discard_tensor_data_func)
    : gpu_info_(gpu_info),
      create_info_(create_info),
      graph_(graph),
      create_tensor_func_(create_tensor_func),
      context_(context),
      buffer_id_to_spatial_tensor_(buffer_id_to_spatial_tensor),
      quant_param_id_to_spatial_tensor_(quant_param_id_to_spatial_tensor),
      has_prepacked_tflite_tensors_(has_prepacked_tflite_tensors),
      serialization_cache_(serialization_cache),
      madvise_original_tensors_(madvise_original_tensors),
      experimental_int4_unpacking_(experimental_int4_unpacking),
      experimental_int2_unpacking_(experimental_int2_unpacking),
      maybe_bind_tensor_data_func_(std::move(maybe_bind_tensor_data_func)),
      packing_lookup_func_(std::move(packing_lookup_func)),
      create_tensor_from_device_buffer_func_(
          std::move(create_tensor_from_device_buffer_func)),
      maybe_get_external_buffer_id_func_(
          std::move(maybe_get_external_buffer_id_func)),
      discard_tensor_data_func_(std::move(discard_tensor_data_func)) {
  next_const_tensor_id_ = quant_param_id_to_spatial_tensor_.size();
  data_type_ = DeduceDataTypeFromPrecision(create_info_.precision);
}

absl::Status SharedMemoryManager::TryRestoringSerializedTensor(
    const uint32_t global_tensor_id, SharedConstTensor& shared_tensor) {
  if (!serialization_cache_) {
    return absl::FailedPreconditionError("Serialization is not supported.");
  }

  size_t page_adjusted_offset = 0;
  ReleaseDataCallback release_data_callback;
#if defined(__APPLE__)
  // TODO: b/421171145 - Evaluate non-Apple platforms to see if this is a
  // memory or init time improvement. At the very least, Web will need to
  // continue using the old path because it does not support mmaping.
  UnownedDataTensorDescriptor tensor_desc;
  ABSL_RETURN_IF_ERROR(serialization_cache_->LookUp(
      global_tensor_id, /*is_quantization_param_tensor=*/false, tensor_desc,
      page_adjusted_offset, release_data_callback));
#else
  TensorDescriptor tensor_desc;
  ABSL_RETURN_IF_ERROR(serialization_cache_->LookUp(
      global_tensor_id, /*is_quantization_param_tensor=*/false, tensor_desc));
#endif

  // If the tensor was prepacked and serialized previously, restore it from
  // the serialized data.
  return create_tensor_func_(tensor_desc, page_adjusted_offset,
                             std::move(release_data_callback),
                             shared_tensor.weights);
}

absl::Status SharedMemoryManager::TryStoringSerializedTensor(
    const uint32_t global_tensor_id, const TensorDescriptor& tensor_desc) {
  // If serialization is enabled, store the prepacked tensor descriptor.
  if (serialization_cache_ && serialization_cache_->IsReadyForInsert()) {
    ABSL_RETURN_IF_ERROR(serialization_cache_->Insert(
        global_tensor_id,
        /*is_quantization_param_tensor=*/false, tensor_desc));
  }
  return absl::OkStatus();
}

absl::Status SharedMemoryManager::CreateQuantizedInt8WeightsTensor(
    const ValueId& shared_tensor_id, const uint32_t global_tensor_id,
    const TfLiteTensor& tflite_tensor, SharedConstTensor& shared_tensor,
    bool is_weight_sum_i_required,
    Tensor<Linear, DataType::INT32>* weights_sum_i) {
  Value* shared_const_tensor = graph_.GetValue(shared_tensor_id);
  shared_const_tensor->tensor.type = DataType::UINT8;
  BHWC& bhwc_shape = shared_const_tensor->tensor.shape;
  OHWI shape(bhwc_shape.b, bhwc_shape.h, bhwc_shape.w, bhwc_shape.c);

  uint32_t external_buffer_id = MaybeGetExternalBufferId(shared_tensor_id);
  if (!weights_manager_ && IsValidExternalBufferId(external_buffer_id)) {
    return absl::FailedPreconditionError(
        "Externalized weights are not supported without weights manager.");
  }

  WeightsDescription weights_desc = GetFullyConnectedInt8WeightsDesc(
      gpu_info_, shape,
      create_info_.hints.Check(ModelHints::kPreferTextureWeights));

  // ML Drift's optimal performance requires the weights to be arranged in a
  // certain layout, which is typically different from TFL flatbuffer's weight
  // layouts.
  // If the packed weight tensor was serialized previously, restore from it
  // directly.
  bool serialized_tensor_found =
      TryRestoringSerializedTensor(global_tensor_id, shared_tensor).ok();
  if (!serialized_tensor_found) {
    if (has_prepacked_tflite_tensors_) {
      // If the tflite tensor is prepacked, restore from it directly.
      ABSL_ASSIGN_OR_RETURN(shared_tensor.weights,
                            CreatePrepackedWeightsTensorFromTfliteTensor(
                                weights_desc, shape, tflite_tensor));
    } else {
      // If weights manager is set, we will utilize Gpu to pack (rearrange) the
      // weights; otherwise, we will pack the weights on CPU.
      if (weights_manager_) {
        uint32_t external_buffer_id =
            MaybeGetExternalBufferId(shared_tensor_id);
        weight_id_to_external_buffer_id_[shared_tensor_id] = external_buffer_id;
        weights_manager_->RegisterWeightsConversion(
            {shared_tensor_id}, weights_desc, shape, DataType::INT8,
            tflite_tensor.data.int8, {}, {});

        auto weights_i8_tensor_desc =
            GetTensorDescriptorsForWeightsLayout(shape, weights_desc)[0];
        shared_const_tensor->tensor.shape =
            weights_i8_tensor_desc.GetBHWCShape();
        shared_const_tensor->tensor.type = weights_i8_tensor_desc.GetDataType();
        return absl::OkStatus();
      }
      TensorDescriptor weights_i8_td = GetInt8TensorDesc(
          gpu_info_, create_info_, shape, tflite_tensor.data.int8,
          is_weight_sum_i_required, weights_sum_i);
      ABSL_RETURN_IF_ERROR(create_tensor_func_(
          weights_i8_td, /*page_adjusted_offset=*/0,
          /*release_data_callback=*/nullptr, shared_tensor.weights));

      // If serialization is enabled, store the serialized tensor descriptor.
      ABSL_RETURN_IF_ERROR(
          TryStoringSerializedTensor(global_tensor_id, weights_i8_td));
    }
  }

  shared_const_tensor->tensor.shape =
      shared_tensor.weights->GetDescriptor().GetBHWCShape();
  shared_const_tensor->tensor.type =
      shared_tensor.weights->GetDescriptor().GetDataType();
  return absl::OkStatus();
}

absl::Status SharedMemoryManager::CreateQuantizedInt4WeightsTensor(
    const ValueId& shared_tensor_id, const uint32_t global_tensor_id,
    const TfLiteTensor& tflite_tensor, SharedConstTensor& shared_tensor,
    bool is_weight_sum_i_required,
    Tensor<Linear, DataType::INT32>* weights_sum_i) {
  Value* shared_const_tensor = graph_.GetValue(shared_tensor_id);
  BHWC& bhwc_shape = shared_const_tensor->tensor.shape;
  OHWI shape(bhwc_shape.b, bhwc_shape.h, bhwc_shape.w, bhwc_shape.c);

  uint32_t external_buffer_id = MaybeGetExternalBufferId(shared_tensor_id);
  if (!weights_manager_ && IsValidExternalBufferId(external_buffer_id)) {
    return absl::FailedPreconditionError(
        "Externalized weights are not supported without weights manager.");
  }

  WeightsDescription weights_desc = GetFullyConnectedInt4WeightsDesc(
      gpu_info_, shape,
      create_info_.hints.Check(ModelHints::kPreferTextureWeights));

  // If the tensor was serialized previously, restore from it directly.
  bool serialized_tensor_found =
      TryRestoringSerializedTensor(global_tensor_id, shared_tensor).ok();
  if (!serialized_tensor_found) {
    if (has_prepacked_tflite_tensors_) {
      // If the tflite tensor is prepacked, restore from it directly.
      ABSL_ASSIGN_OR_RETURN(shared_tensor.weights,
                            CreatePrepackedWeightsTensorFromTfliteTensor(
                                weights_desc, shape, tflite_tensor));
    } else {
      // If weights manager is set, we will utilize Gpu to pack (rearrange) the
      // weights; otherwise, we will pack the weights on CPU.
      // TODO(linchan): Support Int4 weights conversion on Gpu for unaligned
      // shapes.
      if (weights_manager_ && shape.i % 4 == 0) {
        uint32_t external_buffer_id =
            MaybeGetExternalBufferId(shared_tensor_id);
        weight_id_to_external_buffer_id_[shared_tensor_id] = external_buffer_id;
        weights_manager_->RegisterWeightsConversion(
            {shared_tensor_id}, weights_desc, shape, DataType::INT4,
            tflite_tensor.data.int8, {}, {});

        auto weights_i4_tensor_desc =
            GetTensorDescriptorsForWeightsLayout(shape, weights_desc)[0];
        shared_const_tensor->tensor.shape =
            weights_i4_tensor_desc.GetBHWCShape();
        shared_const_tensor->tensor.type = weights_i4_tensor_desc.GetDataType();
        return absl::OkStatus();
      }
      // If the tensor is not prepacked, prepack it now.
      TensorDescriptor weights_i4_td = GetInt4TensorDesc(
          gpu_info_, create_info_, shape, tflite_tensor.data.int8,
          tflite_tensor.bytes, is_weight_sum_i_required, weights_sum_i,
          experimental_int4_unpacking_);
      ABSL_RETURN_IF_ERROR(create_tensor_func_(
          weights_i4_td, /*page_adjusted_offset=*/0,
          /*release_data_callback=*/nullptr, shared_tensor.weights));

      // If serialization is enabled, store the serialized tensor descriptor.
      ABSL_RETURN_IF_ERROR(
          TryStoringSerializedTensor(global_tensor_id, weights_i4_td));
    }
  }

  shared_const_tensor->tensor.shape =
      shared_tensor.GetWeights()->GetDescriptor().GetBHWCShape();
  shared_const_tensor->tensor.type =
      shared_tensor.GetWeights()->GetDescriptor().GetDataType();
  return absl::OkStatus();
}

absl::Status SharedMemoryManager::CreateQuantizedInt2WeightsTensor(
    const ValueId& shared_tensor_id, const uint32_t global_tensor_id,
    const TfLiteTensor& tflite_tensor, SharedConstTensor& shared_tensor,
    bool is_weight_sum_i_required,
    Tensor<Linear, DataType::INT32>* weights_sum_i) {
  Value* shared_const_tensor = graph_.GetValue(shared_tensor_id);
  BHWC& bhwc_shape = shared_const_tensor->tensor.shape;
  OHWI shape(bhwc_shape.b, bhwc_shape.h, bhwc_shape.w, bhwc_shape.c);

  uint32_t external_buffer_id = MaybeGetExternalBufferId(shared_tensor_id);
  if (!weights_manager_ && IsValidExternalBufferId(external_buffer_id)) {
    return absl::FailedPreconditionError(
        "Externalized weights are not supported without weights manager.");
  }

  WeightsDescription weights_desc = GetFullyConnectedInt2WeightsDesc(
      gpu_info_, shape,
      create_info_.hints.Check(ModelHints::kPreferTextureWeights));

  // If the tensor was serialized previously, restore from it directly.
  bool serialized_tensor_found =
      TryRestoringSerializedTensor(global_tensor_id, shared_tensor).ok();
  if (!serialized_tensor_found) {
    if (has_prepacked_tflite_tensors_) {
      // If the tflite tensor is prepacked, restore from it directly.
      ABSL_ASSIGN_OR_RETURN(shared_tensor.weights,
                            CreatePrepackedWeightsTensorFromTfliteTensor(
                                weights_desc, shape, tflite_tensor));
    } else {
      // If the tensor is not prepacked, prepack it now.
      if (weights_manager_) {
        uint32_t external_buffer_id =
            MaybeGetExternalBufferId(shared_tensor_id);
        weight_id_to_external_buffer_id_[shared_tensor_id] = external_buffer_id;
        weights_manager_->RegisterWeightsConversion(
            {shared_tensor_id}, weights_desc, shape, DataType::INT2,
            tflite_tensor.data.int8, {}, {});

        auto weights_i2_tensor_desc =
            GetTensorDescriptorsForWeightsLayout(shape, weights_desc)[0];
        shared_const_tensor->tensor.shape =
            weights_i2_tensor_desc.GetBHWCShape();
        shared_const_tensor->tensor.type = weights_i2_tensor_desc.GetDataType();
        return absl::OkStatus();
      }
      TensorDescriptor weights_i2_td = GetInt2TensorDesc(
          gpu_info_, create_info_, shape, tflite_tensor.data.int8,
          tflite_tensor.bytes, is_weight_sum_i_required, weights_sum_i,
          experimental_int2_unpacking_);
      ABSL_RETURN_IF_ERROR(create_tensor_func_(
          weights_i2_td, /*page_adjusted_offset=*/0,
          /*release_data_callback=*/nullptr, shared_tensor.weights));

      // If serialization is enabled, store the serialized tensor descriptor.
      ABSL_RETURN_IF_ERROR(
          TryStoringSerializedTensor(global_tensor_id, weights_i2_td));
    }
  }

  shared_const_tensor->tensor.shape =
      shared_tensor.GetWeights()->GetDescriptor().GetBHWCShape();
  shared_const_tensor->tensor.type =
      shared_tensor.GetWeights()->GetDescriptor().GetDataType();
  return absl::OkStatus();
}

absl::Status SharedMemoryManager::CreateAffineQuantizationParams(
    const ValueId& shared_tensor_id, uint32_t global_tensor_id,
    const TfLiteTensor& tflite_tensor,
    absl::flat_hash_map<ValueId, GlobalId>* external_tensors,
    bool is_weight_sum_i_required, Node* fc_node, DataType data_type,
    const OHWI& weights_shape) {
  TfLiteAffineQuantization* quant_params =
      static_cast<TfLiteAffineQuantization*>(tflite_tensor.quantization.params);
  auto* shared_tensor = &buffer_id_to_spatial_tensor_.at(global_tensor_id);

  // Copy scales and zero points.
  std::vector<float> scales(
      quant_params->scale->data,
      quant_params->scale->data + quant_params->scale->size);
  std::vector<int> zero_points(
      quant_params->zero_point->data,
      quant_params->zero_point->data + quant_params->zero_point->size);
  if (zero_points.size() != scales.size()) {
    zero_points.resize(scales.size(), zero_points[0]);
  }

  Linear scale_shape(scales.size());
  RewriteDenormalScales(gpu_info_, create_info_, scales.data(), scales.size());

  shared_tensor->scale_global_tensor_id = ++next_const_tensor_id_;
  ABSL_ASSIGN_OR_RETURN(
      ValueId scale_value_id,
      AddInputWithData(shared_tensor->scale_global_tensor_id.value(),
                       scale_shape, *fc_node, scales.data(), data_type));
  // We refresh the shared_tensor pointer as the object could have been moved
  // after AddInputWithData() call added  a scale tensor to the
  // buffer_id_to_spatial_tensor_.
  shared_tensor = &buffer_id_to_spatial_tensor_.at(global_tensor_id);
  (*external_tensors)[scale_value_id] =
      GlobalId::BuildParamId(shared_tensor->scale_global_tensor_id.value());

  shared_tensor->zero_point_global_tensor_id = ++next_const_tensor_id_;
  Linear zero_point_shape(zero_points.size());
  ABSL_ASSIGN_OR_RETURN(
      ValueId zero_point_value_id,
      AddInputWithData(shared_tensor->zero_point_global_tensor_id.value(),
                       zero_point_shape, *fc_node, zero_points.data(),
                       data_type));
  shared_tensor = &buffer_id_to_spatial_tensor_.at(global_tensor_id);
  (*external_tensors)[zero_point_value_id] = GlobalId::BuildParamId(
      shared_tensor->zero_point_global_tensor_id.value());

  // Checks serialized cache for weights_sum_i if not already populated.
  if (is_weight_sum_i_required) {
    shared_tensor->weights_sum_i_global_tensor_id = ++next_const_tensor_id_;
    ValueId weights_sum_i_value_id;

    TensorDescriptor serialized_tensor_desc;
    bool serialized_tensor_found =
        serialization_cache_ &&
        serialization_cache_
            ->LookUp(shared_tensor->weights_sum_i_global_tensor_id.value(),
                     /*is_quantization_param_tensor=*/true,
                     serialized_tensor_desc)
            .ok();

    if (serialized_tensor_found) {
      ABSL_ASSIGN_OR_RETURN(
          weights_sum_i_value_id,
          AddInputWithData<int32_t>(
              shared_tensor->weights_sum_i_global_tensor_id.value(),
              scale_shape, *fc_node, nullptr, DataType::INT32));
    } else if (weights_manager_) {
      BHWC scale_or_zero_point_shape = BHWC(
          1, 1, 1, tflite_tensor.dims->data[quant_params->quantized_dimension]);
      ABSL_ASSIGN_OR_RETURN(
          weights_sum_i_value_id,
          AddInputNode(shared_tensor->weights_sum_i_global_tensor_id.value(),
                       scale_or_zero_point_shape, *fc_node, DataType::INT32));
      DataType src_data_type = DataType::INT4;
      if (tflite_tensor.type == TfLiteType::kTfLiteInt8) {
        src_data_type = DataType::INT8;
      } else if (tflite_tensor.type == TfLiteType::kTfLiteInt4) {
        src_data_type = DataType::INT4;
      } else if (tflite_tensor.type == TfLiteType::kTfLiteInt2) {
        src_data_type = DataType::INT2;
      }
      weights_manager_->RegisterWeightsSumIConversion(
          {weights_sum_i_value_id}, weights_shape, src_data_type,
          tflite_tensor.data.int8);
    } else {
      auto* weights_sum_i_data = shared_tensor->weights_sum_i.data.empty()
                                     ? nullptr
                                     : shared_tensor->weights_sum_i.data.data();
      absl::StatusOr<ValueId> status_or_value_id = AddInputWithData(
          shared_tensor->weights_sum_i_global_tensor_id.value(), scale_shape,
          *fc_node, weights_sum_i_data, DataType::INT32);

      // If the weights_sum_i is empty and there is no entry in the cache, we
      // will get an error when trying to add the input. In this case, we
      // should just skip adding the input.
      if (!status_or_value_id.ok()) {
        if (shared_tensor->weights_sum_i.data.empty()) {
          return absl::OkStatus();
        } else {
          return status_or_value_id.status();
        }
      }
      weights_sum_i_value_id = status_or_value_id.value();
    }
    shared_tensor = &buffer_id_to_spatial_tensor_.at(global_tensor_id);
    (*external_tensors)[weights_sum_i_value_id] = GlobalId::BuildParamId(
        shared_tensor->weights_sum_i_global_tensor_id.value());
  }
  return absl::OkStatus();
}

absl::Status SharedMemoryManager::CreateBlockwiseQuantizationParams(
    const ValueId& shared_tensor_id, uint32_t global_tensor_id,
    const TfLiteTensor& tflite_tensor,
    absl::flat_hash_map<ValueId, GlobalId>* external_tensors, Node* fc_node,
    DataType data_type) {
  TfLiteBlockwiseQuantization* quant_params =
      static_cast<TfLiteBlockwiseQuantization*>(
          tflite_tensor.quantization.params);
  if (quant_params == nullptr) {
    return absl::InternalError("Blockwise quantization params are null.");
  }
  if (quant_params->scale < 0 ||
      quant_params->scale >= context_->tensors_size) {
    return absl::InternalError("Invalid blockwise scale tensor index.");
  }

  auto* shared_tensor = &buffer_id_to_spatial_tensor_.at(global_tensor_id);
  int num_channels = tflite_tensor.dims->data[0];
  int num_blocks = tflite_tensor.dims->data[1] / quant_params->blocksize;
  size_t scale_size = num_channels * num_blocks;

  // Prepare scale data.
  float* scale_data = nullptr;
  TfLiteTensor* scale_tensor = &context_->tensors[quant_params->scale];
  std::vector<float> converted_scales;
  if (scale_tensor->type == kTfLiteFloat32) {
    scale_data = scale_tensor->data.f;
  } else if (scale_tensor->type == kTfLiteFloat16) {
    const auto* scale_f16 =
        reinterpret_cast<TfLiteFloat16*>(scale_tensor->data.f16);
    converted_scales.resize(scale_size);
    for (int i = 0; i < scale_size; ++i) {
      converted_scales[i] = fp16_ieee_to_fp32_value(scale_f16[i].data);
    }
    scale_data = converted_scales.data();
  } else {
    return absl::InternalError(
        absl::StrCat("Unimplemented scale dtype: ", scale_tensor->type));
  }
  RewriteDenormalScales(gpu_info_, create_info_, scale_data, scale_size);

  // Prepare zero point data.
  std::vector<int> zero_points;
  const int* zero_point_data = nullptr;
  if (quant_params->zero_point < context_->tensors_size &&
      quant_params->zero_point >= 0) {
    TfLiteTensor* zero_point_tensor =
        &context_->tensors[quant_params->zero_point];
    if (zero_point_tensor->type != kTfLiteInt32) {
      return absl::InternalError(absl::StrCat(
          "Unimplemented zero point dtype: ", zero_point_tensor->type,
          ". Only Int32 is supported."));
    }
    const int32_t* zp_data = zero_point_tensor->data.i32;
    if (zp_data == nullptr) {
      return absl::InternalError("Zero point tensor data is null.");
    }
    zero_points.resize(scale_size);
    int zp_size = zero_point_tensor->bytes / sizeof(int32_t);
    if (zp_size == 1) {
      std::fill(zero_points.begin(), zero_points.end(), zp_data[0]);
    } else if (zp_size == scale_size) {
      std::copy(zp_data, zp_data + scale_size, zero_points.begin());
    } else {
      return absl::InternalError(
          absl::StrCat("Invalid zero point tensor size: ", zp_size,
                       ", expected 1 or ", scale_size));
    }
    zero_point_data = zero_points.data();
  } else {
    // Currently, the underlying call stacks always require zero point for
    // quantized weights.
    zero_points.assign(scale_size, 0);
    zero_point_data = zero_points.data();
  }

  shared_tensor->scale_global_tensor_id = ++next_const_tensor_id_;
  ABSL_ASSIGN_OR_RETURN(
      ValueId scale_value_id,
      AddScaleNodeWithData(shared_tensor->scale_global_tensor_id.value(),
                           *fc_node, scale_data, data_type, num_channels,
                           num_blocks));
  shared_tensor = &buffer_id_to_spatial_tensor_.at(global_tensor_id);
  (*external_tensors)[scale_value_id] =
      GlobalId::BuildParamId(shared_tensor->scale_global_tensor_id.value());

  shared_tensor->zero_point_global_tensor_id = ++next_const_tensor_id_;
  ABSL_ASSIGN_OR_RETURN(
      ValueId zero_point_value_id,
      AddScaleNodeWithData(shared_tensor->zero_point_global_tensor_id.value(),
                           *fc_node, zero_point_data, data_type, num_channels,
                           num_blocks));
  shared_tensor = &buffer_id_to_spatial_tensor_.at(global_tensor_id);
  (*external_tensors)[zero_point_value_id] = GlobalId::BuildParamId(
      shared_tensor->zero_point_global_tensor_id.value());

  return absl::OkStatus();
}

absl::Status SharedMemoryManager::CreateQuantizedTensorWithScaleAndZeroPoint(
    const ValueId& shared_tensor_id, uint32_t global_tensor_id,
    const TfLiteTensor& tflite_tensor,
    absl::flat_hash_map<ValueId, GlobalId>* external_tensors) {
  if (tflite_tensor.type != TfLiteType::kTfLiteInt8 &&
      tflite_tensor.type != TfLiteType::kTfLiteInt4 &&
      tflite_tensor.type != TfLiteType::kTfLiteInt2) {
    return absl::InternalError(
        "Only quantized int8, int4, and int2 weights are supported for "
        "sharing.");
  }

  Value* shared_const_tensor = graph_.GetValue(shared_tensor_id);
  OHWI weights_shape(
      shared_const_tensor->tensor.shape.b, shared_const_tensor->tensor.shape.h,
      shared_const_tensor->tensor.shape.w, shared_const_tensor->tensor.shape.c);
  std::vector<Node*> weight_consumers = graph_.FindConsumers(shared_tensor_id);
  if (weight_consumers.size() != 1) {
    return absl::InternalError(
        "Expected to have only one shared weight consumer.");
  }
  Node* fc_node = weight_consumers[0];
  DataType data_type = data_type_;
  if (data_type_ == DataType::FLOAT32) {
    if (!graph_.FindInputs(fc_node->id).empty()) {
      DataType input_type = graph_.FindInputs(fc_node->id)[0]->tensor.type;
      if (IsFloatType(input_type)) {
        data_type = input_type;
      }
    }
  }
  auto* shared_tensor = &buffer_id_to_spatial_tensor_.at(global_tensor_id);

  bool is_weight_sum_i_required = false;
  uint32_t external_buffer_id = MaybeGetExternalBufferId(shared_tensor_id);
  bool is_streamed = IsValidExternalBufferId(external_buffer_id);
  // When weights are streamed, they have null data pointers
  // (tflite_tensor.data.int8 is null). Bypassing weights_sum_i calculation is
  // required to avoid dereferencing null data during initialization.
  if (!is_streamed) {
    if (fc_node->operation.type ==
            ToString(OperationType::FULLY_CONNECTED_INT8) ||
        fc_node->operation.type ==
            ToString(OperationType::FULLY_CONNECTED_INT4)) {
      const bool is_conv_int8_supported =
          IsConvInt8Supported(create_info_, gpu_info_, graph_, fc_node) &&
        tflite_tensor.quantization.type ==
            TfLiteQuantizationType::kTfLiteAffineQuantization;

      // In case of preparing weights (repacking and computing intermediate
      // constants like weights_sum_i) on Gpu, we always compute weights_sum_i
      // for quantized FullyConnected ops if weights_sum_i is supported.
      //
      // This improves performance for preparing weights on Gpu, because it puts
      // weight-repacking and weights_sum_i computations for LLM's shared
      // weights into one single InferenceContext:
      // * The Gpu memory allocation is reduced, because in the same
      //   InferenceContext, the SRC weights on Gpu can be reused between
      //   weight-repacking and weights_sum_i computations.
      // * The latency is reduced, because (1) saves time to allocate Gpu memory
      //   for the identical SRC weights in different InferenceContexts; (2)
      //   eliminates the latency to build and execute the second
      //   InferenceContext for LLMs' weights preparation, as the first
      //   InferenceContext for weights preparation already covers all the
      //   weights preparation requests.
      const bool is_conv_int8_recommended =
          IsConvInt8KernelRecommendedForFullyConnectedOp(
              create_info_, gpu_info_, graph_, fc_node) ||
          weights_manager_;

      is_weight_sum_i_required =
          is_conv_int8_supported && is_conv_int8_recommended;
    }
  }

  if (tflite_tensor.type == TfLiteType::kTfLiteInt8) {
    ABSL_RETURN_IF_ERROR(CreateQuantizedInt8WeightsTensor(
        shared_tensor_id, global_tensor_id, tflite_tensor, *shared_tensor,
        is_weight_sum_i_required, &shared_tensor->weights_sum_i));
  } else if (tflite_tensor.type == TfLiteType::kTfLiteInt4) {
    ABSL_RETURN_IF_ERROR(CreateQuantizedInt4WeightsTensor(
        shared_tensor_id, global_tensor_id, tflite_tensor, *shared_tensor,
        is_weight_sum_i_required, &shared_tensor->weights_sum_i));
  } else if (tflite_tensor.type == TfLiteType::kTfLiteInt2) {
    ABSL_RETURN_IF_ERROR(CreateQuantizedInt2WeightsTensor(
        shared_tensor_id, global_tensor_id, tflite_tensor, *shared_tensor,
        is_weight_sum_i_required, &shared_tensor->weights_sum_i));
  }

  if (tflite_tensor.quantization.type ==
      TfLiteQuantizationType::kTfLiteAffineQuantization) {
    return CreateAffineQuantizationParams(
        shared_tensor_id, global_tensor_id, tflite_tensor, external_tensors,
        is_weight_sum_i_required, fc_node, data_type, weights_shape);
  } else if (tflite_tensor.quantization.type ==
             TfLiteQuantizationType::kTfLiteBlockwiseQuantization) {
    return CreateBlockwiseQuantizationParams(shared_tensor_id, global_tensor_id,
                                             tflite_tensor, external_tensors,
                                             fc_node, data_type);
  } else {
    return absl::UnimplementedError(
        "Only affine quantization is supported for the shared tensors.");
  }
}

template <typename InputDataType>
absl::StatusOr<ValueId> SharedMemoryManager::AddInputWithData(
    uint32_t global_tensor_id, const Linear& shape, const Node& consumer_node,
    const InputDataType* data, DataType data_type) {
  auto [tensor_it, tensor_inserted] =
      quant_param_id_to_spatial_tensor_.try_emplace(global_tensor_id);
  if (!tensor_inserted) {
    return absl::InternalError("Could not insert an input tensor.");
  }

  ABSL_ASSIGN_OR_RETURN(ValueId local_value_id,
                        AddInputNode(global_tensor_id, BHWC(1, 1, 1, shape.v),
                                     consumer_node, data_type));

  if (data == nullptr) {
    TensorDescriptor serialized_tensor_desc;
    bool serialized_tensor_found =
        serialization_cache_ &&
        serialization_cache_
            ->LookUp(global_tensor_id,
                     /*is_quantization_param_tensor=*/true,
                     serialized_tensor_desc)
            .ok();
    if (serialized_tensor_found) {
      // If the tensor was prepacked and serialized previously, restore it from
      // the serialized data.
      ABSL_RETURN_IF_ERROR(create_tensor_func_(
          serialized_tensor_desc, /*page_adjusted_offset=*/0,
          /*release_data_callback=*/nullptr, tensor_it->second.weights));
      return local_value_id;
    } else {
      return absl::InvalidArgumentError(
          "Tensor was not found in the cache and the data pointer is null so "
          "there is nothing to upload.");
    }
  }

  TensorDescriptor tensor_desc =
      TensorDescriptor(data_type, create_info_.storage_type, Layout::LINEAR);
  tensor_desc.SetBHWCShape(graph_.GetValue(local_value_id)->tensor.shape);
  ABSL_RETURN_IF_ERROR(tensor_desc.UpdateToSupportedStorageType(
      gpu_info_, tensor_desc.GetBHWCShape()));

  tensor_desc.UploadData(data);
  ABSL_RETURN_IF_ERROR(create_tensor_func_(
      tensor_desc, /*page_adjusted_offset=*/0,
      /*release_data_callback=*/nullptr, tensor_it->second.weights));

  // If serialization is enabled, store the prepacked tensor descriptor.
  if (!weights_manager_) {
    if (serialization_cache_ && serialization_cache_->IsReadyForInsert()) {
      ABSL_RETURN_IF_ERROR(serialization_cache_->Insert(
          global_tensor_id,
          /*is_quantization_param_tensor=*/true, tensor_desc));
    }
  }

  return local_value_id;
}

template <typename InputDataType>
absl::StatusOr<ValueId> SharedMemoryManager::AddScaleNodeWithData(
    uint32_t global_tensor_id, const Node& consumer_node,
    const InputDataType* data, DataType data_type, int num_channels,
    int num_blocks) {
  auto [tensor_it, tensor_inserted] =
      quant_param_id_to_spatial_tensor_.try_emplace(global_tensor_id);
  if (!tensor_inserted) {
    return absl::InternalError("Could not insert an input tensor.");
  }

  const OHWI shape = OHWI(num_channels, 1, 1, num_blocks);
  const BHWC bhwc_shape =
      BHWC(1, 1, DivideRoundUp(num_channels, 4), 4 * num_blocks);
  ABSL_ASSIGN_OR_RETURN(
      ValueId local_value_id,
      AddInputNode(global_tensor_id, bhwc_shape, consumer_node, data_type));

  Tensor<OHWI, DataType::FLOAT32> scale_values;
  scale_values.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < shape.DimensionsProduct(); ++i) {
    scale_values.data[i] = data[i];  // implicit cast of int8 -> fp32
  }
  scale_values.shape = shape;
  TensorDescriptor tensor_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info_, scale_values, data_type);

  ABSL_RETURN_IF_ERROR(create_tensor_func_(
      tensor_desc, /*page_adjusted_offset=*/0,
      /*release_data_callback=*/nullptr, tensor_it->second.weights));
  return local_value_id;
}

absl::StatusOr<ValueId> SharedMemoryManager::AddInputNode(
    uint32_t global_tensor_id, const BHWC& shape, const Node& consumer_node,
    DataType data_type) {
  Value* value = graph_.NewValue();
  value->tensor.ref = global_tensor_id;
  value->tensor.type = data_type;
  value->tensor.shape = shape;
  value->tensor.is_variable_input = false;
  graph_.AddConsumer(consumer_node.id, value->id);
  return value->id;
}

absl::Status SharedMemoryManager::RetrieveTensorWithScaleAndZeroPoint(
    const ValueId& shared_tensor_id, uint32_t global_tensor_id,
    const TfLiteTensor& tflite_tensor,
    absl::flat_hash_map<ValueId, GlobalId>* external_tensors) {
  auto& shared_tensor = buffer_id_to_spatial_tensor_.at(global_tensor_id);
  if (!shared_tensor.scale_global_tensor_id.has_value()) {
    return absl::InternalError("Expected scale tensor id to be set.");
  }
  if (!shared_tensor.zero_point_global_tensor_id.has_value()) {
    return absl::InternalError("Expected zero point tensor id to be set.");
  }
  Value* shared_const_tensor = graph_.GetValue(shared_tensor_id);

  std::vector<Node*> weight_consumers =
      graph_.FindConsumers(shared_const_tensor->id);

  if (weight_consumers.size() != 1) {
    return absl::InternalError("Expected to have only one weights consumer.");
  }
  Node* fc_node = weight_consumers[0];
  DataType data_type = data_type_;
  if (data_type_ == DataType::FLOAT32) {
    if (!graph_.FindInputs(fc_node->id).empty()) {
      data_type = graph_.FindInputs(fc_node->id)[0]->tensor.type;
    }
  }
  bool is_weight_sum_i_required = false;
  if (fc_node->operation.type ==
          ToString(OperationType::FULLY_CONNECTED_INT8) ||
      fc_node->operation.type ==
          ToString(OperationType::FULLY_CONNECTED_INT4)) {
    const bool is_conv_int8_supported =
        IsConvInt8Supported(create_info_, gpu_info_, graph_, fc_node) &&
        tflite_tensor.quantization.type ==
            TfLiteQuantizationType::kTfLiteAffineQuantization;
    is_weight_sum_i_required = is_conv_int8_supported &&
                               IsConvInt8KernelRecommendedForFullyConnectedOp(
                                   create_info_, gpu_info_, graph_, fc_node);
  }

  if (tflite_tensor.quantization.type ==
      TfLiteQuantizationType::kTfLiteAffineQuantization) {
    TfLiteAffineQuantization* quant_params =
        static_cast<TfLiteAffineQuantization*>(
            tflite_tensor.quantization.params);
    BHWC scale_or_zero_point_shape = BHWC(
        1, 1, 1, tflite_tensor.dims->data[quant_params->quantized_dimension]);

    ABSL_ASSIGN_OR_RETURN(
        ValueId scale_value_id,
        AddInputNode(shared_tensor.scale_global_tensor_id.value(),
                     scale_or_zero_point_shape, *fc_node, data_type));
    (*external_tensors)[scale_value_id] =
        GlobalId::BuildParamId(shared_tensor.scale_global_tensor_id.value());

    ABSL_ASSIGN_OR_RETURN(
        ValueId zero_point_value_id,
        AddInputNode(shared_tensor.zero_point_global_tensor_id.value(),
                     scale_or_zero_point_shape, *fc_node, data_type));
    (*external_tensors)[zero_point_value_id] =
      GlobalId::BuildParamId(shared_tensor.zero_point_global_tensor_id.value());

    if (is_weight_sum_i_required) {
      if (shared_tensor.weights_sum_i_global_tensor_id.has_value()) {
        // If the weights_sum_i is already populated, call AddInputNode() for
        // it.
        ABSL_ASSIGN_OR_RETURN(
            ValueId weights_sum_i_value_id,
            AddInputNode(shared_tensor.weights_sum_i_global_tensor_id.value(),
                         scale_or_zero_point_shape, *fc_node, DataType::INT32));
        (*external_tensors)[weights_sum_i_value_id] = GlobalId::BuildParamId(
            shared_tensor.weights_sum_i_global_tensor_id.value());
      } else {
        // If the weights_sum_i is not already populated, and we can support,
        // create it here. This can happen in the case the shared weight creator
        // was an op that didn't support weights_sum_i.
        shared_tensor.weights_sum_i_global_tensor_id = ++next_const_tensor_id_;

        ValueId weights_sum_i_value_id;
        if (weights_manager_) {
          ABSL_ASSIGN_OR_RETURN(
              weights_sum_i_value_id,
              AddInputNode(shared_tensor.weights_sum_i_global_tensor_id.value(),
                           scale_or_zero_point_shape, *fc_node,
                           DataType::INT32));

          OHWI weights_shape = OHWI(shared_const_tensor->tensor.shape.b,
                                    shared_const_tensor->tensor.shape.h,
                                    shared_const_tensor->tensor.shape.w,
                                    shared_const_tensor->tensor.shape.c);
          DataType src_data_type = DataType::INT4;
          if (tflite_tensor.type == TfLiteType::kTfLiteInt8) {
            src_data_type = DataType::INT8;
          } else if (tflite_tensor.type == TfLiteType::kTfLiteInt4) {
            src_data_type = DataType::INT4;
          } else if (tflite_tensor.type == TfLiteType::kTfLiteInt2) {
            src_data_type = DataType::INT2;
          }
          weights_manager_->RegisterWeightsSumIConversion(
              {weights_sum_i_value_id}, weights_shape, src_data_type,
              tflite_tensor.data.int8);

        } else {
          if (shared_tensor.weights_sum_i.data.empty()) {
            // Check if serialized or need to calculate.
            TensorDescriptor serialized_tensor_desc;
            bool serialized_tensor_found =
                serialization_cache_ &&
                serialization_cache_
                    ->LookUp(
                        shared_tensor.weights_sum_i_global_tensor_id.value(),
                        /*is_quantization_param_tensor=*/true,
                        serialized_tensor_desc)
                    .ok();
            // If we do not have the weights_sum_i cached in RAM and it was not
            // serialized, calculate it here.
            if (!serialized_tensor_found) {
              ABSL_RETURN_IF_ERROR(
                  CalculateWeightsSumI(tflite_tensor, shared_const_tensor,
                                       &shared_tensor.weights_sum_i));
            }
          }
          auto* weights_sum_i_data =
              (shared_tensor.weights_sum_i.data.empty())
                  ? nullptr
                  : shared_tensor.weights_sum_i.data.data();
          Linear weights_sum_i_shape = Linear(
              tflite_tensor.dims->data[quant_params->quantized_dimension]);
          absl::StatusOr<ValueId> status_or_value_id = AddInputWithData(
              shared_tensor.weights_sum_i_global_tensor_id.value(),
              weights_sum_i_shape, *fc_node, weights_sum_i_data,
              DataType::INT32);
          if (!status_or_value_id.ok()) {
            return status_or_value_id.status();
          }
          weights_sum_i_value_id = status_or_value_id.value();
        }
        (*external_tensors)[weights_sum_i_value_id] = GlobalId::BuildParamId(
            shared_tensor.weights_sum_i_global_tensor_id.value());
      }
    }
  } else if (tflite_tensor.quantization.type ==
             TfLiteQuantizationType::kTfLiteBlockwiseQuantization) {
    TfLiteBlockwiseQuantization* quant_params =
        static_cast<TfLiteBlockwiseQuantization*>(
            tflite_tensor.quantization.params);
    BHWC bhwc_shape =
        BHWC(1, 1, DivideRoundUp(tflite_tensor.dims->data[0], 4),
             4 * tflite_tensor.dims->data[1] / quant_params->blocksize);

    ABSL_ASSIGN_OR_RETURN(
        ValueId scale_value_id,
        AddInputNode(shared_tensor.scale_global_tensor_id.value(), bhwc_shape,
                     *fc_node, data_type));
    (*external_tensors)[scale_value_id] =
        GlobalId::BuildParamId(shared_tensor.scale_global_tensor_id.value());

    ABSL_ASSIGN_OR_RETURN(
        ValueId zero_point_value_id,
        AddInputNode(shared_tensor.zero_point_global_tensor_id.value(),
                     bhwc_shape, *fc_node, data_type));
    (*external_tensors)[zero_point_value_id] = GlobalId::BuildParamId(
        shared_tensor.zero_point_global_tensor_id.value());
  } else {
    return absl::UnimplementedError(
        "Only affine quantization is supported for the shared tensors.");
  }

  if (weights_manager_) {
    BHWC& bhwc_shape = shared_const_tensor->tensor.shape;
    OHWI shape(bhwc_shape.b, bhwc_shape.h, bhwc_shape.w, bhwc_shape.c);
    WeightsDescription weights_desc;
    switch (tflite_tensor.type) {
      case kTfLiteInt8:
        weights_desc = GetFullyConnectedInt8WeightsDesc(
            gpu_info_, shape,
            create_info_.hints.Check(ModelHints::kPreferTextureWeights));
        break;
      case kTfLiteInt4:
        weights_desc = GetFullyConnectedInt4WeightsDesc(
            gpu_info_, shape,
            create_info_.hints.Check(ModelHints::kPreferTextureWeights));
        break;
      case kTfLiteInt2:
        weights_desc = GetFullyConnectedInt2WeightsDesc(
            gpu_info_, shape,
            create_info_.hints.Check(ModelHints::kPreferTextureWeights));
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported tensor type in SharedMemoryManager for "
                         "the FullyConnected op: ",
                         tflite_tensor.type));
    }
    auto weights_int_tensor_desc =
        GetTensorDescriptorsForWeightsLayout(shape, weights_desc)[0];
    shared_const_tensor->tensor.shape = weights_int_tensor_desc.GetBHWCShape();
    shared_const_tensor->tensor.type = weights_int_tensor_desc.GetDataType();
  } else {
    auto* weights = shared_tensor.GetWeights();
    shared_const_tensor->tensor.shape =
        BHWC(weights->Batch(), weights->Height(), weights->Width(),
             weights->Channels());
    shared_const_tensor->tensor.type = weights->GetDescriptor().GetDataType();
  }
  return absl::OkStatus();
}
absl::Status SharedMemoryManager::MaybeBindTensorData(
    const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
    TfLiteTensor& tensor) {
  if (!maybe_bind_tensor_data_func_) {
    return absl::OkStatus();
  }
  return maybe_bind_tensor_data_func_(shared_tflite_tensor, tensor);
}

absl::Status SharedMemoryManager::DiscardTensorData(
    const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor) {
  if (!discard_tensor_data_func_) {
    return absl::OkStatus();
  }
  return discard_tensor_data_func_(shared_tflite_tensor);
}

absl::Status SharedMemoryManager::CreateSharedTensor(
    const ValueId& shared_tensor_id,
    const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
    std::unique_ptr<GpuSpatialTensor>& gpu_spatial_tensor,
    absl::flat_hash_map<ValueId, GlobalId>* external_tensors) {
  // Weights tensor is quantized and should be uploaded to GPU quantized, go
  // ahead to the early exit.
  if (shared_tflite_tensor.tflite_tensor_id >= context_->tensors_size) {
    return absl::OutOfRangeError("Tensor index is out of range.");
  }
  TfLiteTensor& tensor =
      context_->tensors[shared_tflite_tensor.tflite_tensor_id];
  ABSL_RETURN_IF_ERROR(MaybeBindTensorData(shared_tflite_tensor, tensor));
  if ((tensor.quantization.type ==
           TfLiteQuantizationType::kTfLiteAffineQuantization ||
       tensor.quantization.type ==
           TfLiteQuantizationType::kTfLiteBlockwiseQuantization) &&
      !shared_tflite_tensor.dequant_forced) {
    return CreateQuantizedTensorWithScaleAndZeroPoint(
        shared_tensor_id, shared_tflite_tensor.global_id, tensor,
        external_tensors);
  }

  // If weights tensor is quantized, dequantize it. Otherwise it is float and we
  // just grab the pointer.
  std::vector<float> dequantized_weight;
  float* weights_data_ptr = tensor.data.f;
  if (tensor.quantization.type ==
          TfLiteQuantizationType::kTfLiteAffineQuantization &&
      shared_tflite_tensor.dequant_forced) {
    dequantized_weight.resize(tflite::NumElements(&tensor));
    ::litert::ml_drift::CopyData(tensor, &dequantized_weight[0]);
    weights_data_ptr = dequantized_weight.data();
  } else if (tensor.quantization.type ==
                 TfLiteQuantizationType::kTfLiteBlockwiseQuantization &&
             shared_tflite_tensor.dequant_forced) {
    return absl::UnimplementedError(
        "Enforce dequant Block-wise quantized weights is not supported yet.");
  }

  // Initialize tensor descriptor and upload data.
  TensorRef<BHWC>& tensor_ref = graph_.GetValue(shared_tensor_id)->tensor;
  DataType data_type = data_type_;
  if (data_type_ == DataType::FLOAT32) {
    std::vector<Node*> consumers = graph_.FindConsumers(shared_tensor_id);
    if (consumers.size() == 1 && !graph_.FindInputs(consumers[0]->id).empty()) {
      data_type = DataType::FLOAT16;
      data_type = graph_.FindInputs(consumers[0]->id)[0]->tensor.type;
    }
  }
  ml_drift::TensorDescriptor tensor_desc;
  // Linear layout is forced for shared bias tensors. For other shared tensors,
  // prefer HWC when batch is 1 to avoid external tensor descriptors that carry
  // an unnecessary batch axis.
  Layout layout = shared_tflite_tensor.layout.has_value()
                      ? shared_tflite_tensor.layout.value()
                      : (tensor_ref.shape.b == 1 ? Layout::HWC : Layout::BHWC);
  BHWC shape = layout == Layout::LINEAR ? BHWC(1, 1, 1, tensor_ref.shape.c)
                                        : tensor_ref.shape;
  tensor_desc =
      ml_drift::TensorDescriptor(data_type, create_info_.storage_type, layout);
  tensor_desc.SetBHWCShape(shape);

  tensor_ref.type = tensor_desc.GetDataType();
  ABSL_RETURN_IF_ERROR(tensor_desc.UpdateToSupportedStorageType(
      gpu_info_, tensor_desc.GetBHWCShape()));
  absl::Status device_status = TryCreateTensorFromDeviceBuffer(
      shared_tflite_tensor, tensor_desc, gpu_spatial_tensor);
  if (device_status.ok()) {
    return absl::OkStatus();
  }

  // Support uploading float16 data for float16 tensors, otherwise upload float
  // data. This is used for the models with fp16 weights from MediaPipe. (e.g.
  // inpainting models)
  if (tensor_desc.GetDataType() == DataType::FLOAT16) {
    if (tensor.type == TfLiteType::kTfLiteFloat16) {
      tensor_desc.UploadData<half>(reinterpret_cast<half*>(tensor.data.f16));
    } else {
      int num_elements = tflite::NumElements(&tensor);
      std::vector<half> half_data(num_elements);
      for (int i = 0; i < num_elements; ++i) {
        half_data[i] = half(weights_data_ptr[i]);
      }
      tensor_desc.UploadData<half>(half_data.data());
    }
  } else {  // FLOAT32
    if (tensor.type == TfLiteType::kTfLiteFloat16) {
      int num_elements = tflite::NumElements(&tensor);
      std::vector<float> float_data(num_elements);
      const half* f16_ptr = reinterpret_cast<const half*>(tensor.data.f16);
      for (int i = 0; i < num_elements; ++i) {
        float_data[i] = static_cast<float>(f16_ptr[i]);
      }
      tensor_desc.UploadData<float>(float_data.data());
    } else {
      tensor_desc.UploadData<float>(weights_data_ptr);
    }
  }
  return create_tensor_func_(tensor_desc, /*page_adjusted_offset=*/0,
                             /*release_data_callback=*/nullptr,
                             gpu_spatial_tensor);
}

absl::Status SharedMemoryManager::TryCreateTensorFromDeviceBuffer(
    const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
    const TensorDescriptor& tensor_desc,
    std::unique_ptr<GpuSpatialTensor>& gpu_spatial_tensor) {
  if (!create_tensor_from_device_buffer_func_) {
    return absl::NotFoundError("Device buffer import not supported");
  }
  return create_tensor_from_device_buffer_func_(
      shared_tflite_tensor, tensor_desc, gpu_spatial_tensor);
}

// This function is to register a weight (carried by TfLiteTensor) to
// SharedMemoryManager, SharedMemoryManager cooks it to GpuSpatialTensor, and
// then the GpuSpatialTensor can be shared across subgraphs.
//
// About IDs:
// * shared_tensor_id is the tensor ID in GraphFloat32, only valid for the
// current subgraph.
// * shared_tflite_tensor.global_id is the global unique ID across all
// subgraphs, and will be used to index the GpuSpatialTensor object.
// * shared_tflite_tensor.tflite_tensor_id is the tensor ID in the TfLite
// context, to access the TfLiteTensor object.
// * local_to_global_id_map is used to map the local GraphFloat32 tensor ID (eg.
// shared_tensor_id) to the global ID for GpuSpatialTensor indexing (eg.
// shared_tflite_tensor.global_id).
absl::Status SharedMemoryManager::RegisterExternalConstantTensors(
    const ValueId& shared_tensor_id,
    const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
    absl::flat_hash_map<ValueId, GlobalId>& local_to_global_id_map) {
  auto [tensor_it, new_spatial_tensor_inserted] =
      buffer_id_to_spatial_tensor_.try_emplace(shared_tflite_tensor.global_id);
  local_to_global_id_map[shared_tensor_id] =
      GlobalId::BuildSourceId(shared_tflite_tensor.global_id);
  local_to_tflite_tensor_id_[shared_tensor_id] =
      shared_tflite_tensor.tflite_tensor_id;
  if (shared_tflite_tensor.tflite_tensor_id >= context_->tensors_size) {
    return absl::OutOfRangeError("Tensor index is out of range.");
  }
  TfLiteTensor& tensor =
      context_->tensors[shared_tflite_tensor.tflite_tensor_id];
  if (new_spatial_tensor_inserted) {
    ABSL_RETURN_IF_ERROR(
        CreateSharedTensor(shared_tensor_id, shared_tflite_tensor,
                           tensor_it->second.weights, &local_to_global_id_map));
    // We only call madvise here because we have touched the original tensor's
    // memory. In other cases, we have not touched the memory, the pages likely
    // have not been allocated so madvise would have no effect.
    if (madvise_original_tensors_) {
      TfLiteTensor& tensor =
          context_->tensors[shared_tflite_tensor.tflite_tensor_id];
      MadviseTensor(tensor);
      ABSL_RETURN_IF_ERROR(DiscardTensorData(shared_tflite_tensor));
    }
  } else {
    ABSL_RETURN_IF_ERROR(MaybeBindTensorData(shared_tflite_tensor, tensor));
  }
  if (!new_spatial_tensor_inserted &&
      (tensor.quantization.type ==
           TfLiteQuantizationType::kTfLiteAffineQuantization ||
       tensor.quantization.type ==
           TfLiteQuantizationType::kTfLiteBlockwiseQuantization)) {
    if (!shared_tflite_tensor.dequant_forced) {
      ABSL_RETURN_IF_ERROR(RetrieveTensorWithScaleAndZeroPoint(
          shared_tensor_id, shared_tflite_tensor.global_id, tensor,
          &local_to_global_id_map));
    }
    if (shared_tflite_tensor.dequant_forced) {
      auto& shared_tensor =
          buffer_id_to_spatial_tensor_.at(shared_tflite_tensor.global_id);
      Value* shared_const_tensor = graph_.GetValue(shared_tensor_id);
      auto* weights = shared_tensor.GetWeights();
      shared_const_tensor->tensor.shape =
          BHWC(weights->Batch(), weights->Height(), weights->Width(),
               weights->Channels());
      shared_const_tensor->tensor.type = weights->GetDescriptor().GetDataType();
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<ml_drift::GpuSpatialTensor*>
SharedMemoryManager::GetExternalConstantTensor(const GlobalId& global_id) {
  if (auto it = buffer_id_to_spatial_tensor_.find(global_id.value);
      global_id.IsSourceId() && it != buffer_id_to_spatial_tensor_.end()) {
    return it->second.GetWeights();
  } else if (auto it = quant_param_id_to_spatial_tensor_.find(global_id.value);
             global_id.IsParamId() &&
             it != quant_param_id_to_spatial_tensor_.end()) {
    return it->second.GetWeights();
  }

  std::string global_id_str =
      global_id.IsSourceId() ? " source"
                             : (global_id.IsParamId() ? " param" : " unknown");
  absl::StrAppend(&global_id_str, " id ", global_id.value);
  return absl::InternalError(
      absl::StrCat("Tensor with global_id: ", global_id_str, " not found."));
}

absl::StatusOr<std::unique_ptr<GpuSpatialTensor>>
SharedMemoryManager::CreatePrepackedWeightsTensorFromTfliteTensor(
    const WeightsDescription& weights_desc, const OHWI& shape,
    const TfLiteTensor& tflite_tensor) {
  DataType data_type = weights_desc.type;
  int count = shape.DimensionsProduct();
  if (data_type == DataType::UINT4) {
    count /= 2;
  } else if (data_type == DataType::UINT2 || data_type == DataType::INT2) {
    return absl::InternalError(
        "INT2 and UINT2 data types are not supported for prepacked tensors.");
  }
  absl::Span<const uint8_t> data = absl::MakeSpan(
      reinterpret_cast<const uint8_t*>(tflite_tensor.data.data), count);

  if (weights_desc.IsLinearLayout()) {
    TensorDescriptor tensor_desc(data_type, TensorStorageType::BUFFER,
                                 Layout::LINEAR);
    tensor_desc.SetBHWDCShape(BHWDC(1, 1, 1, 1, count));
    UnownedDataTensorDescriptor unowned_desc(tensor_desc, data);
    std::unique_ptr<GpuSpatialTensor> tensor;
    ABSL_RETURN_IF_ERROR(
        create_tensor_func_(unowned_desc, /*page_adjusted_offset=*/0,
                            /*release_data_callback=*/nullptr, tensor));
    return tensor;
  }

  DataType texture_type = DataType::UINT32;
  if (data_type == DataType::UINT4) {
    texture_type = DataType::UINT16;
  }
  TensorDescriptor tensor_desc(texture_type, TensorStorageType::TEXTURE_2D,
                               Layout::HW);
  uint2 tex_size = Get2dResourceSize(weights_desc, shape);
  // Values are packed: 4 x uint4 = uint16, 4 x uint8 = uint32.
  tex_size.x /= 4;
  tensor_desc.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  UnownedDataTensorDescriptor unowned_desc(tensor_desc, data);
  std::unique_ptr<GpuSpatialTensor> tensor;
  ABSL_RETURN_IF_ERROR(
      create_tensor_func_(unowned_desc, /*page_adjusted_offset=*/0,
                          /*release_data_callback=*/nullptr, tensor));
  return tensor;
}

uint32_t SharedMemoryManager::MaybeGetExternalBufferId(
    const ValueId& shared_tensor_id) {
  uint32_t external_buffer_id = kInvalidExternalBufferId;
  if (maybe_get_external_buffer_id_func_) {
    auto it = local_to_tflite_tensor_id_.find(shared_tensor_id);
    if (it != local_to_tflite_tensor_id_.end()) {
      auto status_or_external_buffer_id =
          maybe_get_external_buffer_id_func_(it->second);
      if (status_or_external_buffer_id.ok()) {
        external_buffer_id = status_or_external_buffer_id.value();
      }
    }
  }
  return external_buffer_id;
}

uint64_t GetSharedMemorySizeFromMap(
    const ::ml_drift::ValueIdToSharedTensorMap& map) {
  uint64_t size = 0;
  for (const auto& [key, shared_tensor] : map) {
    size += shared_tensor.GetWeights()->GetDescriptor().GetMemorySizeInBytes();
  }
  return size;
}

}  // namespace ml_drift
