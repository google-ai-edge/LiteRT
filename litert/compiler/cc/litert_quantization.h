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

#ifndef ODML_LITERT_LITERT_COMPILER_CC_LITERT_QUANTIZATION_H_
#define ODML_LITERT_LITERT_COMPILER_CC_LITERT_QUANTIZATION_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_model.h"

/// @file
/// @brief Quantization helpers for compiler plugins.
///
/// Vendor backends frequently need the float32 values behind a quantized
/// constant tensor, either because the backend does not support the quantized
/// form of an op, or because a transformation constant-folds the dequantization
/// at compile time instead of emitting a runtime `Dequantize` op. These helpers
/// provide a single, shared implementation of that conversion so plugins do not
/// each have to re-derive the layout and q-param handling.

namespace litert::compiler {

namespace quantization_internal {

/// @brief Applies `out[i] = (src[i] - zero_point(i)) * scale(i)` elementwise.
///
/// `scale_fn` and `zero_point_fn` map a flat element index to the q-params that
/// apply to it, which lets the same loop serve both per-tensor and per-channel
/// quantization.
template <typename StorageT, typename ScaleFn, typename ZeroPointFn>
void Dequantize(const void* src_bytes, size_t num_elements,
                const ScaleFn& scale_fn, const ZeroPointFn& zero_point_fn,
                absl::Span<float> out) {
  const auto* src = reinterpret_cast<const StorageT*>(src_bytes);
  for (size_t i = 0; i < num_elements; ++i) {
    out[i] = (static_cast<float>(src[i]) - zero_point_fn(i)) * scale_fn(i);
  }
}

/// @brief Dispatches `Dequantize` on the storage type of the quantized data.
template <typename ScaleFn, typename ZeroPointFn>
Expected<void> DequantizeByType(ElementType element_type, const void* src_bytes,
                                size_t num_elements, const ScaleFn& scale_fn,
                                const ZeroPointFn& zero_point_fn,
                                absl::Span<float> out) {
  switch (element_type) {
    case ElementType::Int8:
      Dequantize<int8_t>(src_bytes, num_elements, scale_fn, zero_point_fn, out);
      return {};
    case ElementType::UInt8:
      Dequantize<uint8_t>(src_bytes, num_elements, scale_fn, zero_point_fn,
                          out);
      return {};
    case ElementType::Int16:
      Dequantize<int16_t>(src_bytes, num_elements, scale_fn, zero_point_fn,
                          out);
      return {};
    case ElementType::UInt16:
      Dequantize<uint16_t>(src_bytes, num_elements, scale_fn, zero_point_fn,
                           out);
      return {};
    case ElementType::Int32:
      Dequantize<int32_t>(src_bytes, num_elements, scale_fn, zero_point_fn,
                          out);
      return {};
    default:
      // Sub-byte types (Int2/Int4/UInt4) are intentionally excluded: their
      // packed representation must be unpacked before it can be strided over.
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported storage type for weights dequantization");
  }
}

}  // namespace quantization_internal

/// @brief Returns the number of elements in `ranked_type`.
///
/// Unlike `Layout::NumElements`, a dynamic or non-positive dimension is
/// reported as `kLiteRtStatusErrorUnsupported`: such a tensor is well formed,
/// it simply cannot be constant-folded at compile time.
inline Expected<size_t> NumElementsForDequantize(
    const RankedTensorType& ranked_type) {
  size_t num_elements = 1;
  for (const auto dim : ranked_type.Layout().Dimensions()) {
    if (dim <= 0) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Dynamic or non-positive dimensions are not supported for "
                   "weights dequantization");
    }
    num_elements *= static_cast<size_t>(dim);
  }
  return num_elements;
}

/// @brief Dequantizes the constant weights of `tensor` into `out`.
///
/// `out` must have room for at least `NumElements()` floats. Prefer this
/// overload over `DequantizeWeights` when the destination buffer already
/// exists, as it avoids an intermediate allocation.
///
/// Supports per-tensor and per-channel quantization over Int8, UInt8, Int16,
/// UInt16 and Int32 storage. Block-wise quantization and sub-byte storage types
/// are rejected with `kLiteRtStatusErrorUnsupported`.
inline Expected<void> DequantizeWeightsInto(const Tensor& tensor,
                                            absl::Span<float> out) {
  if (!tensor.HasWeights()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Tensor has no weights to dequantize");
  }

  LITERT_ASSIGN_OR_RETURN(auto ranked_type, tensor.RankedTensorType());
  const auto dimensions = ranked_type.Layout().Dimensions();
  LITERT_ASSIGN_OR_RETURN(const size_t num_elements,
                          NumElementsForDequantize(ranked_type));

  if (out.size() < num_elements) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Output buffer is too small to hold the dequantized weights");
  }

  const ElementType element_type = ranked_type.ElementType();
  const auto byte_width = GetByteWidth(element_type);
  if (!byte_width.has_value()) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported element type for byte width calculation");
  }

  const absl::Span<const uint8_t> bytes = tensor.Weights().Bytes();
  if (bytes.size() < byte_width->NumBytes(num_elements)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Weight buffer is smaller than the tensor shape requires");
  }

  switch (tensor.QTypeId()) {
    case kLiteRtQuantizationPerTensor: {
      const LiteRtQuantizationPerTensor q = tensor.PerTensorQuantization();
      const float scale = q.scale;
      const float zero_point = static_cast<float>(q.zero_point);
      return quantization_internal::DequantizeByType(
          element_type, bytes.data(), num_elements,
          [scale](size_t) { return scale; },
          [zero_point](size_t) { return zero_point; }, out);
    }

    case kLiteRtQuantizationPerChannel: {
      const LiteRtQuantizationPerChannel q = tensor.PerChannelQuantization();
      if (q.scales == nullptr) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Per-channel quantization is missing scales");
      }
      if (q.quantized_dimension < 0 ||
          static_cast<size_t>(q.quantized_dimension) >= dimensions.size()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Per-channel quantized_dimension is out of range");
      }
      if (static_cast<uint64_t>(dimensions[q.quantized_dimension]) !=
          q.num_channels) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Per-channel num_channels does not match the extent of "
                     "the quantized dimension");
      }

      // Elements are laid out contiguously in row-major order, so the channel
      // owning a flat index is determined purely by the extent of the
      // dimensions that follow the quantized one.
      size_t inner_size = 1;
      for (size_t i = q.quantized_dimension + 1; i < dimensions.size(); ++i) {
        inner_size *= static_cast<size_t>(dimensions[i]);
      }
      const size_t num_channels = q.num_channels;
      const auto channel_of = [inner_size, num_channels](size_t i) {
        return (i / inner_size) % num_channels;
      };

      const float* scales = q.scales;
      const int64_t* zero_points = q.zero_points;
      return quantization_internal::DequantizeByType(
          element_type, bytes.data(), num_elements,
          [scales, channel_of](size_t i) { return scales[channel_of(i)]; },
          [zero_points, channel_of](size_t i) {
            // Zero points are optional; symmetric quantization omits them.
            return zero_points == nullptr
                       ? 0.0f
                       : static_cast<float>(zero_points[channel_of(i)]);
          },
          out);
    }

    case kLiteRtQuantizationBlockWise:
      return Error(kLiteRtStatusErrorUnsupported,
                   "Block-wise quantization is unsupported for weights "
                   "dequantization");

    default:
      return Error(kLiteRtStatusErrorUnsupported,
                   "Tensor is not quantized, or uses an unsupported "
                   "quantization scheme");
  }
}

/// @brief Dequantizes the constant weights of `tensor` into a new float vector.
///
/// See `DequantizeWeightsInto` for the supported quantization schemes and
/// storage types.
inline Expected<std::vector<float>> DequantizeWeights(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto ranked_type, tensor.RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(const size_t num_elements,
                          NumElementsForDequantize(ranked_type));
  std::vector<float> result(num_elements);
  LITERT_RETURN_IF_ERROR(DequantizeWeightsInto(tensor, absl::MakeSpan(result)));
  return result;
}

}  // namespace litert::compiler

#endif  // ODML_LITERT_LITERT_COMPILER_CC_LITERT_QUANTIZATION_H_
