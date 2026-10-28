/* Copyright 2026 Google LLC.

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

#include "tensor/backends/xnnpack/dequantize.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

namespace {

// Maps the flat index of an element of a quantized tensor to the index of its
// scale and zero point.

// A single scale and zero point for the whole tensor.
struct PerTensorIndex {
  size_t operator()(size_t) const { return 0; }
};

// One scale and zero point per index along the quantized dimension, which has
// `channels` indices and is followed by dimensions of `inner` elements in
// total.
struct PerChannelIndex {
  size_t channels;
  size_t inner;
  size_t operator()(size_t i) const { return i / inner % channels; }
};

// One scale and zero point per block of `block_size` consecutive elements of
// each channel of a [rows, cols] matrix, whose channels are its rows if
// `channels_are_rows` and its columns otherwise. The scales are stored channel
// by channel, `num_blocks` per channel.
struct BlockwiseIndex {
  size_t cols;
  bool channels_are_rows;
  size_t block_size;
  size_t num_blocks;
  size_t operator()(size_t i) const {
    const size_t row = i / cols;
    const size_t col = i % cols;
    const size_t channel = channels_are_rows ? row : col;
    const size_t position = channels_are_rows ? col : row;
    return channel * num_blocks + position / block_size;
  }
};

// The validated quantization parameters of a constant tensor.
struct DequantizationParams {
  size_t num_elements;
  absl::Span<const float> scales;
  // One per scale.
  std::vector<int64_t> zero_points;
  std::variant<PerTensorIndex, PerChannelIndex, BlockwiseIndex> index;
};

// Returns one zero point per scale: `zero_points` may be empty, which means
// zeros, have a single zero point shared by all the scales, or one per scale.
absl::StatusOr<std::vector<int64_t>> GetZeroPointPerScale(
    absl::Span<const int64_t> zero_points, size_t num_scales) {
  if (zero_points.empty()) {
    return std::vector<int64_t>(num_scales, 0);
  }
  if (zero_points.size() == 1) {
    return std::vector<int64_t>(num_scales, zero_points[0]);
  }
  if (zero_points.size() != num_scales) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected 0, 1 or %zu zero points, got %zu", num_scales,
                        zero_points.size()));
  }
  return std::vector<int64_t>(zero_points.begin(), zero_points.end());
}

absl::StatusOr<DequantizationParams> GetDequantizationParams(
    const graph::TensorInformation& info,
    absl::Span<const std::byte> raw_data) {
  if (info.quantization == nullptr) {
    return absl::InvalidArgumentError("Missing quantization parameters");
  }
  DequantizationParams params{.num_elements = 1};
  for (const int32_t dim : info.shape) {
    if (dim < 0) {
      return absl::InvalidArgumentError(
          "Only tensors with a static shape can be dequantized");
    }
    params.num_elements *= dim;
  }
  const size_t num_bytes = (BitSize(info.type) * params.num_elements + 7) / 8;
  if (raw_data.size() < num_bytes) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Raw data size %zu is smaller than expected %zu",
                        raw_data.size(), num_bytes));
  }

  const int rank = static_cast<int>(info.shape.size());
  absl::Span<const int64_t> zero_points;
  if (auto pcq =
          info.quantization->As<const graph::PerChannelAffineQuantization>();
      pcq.ok()) {
    params.scales = pcq->scales;
    zero_points = pcq->zero_points;
    if (params.scales.empty()) {
      return absl::InvalidArgumentError("Missing quantization scales");
    }
    if (params.scales.size() == 1) {
      params.index = PerTensorIndex();
    } else {
      const int dim = pcq->quantized_dimension;
      if (dim < 0 || dim >= rank) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Quantized dimension %d is out of range for rank %d", dim, rank));
      }
      const size_t channels = info.shape[dim];
      if (params.scales.size() != channels) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected %zu scales, got %zu", channels, params.scales.size()));
      }
      size_t inner = 1;
      for (int d = dim + 1; d < rank; ++d) {
        inner *= info.shape[d];
      }
      params.index = PerChannelIndex{.channels = channels, .inner = inner};
    }
  } else if (auto bwq =
                 info.quantization->As<const graph::BlockwiseQuantization>();
             bwq.ok()) {
    params.scales = bwq->scales;
    zero_points = bwq->zero_points;
    if (rank != 2) {
      return absl::UnimplementedError(
          "Only rank 2 blockwise quantized tensors can be dequantized");
    }
    const int dim = bwq->quantized_dimension;
    if (dim != 0 && dim != 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Quantized dimension %d is out of range for rank 2", dim));
    }
    const size_t channels = info.shape[dim];
    const size_t blocked = info.shape[1 - dim];
    if (bwq->block_size <= 0 || blocked % bwq->block_size != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block size %d doesn't divide %zu", bwq->block_size, blocked));
    }
    const size_t num_blocks = blocked / bwq->block_size;
    if (params.scales.size() != channels * num_blocks) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected %zu scales, got %zu", channels * num_blocks,
                          params.scales.size()));
    }
    params.index = BlockwiseIndex{
        .cols = static_cast<size_t>(info.shape[1]),
        .channels_are_rows = dim == 0,
        .block_size = static_cast<size_t>(bwq->block_size),
        .num_blocks = num_blocks,
    };
  } else {
    return absl::UnimplementedError("Unsupported quantization");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      params.zero_points,
      GetZeroPointPerScale(zero_points, params.scales.size()));
  return params;
}

// Dequantizes the integers of type `type` in `raw_data`, element `i` with the
// scale and zero point `index(i)`. Works for packed integers and for signed
// and unsigned types.
template <Type type, class Index>
std::vector<float> Dequantize(absl::Span<const std::byte> raw_data,
                              const DequantizationParams& params, Index index) {
  using Storage = NativeStorage<type>;
  static_assert(std::is_integral_v<internal::ElementType<type>>,
                "Only integers can be dequantized.");
  const auto* packed =
      reinterpret_cast<const typename Storage::type*>(raw_data.data());
  std::vector<float> dequantized(params.num_elements);
  for (size_t i = 0; i < params.num_elements; ++i) {
    const int64_t value = internal::GetPackedValue(
        packed[i / Storage::kNumElements], i % Storage::kNumElements);
    const size_t s = index(i);
    dequantized[i] =
        static_cast<float>(value - params.zero_points[s]) * params.scales[s];
  }
  return dequantized;
}

template <Type type>
absl::StatusOr<std::vector<float>> DequantizeAs(
    absl::Span<const std::byte> raw_data, const DequantizationParams& params) {
  // Only instantiates `Dequantize` for the scales that the XNNPACK backend
  // dequantizes, i.e. per-channel ones. The other alternatives of
  // `params.index` can be added here when needed.
  if (const auto* index = std::get_if<PerChannelIndex>(&params.index)) {
    return Dequantize<type>(raw_data, params, *index);
  }
  return absl::UnimplementedError(
      "Only per-channel quantized tensors can be dequantized");
}

}  // namespace

absl::StatusOr<std::vector<float>> DequantizeConstantTensor(
    const graph::TensorInformation& info,
    absl::Span<const std::byte> raw_data) {
  // Only instantiates the template for the types that the XNNPACK backend
  // quantizes per channel. `Dequantize` also supports other integer types,
  // e.g. unsigned ones, which can be added here when needed.
  using DequantizeFn = absl::StatusOr<std::vector<float>> (*)(
      absl::Span<const std::byte>, const DequantizationParams&);
  DequantizeFn dequantize = nullptr;
  switch (info.type) {
    case Type::kI2:
      dequantize = &DequantizeAs<Type::kI2>;
      break;
    case Type::kI4:
      dequantize = &DequantizeAs<Type::kI4>;
      break;
    case Type::kI8:
      dequantize = &DequantizeAs<Type::kI8>;
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Can't dequantize ", ToString(info.type), " values"));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const DequantizationParams params,
                              GetDequantizationParams(info, raw_data));
  return dequantize(raw_data, params);
}

}  // namespace litert::tensor
