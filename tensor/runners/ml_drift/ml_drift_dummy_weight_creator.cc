/* Copyright 2025 Google LLC.

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
#include "tensor/runners/ml_drift/ml_drift_dummy_weight_creator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/parallel.h"

namespace litert::tensor {

namespace {
ml_drift::WeightsDescription GetFullyConnectedWeightsDesc(
    ml_drift::GpuModelBuilder* model_builder, ml_drift::DataType data_type,
    const ml_drift::OHWI& weights_shape) {
  if (data_type == ml_drift::DataType::kInt8) {
    return model_builder->GetFullyConnectedInt8WeightsDesc(weights_shape);
  } else if (data_type == ml_drift::DataType::kInt4) {
    return model_builder->GetFullyConnectedInt4WeightsDesc(weights_shape);
  } else if (data_type == ml_drift::DataType::kInt2) {
    return model_builder->GetFullyConnectedInt2WeightsDesc(weights_shape);
  } else {
    auto desc =
        model_builder->GetFullyConnectedWeightsDesc(data_type, weights_shape);
    return desc;
  }
}

template <typename T>
absl::StatusOr<ml_drift::OHWI> ShapeToOHWI(const T& shape) {
  switch (shape.size()) {
    case 2:
      return ml_drift::OHWI(shape[0], 1, 1, shape[1]);
    case 3:
      return ml_drift::OHWI(shape[1], shape[0], 1, shape[2]);
    case 4:
      return ml_drift::OHWI(shape[0], shape[1], shape[2], shape[3]);
    default:
      return absl::InvalidArgumentError(absl::StrCat("Cannot convert shape ",
                                                     absl::StrJoin(shape, ", "),
                                                     " to MlDrift OHWI."));
  }
}

}  // namespace

bool TryDirectRearrangeInt8AsUint8I4O4(const int8_t* src,
                                       const ml_drift::OHWI& shape,
                                       const ml_drift::WeightsDescription& desc,
                                       absl::Span<uint8_t> dst) {
  if (shape.h != 1 || shape.w != 1 || shape.o <= 0 || shape.i <= 0) {
    return false;
  }
  const bool is_linear_i4o4 =
      desc.layout == ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  const bool is_tex_i4o4 =
      desc.layout == ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  if (!is_linear_i4o4 && !is_tex_i4o4) {
    return false;
  }
  const int dst_group_size = desc.GetOutputGroupSize();
  if (dst_group_size <= 0) return false;
  const int dst_slices = ml_drift::DivideRoundUp(shape.o, 4);
  const int src_slices = ml_drift::DivideRoundUp(shape.i, 4);
  const int dst_groups = ml_drift::DivideRoundUp(dst_slices, dst_group_size);
  const size_t expected_bytes =
      static_cast<size_t>(dst_groups) * dst_group_size * src_slices * 16;
  if (dst.size() < expected_bytes) return false;

  const size_t out_channels = static_cast<size_t>(shape.o);
  const size_t in_channels = static_cast<size_t>(shape.i);
  auto pack_tile = [&](int d, int s, uint8_t* out_tile) {
    for (int d_group = 0; d_group < dst_group_size; ++d_group) {
      const size_t d_ch =
          static_cast<size_t>(4 * (d * dst_group_size + d_group));
      const size_t s_base = static_cast<size_t>(4 * s);
      if (d_ch + 4 <= out_channels && s_base + 4 <= in_channels) {
        const uint8_t* r0 = reinterpret_cast<const uint8_t*>(
            src + (d_ch + 0) * in_channels + s_base);
        const uint8_t* r1 = reinterpret_cast<const uint8_t*>(
            src + (d_ch + 1) * in_channels + s_base);
        const uint8_t* r2 = reinterpret_cast<const uint8_t*>(
            src + (d_ch + 2) * in_channels + s_base);
        const uint8_t* r3 = reinterpret_cast<const uint8_t*>(
            src + (d_ch + 3) * in_channels + s_base);
        for (int k = 0; k < 4; ++k) {
          out_tile[k * 4 + 0] = r0[k] ^ 0x80u;
          out_tile[k * 4 + 1] = r1[k] ^ 0x80u;
          out_tile[k * 4 + 2] = r2[k] ^ 0x80u;
          out_tile[k * 4 + 3] = r3[k] ^ 0x80u;
        }
      } else {
        for (int k = 0; k < 4; ++k) {
          const size_t s_ch = s_base + static_cast<size_t>(k);
          for (int lane = 0; lane < 4; ++lane) {
            const size_t o_ch = d_ch + static_cast<size_t>(lane);
            out_tile[k * 4 + lane] =
                (o_ch < out_channels && s_ch < in_channels)
                    ? (static_cast<uint8_t>(src[o_ch * in_channels + s_ch]) ^
                       0x80u)
                    : 0x80u;
          }
        }
      }
      out_tile += 16;
    }
  };

  const size_t outer_units =
      static_cast<size_t>(is_linear_i4o4 ? dst_groups : src_slices);
  const size_t total_elems = out_channels * in_channels;
  auto run_unit = [&](size_t u) {
    if (is_linear_i4o4) {
      const int d = static_cast<int>(u);
      const size_t d_stride =
          static_cast<size_t>(src_slices) * dst_group_size * 16;
      uint8_t* ptr = dst.data() + static_cast<size_t>(d) * d_stride;
      for (int s = 0; s < src_slices; ++s) {
        pack_tile(d, s, ptr);
        ptr += static_cast<size_t>(dst_group_size) * 16;
      }
    } else {
      const int s = static_cast<int>(u);
      const size_t s_stride =
          static_cast<size_t>(dst_groups) * dst_group_size * 16;
      uint8_t* ptr = dst.data() + static_cast<size_t>(s) * s_stride;
      for (int d = 0; d < dst_groups; ++d) {
        pack_tile(d, s, ptr);
        ptr += static_cast<size_t>(dst_group_size) * 16;
      }
    }
  };
  const size_t max_threads = (total_elems >= (512ULL << 10)) ? 8 : 1;
  ParallelFor(
      outer_units, /*min_grain=*/1,
      [&](size_t begin, size_t end) {
        for (size_t u = begin; u < end; ++u) {
          run_unit(u);
        }
      },
      max_threads);
  return true;
}

ml_drift::TensorDescriptor CreatePackedWeightsTensorDescriptor(
    const ml_drift::WeightsDescription& desc,
    const ml_drift::OHWI& weights_shape, size_t elements_count,
    bool is_embedding) {
  const bool is_supported_texture =
      desc.layout == ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  if (desc.IsLinearLayout() || (is_embedding && !is_supported_texture)) {
    ml_drift::TensorDescriptor weights_td(desc.type,
                                          ml_drift::TensorStorageType::kBuffer,
                                          ml_drift::Layout::kLinear);
    // Sub-byte types are packed: the buffer holds bytes, not elements.
    const size_t bits = ml_drift::SizeInBitsOf(desc.type);
    const int linear_shape_c =
        bits > 0 && bits < 8 ? static_cast<int>((elements_count * bits + 7) / 8)
                             : static_cast<int>(elements_count);
    weights_td.SetBHWDCShape(ml_drift::BHWDC(1, 1, 1, 1, linear_shape_c));
    return weights_td;
  }
  ml_drift::DataType texture_type = ml_drift::DataType::kUint32;
  if (desc.type == ml_drift::DataType::kUint4) {
    texture_type = ml_drift::DataType::kUint16;
  } else if (desc.type == ml_drift::DataType::kUint2) {
    texture_type = ml_drift::DataType::kUint8;
  }
  ml_drift::TensorDescriptor weights_td(texture_type,
                                        ml_drift::TensorStorageType::kTexture2D,
                                        ml_drift::Layout::kHW);
  ml_drift::uint2 tex_size = ml_drift::Get2dResourceSize(desc, weights_shape);
  weights_td.SetBHWDCShape(
      ml_drift::BHWDC(1, tex_size.y, tex_size.x / 4, 1, 4));
  return weights_td;
}

MlDriftDummyWeightCreator::MlDriftDummyWeightCreator(
    ml_drift::GpuModelBuilder* model_builder)
    : model_builder_(model_builder) {}

template <typename T>
absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle>
MlDriftDummyWeightCreator::CreateConstantTensor(
    std::vector<int32_t> shape, absl::Span<const T> weights_data,
    ml_drift::DataType mld_float_data_type) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto bhwc, ShapeToBHWC(shape));

  ml_drift::Layout layout = ml_drift::Layout::kLinear;
  if (!shape.empty()) {
    layout = bhwc.b == 1 ? ml_drift::Layout::kHWC : ml_drift::Layout::kBHWC;
  }

  ml_drift::TensorDescriptor weights_td(
      std::is_same_v<T, float> ? mld_float_data_type
                               : ml_drift::DataType::kInt32,
      ml_drift::TensorStorageType::kBuffer, layout);
  weights_td.SetBHWCShape(bhwc);
  if constexpr (std::is_same_v<T, float>) {
    weights_td.UploadData(weights_data.data());
  } else {
    weights_td.UploadDataRaw<T>(weights_data);
  }
  return model_builder_->AddConstantTensor(std::move(weights_td));
}

template <typename T>
absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle>
MlDriftDummyWeightCreator::CreateFcWeights(
    const Tensor<MlDriftMixinTag>& weight_tensor,
    const absl::Span<const T> weights_data,
    ml_drift::DataType mld_float_data_type,
    ml_drift::TensorDescriptor* out_descriptor) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& t_info,
                              graph::GetInfo(weight_tensor.GetRaw()));
  // ABSL_LOG(WARNING) << "CreateFcWeights: weight_name=" << t_info.name
  //                   << " type=" << (int)t_info.type;
  const auto& t_shape = t_info.shape;
  const auto mld_data_type =
      GetMlDriftDataType(t_info.type, mld_float_data_type);
  if (!mld_data_type.ok()) {
    return mld_data_type.status();
  }
  const auto mld_shape = ShapeToOHWI(t_shape);
  if (!mld_shape.ok()) {
    return mld_shape.status();
  }

  auto desc =
      GetFullyConnectedWeightsDesc(model_builder_, *mld_data_type, *mld_shape);

  auto create_and_rearrange = [&](auto& weights)
      -> absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle> {
    weights.shape = *mld_shape;
    using WeightsDataType = typename decltype(weights.data)::value_type;
    const size_t num_elements = weights.shape.DimensionsProduct();
    const size_t padded_num_elements =
        num_elements + XNN_EXTRA_BYTES / sizeof(WeightsDataType);
    const size_t elements_count = static_cast<size_t>(
        ml_drift::GetTotalElementsCountForLayout(desc, *mld_shape));
    const size_t data_size =
        (elements_count *
             static_cast<size_t>(ml_drift::SizeInBitsOf(desc.type)) +
         7) /
        8;
    std::unique_ptr<uint8_t, decltype(&std::free)> data_ptr(
        static_cast<uint8_t*>(std::malloc(data_size)), &std::free);
    if (!data_ptr) {
      return absl::ResourceExhaustedError("Failed to allocate memory.");
    }
    absl::Span<uint8_t> data(data_ptr.get(), data_size);

    bool packed_directly = false;
    if constexpr (std::is_same_v<T, int8_t>) {
      if constexpr (std::is_same_v<WeightsDataType, int8_t>) {
        if (!weights_data.empty() && t_info.type == Type::kI8 &&
            weights_data.size() >= num_elements &&
            (desc.type == ml_drift::DataType::kUint8 ||
             desc.type == ml_drift::DataType::kInt8)) {
          packed_directly = TryDirectRearrangeInt8AsUint8I4O4(
              weights_data.data(), *mld_shape, desc, data);
        }
      }
    }

    if (!packed_directly) {
      if (weights_data.empty()) {
        weights.data.resize(padded_num_elements);
        if constexpr (std::is_floating_point_v<WeightsDataType>) {
          // std::sin is too slow for dummy data generation.
          int i = 0;
          std::generate(weights.data.begin(),
                        weights.data.begin() + num_elements, [&i]() {
                          return static_cast<WeightsDataType>(i++ % 100 - 50) *
                                 0.1f;
                        });
        } else {
          std::iota(weights.data.begin(), weights.data.begin() + num_elements,
                    0);
        }
        std::fill(weights.data.begin() + num_elements, weights.data.end(),
                  static_cast<WeightsDataType>(0));
      } else {
        bool unpacked_subbyte = false;
        if constexpr (std::is_same_v<T, int8_t>) {
          if (t_info.type == Type::kI4) {
            unpacked_subbyte = true;
            weights.data.resize(padded_num_elements, 0);
            const uint8_t* u8_packed =
                reinterpret_cast<const uint8_t*>(weights_data.data());
            for (size_t i = 0; i < num_elements; ++i) {
              size_t byte_idx = i / 2;
              if (byte_idx < weights_data.size()) {
                size_t bit_shift = (i % 2) * 4;
                int8_t q = (u8_packed[byte_idx] >> bit_shift) & 0x0F;
                if (q & 0x08) {
                  q |= 0xF0;  // Sign extend
                }
                weights.data[i] = q;
              }
            }
          } else if (t_info.type == Type::kI2) {
            unpacked_subbyte = true;
            weights.data.resize(padded_num_elements, 0);
            const uint8_t* u8_packed =
                reinterpret_cast<const uint8_t*>(weights_data.data());
            for (size_t i = 0; i < num_elements; ++i) {
              size_t byte_idx = i / 4;
              if (byte_idx < weights_data.size()) {
                size_t bit_shift = (i % 4) * 2;
                int8_t q = (u8_packed[byte_idx] >> bit_shift) & 0x03;
                if (q & 0x02) {
                  q |= 0xFC;  // Sign extend
                }
                weights.data[i] = q;
              }
            }
          }
        }
        if (!unpacked_subbyte) {
          weights.data.resize(padded_num_elements,
                              static_cast<WeightsDataType>(0));
          const size_t copy_elems =
              std::min(weights_data.size(), padded_num_elements);
          if constexpr (std::is_same_v<T, WeightsDataType>) {
            std::memcpy(weights.data.data(), weights_data.data(),
                        copy_elems * sizeof(WeightsDataType));
          } else {
            std::copy_n(weights_data.data(), copy_elems, weights.data.data());
          }
        }
      }

      if constexpr (std::is_same_v<WeightsDataType, int8_t>) {
        if (desc.type == ml_drift::DataType::kUint4 ||
            desc.type == ml_drift::DataType::kInt4) {
          RearrangeWeightsInt8AsUint4(weights, desc, data, 8, 8u);
        } else if (desc.type == ml_drift::DataType::kUint2 ||
                   desc.type == ml_drift::DataType::kInt2) {
          RearrangeWeightsInt8AsUint2(weights, desc, data, 2, 2u);
        } else {
          RearrangeWeightsInt8AsUint8(weights, desc, data, 128, 128u);
        }
      } else {
        RearrangeWeights(weights, desc, data);
      }
    }

    bool is_embedding = (t_info.name == "model.embed_tokens.weight" ||
                         t_info.name == "token_embd.weight");
    if (!is_embedding) {
      for (const std::weak_ptr<graph::Operation>& weak_op : t_info.consumers) {
        if (std::shared_ptr<graph::Operation> consumer = weak_op.lock();
            consumer != nullptr && consumer->GetName() == "EmbeddingLookup") {
          is_embedding = true;
          break;
        }
      }
    }
    ml_drift::TensorDescriptor weights_td = CreatePackedWeightsTensorDescriptor(
        desc, weights.shape, elements_count, is_embedding);
    const size_t mld_mem_size = weights_td.GetMemorySizeInBytes();
    if (mld_mem_size > data_size) {
      std::vector<uint8_t> safe_data(mld_mem_size, 0);
      std::memcpy(safe_data.data(), data_ptr.get(), data_size);
      weights_td.UploadDataRaw<uint8_t>(safe_data);
    } else {
      absl::Span<const uint8_t> upload_span(data_ptr.get(), mld_mem_size);
      weights_td.UploadDataRaw<uint8_t>(upload_span);
    }

    if (out_descriptor != nullptr) {
      *out_descriptor = std::move(weights_td);
      return model_builder_->AddConstantTensor(
          CopyDescriptorWithoutData(*out_descriptor));
    }
    return model_builder_->AddConstantTensor(std::move(weights_td));
  };

  switch (*mld_data_type) {
    case ml_drift::DataType::kFloat32: {
      ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> weights;
      return create_and_rearrange(weights);
    }
    case ml_drift::DataType::kFloat16: {
      ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> weights;
      return create_and_rearrange(weights);
    }
    case ml_drift::DataType::kInt8: {
      ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> weights;
      return create_and_rearrange(weights);
    }
    case ml_drift::DataType::kInt4: {
      ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> weights;
      return create_and_rearrange(weights);
    }
    case ml_drift::DataType::kInt2: {
      ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> weights;
      return create_and_rearrange(weights);
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported data type for fc weight creation: ",
                       ml_drift::ToString(*mld_data_type)));
  }
}

template absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle>
MlDriftDummyWeightCreator::CreateFcWeights<float>(
    const Tensor<MlDriftMixinTag>& weight_tensor,
    const absl::Span<const float> weights_data,
    ml_drift::DataType mld_float_data_type,
    ml_drift::TensorDescriptor* out_descriptor);

template absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle>
MlDriftDummyWeightCreator::CreateFcWeights<int8_t>(
    const Tensor<MlDriftMixinTag>& weight_tensor,
    const absl::Span<const int8_t> weights_data,
    ml_drift::DataType mld_float_data_type,
    ml_drift::TensorDescriptor* out_descriptor);

template absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle>
MlDriftDummyWeightCreator::CreateConstantTensor<float>(
    std::vector<int32_t> shape, const absl::Span<const float> weights_data,
    ml_drift::DataType mld_float_data_type);

template absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle>
MlDriftDummyWeightCreator::CreateConstantTensor<int32_t>(
    std::vector<int32_t> shape, const absl::Span<const int32_t> weights_data,
    ml_drift::DataType mld_float_data_type);

}  // namespace litert::tensor
