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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSPOSE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSPOSE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/c/common.h"
#include "tflite/kernels/internal/portable_tensor_utils.h"

namespace litert {

template <>
inline constexpr ElementType GetElementType<TfLiteFloat16>() {
  return ElementType::Float16;
}

template <>
inline constexpr ElementType GetElementType<TfLiteBFloat16>() {
  return ElementType::BFloat16;
}

template <>
inline constexpr ElementType GetElementType<TfLiteComplex64>() {
  return ElementType::Complex64;
}

}  // namespace litert

namespace litert::testing {

// clang-format off
template <
    typename Rank,
    typename T,
    typename MaxTensorSize = SizeC<1024>
>
// clang-format on
class Transpose : public TestGraph {
 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static_assert(std::is_same_v<typename MaxTensorSize::value_type, size_t>);
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;

  static constexpr LiteRtOpCode kOpCode = kLiteRtOpCodeTflTranspose;
  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr ElementType kPermutationElementType = ElementType::Int32;

  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};
  static constexpr absl::string_view kPermutationName = "perm";

  struct Params {
    std::array<Layout::Dim, kRank> shape;
    std::array<int32_t, kRank> permutation;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Transpose>;

  static constexpr absl::string_view Name() { return "Transpose"; }

  template <typename Rng, typename = std::enable_if_t<
                              !std::is_same_v<std::decay_t<Rng>, Params>>>
  static Expected<Ptr> Create(Rng& rng) {
    LITERT_ASSIGN_OR_RETURN(auto params, GenerateParams(rng));
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Transpose>(std::move(params), std::move(model));
  }

  static Expected<Ptr> Create(const Params& params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Transpose>(params, std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    (void)device;
    (void)data_builder;
    LITERT_ASSIGN_OR_RETURN(auto input, SimpleBuffer::Create<T>(params_.shape));
    LITERT_RETURN_IF_ERROR(FillDeterministicInput(input));
    VarBuffers inputs;
    inputs.push_back(std::move(input));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));
    return ReferenceImpl(ref_inputs, ref_outputs);
  }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;

    spec.comparator_kind = ConformanceComparatorKind::kExact;
    spec.absolute_tolerance = 0.0;
    spec.relative_tolerance = 0.0;
    spec.bucket_tolerance = 0;
    return spec;
  }

  static std::array<Layout::Dim, kRank> OutputShape(const Params& params) {
    std::array<Layout::Dim, kRank> output_shape;
    for (size_t i = 0; i < kRank; ++i) {
      output_shape[i] = params.shape[params.permutation[i]];
    }
    return output_shape;
  }

  explicit Transpose(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  static constexpr ElementType kElementTypeValue = kElementType;
  static constexpr LiteRtOpCode kOperationCode = kOpCode;

 private:
  template <typename Rng>
  static Expected<Params> GenerateParams(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));

    Params params;
    std::copy(std::cbegin(tensor_type.layout.dimensions),
              std::cbegin(tensor_type.layout.dimensions) + kRank,
              std::begin(params.shape));
    std::iota(params.permutation.begin(), params.permutation.end(), 0);
    std::shuffle(params.permutation.begin(), params.permutation.end(), rng);
    return params;
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    LITERT_RETURN_IF_ERROR(Validate(params));

    const std::vector<int32_t> input_dims(params.shape.begin(),
                                          params.shape.end());
    const auto output_shape = OutputShape(params);
    const std::vector<int32_t> output_dims(output_shape.begin(),
                                           output_shape.end());

    TensorDetails input = {input_dims, LiteRtElementType(kElementType),
                           std::string(kInputNames[0])};
    TensorDetails permutation = {
        {static_cast<int32_t>(kRank)},
        kLiteRtElementTypeInt32,
        std::string(kPermutationName),
        MakeBufferRef(params.permutation.begin(), params.permutation.end())};
    TensorDetails output = {output_dims, LiteRtElementType(kElementType),
                            std::string(kOutputNames[0])};

    return SingleOpModel<kOpCode>({std::move(input), std::move(permutation)},
                                  {std::move(output)});
  }

  static Expected<void> Validate(const Params& params) {
    std::array<bool, kRank> seen = {};
    for (size_t i = 0; i < kRank; ++i) {
      if (params.shape[i] <= 0) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Transpose shape dimensions must be positive");
      }
      if (params.permutation[i] < 0 ||
          params.permutation[i] >= static_cast<int32_t>(kRank)) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Transpose permutation is out of range");
      }
      if (seen[static_cast<size_t>(params.permutation[i])]) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Transpose permutation must be unique");
      }
      seen[static_cast<size_t>(params.permutation[i])] = true;
    }
    return {};
  }

  static Expected<void> FillDeterministicInput(SimpleBuffer& input) {
    auto span = input.template Span<T>();
    for (size_t i = 0; i < span.size(); ++i) {
      if constexpr (std::is_same_v<T, bool>) {
        span[i] = (i % 2) == 0;
      } else if constexpr (std::is_same_v<T, int8_t>) {
        span[i] = static_cast<T>(static_cast<int32_t>(i % 63) - 31);
      } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        span[i] = static_cast<T>(static_cast<int64_t>(i % 127) - 63);
      } else if constexpr (std::is_same_v<T, uint8_t>) {
        span[i] = static_cast<T>(i % 251);
      } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
        span[i] = static_cast<T>(i % 251);
      } else if constexpr (std::is_same_v<T, float> ||
                           std::is_same_v<T, double>) {
        span[i] = static_cast<T>(
            static_cast<double>(static_cast<int32_t>(i % 29) - 14) / 4.0);
      } else if constexpr (std::is_same_v<T, TfLiteFloat16>) {
        span[i].data = static_cast<uint16_t>(0x3C00u + (i % 0x03FFu));
      } else if constexpr (std::is_same_v<T, TfLiteBFloat16>) {
        span[i].data = static_cast<uint16_t>(0x3F80u + (i % 0x007Fu));
      } else if constexpr (std::is_same_v<T, TfLiteComplex64>) {
        span[i].re = static_cast<float>(static_cast<int32_t>(i % 17) - 8);
        span[i].im = static_cast<float>(8 - static_cast<int32_t>(i % 17));
      } else {
        static_assert(!sizeof(T), "Unsupported transpose input type");
      }
    }
    return {};
  }

  static std::array<size_t, kRank> MakeStrides(
      absl::Span<const Layout::Dim> dims) {
    std::array<size_t, kRank> strides;
    size_t stride = 1;
    for (size_t i = kRank; i-- > 0;) {
      strides[i] = stride;
      stride *= dims[i];
    }
    return strides;
  }

  Expected<void> ReferenceImpl(
      const typename Traits::ReferenceInputs& inputs,
      const typename Traits::ReferenceOutputs& outputs) const {
    auto [input] = inputs;
    auto [output] = outputs;

    if (input.dimensions.size() != kRank || output.dimensions.size() != kRank) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Transpose rank does not match template rank");
    }

    const auto expected_output_shape = OutputShape(params_);
    if (!std::equal(expected_output_shape.begin(), expected_output_shape.end(),
                    output.dimensions.begin(), output.dimensions.end())) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Transpose output shape does not match permutation");
    }

    const auto input_strides = MakeStrides(input.dimensions);
    const auto output_strides = MakeStrides(output.dimensions);
    std::array<size_t, kRank> input_indices;
    std::array<size_t, kRank> output_indices;

    for (size_t output_offset = 0; output_offset < output.NumElements();
         ++output_offset) {
      size_t remaining = output_offset;
      for (size_t axis = 0; axis < kRank; ++axis) {
        output_indices[axis] = remaining / output_strides[axis];
        remaining %= output_strides[axis];
      }

      for (size_t axis = 0; axis < kRank; ++axis) {
        input_indices[params_.permutation[axis]] = output_indices[axis];
      }

      size_t input_offset = 0;
      for (size_t axis = 0; axis < kRank; ++axis) {
        input_offset += input_indices[axis] * input_strides[axis];
      }
      output.data[output_offset] = input.data[input_offset];
    }
    return {};
  }

  Params params_;
};

template <typename Rank, typename MaxTensorSize = SizeC<1024>>
class TransposeInt4 : public TestGraph {
 private:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static_assert(std::is_same_v<typename MaxTensorSize::value_type, size_t>);
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;

  static constexpr LiteRtOpCode kOpCode = kLiteRtOpCodeTflTranspose;
  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};
  static constexpr absl::string_view kPermutationName = "perm";

 public:
  struct Params {
    std::array<Layout::Dim, kRank> shape;
    std::array<int32_t, kRank> permutation;
  };

  using Ptr = std::unique_ptr<TransposeInt4>;

  static constexpr absl::string_view Name() { return "TransposeInt4"; }

  template <typename Rng, typename = std::enable_if_t<
                              !std::is_same_v<std::decay_t<Rng>, Params>>>
  static Expected<Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, kLiteRtElementTypeInt4> type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));

    Params params;
    std::copy(std::cbegin(tensor_type.layout.dimensions),
              std::cbegin(tensor_type.layout.dimensions) + kRank,
              std::begin(params.shape));
    std::iota(params.permutation.begin(), params.permutation.end(), 0);
    std::shuffle(params.permutation.begin(), params.permutation.end(), rng);
    return Create(params);
  }

  static Expected<Ptr> Create(const Params& params) {
    LITERT_RETURN_IF_ERROR(Validate(params));
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<TransposeInt4>(params, std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    (void)device;
    (void)data_builder;
    LITERT_ASSIGN_OR_RETURN(
        auto input, SimpleBuffer::Create(MakeTensorType(params_.shape)));
    std::vector<int8_t> unpacked(NumElements(params_.shape));
    for (size_t i = 0; i < unpacked.size(); ++i) {
      unpacked[i] = static_cast<int8_t>(static_cast<int32_t>(i % 16) - 8);
    }
    tflite::tensor_utils::PackInt8IntoDenseInt(
        unpacked.data(), unpacked.size(), /*bit_width=*/4,
        input.template Span<int8_t>().data());
    VarBuffers inputs;
    inputs.push_back(std::move(input));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    if (inputs.size() != 1 || outputs.size() != 1) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "TransposeInt4 expects one input and one output");
    }

    const auto& input = inputs[0];
    auto& output = outputs[0];
    std::vector<int8_t> unpacked_input(NumElements(params_.shape));
    std::vector<int8_t> unpacked_output(unpacked_input.size());

    tflite::tensor_utils::UnpackPackedIntToInt8(
        input.template Span<int8_t>().data(), unpacked_input.size(),
        /*bit_width=*/4, unpacked_input.data());
    TransposeUnpacked(absl::MakeConstSpan(unpacked_input),
                      absl::MakeSpan(unpacked_output));
    tflite::tensor_utils::PackInt8IntoDenseInt(
        unpacked_output.data(), unpacked_output.size(), /*bit_width=*/4,
        output.template Span<int8_t>().data());
    return {};
  }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;

    spec.comparator_kind = ConformanceComparatorKind::kExact;
    return spec;
  }

  static std::array<Layout::Dim, kRank> OutputShape(const Params& params) {
    std::array<Layout::Dim, kRank> output_shape;
    for (size_t i = 0; i < kRank; ++i) {
      output_shape[i] = params.shape[params.permutation[i]];
    }
    return output_shape;
  }

  explicit TransposeInt4(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<void> Validate(const Params& params) {
    std::array<bool, kRank> seen = {};
    for (size_t i = 0; i < kRank; ++i) {
      if (params.shape[i] <= 0) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TransposeInt4 shape dimensions must be positive");
      }
      if (params.permutation[i] < 0 ||
          params.permutation[i] >= static_cast<int32_t>(kRank)) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TransposeInt4 permutation is out of range");
      }
      if (seen[static_cast<size_t>(params.permutation[i])]) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TransposeInt4 permutation must be unique");
      }
      seen[static_cast<size_t>(params.permutation[i])] = true;
    }
    return {};
  }

  static RankedTensorType MakeTensorType(
      const std::array<Layout::Dim, kRank>& shape) {
    return RankedTensorType(
        ElementType::Int4,
        Layout(Dimensions(std::cbegin(shape), std::cend(shape))));
  }

  static size_t NumElements(const std::array<Layout::Dim, kRank>& shape) {
    size_t num_elements = 1;
    for (const auto dim : shape) {
      num_elements *= dim;
    }
    return num_elements;
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    const std::vector<int32_t> input_dims(params.shape.begin(),
                                          params.shape.end());
    const auto output_shape = OutputShape(params);
    const std::vector<int32_t> output_dims(output_shape.begin(),
                                           output_shape.end());

    TensorDetails input = {input_dims, kLiteRtElementTypeInt4,
                           std::string(kInputNames[0])};
    TensorDetails permutation = {
        {static_cast<int32_t>(kRank)},
        kLiteRtElementTypeInt32,
        std::string(kPermutationName),
        MakeBufferRef(params.permutation.begin(), params.permutation.end())};
    TensorDetails output = {output_dims, kLiteRtElementTypeInt4,
                            std::string(kOutputNames[0])};
    return SingleOpModel<kOpCode>({std::move(input), std::move(permutation)},
                                  {std::move(output)});
  }

  static std::array<size_t, kRank> MakeStrides(
      absl::Span<const Layout::Dim> dims) {
    std::array<size_t, kRank> strides;
    size_t stride = 1;
    for (size_t i = kRank; i-- > 0;) {
      strides[i] = stride;
      stride *= dims[i];
    }
    return strides;
  }

  void TransposeUnpacked(absl::Span<const int8_t> input,
                         absl::Span<int8_t> output) const {
    const auto output_shape = OutputShape(params_);
    const auto input_strides = MakeStrides(
        absl::MakeConstSpan(params_.shape.data(), params_.shape.size()));
    const auto output_strides = MakeStrides(
        absl::MakeConstSpan(output_shape.data(), output_shape.size()));
    std::array<size_t, kRank> input_indices;
    std::array<size_t, kRank> output_indices;

    for (size_t output_offset = 0; output_offset < output.size();
         ++output_offset) {
      size_t remaining = output_offset;
      for (size_t axis = 0; axis < kRank; ++axis) {
        output_indices[axis] = remaining / output_strides[axis];
        remaining %= output_strides[axis];
      }
      for (size_t axis = 0; axis < kRank; ++axis) {
        input_indices[params_.permutation[axis]] = output_indices[axis];
      }
      size_t input_offset = 0;
      for (size_t axis = 0; axis < kRank; ++axis) {
        input_offset += input_indices[axis] * input_strides[axis];
      }
      output[output_offset] = input[input_offset];
    }
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSPOSE_H_
