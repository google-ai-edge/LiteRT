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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_DEQUANTIZE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_DEQUANTIZE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/types/half.h"

namespace litert::testing {

// Generates single-op `tfl.dequantize` test graphs supporting per-tensor affine
// quantization (`int8_t`, `uint8_t`, `int4_t`), per-channel affine quantization
// (`int8_t`, `int4_t`), and unquantized `FLOAT16 -> FLOAT32` dequantization.
template <typename Rank, typename T, typename PerChannel = std::false_type,
          typename MaxTensorSize = SizeC<1024>>
class Dequantize : public TestGraph {
 public:
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static_assert(Rank::value >= 1, "Dequantize rank must be at least 1");
  static_assert(Rank::value <= LITERT_TENSOR_MAX_RANK,
                "Dequantize rank must not exceed LITERT_TENSOR_MAX_RANK");
  static_assert(std::is_same_v<typename MaxTensorSize::value_type, size_t>);
  static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
                    std::is_same_v<T, litert::tensor::int4_t> ||
                    std::is_same_v<T, tflite::half>,
                "Dequantize only supports int8_t, uint8_t, int4_t, and half");
  static_assert(!PerChannel::value || std::is_same_v<T, int8_t> ||
                    std::is_same_v<T, litert::tensor::int4_t>,
                "Per-channel Dequantize only supports int8_t and int4_t");

  struct Params {
    std::array<Layout::Dim, Rank::value> shape;
    float scale = 0.05f;
    int64_t zero_point = std::is_same_v<T, uint8_t> ? 128 : 0;
    int32_t quantized_dimension = static_cast<int32_t>(Rank::value - 1);
    std::vector<float> scales;
    std::vector<int64_t> zero_points;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<float>, Params>;
  using Ptr = std::unique_ptr<Dequantize>;

  static constexpr absl::string_view Name() {
    if constexpr (kPerChannel) {
      return "DequantizePerChannel";
    } else {
      return "Dequantize";
    }
  }

  template <typename Rng>
  static Expected<Dequantize::Ptr> Create(Rng& rng) {
    LITERT_ASSIGN_OR_RETURN(Params params, GenerateParams(rng));
    return Create(std::move(params));
  }

  static Expected<Dequantize::Ptr> Create(Params params) {
    const size_t max_elements = std::min<size_t>(
        std::numeric_limits<size_t>::max() / sizeof(float),
        static_cast<size_t>(std::numeric_limits<Layout::Dim>::max()));
    size_t num_elements = 1;
    for (Layout::Dim dim : params.shape) {
      if (dim <= 0) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Dequantize shape dimensions must be positive");
      }
      if (num_elements > max_elements / static_cast<size_t>(dim)) {
        return Error(
            kLiteRtStatusErrorInvalidArgument,
            "Dequantize shape exceeds maximum addressable tensor size");
      }
      num_elements *= static_cast<size_t>(dim);
    }
    if constexpr (kPerChannel) {
      if (params.quantized_dimension < 0) {
        params.quantized_dimension += static_cast<int32_t>(kRank);
      }
      if (params.quantized_dimension < 0 ||
          params.quantized_dimension >= static_cast<int32_t>(kRank)) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "quantized_dimension is out of range");
      }
      const size_t num_channels =
          static_cast<size_t>(params.shape[params.quantized_dimension]);
      if (num_channels < 2) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "per-channel Dequantize requires at least 2 channels");
      }
      if (params.scales.empty()) {
        params.scales.assign(num_channels, params.scale);
      } else if (params.scales.size() != num_channels) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "scales size must match quantized_dimension extent");
      }
      if (params.zero_points.empty()) {
        params.zero_points.assign(num_channels, 0);
      } else if (params.zero_points.size() != num_channels) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "zero_points size must match quantized_dimension extent");
      }
    }
    LITERT_ASSIGN_OR_RETURN(LiteRtModelT::Ptr model, BuildGraph(params));
    return std::make_unique<Dequantize>(std::move(params), std::move(model));
  }

  Dequantize(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    LITERT_ASSIGN_OR_RETURN(SimpleBuffer input,
                            SimpleBuffer::Create<T>(params_.shape));
    RandomTensorDataBuilder constrained_builder = data_builder;
    if constexpr (kIsHalf) {
      if (!data_builder.IsFloatDummy()) {
        constrained_builder.SetFloatRange(-100.0f, 100.0f);
      }
    }
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T>(constrained_builder, device)));
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

 private:
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr bool kPerChannel = PerChannel::value;
  static constexpr bool kIsInt4 = std::is_same_v<T, litert::tensor::int4_t>;
  static constexpr bool kIsHalf = std::is_same_v<T, tflite::half>;

  static constexpr ElementType kInElementType = GetElementType<T>();

  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  template <typename Rng>
  static Expected<Params> GenerateParams(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kInElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params params;
    std::copy(std::cbegin(tensor_type.layout.dimensions),
              std::cbegin(tensor_type.layout.dimensions) + kRank,
              std::begin(params.shape));
    if constexpr (kIsInt4) {
      // Ensure an even last dimension so int4 nibble packing has no trailing
      // half-byte ambiguity.
      if (params.shape[kRank - 1] % 2 != 0) {
        if (params.shape[kRank - 1] > 1) {
          --params.shape[kRank - 1];
        } else {
          ++params.shape[kRank - 1];
        }
      }
    }
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.2f);
    if constexpr (kPerChannel) {
      std::uniform_int_distribution<int32_t> qdim_dist(
          0, static_cast<int32_t>(kRank - 1));
      params.quantized_dimension = qdim_dist(rng);
      if (params.shape[params.quantized_dimension] < 2) {
        params.shape[params.quantized_dimension] = 2;
      }
      const size_t num_channels = params.shape[params.quantized_dimension];
      params.scales.resize(num_channels);
      params.zero_points.assign(num_channels, 0);
      for (size_t channel = 0; channel < num_channels; ++channel) {
        params.scales[channel] = scale_dist(rng);
      }
      if constexpr (!kIsInt4) {
        std::uniform_int_distribution<int64_t> zp_dist(-16, 16);
        for (size_t channel = 0; channel < num_channels; ++channel) {
          params.zero_points[channel] = zp_dist(rng);
        }
      }
    } else if constexpr (!kIsHalf) {
      params.scale = scale_dist(rng);
      if constexpr (kIsInt4) {
        params.zero_point = 0;
      } else if constexpr (std::is_same_v<T, uint8_t>) {
        std::uniform_int_distribution<int64_t> zp_dist(110, 146);
        params.zero_point = zp_dist(rng);
      } else {
        std::uniform_int_distribution<int64_t> zp_dist(-16, 16);
        params.zero_point = zp_dist(rng);
      }
    }
    return params;
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.shape.begin(), params.shape.end());

    TensorTf input = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims);
    if constexpr (kPerChannel) {
      input.SetQuantization(
          std::make_shared<litert::tensor::PerChannelAffineQuantization>(
              params.scales, params.zero_points, params.quantized_dimension));
    } else if constexpr (!kIsHalf) {
      input.SetQuantization(
          std::make_shared<litert::tensor::PerChannelAffineQuantization>(
              std::vector<float>{params.scale},
              std::vector<int64_t>{params.zero_point}, 0));
    }

    TensorTf output = litert::tensor::Dequantize(input);
    output.SetName(std::string(kOutputNames[0]));
    return SaveTensorGraph({output});
  }

  Expected<void> ReferenceImpl(
      const typename Traits::ReferenceInputs& inputs,
      const typename Traits::ReferenceOutputs& outputs) const {
    auto [input] = inputs;
    auto [output] = outputs;
    if (input.dimensions != output.dimensions) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "input and output must have the same dimensions");
    }
    if (!std::equal(input.dimensions.begin(), input.dimensions.end(),
                    params_.shape.begin(), params_.shape.end())) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "input dimensions must match generator shape");
    }
    size_t num_elements = 1;
    for (Layout::Dim dim : params_.shape) {
      if (dim <= 0 || num_elements > std::numeric_limits<size_t>::max() /
                                         static_cast<size_t>(dim)) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Dequantize shape element count overflows size_t");
      }
      num_elements *= static_cast<size_t>(dim);
    }
    const size_t expected_input_size =
        kIsInt4 ? (num_elements / 2 + num_elements % 2) : num_elements;
    if (input.data.size() < expected_input_size) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "input buffer is smaller than expected tensor size");
    }
    if (output.data.size() < num_elements) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "output buffer is smaller than expected tensor size");
    }
    if constexpr (kIsHalf) {
      litert::internal::ReferenceCast<float, tflite::half>(
          input.data.data(), num_elements, output.data.data());
    } else {
      auto run_dequantize = [&](const auto* q_data) {
        if constexpr (kPerChannel) {
          std::array<int32_t, kRank> dims;
          std::copy(params_.shape.begin(), params_.shape.end(), dims.begin());
          litert::internal::ReferenceDequantizePerChannel(
              q_data, dims.data(), static_cast<int>(kRank),
              params_.quantized_dimension, params_.scales.data(),
              params_.zero_points.data(), output.data.data());
        } else {
          litert::internal::ReferenceDequantize(
              q_data, num_elements, output.data.data(), params_.scale,
              params_.zero_point);
        }
      };

      if constexpr (kIsInt4) {
        const uint8_t* raw_bytes =
            reinterpret_cast<const uint8_t*>(input.data.data());
        std::vector<int8_t> unpacked(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
          const uint8_t byte = raw_bytes[i / 2];
          unpacked[i] = (i % 2 == 0) ? (static_cast<int8_t>(byte << 4) >> 4)
                                     : (static_cast<int8_t>(byte) >> 4);
        }
        run_dequantize(unpacked.data());
      } else {
        run_dequantize(input.data.data());
      }
    }
    return {};
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_DEQUANTIZE_H_
