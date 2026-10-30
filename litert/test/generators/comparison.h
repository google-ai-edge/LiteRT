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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_COMPARISON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_COMPARISON_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/simple_binary.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename Rank1, typename Rank2, typename T, typename OpCode>
class Comparison : public TestGraph {
 private:
  static_assert(std::is_same_v<typename Rank1::value_type, size_t>);
  static constexpr size_t kRank1 = Rank1::value;

  static_assert(std::is_same_v<typename Rank2::value_type, size_t>);
  static constexpr size_t kRank2 = Rank2::value;

  static constexpr size_t kMaxRank = std::max(kRank1, kRank2);

  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr LiteRtOpCode kOpCode = OpCode::value;

  static_assert(kOpCode == kLiteRtOpCodeTflEqual ||
                kOpCode == kLiteRtOpCodeTflNotEqual ||
                kOpCode == kLiteRtOpCodeTflGreater ||
                kOpCode == kLiteRtOpCodeTflGreaterEqual ||
                kOpCode == kLiteRtOpCodeTflLess ||
                kOpCode == kLiteRtOpCodeTflLessEqual);

  static constexpr TensorNames<2> kInputNames = {"lhs", "rhs"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

 public:
  struct Params {
    std::array<Layout::Dim, kRank1> input1_shape;
    std::array<Layout::Dim, kRank2> input2_shape;
    std::array<Layout::Dim, kMaxRank> output_shape;
  };

  using Traits = TestLogicTraits<TypeList<T, T>, TypeList<bool>, Params>;
  using Ptr = std::unique_ptr<Comparison>;

  static constexpr absl::string_view Name() { return "Comparison"; }

  template <typename Rng>
  static Expected<Comparison::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 10);
    std::bernoulli_distribution flip_dist(0.5);

    for (size_t i = 0; i < kRank1; ++i) {
      params.input1_shape[i] = dim_dist(rng);
    }
    for (size_t i = 0; i < kRank2; ++i) {
      params.input2_shape[i] = dim_dist(rng);
    }

    // Enforce broadcast compatibility scanning right-to-left.
    for (size_t i = 1; i <= kMaxRank; ++i) {
      int idx1 = static_cast<int>(kRank1) - static_cast<int>(i);
      int idx2 = static_cast<int>(kRank2) - static_cast<int>(i);
      int out_idx = static_cast<int>(kMaxRank) - static_cast<int>(i);

      if (idx1 >= 0 && idx2 >= 0) {
        if (flip_dist(rng)) {
          if (flip_dist(rng)) {
            params.input2_shape[idx2] = 1;
            params.output_shape[out_idx] = params.input1_shape[idx1];
          } else {
            params.input1_shape[idx1] = 1;
            params.output_shape[out_idx] = params.input2_shape[idx2];
          }
        } else {
          params.input2_shape[idx2] = params.input1_shape[idx1];
          params.output_shape[out_idx] = params.input1_shape[idx1];
        }
      } else if (idx2 >= 0) {
        params.output_shape[out_idx] = params.input2_shape[idx2];
      } else if (idx1 >= 0) {
        params.output_shape[out_idx] = params.input1_shape[idx1];
      }
    }

    return Create(std::move(params));
  }

  static Expected<Comparison::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Comparison>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kExact;
    return spec;
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(2);

    LITERT_ASSIGN_OR_RETURN(auto lhs,
                            SimpleBuffer::Create<T>(params_.input1_shape));
    LITERT_ASSIGN_OR_RETURN(auto rhs,
                            SimpleBuffer::Create<T>(params_.input2_shape));

    LITERT_RETURN_IF_ERROR((lhs.template WriteRandom<T>(data_builder, device)));
    LITERT_RETURN_IF_ERROR((rhs.template WriteRandom<T>(data_builder, device)));

    // Inject identical values at select indices to ensure both true and false
    // comparison outcomes occur in the test data.
    if constexpr (!std::is_same_v<T, bool>) {
      auto lhs_span = lhs.template Span<T>();
      auto rhs_span = rhs.template Span<T>();
      size_t min_elements = std::min(lhs_span.size(), rhs_span.size());
      for (size_t i = 0; i < min_elements; i += 3) {
        rhs_span[i] = lhs_span[i];
      }
    }

    inputs.push_back(std::move(lhs));
    inputs.push_back(std::move(rhs));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [in1, in2] = ref_inputs;
    auto [out] = ref_outputs;

    if constexpr (kOpCode == kLiteRtOpCodeTflEqual) {
      litert::internal::ReferenceEqual(
          in1.data.data(), params_.input1_shape.data(), kRank1, in2.data.data(),
          params_.input2_shape.data(), kRank2, out.data.data(),
          params_.output_shape.data(), kMaxRank);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflNotEqual) {
      litert::internal::ReferenceNotEqual(
          in1.data.data(), params_.input1_shape.data(), kRank1, in2.data.data(),
          params_.input2_shape.data(), kRank2, out.data.data(),
          params_.output_shape.data(), kMaxRank);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflGreater) {
      litert::internal::ReferenceGreater(
          in1.data.data(), params_.input1_shape.data(), kRank1, in2.data.data(),
          params_.input2_shape.data(), kRank2, out.data.data(),
          params_.output_shape.data(), kMaxRank);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflGreaterEqual) {
      litert::internal::ReferenceGreaterEqual(
          in1.data.data(), params_.input1_shape.data(), kRank1, in2.data.data(),
          params_.input2_shape.data(), kRank2, out.data.data(),
          params_.output_shape.data(), kMaxRank);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflLess) {
      litert::internal::ReferenceLess(
          in1.data.data(), params_.input1_shape.data(), kRank1, in2.data.data(),
          params_.input2_shape.data(), kRank2, out.data.data(),
          params_.output_shape.data(), kMaxRank);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflLessEqual) {
      litert::internal::ReferenceLessEqual(
          in1.data.data(), params_.input1_shape.data(), kRank1, in2.data.data(),
          params_.input2_shape.data(), kRank2, out.data.data(),
          params_.output_shape.data(), kMaxRank);
    }

    return {};
  }

  Comparison(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

    std::vector<int32_t> in1_dims(params.input1_shape.begin(),
                                  params.input1_shape.end());
    std::vector<int32_t> in2_dims(params.input2_shape.begin(),
                                  params.input2_shape.end());

    TensorTf a = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value,
        in1_dims);
    TensorTf b = litert::tensor::Create(
        std::string(kInputNames[1]), litert::tensor::ApiType<T>::value,
        in2_dims);

    TensorTf output;
    if constexpr (kOpCode == kLiteRtOpCodeTflEqual) {
      output = litert::tensor::Equal(a, b);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflNotEqual) {
      output = litert::tensor::NotEqual(a, b);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflGreater) {
      output = litert::tensor::Greater(a, b);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflGreaterEqual) {
      output = litert::tensor::GreaterEqual(a, b);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflLess) {
      output = litert::tensor::Less(a, b);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflLessEqual) {
      output = litert::tensor::LessEqual(a, b);
    } else {
      static_assert(kOpCode == kLiteRtOpCodeTflEqual, "Unsupported OpCode");
    }
    output.SetName(std::string(kOutputNames[0]));

    return litert::testing::SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_COMPARISON_H_
