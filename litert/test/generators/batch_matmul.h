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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BATCH_MATMUL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BATCH_MATMUL_H_

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
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/matmul.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

template <typename Rank1, typename Rank2, typename T,
          typename AdjX = std::false_type, typename AdjY = std::false_type>
class BatchMatmul : public TestGraph {
  static_assert(std::is_same_v<typename Rank1::value_type, size_t>);
  static constexpr size_t kRank1 = Rank1::value;

  static_assert(std::is_same_v<typename Rank2::value_type, size_t>);
  static constexpr size_t kRank2 = Rank2::value;

  static constexpr size_t kMaxRank = (kRank1 > kRank2) ? kRank1 : kRank2;

  static constexpr bool kAdjX = AdjX::value;
  static constexpr bool kAdjY = AdjY::value;

  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr TensorNames<2> kInputNames = {"lhs", "rhs"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, kRank1> input1_shape;
    std::array<Layout::Dim, kRank2> input2_shape;
    std::array<Layout::Dim, kMaxRank> output_shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T, T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<BatchMatmul>;

  static constexpr absl::string_view Name() { return "BatchMatmul"; }

  template <typename Rng>
  static Expected<BatchMatmul::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 8);
    std::uniform_int_distribution<int> batch_dist(1, 4);
    std::bernoulli_distribution flip_dist(0.5);

    int m = dim_dist(rng);
    int k = dim_dist(rng);
    int n = dim_dist(rng);

    if constexpr (!kAdjX) {
      params.input1_shape[kRank1 - 2] = m;
      params.input1_shape[kRank1 - 1] = k;
    } else {
      params.input1_shape[kRank1 - 2] = k;
      params.input1_shape[kRank1 - 1] = m;
    }

    if constexpr (!kAdjY) {
      params.input2_shape[kRank2 - 2] = k;
      params.input2_shape[kRank2 - 1] = n;
    } else {
      params.input2_shape[kRank2 - 2] = n;
      params.input2_shape[kRank2 - 1] = k;
    }

    // Tracks if any outer dimension was broadcasted to guarantee that the
    // generator actively exercises batch broadcasting kernels.
    bool has_broadcast = false;

    for (size_t i = 1; i <= kMaxRank - 2; ++i) {
      int idx1 = static_cast<int>(kRank1 - 2) - static_cast<int>(i);
      int idx2 = static_cast<int>(kRank2 - 2) - static_cast<int>(i);

      if (idx1 >= 0 && idx2 >= 0) {
        int b = batch_dist(rng);
        bool force_broadcast = (i == 1) && !has_broadcast;

        if (flip_dist(rng) || force_broadcast) {
          if (flip_dist(rng)) {
            params.input2_shape[idx2] = 1;
            params.input1_shape[idx1] = b;
          } else {
            params.input1_shape[idx1] = 1;
            params.input2_shape[idx2] = b;
          }
          has_broadcast = true;
        } else {
          params.input2_shape[idx2] = b;
          params.input1_shape[idx1] = b;
        }
      } else if (idx1 >= 0) {
        params.input1_shape[idx1] = batch_dist(rng);
      } else if (idx2 >= 0) {
        params.input2_shape[idx2] = batch_dist(rng);
      }
    }

    LiteRtOpT op;
    op.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
    auto options = std::make_unique<tflite::BatchMatMulOptionsT>();
    options->adj_x = kAdjX;
    options->adj_y = kAdjY;
    options->asymmetric_quantize_inputs = false;

    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_BatchMatMulOptions;
    tfl_opts.value = options.release();
    litert::internal::SetTflOptions(op, std::move(tfl_opts));

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input1_shape.begin(), params.input1_shape.end()},
        {params.input2_shape.begin(), params.input2_shape.end()}};
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferBatchMatmul(
        op, absl::MakeSpan(input_shapes), output_shapes));

    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<BatchMatmul>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(2);
    RandomTensorDataBuilder modified_data_builder = data_builder;
    modified_data_builder.SetFloatRange(-2.0f, 2.0f);

    LITERT_ASSIGN_OR_RETURN(auto input1,
                            SimpleBuffer::Create<T>(params_.input1_shape));
    LITERT_RETURN_IF_ERROR(
        (input1.template WriteRandom<T>(modified_data_builder, device)));
    inputs.push_back(std::move(input1));

    LITERT_ASSIGN_OR_RETURN(auto input2,
                            SimpleBuffer::Create<T>(params_.input2_shape));
    LITERT_RETURN_IF_ERROR(
        (input2.template WriteRandom<T>(modified_data_builder, device)));
    inputs.push_back(std::move(input2));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [lhs, rhs] = ref_inputs;
    auto [output] = ref_outputs;

    litert::internal::ReferenceBatchMatmul<T>(
        lhs.data.data(),
        reinterpret_cast<const int32_t*>(params_.input1_shape.data()), kRank1,
        rhs.data.data(),
        reinterpret_cast<const int32_t*>(params_.input2_shape.data()), kRank2,
        output.data.data(),
        reinterpret_cast<const int32_t*>(params_.output_shape.data()), kMaxRank,
        kAdjX, kAdjY);

    return {};
  }

  BatchMatmul(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims1(params.input1_shape.begin(),
                               params.input1_shape.end());
    std::vector<int32_t> dims2(params.input2_shape.begin(),
                               params.input2_shape.end());

    TensorTf input1 = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims1);
    TensorTf input2 = litert::tensor::Create(
        std::string(kInputNames[1]), litert::tensor::ApiType<T>::value, dims2);

    TensorTf output =
        litert::tensor::BatchMatMul(input1, input2, kAdjX, kAdjY);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BATCH_MATMUL_H_
