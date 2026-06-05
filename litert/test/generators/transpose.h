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
#include <random>
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
#include "litert/core/model/model.h"
#include "litert/core/model/ops/transpose.h"
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

template <typename Rank, typename T,
          typename OpCode = OpCodeC<kLiteRtOpCodeTflTranspose>>
class Transpose : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    std::array<int32_t, kRank> perm;
    std::array<Layout::Dim, kRank> output_shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Transpose>;

  static constexpr absl::string_view Name() { return "Transpose"; }

  template <typename Rng>
  static Expected<Transpose::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 6);

    for (size_t i = 0; i < kRank; ++i) {
      params.input_shape[i] = dim_dist(rng);
      params.perm[i] = i;
    }

    std::shuffle(params.perm.begin(), params.perm.begin() + kRank, rng);

    LiteRtOpT op;
    op.SetOpCode(kLiteRtOpCodeTflTranspose);
    auto options = std::make_unique<tflite::TransposeOptionsT>();

    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_TransposeOptions;
    tfl_opts.value = options.release();
    litert::internal::SetTflOptions(op, std::move(tfl_opts));

    std::vector<int32_t> perm_vec(params.perm.begin(),
                                  params.perm.begin() + kRank);
    auto perm_buf = MakeOwningBufferRef(perm_vec);

    LiteRtTensorT perm_tensor;
    perm_tensor.SetType(::MakeRankedTensorType(kLiteRtElementTypeInt32,
                                               {static_cast<int32_t>(kRank)}));
    ::SetWeightsFromUnownedBuffer(perm_tensor.Weights(), perm_buf);
    op.Inputs().push_back(nullptr);
    op.Inputs().push_back(&perm_tensor);

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input_shape.begin(), params.input_shape.end()},
        {static_cast<int32_t>(kRank)}};
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferTranspose(
        op, absl::MakeSpan(input_shapes), output_shapes));

    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Transpose>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(1);
    RandomTensorDataBuilder modified_data_builder = data_builder;
    modified_data_builder.SetFloatRange(-5.0f, 5.0f);

    LITERT_ASSIGN_OR_RETURN(auto input,
                            SimpleBuffer::Create<T>(params_.input_shape));
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T>(modified_data_builder, device)));
    inputs.push_back(std::move(input));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [in] = ref_inputs;
    auto [output] = ref_outputs;

    litert::internal::ReferenceTranspose<T>(
        in.data.data(),
        reinterpret_cast<const int32_t*>(params_.input_shape.data()),
        reinterpret_cast<const int32_t*>(params_.perm.data()), kRank,
        output.data.data());

    return {};
  }

  Transpose(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    std::vector<int> perm(params.perm.begin(), params.perm.begin() + kRank);

    TensorTf input = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims);

    TensorTf output = litert::tensor::Transpose(input, perm);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TRANSPOSE_H_
