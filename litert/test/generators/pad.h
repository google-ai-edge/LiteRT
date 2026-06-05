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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_PAD_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_PAD_H_

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
#include "litert/core/model/ops/pad.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename Rank, typename T,
          typename OpCode = OpCodeC<kLiteRtOpCodeTflPad>>
class Pad : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr LiteRtOpCode kOpCode = OpCode::value;

  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr TensorNames<2> kInputNames = {"input", "pad_value"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    std::array<int32_t, kRank * 2> paddings;
    std::array<Layout::Dim, kRank> output_shape;
  };

 public:
  using InputTypesT =
      SelectT<std::bool_constant<kOpCode == kLiteRtOpCodeTflPad>, TypeList<T>,
              std::bool_constant<kOpCode == kLiteRtOpCodeTflPadv2>,
              TypeList<T, T>>;
  using Traits = TestLogicTraits<InputTypesT, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Pad>;

  static constexpr absl::string_view Name() {
    return kOpCode == kLiteRtOpCodeTflPad ? "Pad" : "Padv2";
  }

  template <typename Rng>
  static Expected<Pad::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 5);
    std::uniform_int_distribution<int> pad_dist(0, 2);

    for (size_t i = 0; i < kRank; ++i) {
      params.input_shape[i] = dim_dist(rng);
      params.paddings[i * 2] = pad_dist(rng);
      params.paddings[i * 2 + 1] = pad_dist(rng);
      params.output_shape[i] = params.input_shape[i] + params.paddings[i * 2] +
                               params.paddings[i * 2 + 1];
    }

    LiteRtOpT op;
    op.SetOpCode(kOpCode);

    std::vector<int32_t> paddings_vec(params.paddings.begin(),
                                      params.paddings.begin() + kRank * 2);
    auto paddings_buf = MakeOwningBufferRef(paddings_vec);
    LiteRtTensorT paddings_tensor;
    paddings_tensor.SetType(::MakeRankedTensorType(
        kLiteRtElementTypeInt32, {static_cast<int32_t>(kRank), 2}));
    ::SetWeightsFromUnownedBuffer(paddings_tensor.Weights(), paddings_buf);

    op.Inputs().push_back(nullptr);
    op.Inputs().push_back(&paddings_tensor);

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input_shape.begin(), params.input_shape.end()},
        {static_cast<int32_t>(kRank), 2}};
    std::vector<litert::internal::Dims> output_shapes(1);

    if constexpr (kOpCode == kLiteRtOpCodeTflPad) {
      LITERT_RETURN_IF_ERROR(litert::internal::InferPad(
          op, absl::MakeSpan(input_shapes), output_shapes));
    } else {
      LiteRtTensorT pad_val_tensor;
      pad_val_tensor.SetType(::MakeRankedTensorType(GetElementType<T>(), {1}));
      T zero_val = 0;
      auto val_buf = MakeOwningBufferRef(std::vector<T>{zero_val});
      ::SetWeightsFromUnownedBuffer(pad_val_tensor.Weights(), val_buf);
      op.Inputs().push_back(&pad_val_tensor);
      input_shapes.push_back({1});

      LITERT_RETURN_IF_ERROR(litert::internal::InferPadv2(
          op, absl::MakeSpan(input_shapes), output_shapes));
    }

    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Pad>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(kOpCode == kLiteRtOpCodeTflPad ? 1 : 2);
    RandomTensorDataBuilder modified_data_builder = data_builder;
    modified_data_builder.SetFloatRange(-5.0f, 5.0f);

    LITERT_ASSIGN_OR_RETURN(auto input,
                            SimpleBuffer::Create<T>(params_.input_shape));
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T>(modified_data_builder, device)));
    inputs.push_back(std::move(input));

    if constexpr (kOpCode == kLiteRtOpCodeTflPadv2) {
      LITERT_ASSIGN_OR_RETURN(
          auto pad_val, SimpleBuffer::Create<T>(std::array<Layout::Dim, 1>{1}));
      LITERT_RETURN_IF_ERROR(
          (pad_val.template WriteRandom<T>(modified_data_builder, device)));
      inputs.push_back(std::move(pad_val));
    }

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [output] = ref_outputs;

    if constexpr (kOpCode == kLiteRtOpCodeTflPad) {
      auto [in] = ref_inputs;
      litert::internal::ReferencePad<T>(
          in.data.data(),
          reinterpret_cast<const int32_t*>(params_.input_shape.data()),
          params_.paddings.data(), T{0}, kRank, output.data.data());
    } else {
      auto [in, pad_val] = ref_inputs;
      litert::internal::ReferencePad<T>(
          in.data.data(),
          reinterpret_cast<const int32_t*>(params_.input_shape.data()),
          params_.paddings.data(), pad_val.data[0], kRank, output.data.data());
    }

    return {};
  }

  Pad(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());

    TensorTf input = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims);

    TensorTf paddings_tensor = litert::tensor::Create(
        "paddings", litert::tensor::Type::kI32,
        {static_cast<int32_t>(kRank), 2},
        litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kI32>(
            absl::MakeSpan(params.paddings.data(), kRank * 2)));

    TensorTf output;
    if constexpr (kOpCode == kLiteRtOpCodeTflPad) {
      output = litert::tensor::Pad(input, paddings_tensor);
    } else {
      TensorTf pad_val_tensor = litert::tensor::Create(
          std::string(kInputNames[1]), litert::tensor::ApiType<T>::value, {1});
      output = litert::tensor::PadV2(input, paddings_tensor, pad_val_tensor);
    }
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_PAD_H_
