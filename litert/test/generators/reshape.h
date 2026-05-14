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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_RESHAPE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_RESHAPE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/reshape.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"

namespace litert {
namespace testing {

template <typename InputRank, typename OutputRank, typename T,
          typename MaxTensorSize = SizeC<1024> >
class Reshape : public TestGraph {
 private:
  static constexpr size_t kInputRank = InputRank::value;
  static constexpr size_t kOutputRank = OutputRank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};
  static constexpr absl::string_view kSignatureName = "default";

 public:
  struct Params {
    std::array<Layout::Dim, kInputRank> input_shape;
    std::array<Layout::Dim, kOutputRank> output_shape;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Reshape>;

  static constexpr absl::string_view Name() { return "Reshape"; }

  template <typename Rng>
  static Expected<Reshape::Ptr> Create(Rng& rng) {
    RandomTensorType<kInputRank, kMaxTensorSize,
                     LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));

    Params p;
    std::copy(std::cbegin(tensor_type.layout.dimensions),
              std::cbegin(tensor_type.layout.dimensions) + kInputRank,
              std::begin(p.input_shape));

    // Strategy 1: Flattening (ND -> 1D)
    if constexpr (kOutputRank == 1) {
      int64_t product = 1;
      for (size_t i = 0; i < kInputRank; ++i) {
        product *= p.input_shape[i];
      }
      p.output_shape[0] = product;
    } else {
      // Strategy 2: General Reshape (using -1 and 1s)
      // We place -1 at a random index and set others to 1.
      // InferReshape will compute the missing dimension to match the input
      // volume.
      int minus_one_idx = rng() % kOutputRank;
      for (size_t i = 0; i < kOutputRank; ++i) {
        if (i == minus_one_idx) {
          p.output_shape[i] = -1;
        } else {
          p.output_shape[i] = 1;
        }
      }
    }

    return Create(std::move(p));
  }

  static Expected<Reshape::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Reshape>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    LITERT_ASSIGN_OR_RETURN(auto input,
                            SimpleBuffer::Create<T>(params_.input_shape));
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T>(data_builder, device)));
    inputs.push_back(std::move(input));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [input] = ref_inputs;
    auto [output] = ref_outputs;

    if (input.NumElements() != output.NumElements()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Input and output size mismatch.");
    }

    const T* src = input.data.data();
    T* dst = output.data.data();
    std::copy(src, src + input.NumElements(), dst);

    return {};
  }

  Reshape(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> input_dims(params.input_shape.begin(),
                                    params.input_shape.end());
    std::vector<int32_t> target_dims(params.output_shape.begin(),
                                     params.output_shape.end());

    std::vector<TensorDetails> op_inputs(2);
    op_inputs[0] = TensorDetails{input_dims, LiteRtElementType(kElementType),
                                 std::string(kInputNames[0])};

    std::vector<int32_t> shape_dims = {static_cast<int32_t>(kOutputRank)};
    std::vector<int32_t> shape_data(target_dims.begin(), target_dims.end());
    auto shape_buf = MakeOwningBufferRef(shape_data);
    op_inputs[1] = TensorDetails{shape_dims, kLiteRtElementTypeInt32, "shape",
                                 std::move(shape_buf)};

    // Leverage shape inference to compute or resolve output shape.
    LiteRtOpT dummy_op;
    dummy_op.SetOpCode(kLiteRtOpCodeTflReshape);

    LiteRtTensorT data_tensor;
    data_tensor.SetType(
        ::MakeRankedTensorType(LiteRtElementType(kElementType), input_dims));

    LiteRtTensorT shape_tensor;
    shape_tensor.SetType(
        ::MakeRankedTensorType(kLiteRtElementTypeInt32, shape_dims));
    auto shape_buf_ptr = MakeOwningBufferRef(shape_data);
    ::SetWeightsFromUnownedBuffer(shape_tensor.Weights(), shape_buf_ptr);

    dummy_op.Inputs().push_back(&data_tensor);
    dummy_op.Inputs().push_back(&shape_tensor);

    std::vector<internal::Dims> input_shapes = {input_dims, shape_dims};
    std::vector<internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(internal::InferReshape(
        dummy_op, absl::MakeSpan(input_shapes), output_shapes));

    std::vector<int32_t> resolved_output_dims(output_shapes[0].begin(),
                                              output_shapes[0].end());

    std::vector<TensorDetails> op_outputs(1);
    op_outputs[0] =
        TensorDetails{resolved_output_dims, LiteRtElementType(kElementType),
                      std::string(kOutputNames[0])};

    return SingleOpModel<kLiteRtOpCodeTflReshape>(op_inputs, op_outputs);
  }

  Params params_;
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_RESHAPE_H_
