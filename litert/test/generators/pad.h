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
#include <cstring>
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
#include "litert/core/model/ops/pad.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"

namespace litert {
namespace testing {

template <typename Rank, typename T, typename MaxTensorSize = SizeC<1024>>
class Pad : public TestGraph {
 private:
  static constexpr size_t kRank = Rank::value;
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;
  static constexpr ElementType kElementType = GetElementType<T>();

  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

 public:
  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    std::array<int32_t, kRank * 2> paddings;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Pad>;

  static constexpr absl::string_view Name() { return "Pad"; }

  template <typename Rng>
  static Expected<Pad::Ptr> Create(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));

    Params p;
    std::copy(std::cbegin(tensor_type.layout.dimensions),
              std::cbegin(tensor_type.layout.dimensions) + kRank,
              std::begin(p.input_shape));

    for (size_t i = 0; i < kRank; ++i) {
      p.paddings[2 * i] = rng() % 3;
      p.paddings[2 * i + 1] = rng() % 3;
    }

    return Create(std::move(p));
  }

  static Expected<Pad::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Pad>(std::move(params), std::move(model));
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

    const T* src = input.data.data();
    T* dst = output.data.data();

    std::array<int32_t, kRank> out_dims;
    for (size_t d = 0; d < kRank; ++d) {
      out_dims[d] = params_.input_shape[d] + params_.paddings[2 * d] +
                    params_.paddings[2 * d + 1];
    }

    auto get_strides = [](const auto& dims) {
      std::array<int32_t, kRank> strides;
      int32_t s = 1;
      for (int i = static_cast<int>(kRank) - 1; i >= 0; --i) {
        strides[i] = s;
        s *= dims[i];
      }
      return strides;
    };

    auto in_strides = get_strides(params_.input_shape);
    auto out_strides = get_strides(out_dims);

    int32_t num_out = 1;
    for (size_t d = 0; d < kRank; ++d) num_out *= out_dims[d];

    for (int32_t i = 0; i < num_out; ++i) {
      std::array<int32_t, kRank> coords;
      int32_t rem = i;
      for (size_t d = 0; d < kRank; ++d) {
        coords[d] = rem / out_strides[d];
        rem %= out_strides[d];
      }

      bool in_bounds = true;
      int32_t src_idx = 0;
      for (size_t d = 0; d < kRank; ++d) {
        int32_t in_c = coords[d] - params_.paddings[2 * d];
        if (in_c < 0 || in_c >= params_.input_shape[d]) {
          in_bounds = false;
          break;
        }
        src_idx += in_c * in_strides[d];
      }
      if (in_bounds) {
        dst[i] = src[src_idx];
      } else {
        dst[i] = static_cast<T>(0.0f);
      }
    }

    return {};
  }

  Pad(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<int32_t> input_dims(params.input_shape.begin(),
                                    params.input_shape.end());

    std::vector<TensorDetails> op_inputs(2);
    op_inputs[0] = TensorDetails{input_dims, LiteRtElementType(kElementType),
                                 std::string(kInputNames[0])};

    std::vector<int32_t> pad_dims = {static_cast<int32_t>(kRank), 2};
    std::vector<int32_t> pad_data(params.paddings.begin(),
                                  params.paddings.end());
    auto pad_buf = MakeOwningBufferRef(pad_data);
    op_inputs[1] = TensorDetails{pad_dims, kLiteRtElementTypeInt32, "paddings",
                                 std::move(pad_buf)};

    LiteRtOpT dummy_op;
    dummy_op.SetOpCode(kLiteRtOpCodeTflPad);

    LiteRtTensorT data_tensor;
    data_tensor.SetType(
        ::MakeRankedTensorType(LiteRtElementType(kElementType), input_dims));

    LiteRtTensorT pad_tensor;
    pad_tensor.SetType(
        ::MakeRankedTensorType(kLiteRtElementTypeInt32, pad_dims));
    auto pad_buf_ptr = MakeOwningBufferRef(pad_data);
    ::SetWeightsFromUnownedBuffer(pad_tensor.Weights(), pad_buf_ptr);

    dummy_op.Inputs().push_back(&data_tensor);
    dummy_op.Inputs().push_back(&pad_tensor);

    std::vector<::litert::internal::Dims> input_shapes = {
        ::litert::internal::Dims(input_dims.begin(), input_dims.end()),
        ::litert::internal::Dims(pad_dims.begin(), pad_dims.end())};
    std::vector<::litert::internal::Dims> output_shapes(1);
    LITERT_RETURN_IF_ERROR(
        ::litert::internal::InferPad(dummy_op, input_shapes, output_shapes));

    std::vector<int32_t> resolved_output_dims(output_shapes[0].begin(),
                                              output_shapes[0].end());

    std::vector<TensorDetails> op_outputs(1);
    op_outputs[0] =
        TensorDetails{resolved_output_dims, LiteRtElementType(kElementType),
                      std::string(kOutputNames[0])};

    return SingleOpModel<kLiteRtOpCodeTflPad>(op_inputs, op_outputs);
  }

  Params params_;
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_PAD_H_
