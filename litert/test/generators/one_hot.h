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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ONE_HOT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ONE_HOT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/one_hot.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::testing {

template <>
struct OpDetails<kLiteRtOpCodeTflOneHot> {
  explicit OpDetails(int32_t axis) : axis(axis) {}

  TflOptions MakeTflOptions() const {
    TflOptions res;
    res.type = tflite::BuiltinOptions_OneHotOptions;
    tflite::OneHotOptionsT options;
    options.axis = axis;
    res.Set(std::move(options));
    return res;
  }

  TflOpCodePtr MakeTflCode() const {
    auto code = std::make_unique<TflOpCode>();
    code->builtin_code = tflite::BuiltinOperator_ONE_HOT;
    code->version = 1;
    return code;
  }

 private:
  int32_t axis;
};

template <typename T, typename Rank, typename Axis>
class OneHot : public litert::testing::TestGraph {
 public:
  using Ptr = std::unique_ptr<OneHot>;

  static constexpr absl::string_view Name() { return "OneHot"; }

  struct Params {
    std::vector<int32_t> indices_shape;
    int32_t depth;
    T on_value;
    T off_value;
    int32_t axis;
  };

  static constexpr size_t kRank = Rank::value;
  static constexpr int32_t kAxis = Axis::value;

  static constexpr LiteRtOpCode kOpCode = kLiteRtOpCodeTflOneHot;

  static constexpr ElementType kElementType = GetElementType<T>();

  template <typename Rng>
  static Expected<typename OneHot::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 10);
    std::bernoulli_distribution large_dim_dist(0.5);
    std::uniform_int_distribution<int> large_size_dist(100, 300);

    params.indices_shape.resize(kRank);
    for (size_t i = 0; i < kRank; ++i) {
      if (i == 0 && large_dim_dist(rng)) {
        params.indices_shape[i] = large_size_dist(rng);
      } else {
        params.indices_shape[i] = dim_dist(rng);
      }
    }

    params.depth = dim_dist(rng);
    params.on_value = T(1);
    params.off_value = T(0);
    params.axis = kAxis;

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<OneHot>(std::move(params), std::move(model));
  }

  using Inputs = std::vector<std::vector<uint8_t>>;
  using Outputs = std::vector<std::vector<uint8_t>>;

  static constexpr litert::testing::TensorNames<4> kInputNames = {
      "indices", "depth", "on_value", "off_value"};
  static constexpr litert::testing::TensorNames<1> kOutputNames = {"output"};

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    LITERT_ASSIGN_OR_RETURN(
        auto input, SimpleBuffer::Create<int32_t>(params_.indices_shape));

    auto span = input.template Span<int32_t>();
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int32_t> idx_dist(0, params_.depth - 1);
    for (size_t i = 0; i < span.size(); ++i) {
      span[i] = idx_dist(rng);
    }

    inputs.push_back(std::move(input));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    const auto& indices = inputs[0];
    auto& out = outputs[0];

    const int32_t* indices_data = indices.template Span<int32_t>().data();
    T* output_data = out.template Span<T>().data();

    size_t num_indices = 1;
    for (auto dim : params_.indices_shape) {
      num_indices *= dim;
    }

    int positive_axis = params_.axis;
    if (positive_axis < 0) {
      positive_axis += kRank + 1;
    }

    litert::internal::ReferenceOneHot<T>(
        indices_data, num_indices, params_.depth, params_.on_value,
        params_.off_value, positive_axis, kRank, params_.indices_shape,
        output_data);

    return {};
  }

  bool HasReference() const override { return true; }

  OneHot(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<TensorDetails> inputs(4);
    std::vector<TensorDetails> outputs(1);

    // 0. Indices
    inputs[0] = TensorDetails{params.indices_shape,
                              LiteRtElementType(ElementType::Int32),
                              std::string(kInputNames[0])};

    // 1. Depth
    std::vector<int32_t> depth_data = {params.depth};
    inputs[1] = TensorDetails{{},
                              LiteRtElementType(ElementType::Int32),
                              std::string(kInputNames[1]),
                              MakeOwningBufferRef(depth_data)};

    // 2. On Value
    std::vector<T> on_data = {params.on_value};
    inputs[2] = TensorDetails{{},
                              LiteRtElementType(kElementType),
                              std::string(kInputNames[2]),
                              MakeOwningBufferRef(on_data)};

    // 3. Off Value
    std::vector<T> off_data = {params.off_value};
    inputs[3] = TensorDetails{{},
                              LiteRtElementType(kElementType),
                              std::string(kInputNames[3]),
                              MakeOwningBufferRef(off_data)};

    // Output
    int positive_axis = params.axis;
    if (positive_axis < 0) {
      positive_axis += kRank + 1;
    }
    std::vector<int32_t> output_shape = params.indices_shape;
    output_shape.insert(output_shape.begin() + positive_axis, params.depth);

    outputs[0] = TensorDetails{output_shape, LiteRtElementType(kElementType),
                               std::string(kOutputNames[0])};

    return SingleOpModel<kLiteRtOpCodeTflOneHot>(inputs, outputs, params.axis);
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ONE_HOT_H_
