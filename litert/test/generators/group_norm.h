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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_GROUP_NORM_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_GROUP_NORM_H_

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/generators/reference_evaluator.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

// Generates a 4D NHWC `odml.group_norm` StableHLO composite operation graph
// along with its decomposed reference subgraph.
template <typename T>
class GroupNorm : public TestGraph {
 public:
  struct Params {
    size_t batch = 1;
    size_t height = 8;
    size_t width = 8;
    size_t channels = 32;
    size_t num_groups = 8;
    float epsilon = 1e-5f;
    bool has_gamma = true;
    bool has_beta = true;
  };

  using Ptr = std::unique_ptr<GroupNorm>;

  static constexpr absl::string_view Name() { return "GroupNorm"; }

  struct GridConfig {
    size_t batch;
    size_t height;
    size_t width;
    size_t channels;
    size_t num_groups;
    float epsilon;
    bool has_gamma;
    bool has_beta;
  };

  // Stratified grid of 16 GroupNorm configurations covering:
  // - Standard multi-group 4D NHWC shapes
  // - Group size (channels / num_groups) variations: 1, 2, 4, 6, 8
  // - Single-group (num_groups = 1) and per-channel (num_groups == channels)
  // - Optional constant gamma and beta weight combinations (1, 2, and 3 inputs)
  // - Single-batch and multi-batch shapes
  static constexpr GridConfig kStratifiedGrid[] = {
      // ─── 1. Standard Multi-Group 4D NHWC Shapes ───
      {1, 8, 8, 32, 8, 1e-5f, true, true},     // 0: G=8, D=4
      {1, 16, 16, 64, 16, 1e-5f, true, true},  // 1: G=16, D=4
      {1, 8, 8, 64, 8, 1e-5f, true, true},     // 2: G=8, D=8
      {1, 4, 4, 128, 32, 1e-5f, true, true},   // 3: G=32, D=4
      {1, 8, 8, 32, 1, 1e-6f, true, true},     // 4: G=1 (single group)

      // ─── 2. Group Size Variations (D = channels / num_groups) ───
      {1, 8, 8, 8, 8, 1e-5f, true, true},      // 5: G=8, D=1
      {1, 6, 8, 16, 16, 1e-5f, true, true},    // 6: G=16, D=1
      {1, 16, 16, 64, 32, 1e-5f, true, true},  // 7: G=32, D=2
      {1, 6, 8, 24, 12, 1e-5f, true, true},    // 8: G=12, D=2
      {1, 4, 6, 48, 8, 1e-5f, true, true},     // 9: G=8, D=6

      // ─── 3. Optional Affine Weight Combinations ───
      {1, 8, 8, 32, 8, 1e-5f, true, false},   // 10: Gamma only (D=4)
      {1, 8, 8, 24, 6, 1e-5f, true, false},   // 11: Gamma only (G=6, D=4)
      {1, 8, 8, 32, 8, 1e-5f, false, false},  // 12: No gamma or beta (D=4)
      {1, 6, 8, 16, 8, 1e-5f, false, false},  // 13: No gamma or beta (D=2)

      // ─── 4. Multi-Batch Shapes (batch > 1) ───
      {2, 8, 8, 32, 8, 1e-5f, true, true},  // 14: Batch-2
      {4, 4, 4, 16, 4, 1e-5f, true, true},  // 15: Batch-4
  };

  template <typename Rng>
  static Expected<GroupNorm::Ptr> Create(Rng& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    const auto& entry = kStratifiedGrid[(sample_counter++) % kGridSize];

    Params params;
    params.batch = entry.batch;
    params.height = entry.height;
    params.width = entry.width;
    params.channels = entry.channels;
    params.num_groups = entry.num_groups;
    params.epsilon = entry.epsilon;
    params.has_gamma = entry.has_gamma;
    params.has_beta = entry.has_beta;

    return Create(std::move(params));
  }

  static Expected<GroupNorm::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<GroupNorm>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kFloatElementwise;
    if constexpr (std::is_same_v<T, float>) {
      spec.relative_tolerance = 1e-3;
      spec.absolute_tolerance = 1e-3;
    } else {
      spec.relative_tolerance = 1e-2;
      spec.absolute_tolerance = 1e-2;
    }
    return spec;
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(1);

    std::array<Layout::Dim, 4> shape = {
        static_cast<Layout::Dim>(params_.batch),
        static_cast<Layout::Dim>(params_.height),
        static_cast<Layout::Dim>(params_.width),
        static_cast<Layout::Dim>(params_.channels)};

    auto builder = data_builder;
    if (!builder.IsFloatDummy()) {
      builder.SetFloatRange(-2.0f, 2.0f);
    }

    LITERT_ASSIGN_OR_RETURN(auto in, SimpleBuffer::Create<T>(shape));
    LITERT_RETURN_IF_ERROR((in.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(in));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return ReferenceEvaluator::Evaluate(Graph(), inputs, outputs);
  }

  GroupNorm(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    constexpr auto kType = litert::tensor::ApiType<T>::value;

    const int b = static_cast<int>(params.batch);
    const int h = static_cast<int>(params.height);
    const int w = static_cast<int>(params.width);
    const int c = static_cast<int>(params.channels);
    const int g = static_cast<int>(params.num_groups);
    const int d = c / g;

    TensorTf input = litert::tensor::Create("input", kType, {b, h, w, c});

    std::vector<float> gamma_data(c);
    std::vector<float> beta_data(c);
    for (int i = 0; i < c; ++i) {
      gamma_data[i] = 1.0f + 0.1f * static_cast<float>((i % 5) - 2);
      beta_data[i] = 0.05f * static_cast<float>((i % 7) - 3);
    }
    TensorTf gamma = litert::tensor::Create("gamma", kType, {c}, gamma_data);
    TensorTf beta = litert::tensor::Create("beta", kType, {c}, beta_data);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("num_groups", g);
      fbb.Float("epsilon", params.epsilon);
      fbb.Int("channel_axis", 3);
      fbb.Int("sub_type", 0);
    });
    fbb.Finish();

    litert::tensor::StableHLOCompositeOptions options{
        .name = "odml.group_norm",
        .composite_attributes = fbb.GetBuffer(),
    };

    auto normalize = [&](auto x) {
      // Reshape [B, H, W, C] -> [B, H * W, G, D] so each group (b, g) is
      // normalized over axes {1, 3} (spatial H*W and within-group channels D).
      auto reshaped = litert::tensor::Reshape(x, {b, h * w, g, d});
      auto mean = litert::tensor::Mean(reshaped, {1, 3}, /*keep_dims=*/true);
      auto centered = litert::tensor::Sub(reshaped, mean);
      auto variance = litert::tensor::Mean(
          litert::tensor::Mul(centered, centered), {1, 3}, /*keep_dims=*/true);
      TensorTf eps =
          litert::tensor::Create("gn_eps", kType, {1}, params.epsilon);
      auto normed = litert::tensor::Mul(
          centered, litert::tensor::Rsqrt(litert::tensor::Add(variance, eps)));
      return litert::tensor::Reshape(normed, {b, h, w, c});
    };

    TensorTf output;
    if (!params.has_gamma) {
      output = litert::tensor::StableHLOComposite(options, normalize, input);
    } else if (!params.has_beta) {
      output = litert::tensor::StableHLOComposite(
          options,
          [&](auto x, auto s) { return litert::tensor::Mul(normalize(x), s); },
          input, gamma);
    } else {
      output = litert::tensor::StableHLOComposite(
          options,
          [&](auto x, auto s, auto b_in) {
            return litert::tensor::Add(litert::tensor::Mul(normalize(x), s),
                                       b_in);
          },
          input, gamma, beta);
    }

    output.SetName("output");
    return litert::testing::SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_GROUP_NORM_H_
