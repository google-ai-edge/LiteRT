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
#include "litert/c/litert_common.h"
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
#include "tensor/buffer.h"
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
    if (params.batch == 0 || params.height == 0 || params.width == 0 ||
        params.channels == 0 || params.num_groups == 0) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "GroupNorm dimensions and num_groups must be positive");
    }
    if (params.channels % params.num_groups != 0) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "GroupNorm channels must be divisible by num_groups");
    }
    if (!params.has_gamma && params.has_beta) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "GroupNorm beta requires gamma to be present");
    }

    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

    int b = static_cast<int>(params.batch);
    int h = static_cast<int>(params.height);
    int w = static_cast<int>(params.width);
    int c = static_cast<int>(params.channels);
    int g = static_cast<int>(params.num_groups);
    int d = c / g;
    float eps = params.epsilon;

    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, {b, h, w, c});

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("num_groups", g);
      fbb.Float("epsilon", eps);
      fbb.Int("channel_axis", 3);
      fbb.Int("sub_type", 0);
    });
    fbb.Finish();
    auto composite_attributes = fbb.GetBuffer();

    auto decompose_core = [b, h, w, c, g, d, eps](auto x) {
      // Reshape [B, H, W, C] -> [B, H * W, G, D] so each group (b, g) is
      // normalized over axes {1, 3} (spatial H*W and within-group channels D).
      auto reshaped = litert::tensor::Reshape(x, {b, h * w, g, d});
      auto mean = litert::tensor::Mean(reshaped, {1, 3}, /*keep_dims=*/true);
      auto centered = litert::tensor::Sub(reshaped, mean);
      auto sq_diff = litert::tensor::Mul(centered, centered);
      auto variance = litert::tensor::Mean(sq_diff, {1, 3}, /*keep_dims=*/true);

      TensorTf eps_tensor = litert::tensor::Create(
          "gn_eps", litert::tensor::ApiType<T>::value, {1, 1, 1, 1},
          litert::tensor::OwningCpuBuffer::CopyAs(
              litert::tensor::ApiType<T>::value, std::vector<float>{eps}));
      auto rstd =
          litert::tensor::Rsqrt(litert::tensor::Add(variance, eps_tensor));
      auto normed_4d = litert::tensor::Mul(centered, rstd);
      return litert::tensor::Reshape(normed_4d, {b, h, w, c});
    };

    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.group_norm",
        .composite_attributes = composite_attributes,
    };

    std::vector<float> gamma_data(c);
    std::vector<float> beta_data(c);
    for (int i = 0; i < c; ++i) {
      gamma_data[i] = 1.0f + 0.1f * static_cast<float>((i % 5) - 2);
      beta_data[i] = 0.05f * static_cast<float>((i % 7) - 3);
    }

    TensorTf output;
    if (!params.has_gamma) {
      output = litert::tensor::StableHLOComposite(
          composite_options,
          [decompose_core](auto x) { return decompose_core(x); }, input);
    } else if (!params.has_beta) {
      TensorTf gamma = litert::tensor::Create(
          "gamma", litert::tensor::ApiType<T>::value, {c},
          litert::tensor::OwningCpuBuffer::CopyAs(
              litert::tensor::ApiType<T>::value, gamma_data));
      output = litert::tensor::StableHLOComposite(
          composite_options,
          [decompose_core, c](auto x, auto gamma_in) {
            auto normed = decompose_core(x);
            auto gamma_4d = litert::tensor::Reshape(gamma_in, {1, 1, 1, c});
            return litert::tensor::Mul(normed, gamma_4d);
          },
          input, gamma);
    } else {
      TensorTf gamma = litert::tensor::Create(
          "gamma", litert::tensor::ApiType<T>::value, {c},
          litert::tensor::OwningCpuBuffer::CopyAs(
              litert::tensor::ApiType<T>::value, gamma_data));
      TensorTf beta = litert::tensor::Create(
          "beta", litert::tensor::ApiType<T>::value, {c},
          litert::tensor::OwningCpuBuffer::CopyAs(
              litert::tensor::ApiType<T>::value, beta_data));
      output = litert::tensor::StableHLOComposite(
          composite_options,
          [decompose_core, c](auto x, auto gamma_in, auto beta_in) {
            auto normed = decompose_core(x);
            auto gamma_4d = litert::tensor::Reshape(gamma_in, {1, 1, 1, c});
            auto beta_4d = litert::tensor::Reshape(beta_in, {1, 1, 1, c});
            return litert::tensor::Add(litert::tensor::Mul(normed, gamma_4d),
                                       beta_4d);
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
