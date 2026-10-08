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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_LAYER_NORM_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_LAYER_NORM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
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

// Generates a 2D, 3D, or 4D LayerNorm StableHLO composite operation graph
// (`odml.group_norm` with `num_groups = 1` and `sub_type = 1`) along with its
// decomposed reference subgraph.
template <typename T>
class LayerNorm : public TestGraph {
 public:
  struct Params {
    std::vector<int32_t> shape = {1, 8, 8, 32};
    float epsilon = 1e-5f;
    bool has_scale = true;
    bool has_bias = true;
  };

  using Ptr = std::unique_ptr<LayerNorm>;

  static constexpr absl::string_view Name() { return "LayerNorm"; }

  struct GridConfig {
    size_t rank;
    int32_t dims[4];
    float epsilon;
    bool has_scale;
    bool has_bias;
  };

  // Stratified grid of 16 LayerNorm configurations covering:
  // - 2D [B, C], 3D [B, S, C], and 4D [B, H, W, C] tensor ranks
  // - Single-batch and multi-batch shapes
  // - Optional constant scale (gamma) and bias (beta) weight combinations
  static constexpr GridConfig kStratifiedGrid[] = {
      // ─── 1. 3D Shapes [B, S, C] ───
      {3, {1, 1, 64, 0}, 1e-5f, true, true},    // 0: S=1, C=64
      {3, {1, 16, 128, 0}, 1e-5f, true, true},  // 1: S=16, C=128
      {3, {1, 64, 256, 0}, 1e-5f, true, true},  // 2: S=64, C=256
      {3, {2, 8, 64, 0}, 1e-5f, true, true},    // 3: Batch-2 3D
      {3, {1, 16, 64, 0}, 1e-6f, true, true},   // 4: S=16, C=64 (eps=1e-6)

      // ─── 2. 4D Shapes [B, H, W, C] ───
      {4, {1, 8, 8, 32}, 1e-5f, true, true},    // 5: 8x8, C=32
      {4, {1, 14, 14, 64}, 1e-6f, true, true},  // 6: 14x14, C=64
      {4, {2, 4, 4, 48}, 1e-5f, true, true},    // 7: Batch-2 4D
      {4, {1, 4, 6, 96}, 1e-5f, true, true},    // 8: 4x6, C=96

      // ─── 3. 2D Shapes [B, C] ───
      {2, {1, 128, 0, 0}, 1e-5f, true, true},  // 9: Batch-1 2D
      {2, {4, 64, 0, 0}, 1e-5f, true, true},   // 10: Batch-4 2D
      {2, {2, 64, 0, 0}, 1e-6f, true, true},   // 11: Batch-2 2D

      // ─── 4. Optional Scale / Bias Combinations ───
      {4, {1, 8, 8, 32}, 1e-5f, true, false},   // 12: 4D scale only
      {4, {1, 8, 8, 32}, 1e-5f, false, false},  // 13: 4D no scale or bias
      {3, {1, 8, 64, 0}, 1e-5f, true, false},   // 14: 3D scale only
      {3, {1, 8, 64, 0}, 1e-5f, false, false},  // 15: 3D no scale or bias
  };

  template <typename Rng>
  static Expected<LayerNorm::Ptr> Create(Rng& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    const auto& entry = kStratifiedGrid[(sample_counter++) % kGridSize];

    Params params;
    params.shape.assign(entry.dims, entry.dims + entry.rank);
    params.epsilon = entry.epsilon;
    params.has_scale = entry.has_scale;
    params.has_bias = entry.has_bias;

    return Create(std::move(params));
  }

  static Expected<LayerNorm::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<LayerNorm>(std::move(params), std::move(model));
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

    std::vector<Layout::Dim> dims(params_.shape.begin(), params_.shape.end());

    auto builder = data_builder;
    if (!builder.IsFloatDummy()) {
      builder.SetFloatRange(-2.0f, 2.0f);
    }

    LITERT_ASSIGN_OR_RETURN(auto in,
                            SimpleBuffer::Create<T>(absl::MakeConstSpan(dims)));
    LITERT_RETURN_IF_ERROR((in.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(in));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return ReferenceEvaluator::Evaluate(Graph(), inputs, outputs);
  }

  LayerNorm(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    if (params.shape.size() < 2 || params.shape.size() > 4) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "LayerNorm shape rank must be between 2 and 4");
    }
    for (int32_t d : params.shape) {
      if (d <= 0) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "LayerNorm dimensions must be positive");
      }
    }
    if (!params.has_scale && params.has_bias) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "LayerNorm bias requires scale to be present");
    }

    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

    int rank = static_cast<int>(params.shape.size());
    int last_axis = rank - 1;
    int channels = params.shape.back();
    float eps = params.epsilon;

    std::vector<int> tensor_shape(params.shape.begin(), params.shape.end());
    TensorTf input = litert::tensor::Create(
        "input", litert::tensor::ApiType<T>::value, tensor_shape);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("num_groups", 1);
      fbb.Float("epsilon", eps);
      fbb.Int("channel_axis", last_axis);
      fbb.Int("sub_type", 1);
    });
    fbb.Finish();
    auto composite_attributes = fbb.GetBuffer();

    std::vector<int> scalar_bcast_shape(rank, 1);
    std::vector<int> channel_bcast_shape(rank, 1);
    channel_bcast_shape.back() = channels;

    auto decompose_core = [last_axis, eps, scalar_bcast_shape](auto x) {
      auto mean = litert::tensor::Mean(x, {last_axis}, /*keep_dims=*/true);
      auto centered = litert::tensor::Sub(x, mean);
      auto sq_diff = litert::tensor::Mul(centered, centered);
      auto variance =
          litert::tensor::Mean(sq_diff, {last_axis}, /*keep_dims=*/true);

      TensorTf eps_tensor = litert::tensor::Create(
          "ln_eps", litert::tensor::ApiType<T>::value, scalar_bcast_shape,
          litert::tensor::OwningCpuBuffer::CopyAs(
              litert::tensor::ApiType<T>::value, std::vector<float>{eps}));
      auto rstd =
          litert::tensor::Rsqrt(litert::tensor::Add(variance, eps_tensor));
      return litert::tensor::Mul(centered, rstd);
    };

    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.group_norm",
        .composite_attributes = composite_attributes,
    };

    std::vector<float> scale_data(channels);
    std::vector<float> bias_data(channels);
    for (int i = 0; i < channels; ++i) {
      scale_data[i] = 1.0f + 0.1f * static_cast<float>((i % 5) - 2);
      bias_data[i] = 0.05f * static_cast<float>((i % 7) - 3);
    }

    TensorTf output;
    if (!params.has_scale) {
      output = litert::tensor::StableHLOComposite(
          composite_options,
          [decompose_core](auto x) { return decompose_core(x); }, input);
    } else if (!params.has_bias) {
      TensorTf scale = litert::tensor::Create(
          "scale", litert::tensor::ApiType<T>::value, {channels},
          litert::tensor::OwningCpuBuffer::CopyAs(
              litert::tensor::ApiType<T>::value, scale_data));
      output = litert::tensor::StableHLOComposite(
          composite_options,
          [decompose_core, channel_bcast_shape](auto x, auto scale_in) {
            auto normed = decompose_core(x);
            auto scale_bcast =
                litert::tensor::Reshape(scale_in, channel_bcast_shape);
            return litert::tensor::Mul(normed, scale_bcast);
          },
          input, scale);
    } else {
      TensorTf scale = litert::tensor::Create(
          "scale", litert::tensor::ApiType<T>::value, {channels},
          litert::tensor::OwningCpuBuffer::CopyAs(
              litert::tensor::ApiType<T>::value, scale_data));
      TensorTf bias = litert::tensor::Create(
          "bias", litert::tensor::ApiType<T>::value, {channels},
          litert::tensor::OwningCpuBuffer::CopyAs(
              litert::tensor::ApiType<T>::value, bias_data));
      output = litert::tensor::StableHLOComposite(
          composite_options,
          [decompose_core, channel_bcast_shape](auto x, auto scale_in,
                                                auto bias_in) {
            auto normed = decompose_core(x);
            auto scale_bcast =
                litert::tensor::Reshape(scale_in, channel_bcast_shape);
            auto bias_bcast =
                litert::tensor::Reshape(bias_in, channel_bcast_shape);
            return litert::tensor::Add(litert::tensor::Mul(normed, scale_bcast),
                                       bias_bcast);
          },
          input, scale, bias);
    }

    output.SetName("output");
    return litert::testing::SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_LAYER_NORM_H_
