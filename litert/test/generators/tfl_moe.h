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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TFL_MOE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TFL_MOE_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/generators/reference_evaluator.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/types/half.h"

namespace litert::testing {

// Generates a single `kTfLiteBuiltinCustom` op (`custom_code = "moe"`) graph
// with Flexbuffers `custom_options` matching ML Drift GPU's
// `MoeExpertsOperationParser` and `ir::GetMoeExpertsParser()`.
//
// Supports both:
//   - 7-input non-quantized layout (`TWt` in `{float, tflite::half}`,
//     `weight_type = "fp32"`):
//       [0] src:              [1, seq_len, d_in]             (TIn)
//       [1] top_weights:      [1, seq_len, top_k]            (TIn)
//       [2] top_indices:      [1, seq_len, top_k]            (int32_t)
//       [3] ff_gate_weight:   [d_mid, num_experts, 1, d_in]  (const TWt)
//       [4] ff1_weight:       [d_mid, num_experts, 1, d_in]  (const TWt)
//       [5] linear_weight:    [d_in, num_experts, 1, d_mid]  (const TWt)
//       [6] per_expert_scale: [1, 1, 1, num_experts]         (const TIn)
//   - 10-input quantized layout (`TWt` in `{int8_t, litert::tensor::int4_t}`,
//     `weight_type` in `{"int8", "int4"}`):
//       [0] src:              [1, seq_len, d_in]             (TIn)
//       [1] top_weights:      [1, seq_len, top_k]            (TIn)
//       [2] top_indices:      [1, seq_len, top_k]            (int32_t)
//       [3] ff_gate_weight:   [d_mid, num_experts, 1, d_in]  (const TWt)
//       [4] ff_gate_scale:    [d_mid, num_experts, 1, B]     (const float)
//       [5] ff1_weight:       [d_mid, num_experts, 1, d_in]  (const TWt)
//       [6] ff1_scale:        [d_mid, num_experts, 1, B]     (const float)
//       [7] linear_weight:    [d_in, num_experts, 1, d_mid]  (const TWt)
//       [8] linear_scale:     [d_in, num_experts, 1, B]      (const float)
//       [9] per_expert_scale: [1, 1, 1, num_experts]         (const TIn)
//     where `B = scale_blocks` covers both per-channel (`B = 1`) and blockwise
//     (`B > 1`) scales, and quantized weight tensors carry symmetric
//     per-channel affine quantization metadata (`zero_point == 0`).
template <typename TIn, typename TWt = TIn,
          typename ShareConstantTensors = std::true_type>
class TflMoe : public TestGraph {
 private:
  static_assert(std::is_same_v<TIn, float> || std::is_same_v<TIn, tflite::half>,
                "TflMoe TIn must be float or tflite::half.");
  static_assert(std::is_same_v<TWt, float> ||
                    std::is_same_v<TWt, tflite::half> ||
                    std::is_same_v<TWt, int8_t> ||
                    std::is_same_v<TWt, litert::tensor::int4_t>,
                "TflMoe TWt must be float, tflite::half, int8_t, or int4_t.");

  static constexpr litert::tensor::Type kInputType =
      litert::tensor::ApiType<TIn>::value;
  static constexpr litert::tensor::Type kWeightType =
      litert::tensor::ApiType<TWt>::value;

 public:
  static constexpr bool kIsQuantizedWeight =
      std::is_same_v<TWt, int8_t> ||
      std::is_same_v<TWt, litert::tensor::int4_t>;

 private:
  static constexpr absl::string_view WeightTypeStr() {
    if constexpr (std::is_same_v<TWt, int8_t>) {
      return "int8";
    } else if constexpr (std::is_same_v<TWt, litert::tensor::int4_t>) {
      return "int4";
    } else {
      return "fp32";
    }
  }

 public:
  struct Params {
    int32_t batch;
    int32_t seq_len;
    int32_t d_in;
    int32_t d_mid;
    int32_t num_experts;
    int32_t top_k;
    int32_t scale_blocks = 1;
  };

  using Traits =
      TestLogicTraits<TypeList<TIn, TIn, int32_t>, TypeList<TIn>, Params>;
  using Ptr = std::unique_ptr<TflMoe>;

  static constexpr absl::string_view Name() { return "TflMoe"; }

  // Stratified grid of 16 MoE configurations:
  // - All entries have `batch = 1` (required by ML Drift's `"moe"` parser) and
  //   channel dims (`d_in`, `d_mid`) as multiples of 32.
  // - Covers single-token decode (`seq_len = 1`), multi-token decode
  //   (`seq_len = 2..4`), and prefill (`seq_len = 8..16`).
  // - Covers both unpacked groups (`seq_len * top_k <= num_experts`) and packed
  //   groups (`seq_len * top_k > num_experts`).
  // - Covers both per-channel quantized scales (`scale_blocks = 1`) and
  //   blockwise quantized scales (`scale_blocks = 2, 4`).
  static constexpr Params kStratifiedGrid[] = {
      // ─── 1. Decode Workloads (seq_len = 1, unpacked groups) ───
      {1, 1, 32, 64, 4, 1, 1},
      {1, 1, 32, 64, 4, 2, 1},
      {1, 1, 64, 128, 4, 2, 2},
      {1, 1, 64, 128, 8, 2, 1},
      {1, 1, 32, 64, 8, 4, 1},
      {1, 1, 64, 128, 8, 4, 2},

      // ─── 2. Short / Chunked Context Workloads (seq_len = 2..4) ───
      {1, 2, 32, 64, 4, 2, 1},
      {1, 2, 64, 128, 8, 2, 2},
      {1, 4, 32, 64, 4, 1, 1},
      {1, 4, 32, 64, 4, 2, 2},
      {1, 4, 64, 128, 8, 2, 1},
      {1, 4, 64, 128, 8, 4, 4},

      // ─── 3. Prefill Workloads (seq_len = 8..16, packed groups) ───
      {1, 8, 32, 64, 4, 2, 1},
      {1, 8, 64, 128, 8, 2, 2},
      {1, 16, 32, 64, 4, 2, 1},
      {1, 16, 64, 96, 8, 2, 2},
  };

  template <typename RNG>
  static Expected<TflMoe::Ptr> Create(RNG& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    return Create(kStratifiedGrid[(sample_counter++) % kGridSize]);
  }

  static Expected<TflMoe::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model,
                            BuildGraph(params, /*decompose=*/false));
    return std::make_unique<TflMoe>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<void> ConfigureOptions(Options& options) const override {
    if constexpr (ShareConstantTensors::value) {
      LITERT_ASSIGN_OR_RETURN(auto& gpu_opts, options.GetGpuOptions());
      LITERT_RETURN_IF_ERROR(gpu_opts.EnableConstantTensorSharing(true));
      LITERT_RETURN_IF_ERROR(gpu_opts.SetMadviseOriginalSharedTensors(false));
    }
    return {};
  }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kFloatElementwise;
    if constexpr (std::is_same_v<TIn, float> && std::is_same_v<TWt, float>) {
      spec.relative_tolerance = 1e-3;
      spec.absolute_tolerance = 1e-3;
    } else {
      spec.relative_tolerance = 2e-2;
      spec.absolute_tolerance = 2e-2;
    }
    return spec;
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    RandomTensorDataBuilder builder = data_builder;
    if (!builder.IsFloatDummy()) {
      builder.SetFloatRange(-0.5f, 0.5f);
    }
    builder.SetIntRange(0, params_.num_experts - 1);

    const std::vector<int32_t> tokens_dims = {params_.batch, params_.seq_len,
                                              params_.d_in};
    LITERT_ASSIGN_OR_RETURN(auto tokens,
                            SimpleBuffer::Create<TIn>(tokens_dims));
    LITERT_RETURN_IF_ERROR((tokens.template WriteRandom<TIn>(builder, device)));
    inputs.push_back(std::move(tokens));

    const std::vector<int32_t> routing_dims = {params_.batch, params_.seq_len,
                                               params_.top_k};
    LITERT_ASSIGN_OR_RETURN(auto routing_weights,
                            SimpleBuffer::Create<TIn>(routing_dims));
    LITERT_RETURN_IF_ERROR(
        (routing_weights.template WriteRandom<TIn>(builder, device)));
    inputs.push_back(std::move(routing_weights));

    LITERT_ASSIGN_OR_RETURN(auto expert_indices,
                            SimpleBuffer::Create<int32_t>(routing_dims));
    LITERT_RETURN_IF_ERROR(
        (expert_indices.template WriteRandom<int32_t>(builder, device)));
    // Ensure distinct valid expert indices per token within [0, num_experts).
    auto indices_span = expert_indices.Span<int32_t>();
    const size_t tokens_count =
        static_cast<size_t>(params_.batch) * params_.seq_len;
    for (size_t t = 0; t < tokens_count; ++t) {
      const int32_t base =
          ((indices_span[t * params_.top_k] % params_.num_experts) +
           params_.num_experts) %
          params_.num_experts;
      for (int32_t k = 0; k < params_.top_k; ++k) {
        indices_span[t * params_.top_k + k] = (base + k) % params_.num_experts;
      }
    }
    inputs.push_back(std::move(expert_indices));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto reference_model,
                            BuildGraph(params_, /*decompose=*/true));
    return ReferenceEvaluator::Evaluate(*reference_model, inputs, outputs);
  }

  TflMoe(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  // Base dequantization scale for quantized integer weights (`int8` in
  // [-15, 15], `int4` in [-7, 7]) to keep effective weights around [-0.3, 0.3]
  // and prevent Gelu saturation or FP16 overflow across the 3-layer MLP.
  static constexpr float WeightScale() {
    if constexpr (std::is_same_v<TWt, int8_t>) {
      return 0.02f;
    } else if constexpr (std::is_same_v<TWt, litert::tensor::int4_t>) {
      return 0.04f;
    } else {
      return 1.0f;
    }
  }

  static std::unique_ptr<litert::tensor::Quantization> MakePerChannelQuant(
      int32_t num_channels, float base_scale, uint32_t seed) {
    auto quant =
        std::make_unique<litert::tensor::PerChannelAffineQuantization>();
    quant->scales.resize(num_channels);
    for (int32_t c = 0; c < num_channels; ++c) {
      float factor = 0.75f + 0.05f * static_cast<float>((c + seed * 3) % 11);
      quant->scales[c] = base_scale * factor;
    }
    quant->zero_points.assign(num_channels, 0);
    quant->quantized_dimension = 0;
    return quant;
  }

  static TensorTf MakeExplicitScaleTensor(const char* name, int d_out,
                                          int num_experts, int scale_blocks,
                                          uint32_t seed) {
    const float base_scale = WeightScale();
    std::vector<float> scales(d_out * num_experts * scale_blocks);
    for (int d = 0; d < d_out; ++d) {
      for (int e = 0; e < num_experts; ++e) {
        for (int b = 0; b < scale_blocks; ++b) {
          float factor = 0.75f + 0.05f * static_cast<float>(
                                             (d + e * 3 + b * 5 + seed) % 11);
          scales[(d * num_experts + e) * scale_blocks + b] =
              base_scale * factor;
        }
      }
    }
    return litert::tensor::Create(name, litert::tensor::Type::kFP32,
                                  {d_out, num_experts, 1, scale_blocks},
                                  std::move(scales));
  }

  static TensorTf MakeWeightTensor(const char* name,
                                   litert::tensor::Shape shape, uint32_t seed) {
    size_t num_elements = 1;
    for (int32_t d : shape) {
      num_elements *= static_cast<size_t>(d);
    }
    if constexpr (!kIsQuantizedWeight) {
      std::vector<float> f32_values(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        f32_values[i] = 0.5f * std::sin(static_cast<float>(i) + seed);
      }
      return litert::tensor::Create(name, kWeightType, std::move(shape),
                                    std::move(f32_values));
    } else {
      constexpr size_t kNum = litert::tensor::ApiType<TWt>::kNumElements;
      std::vector<TWt> packed((num_elements + kNum - 1) / kNum);
      uint32_t state = seed * 2654435761u + 1u;
      for (size_t i = 0; i < num_elements; ++i) {
        state = state * 1664525u + 1013904223u;
        int8_t val = 0;
        if constexpr (std::is_same_v<TWt, int8_t>) {
          val = static_cast<int8_t>(static_cast<int32_t>(state % 31u) - 15);
        } else if constexpr (std::is_same_v<TWt, litert::tensor::int4_t>) {
          val = static_cast<int8_t>(static_cast<int32_t>(state % 15u) - 7);
        }
        litert::tensor::internal::SetPackedValue(packed[i / kNum], i % kNum,
                                                 val);
      }
      const int32_t num_channels = shape[0];
      TensorTf w = litert::tensor::Create(name, kWeightType, std::move(shape),
                                          std::move(packed));
      w.SetQuantization(MakePerChannelQuant(num_channels, WeightScale(), seed));
      return w;
    }
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params,
                                                bool decompose = false) {
    const int B = params.batch;
    const int N = params.seq_len;
    const int D_in = params.d_in;
    const int D_mid = params.d_mid;
    const int E = params.num_experts;
    const int K = params.top_k;
    const int S_blocks = params.scale_blocks;

    TensorTf tokens =
        litert::tensor::Create("tokens", kInputType, {B, N, D_in});
    TensorTf routing_weights =
        litert::tensor::Create("routing_weights", kInputType, {B, N, K});
    TensorTf expert_indices = litert::tensor::Create(
        "expert_indices", litert::tensor::Type::kI32, {B, N, K});

    TensorTf gate = MakeWeightTensor("gate", {D_mid, E, 1, D_in}, /*seed=*/1);
    TensorTf up = MakeWeightTensor("up", {D_mid, E, 1, D_in}, /*seed=*/2);
    TensorTf down = MakeWeightTensor("down", {D_in, E, 1, D_mid}, /*seed=*/3);

    std::vector<float> scale_vals(E);
    for (int e = 0; e < E; ++e) {
      scale_vals[e] = 0.5f + 0.1f * static_cast<float>(e);
    }
    TensorTf scale =
        litert::tensor::Create("scale", kInputType, {1, 1, 1, E}, scale_vals);

    std::optional<TensorTf> gate_scale;
    std::optional<TensorTf> up_scale;
    std::optional<TensorTf> down_scale;
    if constexpr (kIsQuantizedWeight) {
      gate_scale = MakeExplicitScaleTensor("gate_scale", D_mid, E, S_blocks,
                                           /*seed=*/1);
      up_scale =
          MakeExplicitScaleTensor("up_scale", D_mid, E, S_blocks, /*seed=*/2);
      down_scale =
          MakeExplicitScaleTensor("down_scale", D_in, E, S_blocks, /*seed=*/3);
    }

    if (!decompose) {
      flexbuffers::Builder fbb;
      fbb.Map([&]() {
        fbb.Int("num_experts", E);
        fbb.Int("num_active_experts", K);
        fbb.Int("model_dim", D_in);
        fbb.Int("hidden_dim", D_mid);
        fbb.String("weight_type", std::string(WeightTypeStr()));
        fbb.String("activation", "gelu_tanh");
        fbb.Bool("renormalized_top_weights", true);
      });
      fbb.Finish();

      std::vector<TensorTf> inputs;
      if constexpr (!kIsQuantizedWeight) {
        inputs = {tokens, routing_weights, expert_indices, gate, up, down,
                  scale};
      } else {
        inputs = {
            tokens, routing_weights, expert_indices, gate,        *gate_scale,
            up,     *up_scale,       down,           *down_scale, scale};
      }

      auto outputs =
          litert::tensor::Custom(absl::MakeSpan(inputs), "moe", fbb.GetBuffer(),
                                 {{B, N, D_in}}, {kInputType});
      outputs[0].SetName("output");
      return litert::testing::SaveTensorGraph(
          {tokens, routing_weights, expert_indices}, {outputs[0]});
    }

    // Decomposed MoE graph using primitive TFLite ops for reference evaluation.
    auto tokens_f32 = litert::tensor::Cast(tokens, litert::tensor::Type::kFP32);
    auto routing_f32 =
        litert::tensor::Cast(routing_weights, litert::tensor::Type::kFP32);
    auto scale_f32 = litert::tensor::Cast(scale, litert::tensor::Type::kFP32);

    auto gather_expert_weight = [&](TensorTf w,
                                    const std::optional<TensorTf>& w_scale,
                                    int d_out, int d_in) {
      TensorTf w_f32 = litert::tensor::Cast(w, litert::tensor::Type::kFP32);
      if constexpr (kIsQuantizedWeight) {
        if (S_blocks == 1) {
          w_f32 = litert::tensor::Mul(w_f32, *w_scale);
        } else {
          auto w_blocked = litert::tensor::Reshape(
              w_f32, {d_out, E, S_blocks, d_in / S_blocks});
          auto s_blocked =
              litert::tensor::Reshape(*w_scale, {d_out, E, S_blocks, 1});
          w_f32 = litert::tensor::Reshape(
              litert::tensor::Mul(w_blocked, s_blocked), {d_out, E, 1, d_in});
        }
      }
      auto w_3d = litert::tensor::Reshape(w_f32, {d_out, E, d_in});
      auto w_t = litert::tensor::Transpose(w_3d, {1, 0, 2});
      return litert::tensor::Gather(w_t, expert_indices, /*axis=*/0,
                                    /*batch_dims=*/0);
    };

    auto w_gate = gather_expert_weight(gate, gate_scale, D_mid, D_in);
    auto w_up = gather_expert_weight(up, up_scale, D_mid, D_in);
    auto w_down = gather_expert_weight(down, down_scale, D_in, D_mid);

    auto x = litert::tensor::Reshape(tokens_f32, {B, N, 1, 1, D_in});
    auto gate_proj =
        litert::tensor::BatchMatMul(x, w_gate, /*adj_x=*/false, /*adj_y=*/true);
    auto up_proj =
        litert::tensor::BatchMatMul(x, w_up, /*adj_x=*/false, /*adj_y=*/true);
    auto hidden = litert::tensor::Mul(
        litert::tensor::Gelu(gate_proj, /*approximate=*/true), up_proj);
    auto down_proj = litert::tensor::BatchMatMul(
        hidden, w_down, /*adj_x=*/false, /*adj_y=*/true);
    auto expert_out = litert::tensor::Reshape(down_proj, {B, N, K, D_in});

    auto expert_scale = litert::tensor::Gather(
        litert::tensor::Reshape(scale_f32, {E}), expert_indices, /*axis=*/0,
        /*batch_dims=*/0);
    auto weights = litert::tensor::Reshape(
        litert::tensor::Mul(routing_f32, expert_scale), {B, N, K, 1});
    auto weighted_out = litert::tensor::Mul(expert_out, weights);
    auto summed = litert::tensor::Sum(weighted_out, /*axes=*/{2},
                                      /*keep_dims=*/false);
    TensorTf output = litert::tensor::Cast(summed, kInputType);
    output.SetName("output");
    return litert::testing::SaveTensorGraph(
        {tokens, routing_weights, expert_indices}, {output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_TFL_MOE_H_
