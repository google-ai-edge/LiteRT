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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MOE_EXPERTS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MOE_EXPERTS_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
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
#include "tflite/types/half.h"

namespace litert::testing {

// Generates a STABLEHLO_COMPOSITE "odml.moe_experts" test graph with both a
// composite entrypoint and a decomposed reference subgraph.
//
// Template parameters:
//   TIn: Activation type (float or tflite::half).
//   TWt: Expert weight storage type (float, tflite::half, int8_t,
//        litert::tensor::int4_t, or litert::tensor::int2_t).
//   ExplicitScales: std::false_type for the 7-input layout (where quantized
//        weights carry per-channel affine quantization parameters on the
//        tensor), or std::true_type for the 10-input layout (where each
//        weight tensor is followed by an explicit float scale tensor).
template <typename TIn, typename TWt = TIn,
          typename ExplicitScales = std::false_type>
class MoeExperts : public TestGraph {
 private:
  static_assert(std::is_same_v<TIn, float> ||
                    std::is_same_v<TIn, tflite::half>,
                "MoeExperts input type must be float or tflite::half.");
  static_assert(std::is_same_v<TWt, float> ||
                    std::is_same_v<TWt, tflite::half> ||
                    std::is_same_v<TWt, int8_t> ||
                    std::is_same_v<TWt, litert::tensor::int4_t> ||
                    std::is_same_v<TWt, litert::tensor::int2_t>,
                "Unsupported MoeExperts weight type.");

  static constexpr litert::tensor::Type kInputType =
      litert::tensor::ApiType<TIn>::value;
  static constexpr litert::tensor::Type kWeightType =
      litert::tensor::ApiType<TWt>::value;
  static constexpr bool kIsQuantizedWeight =
      std::is_same_v<TWt, int8_t> ||
      std::is_same_v<TWt, litert::tensor::int4_t> ||
      std::is_same_v<TWt, litert::tensor::int2_t>;

  struct Params {
    int32_t batch;
    int32_t seq_len;
    int32_t d_in;
    int32_t d_mid;
    int32_t num_experts;
    int32_t top_k;
  };

 public:
  using Traits = TestLogicTraits<TypeList<TIn, TIn, int32_t>, TypeList<TIn>,
                                 Params>;
  using Ptr = std::unique_ptr<MoeExperts>;

  static constexpr absl::string_view Name() { return "MoeExperts"; }

  struct GridConfig {
    int32_t batch;
    int32_t seq_len;
    int32_t d_in;
    int32_t d_mid;
    int32_t num_experts;
    int32_t top_k;
  };

  // Stratified grid of 16 representative MoE workloads.
  // Uses multiples of 32 for d_in and d_mid so packed sub-byte weights
  // (int4, int2) and YNNPACK kernels are always aligned.
  static constexpr GridConfig kStratifiedGrid[] = {
      // ─── 1. Decode Workloads (Single Token, seq_len = 1) ───
      {1, 1, 32, 32, 2, 1},   // 0: Tiny top-1 decode
      {1, 1, 32, 64, 4, 1},   // 1: Small top-1 decode
      {1, 1, 32, 64, 4, 2},   // 2: Small top-2 decode
      {1, 1, 64, 64, 4, 2},   // 3: Medium top-2 decode
      {1, 1, 64, 128, 8, 2},  // 4: 8-expert top-2 decode

      // ─── 2. Short / Chunked Context Workloads ───
      {1, 2, 32, 32, 3, 1},   // 5: 2-token top-1 chunk
      {1, 2, 32, 64, 4, 2},   // 6: 2-token top-2 chunk
      {1, 4, 32, 64, 4, 2},   // 7: 4-token top-2 chunk
      {1, 4, 64, 64, 4, 3},   // 8: 4-token top-3 chunk

      // ─── 3. Prefill Workloads ───
      {1, 8, 32, 32, 2, 1},   // 9: 8-token top-1 prefill
      {1, 8, 32, 64, 4, 2},   // 10: 8-token top-2 prefill
      {1, 16, 32, 32, 2, 2},  // 11: 16-token all-experts prefill
      {1, 16, 32, 64, 4, 1},  // 12: 16-token top-1 prefill

      // ─── 4. Batched Workloads ───
      {2, 1, 32, 64, 4, 2},   // 13: Batch-2 decode
      {2, 2, 32, 64, 4, 2},   // 14: Batch-2 sequence
      {4, 1, 32, 32, 3, 2},   // 15: Batch-4 decode
  };

  template <typename RNG>
  static Expected<MoeExperts::Ptr> Create(RNG& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    const auto& entry = kStratifiedGrid[(sample_counter++) % kGridSize];

    return Create(Params{
        .batch = entry.batch,
        .seq_len = entry.seq_len,
        .d_in = entry.d_in,
        .d_mid = entry.d_mid,
        .num_experts = entry.num_experts,
        .top_k = entry.top_k,
    });
  }

  static Expected<MoeExperts::Ptr> Create(int32_t batch, int32_t seq_len,
                                          int32_t d_in, int32_t d_mid,
                                          int32_t num_experts, int32_t top_k) {
    return Create(Params{
        .batch = batch,
        .seq_len = seq_len,
        .d_in = d_in,
        .d_mid = d_mid,
        .num_experts = num_experts,
        .top_k = top_k,
    });
  }

  static Expected<MoeExperts::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<MoeExperts>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

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
    LITERT_ASSIGN_OR_RETURN(
        auto tokens, SimpleBuffer::Create<TIn>(tokens_dims));
    LITERT_RETURN_IF_ERROR((tokens.template WriteRandom<TIn>(builder, device)));
    inputs.push_back(std::move(tokens));

    const std::vector<int32_t> routing_dims = {params_.batch, params_.seq_len,
                                               params_.top_k};
    LITERT_ASSIGN_OR_RETURN(
        auto routing_weights, SimpleBuffer::Create<TIn>(routing_dims));
    LITERT_RETURN_IF_ERROR(
        (routing_weights.template WriteRandom<TIn>(builder, device)));
    inputs.push_back(std::move(routing_weights));

    LITERT_ASSIGN_OR_RETURN(
        auto expert_indices, SimpleBuffer::Create<int32_t>(routing_dims));
    LITERT_RETURN_IF_ERROR(
        (expert_indices.template WriteRandom<int32_t>(builder, device)));
    // Ensure indices are always within [0, num_experts) even if dummy integer
    // generation was enabled on the caller's builder.
    auto indices_span = expert_indices.Span<int32_t>();
    for (int32_t& idx : indices_span) {
      idx = ((idx % params_.num_experts) + params_.num_experts) %
            params_.num_experts;
    }
    inputs.push_back(std::move(expert_indices));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return ReferenceEvaluator::Evaluate(Graph(), inputs, outputs);
  }

  MoeExperts(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  static constexpr float WeightScale() {
    if constexpr (std::is_same_v<TWt, int8_t>) {
      return 0.02f;
    } else if constexpr (std::is_same_v<TWt, litert::tensor::int4_t>) {
      return 0.04f;
    } else if constexpr (std::is_same_v<TWt, litert::tensor::int2_t>) {
      return 0.15f;
    } else {
      return 1.0f;
    }
  }

  static std::shared_ptr<litert::tensor::Quantization> MakePerChannelQuant(
      int32_t num_channels, float base_scale, uint32_t seed) {
    auto quant =
        std::make_shared<litert::tensor::PerChannelAffineQuantization>();
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
                                          int num_experts, uint32_t seed) {
    const float base_scale = WeightScale();
    std::vector<float> scales(d_out * num_experts);
    for (int d = 0; d < d_out; ++d) {
      for (int e = 0; e < num_experts; ++e) {
        float factor =
            0.75f + 0.05f * static_cast<float>((d + e * 3 + seed) % 11);
        scales[d * num_experts + e] = base_scale * factor;
      }
    }
    return litert::tensor::Create(name, litert::tensor::Type::kFP32,
                                  {d_out, num_experts, 1, 1},
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
        f32_values[i] = 0.1f * std::sin(static_cast<float>(i) + seed);
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
        } else if constexpr (std::is_same_v<TWt, litert::tensor::int2_t>) {
          val = static_cast<int8_t>(static_cast<int32_t>(state % 4u) - 2);
        }
        litert::tensor::internal::SetPackedValue(packed[i / kNum], i % kNum,
                                                 val);
      }
      const int32_t num_channels = shape[0];
      TensorTf w = litert::tensor::Create(name, kWeightType, std::move(shape),
                                          std::move(packed));
      if constexpr (!ExplicitScales::value) {
        w.SetQuantization(
            MakePerChannelQuant(num_channels, WeightScale(), seed));
      }
      return w;
    }
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    const int B = params.batch;
    const int N = params.seq_len;
    const int D_in = params.d_in;
    const int D_mid = params.d_mid;
    const int E = params.num_experts;
    const int K = params.top_k;

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

    auto decompose_core =
        [B, N, D_in, D_mid, E, K](
            TensorTf tokens_in, TensorTf routing_weights_in,
            TensorTf expert_indices_in, TensorTf gate_in,
            const TensorTf* gate_scale_in, TensorTf up_in,
            const TensorTf* up_scale_in, TensorTf down_in,
            const TensorTf* down_scale_in, TensorTf scale_in) -> TensorTf {
      auto tokens_f32 =
          litert::tensor::Cast(tokens_in, litert::tensor::Type::kFP32);
      auto routing_f32 =
          litert::tensor::Cast(routing_weights_in, litert::tensor::Type::kFP32);
      auto scale_f32 =
          litert::tensor::Cast(scale_in, litert::tensor::Type::kFP32);

      // Convert weights from [D_out, E, 1, D_in] to [E, D_out, D_in] in FP32
      // and gather the K selected experts per token -> [B, N, K, D_out, D_in].
      auto gather_expert_weight = [&](TensorTf w, const TensorTf* w_scale,
                                      int d_out, int d_in) {
        TensorTf w_f32;
        if constexpr (kIsQuantizedWeight && !ExplicitScales::value) {
          w_f32 = litert::tensor::Dequantize(w);
        } else {
          w_f32 = litert::tensor::Cast(w, litert::tensor::Type::kFP32);
          if (w_scale != nullptr) {
            w_f32 = litert::tensor::Mul(w_f32, *w_scale);
          }
        }
        auto w_3d = litert::tensor::Reshape(w_f32, {d_out, E, d_in});
        auto w_t = litert::tensor::Transpose(w_3d, {1, 0, 2});
        return litert::tensor::Gather(w_t, expert_indices_in, /*axis=*/0,
                                      /*batch_dims=*/0);
      };
      auto w_gate = gather_expert_weight(gate_in, gate_scale_in, D_mid, D_in);
      auto w_up = gather_expert_weight(up_in, up_scale_in, D_mid, D_in);
      auto w_down = gather_expert_weight(down_in, down_scale_in, D_in, D_mid);

      // Per-expert gated MLP:
      //   x:      [B, N, 1, 1, D_in] (broadcasts across K)
      //   w_gate: [B, N, K, D_mid, D_in]
      //   hidden: Gelu(x @ w_gate^T) * (x @ w_up^T) -> [B, N, K, 1, D_mid]
      //   down:   hidden @ w_down^T                 -> [B, N, K, 1, D_in]
      auto x = litert::tensor::Reshape(tokens_f32, {B, N, 1, 1, D_in});
      auto gate_proj = litert::tensor::BatchMatMul(x, w_gate, /*adj_x=*/false,
                                                   /*adj_y=*/true);
      auto up_proj =
          litert::tensor::BatchMatMul(x, w_up, /*adj_x=*/false, /*adj_y=*/true);
      auto hidden = litert::tensor::Mul(
          litert::tensor::Gelu(gate_proj, /*approximate=*/true), up_proj);
      auto down_proj = litert::tensor::BatchMatMul(
          hidden, w_down, /*adj_x=*/false, /*adj_y=*/true);
      auto expert_out = litert::tensor::Reshape(down_proj, {B, N, K, D_in});

      // Scale each expert output by routing_weights * per_expert_scale and
      // reduce-sum across the K active experts -> [B, N, D_in].
      auto expert_scale = litert::tensor::Gather(
          litert::tensor::Reshape(scale_f32, {E}), expert_indices_in,
          /*axis=*/0, /*batch_dims=*/0);
      auto weights = litert::tensor::Reshape(
          litert::tensor::Mul(routing_f32, expert_scale), {B, N, K, 1});
      auto weighted_out = litert::tensor::Mul(expert_out, weights);
      auto summed = litert::tensor::Sum(weighted_out, /*axes=*/{2},
                                        /*keep_dims=*/false);
      return litert::tensor::Cast(summed, kInputType);
    };

    flexbuffers::Builder fbb;
    fbb.Map([&]() {});
    fbb.Finish();
    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.moe_experts",
        .composite_attributes = fbb.GetBuffer(),
    };

    TensorTf output;
    if constexpr (!ExplicitScales::value) {
      output = litert::tensor::StableHLOComposite(
          composite_options,
          [decompose_core](TensorTf tokens_in, TensorTf routing_weights_in,
                           TensorTf expert_indices_in, TensorTf gate_in,
                           TensorTf up_in, TensorTf down_in,
                           TensorTf scale_in) {
            return decompose_core(tokens_in, routing_weights_in,
                                  expert_indices_in, gate_in, nullptr, up_in,
                                  nullptr, down_in, nullptr, scale_in);
          },
          tokens, routing_weights, expert_indices, gate, up, down, scale);
    } else {
      TensorTf gate_scale =
          MakeExplicitScaleTensor("gate_scale", D_mid, E, /*seed=*/1);
      TensorTf up_scale =
          MakeExplicitScaleTensor("up_scale", D_mid, E, /*seed=*/2);
      TensorTf down_scale =
          MakeExplicitScaleTensor("down_scale", D_in, E, /*seed=*/3);

      output = litert::tensor::StableHLOComposite(
          composite_options,
          [decompose_core](TensorTf tokens_in, TensorTf routing_weights_in,
                           TensorTf expert_indices_in, TensorTf gate_in,
                           TensorTf gate_scale_in, TensorTf up_in,
                           TensorTf up_scale_in, TensorTf down_in,
                           TensorTf down_scale_in, TensorTf scale_in) {
            return decompose_core(
                tokens_in, routing_weights_in, expert_indices_in, gate_in,
                &gate_scale_in, up_in, &up_scale_in, down_in, &down_scale_in,
                scale_in);
          },
          tokens, routing_weights, expert_indices, gate, gate_scale, up,
          up_scale, down, down_scale, scale);
    }

    output.SetName("output");
    return litert::testing::SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MOE_EXPERTS_H_
