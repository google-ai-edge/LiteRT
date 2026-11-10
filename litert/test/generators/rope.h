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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ROPE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ROPE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_detail.h"
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

// Generates test graphs for the `odml.rope` StableHLO composite op.
//
// Template parameters:
// - `T`: Activation element type (`float` or `tflite::half`).
// - `SplitInputs`:
//   - `std::false_type` (2 inputs -> 1 output, `SplitRoPEConcat`):
//     Inputs: `input` [B, H, S, D], `pos` [B, S] (int32)
//     Outputs: `output` [B, H, S, D]
//   - `std::true_type` (3 inputs -> 2 outputs, `RoPE`):
//     Inputs: `input_l` [B, H, S, D/2], `input_r` [B, H, S, D/2], `pos` [B, S]
//     Outputs: `output_l` [B, H, S, D/2], `output_r` [B, H, S, D/2]
// - `ConstantPos`:
//   - `std::false_type`: `pos` is a dynamic runtime subgraph input.
//   - `std::true_type`: `pos` is a static constant tensor baked into the
//     flatbuffer (matching static-prefill / encoder exports in Gemma 4,
//     Qwen 3.5, LFM2, and SAM2).
template <typename T, typename SplitInputs = std::false_type,
          typename ConstantPos = std::false_type>
class Rope : public TestGraph {
 public:
  static constexpr bool kSplitInputs = SplitInputs::value;
  static constexpr bool kConstantPos = ConstantPos::value;

  // Note: ML Drift's GPU RoPE kernels operate at 4-channel slice granularity
  // (`head_dim >= 8 && head_dim % 8 == 0` so `half_dim % 4 == 0`, and
  // `rotary_half_dim % 4 == 0`).
  struct Params {
    size_t batch = 1;
    size_t seq_len = 1;
    size_t num_heads = 8;
    size_t head_dim = 128;
    float min_timescale = 1.0f;
    float max_timescale = 10000.0f;
    float proportion = 1.0f;
    int32_t pos_offset = 1;
  };

  using InputTypes = std::conditional_t<
      kConstantPos,
      std::conditional_t<kSplitInputs, TypeList<T, T>, TypeList<T>>,
      std::conditional_t<kSplitInputs, TypeList<T, T, int32_t>,
                         TypeList<T, int32_t>>>;
  using OutputTypes =
      std::conditional_t<kSplitInputs, TypeList<T, T>, TypeList<T>>;
  using Traits = TestLogicTraits<InputTypes, OutputTypes, Params>;
  using Ptr = std::unique_ptr<Rope>;

  static constexpr absl::string_view Name() { return "Rope"; }

  struct GridConfig {
    size_t batch;
    size_t seq_len;
    size_t num_heads;
    size_t head_dim;
    float min_timescale;
    float max_timescale;
    float proportion;
    int32_t pos_offset = 1;
  };

  // Stratified grid of 16 representative production LLM RoPE workloads.
  // Covers Llama/Gemma standard, Qwen 3 (1M timescale), Gemma 3 linear scaling
  // (min_timescale = 8, max_timescale = 8M), Embedding Gemma v2 (D=256, 512),
  // Gemma 4 / Qwen 3.5 / Phi partial rotary (proportion < 1.0), multi-batch
  // workloads, and varied decode token positions (`pos_offset`). All entries
  // satisfy `head_dim % 8 == 0` and `(proportion * (head_dim / 2)) % 4 == 0`.
  // ATS test suites should specify iters >= 16 to ensure full coverage.
  static constexpr GridConfig kStratifiedGrid[] = {
      // ─── 1. Standard Llama / Gemma Workloads ───
      {1, 1, 4, 64, 1.0f, 10000.0f, 1.0f, 1},    // 0: Llama/Gemma D=64 decode
      {1, 1, 8, 128, 1.0f, 10000.0f, 1.0f, 16},  // 1: Llama/Gemma D=128 decode
      {1, 16, 8, 128, 1.0f, 10000.0f, 1.0f, 1},  // 2: Llama/Gemma D=128 prefill

      // ─── 2. Qwen 3 Workloads (max_timescale = 1M) ───
      {1, 1, 16, 128, 1.0f, 1000000.0f, 1.0f, 100},  // 3: Qwen 3 D=128 decode
      {1, 32, 16, 128, 1.0f, 1000000.0f, 1.0f, 1},   // 4: Qwen 3 D=128 prefill

      // ─── 3. Gemma 3 Linear Scaling (min=8, max=8M, D=256) ───
      {1, 1, 8, 256, 8.0f, 8000000.0f, 1.0f, 512},  // 5: Gemma 3 D=256 decode
      {1, 32, 8, 256, 8.0f, 8000000.0f, 1.0f, 1},   // 6: Gemma 3 D=256 prefill

      // ─── 4. Embedding Gemma v2 (D=256 and D=512) ───
      {1, 16, 4, 256, 1.0f, 10000.0f, 1.0f, 1},  // 7: Embedding Gemma v2 D=256
      {1, 8, 4, 512, 1.0f, 10000.0f, 1.0f, 1},   // 8: Embedding Gemma v2 D=512

      // ─── 5. Production Partial Rotary (proportion < 1.0) ───
      {1, 1, 8, 256, 1.0f, 1000000.0f, 0.25f, 1024},  // 9: Gemma 4 D=256
      {1, 8, 8, 512, 1.0f, 1000000.0f, 0.25f, 1},     // 10: Gemma 4 D=512
      {1, 16, 8, 256, 1.0f, 10000000.0f, 0.25f, 1},   // 11: Qwen 3.5 D=256
      {1, 4, 8, 80, 1.0f, 10000.0f, 0.4f, 1},         // 12: Phi-2 D=80
      {1, 8, 8, 96, 1.0f, 10000.0f, 0.5f, 1},         // 13: Phi-3 D=96

      // ─── 6. Batched Workloads (batch > 1) ───
      {2, 1, 8, 64, 1.0f, 10000.0f, 1.0f, 4095},  // 14: Batch-2 D=64 decode
      {2, 8, 8, 128, 1.0f, 500000.0f, 1.0f, 1},   // 15: Batch-2 D=128 prefill
  };

  template <typename Rng>
  static Expected<Rope::Ptr> Create(Rng& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    const auto& entry = kStratifiedGrid[(sample_counter++) % kGridSize];

    Params params;
    params.batch = entry.batch;
    params.seq_len = entry.seq_len;
    params.num_heads = entry.num_heads;
    params.head_dim = entry.head_dim;
    params.min_timescale = entry.min_timescale;
    params.max_timescale = entry.max_timescale;
    params.proportion = entry.proportion;
    params.pos_offset = entry.pos_offset;

    return Create(std::move(params));
  }

  static Expected<Rope::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Rope>(std::move(params), std::move(model));
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
    auto builder = data_builder;
    if (!builder.IsFloatDummy()) {
      builder.SetFloatRange(-1.0f, 1.0f);
    }

    if constexpr (!kSplitInputs) {
      inputs.reserve(kConstantPos ? 1 : 2);
      std::array<Layout::Dim, 4> input_shape = {
          static_cast<Layout::Dim>(params_.batch),
          static_cast<Layout::Dim>(params_.num_heads),
          static_cast<Layout::Dim>(params_.seq_len),
          static_cast<Layout::Dim>(params_.head_dim)};
      LITERT_ASSIGN_OR_RETURN(auto input, SimpleBuffer::Create<T>(input_shape));
      LITERT_RETURN_IF_ERROR((input.template WriteRandom<T>(builder, device)));
      inputs.push_back(std::move(input));
    } else {
      inputs.reserve(kConstantPos ? 2 : 3);
      std::array<Layout::Dim, 4> half_shape = {
          static_cast<Layout::Dim>(params_.batch),
          static_cast<Layout::Dim>(params_.num_heads),
          static_cast<Layout::Dim>(params_.seq_len),
          static_cast<Layout::Dim>(params_.head_dim / 2)};
      LITERT_ASSIGN_OR_RETURN(auto input_l,
                              SimpleBuffer::Create<T>(half_shape));
      LITERT_RETURN_IF_ERROR(
          (input_l.template WriteRandom<T>(builder, device)));
      inputs.push_back(std::move(input_l));

      LITERT_ASSIGN_OR_RETURN(auto input_r,
                              SimpleBuffer::Create<T>(half_shape));
      LITERT_RETURN_IF_ERROR(
          (input_r.template WriteRandom<T>(builder, device)));
      inputs.push_back(std::move(input_r));
    }

    if constexpr (!kConstantPos) {
      std::array<Layout::Dim, 2> pos_shape = {
          static_cast<Layout::Dim>(params_.batch),
          static_cast<Layout::Dim>(params_.seq_len)};
      LITERT_ASSIGN_OR_RETURN(auto pos,
                              SimpleBuffer::Create<int32_t>(pos_shape));
      auto pos_span = pos.Span<int32_t>();
      for (size_t b = 0; b < params_.batch; ++b) {
        for (size_t t = 0; t < params_.seq_len; ++t) {
          pos_span[b * params_.seq_len + t] =
              params_.pos_offset + static_cast<int32_t>(t + 7 * b);
        }
      }
      inputs.push_back(std::move(pos));
    }

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return ReferenceEvaluator::EvaluateCompositeReference(Graph(), inputs,
                                                          outputs);
  }

  Rope(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    constexpr auto kType = litert::tensor::ApiType<T>::value;

    int b = static_cast<int>(params.batch);
    int s = static_cast<int>(params.seq_len);
    int num_heads = static_cast<int>(params.num_heads);
    int head_dim = static_cast<int>(params.head_dim);
    if (head_dim < 8 || head_dim % 8 != 0) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          "Rope head_dim must be a positive multiple of 8 for 4-channel slice "
          "alignment");
    }
    int half_dim = head_dim / 2;
    int rotary_half_dim =
        std::clamp(static_cast<int>(params.proportion * half_dim), 0, half_dim);
    if (rotary_half_dim % 4 != 0) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          "Rope rotary_half_dim (proportion * head_dim / 2) must be a multiple "
          "of 4 for 4-channel slice alignment");
    }

    // Precompute RoPE timescale [1, 1, 1, half_dim]. Unrotated channels past
    // `rotary_half_dim` receive +inf so `pos / +inf = 0` (`cos = 1, sin = 0`),
    // keeping a single uniform 2-way slice/concat decomposition graph.
    std::vector<float> timescale_data(half_dim,
                                      std::numeric_limits<float>::infinity());
    float inv_half_dim = 1.0f / static_cast<float>(half_dim);
    for (int i = 0; i < rotary_half_dim; ++i) {
      float fraction = static_cast<float>(i) * inv_half_dim;
      timescale_data[i] =
          params.min_timescale *
          std::pow(params.max_timescale / params.min_timescale, fraction);
    }

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Float("min_timescale", params.min_timescale);
      fbb.Float("max_timescale", params.max_timescale);
      fbb.Float("proportion", params.proportion);
    });
    fbb.Finish();

    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.rope",
        .composite_attributes = fbb.GetBuffer(),
    };

    auto rotate_pair = [&](auto x0, auto x1, auto pos_in) {
      TensorTf timescale =
          litert::tensor::Create("timescale", litert::tensor::Type::kFP32,
                                 {1, 1, 1, half_dim}, timescale_data);
      auto pos_4d = litert::tensor::Reshape(
          litert::tensor::Cast(pos_in, litert::tensor::Type::kFP32),
          {b, 1, s, 1});
      auto sinusoid = litert::tensor::Div(pos_4d, timescale);
      auto cos_val = litert::tensor::Cast(litert::tensor::Cos(sinusoid), kType);
      auto sin_val = litert::tensor::Cast(litert::tensor::Sin(sinusoid), kType);
      auto out0 = litert::tensor::Sub(litert::tensor::Mul(x0, cos_val),
                                      litert::tensor::Mul(x1, sin_val));
      auto out1 = litert::tensor::Add(litert::tensor::Mul(x1, cos_val),
                                      litert::tensor::Mul(x0, sin_val));
      return std::make_tuple(out0, out1);
    };

    std::vector<int32_t> pos_data(b * s);
    for (int bi = 0; bi < b; ++bi) {
      for (int ti = 0; ti < s; ++ti) {
        pos_data[bi * s + ti] = params.pos_offset + ti + 7 * bi;
      }
    }
    TensorTf pos =
        kConstantPos
            ? litert::tensor::Create("pos", litert::tensor::Type::kI32, {b, s},
                                     pos_data)
            : litert::tensor::Create("pos", litert::tensor::Type::kI32, {b, s});

    if constexpr (!kSplitInputs) {
      TensorTf input =
          litert::tensor::Create("input", kType, {b, num_heads, s, head_dim});
      auto decompose = [&](auto x_in, auto pos_in) {
        auto x0 = litert::tensor::Slice(x_in, {0, 0, 0, 0},
                                        {b, num_heads, s, half_dim});
        auto x1 = litert::tensor::Slice(x_in, {0, 0, 0, half_dim},
                                        {b, num_heads, s, half_dim});
        auto [out0, out1] = rotate_pair(x0, x1, pos_in);
        return litert::tensor::Concatenation({out0, out1}, /*axis=*/3);
      };
      auto output = litert::tensor::StableHLOComposite(composite_options,
                                                       decompose, input, pos);
      output.SetName("output");
      return litert::testing::SaveTensorGraph({output});
    } else {
      TensorTf input_l =
          litert::tensor::Create("input_l", kType, {b, num_heads, s, half_dim});
      TensorTf input_r =
          litert::tensor::Create("input_r", kType, {b, num_heads, s, half_dim});
      auto [out_l, out_r] = litert::tensor::StableHLOComposite(
          composite_options, rotate_pair, input_l, input_r, pos);
      out_l.SetName("output_l");
      out_r.SetName("output_r");
      return litert::testing::SaveTensorGraph({out_l, out_r});
    }
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_ROPE_H_
