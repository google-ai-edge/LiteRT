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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_QKV_NORM_ROPE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_QKV_NORM_ROPE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
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

template <typename T>
class QkvNormRope : public TestGraph {
 public:
  struct Params {
    size_t batch = 1;
    size_t seq_len = 1;
    size_t num_heads = 16;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    float min_timescale = 1.0f;
    float max_timescale = 1000000.0f;
    float proportion = 1.0f;
    float epsilon = 1e-6f;
  };

  using Ptr = std::unique_ptr<QkvNormRope>;

  static constexpr absl::string_view Name() { return "QkvNormRope"; }

  struct GridConfig {
    size_t batch;
    size_t seq_len;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    float min_timescale;
    float max_timescale;
    float epsilon;
  };

  // Stratified grid of 16 representative LLM QKV-norm-rope workloads.
  // Includes single-token decode, chunked, prefill, and batched workloads.
  // ATS test suites should specify iters >= 16 to ensure full coverage.
  static constexpr GridConfig kStratifiedGrid[] = {
      // ─── 1. Decode Workloads (Single Token, seq_len = 1) ───
      {1, 1, 4, 2, 64, 1.0f, 10000.0f, 1e-6f},      // 0: Tiny / toy decode
      {1, 1, 8, 2, 128, 1.0f, 500000.0f, 1e-6f},    // 1: GQA 8:2 decode
      {1, 1, 12, 2, 128, 1.0f, 1000000.0f, 1e-6f},  // 2: GQA 12:2 decode
      {1, 1, 16, 8, 128, 1.0f, 1000000.0f, 1e-6f},  // 3: GQA 16:8 decode
      {1, 1, 32, 8, 128, 1.0f, 500000.0f, 1e-5f},   // 4: GQA 32:8 decode
      {1, 1, 8, 4, 256, 1.0f, 10000.0f, 1e-6f},     // 5: GQA 8:4
                                                    // (head_dim=256) decode

      // ─── 2. Short / Chunked Context Workloads ───
      {1, 8, 8, 2, 128, 1.0f, 1000000.0f, 1e-6f},    // 6: 8-token chunk
      {1, 16, 16, 8, 128, 1.0f, 1000000.0f, 1e-6f},  // 7: 16-token chunk
      {1, 64, 8, 4, 256, 1.0f, 10000.0f, 1e-6f},     // 8: 64-token chunk

      // ─── 3. Prefill Workloads (Extended Sequence) ───
      {1, 128, 16, 8, 128, 1.0f, 1000000.0f, 1e-6f},  // 9: 128-token prefill
      {1, 256, 8, 2, 128, 1.0f, 500000.0f, 1e-6f},    // 10: 256-token prefill
      {1, 512, 16, 4, 128, 1.0f, 1000000.0f, 1e-6f},  // 11: 512-token prefill

      // ─── 4. Batched Workloads (Batch > 1) ───
      {2, 1, 8, 2, 128, 1.0f, 500000.0f, 1e-6f},     // 12: Batch-2 decode
      {2, 16, 16, 8, 128, 1.0f, 1000000.0f, 1e-6f},  // 13: Batch-2 chunked
      {4, 1, 8, 4, 64, 1.0f, 10000.0f, 1e-6f},       // 14: Batch-4 decode
      {4, 1, 16, 8, 128, 1.0f, 1000000.0f, 1e-6f},   // 15: Batch-4 decode
  };

  template <typename Rng>
  static Expected<QkvNormRope::Ptr> Create(Rng& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    const auto& entry = kStratifiedGrid[(sample_counter++) % kGridSize];

    Params params;
    params.batch = entry.batch;
    params.seq_len = entry.seq_len;
    params.num_heads = entry.num_heads;
    params.num_kv_heads = entry.num_kv_heads;
    params.head_dim = entry.head_dim;
    params.min_timescale = entry.min_timescale;
    params.max_timescale = entry.max_timescale;
    params.epsilon = entry.epsilon;

    return Create(std::move(params));
  }

  static Expected<QkvNormRope::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<QkvNormRope>(std::move(params), std::move(model));
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
    inputs.reserve(4);

    size_t total_dim =
        (params_.num_heads + 2 * params_.num_kv_heads) * params_.head_dim;
    std::array<Layout::Dim, 4> qkv_shape = {
        static_cast<Layout::Dim>(params_.batch), 1,
        static_cast<Layout::Dim>(params_.seq_len),
        static_cast<Layout::Dim>(total_dim)};

    auto builder = data_builder;
    if (!builder.IsFloatDummy()) {
      builder.SetFloatRange(-1.0f, 1.0f);
    }

    LITERT_ASSIGN_OR_RETURN(auto qkv, SimpleBuffer::Create<T>(qkv_shape));
    LITERT_RETURN_IF_ERROR((qkv.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(qkv));

    std::array<Layout::Dim, 2> pos_shape = {
        static_cast<Layout::Dim>(params_.batch),
        static_cast<Layout::Dim>(params_.seq_len)};
    LITERT_ASSIGN_OR_RETURN(auto pos, SimpleBuffer::Create<int32_t>(pos_shape));
    auto pos_span = pos.Span<int32_t>();
    for (size_t b = 0; b < params_.batch; ++b) {
      for (size_t t = 0; t < params_.seq_len; ++t) {
        pos_span[b * params_.seq_len + t] = static_cast<int32_t>(t);
      }
    }
    inputs.push_back(std::move(pos));

    std::array<Layout::Dim, 1> weight_shape = {
        static_cast<Layout::Dim>(params_.head_dim)};
    auto weight_builder = data_builder;
    if (!weight_builder.IsFloatDummy()) {
      weight_builder.SetFloatRange(0.5f, 1.5f);
    }

    LITERT_ASSIGN_OR_RETURN(auto q_w, SimpleBuffer::Create<T>(weight_shape));
    LITERT_RETURN_IF_ERROR(
        (q_w.template WriteRandom<T>(weight_builder, device)));
    inputs.push_back(std::move(q_w));

    LITERT_ASSIGN_OR_RETURN(auto k_w, SimpleBuffer::Create<T>(weight_shape));
    LITERT_RETURN_IF_ERROR(
        (k_w.template WriteRandom<T>(weight_builder, device)));
    inputs.push_back(std::move(k_w));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return ReferenceEvaluator::EvaluateCompositeReference(Graph(), inputs,
                                                           outputs);
  }

  QkvNormRope(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

    int b = static_cast<int>(params.batch);
    int s = static_cast<int>(params.seq_len);
    int num_heads = static_cast<int>(params.num_heads);
    int num_kv_heads = static_cast<int>(params.num_kv_heads);
    int head_dim = static_cast<int>(params.head_dim);
    int total_dim = (num_heads + 2 * num_kv_heads) * head_dim;

    TensorTf qkv = litert::tensor::Create(
        "qkv", litert::tensor::ApiType<T>::value, {b, 1, s, total_dim});
    TensorTf position = litert::tensor::Create(
        "position", litert::tensor::ApiType<int32_t>::value, {b, s});
    TensorTf q_weight = litert::tensor::Create(
        "q_weight", litert::tensor::ApiType<T>::value, {head_dim});
    TensorTf k_weight = litert::tensor::Create(
        "k_weight", litert::tensor::ApiType<T>::value, {head_dim});

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("num_heads", num_heads);
      fbb.Int("num_kv_heads", num_kv_heads);
      fbb.Int("head_dim", head_dim);
      fbb.Float("min_timescale", params.min_timescale);
      fbb.Float("max_timescale", params.max_timescale);
      fbb.Float("proportion", params.proportion);
      fbb.Float("epsilon", params.epsilon);
    });
    fbb.Finish();
    auto composite_attributes = fbb.GetBuffer();

    auto decompose = [b, s, num_heads, num_kv_heads, head_dim,
                      &params](auto qkv_in, auto pos_in, auto q_w_in,
                               auto k_w_in) {
      int q_channels = num_heads * head_dim;
      int kv_channels = num_kv_heads * head_dim;
      int half_dim = head_dim / 2;

      // 1. Slice Q, K, V from fused QKV input [B, 1, S, total_dim].
      auto q_slice = litert::tensor::Slice(
          qkv_in, {0, 0, 0, 0}, {b, 1, s, q_channels});
      auto k_slice = litert::tensor::Slice(
          qkv_in, {0, 0, 0, q_channels}, {b, 1, s, kv_channels});
      auto v_slice = litert::tensor::Slice(
          qkv_in, {0, 0, 0, q_channels + kv_channels}, {b, 1, s, kv_channels});

      // 2. Reshape and Transpose into [B, num_heads, S, head_dim].
      auto q_reshaped = litert::tensor::Reshape(
          q_slice, {b, s, num_heads, head_dim});
      auto q_transposed = litert::tensor::Transpose(
          q_reshaped, {0, 2, 1, 3});

      auto k_reshaped = litert::tensor::Reshape(
          k_slice, {b, s, num_kv_heads, head_dim});
      auto k_transposed = litert::tensor::Transpose(
          k_reshaped, {0, 2, 1, 3});

      auto v_reshaped = litert::tensor::Reshape(
          v_slice, {b, s, num_kv_heads, head_dim});
      auto v_out = litert::tensor::Transpose(
          v_reshaped, {0, 2, 1, 3});

      // 3. RMSNorm on Q and K along head_dim (axis 3).
      TensorTf eps_tensor = litert::tensor::Create(
          "rms_eps", litert::tensor::ApiType<T>::value, {1, 1, 1, 1},
          litert::tensor::OwningCpuBuffer::CopyAs(
              litert::tensor::ApiType<T>::value,
              std::vector<float>{params.epsilon}));

      auto q_w_4d = litert::tensor::Reshape(q_w_in, {1, 1, 1, head_dim});
      auto q_sq = litert::tensor::Mul(q_transposed, q_transposed);
      auto q_var = litert::tensor::Mean(q_sq, {3}, /*keep_dims=*/true);
      auto q_rstd =
          litert::tensor::Rsqrt(litert::tensor::Add(q_var, eps_tensor));
      auto q_normed = litert::tensor::Mul(
          litert::tensor::Mul(q_transposed, q_rstd), q_w_4d);

      auto k_w_4d = litert::tensor::Reshape(k_w_in, {1, 1, 1, head_dim});
      auto k_sq = litert::tensor::Mul(k_transposed, k_transposed);
      auto k_var = litert::tensor::Mean(k_sq, {3}, /*keep_dims=*/true);
      auto k_rstd =
          litert::tensor::Rsqrt(litert::tensor::Add(k_var, eps_tensor));
      auto k_normed = litert::tensor::Mul(
          litert::tensor::Mul(k_transposed, k_rstd), k_w_4d);

      // 4. RoPE on Q and K.
      auto q0 = litert::tensor::Slice(
          q_normed, {0, 0, 0, 0}, {b, num_heads, s, half_dim});
      auto q1 = litert::tensor::Slice(
          q_normed, {0, 0, 0, half_dim}, {b, num_heads, s, half_dim});

      auto k0 = litert::tensor::Slice(
          k_normed, {0, 0, 0, 0}, {b, num_kv_heads, s, half_dim});
      auto k1 = litert::tensor::Slice(
          k_normed, {0, 0, 0, half_dim}, {b, num_kv_heads, s, half_dim});

      std::vector<float> timescale_data(half_dim);
      float inv_dst_ch = 1.0f / static_cast<float>(head_dim);
      for (int i = 0; i < half_dim; ++i) {
        float fraction = 2.0f * static_cast<float>(i) * inv_dst_ch;
        timescale_data[i] =
            params.min_timescale *
            std::pow(params.max_timescale / params.min_timescale, fraction);
      }
      TensorTf timescale = litert::tensor::Create(
          "timescale", litert::tensor::Type::kFP32, {1, 1, 1, half_dim},
          litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kFP32>(
              timescale_data));

      auto pos_fp = litert::tensor::Cast(pos_in, litert::tensor::Type::kFP32);
      auto pos_4d = litert::tensor::Reshape(pos_fp, {b, 1, s, 1});
      auto sinusoid = litert::tensor::Div(pos_4d, timescale);
      auto cos_fp = litert::tensor::Cos(sinusoid);
      auto sin_fp = litert::tensor::Sin(sinusoid);
      auto cos_val =
          litert::tensor::Cast(cos_fp, litert::tensor::ApiType<T>::value);
      auto sin_val =
          litert::tensor::Cast(sin_fp, litert::tensor::ApiType<T>::value);

      auto q_out0 = litert::tensor::Sub(
          litert::tensor::Mul(q0, cos_val), litert::tensor::Mul(q1, sin_val));
      auto q_out1 = litert::tensor::Add(
          litert::tensor::Mul(q1, cos_val), litert::tensor::Mul(q0, sin_val));
      auto q_out = litert::tensor::Concatenation({q_out0, q_out1}, /*axis=*/3);

      auto k_out0 = litert::tensor::Sub(
          litert::tensor::Mul(k0, cos_val), litert::tensor::Mul(k1, sin_val));
      auto k_out1 = litert::tensor::Add(
          litert::tensor::Mul(k1, cos_val), litert::tensor::Mul(k0, sin_val));
      auto k_out = litert::tensor::Concatenation({k_out0, k_out1}, /*axis=*/3);

      return std::make_tuple(q_out, k_out, v_out);
    };

    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.qkv_norm_rope",
        .composite_attributes = composite_attributes,
    };

    auto [q_out, k_out, v_out] = litert::tensor::StableHLOComposite(
        composite_options, decompose, qkv, position, q_weight, k_weight);

    q_out.SetName("query_states");
    k_out.SetName("key_states");
    v_out.SetName("value_states");

    return litert::testing::SaveTensorGraph({q_out, k_out, v_out});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_QKV_NORM_ROPE_H_
