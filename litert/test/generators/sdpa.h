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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SDPA_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SDPA_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/matmul.h"
#include "litert/core/model/ops/simple_binary.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/core/model/ops/tile.h"
#include "litert/core/model/ops/transpose.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/types/half.h"

namespace litert::testing {

template <typename T, typename WithMask = std::true_type,
          typename SoftCap = std::false_type>
class Sdpa : public TestGraph {
 private:
  static constexpr bool kWithMask = WithMask::value;
  static constexpr bool kSoftCap = SoftCap::value;

 public:
  struct Params {
    size_t batch = 1;
    size_t num_q_heads = 4;
    size_t num_kv_heads = 4;
    size_t seq_q = 4;
    size_t seq_k = 4;
    size_t head_dim = 64;
    float scale = 0.125f;
    float softcap_val = 50.0f;
  };

  using Ptr = std::unique_ptr<Sdpa>;

  static constexpr absl::string_view Name() { return "Sdpa"; }

  struct GridConfig {
    int batch;
    int num_q_heads;
    int num_kv_heads;
    int seq_q;
    int seq_k;
    int head_dim;
  };

  // Stratified grid of 15 realistic LLM attention workloads (MQA/GQA/MHA).
  // ATS test suites should specify iters >= 15 to ensure full coverage.
  static constexpr GridConfig kStratifiedGrid[] = {
      // --- Decode Configurations (S_Q = 1) ---
      {1, 2, 1, 1, 32, 128},    // 0: MQA (2:1), Short Decode (S_K = 32)
      {1, 8, 1, 1, 128, 128},   // 1: MQA (8:1), Short Decode (S_K = 128)
      {1, 8, 1, 1, 1024, 128},  // 2: MQA (8:1), Long Decode (S_K = 1024)
      {1, 4, 2, 1, 64, 128},    // 3: GQA (4:2), Short Decode (S_K = 64)
      {1, 8, 2, 1, 1024, 128},  // 4: GQA (8:2), Long Decode (S_K = 1024)
      {1, 8, 4, 1, 128, 128},   // 5: GQA (8:4), Medium Decode (S_K = 128)
      {1, 4, 4, 1, 256, 64},    // 6: MHA (4:4), Medium Decode (S_K = 256)
      {1, 8, 8, 1, 1024, 128},  // 7: MHA (8:8), Long Decode (S_K = 1024)

      // --- Prefill Configurations (S_Q = S_K) ---
      {1, 2, 1, 32, 32, 256},    // 8: MQA (2:1), Short Prefill (S = 32)
      {1, 8, 1, 64, 64, 128},    // 9: MQA (8:1), Short Prefill (S = 64)
      {1, 4, 2, 128, 128, 128},  // 10: GQA (4:2), Medium Prefill (S = 128)
      {1, 8, 2, 256, 256, 128},  // 11: GQA (8:2), Long Prefill (S = 256)
      {1, 8, 4, 15, 15, 256},    // 12: GQA (8:4), Short Prefill (S = 15)
      {1, 4, 4, 256, 256, 64},   // 13: MHA (4:4), Medium Prefill (S = 256)
      {1, 8, 8, 256, 256, 128},  // 14: MHA (8:8), Long Prefill (S = 256)
  };

  template <typename Rng>
  static Expected<Sdpa::Ptr> Create(Rng& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    const auto& entry = kStratifiedGrid[(sample_counter++) % kGridSize];

    Params params;
    params.batch = entry.batch;
    params.num_q_heads = entry.num_q_heads;
    params.num_kv_heads = entry.num_kv_heads;
    params.seq_q = entry.seq_q;
    params.seq_k = entry.seq_k;
    params.head_dim = entry.head_dim;
    params.scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
    params.softcap_val = 50.0f;

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Sdpa>(std::move(params), std::move(model));
  }

  static Expected<Sdpa::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Sdpa>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kFloatAccumulationAware;
    spec.accumulation_depth = std::max(params_.head_dim, params_.seq_k);
    if constexpr (std::is_same_v<T, float>) {
      spec.relative_tolerance = 1e-3;
      spec.absolute_tolerance = 1e-3;
    } else {
      spec.relative_tolerance = 5e-2;
      spec.absolute_tolerance = 5e-2;
    }
    return spec;
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(kWithMask ? 4 : 3);

    std::array<Layout::Dim, 4> q_shape = {
        static_cast<Layout::Dim>(params_.batch),
        static_cast<Layout::Dim>(params_.seq_q),
        static_cast<Layout::Dim>(params_.num_q_heads),
        static_cast<Layout::Dim>(params_.head_dim)};
    std::array<Layout::Dim, 4> kv_shape = {
        static_cast<Layout::Dim>(params_.batch),
        static_cast<Layout::Dim>(params_.seq_k),
        static_cast<Layout::Dim>(params_.num_kv_heads),
        static_cast<Layout::Dim>(params_.head_dim)};

    auto builder = data_builder;
    if (!builder.IsFloatDummy()) {
      builder.SetFloatRange(-1.5f, 1.5f);
    }

    // 1. Query input
    LITERT_ASSIGN_OR_RETURN(auto q, SimpleBuffer::Create<T>(q_shape));
    LITERT_RETURN_IF_ERROR((q.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(q));

    // 2. Key input
    LITERT_ASSIGN_OR_RETURN(auto k, SimpleBuffer::Create<T>(kv_shape));
    LITERT_RETURN_IF_ERROR((k.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(k));

    // 3. Value input
    LITERT_ASSIGN_OR_RETURN(auto v, SimpleBuffer::Create<T>(kv_shape));
    LITERT_RETURN_IF_ERROR((v.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(v));

    // 4. Optional Mask input
    if constexpr (kWithMask) {
      std::array<Layout::Dim, 4> mask_shape = {
          static_cast<Layout::Dim>(params_.batch), 1,
          static_cast<Layout::Dim>(params_.seq_q),
          static_cast<Layout::Dim>(params_.seq_k)};
      LITERT_ASSIGN_OR_RETURN(auto mask, SimpleBuffer::Create<T>(mask_shape));
      auto mask_span = mask.Span<T>();

      for (size_t b = 0; b < params_.batch; ++b) {
        for (size_t i = 0; i < params_.seq_q; ++i) {
          for (size_t j = 0; j < params_.seq_k; ++j) {
            size_t idx = (b * params_.seq_q + i) * params_.seq_k + j;
            mask_span[idx] = (params_.seq_q == params_.seq_k && j > i)
                                 ? static_cast<T>(-10000.0f)
                                 : static_cast<T>(0.0f);
          }
        }
      }
      inputs.push_back(std::move(mask));
    }

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    if (outputs.empty()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Outputs cannot be empty");
    }
    auto& out_buf = outputs[0];

    std::vector<float> q_f32;
    if constexpr (std::is_same_v<T, float>) {
      auto span = inputs[0].Span<float>();
      q_f32.assign(span.begin(), span.end());
    } else if constexpr (std::is_same_v<T, tflite::half>) {
      auto span = inputs[0].Span<tflite::half>();
      q_f32.resize(span.size());
      for (size_t i = 0; i < span.size(); ++i) {
        q_f32[i] = static_cast<float>(span[i]);
      }
    }

    std::vector<float> k_f32;
    std::vector<float> v_f32;
    if constexpr (std::is_same_v<T, float>) {
      auto k_span = inputs[1].Span<float>();
      auto v_span = inputs[2].Span<float>();
      k_f32.assign(k_span.begin(), k_span.end());
      v_f32.assign(v_span.begin(), v_span.end());
    } else if constexpr (std::is_same_v<T, tflite::half>) {
      auto k_span = inputs[1].Span<tflite::half>();
      auto v_span = inputs[2].Span<tflite::half>();
      k_f32.resize(k_span.size());
      v_f32.resize(v_span.size());
      for (size_t i = 0; i < k_span.size(); ++i) {
        k_f32[i] = static_cast<float>(k_span[i]);
      }
      for (size_t i = 0; i < v_span.size(); ++i) {
        v_f32[i] = static_cast<float>(v_span[i]);
      }
    }

    std::vector<float> mask_f32;
    if constexpr (kWithMask) {
      if constexpr (std::is_same_v<T, float>) {
        auto span = inputs[3].Span<float>();
        mask_f32.assign(span.begin(), span.end());
      } else if constexpr (std::is_same_v<T, tflite::half>) {
        auto span = inputs[3].Span<tflite::half>();
        mask_f32.resize(span.size());
        for (size_t i = 0; i < span.size(); ++i) {
          mask_f32[i] = static_cast<float>(span[i]);
        }
      }
    }

    size_t B = params_.batch;
    size_t N_Q = params_.num_q_heads;
    size_t N_KV = params_.num_kv_heads;
    size_t SQ = params_.seq_q;
    size_t SK = params_.seq_k;
    size_t D = params_.head_dim;

    // Step 1: Prepare Q, K, V in [B * N_Q, S, D] layout (with GQA/MQA broadcast).
    auto q_bnhd = PrepareAttentionInput(q_f32, B, SQ, N_Q, D, N_Q);
    auto k_bnhd = PrepareAttentionInput(k_f32, B, SK, N_KV, D, N_Q);
    auto v_bnhd = PrepareAttentionInput(v_f32, B, SK, N_KV, D, N_Q);

    // Step 2: BatchMatmul(Q, K^T) -> [B * N_Q, SQ, SK].
    int32_t q_dims[3] = {static_cast<int32_t>(B * N_Q),
                         static_cast<int32_t>(SQ), static_cast<int32_t>(D)};
    int32_t k_dims[3] = {static_cast<int32_t>(B * N_Q),
                         static_cast<int32_t>(SK), static_cast<int32_t>(D)};
    int32_t qk_dims[3] = {static_cast<int32_t>(B * N_Q),
                          static_cast<int32_t>(SQ), static_cast<int32_t>(SK)};

    std::vector<float> scores(B * N_Q * SQ * SK);
    litert::internal::ReferenceBatchMatmul(
        q_bnhd.data(), q_dims, 3, k_bnhd.data(), k_dims, 3, scores.data(),
        qk_dims, 3, /*adj_x=*/false, /*adj_y=*/true);

    // Step 3: Mul(scores, scale).
    int32_t scalar_dims[1] = {1};
    float scale = params_.scale;
    litert::internal::ReferenceBinaryGeneric(
        scores.data(), qk_dims, 3, &scale, scalar_dims, 1, scores.data(),
        qk_dims, 3, std::multiplies<float>());

    // Step 4: Softcap (Div -> Tanh -> Mul).
    if constexpr (kSoftCap) {
      float cap = params_.softcap_val;
      litert::internal::ReferenceBinaryGeneric(
          scores.data(), qk_dims, 3, &cap, scalar_dims, 1, scores.data(),
          qk_dims, 3, std::divides<float>());
      litert::internal::ReferenceTanh(scores.data(), scores.size(),
                                      scores.data());
      litert::internal::ReferenceBinaryGeneric(
          scores.data(), qk_dims, 3, &cap, scalar_dims, 1, scores.data(),
          qk_dims, 3, std::multiplies<float>());
    }

    // Step 5: Add Mask (if present) via broadcast Add.
    if constexpr (kWithMask) {
      int32_t scores_4d_dims[4] = {static_cast<int32_t>(B),
                                   static_cast<int32_t>(N_Q),
                                   static_cast<int32_t>(SQ),
                                   static_cast<int32_t>(SK)};
      int32_t mask_4d_dims[4] = {static_cast<int32_t>(B), 1,
                                 static_cast<int32_t>(SQ),
                                 static_cast<int32_t>(SK)};
      litert::internal::ReferenceBinaryGeneric(
          scores.data(), scores_4d_dims, 4, mask_f32.data(), mask_4d_dims, 4,
          scores.data(), scores_4d_dims, 4, std::plus<float>());
    }

    // Step 6: Softmax along SK axis.
    std::vector<float> probs(scores.size());
    litert::internal::ReferenceSoftmax(
        scores.data(), probs.data(),
        /*batch=*/static_cast<int>(B * N_Q * SQ),
        /*depth=*/static_cast<int>(SK),
        /*beta=*/1.0f);

    // Step 7: BatchMatmul(Probs, V) -> [B * N_Q, SQ, D].
    int32_t probs_dims[3] = {static_cast<int32_t>(B * N_Q),
                             static_cast<int32_t>(SQ),
                             static_cast<int32_t>(SK)};
    int32_t v_dims[3] = {static_cast<int32_t>(B * N_Q),
                         static_cast<int32_t>(SK), static_cast<int32_t>(D)};
    int32_t out_bnhd_dims[3] = {static_cast<int32_t>(B * N_Q),
                                static_cast<int32_t>(SQ),
                                static_cast<int32_t>(D)};

    std::vector<float> out_bnhd(B * N_Q * SQ * D);
    litert::internal::ReferenceBatchMatmul(
        probs.data(), probs_dims, 3, v_bnhd.data(), v_dims, 3, out_bnhd.data(),
        out_bnhd_dims, 3, /*adj_x=*/false, /*adj_y=*/false);

    // Step 8: Transpose [B, N_Q, SQ, D] back to [B, SQ, N_Q, D].
    std::vector<float> out_f32(B * SQ * N_Q * D);
    int32_t bnhd_dims[4] = {static_cast<int32_t>(B),
                            static_cast<int32_t>(N_Q),
                            static_cast<int32_t>(SQ),
                            static_cast<int32_t>(D)};
    int32_t out_perm[4] = {0, 2, 1, 3};
    litert::internal::ReferenceTranspose(out_bnhd.data(), bnhd_dims, out_perm,
                                         4, out_f32.data());

    if constexpr (std::is_same_v<T, float>) {
      auto span = out_buf.Span<float>();
      absl::c_copy(out_f32, span.begin());
    } else if constexpr (std::is_same_v<T, tflite::half>) {
      auto span = out_buf.Span<tflite::half>();
      for (size_t i = 0; i < out_f32.size(); ++i) {
        span[i] = tflite::half(out_f32[i]);
      }
    }
    return {};
  }

  Sdpa(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  // Prepares input [B, S, N, D] into [B * target_N, S, D] layout.
  // Mirrors the exact decomposed subgraph in BuildGraph:
  //   1. Transpose [B, S, N, D] -> [B, N, S, D]
  //   2. (If target_N > N): Tile along head axis -> [B, G * N, S, D]
  //   3. (If N > 1): Reorder heads via 5D Transpose
  //      [B, G, N, S, D] -> [B, N, G, S, D]
  static std::vector<float> PrepareAttentionInput(
      const std::vector<float>& input, size_t B, size_t S, size_t N, size_t D,
      size_t target_N) {
    int32_t b = static_cast<int32_t>(B);
    int32_t s = static_cast<int32_t>(S);
    int32_t n = static_cast<int32_t>(N);
    int32_t d = static_cast<int32_t>(D);
    int32_t target_n = static_cast<int32_t>(target_N);

    // 1. Transpose [B, S, N, D] -> [B, N, S, D]
    std::vector<float> transposed(B * N * S * D);
    int32_t in_dims[4] = {b, s, n, d};
    int32_t perm[4] = {0, 2, 1, 3};
    litert::internal::ReferenceTranspose(input.data(), in_dims, perm, 4,
                                         transposed.data());
    if (N == target_N) {
      return transposed;
    }

    int32_t g = target_n / n;

    // 2. Tile along head axis -> [B, G * N, S, D]
    std::vector<float> tiled(B * target_N * S * D);
    int32_t transposed_dims[4] = {b, n, s, d};
    int32_t multiples[4] = {1, g, 1, 1};
    litert::internal::ReferenceTile(transposed.data(), transposed_dims,
                                    multiples, 4, tiled.data());
    if (N == 1) {
      return tiled;
    }

    // 3. Reorder heads: Transpose 5D [B, G, N, S, D] -> [B, N, G, S, D]
    std::vector<float> output(B * target_N * S * D);
    int32_t dims_5d[5] = {b, g, n, s, d};
    int32_t perm_5d[5] = {0, 2, 1, 3, 4};
    litert::internal::ReferenceTranspose(tiled.data(), dims_5d, perm_5d, 5,
                                         output.data());
    return output;
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

    int b = static_cast<int>(params.batch);
    int q_h = static_cast<int>(params.num_q_heads);
    int kv_h = static_cast<int>(params.num_kv_heads);
    int sq = static_cast<int>(params.seq_q);
    int sk = static_cast<int>(params.seq_k);
    int d = static_cast<int>(params.head_dim);

    TensorTf q = litert::tensor::Create(
        "query", litert::tensor::ApiType<T>::value, {b, sq, q_h, d});
    TensorTf k = litert::tensor::Create(
        "key", litert::tensor::ApiType<T>::value, {b, sk, kv_h, d});
    TensorTf v = litert::tensor::Create(
        "value", litert::tensor::ApiType<T>::value, {b, sk, kv_h, d});

    std::vector<litert::tensor::TensorHandle> in_tensors = {q, k, v};

    TensorTf mask;
    if constexpr (kWithMask) {
      mask = litert::tensor::Create("mask", litert::tensor::ApiType<T>::value,
                                    {b, 1, sq, sk});
      in_tensors.push_back(mask);
    }

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Float("scale", params.scale);
      if constexpr (kSoftCap) {
        fbb.Float("logit_cap", params.softcap_val);
        fbb.Float("softcap", params.softcap_val);
      }
    });
    fbb.Finish();
    auto composite_attributes = fbb.GetBuffer();

    auto repeat_kv = [&params](auto tensor_in, int sk_dim, int d_dim) {
      auto transposed = litert::tensor::Transpose(tensor_in, {0, 2, 1, 3});
      if (params.num_q_heads == params.num_kv_heads) {
        return transposed;
      }
      int b_dim = static_cast<int>(params.batch);
      int g = static_cast<int>(params.num_q_heads / params.num_kv_heads);
      int kv_heads = static_cast<int>(params.num_kv_heads);

      std::vector<decltype(transposed)> repeated_heads;
      repeated_heads.reserve(g);
      for (int i = 0; i < g; ++i) {
        repeated_heads.push_back(transposed);
      }
      auto concat = litert::tensor::Concatenation(
          absl::MakeSpan(repeated_heads), /*axis=*/1);
      if (kv_heads == 1) {
        return concat;
      }

      auto reshaped_5d =
          litert::tensor::Reshape(concat, {b_dim, g, kv_heads, sk_dim, d_dim});
      auto transposed_5d =
          litert::tensor::Transpose(reshaped_5d, {0, 2, 1, 3, 4});
      return litert::tensor::Reshape(
          transposed_5d,
          {b_dim, static_cast<int>(params.num_q_heads), sk_dim, d_dim});
    };

    TensorTf output;
    if constexpr (kWithMask) {
      output = litert::tensor::StableHLOComposite(
          litert::tensor::StableHLOCompositeOptions{
              .name = "odml.scaled_dot_product_attention",
              .composite_attributes = composite_attributes,
          },
          [&params, repeat_kv, sk, d](auto q_in, auto k_in, auto v_in,
                                      auto mask_in) {
            auto q_bnhd = litert::tensor::Transpose(q_in, {0, 2, 1, 3});
            auto k_bnhd = repeat_kv(k_in, sk, d);
            auto v_bnhd = repeat_kv(v_in, sk, d);
            auto qk = litert::tensor::BatchMatMul(
                q_bnhd, k_bnhd, /*adj_x=*/false, /*adj_y=*/true);
            TensorTf scale_tensor = litert::tensor::Create(
                "sdpa_scale", litert::tensor::ApiType<T>::value, /*shape=*/{1},
                litert::tensor::OwningCpuBuffer::CopyAs(
                    litert::tensor::ApiType<T>::value,
                    std::vector<float>{params.scale}));
            auto scaled_qk = litert::tensor::Mul(qk, scale_tensor);
            auto pre_mask_scores = scaled_qk;
            if constexpr (kSoftCap) {
              TensorTf cap_tensor = litert::tensor::Create(
                  "sdpa_softcap", litert::tensor::ApiType<T>::value,
                  /*shape=*/{1},
                  litert::tensor::OwningCpuBuffer::CopyAs(
                      litert::tensor::ApiType<T>::value,
                      std::vector<float>{params.softcap_val}));
              auto div_cap = litert::tensor::Div(scaled_qk, cap_tensor);
              auto tanh_scores = litert::tensor::Tanh(div_cap);
              pre_mask_scores = litert::tensor::Mul(tanh_scores, cap_tensor);
            }
            auto masked_scores = litert::tensor::Add(pre_mask_scores, mask_in);
            auto probs = litert::tensor::Softmax(masked_scores, /*beta=*/1.0f);
            auto out_bnhd = litert::tensor::BatchMatMul(
                probs, v_bnhd, /*adj_x=*/false, /*adj_y=*/false);
            return litert::tensor::Transpose(out_bnhd, {0, 2, 1, 3});
          },
          q, k, v, mask);
    } else {
      output = litert::tensor::StableHLOComposite(
          litert::tensor::StableHLOCompositeOptions{
              .name = "odml.scaled_dot_product_attention",
              .composite_attributes = composite_attributes,
          },
          [&params, repeat_kv, sk, d](auto q_in, auto k_in, auto v_in) {
            auto q_bnhd = litert::tensor::Transpose(q_in, {0, 2, 1, 3});
            auto k_bnhd = repeat_kv(k_in, sk, d);
            auto v_bnhd = repeat_kv(v_in, sk, d);
            auto qk = litert::tensor::BatchMatMul(
                q_bnhd, k_bnhd, /*adj_x=*/false, /*adj_y=*/true);
            TensorTf scale_tensor = litert::tensor::Create(
                "sdpa_scale", litert::tensor::ApiType<T>::value, /*shape=*/{1},
                litert::tensor::OwningCpuBuffer::CopyAs(
                    litert::tensor::ApiType<T>::value,
                    std::vector<float>{params.scale}));
            auto scaled_qk = litert::tensor::Mul(qk, scale_tensor);
            auto pre_mask_scores = scaled_qk;
            if constexpr (kSoftCap) {
              TensorTf cap_tensor = litert::tensor::Create(
                  "sdpa_softcap", litert::tensor::ApiType<T>::value,
                  /*shape=*/{1},
                  litert::tensor::OwningCpuBuffer::CopyAs(
                      litert::tensor::ApiType<T>::value,
                      std::vector<float>{params.softcap_val}));
              auto div_cap = litert::tensor::Div(scaled_qk, cap_tensor);
              auto tanh_scores = litert::tensor::Tanh(div_cap);
              pre_mask_scores = litert::tensor::Mul(tanh_scores, cap_tensor);
            }
            auto probs =
                litert::tensor::Softmax(pre_mask_scores, /*beta=*/1.0f);
            auto out_bnhd = litert::tensor::BatchMatMul(
                probs, v_bnhd, /*adj_x=*/false, /*adj_y=*/false);
            return litert::tensor::Transpose(out_bnhd, {0, 2, 1, 3});
          },
          q, k, v);
    }

    output.SetName("output");
    return litert::testing::SaveTensorGraph(std::move(in_tensors), {output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SDPA_H_
