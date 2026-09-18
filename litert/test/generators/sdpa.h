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
#include <memory>
#include <string>
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
    return ReferenceEvaluator::EvaluateCompositeReference(Graph(), inputs,
                                                          outputs);
  }

  Sdpa(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
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

    TensorTf mask;
    if constexpr (kWithMask) {
      mask = litert::tensor::Create("mask", litert::tensor::ApiType<T>::value,
                                    {b, 1, sq, sk});
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

    auto decompose = [&params, repeat_kv, sk, d](auto q_in, auto k_in,
                                                 auto v_in, auto... rest) {
      auto q_bnhd = litert::tensor::Transpose(q_in, {0, 2, 1, 3});
      auto k_bnhd = repeat_kv(k_in, sk, d);
      auto v_bnhd = repeat_kv(v_in, sk, d);
      auto qk = litert::tensor::BatchMatMul(q_bnhd, k_bnhd, /*adj_x=*/false,
                                            /*adj_y=*/true);
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
      auto scores = pre_mask_scores;
      if constexpr (kWithMask) {
        auto rest_tuple = std::forward_as_tuple(rest...);
        scores = litert::tensor::Add(pre_mask_scores, std::get<0>(rest_tuple));
      }
      auto probs = litert::tensor::Softmax(scores, /*beta=*/1.0f);
      auto out_bnhd = litert::tensor::BatchMatMul(
          probs, v_bnhd, /*adj_x=*/false, /*adj_y=*/false);
      return litert::tensor::Transpose(out_bnhd, {0, 2, 1, 3});
    };

    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.scaled_dot_product_attention",
        .composite_attributes = composite_attributes,
    };

    TensorTf output;
    if constexpr (kWithMask) {
      output = litert::tensor::StableHLOComposite(composite_options, decompose,
                                                  q, k, v, mask);
    } else {
      output = litert::tensor::StableHLOComposite(composite_options, decompose,
                                                  q, k, v);
    }

    output.SetName("output");
    return litert::testing::SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SDPA_H_
