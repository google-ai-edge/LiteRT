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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SDPA_TRANSPOSED_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SDPA_TRANSPOSED_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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
          typename SoftCap = std::false_type,
          typename WithParamTensor = std::false_type>
class SdpaTransposed : public TestGraph {
 private:
  static constexpr bool kWithMask = WithMask::value;
  static constexpr bool kSoftCap = SoftCap::value;
  static constexpr bool kWithParamTensor = WithParamTensor::value;

 public:
  struct Params {
    size_t batch = 1;
    size_t num_kv_heads = 1;
    size_t num_q_heads = 8;
    size_t kv_len = 128;
    size_t head_dim = 128;
    float scale = 1.0f;
    float softcap_val = 50.0f;
    bool bool_mask = true;
  };

  using Ptr = std::unique_ptr<SdpaTransposed>;

  static constexpr absl::string_view Name() { return "SdpaTransposed"; }

  struct GridConfig {
    int batch;
    int num_kv_heads;
    int num_q_heads;
    int kv_len;
    int head_dim;
    bool bool_mask;
  };

  // Stratified grid of 16 realistic transposed LLM attention workloads.
  // ATS test suites should specify iters >= 16 to ensure full coverage.
  static constexpr GridConfig kStratifiedGrid[] = {
      // ─── 1. Standard Head Dimension (Head Dim = 128) Decode Workloads ───
      {1, 1, 8, 32, 128, true},     // 0: MQA (8:1) Short Decode
      {1, 1, 8, 128, 128, true},    // 1: MQA (8:1) Medium Decode
      {1, 1, 8, 1024, 128, true},   // 2: MQA (8:1) Long Context Decode
      {1, 1, 8, 2048, 128, false},  // 3: MQA (8:1) Ultra-Long Float Mask
      {1, 2, 8, 128, 128, true},    // 4: GQA (8:2) Medium Decode
      {1, 2, 8, 1024, 128, true},   // 5: GQA (8:2) Long Decode
      {1, 4, 8, 256, 128, false},   // 6: GQA (8:4) Decode w/ Float Mask
      {1, 8, 8, 512, 128, true},    // 7: MHA (8:8) Decode

      // ─── 2. Non-Standard Head Dimensions (Head Dim = 64, 256) ───
      {1, 4, 4, 256, 64, true},   // 8: MHA D=64 Decode (Tiny LLM)
      {1, 1, 8, 512, 256, true},  // 9: MQA D=256 Decode (Large Head Dim)

      // ─── 3. Prefill & Extended Sequence Workloads ───
      {1, 1, 8, 32, 128, true},    // 10: MQA Short Sequence
      {1, 2, 8, 64, 128, true},    // 11: GQA Medium Sequence
      {1, 2, 8, 128, 128, false},  // 12: GQA Long Sequence w/ Float Mask
      {1, 4, 8, 256, 128, true},   // 13: GQA Large Context
      {1, 4, 4, 64, 64, true},     // 14: MHA D=64
      {1, 8, 8, 128, 256, true},   // 15: MHA D=256
  };

  template <typename Rng>
  static Expected<SdpaTransposed::Ptr> Create(Rng& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    const auto& entry = kStratifiedGrid[(sample_counter++) % kGridSize];

    Params params;
    params.batch = entry.batch;
    params.num_kv_heads = entry.num_kv_heads;
    params.num_q_heads = entry.num_q_heads;
    params.kv_len = entry.kv_len;
    params.head_dim = entry.head_dim;
    params.scale = 1.0f;
    params.softcap_val = 50.0f;
    params.bool_mask = entry.bool_mask;

    return Create(std::move(params));
  }

  static Expected<SdpaTransposed::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<SdpaTransposed>(std::move(params),
                                            std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kFloatAccumulationAware;
    spec.accumulation_depth = std::max(params_.head_dim, params_.kv_len);
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
    size_t expected_inputs =
        3 + (kWithMask ? 1 : 0) + (kWithParamTensor ? 1 : 0);
    inputs.reserve(expected_inputs);

    std::array<Layout::Dim, 4> q_shape = {
        static_cast<Layout::Dim>(params_.batch),
        static_cast<Layout::Dim>(params_.num_kv_heads),
        static_cast<Layout::Dim>(params_.num_q_heads),
        static_cast<Layout::Dim>(params_.head_dim)};
    std::array<Layout::Dim, 4> k_shape = {
        static_cast<Layout::Dim>(params_.batch),
        static_cast<Layout::Dim>(params_.num_kv_heads),
        static_cast<Layout::Dim>(params_.kv_len),
        static_cast<Layout::Dim>(params_.head_dim)};
    std::array<Layout::Dim, 4> v_shape = {
        static_cast<Layout::Dim>(params_.batch),
        static_cast<Layout::Dim>(params_.num_kv_heads),
        static_cast<Layout::Dim>(params_.head_dim),
        static_cast<Layout::Dim>(params_.kv_len)};

    auto builder = data_builder;
    if (!builder.IsFloatDummy()) {
      builder.SetFloatRange(-1.5f, 1.5f);
    }

    auto q_builder = builder;
    if (!q_builder.IsFloatDummy()) {
      float q_scale = 1.0f / std::sqrt(static_cast<float>(params_.head_dim));
      q_builder.SetFloatRange(-1.5f * q_scale, 1.5f * q_scale);
    }

    // 1. Query input [B, H_kv, H_q, D]
    LITERT_ASSIGN_OR_RETURN(auto q, SimpleBuffer::Create<T>(q_shape));
    LITERT_RETURN_IF_ERROR((q.template WriteRandom<T>(q_builder, device)));
    inputs.push_back(std::move(q));

    // 2. Key input [B, H_kv, KV_LEN, D]
    LITERT_ASSIGN_OR_RETURN(auto k, SimpleBuffer::Create<T>(k_shape));
    LITERT_RETURN_IF_ERROR((k.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(k));

    // 3. Transposed Value input [B, H_kv, D, KV_LEN]
    LITERT_ASSIGN_OR_RETURN(auto v, SimpleBuffer::Create<T>(v_shape));
    LITERT_RETURN_IF_ERROR((v.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(v));

    // 4. Optional Mask input [B, 1, H_q, KV_LEN]
    if constexpr (kWithMask) {
      std::array<Layout::Dim, 4> mask_shape = {
          static_cast<Layout::Dim>(params_.batch), 1,
          static_cast<Layout::Dim>(params_.num_q_heads),
          static_cast<Layout::Dim>(params_.kv_len)};
      LITERT_ASSIGN_OR_RETURN(auto mask, SimpleBuffer::Create<T>(mask_shape));
      auto mask_span = mask.Span<T>();

      for (size_t b = 0; b < params_.batch; ++b) {
        for (size_t n = 0; n < params_.num_q_heads; ++n) {
          for (size_t j = 0; j < params_.kv_len; ++j) {
            size_t idx = (b * params_.num_q_heads + n) * params_.kv_len + j;
            mask_span[idx] = static_cast<T>(0.0f);
          }
        }
      }
      inputs.push_back(std::move(mask));
    }

    // 5. Optional Param Tensor input [1, 1, 1, 7]
    // Shape [1, 1, 1, 7] is the standard 4D control buffer layout exported by
    // litert_torch for LLM runtime parameters.
    if constexpr (kWithParamTensor) {
      std::array<Layout::Dim, 4> param_shape = {1, 1, 1, 7};
      LITERT_ASSIGN_OR_RETURN(auto param,
                              SimpleBuffer::Create<int32_t>(param_shape));
      auto param_span = param.Span<int32_t>();
      for (size_t i = 0; i < 7; ++i) {
        param_span[i] = 0;
      }
      // Index 2 specifies the active token count for runtime bounds checks,
      // informing delegates of the populated KV cache size without reshaping.
      param_span[2] = static_cast<int32_t>(params_.kv_len);
      inputs.push_back(std::move(param));
    }

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return ReferenceEvaluator::EvaluateCompositeReference(Graph(), inputs,
                                                           outputs);
  }

  SdpaTransposed(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

    int b = static_cast<int>(params.batch);
    int num_kv_heads = static_cast<int>(params.num_kv_heads);
    int num_q_heads = static_cast<int>(params.num_q_heads);
    int kv_len = static_cast<int>(params.kv_len);
    int head_dim = static_cast<int>(params.head_dim);

    TensorTf q = litert::tensor::Create(
        "query", litert::tensor::ApiType<T>::value,
        {b, num_kv_heads, num_q_heads, head_dim});
    TensorTf k = litert::tensor::Create(
        "key", litert::tensor::ApiType<T>::value,
        {b, num_kv_heads, kv_len, head_dim});
    TensorTf v = litert::tensor::Create(
        "value", litert::tensor::ApiType<T>::value,
        {b, num_kv_heads, head_dim, kv_len});

    TensorTf mask;
    if constexpr (kWithMask) {
      mask = litert::tensor::Create(
          "mask", litert::tensor::ApiType<T>::value,
          {b, 1, num_q_heads, kv_len});
    }

    TensorTf param_tensor;
    if constexpr (kWithParamTensor) {
      // 4D control tensor [1, 1, 1, 7] per litert_torch runtime parameter
      // specification for dynamic cache bounds.
      param_tensor = litert::tensor::Create(
          "param_tensor", litert::tensor::ApiType<int32_t>::value,
          {1, 1, 1, 7});
    }

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      // Tests the Transposed-V layout exported by litert_torch.
      // Indices reflect the shapes defined above: k has time step kv_len at
      // axis 2 [B, H_kv, KV_LEN, D], and v has time step kv_len transposed to
      // axis 3 [B, H_kv, D, KV_LEN].
      fbb.Int("k_ts_idx", 2);
      fbb.Int("v_ts_idx", 3);
      if constexpr (kSoftCap) {
        fbb.Float("softcap", params.softcap_val);
      }
    });
    fbb.Finish();
    auto composite_attributes = fbb.GetBuffer();

    auto decompose = [&params](auto q_in, auto k_in, auto v_in, auto... rest) {
      auto qk = litert::tensor::BatchMatMul(q_in, k_in, /*adj_x=*/false,
                                            /*adj_y=*/true);
      auto pre_mask_scores = qk;
      if constexpr (kSoftCap) {
        TensorTf cap_tensor = litert::tensor::Create(
            "sdpa_softcap", litert::tensor::ApiType<T>::value,
            /*shape=*/{1},
            litert::tensor::OwningCpuBuffer::CopyAs(
                litert::tensor::ApiType<T>::value,
                std::vector<float>{params.softcap_val}));
        auto div_cap = litert::tensor::Div(qk, cap_tensor);
        auto tanh_scores = litert::tensor::Tanh(div_cap);
        pre_mask_scores = litert::tensor::Mul(tanh_scores, cap_tensor);
      }
      auto scores = pre_mask_scores;
      if constexpr (kWithMask) {
        auto rest_tuple = std::forward_as_tuple(rest...);
        scores = litert::tensor::Add(pre_mask_scores, std::get<0>(rest_tuple));
      }
      auto probs = litert::tensor::Softmax(scores, /*beta=*/1.0f);
      return litert::tensor::BatchMatMul(probs, v_in, /*adj_x=*/false,
                                         /*adj_y=*/true);
    };

    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.sdpa_transposed",
        .composite_attributes = composite_attributes,
    };

    TensorTf output;
    if constexpr (kWithMask && kWithParamTensor) {
      output = litert::tensor::StableHLOComposite(composite_options, decompose,
                                                  q, k, v, mask, param_tensor);
    } else if constexpr (kWithMask && !kWithParamTensor) {
      output = litert::tensor::StableHLOComposite(composite_options, decompose,
                                                  q, k, v, mask);
    } else if constexpr (!kWithMask && kWithParamTensor) {
      output = litert::tensor::StableHLOComposite(composite_options, decompose,
                                                  q, k, v, param_tensor);
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

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SDPA_TRANSPOSED_H_
