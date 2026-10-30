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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_CACHE_UPDATE_SDPA_TRANSPOSED_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_CACHE_UPDATE_SDPA_TRANSPOSED_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/generators/reference_evaluator.h"
#include "litert/test/generators/sdpa_transposed.h"
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

// Compound generator exercising the fused GPU KV-cache update + transposed SDPA
// pipeline (`odml.cache_update` -> `odml.sdpa_transposed`, where
// `from_cache_update == true`).
template <typename T, typename WithMask = std::true_type,
          typename SoftCap = std::false_type>
class CacheUpdateSdpaTransposed : public TestGraph {
 private:
  static constexpr bool kWithMask = WithMask::value;
  using BaseSdpa = SdpaTransposed<T, WithMask, SoftCap, std::true_type>;
  using TensorTf = typename BaseSdpa::TensorTf;

 public:
  using Params = typename BaseSdpa::Params;
  using GridConfig = typename BaseSdpa::GridConfig;
  using Ptr = std::unique_ptr<CacheUpdateSdpaTransposed>;

  static constexpr absl::string_view Name() {
    return "CacheUpdateSdpaTransposed";
  }

  template <typename Rng>
  static Expected<CacheUpdateSdpaTransposed::Ptr> Create(Rng& /*rng*/) {
    static size_t sample_counter = 0;
    return Create(BaseSdpa::SampleParams(sample_counter));
  }

  static Expected<CacheUpdateSdpaTransposed::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<CacheUpdateSdpaTransposed>(std::move(params),
                                                       std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    return BaseSdpa::GetConformanceSpecForParams(params_);
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    return BaseSdpa::MakeInputsForParams(params_, /*from_cache_update=*/true,
                                         device, data_builder);
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return ReferenceEvaluator::Evaluate(Graph(), inputs, outputs);
  }

  CacheUpdateSdpaTransposed(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    int b = static_cast<int>(params.batch);
    int num_kv_heads = static_cast<int>(params.num_kv_heads);
    int num_q_heads = static_cast<int>(params.num_q_heads);
    int q_seq_len = static_cast<int>(params.q_seq_len);
    int kv_len = static_cast<int>(params.kv_len);
    int head_dim = static_cast<int>(params.head_dim);

    TensorTf q =
        litert::tensor::Create("query", litert::tensor::ApiType<T>::value,
                               {b, num_q_heads, q_seq_len, head_dim});
    TensorTf k =
        litert::tensor::Create("key", litert::tensor::ApiType<T>::value,
                               {b, num_kv_heads, kv_len, head_dim});
    TensorTf v =
        litert::tensor::Create("value", litert::tensor::ApiType<T>::value,
                               {b, num_kv_heads, kv_len, head_dim});

    TensorTf mask;
    if constexpr (kWithMask) {
      const auto mask_type = params.bool_mask
                                 ? litert::tensor::Type::kBOOL
                                 : litert::tensor::ApiType<T>::value;
      mask =
          litert::tensor::Create("mask", mask_type, {b, 1, q_seq_len, kv_len});
    }

    TensorTf param_tensor = litert::tensor::Create(
        "param_tensor", litert::tensor::ApiType<int32_t>::value, {1, 1, 1, 7});

    // 1. odml.cache_update composite op
    flexbuffers::Builder cache_fbb;
    cache_fbb.Map([&]() {
      cache_fbb.Int("kv_cache_batch_size", b * num_kv_heads);
      cache_fbb.Int("cache_size", kv_len);
      cache_fbb.Int("head_size", head_dim);
    });
    cache_fbb.Finish();

    auto decompose_cache_update = [b, num_kv_heads, kv_len, head_dim](
                                      auto k_in, auto v_in, auto /*param_in*/) {
      auto cache_k =
          litert::tensor::Reshape(k_in, {b, num_kv_heads, kv_len, head_dim});
      auto cache_v = litert::tensor::Transpose(v_in, {0, 1, 3, 2});
      return std::make_tuple(cache_k, cache_v);
    };

    litert::tensor::StableHLOCompositeOptions cache_options{
        .name = "odml.cache_update",
        .composite_attributes = cache_fbb.GetBuffer(),
    };
    auto [cache_k, cache_v] = litert::tensor::StableHLOComposite(
        cache_options, decompose_cache_update, k, v, param_tensor);
    cache_k.SetName("cache_k");
    cache_v.SetName("cache_v");

    // 2. odml.sdpa_transposed composite op (with from_cache_update = true)
    TensorTf output =
        BaseSdpa::BuildSdpaComposite(params, /*from_cache_update=*/true, q,
                                     cache_k, cache_v, mask, param_tensor);

    std::vector<litert::tensor::TensorHandle> graph_inputs = {q, k, v};
    if constexpr (kWithMask) {
      graph_inputs.push_back(mask);
    }
    graph_inputs.push_back(param_tensor);
    return SaveTensorGraph(std::move(graph_inputs), {output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_CACHE_UPDATE_SDPA_TRANSPOSED_H_
