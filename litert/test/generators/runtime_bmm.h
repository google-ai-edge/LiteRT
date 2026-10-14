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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_RUNTIME_BMM_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_RUNTIME_BMM_H_

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
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/types/half.h"

namespace litert::testing {

template <typename T_in, typename T_rhs = T_in,
          typename IsSrc = std::false_type>
class RuntimeBmm : public TestGraph {
 private:
  static constexpr bool kIsSrc = IsSrc::value;
  static constexpr bool kIsInt4Rhs =
      std::is_same_v<T_rhs, litert::tensor::int4_t>;
  static constexpr bool kIsQuantizedRhs =
      std::is_integral_v<T_rhs> || kIsInt4Rhs;
  static constexpr float kRhsQuantScale = 0.01f;
  static constexpr TensorNames<2> kInputNames = {"lhs", "rhs"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

 public:
  struct Params {
    size_t batch = 1;
    size_t num_heads = 8;
    size_t q_seq_len = 1;
    size_t kv_len = 128;
    size_t head_dim = 64;
  };

  using Traits = TestLogicTraits<TypeList<T_in, T_rhs>, TypeList<T_in>, Params>;
  using Ptr = std::unique_ptr<RuntimeBmm>;

  static constexpr absl::string_view Name() { return "RuntimeBmm"; }

  struct GridConfig {
    size_t batch;
    size_t num_heads;
    size_t q_seq_len;
    size_t kv_len;
    size_t head_dim;
  };

  // Stratified grid of 16 representative LLM runtime batched matmul workloads.
  // Both kv_len and head_dim are even (multiples of 32) so that packed INT4
  // RHS tensors remain byte-aligned both before and after static transpose.
  // ATS test suites should specify iters >= 16 to ensure full coverage.
  static constexpr GridConfig kStratifiedGrid[] = {
      // ─── 1. Single-Token Decode Workloads (q_seq_len = 1) ───
      {1, 1, 1, 32, 64},    // 0: Single-head short decode (D=64)
      {1, 1, 1, 64, 128},   // 1: Single-head short decode (D=128)
      {1, 2, 1, 64, 64},    // 2: 2-head decode (D=64)
      {1, 2, 1, 128, 128},  // 3: 2-head medium decode (D=128)
      {1, 4, 1, 64, 64},    // 4: 4-head short decode (D=64)
      {1, 4, 1, 128, 128},  // 5: 4-head medium decode (D=128)
      {1, 8, 1, 64, 64},    // 6: 8-head short decode (D=64)
      {1, 8, 1, 128, 128},  // 7: 8-head medium decode (D=128)

      // ─── 2. Chunked Decode Workloads (q_seq_len = 4..16) ───
      {1, 2, 4, 64, 64},    // 8: 4-token chunked decode (D=64)
      {1, 4, 4, 64, 128},   // 9: 4-token chunked decode (D=128)
      {1, 4, 8, 128, 64},   // 10: 8-token chunked decode (D=64)
      {1, 8, 16, 128, 64},  // 11: 16-token chunked decode (D=64)

      // ─── 3. Prefill Workloads (q_seq_len == kv_len) ───
      {1, 2, 32, 32, 64},  // 12: 32-token prefill (D=64)
      {1, 4, 64, 64, 64},  // 13: 64-token prefill (D=64)

      // ─── 4. Batched Workloads (batch > 1) ───
      {2, 4, 1, 64, 64},  // 14: Batch-2 decode
      {2, 2, 8, 64, 64},  // 15: Batch-2 chunked decode
  };

  template <typename Rng>
  static Expected<RuntimeBmm::Ptr> Create(Rng& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    const auto& entry = kStratifiedGrid[(sample_counter++) % kGridSize];

    Params params;
    params.batch = entry.batch;
    params.num_heads = entry.num_heads;
    params.q_seq_len = entry.q_seq_len;
    params.kv_len = entry.kv_len;
    params.head_dim = entry.head_dim;

    return Create(std::move(params));
  }

  static Expected<RuntimeBmm::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<RuntimeBmm>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kFloatAccumulationAware;
    spec.accumulation_depth =
        static_cast<int64_t>(kIsSrc ? params_.kv_len : params_.head_dim);
    if constexpr (kIsQuantizedRhs) {
      spec.relative_tolerance = 5e-2;
      spec.absolute_tolerance = 5e-2;
    } else if constexpr (std::is_same_v<T_in, tflite::half>) {
      spec.relative_tolerance = 5e-3;
      spec.absolute_tolerance = 5e-3;
    } else {
      spec.relative_tolerance = 1e-4;
      spec.absolute_tolerance = 1e-4;
    }
    return spec;
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(2);

    const auto lhs_shape = LhsShape(params_);
    const auto rhs_shape = RhsShape(params_);

    auto lhs_builder = data_builder;
    if (!lhs_builder.IsFloatDummy()) {
      const float lhs_scale =
          1.0f / std::sqrt(static_cast<float>(lhs_shape[3]));
      lhs_builder.SetFloatRange(-1.5f * lhs_scale, 1.5f * lhs_scale);
    }

    auto rhs_builder = data_builder;
    if (!rhs_builder.IsFloatDummy()) {
      rhs_builder.SetFloatRange(-1.5f, 1.5f);
    }

    LITERT_ASSIGN_OR_RETURN(auto lhs, SimpleBuffer::Create<T_in>(lhs_shape));
    LITERT_RETURN_IF_ERROR(
        (lhs.template WriteRandom<T_in>(lhs_builder, device)));
    inputs.push_back(std::move(lhs));

    LITERT_ASSIGN_OR_RETURN(auto rhs, SimpleBuffer::Create<T_rhs>(rhs_shape));
    LITERT_RETURN_IF_ERROR(
        (rhs.template WriteRandom<T_rhs>(rhs_builder, device)));
    inputs.push_back(std::move(rhs));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return ReferenceEvaluator::Evaluate(Graph(), inputs, outputs);
  }

  RuntimeBmm(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static std::array<Layout::Dim, 4> LhsShape(const Params& params) {
    return {static_cast<Layout::Dim>(params.batch),
            static_cast<Layout::Dim>(params.num_heads),
            static_cast<Layout::Dim>(params.q_seq_len),
            static_cast<Layout::Dim>(kIsSrc ? params.kv_len : params.head_dim)};
  }

  static std::array<Layout::Dim, 4> RhsShape(const Params& params) {
    return {static_cast<Layout::Dim>(params.batch),
            static_cast<Layout::Dim>(params.num_heads),
            static_cast<Layout::Dim>(kIsSrc ? params.head_dim : params.kv_len),
            static_cast<Layout::Dim>(kIsSrc ? params.kv_len : params.head_dim)};
  }

  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

    const auto lhs_shape = LhsShape(params);
    const auto rhs_shape = RhsShape(params);
    std::vector<int32_t> dims_lhs(lhs_shape.begin(), lhs_shape.end());
    std::vector<int32_t> dims_rhs(rhs_shape.begin(), rhs_shape.end());

    TensorTf lhs =
        litert::tensor::Create(std::string(kInputNames[0]),
                               litert::tensor::ApiType<T_in>::value, dims_lhs);
    TensorTf rhs =
        litert::tensor::Create(std::string(kInputNames[1]),
                               litert::tensor::ApiType<T_rhs>::value, dims_rhs);
    if constexpr (kIsQuantizedRhs) {
      rhs.SetQuantization(
          std::make_shared<litert::tensor::PerChannelAffineQuantization>(
              std::vector<float>{kRhsQuantScale}, std::vector<int64_t>{0}, 0));
    }

    const int32_t seq_len = static_cast<int32_t>(params.kv_len);
    std::vector<int32_t> params_data = {0, seq_len, seq_len, 0, 0, 0, 0};
    auto params_buf =
        litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kI32>(
            params_data);
    TensorTf params_tensor =
        litert::tensor::Create("params", litert::tensor::Type::kI32,
                               {1, 1, 1, 7}, std::move(params_buf));

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Bool("is_global", true);
      fbb.Bool("is_src", kIsSrc);
      if constexpr (kIsQuantizedRhs) {
        fbb.Float("scale", kRhsQuantScale);
      }
    });
    fbb.Finish();

    auto decompose = [](TensorTf d_lhs, TensorTf d_rhs, TensorTf d_params) {
      (void)d_params;
      if constexpr (kIsQuantizedRhs) {
        return litert::tensor::BatchMatMul(
            d_lhs, litert::tensor::Dequantize(d_rhs), /*adj_x=*/false,
            /*adj_y=*/true);
      } else {
        return litert::tensor::BatchMatMul(d_lhs, d_rhs, /*adj_x=*/false,
                                           /*adj_y=*/true);
      }
    };

    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.runtime_bmm",
        .composite_attributes = fbb.GetBuffer(),
    };

    TensorTf output = litert::tensor::StableHLOComposite(
        composite_options, decompose, lhs, rhs, params_tensor);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_RUNTIME_BMM_H_
