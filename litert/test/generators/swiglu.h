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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SWIGLU_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SWIGLU_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
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
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename T>
class Swiglu : public TestGraph {
 public:
  struct Params {
    size_t batch = 1;
    size_t seq_len = 1;
    size_t hidden_dim = 256;
  };

  using Ptr = std::unique_ptr<Swiglu>;

  static constexpr absl::string_view Name() { return "Swiglu"; }

  struct GridConfig {
    size_t batch;
    size_t seq_len;
    size_t hidden_dim;
  };

  // Stratified grid of 16 representative LLM MLP / SwiGLU workloads.
  // ATS test suites should specify iters >= 16 to ensure full coverage.
  static constexpr GridConfig kStratifiedGrid[] = {
      // ─── 1. Decode Workloads (Single Token, seq_len = 1) ───
      {1, 1, 256},   // 0: Tiny / mobile decode
      {1, 1, 512},   // 1: Small decode
      {1, 1, 1024},  // 2: Medium decode
      {1, 1, 2048},  // 3: Large decode
      {1, 1, 4096},  // 4: Extra-large decode

      // ─── 2. Short / Chunked Context Workloads ───
      {1, 8, 512},    // 5: 8-token chunk
      {1, 16, 1024},  // 6: 16-token chunk
      {1, 32, 2048},  // 7: 32-token chunk
      {1, 64, 1024},  // 8: 64-token chunk

      // ─── 3. Prefill Workloads (Extended Sequence) ───
      {1, 128, 1024},  // 9: 128-token prefill
      {1, 256, 1024},  // 10: 256-token prefill
      {1, 512, 512},   // 11: 512-token prefill
      {1, 1024, 256},  // 12: 1K-token long prefill

      // ─── 4. Batched Workloads ───
      {2, 1, 1024},  // 13: Batch-2 decode
      {2, 64, 512},  // 14: Batch-2 sequence
      {4, 1, 512},   // 15: Batch-4 decode
  };

  template <typename Rng>
  static Expected<Swiglu::Ptr> Create(Rng& /*rng*/) {
    static constexpr size_t kGridSize =
        sizeof(kStratifiedGrid) / sizeof(kStratifiedGrid[0]);
    static size_t sample_counter = 0;
    const auto& entry = kStratifiedGrid[(sample_counter++) % kGridSize];

    Params params;
    params.batch = entry.batch;
    params.seq_len = entry.seq_len;
    params.hidden_dim = entry.hidden_dim;

    return Create(std::move(params));
  }

  static Expected<Swiglu::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Swiglu>(std::move(params), std::move(model));
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

    std::array<Layout::Dim, 3> shape = {
        static_cast<Layout::Dim>(params_.batch),
        static_cast<Layout::Dim>(params_.seq_len),
        static_cast<Layout::Dim>(2 * params_.hidden_dim)};

    auto builder = data_builder;
    if (!builder.IsFloatDummy()) {
      builder.SetFloatRange(-2.0f, 2.0f);
    }

    LITERT_ASSIGN_OR_RETURN(auto in, SimpleBuffer::Create<T>(shape));
    LITERT_RETURN_IF_ERROR((in.template WriteRandom<T>(builder, device)));
    inputs.push_back(std::move(in));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return ReferenceEvaluator::EvaluateCompositeReference(Graph(), inputs,
                                                           outputs);
  }

  Swiglu(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

    int b = static_cast<int>(params.batch);
    int s = static_cast<int>(params.seq_len);
    int h = static_cast<int>(params.hidden_dim);

    TensorTf gate_up = litert::tensor::Create(
        "gate_up", litert::tensor::ApiType<T>::value, {b, s, 2 * h});

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("gate_size", h);
    });
    fbb.Finish();
    auto composite_attributes = fbb.GetBuffer();

    auto decompose = [b, s, h](auto in) {
      auto gate = litert::tensor::Slice(in, {0, 0, 0}, {b, s, h});
      auto up = litert::tensor::Slice(in, {0, 0, h}, {b, s, h});
      auto silu_gate =
          litert::tensor::Mul(gate, litert::tensor::Logistic(gate));
      return litert::tensor::Mul(silu_gate, up);
    };

    litert::tensor::StableHLOCompositeOptions composite_options{
        .name = "odml.swiglu",
        .composite_attributes = composite_attributes,
    };

    TensorTf output = litert::tensor::StableHLOComposite(composite_options,
                                                         decompose, gate_up);
    output.SetName("output");
    return litert::testing::SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SWIGLU_H_
