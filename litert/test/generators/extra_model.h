// Copyright 2024 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_EXTRA_MODEL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_EXTRA_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/test/generators/common.h"
#include "litert/test/simple_buffer.h"

namespace litert::testing {

// Container for test graphs that originate from raw .tflite passed to the test.
class ExtraModel : public TestGraph {
 public:
  using Ptr = std::unique_ptr<ExtraModel>;

  static constexpr absl::string_view Name() { return "ExtraModel"; }

  bool HasReference() const override { return false; }

  static Expected<ExtraModel::Ptr> Create(absl::string_view model_path) {
    LITERT_ASSIGN_OR_RETURN(auto model,
                            internal::LoadModelFromFile(model_path));
    return std::make_unique<ExtraModel>(std::move(model));
  }

  // Defers loading the .tflite file from disk until `EnsureLoaded()` is called
  // during test SetUp(), avoiding OOMs from keeping all registered models in
  // memory simultaneously.
  static ExtraModel::Ptr CreateLazy(absl::string_view model_path) {
    return std::make_unique<ExtraModel>(model_path);
  }

  Expected<void> EnsureLoaded() override {
    if (model_ != nullptr) {
      return {};
    }
    LITERT_ASSIGN_OR_RETURN(model_, internal::LoadModelFromFile(model_path_));
    InitInputBounds();
    return {};
  }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    const auto& sg_inputs = Graph().Subgraph(0).Inputs();
    for (size_t i = 0; i < sg_inputs.size(); ++i) {
      LITERT_ASSIGN_OR_RETURN(auto t, (sg_inputs[i]->Ranked()));
      LITERT_ASSIGN_OR_RETURN(auto b,
                              SimpleBuffer::Create(RankedTensorType(t)));
      auto builder = data_builder;
      const int32_t upper_bound =
          (i < input_upper_bounds_.size() && input_upper_bounds_[i].has_value())
              ? *input_upper_bounds_[i]
              : kDefaultIntUpperBound;
      if (t.element_type == kLiteRtElementTypeInt32) {
        builder.SetIntRange(0, upper_bound - 1);
      } else if (t.element_type == kLiteRtElementTypeInt64) {
        builder.SetInt64Range(0, upper_bound - 1);
      }
      LITERT_RETURN_IF_ERROR((b.WriteRandom(builder, device)));
      inputs.push_back(std::move(b));
    }
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    return Error(kLiteRtStatusErrorUnsupported);
  }

  explicit ExtraModel(LiteRtModelT::Ptr model) : TestGraph(std::move(model)) {
    InitInputBounds();
  }
  explicit ExtraModel(absl::string_view model_path)
      : model_path_(model_path) {}

 private:
  // Default exclusive upper bound (`[0, 1]`) for integer inputs when no lookup
  // table dimension bound is found, avoiding overflow in downstream arithmetic
  // or mask ops.
  static constexpr int32_t kDefaultIntUpperBound = 2;

  static bool IsIntTensor(const LiteRtTensorT& tensor) {
    auto t = tensor.Ranked();
    return t && (t->element_type == kLiteRtElementTypeInt32 ||
                 t->element_type == kLiteRtElementTypeInt64);
  }

  // Returns the exclusive upper bound imposed by a table lookup `op` on its
  // integer index operand at `arg_ind`:
  // - `EmbeddingLookup` (`arg_ind == 0`): table dimension 0 (`vocab_size`).
  // - `Gather` / `GatherNd` (`arg_ind == 1`): table dimension `axis`.
  static std::optional<int32_t> GetLookupOpUpperBound(
      const LiteRtOpT& op, LiteRtParamIndex arg_ind) {
    if (op.OpCode() == kLiteRtOpCodeTflEmbeddingLookup && arg_ind == 0 &&
        op.NumInputs() >= 2) {
      auto t = op.Input(1).Ranked();
      if (t && t->layout.rank >= 1 && t->layout.dimensions[0] > 0) {
        return t->layout.dimensions[0];
      }
    } else if ((op.OpCode() == kLiteRtOpCodeTflGather ||
                op.OpCode() == kLiteRtOpCodeTflGatherNd) &&
               arg_ind == 1 && op.NumInputs() >= 2) {
      auto t = op.Input(0).Ranked();
      const auto* opts = internal::GetTflOptions(op).AsGatherOptions();
      int32_t axis = opts ? opts->axis : 0;
      if (t && axis < 0) axis += t->layout.rank;
      if (t && axis >= 0 && axis < static_cast<int32_t>(t->layout.rank) &&
          t->layout.dimensions[axis] > 0) {
        return t->layout.dimensions[axis];
      }
    }
    return std::nullopt;
  }

  // Returns true if `op` is an integer `Cast` or `Reshape` adapter on its
  // primary input (`arg_ind == 0`), which TFLite inserts on input edges before
  // lookup ops (e.g. `INT64 -> Cast(INT32) -> Reshape -> Gather`).
  static bool IsIntAdapterOp(const LiteRtOpT& op, LiteRtParamIndex arg_ind) {
    return arg_ind == 0 && op.NumOutputs() == 1 &&
           (op.OpCode() == kLiteRtOpCodeTflCast ||
            op.OpCode() == kLiteRtOpCodeTflReshape) &&
           IsIntTensor(op.Output(0));
  }

  // Inspects an integer subgraph input's consumer ops (stepping across any
  // `Cast` / `Reshape` input adapters) to find its exclusive upper bound.
  static std::optional<int32_t> FindInputUpperBound(
      const LiteRtTensorT& input) {
    if (!IsIntTensor(input)) return std::nullopt;
    for (size_t i = 0; i < input.NumUses(); ++i) {
      auto [op, arg_ind] = input.GetUse(i);
      while (IsIntAdapterOp(*op, arg_ind)) {
        const auto& out = op->Output(0);
        if (out.NumUses() != 1) break;
        std::tie(op, arg_ind) = out.GetUse(0);
      }
      if (auto bound = GetLookupOpUpperBound(*op, arg_ind)) {
        return bound;
      }
    }
    return std::nullopt;
  }

  // Precomputes per-input integer upper bounds from `subgraph.Inputs()`.
  void InitInputBounds() {
    if (model_ == nullptr || model_->NumSubgraphs() == 0) return;
    for (const auto* input : Graph().Subgraph(0).Inputs()) {
      input_upper_bounds_.push_back(FindInputUpperBound(*input));
    }
  }

  std::string model_path_;
  // Exclusive upper bound `N` (clamping generated values to `[0, N)`) for each
  // `Subgraph(0)` input, or `std::nullopt` if the input is unconstrained.
  std::vector<std::optional<int32_t>> input_upper_bounds_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_EXTRA_MODEL_H_
