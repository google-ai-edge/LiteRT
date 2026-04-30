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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_NPU_OPTIMIZER_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_NPU_OPTIMIZER_H_

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace litert {
namespace openvino {

// Eliminates FakeQuantize nodes that immediately follow MatMul operations.
//
// The FakeQuantize directly after a MatMul re-quantize its output to a
// calibrated range. This pass detects the MatMul -> FakeQuantize pattern
// and removes the FakeQuantize, rewiring downstream consumers to read
// directly from the MatMul output to reduce NPU overhead.
class EliminateMatMulFakeQuantize : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("EliminateMatMulFakeQuantize");
  EliminateMatMulFakeQuantize();
};

// The Intel NPU compiler's IE.Sign op only accepts float operands
// (f16/f32/f64). When the TFLite frontend produces a Sign node with integer
// input, NPU compilation fails. This pass wraps such Sign nodes with Convert
// ops:
//   Convert(int->f32) -> Sign -> Convert(f32->original_int_type)
class CastIntegerSignToFloat : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("CastIntegerSignToFloat");
  CastIntegerSignToFloat();
};

// Configurable runner for NPU-specific optimization passes.
// Use the setter APIs to toggle individual optimizations, then call Run() to
// apply the enabled passes to a model.
class NpuOptimizer {
 public:
  // Toggles the EliminateMatMulFakeQuantize pass. Disabled by default.
  NpuOptimizer& SetEliminateMatMulFakeQuantize(bool enable) {
    eliminate_matmul_fq_ = enable;
    return *this;
  }

  // Toggles the CastIntegerSignToFloat pass. Enabled by default because the
  // NPU plugin cannot lower integer Sign operations.
  NpuOptimizer& SetCastIntegerSignToFloat(bool enable) {
    cast_integer_sign_to_float_ = enable;
    return *this;
  }

  // Runs all currently-enabled passes on |model|.
  void Run(const std::shared_ptr<ov::Model>& model) const;

 private:
  bool eliminate_matmul_fq_ = false;
  bool cast_integer_sign_to_float_ = true;
};

}  // namespace openvino
}  // namespace litert

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_NPU_OPTIMIZER_H_
