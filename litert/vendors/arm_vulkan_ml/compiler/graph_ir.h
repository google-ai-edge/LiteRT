// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_GRAPH_IR_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_GRAPH_IR_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "litert/c/options/litert_arm_vulkan_ml_options.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"

namespace litert::arm_vulkan_ml {
using OptimizationMode = LrtArmVulkanMLOptimizationMode;
using ValueId = size_t;

// Owned, target-independent values. Model handles are identities only: no
// rewrite or emitter dereferences them after import. INT16 may be represented
// here in future, but is not admitted by the current operation analysis.
struct GraphValue {
  uintptr_t original_tensor = 0;
  LiteRtElementType logical_type = kLiteRtElementTypeInt8;
  tfsc::tosa::DataType storage_type = tfsc::tosa::DataType::int8_t;
  std::vector<uint32_t> shape;
  bool logical_scalar = false;
  std::vector<double> scales;
  std::vector<int64_t> zero_points;
  int32_t quantized_dimension = 0;
  std::vector<uint8_t> constant;
  bool is_constant = false;
  bool graph_input = false;
  bool graph_output = false;
  // Reserved for mutable/persistent values if a future frontend admits them.
  bool persistent = false;
};

// Ordered INT8 epilogues retain intermediate rounding and saturation. Never
// combine two RESCALEs by multiplying scales. A composite's analyzed body owns
// its original accumulator/bias/rescale recipe; these are subsequent transforms.
struct OutputTransform {
  enum class Kind { kClamp, kTable, kRescale };
  Kind kind = Kind::kClamp;
  QuantizedTensor input{}, output{};
  Int8Range clamp{-128, 127};
  std::vector<int8_t> table;
  std::vector<Scale32> scales;
  std::vector<AccumulatorRange> ranges;
  bool affine_requantization = false;
  bool approximate = false;
};

enum class CompositeKind {
  kAnalyzedOperation, kConvolution, kDepthwiseConvolution,
  kTransposeConvolution, kBinary, kFullyConnected, kPool, kInt8Function, kSoftmax
};
struct GraphNode {
  CompositeKind kind = CompositeKind::kAnalyzedOperation;
  OperationPlan body{};
  std::vector<ValueId> inputs, outputs;
  std::vector<size_t> source_operators;
  std::vector<OutputTransform> output_transforms;
  bool transform_only = false;
  bool removed = false;
};
struct CompilerGraph {
  std::vector<GraphValue> values;
  std::vector<GraphNode> nodes;
  std::vector<ValueId> inputs, outputs;
  size_t original_operator_count = 0;
};
struct OptimizationDecision {
  std::string pass;
  std::vector<size_t> source_operators;
  bool applied = false;
  std::string reason;
};
struct OptimizationReport {
  std::string before_ir, after_ir;
  std::vector<OptimizationDecision> decisions;
  size_t emitted_operators = 0, constant_count = 0, constant_bytes = 0;
  double compile_ms = 0;
};

const char* OptimizationModeName(OptimizationMode mode);
std::optional<OptimizationMode> ParseOptimizationMode(const std::string& name);
const char* CompositeKindName(CompositeKind kind);
std::optional<CompilerGraph> ImportGraph(const Subgraph& graph,
                                       std::string& reason, SoftmaxLutType softmax_lut = kLrtArmVulkanMLSoftmaxLutAuto);
bool ValidateGraph(const CompilerGraph& graph, std::string& reason);
bool OptimizeGraph(CompilerGraph& graph, OptimizationMode mode,
                   OptimizationReport* report, std::string& reason);
std::string DumpGraph(const CompilerGraph& graph);
tfsc::tosa::ResId EmitOutputTransform(tfsc::tosa::Graph& graph,
    tfsc::tosa::ResId input, const OutputTransform& transform,
    GraphConstantPool& constants);
}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_GRAPH_IR_H_
