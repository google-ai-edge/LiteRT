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

#include "litert/vendors/arm_vulkan_ml/compiler/graph_ir.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace litert::arm_vulkan_ml {
namespace {
GraphValue Value(std::vector<uint32_t> shape = {1, 256}, double scale = .125,
                 int32_t zp = -17) {
  GraphValue value;
  value.shape = std::move(shape); value.scales = {scale}; value.zero_points = {zp};
  return value;
}
QuantizedTensor Q(const GraphValue& value) {
  return {value.shape, value.scales[0], static_cast<int32_t>(value.zero_points[0]), false};
}
GraphNode Function(ValueId input, ValueId output, size_t source, OutputTransform transform) {
  GraphNode node;
  node.kind = CompositeKind::kInt8Function; node.transform_only = true;
  node.inputs = {input}; node.outputs = {output}; node.source_operators = {source};
  node.body.code = kLiteRtOpCodeTflQuantize; node.body.runtime_inputs = {0};
  node.body.inputs = {transform.input}; node.body.output = transform.output;
  node.output_transforms = {std::move(transform)};
  return node;
}
OutputTransform Clamp(const GraphValue& value, int minimum, int maximum) {
  OutputTransform result;
  result.input = result.output = Q(value); result.clamp = {minimum, maximum};
  return result;
}
CompilerGraph Chain(OutputTransform first, OutputTransform second) {
  CompilerGraph graph;
  graph.values = {Value(), Value(), Value()};
  graph.values[0].graph_input = true; graph.values[2].graph_output = true;
  graph.inputs = {0}; graph.outputs = {2}; graph.original_operator_count = 2;
  graph.nodes = {Function(0, 1, 0, std::move(first)), Function(1, 2, 1, std::move(second))};
  return graph;
}
int Apply(int code, const OutputTransform& transform) {
  if (transform.kind == OutputTransform::Kind::kTable) return transform.table[code + 128];
  if (transform.kind == OutputTransform::Kind::kRescale) {
    auto scaled = ApplyScale32(code - transform.input.zero_point, transform.scales[0]);
    EXPECT_TRUE(scaled);
    if (!scaled) return 0;
    code = static_cast<int>(std::clamp(int64_t{*scaled} + transform.output.zero_point,
                                     int64_t{-128}, int64_t{127}));
  }
  return std::clamp(code, transform.clamp.minimum, transform.clamp.maximum);
}
OutputTransform Table(const GraphValue& input, const GraphValue& output, bool affine) {
  auto transform = Clamp(input, -128, 127);
  transform.kind = OutputTransform::Kind::kTable; transform.output = Q(output);
  transform.affine_requantization = affine;
  if (affine) {
    auto table = RequantizeTable(static_cast<float>(input.scales[0]),
        static_cast<int32_t>(input.zero_points[0]), static_cast<float>(output.scales[0]),
        static_cast<int32_t>(output.zero_points[0]));
    EXPECT_TRUE(table);
    if (table) transform.table = *table;
  } else for (int code = -128; code <= 127; ++code)
    transform.table.push_back(static_cast<int8_t>(std::clamp(code * 2 + 9, -128, 127)));
  return transform;
}

TEST(GraphOptimizer, OffPreservesTheImportedGraph) {
  auto graph = Chain(Clamp(Value(), 0, 90), Clamp(Value(), 20, 70));
  const auto before = DumpGraph(graph);
  std::string reason; OptimizationReport report;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationOff, &report, reason)) << reason;
  EXPECT_EQ(DumpGraph(graph), before); EXPECT_TRUE(report.decisions.empty());
}
TEST(GraphOptimizer, ConstantInterningPreservesScalarAndVectorOutputRanks) {
  auto graph = Chain(Clamp(Value(), 0, 90), Clamp(Value(), 20, 70));
  for (bool scalar : {true, false, false}) {
    auto constant = Value({1});
    constant.logical_scalar = scalar;
    constant.is_constant = true;
    constant.graph_output = true;
    constant.constant = {17};
    graph.outputs.push_back(graph.values.size());
    graph.values.push_back(std::move(constant));
  }
  std::string reason;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
  EXPECT_NE(graph.outputs[1], graph.outputs[2]);
  EXPECT_TRUE(graph.values[graph.outputs[1]].logical_scalar);
  EXPECT_FALSE(graph.values[graph.outputs[2]].logical_scalar);
  EXPECT_EQ(graph.outputs[2], graph.outputs[3]);
}
TEST(GraphOptimizer, ClampsComposeExactlyEvenWhenRangesAreDisjoint) {
  for (auto range : {Int8Range{-80, -40}, Int8Range{10, 30}, Int8Range{60, 120}}) {
    auto first = Clamp(Value(), -10, 50), second = Clamp(Value(), range.minimum, range.maximum);
    auto graph = Chain(first, second);
    std::string reason;
    ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
    ASSERT_TRUE(graph.nodes[1].removed);
    ASSERT_EQ(graph.nodes[0].output_transforms.size(), 1);
    for (int code = -128; code <= 127; ++code)
      EXPECT_EQ(Apply(Apply(code, first), second), Apply(code, graph.nodes[0].output_transforms[0]));
    EXPECT_EQ(graph.nodes[0].source_operators, (std::vector<size_t>{0, 1}));
  }
}
TEST(GraphOptimizer, GraphOutputsAndFanoutAreFusionBarriers) {
  for (bool exposed : {false, true}) {
    auto graph = Chain(Clamp(Value(), -10, 50), Clamp(Value(), 0, 30));
    if (exposed) { graph.outputs.push_back(1); graph.values[1].graph_output = true; }
    else {
      graph.values.push_back(Value()); graph.values[3].graph_output = true;
      graph.nodes.push_back(Function(1, 3, 2, Clamp(Value(), -30, 0)));
      graph.outputs.push_back(3); graph.original_operator_count = 3;
    }
    std::string reason; OptimizationReport report;
    ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, &report, reason)) << reason;
    EXPECT_FALSE(graph.nodes[1].removed);
    EXPECT_TRUE(std::any_of(report.decisions.begin(), report.decisions.end(),
        [](const auto& d) { return !d.applied && d.reason.find("additional consumers") != std::string::npos; }));
  }
}
TEST(GraphOptimizer, LookupCompositionPreservesAllIntermediateRoundingAndSaturation) {
  auto first = Table(Value(), Value(), false), second = first;
  for (size_t i = 0; i < second.table.size(); ++i) second.table[i] = static_cast<int8_t>(127 - static_cast<int>(i));
  auto graph = Chain(first, second);
  std::string reason;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
  ASSERT_EQ(graph.nodes[0].output_transforms.size(), 1);
  for (int code = -128; code <= 127; ++code)
    EXPECT_EQ(Apply(Apply(code, first), second), Apply(code, graph.nodes[0].output_transforms[0]));
}
TEST(GraphOptimizer, NativeRequantizationIsExhaustivelyBounded) {
  for (double ratio : {.25, .3, .5, 1.0, 2.0, 6.0, 1e-8}) {
    for (int32_t izp : {-128, -17, 127}) for (int32_t ozp : {-128, 7, 127}) {
      auto input = Value({1, 256}, ratio, izp), output = Value({1, 256}, 1, ozp);
      auto table = Table(input, output, true);
      for (auto mode : {kLrtArmVulkanMLOptimizationExact, kLrtArmVulkanMLOptimizationQualified}) {
        CompilerGraph graph;
        graph.values = {input, output}; graph.values[0].graph_input = true; graph.values[1].graph_output = true;
        graph.inputs = {0}; graph.outputs = {1}; graph.original_operator_count = 1;
        graph.nodes = {Function(0, 1, 0, table)};
        std::string reason;
        ASSERT_TRUE(OptimizeGraph(graph, mode, nullptr, reason)) << reason;
        const auto& transformed = graph.nodes[0].output_transforms[0];
        int total = 0, maximum = 0;
        for (int code = -128; code <= 127; ++code) {
          int error = std::abs(Apply(code, table) - Apply(code, transformed));
          maximum = std::max(maximum, error); total += error;
        }
        EXPECT_LE(maximum, mode == kLrtArmVulkanMLOptimizationExact ? 0 : 1);
        EXPECT_LE(total, mode == kLrtArmVulkanMLOptimizationExact ? 0 : 64);
      }
    }
  }
}
TEST(GraphOptimizer, LocallyBoundedApproximationRemainsDisabledAfterCorpusFailure) {
  auto input = Value({1, 256}, 0.0071264510042965412, -128);
  auto output = Value({1, 256}, 0.014890247955918312, -6);
  auto transform = Table(input, output, true);
  CompilerGraph graph;
  graph.values = {input, output}; graph.values[0].graph_input = true; graph.values[1].graph_output = true;
  graph.inputs = {0}; graph.outputs = {1}; graph.original_operator_count = 1;
  graph.nodes = {Function(0, 1, 0, transform)};
  std::string reason; OptimizationReport report;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationQualified, &report, reason)) << reason;
  EXPECT_EQ(graph.nodes[0].output_transforms[0].kind, OutputTransform::Kind::kTable);
  EXPECT_TRUE(std::any_of(report.decisions.begin(), report.decisions.end(), [](const auto& d) {
    return !d.applied && d.reason.find("corpus gate failed") != std::string::npos;
  }));
}

TEST(GraphOptimizer, NativeTableNormalizationPrecedesLookupFolding) {
  auto input = Value({1, 256}, 2, 0), output = Value({1, 256}, 1, 0);
  auto graph = Chain(Table(input, output, true), Table(output, output, false));
  graph.values[0] = input; graph.values[0].graph_input = true;
  graph.values[1] = output; graph.values[2] = output; graph.values[2].graph_output = true;
  std::string reason; OptimizationReport report;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, &report, reason)) << reason;
  ASSERT_EQ(graph.nodes[0].output_transforms.size(), 2);
  EXPECT_EQ(graph.nodes[0].output_transforms[0].kind, OutputTransform::Kind::kRescale);
  EXPECT_EQ(graph.nodes[0].output_transforms[1].kind, OutputTransform::Kind::kTable);
}
TEST(GraphOptimizer, RoundedScalesAreNeverMultiplied) {
  auto value = Value(); auto transform = Clamp(value, -120, 110);
  transform.kind = OutputTransform::Kind::kRescale;
  transform.scales = {*MakeScale32(.3)}; transform.ranges = {{-111, 144}};
  auto graph = Chain(transform, transform);
  std::string reason;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
  ASSERT_EQ(graph.nodes[0].output_transforms.size(), 2);
  for (int code = -128; code <= 127; ++code)
    EXPECT_EQ(Apply(Apply(code, transform), transform),
        Apply(Apply(code, graph.nodes[0].output_transforms[0]), graph.nodes[0].output_transforms[1]));
}

GraphNode Shape(ValueId input, ValueId output, size_t source,
                const GraphValue& in, const GraphValue& out,
                std::vector<int32_t> permutation) {
  GraphNode node;
  node.body.code = permutation.empty() ? kLiteRtOpCodeTflReshape : kLiteRtOpCodeTflTranspose;
  node.body.inputs = {Q(in)}; node.body.output = Q(out); node.body.parameters = std::move(permutation);
  node.body.runtime_inputs = {0}; node.inputs = {input}; node.outputs = {output}; node.source_operators = {source};
  return node;
}
TEST(GraphOptimizer, TransposeCompositionAndUnitAxesUseMetadataReshapes) {
  CompilerGraph graph;
  graph.values = {Value({1, 2, 3, 1}), Value({1, 3, 2, 1}), Value({1, 2, 3, 1})};
  graph.values[0].graph_input = true; graph.values[2].graph_output = true;
  graph.inputs = {0}; graph.outputs = {2}; graph.original_operator_count = 2;
  graph.nodes = {Shape(0, 1, 0, graph.values[0], graph.values[1], {0, 2, 1, 3}),
                 Shape(1, 2, 1, graph.values[1], graph.values[2], {0, 2, 1, 3})};
  std::string reason;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
  EXPECT_TRUE(graph.nodes[0].removed); EXPECT_EQ(graph.nodes[1].inputs[0], 0);
  EXPECT_EQ(graph.nodes[1].body.code, kLiteRtOpCodeTflReshape);
}
TEST(GraphOptimizer, IdentityShapeDoesNotAttributeUnrelatedInputConsumers) {
  CompilerGraph graph;
  graph.values = {Value(), Value(), Value(), Value()};
  graph.values[0].graph_input = true; graph.values[2].graph_output = true; graph.values[3].graph_output = true;
  graph.inputs = {0}; graph.outputs = {2, 3}; graph.original_operator_count = 3;
  graph.nodes = {Shape(0, 1, 0, graph.values[0], graph.values[1], {}),
      Function(0, 2, 1, Clamp(Value(), -10, 50)), Function(1, 3, 2, Clamp(Value(), 0, 30))};
  std::string reason;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
  EXPECT_TRUE(graph.nodes[0].removed);
  EXPECT_EQ(graph.nodes[1].source_operators, (std::vector<size_t>{1}));
  EXPECT_EQ(graph.nodes[2].source_operators, (std::vector<size_t>{0, 2}));
}

TEST(GraphOptimizer, ConstantTransposeFoldsOwnedBytesAndDeadConstantsDisappear) {
  CompilerGraph graph;
  graph.values = {Value({1, 2, 3, 1}), Value({1, 3, 2, 1})};
  graph.values[0].is_constant = true; graph.values[0].constant = {0, 1, 2, 3, 4, 5};
  graph.values[1].graph_output = true; graph.outputs = {1}; graph.original_operator_count = 1;
  graph.nodes = {Shape(0, 1, 0, graph.values[0], graph.values[1], {0, 2, 1, 3})};
  std::string reason;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
  EXPECT_TRUE(graph.nodes[0].removed);
  EXPECT_EQ(graph.values[1].constant, (std::vector<uint8_t>{0, 3, 1, 4, 2, 5}));
  EXPECT_TRUE(graph.values[0].constant.empty());
}
CompilerGraph PaddedConvolution() {
  CompilerGraph graph;
  graph.values = {Value({1, 4, 4, 1}), Value({1, 6, 6, 1}), Value({1, 4, 4, 2})};
  graph.inputs = {0}; graph.outputs = {2}; graph.values[0].graph_input = true; graph.values[2].graph_output = true;
  graph.original_operator_count = 2;
  GraphNode pad = Shape(0, 1, 0, graph.values[0], graph.values[1], {});
  pad.body.code = kLiteRtOpCodeTflPad; pad.body.pad_value = -17;
  pad.body.parameters = {0, 0, 1, 1, 1, 1, 0, 0};
  Conv2dSpec spec{};
  spec.input_shape = {1, 6, 6, 1}; spec.weight_shape = {2, 3, 3, 1}; spec.output_shape = {1, 4, 4, 2};
  spec.stride = {1, 1}; spec.dilation = {1, 1}; spec.weights.assign(18, 1);
  spec.bias = {0, 1073741824}; spec.input_scale = .125; spec.output_scale = .25;
  spec.weight_scales = {.5, 1e-12}; spec.bias_scales = {.0625, 1.25e-13};
  spec.input_zero_point = -17; spec.output_zero_point = -17; spec.activation = Activation::kRelu6;
  std::string reason; auto analyzed = AnalyzeConv2d(spec, reason); EXPECT_TRUE(analyzed) << reason;
  GraphNode conv = Shape(1, 2, 1, graph.values[1], graph.values[2], {});
  conv.kind = CompositeKind::kConvolution; conv.body.code = kLiteRtOpCodeTflConv2d;
  conv.body.convolution = std::move(analyzed);
  graph.nodes = {pad, conv}; return graph;
}
TEST(GraphOptimizer, RealZeroSpatialPadFoldsWithPerChannelTinyScalesAndLargeBias) {
  auto graph = PaddedConvolution();
  const auto original = *graph.nodes[1].body.convolution;
  std::string reason;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
  EXPECT_TRUE(graph.nodes[0].removed); EXPECT_EQ(graph.nodes[1].inputs[0], 0);
  const auto& folded = *graph.nodes[1].body.convolution;
  EXPECT_EQ(folded.spec.padding, (std::array<int32_t, 4>{1, 1, 1, 1}));
  EXPECT_EQ(folded.spec.input_shape, (std::array<uint32_t, 4>{1, 4, 4, 1}));
  for (size_t c = 0; c < original.scales.size(); ++c)
    for (int32_t value : {original.accumulator_ranges[c].minimum, 0, original.accumulator_ranges[c].maximum})
      EXPECT_EQ(ApplyScale32(value, original.scales[c]), ApplyScale32(value, folded.scales[c]));
}
TEST(GraphOptimizer, PadFoldingRejectsNonzeroFillNonspatialPaddingAndExposedValues) {
  for (int variant = 0; variant < 4; ++variant) {
    auto graph = PaddedConvolution();
    if (variant == 0) graph.nodes[0].body.pad_value = 0;
    if (variant == 1) graph.nodes[0].body.parameters[0] = 1;
    if (variant == 2) graph.values[1].graph_output = true;
    if (variant == 3) graph.nodes[1].body.convolution->spec.padding[0] = 1;
    std::string reason;
    ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
    EXPECT_FALSE(graph.nodes[0].removed);
  }
}
TEST(GraphOptimizer, PersistentValuesAreFusionAndCleanupBarriers) {
  auto graph = Chain(Clamp(Value(), -10, 50), Clamp(Value(), 0, 30));
  graph.values[1].persistent = true;
  std::string reason;
  ASSERT_TRUE(OptimizeGraph(graph, kLrtArmVulkanMLOptimizationExact, nullptr, reason)) << reason;
  EXPECT_FALSE(graph.nodes[0].removed); EXPECT_FALSE(graph.nodes[1].removed);
}

TEST(GraphOptimizer, ValidationRejectsUnboundValuesAndMalformedTransforms) {
  auto graph = Chain(Clamp(Value(), 0, 90), Clamp(Value(), 20, 70));
  std::string reason;
  graph.nodes[0].inputs = {2}; EXPECT_FALSE(ValidateGraph(graph, reason));
  graph.nodes[0].inputs = {0}; graph.nodes[0].output_transforms[0].clamp = {20, 10};
  EXPECT_FALSE(ValidateGraph(graph, reason));
  graph.nodes[0].output_transforms[0].clamp = {0, 90};
  graph.nodes[0].output_transforms[0].output.zero_point = 3;
  EXPECT_FALSE(ValidateGraph(graph, reason));
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
