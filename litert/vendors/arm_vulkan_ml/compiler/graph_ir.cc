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
#include <cstring>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "litert/vendors/arm_vulkan_ml/compiler/portable_ops.h"

namespace litert::arm_vulkan_ml {
namespace {
using tfsc::tosa::DataType;
using tfsc::tosa::Attribute;
using tfsc::tosa::ResId;
using TosaTensor = tfsc::tosa::Tensor;

bool SameQuantization(const GraphValue& a, const GraphValue& b) {
  return a.logical_type == b.logical_type && a.storage_type == b.storage_type &&
         a.scales == b.scales && a.zero_points == b.zero_points &&
         a.quantized_dimension == b.quantized_dimension;
}
QuantizedTensor Quantized(const GraphValue& value) {
  return {value.shape, value.scales.empty() ? 1 : value.scales[0],
          value.zero_points.empty() ? 0 : static_cast<int32_t>(value.zero_points[0]),
          value.storage_type == DataType::float32_t};
}
size_t ElementSize(const GraphValue& value) {
  return value.storage_type == DataType::int32_t ||
                 value.storage_type == DataType::float32_t ? 4 : 1;
}
bool IsShape(const GraphNode& node) {
  return !node.body.portable && !node.body.typed && !node.transform_only &&
         node.output_transforms.empty() && node.inputs.size() == 1 &&
         node.outputs.size() == 1 &&
         (node.body.code == kLiteRtOpCodeTflReshape ||
          node.body.code == kLiteRtOpCodeTflTranspose);
}
CompositeKind Classify(const OperationPlan& plan) {
  if (plan.typed || plan.portable || plan.decomposition || plan.inputs[0].floating || plan.output.floating)
    return CompositeKind::kAnalyzedOperation;
  switch (plan.code) {
    case kLiteRtOpCodeTflSoftmax: return CompositeKind::kSoftmax;
    case kLiteRtOpCodeTflConv2d: return CompositeKind::kConvolution;
    case kLiteRtOpCodeTflDepthwiseConv2d: return CompositeKind::kDepthwiseConvolution;
    case kLiteRtOpCodeTflTransposeConv: return CompositeKind::kTransposeConvolution;
    case kLiteRtOpCodeTflFullyConnected: return CompositeKind::kFullyConnected;
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflMul: return CompositeKind::kBinary;
    case kLiteRtOpCodeTflAveragePool2d:
    case kLiteRtOpCodeTflMaxPool2d: return CompositeKind::kPool;
    default: return CompositeKind::kAnalyzedOperation;
  }
}
void Origins(GraphNode& target, const GraphNode& other) {
  target.source_operators.insert(target.source_operators.end(),
      other.source_operators.begin(), other.source_operators.end());
  std::sort(target.source_operators.begin(), target.source_operators.end());
  target.source_operators.erase(std::unique(target.source_operators.begin(),
      target.source_operators.end()), target.source_operators.end());
}
void Decision(OptimizationReport* report, const char* pass,
              const GraphNode& node, bool applied, const std::string& reason) {
  if (report) report->decisions.push_back({pass, node.source_operators, applied, reason});
}
void ReplaceUses(CompilerGraph& graph, ValueId old_value, ValueId new_value) {
  for (auto& node : graph.nodes) if (!node.removed)
    for (auto& input : node.inputs) if (input == old_value) input = new_value;
  for (auto& output : graph.outputs) if (output == old_value) output = new_value;
}
struct Uses {
  std::vector<size_t> count;
  std::vector<size_t> producer;
  explicit Uses(const CompilerGraph& graph)
      : count(graph.values.size()), producer(graph.values.size(), graph.nodes.size()) {
    for (size_t i = 0; i < graph.nodes.size(); ++i) if (!graph.nodes[i].removed) {
      for (auto input : graph.nodes[i].inputs) ++count[input];
      for (auto output : graph.nodes[i].outputs) producer[output] = i;
    }
  }
};
bool Removable(const CompilerGraph& graph, ValueId value, const Uses& uses) {
  return uses.count[value] == 1 && !graph.values[value].graph_input &&
         !graph.values[value].graph_output && !graph.values[value].persistent;
}
Int8Range ComposeClamps(Int8Range first, Int8Range second) {
  return {std::clamp(first.minimum, second.minimum, second.maximum),
          std::clamp(first.maximum, second.minimum, second.maximum)};
}
// All currently admitted scales are SINGLE_ROUND. Reducing an even multiplier
// and its shift leaves the exact rational/rounding boundary unchanged. Recheck
// the input domain, since shifts below 32 have a narrower legal range. Never
// apply this rule to a future DOUBLE_ROUND stage without a separate proof.
bool NormalizeScale(Scale32& scale, AccumulatorRange range) {
  bool changed = false;
  while (scale.shift > 2 && scale.multiplier > 0 && scale.multiplier % 2 == 0) {
    Scale32 candidate{scale.multiplier / 2, static_cast<int8_t>(scale.shift - 1)};
    if (!IsScale32RangeValid(candidate, range.minimum, range.maximum)) break;
    scale = candidate;
    changed = true;
  }
  return changed;
}
void NormalizeScales(GraphNode& node, OptimizationReport* report) {
  bool changed = false;
  for (auto& scale : node.body.scales)
    changed |= NormalizeScale(scale, {INT32_MIN, INT32_MAX});
  if (node.body.convolution) {
    auto& conv = *node.body.convolution;
    for (size_t i = 0; i < conv.scales.size(); ++i)
      changed |= NormalizeScale(conv.scales[i], conv.accumulator_ranges[i]);
  }
  if (changed) Decision(report, "normalize_scales", node, true,
                         "exact rational value, SINGLE_ROUND and legal input domain preserved");
}

bool FoldConstantLayout(CompilerGraph& graph, GraphNode& node,
                        OptimizationReport* report) {
  if (!IsShape(node) || !graph.values[node.inputs[0]].is_constant ||
      graph.values[node.inputs[0]].persistent || graph.values[node.outputs[0]].persistent) return false;
  const auto& input = graph.values[node.inputs[0]];
  auto& output = graph.values[node.outputs[0]];
  if (!SameQuantization(input, output)) return false;
  output.constant.resize(input.constant.size());
  if (node.body.code == kLiteRtOpCodeTflReshape) output.constant = input.constant;
  else {
    const size_t rank = input.shape.size(), width = ElementSize(input);
    std::vector<size_t> strides(rank, 1);
    for (size_t i = rank; i > 1; --i) strides[i - 2] = strides[i - 1] * input.shape[i - 1];
    for (size_t i = 0; i < output.constant.size() / width; ++i) {
      size_t remaining = i, offset = 0;
      for (size_t j = rank; j > 0; --j) {
        size_t axis = j - 1;
        offset += (remaining % output.shape[axis]) * strides[node.body.parameters[axis]];
        remaining /= output.shape[axis];
      }
      std::memcpy(output.constant.data() + i * width, input.constant.data() + offset * width, width);
    }
  }
  output.is_constant = true;
  node.removed = true;
  Decision(report, "fold_constant_layout", node, true, "owned immutable bytes transformed before emission");
  return true;
}

bool ShapePass(CompilerGraph& graph, OptimizationReport* report) {
  bool changed = false;
  Uses uses(graph);
  for (auto& node : graph.nodes) {
    if (node.removed || !IsShape(node)) continue;
    if (FoldConstantLayout(graph, node, report)) { changed = true; continue; }
    if (node.body.code == kLiteRtOpCodeTflTranspose) {
      const auto& shape = graph.values[node.inputs[0]].shape;
      std::vector<int32_t> before, after;
      for (size_t i = 0; i < shape.size(); ++i) if (shape[i] != 1) before.push_back(static_cast<int32_t>(i));
      for (auto axis : node.body.parameters) if (shape[axis] != 1) after.push_back(axis);
      if (before == after) {
        node.body.code = kLiteRtOpCodeTflReshape;
        node.body.parameters.assign(node.body.output.shape.begin(), node.body.output.shape.end());
        Decision(report, "metadata_transpose", node, true, "only unit axes move; element order unchanged");
        changed = true;
      }
    }
    auto input = node.inputs[0], output = node.outputs[0];
    if (node.body.code == kLiteRtOpCodeTflReshape &&
        graph.values[input].shape == graph.values[output].shape &&
        !graph.values[output].graph_output && !graph.values[output].persistent &&
        uses.count[output] == 1) {
      node.removed = true;
      // Transfer provenance only to the consumers being rewired. Other users
      // of the original input do not belong to this eliminated operation.
      for (auto& consumer : graph.nodes) if (!consumer.removed &&
          std::find(consumer.inputs.begin(), consumer.inputs.end(), output) != consumer.inputs.end()) Origins(consumer, node);
      ReplaceUses(graph, output, input);
      Decision(report, "identity_shape", node, true, "same shape and element order");
      changed = true;
      continue;
    }
    size_t pi = uses.producer[input];
    if (pi == graph.nodes.size() || graph.nodes[pi].removed || !IsShape(graph.nodes[pi])) continue;
    auto& previous = graph.nodes[pi];
    if (!Removable(graph, input, uses)) {
      Decision(report, "compose_shapes", node, false, "intermediate is graph IO or has additional consumers");
      continue;
    }
    if (node.body.code != previous.body.code) continue;
    if (node.body.code == kLiteRtOpCodeTflTranspose) {
      auto permutation = node.body.parameters;
      for (size_t i = 0; i < permutation.size(); ++i)
        node.body.parameters[i] = previous.body.parameters[permutation[i]];
    }
    node.inputs[0] = previous.inputs[0];
    node.body.inputs[0] = Quantized(graph.values[node.inputs[0]]);
    previous.removed = true;
    Origins(node, previous);
    Decision(report, "compose_shapes", node, true, "single consumer; no intermediate element conversion");
    changed = true;
  }
  return changed;
}

bool PadPass(CompilerGraph& graph, OptimizationReport* report) {
  Uses uses(graph);
  bool changed = false;
  for (auto& conv : graph.nodes) {
    if (conv.removed || (conv.kind != CompositeKind::kConvolution &&
                        conv.kind != CompositeKind::kDepthwiseConvolution)) continue;
    size_t pi = uses.producer[conv.inputs[0]];
    if (pi == graph.nodes.size()) continue;
    auto& pad = graph.nodes[pi];
    if (pad.removed || (pad.body.code != kLiteRtOpCodeTflPad && pad.body.code != kLiteRtOpCodeTflPadv2)) continue;
    auto skip = [&](const char* message) { Decision(report, "fold_zero_pad", conv, false, message); };
    if (!Removable(graph, conv.inputs[0], uses) || !pad.output_transforms.empty()) {
      skip("intermediate is graph IO, has additional consumers or an output transform"); continue;
    }
    const auto& p = pad.body.parameters;
    const auto& input = graph.values[pad.inputs[0]];
    const auto& padded = graph.values[pad.outputs[0]];
    if (p.size() != 8 || p[0] || p[1] || p[6] || p[7] ||
        !SameQuantization(input, padded) || pad.body.pad_value != pad.body.inputs[0].zero_point) {
      skip("requires spatial-only padding of real zero with unchanged quantization"); continue;
    }
    auto spec = conv.body.convolution->spec;
    if (std::any_of(spec.padding.begin(), spec.padding.end(), [](int32_t x) { return x != 0; })) {
      skip("convolution already has padding"); continue;
    }
    bool legal = true;
    for (size_t axis = 0; axis < 2; ++axis) {
      int64_t effective = (int64_t{spec.weight_shape[axis + 1]} - 1) * spec.dilation[axis] + 1;
      if (p[2 + 2 * axis] >= effective || p[3 + 2 * axis] >= effective) legal = false;
      spec.padding[2 * axis] = p[2 + 2 * axis];
      spec.padding[2 * axis + 1] = p[3 + 2 * axis];
    }
    if (!legal) { skip("padding must be smaller than the effective kernel"); continue; }
    std::copy(input.shape.begin(), input.shape.end(), spec.input_shape.begin());
    std::string reason;
    auto plan = AnalyzeConv2d(std::move(spec), reason);
    if (!plan) { skip("folded geometry/range rejected by convolution analysis"); continue; }
    plan->activation_range = conv.body.convolution->activation_range;
    conv.body.convolution = std::move(plan);
    conv.body.inputs[0] = Quantized(input);
    conv.inputs[0] = pad.inputs[0];
    pad.removed = true;
    Origins(conv, pad);
    Decision(report, "fold_zero_pad", conv, true, "real-zero spatial PAD absorbed; geometry and accumulator ranges rechecked");
    changed = true;
  }
  return changed;
}

bool QualifyRequantization(GraphNode& node, OptimizationMode mode,
                           OptimizationReport* report) {
  bool changed = false;
  for (auto& t : node.output_transforms) {
    if (t.kind != OutputTransform::Kind::kTable || !t.affine_requantization) continue;
    auto scale = MakeScale32(t.input.scale / t.output.scale);
    if (!scale || !IsScale32RangeValid(*scale, -128 - t.input.zero_point, 127 - t.input.zero_point)) {
      Decision(report, "native_requantization", node, false, "native scale/input range not representable"); continue;
    }
    int maximum = 0, total = 0;
    for (int code = -128; code <= 127; ++code) {
      auto scaled = ApplyScale32(code - t.input.zero_point, *scale);
      if (!scaled) { maximum = 256; break; }
      int64_t value = std::clamp(int64_t{*scaled} + t.output.zero_point,
          int64_t{t.clamp.minimum}, int64_t{t.clamp.maximum});
      int error = std::abs(static_cast<int>(value) - t.table[code + 128]);
      maximum = std::max(maximum, error); total += error;
    }
    bool approximate = maximum != 0;
    const bool locally_qualified = maximum <= 1 && total <= 64;
    // The bounded approximate candidate failed the end-to-end corpus gate:
    // downstream operations amplified a <=1-code local difference. Keep it
    // disabled in production, including qualified mode, until a replacement
    // rule passes both local and model gates. Exact replacements remain on.
    constexpr bool kApproximateRequantizationQualified = false;
    bool accepted = maximum == 0 || (mode == kLrtArmVulkanMLOptimizationQualified &&
        locally_qualified && kApproximateRequantizationQualified);
    std::ostringstream result;
    result << "all 256 codes: max=" << maximum << ", mean=" << total / 256.0;
    if (!accepted) { Decision(report, "native_requantization", node, false, result.str() + (approximate && locally_qualified && mode == kLrtArmVulkanMLOptimizationQualified
        ? "; approximate rule disabled: end-to-end corpus gate failed"
        : "; mode/local error bound rejected")); continue; }
    t.kind = OutputTransform::Kind::kRescale;
    t.scales = {*scale}; t.ranges = {{-128 - t.input.zero_point, 127 - t.input.zero_point}};
    t.table.clear(); t.approximate = approximate;
    Decision(report, "native_requantization", node, true, result.str() + (approximate ? "; qualified bounded difference" : "; exact"));
    changed = true;
  }
  return changed;
}

bool HasBodyClamp(const GraphNode& node) {
  switch (node.kind) {
    case CompositeKind::kConvolution:
      // Grouped convolution reconstructs each group's original activation.
      return node.body.convolution->spec.input_shape[3] == node.body.convolution->spec.weight_shape[3];
    case CompositeKind::kDepthwiseConvolution:
    case CompositeKind::kFullyConnected:
    case CompositeKind::kTransposeConvolution:
    case CompositeKind::kBinary:
    case CompositeKind::kPool: return true;
    default: return false;
  }
}
void NormalizeTransforms(GraphNode& node, OptimizationReport* report) {
  auto& transforms = node.output_transforms;
  if (!node.transform_only && HasBodyClamp(node) && !transforms.empty() &&
      transforms[0].kind == OutputTransform::Kind::kClamp) {
    auto& range = (node.kind == CompositeKind::kConvolution || node.kind == CompositeKind::kDepthwiseConvolution)
        ? node.body.convolution->activation_range : node.body.activation;
    range = ComposeClamps(range, transforms[0].clamp);
    transforms.erase(transforms.begin());
    Decision(report, "compose_clamps", node, true, "clamp absorbed into quantized body; rounding and saturation retained");
  }
  for (size_t i = 1; i < transforms.size();) {
    auto& first = transforms[i - 1];
    const auto& second = transforms[i];
    if (first.kind == OutputTransform::Kind::kClamp && second.kind == OutputTransform::Kind::kClamp) {
      first.clamp = ComposeClamps(first.clamp, second.clamp);
      first.output = second.output;
      Decision(report, "compose_clamps", node, true, "ordered clamp composition, including disjoint ranges");
    } else if (first.kind == OutputTransform::Kind::kTable && second.kind == OutputTransform::Kind::kTable) {
      for (auto& code : first.table) code = second.table[static_cast<int>(code) + 128];
      first.output = second.output;
      first.affine_requantization = false;
      Decision(report, "compose_int8_tables", node, true, "exact composition of all 256 codes including intermediate rounding/saturation");
    } else { ++i; continue; }
    transforms.erase(transforms.begin() + static_cast<ptrdiff_t>(i));
  }
}

bool FuseTransforms(CompilerGraph& graph, OptimizationReport* report) {
  Uses uses(graph);
  bool changed = false;
  for (auto& node : graph.nodes) {
    if (node.removed || !node.transform_only || node.inputs.size() != 1) continue;
    size_t pi = uses.producer[node.inputs[0]];
    if (pi == graph.nodes.size()) continue;
    auto& previous = graph.nodes[pi];
    if (previous.removed || previous.outputs.size() != 1 ||
        (previous.kind == CompositeKind::kAnalyzedOperation && !previous.transform_only)) continue;
    if (!Removable(graph, node.inputs[0], uses)) {
      Decision(report, "fuse_output_transform", node, false, "intermediate is graph IO or has additional consumers"); continue;
    }
    // Body output quantization remains unchanged; the ordered transforms own
    // every subsequent quantization boundary. No rounded scale is multiplied.
    previous.output_transforms.insert(previous.output_transforms.end(),
        node.output_transforms.begin(), node.output_transforms.end());
    previous.outputs = node.outputs;
    Origins(previous, node);
    node.removed = true;
    NormalizeTransforms(previous, report);
    Decision(report, "fuse_output_transform", previous, true, "single-consumer quantized composite; ordered transforms preserved");
    changed = true;
  }
  return changed;
}
void Cleanup(CompilerGraph& graph, OptimizationReport* report) {
  std::vector<bool> live(graph.values.size(), false);
  for (auto output : graph.outputs) live[output] = true;
  for (size_t i = 0; i < graph.values.size(); ++i) if (graph.values[i].persistent) live[i] = true;
  for (size_t i = graph.nodes.size(); i > 0; --i) {
    auto& node = graph.nodes[i - 1];
    if (node.removed) continue;
    bool needed = false;
    for (auto output : node.outputs) needed |= live[output];
    if (!needed) { node.removed = true; Decision(report, "dead_nodes", node, true, "no live output consumers"); }
    else for (auto input : node.inputs) live[input] = true;
  }
  for (size_t i = 0; i < graph.values.size(); ++i) if (!live[i] && graph.values[i].is_constant)
    graph.values[i].constant.clear();
}
}  // namespace

const char* OptimizationModeName(OptimizationMode mode) {
  switch (mode) {
    case kLrtArmVulkanMLOptimizationOff: return "off";
    case kLrtArmVulkanMLOptimizationExact: return "exact";
    case kLrtArmVulkanMLOptimizationQualified: return "qualified";
  }
  return "invalid";
}
std::optional<OptimizationMode> ParseOptimizationMode(const std::string& name) {
  if (name == "off") return kLrtArmVulkanMLOptimizationOff;
  if (name == "exact") return kLrtArmVulkanMLOptimizationExact;
  if (name == "qualified") return kLrtArmVulkanMLOptimizationQualified;
  return std::nullopt;
}
const char* CompositeKindName(CompositeKind kind) {
  switch (kind) {
    case CompositeKind::kAnalyzedOperation: return "analyzed_operation";
    case CompositeKind::kConvolution: return "quantized_convolution";
    case CompositeKind::kDepthwiseConvolution: return "quantized_depthwise_convolution";
    case CompositeKind::kTransposeConvolution: return "quantized_transpose_convolution";
    case CompositeKind::kBinary: return "quantized_binary";
    case CompositeKind::kFullyConnected: return "quantized_fully_connected";
    case CompositeKind::kPool: return "quantized_pool";
    case CompositeKind::kInt8Function: return "int8_function";
    case CompositeKind::kSoftmax: return "quantized_softmax";
  }
  return "invalid";
}

std::optional<CompilerGraph> ImportGraph(const Subgraph& subgraph, std::string& reason, SoftmaxLutType softmax_lut) {
  CompilerGraph graph;
  std::unordered_map<LiteRtTensor, ValueId> tensors;
  auto value = [&](const Tensor& tensor) -> std::optional<ValueId> {
    auto existing = tensors.find(tensor.Get());
    if (existing != tensors.end()) return existing->second;
    auto type = tensor.RankedTensorType();
    if (!type || type->Layout().HasStrides()) return std::nullopt;
    GraphValue result;
    result.original_tensor = reinterpret_cast<uintptr_t>(tensor.Get());
    result.logical_type = static_cast<LiteRtElementType>(tensor.ElementType());
    switch (tensor.ElementType()) {
      case ElementType::Int8:
      case ElementType::Bool: result.storage_type = DataType::int8_t; break;
      case ElementType::Int32:
      case ElementType::UInt32: result.storage_type = DataType::int32_t; break;
      case ElementType::Float32:
      case ElementType::Complex64: result.storage_type = DataType::float32_t; break;
      default: return std::nullopt;
    }
    uint64_t count = 1;
    for (auto dim : type->Layout().Dimensions()) {
      if (dim <= 0 || count > uint64_t{INT32_MAX} / dim) return std::nullopt;
      count *= dim; result.shape.push_back(static_cast<uint32_t>(dim));
    }
    result.logical_scalar = result.shape.empty();
    if (tensor.ElementType() == ElementType::Complex64) { result.shape.push_back(2); count *= 2; }
    else if (result.logical_scalar) result.shape = {1};
    if (tensor.QTypeId() == QuantizationTypeId::PerTensor) {
      auto q = tensor.PerTensorQuantization();
      result.scales = {q.scale}; result.zero_points = {q.zero_point};
    } else if (tensor.QTypeId() == QuantizationTypeId::PerChannel) {
      auto q = tensor.PerChannelQuantization();
      result.scales.assign(q.scales, q.scales + q.num_channels);
      result.zero_points.assign(q.zero_points, q.zero_points + q.num_channels);
      result.quantized_dimension = q.quantized_dimension;
    }
    if (tensor.HasWeights()) {
      auto bytes = tensor.Weights().Bytes();
      if (bytes.size() != count * ElementSize(result)) return std::nullopt;
      result.constant.assign(bytes.begin(), bytes.end()); result.is_constant = true;
    }
    ValueId id = graph.values.size();
    tensors.emplace(tensor.Get(), id); graph.values.push_back(std::move(result));
    return id;
  };
  for (const auto& tensor : subgraph.Inputs()) {
    if (tensor.HasWeights()) continue;
    auto id = value(tensor);
    if (!id) { reason = "Invalid partition input layout/type"; return std::nullopt; }
    graph.inputs.push_back(*id); graph.values[*id].graph_input = true;
  }
  for (const auto& op : subgraph.Ops()) {
    auto plan = AnalyzeOperation(op, reason, softmax_lut);
    if (!plan) return std::nullopt;
    GraphNode node;
    node.kind = Classify(*plan);
    node.source_operators = {graph.original_operator_count++};
    for (auto index : plan->runtime_inputs) {
      auto id = value(op.Inputs()[index]);
      if (!id) { reason = "Invalid runtime input/constant"; return std::nullopt; }
      node.inputs.push_back(*id);
    }
    for (const auto& tensor : op.Outputs()) {
      auto id = value(tensor);
      if (!id) { reason = "Invalid operation output"; return std::nullopt; }
      node.outputs.push_back(*id);
    }
    const bool int8_unary = node.inputs.size() == 1 && node.outputs.size() == 1 &&
        !plan->typed && !plan->portable && !plan->inputs[0].floating && !plan->output.floating;
    if (int8_unary && (!plan->table.empty() || plan->code == kLiteRtOpCodeTflRelu ||
        plan->code == kLiteRtOpCodeTflRelu6 || plan->code == kLiteRtOpCodeTflQuantize)) {
      OutputTransform transform;
      transform.input = plan->inputs[0]; transform.output = plan->output;
      transform.clamp = plan->activation;
      transform.table = plan->table;
      transform.kind = transform.table.empty() ? OutputTransform::Kind::kClamp : OutputTransform::Kind::kTable;
      transform.affine_requantization = plan->code == kLiteRtOpCodeTflQuantize ||
          plan->code == kLiteRtOpCodeTflRelu || plan->code == kLiteRtOpCodeTflRelu6;
      // A same-quantization QUANTIZE is already an identity in the legacy path.
      if (plan->code != kLiteRtOpCodeTflQuantize || !transform.table.empty())
        node.output_transforms.push_back(std::move(transform));
      node.transform_only = true; node.kind = CompositeKind::kInt8Function;
    }
    node.body = std::move(*plan); graph.nodes.push_back(std::move(node));
  }
  for (const auto& tensor : subgraph.Outputs()) {
    auto id = value(tensor);
    if (!id) { reason = "Invalid partition output layout/type"; return std::nullopt; }
    graph.outputs.push_back(*id); graph.values[*id].graph_output = true;
  }
  if (!ValidateGraph(graph, reason)) return std::nullopt;
  return graph;
}

bool ValidateGraph(const CompilerGraph& graph, std::string& reason) {
  auto reject = [&](const char* message) { reason = message; return false; };
  if (graph.nodes.empty() || graph.outputs.empty()) return reject("Empty compiler graph or outputs");
  std::vector<bool> defined(graph.values.size(), false);
  for (size_t i = 0; i < graph.values.size(); ++i) defined[i] = graph.values[i].is_constant;
  for (auto id : graph.inputs) {
    if (id >= defined.size()) return reject("Invalid graph input value");
    defined[id] = true;
  }
  for (const auto& node : graph.nodes) if (!node.removed) {
    if (node.inputs.empty() || node.outputs.empty() || node.source_operators.empty()) return reject("Invalid IR node");
    for (auto input : node.inputs) if (input >= defined.size() || !defined[input])
      return reject("Partition is not topologically ordered or has an unbound input");
    for (auto source : node.source_operators)
      if (source >= graph.original_operator_count) return reject("Invalid original operator provenance");
    for (auto output : node.outputs) {
      if (output >= defined.size() || defined[output]) return reject("Invalid or multiply-defined IR output");
      defined[output] = true;
    }
    if (node.inputs.size() != node.body.runtime_inputs.size()) return reject("IR/body runtime input count mismatch");
    if (node.body.portable) {
      if (!node.output_transforms.empty() || node.outputs.size() != node.body.portable->outputs.size())
        return reject("Invalid portable IR expansion");
      for (size_t i = 0; i < node.outputs.size(); ++i)
        if (graph.values[node.outputs[i]].shape != node.body.portable->outputs[i])
          return reject("Portable IR output shape mismatch");
      continue;
    }
    if (node.outputs.size() != 1 || node.body.inputs.empty()) return reject("Invalid analyzed IR expansion");
    auto same = [](const QuantizedTensor& a, const QuantizedTensor& b) {
      return a.shape == b.shape && a.floating == b.floating &&
             a.scale == b.scale && a.zero_point == b.zero_point;
    };
    QuantizedTensor current = node.transform_only ? Quantized(graph.values[node.inputs[0]]) : node.body.output;
    for (const auto& transform : node.output_transforms) {
      if (!same(current, transform.input) || transform.input.shape != transform.output.shape ||
          transform.input.floating || transform.output.floating ||
          !std::isfinite(transform.output.scale) || transform.output.scale <= 0 ||
          transform.input.zero_point < -128 || transform.input.zero_point > 127 ||
          transform.output.zero_point < -128 || transform.output.zero_point > 127)
        return reject("Output transform shape/type/quantization boundary mismatch");
      if (transform.kind == OutputTransform::Kind::kClamp && !same(transform.input, transform.output))
        return reject("CLAMP cannot change quantization");
      if (transform.clamp.minimum < -128 || transform.clamp.maximum > 127 ||
          transform.clamp.minimum > transform.clamp.maximum) return reject("Invalid output clamp");
      if (transform.kind == OutputTransform::Kind::kTable && transform.table.size() != 256) return reject("Invalid INT8 lookup table");
      if (transform.kind == OutputTransform::Kind::kRescale &&
          (transform.scales.size() != 1 || transform.ranges.size() != 1 ||
           !IsScale32RangeValid(transform.scales[0], transform.ranges[0].minimum, transform.ranges[0].maximum)))
        return reject("Invalid output rescale range");
      if (transform.kind == OutputTransform::Kind::kRescale &&
          (transform.ranges[0].minimum > -128 - transform.input.zero_point ||
           transform.ranges[0].maximum < 127 - transform.input.zero_point))
        return reject("Output rescale proof does not cover every INT8 input code");
      current = transform.output;
    }
    const auto& output = graph.values[node.outputs[0]];
    if (current.shape != output.shape ||
        (output.logical_type == kLiteRtElementTypeInt8 && !same(current, Quantized(output))))
      return reject("IR output shape/quantization mismatch");
  }
  for (auto output : graph.outputs) if (output >= defined.size() || !defined[output]) return reject("Unbound graph output");
  return true;
}

bool OptimizeGraph(CompilerGraph& graph, OptimizationMode mode,
                   OptimizationReport* report, std::string& reason) {
  if (!ParseOptimizationMode(OptimizationModeName(mode))) { reason = "Invalid optimization mode"; return false; }
  if (!ValidateGraph(graph, reason)) return false;
  if (report) { *report = {}; report->before_ir = DumpGraph(graph); }
  if (mode != kLrtArmVulkanMLOptimizationOff) {
    for (auto& node : graph.nodes) NormalizeScales(node, report);
    // Normalize native TABLE representation before table composition. Otherwise
    // affine semantic tags are lost and the native RESCALE opportunity vanishes.
    for (auto& node : graph.nodes) QualifyRequantization(node, mode, report);
    const size_t limit = graph.nodes.size() * 2 + 2;
    for (size_t iteration = 0; iteration < limit; ++iteration) {
      bool changed = ShapePass(graph, report);
      changed |= PadPass(graph, report);
      changed |= FuseTransforms(graph, report);
      if (!changed) break;
      if (iteration + 1 == limit) { reason = "Optimizer failed to converge"; return false; }
    }
    // Deterministic deduplication of immutable values, preserving type/shape/q.
    for (size_t i = 0; i < graph.values.size(); ++i) if (graph.values[i].is_constant && !graph.values[i].persistent)
      for (size_t j = 0; j < i; ++j) if (graph.values[j].is_constant && !graph.values[j].persistent &&
          graph.values[i].shape == graph.values[j].shape &&
          graph.values[i].logical_scalar == graph.values[j].logical_scalar &&
          SameQuantization(graph.values[i], graph.values[j]) && graph.values[i].constant == graph.values[j].constant) {
        ReplaceUses(graph, i, j);
        if (report) report->decisions.push_back({"intern_constant_values", {}, true,
            "v" + std::to_string(i) + " aliases immutable v" + std::to_string(j) + "; same logical/storage type, logical rank, shape, quantization and bytes"});
        break;
      }
    Cleanup(graph, report);
  }
  if (!ValidateGraph(graph, reason)) return false;
  if (report) report->after_ir = DumpGraph(graph);
  return true;
}

std::string DumpGraph(const CompilerGraph& graph) {
  std::ostringstream out;
  out << std::setprecision(17);
  for (size_t i = 0; i < graph.values.size(); ++i) {
    const auto& v = graph.values[i];
    out << "v" << i << " logical=" << static_cast<int>(v.logical_type)
        << " storage=" << static_cast<int>(v.storage_type) << " shape=[";
    for (auto dim : v.shape) out << dim << ",";
    out << "] scalar=" << v.logical_scalar << " scales=["; for (auto scale : v.scales) out << scale << ",";
    out << "] zero_points=["; for (auto zp : v.zero_points) out << zp << ",";
    out << "] qdim=" << v.quantized_dimension << " constant_bytes=" << v.constant.size()
        << " input=" << v.graph_input << " output=" << v.graph_output << " persistent=" << v.persistent << "\n";
  }
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto& n = graph.nodes[i];
    out << "n" << i << " " << CompositeKindName(n.kind) << " code=" << n.body.code
        << " removed=" << n.removed << " sources=[";
    for (auto source : n.source_operators) out << source << ",";
    out << "] inputs=["; for (auto input : n.inputs) out << "v" << input << ",";
    out << "] outputs=["; for (auto output : n.outputs) out << "v" << output << ",";
    out << "] body_clamp=[" << n.body.activation.minimum << "," << n.body.activation.maximum << "]";
    if (n.body.softmax) out << " softmax_lut=" << SoftmaxLutTypeName(n.body.softmax->lut_type)
        << " normalization_shift=" << static_cast<int>(n.body.softmax->normalization_shift);
    if (n.body.convolution) {
      const auto& conv = *n.body.convolution;
      out << " convolution_padding=["; for (auto pad : conv.spec.padding) out << pad << ",";
      out << "] accumulator_ranges=[";
      for (auto range : conv.accumulator_ranges) out << range.minimum << ":" << range.maximum << ",";
      out << "] scales=[";
      for (auto scale : conv.scales) out << scale.multiplier << "/2^" << static_cast<int>(scale.shift) << ",";
      out << "] weights=" << conv.spec.weights.size() << " bias=" << conv.spec.bias.size();
    }
    out << " transforms=[";
    for (const auto& t : n.output_transforms) {
      out << (t.kind == OutputTransform::Kind::kClamp ? "CLAMP" : t.kind == OutputTransform::Kind::kTable ? "TABLE" : "RESCALE")
          << "(zp=" << t.input.zero_point << "->" << t.output.zero_point
          << ",clamp=" << t.clamp.minimum << ":" << t.clamp.maximum << ",approx=" << t.approximate << "),";
    }
    out << "]\n";
  }
  return out.str();
}

ResId EmitOutputTransform(tfsc::tosa::Graph& graph, ResId input,
                          const OutputTransform& t, GraphConstantPool& constants) {
  TosaTensor output(DataType::int8_t, t.output.shape);
  if (t.kind == OutputTransform::Kind::kTable) {
    std::vector<uint8_t> bytes(t.table.size());
    std::memcpy(bytes.data(), t.table.data(), bytes.size());
    return graph.AddTableOperator(input, AddGraphConstant(graph,
        TosaTensor(DataType::int8_t, {256}), std::move(bytes), constants), output);
  }
  if (t.kind == OutputTransform::Kind::kRescale) {
    int32_t multiplier = t.scales[0].multiplier;
    std::vector<uint8_t> bytes(sizeof(multiplier));
    std::memcpy(bytes.data(), &multiplier, bytes.size());
    auto mult = AddGraphConstant(graph, TosaTensor(DataType::int32_t, {1}), std::move(bytes), constants);
    auto shift = AddGraphConstant(graph, TosaTensor(DataType::int8_t, {1}),
        {static_cast<uint8_t>(t.scales[0].shift)}, constants);
    input = graph.AddRescaleOperator(input, mult, shift,
        graph.AddTensorConstant(Attribute(static_cast<int8_t>(t.input.zero_point))),
        graph.AddTensorConstant(Attribute(static_cast<int8_t>(t.output.zero_point))),
        Attribute(true), Attribute(int32_t{1}), Attribute(false), Attribute(false), Attribute(false), output);
    if (t.clamp.minimum == -128 && t.clamp.maximum == 127) return input;
  }
  return graph.AddClampOperator(input, Attribute(static_cast<int8_t>(t.clamp.minimum)),
      Attribute(static_cast<int8_t>(t.clamp.maximum)), Attribute(int32_t{1}), output);
}
}  // namespace litert::arm_vulkan_ml
