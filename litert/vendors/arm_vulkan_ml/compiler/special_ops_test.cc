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

#include "litert/vendors/arm_vulkan_ml/compiler/special_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "TosaForSpirvCodegen.hpp"
#include "flatbuffers/flexbuffers.h"
#include "gtest/gtest.h"
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "litert/vendors/arm_vulkan_ml/compiler/tosa_requirements.h"
#include "tflite/schema/schema_generated.h"

namespace litert::arm_vulkan_ml {
namespace {
using O = tfsc::tosa::OperatorEnum;

template <typename T>
uint32_t Buffer(tflite::ModelT& model, const std::vector<T>& values) {
  auto buffer = std::make_unique<tflite::BufferT>();
  buffer->data.resize(values.size() * sizeof(T));
  std::memcpy(buffer->data.data(), values.data(), buffer->data.size());
  model.buffers.push_back(std::move(buffer));
  return static_cast<uint32_t>(model.buffers.size() - 1);
}
void Tensor(tflite::SubGraphT& graph, tflite::TensorType type,
            std::vector<int32_t> shape, uint32_t buffer = 0, float scale = 0,
            int64_t zp = 0) {
  auto tensor = std::make_unique<tflite::TensorT>();
  tensor->name = "value_" + std::to_string(graph.tensors.size());
  tensor->shape = std::move(shape);
  tensor->type = type;
  tensor->buffer = buffer;
  if (scale) {
    tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
    tensor->quantization->scale = {scale};
    tensor->quantization->zero_point = {zp};
  }
  graph.tensors.push_back(std::move(tensor));
}
tflite::ModelT Base(tflite::BuiltinOperator code) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  model.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
  auto opcode = std::make_unique<tflite::OperatorCodeT>();
  opcode->builtin_code = code;
  opcode->version = 1;
  model.operator_codes.push_back(std::move(opcode));
  model.subgraphs[0]->operators.push_back(
      std::make_unique<tflite::OperatorT>());
  return model;
}
std::vector<uint8_t> Pack(const tflite::ModelT& model) {
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  return {builder.GetBufferPointer(),
          builder.GetBufferPointer() + builder.GetSize()};
}
tflite::ModelT Norm(const std::string& name, double epsilon = 1e-5,
                    int axis = -1, int groups = 1, int version = -1) {
  auto model = Base(tflite::BuiltinOperator_STABLEHLO_COMPOSITE);
  auto& graph = *model.subgraphs[0];
  Tensor(graph, tflite::TensorType_FLOAT32, {2, 3, 4});
  Tensor(graph, tflite::TensorType_FLOAT32, {4});
  Tensor(graph, tflite::TensorType_FLOAT32, {4});
  Tensor(graph, tflite::TensorType_FLOAT32, {2, 3, 4});
  auto& op = *graph.operators[0];
  op.inputs = name == "odml.l2_norm"    ? std::vector<int32_t>{0}
              : name == "odml.rms_norm" ? std::vector<int32_t>{0, 1}
                                        : std::vector<int32_t>{0, 1, 2};
  op.outputs = {3};
  graph.inputs = op.inputs;
  graph.outputs = {3};
  flexbuffers::Builder attributes;
  attributes.Map([&] {
    attributes.Double("epsilon", epsilon);
    attributes.Int("channel_axis", axis);
    attributes.Int("num_groups", groups);
    attributes.Int("sub_type", 1);
  });
  attributes.Finish();
  tflite::StableHLOCompositeOptionsT options;
  options.name = name;
  options.version = version < 0 ? (name == "odml.group_norm" ? 1 : 0) : version;
  options.decomposition_subgraph_index = 1;
  options.composite_attributes = attributes.GetBuffer();
  op.builtin_options_2.Set(options);
  // Direct selection does not depend on the decomposition body.
  model.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
  return model;
}
tflite::ModelT Fc(bool wide, bool keep) {
  auto model = Base(tflite::BuiltinOperator_FULLY_CONNECTED);
  auto& graph = *model.subgraphs[0];
  const auto data_type =
      wide ? tflite::TensorType_INT16 : tflite::TensorType_INT8;
  Tensor(graph, data_type, {2, 3, 4}, 0, .125f);
  Tensor(graph, tflite::TensorType_INT8, {2, 4},
         Buffer(model, std::vector<int8_t>{1, -2, 3, 4, -4, 3, -2, 1}), .25f);
  Tensor(graph, wide ? tflite::TensorType_INT64 : tflite::TensorType_INT32, {2},
         wide ? Buffer(model, std::vector<int64_t>{17, -23})
              : Buffer(model, std::vector<int32_t>{17, -23}),
         .03125f);
  Tensor(graph, data_type,
         keep ? std::vector<int32_t>{2, 3, 2} : std::vector<int32_t>{6, 2}, 0,
         .25f);
  graph.inputs = {0};
  graph.outputs = {3};
  auto& op = *graph.operators[0];
  op.inputs = {0, 1, 2};
  op.outputs = {3};
  tflite::FullyConnectedOptionsT options;
  options.keep_num_dims = keep;
  op.builtin_options.Set(options);
  return model;
}
std::shared_ptr<const CompiledPartition> Compile(const tflite::ModelT& object,
                                                 std::string& reason,
                                                 int bits = 16) {
  auto bytes = Pack(object);
  auto model = ExtendedModel::CreateFromBuffer(
      BufferRef<uint8_t>(bytes.data(), bytes.size()));
  if (!model) return nullptr;
  auto graph = model->Subgraph(0);
  if (!graph) return nullptr;
  return CompilePartition(*graph, reason, nullptr,
                          kLrtArmVulkanMLOptimizationQualified, nullptr,
                          kLrtArmVulkanMLSoftmaxLutAuto, bits);
}
TEST(SpecialOps, DirectNamedNormalizationAndOwnedPlans) {
  for (const auto* name :
       {"odml.group_norm", "odml.rms_norm", "odml.l2_norm"}) {
    std::string reason;
    auto artifact = Compile(Norm(name), reason);
    ASSERT_NE(artifact, nullptr) << name << ": " << reason;
    EXPECT_LT(artifact->metrics.operators.size(), 10);
    EXPECT_EQ(artifact->metrics.identity_reshapes, 0);
    EXPECT_EQ(artifact->profiles.back(), "PRO-FP");
    // The compiled artifact owns everything after the source model is gone.
    EXPECT_FALSE(artifact->spirv.empty());
  }
}
TEST(SpecialOps, NormalizationRejectsUnsupportedMetadataAndLayouts) {
  for (auto object :
       {Norm("odml.group_norm", 0), Norm("odml.group_norm", -1),
        Norm("odml.group_norm", 1e-5, 1), Norm("odml.group_norm", 1e-5, -1, 2),
        Norm("odml.group_norm", 1e-5, -1, 1, 2), Norm("unknown.norm")}) {
    std::string reason;
    EXPECT_EQ(Compile(object, reason), nullptr);
    EXPECT_FALSE(reason.empty());
  }
  auto object = Norm("odml.rms_norm");
  object.subgraphs[0]->tensors[1]->shape = {1, 4};
  std::string reason;
  EXPECT_EQ(Compile(object, reason), nullptr);
  object = Norm("odml.group_norm");
  object.subgraphs[0]->tensors[0]->shape = {-1, 3, 4};
  EXPECT_EQ(Compile(object, reason), nullptr);
}
TEST(SpecialOps, QuantizedFcPreservesDimensionsAndAttachesBiasToConv) {
  for (bool wide : {false, true})
    for (bool keep : {false, true})
      for (int bits : {16, 32}) {
        std::string reason;
        auto artifact = Compile(Fc(wide, keep), reason, bits);
        ASSERT_NE(artifact, nullptr) << reason;
        EXPECT_EQ(
            artifact->metrics.operators.at(static_cast<uint32_t>(O::Conv2d)),
            1);
        EXPECT_EQ(
            artifact->metrics.operators.at(static_cast<uint32_t>(O::Rescale)),
            1);
        EXPECT_EQ(
            artifact->metrics.operators.count(static_cast<uint32_t>(O::Matmul)),
            0);
        EXPECT_EQ(
            artifact->metrics.operators.count(static_cast<uint32_t>(O::Add)),
            0);
        EXPECT_EQ(
            artifact->metrics.operators.count(static_cast<uint32_t>(O::Clamp)),
            0);
        EXPECT_EQ(artifact->inputs[0].element,
                  wide ? TensorElement::kInt16 : TensorElement::kInt8);
        EXPECT_EQ(artifact->outputs[0].shape,
                  keep ? (std::vector<uint32_t>{2, 3, 2})
                       : (std::vector<uint32_t>{6, 2}));
        EXPECT_EQ(artifact->requires_shader_int64, wide);
        EXPECT_EQ(artifact->requires_shader_int16, wide || bits == 16);
        EXPECT_EQ(artifact->extensions.empty(), !wide);
        EXPECT_EQ(artifact->constants[1].descriptor.element,
                  wide ? TensorElement::kInt64 : TensorElement::kInt32);
      }
}
TEST(SpecialOps, Int16FcRescaleAgreesWithIndependentDoubleOracle) {
  auto object = Fc(true, true);
  auto& tensors = object.subgraphs[0]->tensors;
  tensors[0]->quantization->scale = {.00137f};
  tensors[1]->quantization->scale = {.00385f};
  tensors[2]->quantization->scale = {.00137f * .00385f};
  tensors[3]->quantization->scale = {.0006913f};
  std::string reason;
  auto artifact = Compile(object, reason);
  ASSERT_TRUE(artifact) << reason;
  ASSERT_EQ(artifact->constants[2].descriptor.element, TensorElement::kInt16);
  int16_t multipliers[2];
  int8_t shifts[2];
  std::memcpy(multipliers, artifact->constants[2].bytes.data(),
              sizeof(multipliers));
  std::memcpy(shifts, artifact->constants[3].bytes.data(), sizeof(shifts));
  const double scale = static_cast<double>(.00137f) * .00385f / .0006913f;
  const int weights[2][4] = {{1, -2, 3, 4}, {-4, 3, -2, 1}};
  for (int x : {-32768, -32767, -171, -1, 0, 1, 171, 32766, 32767}) {
    for (int c = 0; c < 2; ++c) {
      const int inputs[] = {x, -x - 1, x / 3, -x / 7};
      int64_t accumulator = c ? -23 : 17;
      for (int k = 0; k < 4; ++k)
        accumulator += int64_t{inputs[k]} * weights[c][k];
      const int64_t denominator = int64_t{1} << shifts[c];
      const int64_t numerator = accumulator * multipliers[c] + denominator / 2;
      const int64_t rounded = numerator / denominator -
                              (numerator < 0 && numerator % denominator != 0);
      const auto actual = std::clamp(rounded, int64_t{-32768}, int64_t{32767});
      const auto expected =
          std::clamp(static_cast<int64_t>(std::floor(accumulator * scale + .5)),
                     int64_t{-32768}, int64_t{32767});
      EXPECT_LE(std::abs(actual - expected), 1);
    }
  }
}
TEST(SpecialOps, Int16ConversionsPreserveLogicalShapesAndRequireSymmetry) {
  for (bool quantize : {true, false}) {
    auto object = Base(quantize ? tflite::BuiltinOperator_QUANTIZE
                                : tflite::BuiltinOperator_DEQUANTIZE);
    auto& graph = *object.subgraphs[0];
    Tensor(graph,
           quantize ? tflite::TensorType_FLOAT32 : tflite::TensorType_INT16,
           {2, 4}, 0, quantize ? 0 : .125f);
    Tensor(graph,
           quantize ? tflite::TensorType_INT16 : tflite::TensorType_FLOAT32,
           {2, 4}, 0, quantize ? .125f : 0);
    graph.inputs = {0};
    graph.outputs = {1};
    graph.operators[0]->inputs = {0};
    graph.operators[0]->outputs = {1};
    std::string reason;
    auto artifact = Compile(object, reason);
    ASSERT_TRUE(artifact) << reason;
    EXPECT_EQ(artifact->extensions, (std::vector<std::string>{"EXT-INT16"}));
    EXPECT_EQ(artifact->metrics.identity_reshapes, 0);
    graph.tensors[quantize ? 1 : 0]->quantization->zero_point = {1};
    EXPECT_EQ(Compile(object, reason), nullptr);
  }
  auto object = Norm("odml.group_norm");
  object.subgraphs[0]->tensors[0]->shape = {8192, 8192, 8192, 8192, 8192, 8192};
  std::string reason;
  EXPECT_EQ(Compile(object, reason), nullptr);
}

TEST(SpecialOps, Int16SubtractWidensBeforeSaturatingAndChecksQuantization) {
  auto object = Base(tflite::BuiltinOperator_SUB);
  auto& graph = *object.subgraphs[0];
  Tensor(graph, tflite::TensorType_INT16, {2, 4}, 0, .125f);
  Tensor(graph, tflite::TensorType_INT16, {},
         Buffer(object, std::vector<int16_t>{32767}), .125f);
  Tensor(graph, tflite::TensorType_INT16, {2, 4}, 0, .125f);
  graph.inputs = {0};
  graph.outputs = {2};
  graph.operators[0]->inputs = {0, 1};
  graph.operators[0]->outputs = {2};
  graph.operators[0]->builtin_options.Set(tflite::SubOptionsT{});
  std::string reason;
  auto artifact = Compile(object, reason);
  ASSERT_TRUE(artifact) << reason;
  EXPECT_EQ(artifact->metrics.operators.at(static_cast<uint32_t>(O::Sub)), 1);
  EXPECT_EQ(artifact->metrics.operators.at(static_cast<uint32_t>(O::Rescale)),
            1);
  EXPECT_EQ(artifact->metrics.identity_reshapes, 0);
  graph.tensors[1]->quantization->scale = {.25f};
  EXPECT_EQ(Compile(object, reason), nullptr);
}
TEST(SpecialOps, RescaleAmplificationRequiresAProvenRange) {
  for (bool proven : {false, true}) {
    auto module = tfsc::CreateModule();
    tfsc::tosa::Graph graph(module, "main");
    auto input = graph.AddInput(
        tfsc::tosa::Tensor(tfsc::tosa::DataType::int32_t, {4}), 0);
    GraphConstantPool pool;
    pool.rescale_multiplier_bits = 16;
    auto value = EmitNativeRescale(
        graph, input, tfsc::tosa::DataType::int32_t,
        tfsc::tosa::DataType::int32_t, {4}, 0, 0, {*MakeScale32(16384)}, pool,
        proven ? std::vector<int64_t>{-128} : std::vector<int64_t>{},
        proven ? std::vector<int64_t>{127} : std::vector<int64_t>{});
    ASSERT_EQ(value != nullptr, proven);
    if (!proven) continue;
    ASSERT_FALSE(tfsc::IsError(value)) << tfsc::GetErrorMsg(value);
    graph.AddOutput(value, 1);
    graph.FinalizeGraph();
    EXPECT_EQ(CollectGraphMetrics(*module).operators.at(
                  static_cast<uint32_t>(O::Rescale)),
              2);
  }
}

TEST(SpecialOps, FcRejectsMismatchedBiasScalesAndKeepDimensions) {
  auto object = Fc(false, true);
  object.subgraphs[0]->tensors[3]->shape = {6, 2};
  std::string reason;
  EXPECT_EQ(Compile(object, reason), nullptr);
  object = Fc(true, true);
  object.subgraphs[0]->tensors[2]->quantization->scale = {.5f};
  EXPECT_EQ(Compile(object, reason), nullptr);
  object = Fc(true, true);
  object.subgraphs[0]->tensors[0]->quantization->zero_point = {1};
  EXPECT_EQ(Compile(object, reason), nullptr);
  object = Fc(true, true);
  const int64_t bias = int64_t{1} << 47;
  std::memcpy(object.buffers[2]->data.data(), &bias, sizeof(bias));
  EXPECT_EQ(Compile(object, reason), nullptr);
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
