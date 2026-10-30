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

#include "litert/vendors/arm_vulkan_ml/compiler/decomposition.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "tflite/schema/schema_generated.h"

namespace litert::arm_vulkan_ml {
namespace {
struct Fixture {
  tflite::ModelT model;
  tflite::SubGraphT* graph;
  tflite::OperatorT* op;
  std::vector<uint8_t> bytes;

  explicit Fixture(tflite::BuiltinOperator code) {
    model.version = 3;
    model.buffers.push_back(std::make_unique<tflite::BufferT>());
    model.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
    graph = model.subgraphs[0].get();
    graph->operators.push_back(std::make_unique<tflite::OperatorT>());
    op = graph->operators[0].get();
    auto opcode = std::make_unique<tflite::OperatorCodeT>();
    opcode->builtin_code = code;
    model.operator_codes.push_back(std::move(opcode));
  }
  int Tensor(tflite::TensorType type, std::vector<int32_t> shape, float scale = .125f) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->type = type; tensor->shape = std::move(shape);
    if (type == tflite::TensorType_INT8) {
      tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
      tensor->quantization->scale = {scale};
      tensor->quantization->zero_point = {0};
    }
    graph->tensors.push_back(std::move(tensor));
    return graph->tensors.size() - 1;
  }
  int Control(std::vector<int32_t> shape, const std::vector<int32_t>& values) {
    const int index = Tensor(tflite::TensorType_INT32, std::move(shape));
    auto buffer = std::make_unique<tflite::BufferT>();
    buffer->data.resize(values.size() * sizeof(int32_t));
    std::memcpy(buffer->data.data(), values.data(), buffer->data.size());
    graph->tensors[index]->buffer = model.buffers.size();
    model.buffers.push_back(std::move(buffer));
    return index;
  }
  int FloatConstant(std::vector<int32_t> shape, const std::vector<float>& values) {
    const int index = Tensor(tflite::TensorType_FLOAT32, std::move(shape));
    auto buffer = std::make_unique<tflite::BufferT>();
    buffer->data.resize(values.size() * sizeof(float));
    std::memcpy(buffer->data.data(), values.data(), buffer->data.size());
    graph->tensors[index]->buffer = model.buffers.size();
    model.buffers.push_back(std::move(buffer));
    return index;
  }
  Expected<ExtendedModel> Load() {
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
    bytes.assign(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
    return ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(bytes.data(), bytes.size()));
  }
};

TEST(DecompositionTest, GatherNdProvesEveryIndexAndPreservesQuantization) {
  for (int variant = 0; variant < 5; ++variant) {
    Fixture f(tflite::BuiltinOperator_GATHER_ND);
    const int data = f.Tensor(tflite::TensorType_INT8, {2, 3, 4});
    const int index = variant == 1 ? f.Tensor(tflite::TensorType_INT32, {2, 2}) :
        f.Control({2, 2}, {0, 1, variant == 2 ? -1 : 1, variant == 3 ? 3 : 2});
    const int output = f.Tensor(tflite::TensorType_INT8, {2, 4}, variant == 4 ? .25f : .125f);
    f.op->inputs = {data, index}; f.op->outputs = {output};
    f.graph->inputs = variant == 1 ? std::vector<int32_t>{data, index} : std::vector<int32_t>{data};
    f.graph->outputs = {output};
    auto model = f.Load(); ASSERT_TRUE(model);
    auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*graph, reason)), variant == 0) << reason;
  }
}

TEST(DecompositionTest, BinaryArithmeticRejectsComplexOperandsWithMatchingStorage) {
  for (auto code : {tflite::BuiltinOperator_PRELU, tflite::BuiltinOperator_ADD_N,
                    tflite::BuiltinOperator_SQUARED_DIFFERENCE}) {
    SCOPED_TRACE(tflite::EnumNameBuiltinOperator(code));
    Fixture f(code);
    const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 2});
    // COMPLEX64 [2] and FLOAT32 [2, 2] share a physical FP32 shape, but
    // they are different logical types and cannot be used interchangeably.
    const int other = f.Tensor(tflite::TensorType_COMPLEX64, {2});
    const int output = f.Tensor(tflite::TensorType_FLOAT32, {2, 2});
    f.op->inputs = {data, other}; f.op->outputs = {output};
    f.graph->inputs = {data, other}; f.graph->outputs = {output};
    auto model = f.Load(); ASSERT_TRUE(model);
    auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason));
  }
}

TEST(DecompositionTest, DirectAnalysisEnforcesRegisteredTypePairs) {
  Fixture f(tflite::BuiltinOperator_PRELU);
  const int data = f.Tensor(tflite::TensorType_INT32, {2});
  const int alpha = f.Control({1}, {1});
  const int output = f.Tensor(tflite::TensorType_INT32, {2});
  f.op->inputs = {data, alpha}; f.op->outputs = {output};
  f.graph->inputs = {data}; f.graph->outputs = {output};
  auto model = f.Load(); ASSERT_TRUE(model);
  auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
  std::string reason;
  EXPECT_FALSE(AnalyzeDecomposition(graph->Ops()[0], reason));
  EXPECT_NE(reason.find("types/shapes"), std::string::npos);
}

TEST(DecompositionTest, FloatingFloorArithmeticRequiresExactConstantReciprocals) {
  for (auto code : {tflite::BuiltinOperator_FLOOR_DIV, tflite::BuiltinOperator_FLOOR_MOD}) {
    for (float divisor : {2.0f, -4.0f, .125f, 41.0f, 0.0f,
                          std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::denorm_min()}) {
      SCOPED_TRACE(tflite::EnumNameBuiltinOperator(code));
      SCOPED_TRACE(divisor);
      Fixture f(code);
      const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
      const int rhs = f.FloatConstant({1}, {divisor});
      const int output = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
      f.op->inputs = {data, rhs}; f.op->outputs = {output};
      f.graph->inputs = {data}; f.graph->outputs = {output};
      auto model = f.Load(); ASSERT_TRUE(model);
      auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
      std::string reason;
      const bool supported = divisor == 2.0f || divisor == -4.0f || divisor == .125f;
      EXPECT_EQ(static_cast<bool>(CompilePartition(*graph, reason)), supported) << reason;
      if (!supported) EXPECT_NE(reason.find("power-of-two"), std::string::npos);
    }
    Fixture f(code);
    const int data = f.Tensor(tflite::TensorType_FLOAT32, {2});
    const int rhs = f.Tensor(tflite::TensorType_FLOAT32, {2});
    const int output = f.Tensor(tflite::TensorType_FLOAT32, {2});
    f.op->inputs = {data, rhs}; f.op->outputs = {output};
    f.graph->inputs = {data, rhs}; f.graph->outputs = {output};
    auto model = f.Load(); ASSERT_TRUE(model);
    auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason));
    EXPECT_NE(reason.find("constant power-of-two"), std::string::npos);
  }
}

TEST(DecompositionTest, NormalizedScalarStorageDoesNotHideInvalidLogicalRanks) {
  for (auto code : {tflite::BuiltinOperator_ARG_MIN, tflite::BuiltinOperator_SHAPE,
                    tflite::BuiltinOperator_GREATER,
                    tflite::BuiltinOperator_ADD_N, tflite::BuiltinOperator_FILL,
                    tflite::BuiltinOperator_BROADCAST_TO, tflite::BuiltinOperator_GATHER_ND}) {
    SCOPED_TRACE(tflite::EnumNameBuiltinOperator(code));
    Fixture f(code);
    const bool scalar_data = code != tflite::BuiltinOperator_ARG_MIN &&
        code != tflite::BuiltinOperator_SHAPE && code != tflite::BuiltinOperator_ADD_N;
    const int data = f.Tensor(tflite::TensorType_FLOAT32,
        scalar_data ? std::vector<int32_t>{} : std::vector<int32_t>{1});
    const bool vector_output = code == tflite::BuiltinOperator_ARG_MIN ||
        code == tflite::BuiltinOperator_GREATER ||
        code == tflite::BuiltinOperator_ADD_N;
    const auto type = code == tflite::BuiltinOperator_ARG_MIN || code == tflite::BuiltinOperator_SHAPE
        ? tflite::TensorType_INT32 : code == tflite::BuiltinOperator_GREATER
        ? tflite::TensorType_BOOL : tflite::TensorType_FLOAT32;
    const int output = f.Tensor(type, vector_output ? std::vector<int32_t>{1} : std::vector<int32_t>{});
    f.graph->inputs = {data}; f.graph->outputs = {output};
    f.op->inputs = {data}; f.op->outputs = {output};
    if (code == tflite::BuiltinOperator_ARG_MIN) {
      f.op->inputs.push_back(f.Control({}, {0}));
      tflite::ArgMinOptionsT options; options.output_type = tflite::TensorType_INT32;
      f.op->builtin_options.Set(options);
    } else if (code == tflite::BuiltinOperator_SHAPE) {
      tflite::ShapeOptionsT options; options.out_type = tflite::TensorType_INT32;
      f.op->builtin_options.Set(options);
    } else if (code == tflite::BuiltinOperator_GREATER || code == tflite::BuiltinOperator_ADD_N) {
      const int other = f.Tensor(tflite::TensorType_FLOAT32, {});
      f.graph->inputs.push_back(other); f.op->inputs.push_back(other);
    } else if (code == tflite::BuiltinOperator_FILL) {
      f.op->inputs = {f.Control({1}, {1}), data};
    } else {
      f.op->inputs.push_back(f.Control({1}, {code == tflite::BuiltinOperator_GATHER_ND ? 0 : 1}));
    }
    auto model = f.Load(); ASSERT_TRUE(model);
    auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason)) << reason;
  }
}

TEST(DecompositionTest, ScalarSelectConditionsBroadcastWithoutChangingDataRank) {
  for (auto code : {tflite::BuiltinOperator_SELECT, tflite::BuiltinOperator_SELECT_V2}) {
    Fixture f(code);
    const int condition = f.Tensor(tflite::TensorType_BOOL, {});
    const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int output = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    f.op->inputs = {condition, data, data}; f.op->outputs = {output};
    f.graph->inputs = {condition, data}; f.graph->outputs = {output};
    auto model = f.Load(); ASSERT_TRUE(model);
    auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
    std::string reason;
    auto compiled = CompilePartition(*graph, reason); ASSERT_TRUE(compiled) << reason;
    EXPECT_FALSE(compiled->outputs[0].logical_scalar);
    EXPECT_EQ(compiled->outputs[0].shape, (std::vector<uint32_t>{2, 3}));
  }
}

TEST(DecompositionTest, SingletonSelectPreservesTheDeclaredScalarOrVectorOutput) {
  for (auto code : {tflite::BuiltinOperator_SELECT, tflite::BuiltinOperator_SELECT_V2}) {
    for (bool scalar : {false, true}) {
      Fixture f(code);
      const int condition = f.Tensor(tflite::TensorType_BOOL, {1, 1, 1});
      const int data = f.Tensor(tflite::TensorType_FLOAT32, {1, 1});
      const int other = f.Tensor(tflite::TensorType_FLOAT32, {});
      const int output = f.Tensor(tflite::TensorType_FLOAT32,
          scalar ? std::vector<int32_t>{} : std::vector<int32_t>{1});
      f.op->inputs = {condition, data, other}; f.op->outputs = {output};
      f.graph->inputs = {condition, data, other}; f.graph->outputs = {output};
      auto model = f.Load(); ASSERT_TRUE(model);
      auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
      std::string reason;
      auto compiled = CompilePartition(*graph, reason); ASSERT_TRUE(compiled) << reason;
      EXPECT_EQ(compiled->outputs[0].logical_scalar, scalar);
      EXPECT_EQ(compiled->outputs[0].shape, (std::vector<uint32_t>{1}));
    }
  }
}

TEST(DecompositionTest, SparseToDenseRejectsCollidingAndOutOfRangeIndices) {
  for (int variant = 0; variant < 4; ++variant) {
    Fixture f(tflite::BuiltinOperator_SPARSE_TO_DENSE);
    const int index = f.Control({2, 2}, {0, 1, variant == 1 ? 0 : 1,
                                       variant == 1 ? 1 : variant == 2 ? 3 : 2});
    const int shape = f.Control({2}, {2, 3});
    const int data = f.Tensor(tflite::TensorType_INT32, {2});
    const int fill = f.Control({}, {-7});
    const int output = f.Tensor(tflite::TensorType_INT32, {2, variant == 3 ? 4 : 3});
    f.op->inputs = {index, shape, data, fill}; f.op->outputs = {output};
    f.op->builtin_options.Set(tflite::SparseToDenseOptionsT{});
    f.graph->inputs = {data}; f.graph->outputs = {output};
    auto model = f.Load(); ASSERT_TRUE(model);
    auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*graph, reason)), variant == 0) << reason;
  }
}

TEST(DecompositionTest, QuantizedMeanRejectsPossibleRuntimeAccumulatorOverflow) {
  Fixture f(tflite::BuiltinOperator_MEAN);
  const int data = f.Tensor(tflite::TensorType_INT8, {8192, 8192});
  const int axes = f.Control({2}, {0, 1});
  const int output = f.Tensor(tflite::TensorType_INT8, {});
  f.op->inputs = {data, axes}; f.op->outputs = {output};
  f.op->builtin_options.Set(tflite::ReducerOptionsT{});
  f.graph->inputs = {data}; f.graph->outputs = {output};
  auto model = f.Load(); ASSERT_TRUE(model);
  auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
  std::string reason;
  EXPECT_FALSE(AnalyzeDecomposition(graph->Ops()[0], reason));
  EXPECT_NE(reason.find("range"), std::string::npos);
}

TEST(DecompositionTest, SquareTableHandlesSignsRoundingAndSaturation) {
  // The local TFLite SQUARE reference has no INT8 kernel. Check its owned
  // mathematical table against explicit values, including saturation and the
  // first rounding transition, rather than accepting an unavailable oracle.
  Fixture f(tflite::BuiltinOperator_SQUARE);
  const int data = f.Tensor(tflite::TensorType_INT8, {256}, .03125f);
  const int output = f.Tensor(tflite::TensorType_INT8, {256}, .0625f);
  f.op->inputs = {data}; f.op->outputs = {output};
  f.graph->inputs = {data}; f.graph->outputs = {output};
  auto model = f.Load(); ASSERT_TRUE(model);
  auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
  std::string reason;
  auto plan = AnalyzeDecomposition(graph->Ops()[0], reason); ASSERT_TRUE(plan) << reason;
  ASSERT_EQ(plan->table.size(), 256);
  const std::vector<int> inputs{-128, -127, -9, -8, -7, -6, -5, -1, 0, 1, 5, 6, 7, 8, 9, 127};
  const std::vector<int> expected{127, 127, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 127};
  for (size_t i = 0; i < inputs.size(); ++i) EXPECT_EQ(plan->table[inputs[i] + 128], expected[i]);
  EXPECT_TRUE(CompilePartition(*graph, reason)) << reason;
}

TEST(DecompositionTest, RankScalarRetainsTheLogicalBufferRank) {
  Fixture f(tflite::BuiltinOperator_RANK);
  const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
  const int output = f.Tensor(tflite::TensorType_INT32, {});
  f.op->inputs = {data}; f.op->outputs = {output};
  f.graph->inputs = {data}; f.graph->outputs = {output};
  auto model = f.Load(); ASSERT_TRUE(model);
  auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
  std::string reason;
  auto compiled = CompilePartition(*graph, reason); ASSERT_TRUE(compiled) << reason;
  ASSERT_EQ(compiled->outputs.size(), 1);
  EXPECT_TRUE(compiled->outputs[0].logical_scalar);
  EXPECT_EQ(compiled->outputs[0].shape, (std::vector<uint32_t>{1}));
  EXPECT_EQ(compiled->outputs[0].logical_element, kLiteRtElementTypeInt32);
}

TEST(DecompositionTest, InternalFloatingBucketizeRequiresTheFloatingProfile) {
  Fixture f(tflite::BuiltinOperator_BUCKETIZE);
  const int data = f.Tensor(tflite::TensorType_INT32, {2, 3});
  const int output = f.Tensor(tflite::TensorType_INT32, {2, 3});
  tflite::BucketizeOptionsT options; options.boundaries = {0, 1.5f};
  f.op->builtin_options.Set(options);
  f.op->inputs = {data}; f.op->outputs = {output};
  f.graph->inputs = {data}; f.graph->outputs = {output};
  auto model = f.Load(); ASSERT_TRUE(model);
  auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
  std::string reason;
  auto compiled = CompilePartition(*graph, reason); ASSERT_TRUE(compiled) << reason;
  EXPECT_EQ(compiled->profiles, (std::vector<std::string>{"PRO-INT", "PRO-FP"}));
}

TEST(DecompositionTest, RfftRetainsExtensionAndTransformExtentRequirements) {
  Fixture f(tflite::BuiltinOperator_RFFT2D);
  const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 4, 8});
  const int length = f.Control({2}, {4, 8});
  const int output = f.Tensor(tflite::TensorType_COMPLEX64, {2, 4, 5});
  f.op->inputs = {data, length}; f.op->outputs = {output};
  f.graph->inputs = {data}; f.graph->outputs = {output};
  auto model = f.Load(); ASSERT_TRUE(model);
  auto graph = model->Subgraph(0); ASSERT_TRUE(graph);
  std::string reason;
  auto compiled = CompilePartition(*graph, reason); ASSERT_TRUE(compiled) << reason;
  EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-FFT"}));
  EXPECT_EQ(compiled->level_requirements.max_kernel, 0);
  EXPECT_GE(compiled->level_requirements.max_tensor_bytes, 320);
  EXPECT_GE(compiled->level_requirements.max_rank, 4);
  EXPECT_EQ(compiled->outputs[0].shape, (std::vector<uint32_t>{2, 4, 5, 2}));
  EXPECT_EQ(compiled->outputs[0].logical_element, kLiteRtElementTypeComplex64);
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
