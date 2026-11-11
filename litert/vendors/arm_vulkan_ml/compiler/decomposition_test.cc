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

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tosa-for-spirv-codegen/tosa/OperatorEnum.hpp"  // from @tosa_for_spirv_codegen
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "tflite/schema/schema_generated.h"

namespace litert::arm_vulkan_ml {
namespace {
Subgraph FirstSubgraph(Expected<ExtendedModel>& model) {
  auto sg = model->Subgraph(0);
  return std::move(*sg);
}

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
  int Tensor(tflite::TensorType type, std::vector<int32_t> shape,
             float scale = .125f, int64_t zp = 0) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->type = type;
    tensor->shape = std::move(shape);
    if (type == tflite::TensorType_INT8 || type == tflite::TensorType_INT16) {
      tensor->quantization =
          std::make_unique<tflite::QuantizationParametersT>();
      tensor->quantization->scale = {scale};
      tensor->quantization->zero_point = {zp};
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
  int FloatConstant(std::vector<int32_t> shape,
                    const std::vector<float>& values) {
    const int index = Tensor(tflite::TensorType_FLOAT32, std::move(shape));
    auto buffer = std::make_unique<tflite::BufferT>();
    buffer->data.resize(values.size() * sizeof(float));
    std::memcpy(buffer->data.data(), values.data(), buffer->data.size());
    graph->tensors[index]->buffer = model.buffers.size();
    model.buffers.push_back(std::move(buffer));
    return index;
  }
  template <typename T>
  int Constant(tflite::TensorType type, std::vector<int32_t> shape,
               const std::vector<T>& values, float scale = .125f,
               int64_t zp = 0) {
    const int index = Tensor(type, std::move(shape), scale, zp);
    auto buffer = std::make_unique<tflite::BufferT>();
    buffer->data.resize(values.size() * sizeof(T));
    if (!values.empty()) {
      std::memcpy(buffer->data.data(), values.data(), buffer->data.size());
    }
    graph->tensors[index]->buffer = model.buffers.size();
    model.buffers.push_back(std::move(buffer));
    return index;
  }
  Expected<ExtendedModel> Load() {
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
    bytes.assign(builder.GetBufferPointer(),
                 builder.GetBufferPointer() + builder.GetSize());
    return ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(bytes.data(), bytes.size()));
  }
};

TEST(DecompositionTest, GatherNdProvesEveryIndexAndPreservesQuantization) {
  for (int variant = 0; variant < 5; ++variant) {
    Fixture f(tflite::BuiltinOperator_GATHER_ND);
    const int data = f.Tensor(tflite::TensorType_INT8, {2, 3, 4});
    const int index = variant == 1
                          ? f.Tensor(tflite::TensorType_INT32, {2, 2})
                          : f.Control({2, 2}, {0, 1, variant == 2 ? -1 : 1,
                                               variant == 3 ? 3 : 2});
    const int output =
        f.Tensor(tflite::TensorType_INT8, {2, 4}, variant == 4 ? .25f : .125f);
    f.op->inputs = {data, index};
    f.op->outputs = {output};
    f.graph->inputs = variant == 1 ? std::vector<int32_t>{data, index}
                                   : std::vector<int32_t>{data};
    f.graph->outputs = {output};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*graph, reason)), variant == 0)
        << reason;
  }
}

TEST(DecompositionTest,
     BinaryArithmeticRejectsComplexOperandsWithMatchingStorage) {
  for (auto code :
       {tflite::BuiltinOperator_PRELU, tflite::BuiltinOperator_ADD_N,
        tflite::BuiltinOperator_SQUARED_DIFFERENCE}) {
    SCOPED_TRACE(tflite::EnumNameBuiltinOperator(code));
    Fixture f(code);
    const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 2});
    // COMPLEX64 [2] and FLOAT32 [2, 2] share a physical FP32 shape, but
    // they are different logical types and cannot be used interchangeably.
    const int other = f.Tensor(tflite::TensorType_COMPLEX64, {2});
    const int output = f.Tensor(tflite::TensorType_FLOAT32, {2, 2});
    f.op->inputs = {data, other};
    f.op->outputs = {output};
    f.graph->inputs = {data, other};
    f.graph->outputs = {output};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason));
  }
}

TEST(DecompositionTest, DirectAnalysisEnforcesRegisteredTypePairs) {
  Fixture f(tflite::BuiltinOperator_PRELU);
  const int data = f.Tensor(tflite::TensorType_INT32, {2});
  const int alpha = f.Control({1}, {1});
  const int output = f.Tensor(tflite::TensorType_INT32, {2});
  f.op->inputs = {data, alpha};
  f.op->outputs = {output};
  f.graph->inputs = {data};
  f.graph->outputs = {output};
  auto model = f.Load();
  ASSERT_TRUE(model);
  auto graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  std::string reason;
  EXPECT_FALSE(AnalyzeDecomposition(graph->Ops()[0], reason));
  EXPECT_NE(reason.find("types/shapes"), std::string::npos);
}

TEST(DecompositionTest,
     FloatingFloorArithmeticRequiresExactConstantReciprocals) {
  for (auto code :
       {tflite::BuiltinOperator_FLOOR_DIV, tflite::BuiltinOperator_FLOOR_MOD}) {
    for (float divisor : {2.0f, -4.0f, .125f, 41.0f, 0.0f,
                          std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::denorm_min()}) {
      SCOPED_TRACE(tflite::EnumNameBuiltinOperator(code));
      SCOPED_TRACE(divisor);
      Fixture f(code);
      const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
      const int rhs = f.FloatConstant({1}, {divisor});
      const int output = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
      f.op->inputs = {data, rhs};
      f.op->outputs = {output};
      f.graph->inputs = {data};
      f.graph->outputs = {output};
      auto model = f.Load();
      ASSERT_TRUE(model);
      auto graph = model->Subgraph(0);
      ASSERT_TRUE(graph);
      std::string reason;
      const bool supported =
          divisor == 2.0f || divisor == -4.0f || divisor == .125f;
      EXPECT_EQ(static_cast<bool>(CompilePartition(*graph, reason)), supported)
          << reason;
      if (!supported) EXPECT_NE(reason.find("power-of-two"), std::string::npos);
    }
    Fixture f(code);
    const int data = f.Tensor(tflite::TensorType_FLOAT32, {2});
    const int rhs = f.Tensor(tflite::TensorType_FLOAT32, {2});
    const int output = f.Tensor(tflite::TensorType_FLOAT32, {2});
    f.op->inputs = {data, rhs};
    f.op->outputs = {output};
    f.graph->inputs = {data, rhs};
    f.graph->outputs = {output};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason));
    EXPECT_NE(reason.find("constant power-of-two"), std::string::npos);
  }
}

TEST(DecompositionTest, NormalizedScalarStorageDoesNotHideInvalidLogicalRanks) {
  for (auto code :
       {tflite::BuiltinOperator_ARG_MIN, tflite::BuiltinOperator_SHAPE,
        tflite::BuiltinOperator_GREATER, tflite::BuiltinOperator_ADD_N,
        tflite::BuiltinOperator_FILL, tflite::BuiltinOperator_BROADCAST_TO,
        tflite::BuiltinOperator_GATHER_ND}) {
    SCOPED_TRACE(tflite::EnumNameBuiltinOperator(code));
    Fixture f(code);
    const bool scalar_data = code != tflite::BuiltinOperator_ARG_MIN &&
                             code != tflite::BuiltinOperator_SHAPE &&
                             code != tflite::BuiltinOperator_ADD_N;
    const int data = f.Tensor(
        tflite::TensorType_FLOAT32,
        scalar_data ? std::vector<int32_t>{} : std::vector<int32_t>{1});
    const bool vector_output = code == tflite::BuiltinOperator_ARG_MIN ||
                               code == tflite::BuiltinOperator_GREATER ||
                               code == tflite::BuiltinOperator_ADD_N;
    const auto type = code == tflite::BuiltinOperator_ARG_MIN ||
                              code == tflite::BuiltinOperator_SHAPE
                          ? tflite::TensorType_INT32
                      : code == tflite::BuiltinOperator_GREATER
                          ? tflite::TensorType_BOOL
                          : tflite::TensorType_FLOAT32;
    const int output = f.Tensor(
        type, vector_output ? std::vector<int32_t>{1} : std::vector<int32_t>{});
    f.graph->inputs = {data};
    f.graph->outputs = {output};
    f.op->inputs = {data};
    f.op->outputs = {output};
    if (code == tflite::BuiltinOperator_ARG_MIN) {
      f.op->inputs.push_back(f.Control({}, {0}));
      tflite::ArgMinOptionsT options;
      options.output_type = tflite::TensorType_INT32;
      f.op->builtin_options.Set(options);
    } else if (code == tflite::BuiltinOperator_SHAPE) {
      tflite::ShapeOptionsT options;
      options.out_type = tflite::TensorType_INT32;
      f.op->builtin_options.Set(options);
    } else if (code == tflite::BuiltinOperator_GREATER ||
               code == tflite::BuiltinOperator_ADD_N) {
      const int other = f.Tensor(tflite::TensorType_FLOAT32, {});
      f.graph->inputs.push_back(other);
      f.op->inputs.push_back(other);
    } else if (code == tflite::BuiltinOperator_FILL) {
      f.op->inputs = {f.Control({1}, {1}), data};
    } else {
      f.op->inputs.push_back(
          f.Control({1}, {code == tflite::BuiltinOperator_GATHER_ND ? 0 : 1}));
    }
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason)) << reason;
  }
}

TEST(DecompositionTest,
     ScalarSelectConditionsBroadcastWithoutChangingDataRank) {
  for (auto code :
       {tflite::BuiltinOperator_SELECT, tflite::BuiltinOperator_SELECT_V2}) {
    Fixture f(code);
    const int condition = f.Tensor(tflite::TensorType_BOOL, {});
    const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int output = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    f.op->inputs = {condition, data, data};
    f.op->outputs = {output};
    f.graph->inputs = {condition, data};
    f.graph->outputs = {output};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    auto compiled = CompilePartition(*graph, reason);
    ASSERT_TRUE(compiled) << reason;
    EXPECT_FALSE(compiled->outputs[0].logical_scalar);
    EXPECT_EQ(compiled->outputs[0].shape, (std::vector<uint32_t>{2, 3}));
  }
}

TEST(DecompositionTest,
     SingletonSelectPreservesTheDeclaredScalarOrVectorOutput) {
  for (auto code :
       {tflite::BuiltinOperator_SELECT, tflite::BuiltinOperator_SELECT_V2}) {
    for (bool scalar : {false, true}) {
      Fixture f(code);
      const int condition = f.Tensor(tflite::TensorType_BOOL, {1, 1, 1});
      const int data = f.Tensor(tflite::TensorType_FLOAT32, {1, 1});
      const int other = f.Tensor(tflite::TensorType_FLOAT32, {});
      const int output =
          f.Tensor(tflite::TensorType_FLOAT32,
                   scalar ? std::vector<int32_t>{} : std::vector<int32_t>{1});
      f.op->inputs = {condition, data, other};
      f.op->outputs = {output};
      f.graph->inputs = {condition, data, other};
      f.graph->outputs = {output};
      auto model = f.Load();
      ASSERT_TRUE(model);
      auto graph = model->Subgraph(0);
      ASSERT_TRUE(graph);
      std::string reason;
      auto compiled = CompilePartition(*graph, reason);
      ASSERT_TRUE(compiled) << reason;
      EXPECT_EQ(compiled->outputs[0].logical_scalar, scalar);
      EXPECT_EQ(compiled->outputs[0].shape, (std::vector<uint32_t>{1}));
    }
  }
}

TEST(DecompositionTest, SparseToDenseRejectsCollidingAndOutOfRangeIndices) {
  for (int variant = 0; variant < 4; ++variant) {
    Fixture f(tflite::BuiltinOperator_SPARSE_TO_DENSE);
    const int index = f.Control({2, 2}, {0, 1, variant == 1 ? 0 : 1,
                                         variant == 1   ? 1
                                         : variant == 2 ? 3
                                                        : 2});
    const int shape = f.Control({2}, {2, 3});
    const int data = f.Tensor(tflite::TensorType_INT32, {2});
    const int fill = f.Control({}, {-7});
    const int output =
        f.Tensor(tflite::TensorType_INT32, {2, variant == 3 ? 4 : 3});
    f.op->inputs = {index, shape, data, fill};
    f.op->outputs = {output};
    f.op->builtin_options.Set(tflite::SparseToDenseOptionsT{});
    f.graph->inputs = {data};
    f.graph->outputs = {output};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*graph, reason)), variant == 0)
        << reason;
  }
}

TEST(DecompositionTest,
     QuantizedMeanRejectsPossibleRuntimeAccumulatorOverflow) {
  Fixture f(tflite::BuiltinOperator_MEAN);
  const int data = f.Tensor(tflite::TensorType_INT8, {8192, 8192});
  const int axes = f.Control({2}, {0, 1});
  const int output = f.Tensor(tflite::TensorType_INT8, {});
  f.op->inputs = {data, axes};
  f.op->outputs = {output};
  f.op->builtin_options.Set(tflite::ReducerOptionsT{});
  f.graph->inputs = {data};
  f.graph->outputs = {output};
  auto model = f.Load();
  ASSERT_TRUE(model);
  auto graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
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
  f.op->inputs = {data};
  f.op->outputs = {output};
  f.graph->inputs = {data};
  f.graph->outputs = {output};
  auto model = f.Load();
  ASSERT_TRUE(model);
  auto graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  std::string reason;
  auto plan = AnalyzeDecomposition(graph->Ops()[0], reason);
  ASSERT_TRUE(plan) << reason;
  ASSERT_EQ(plan->table.size(), 256);
  const std::vector<int> inputs{-128, -127, -9, -8, -7, -6, -5, -1,
                                0,    1,    5,  6,  7,  8,  9,  127};
  const std::vector<int> expected{127, 127, 1, 1, 1, 1, 0, 0,
                                  0,   0,   0, 1, 1, 1, 1, 127};
  for (size_t i = 0; i < inputs.size(); ++i)
    EXPECT_EQ(plan->table[inputs[i] + 128], expected[i]);
  EXPECT_TRUE(CompilePartition(*graph, reason)) << reason;
}

TEST(DecompositionTest, RankScalarRetainsTheLogicalBufferRank) {
  Fixture f(tflite::BuiltinOperator_RANK);
  const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
  const int output = f.Tensor(tflite::TensorType_INT32, {});
  f.op->inputs = {data};
  f.op->outputs = {output};
  f.graph->inputs = {data};
  f.graph->outputs = {output};
  auto model = f.Load();
  ASSERT_TRUE(model);
  auto graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  std::string reason;
  auto compiled = CompilePartition(*graph, reason);
  ASSERT_TRUE(compiled) << reason;
  ASSERT_EQ(compiled->outputs.size(), 1);
  EXPECT_TRUE(compiled->outputs[0].logical_scalar);
  EXPECT_EQ(compiled->outputs[0].shape, (std::vector<uint32_t>{1}));
  EXPECT_EQ(compiled->outputs[0].logical_element, kLiteRtElementTypeInt32);
}

TEST(DecompositionTest, InternalFloatingBucketizeRequiresTheFloatingProfile) {
  Fixture f(tflite::BuiltinOperator_BUCKETIZE);
  const int data = f.Tensor(tflite::TensorType_INT32, {2, 3});
  const int output = f.Tensor(tflite::TensorType_INT32, {2, 3});
  tflite::BucketizeOptionsT options;
  options.boundaries = {0, 1.5f};
  f.op->builtin_options.Set(options);
  f.op->inputs = {data};
  f.op->outputs = {output};
  f.graph->inputs = {data};
  f.graph->outputs = {output};
  auto model = f.Load();
  ASSERT_TRUE(model);
  auto graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  std::string reason;
  auto compiled = CompilePartition(*graph, reason);
  ASSERT_TRUE(compiled) << reason;
  EXPECT_EQ(compiled->profiles,
            (std::vector<std::string>{"PRO-INT", "PRO-FP"}));
}

TEST(DecompositionTest, RfftRetainsExtensionAndTransformExtentRequirements) {
  Fixture f(tflite::BuiltinOperator_RFFT2D);
  const int data = f.Tensor(tflite::TensorType_FLOAT32, {2, 4, 8});
  const int length = f.Control({2}, {4, 8});
  const int output = f.Tensor(tflite::TensorType_COMPLEX64, {2, 4, 5});
  f.op->inputs = {data, length};
  f.op->outputs = {output};
  f.graph->inputs = {data};
  f.graph->outputs = {output};
  auto model = f.Load();
  ASSERT_TRUE(model);
  auto graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  std::string reason;
  auto compiled = CompilePartition(*graph, reason);
  ASSERT_TRUE(compiled) << reason;
  EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-FFT"}));
  EXPECT_EQ(compiled->level_requirements.max_kernel, 0);
  EXPECT_GE(compiled->level_requirements.max_tensor_bytes, 320);
  EXPECT_GE(compiled->level_requirements.max_rank, 4);
  EXPECT_EQ(compiled->outputs[0].shape, (std::vector<uint32_t>{2, 4, 5, 2}));
  EXPECT_EQ(compiled->outputs[0].logical_element, kLiteRtElementTypeComplex64);
}

TEST(DecompositionTest, Int16BatchMatmulRescales3dInt48BeforeOutputReshape) {
  using O = tfsc::tosa::OperatorEnum;
  // ViT QK^T attention shape: [1, 12, 197, 64] x [1, 12, 197, 64] (adj_y=true)
  // -> [1, 12, 197, 197].
  Fixture f(tflite::BuiltinOperator_BATCH_MATMUL);
  const int lhs = f.Tensor(tflite::TensorType_INT16, {1, 12, 197, 64}, 0.025f);
  const int rhs = f.Tensor(tflite::TensorType_INT16, {1, 12, 197, 64}, 0.025f);
  const int out =
      f.Tensor(tflite::TensorType_INT16, {1, 12, 197, 197}, 0.05f);
  tflite::BatchMatMulOptionsT opts;
  opts.adj_x = false;
  opts.adj_y = true;
  opts.asymmetric_quantize_inputs = false;
  f.op->builtin_options.Set(opts);
  f.op->inputs = {lhs, rhs};
  f.op->outputs = {out};
  f.graph->inputs = {lhs, rhs};
  f.graph->outputs = {out};
  auto model = f.Load();
  ASSERT_TRUE(model);
  auto graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  std::string reason;
  auto compiled = CompilePartition(*graph, reason);
  ASSERT_TRUE(compiled) << reason;
  EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-INT16"}));
  EXPECT_EQ(compiled->requires_shader_int64, true);
  EXPECT_EQ(compiled->metrics.operators.at(static_cast<uint32_t>(O::Matmul)),
            1);
  EXPECT_EQ(compiled->metrics.operators.at(static_cast<uint32_t>(O::Rescale)),
            1);
  EXPECT_EQ(compiled->metrics.operators.count(static_cast<uint32_t>(O::Clamp)),
            0);
  EXPECT_EQ(compiled->level_requirements.max_rank, 4);

  // asymmetric_quantize_inputs = true or nonzero zero_point must be rejected.
  opts.asymmetric_quantize_inputs = true;
  f.op->builtin_options.Set(opts);
  model = f.Load();
  ASSERT_TRUE(model);
  graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  EXPECT_FALSE(CompilePartition(*graph, reason));

  opts.asymmetric_quantize_inputs = false;
  f.op->builtin_options.Set(opts);
  f.graph->tensors[lhs]->quantization->zero_point = {1};
  model = f.Load();
  ASSERT_TRUE(model);
  graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  EXPECT_FALSE(CompilePartition(*graph, reason));
}

TEST(DecompositionTest, Int16GeluMeanAndSelectCompileWithExtInt16) {
  for (bool approximate : {false, true}) {
    Fixture f(tflite::BuiltinOperator_GELU);
    const int data = f.Tensor(tflite::TensorType_INT16, {1, 197, 64}, 0.025f);
    const int out = f.Tensor(tflite::TensorType_INT16, {1, 197, 64}, 0.025f);
    tflite::GeluOptionsT opts;
    opts.approximate = approximate;
    f.op->builtin_options.Set(opts);
    f.op->inputs = {data};
    f.op->outputs = {out};
    f.graph->inputs = {data};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    auto plan = AnalyzeDecomposition(graph->Ops()[0], reason);
    ASSERT_TRUE(plan) << reason;
    EXPECT_EQ(plan->table16.size(), 513);
    auto compiled = CompilePartition(*graph, reason);
    ASSERT_TRUE(compiled) << reason;
    EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-INT16"}));
  }

  {
    Fixture f(tflite::BuiltinOperator_MEAN);
    const int data = f.Tensor(tflite::TensorType_INT16, {1, 197, 64}, 0.05f);
    const int axes = f.Control({1}, {1});
    const int out = f.Tensor(tflite::TensorType_INT16, {1, 64}, 0.05f);
    f.op->builtin_options.Set(tflite::ReducerOptionsT{});
    f.op->inputs = {data, axes};
    f.op->outputs = {out};
    f.graph->inputs = {data};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    auto compiled = CompilePartition(*graph, reason);
    ASSERT_TRUE(compiled) << reason;
    EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-INT16"}));
  }

  {
    Fixture f(tflite::BuiltinOperator_SELECT_V2);
    const int cond = f.Tensor(tflite::TensorType_BOOL, {1, 4});
    const int lhs = f.Tensor(tflite::TensorType_INT16, {2, 4}, 0.125f);
    const int rhs = f.Tensor(tflite::TensorType_INT16, {2, 4}, 0.125f);
    const int out = f.Tensor(tflite::TensorType_INT16, {2, 4}, 0.125f);
    f.op->inputs = {cond, lhs, rhs};
    f.op->outputs = {out};
    f.graph->inputs = {cond, lhs, rhs};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    auto compiled = CompilePartition(*graph, reason);
    ASSERT_TRUE(compiled) << reason;
    EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-INT16"}));
  }
}

TEST(DecompositionTest,
     Int8Conv3dSupportsSameCropPerChannelAndValidAndRejectsInvalidGeometry) {
  using OpEnum = tfsc::tosa::OperatorEnum;
  {
    Fixture f(tflite::BuiltinOperator_CONV_3D);
    const int input = f.Tensor(tflite::TensorType_INT8, {1, 5, 5, 5, 2}, .125f, -2);
    const std::vector<int8_t> w_vals(32, 1);
    const int weight =
        f.Constant<int8_t>(tflite::TensorType_INT8, {2, 2, 2, 2, 2}, w_vals);
    f.graph->tensors[weight]->quantization->quantized_dimension = 4;
    f.graph->tensors[weight]->quantization->scale = {.25f, .5f};
    f.graph->tensors[weight]->quantization->zero_point = {0, 0};
    const int bias =
        f.Constant<int32_t>(tflite::TensorType_INT32, {2}, {4, -4});
    f.graph->tensors[bias]->quantization =
        std::make_unique<tflite::QuantizationParametersT>();
    f.graph->tensors[bias]->quantization->quantized_dimension = 0;
    f.graph->tensors[bias]->quantization->scale = {.03125f, .0625f};
    f.graph->tensors[bias]->quantization->zero_point = {0, 0};
    const int output = f.Tensor(tflite::TensorType_INT8, {1, 3, 3, 3, 2}, .25f, 0);
    f.op->inputs = {input, weight, bias};
    f.op->outputs = {output};
    f.graph->inputs = {input};
    f.graph->outputs = {output};
    tflite::Conv3DOptionsT opts;
    opts.padding = tflite::Padding_SAME;
    opts.stride_d = 2;
    opts.stride_h = 2;
    opts.stride_w = 2;
    opts.dilation_d_factor = 1;
    opts.dilation_h_factor = 1;
    opts.dilation_w_factor = 1;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    auto compiled = CompilePartition(*graph, reason);
    ASSERT_TRUE(compiled) << reason;
    EXPECT_GE(compiled->metrics.operators.at(static_cast<uint32_t>(OpEnum::Conv3d)), 1);
    EXPECT_EQ(compiled->metrics.operators.count(static_cast<uint32_t>(OpEnum::Slice)), 0);
    EXPECT_GE(compiled->metrics.operators.at(static_cast<uint32_t>(OpEnum::Rescale)), 1);
    EXPECT_GE(compiled->metrics.operators.at(static_cast<uint32_t>(OpEnum::Clamp)), 1);
  }
  {
    Fixture f(tflite::BuiltinOperator_CONV_3D);
    const int input = f.Tensor(tflite::TensorType_INT8, {1, 5, 5, 5, 2}, .125f, 0);
    const int weight = f.Constant<int8_t>(
        tflite::TensorType_INT8, {2, 2, 2, 2, 2}, std::vector<int8_t>(32, 1),
        .25f, 0);
    const int bias =
        f.Constant<int32_t>(tflite::TensorType_INT32, {2}, {0, 2});
    f.graph->tensors[bias]->quantization =
        std::make_unique<tflite::QuantizationParametersT>();
    f.graph->tensors[bias]->quantization->scale = {.03125f};
    f.graph->tensors[bias]->quantization->zero_point = {0};
    const int output = f.Tensor(tflite::TensorType_INT8, {1, 2, 2, 2, 2}, .25f, 0);
    f.op->inputs = {input, weight, bias};
    f.op->outputs = {output};
    f.graph->inputs = {input};
    f.graph->outputs = {output};
    tflite::Conv3DOptionsT opts;
    opts.padding = tflite::Padding_VALID;
    opts.stride_d = opts.stride_h = opts.stride_w = 2;
    opts.dilation_d_factor = opts.dilation_h_factor = opts.dilation_w_factor = 1;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    auto compiled = CompilePartition(*graph, reason);
    ASSERT_TRUE(compiled) << reason;
    EXPECT_GE(compiled->metrics.operators.at(static_cast<uint32_t>(OpEnum::Slice)), 1);
    opts.stride_d = opts.stride_h = opts.stride_w = 1;

    // Rejections: nonzero weight zp, mismatched bias scale, kernel > input, bad output.
    for (int bad = 0; bad < 4; ++bad) {
      SCOPED_TRACE(bad);
      Fixture bf(tflite::BuiltinOperator_CONV_3D);
      const int bi = bf.Tensor(tflite::TensorType_INT8,
                               bad == 2 ? std::vector<int32_t>{1, 1, 3, 3, 2}
                                        : std::vector<int32_t>{1, 3, 3, 3, 2},
                               .125f, 0);
      const int bw = bf.Constant<int8_t>(
          tflite::TensorType_INT8, {2, 2, 2, 2, 2}, std::vector<int8_t>(32, 1),
          .25f, bad == 0 ? 1 : 0);
      const int bb =
          bf.Constant<int32_t>(tflite::TensorType_INT32, {2}, {0, 0});
      bf.graph->tensors[bb]->quantization =
          std::make_unique<tflite::QuantizationParametersT>();
      bf.graph->tensors[bb]->quantization->scale = {bad == 1 ? .5f : .03125f};
      bf.graph->tensors[bb]->quantization->zero_point = {0};
      const int bo = bf.Tensor(tflite::TensorType_INT8,
                               bad == 3 ? std::vector<int32_t>{1, 3, 2, 2, 2}
                                        : std::vector<int32_t>{1, 2, 2, 2, 2},
                               .25f, 0);
      bf.op->inputs = {bi, bw, bb};
      bf.op->outputs = {bo};
      bf.graph->inputs = {bi};
      bf.graph->outputs = {bo};
      bf.op->builtin_options.Set(opts);
      auto bm = bf.Load();
      ASSERT_TRUE(bm);
      auto bg = bm->Subgraph(0);
      ASSERT_TRUE(bg);
      std::string err;
      EXPECT_FALSE(AnalyzeDecomposition(bg->Ops()[0], err));
    }
  }
}

TEST(DecompositionTest,
     SpaceBatchDepthAndMirrorPadLoweringsCoverValidAndInvalidGeometries) {
  {
    Fixture f(tflite::BuiltinOperator_SPACE_TO_DEPTH);
    const int in = f.Tensor(tflite::TensorType_INT8, {1, 4, 6, 3});
    const int out = f.Tensor(tflite::TensorType_INT8, {1, 2, 3, 12});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::SpaceToDepthOptionsT opts;
    opts.block_size = 2;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;

    f.graph->tensors[in]->shape = {1, 5, 6, 3};
    auto bad_model = f.Load();
    ASSERT_TRUE(bad_model);
    EXPECT_FALSE(AnalyzeDecomposition(FirstSubgraph(bad_model).Ops()[0], reason));
  }
  {
    Fixture f(tflite::BuiltinOperator_DEPTH_TO_SPACE);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 2, 3, 12});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {1, 4, 6, 3});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::DepthToSpaceOptionsT opts;
    opts.block_size = 2;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;

    f.graph->tensors[in]->shape = {1, 2, 3, 10};
    auto bad_model = f.Load();
    ASSERT_TRUE(bad_model);
    EXPECT_FALSE(AnalyzeDecomposition(FirstSubgraph(bad_model).Ops()[0], reason));
  }
  for (auto dtype : {tflite::TensorType_INT8, tflite::TensorType_FLOAT32}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    {
      Fixture f(tflite::BuiltinOperator_SPACE_TO_BATCH_ND);
      const int in = f.Tensor(dtype, {1, 3, 3, 2}, .125f, -3);
      const int block = f.Control({2}, {2, 2});
      const int pad = f.Control({2, 2}, {1, 0, 0, 1});
      const int out = f.Tensor(dtype, {4, 2, 2, 2}, .125f, -3);
      f.op->inputs = {in, block, pad};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
    {
      Fixture f(tflite::BuiltinOperator_BATCH_TO_SPACE_ND);
      const int in = f.Tensor(dtype, {4, 2, 2, 2}, .125f, -3);
      const int block = f.Control({2}, {2, 2});
      const int crop = f.Control({2, 2}, {1, 0, 0, 1});
      const int out = f.Tensor(dtype, {1, 3, 3, 2}, .125f, -3);
      f.op->inputs = {in, block, crop};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
    for (auto mode :
         {tflite::MirrorPadMode_REFLECT, tflite::MirrorPadMode_SYMMETRIC}) {
      SCOPED_TRACE(static_cast<int>(mode));
      Fixture f(tflite::BuiltinOperator_MIRROR_PAD);
      const int in = f.Tensor(dtype, {2, 4, 3}, .125f, -1);
      const int pad = f.Control({3, 2}, {0, 0, 2, 1, 0, 2});
      const int out = f.Tensor(dtype, {2, 7, 5}, .125f, -1);
      f.op->inputs = {in, pad};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      tflite::MirrorPadOptionsT opts;
      opts.mode = mode;
      f.op->builtin_options.Set(opts);
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
  }
  // MirrorPad REFLECT vs SYMMETRIC boundary check (pad=4 on dim=4).
  for (auto mode :
       {tflite::MirrorPadMode_REFLECT, tflite::MirrorPadMode_SYMMETRIC}) {
    SCOPED_TRACE(static_cast<int>(mode));
    Fixture f(tflite::BuiltinOperator_MIRROR_PAD);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 4, 3});
    const int pad = f.Control({3, 2}, {0, 0, 4, 0, 0, 0});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 8, 3});
    f.op->inputs = {in, pad};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::MirrorPadOptionsT opts;
    opts.mode = mode;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(
                  AnalyzeDecomposition(FirstSubgraph(model).Ops()[0], reason)),
              mode == tflite::MirrorPadMode_SYMMETRIC);
  }
}

TEST(DecompositionTest,
     OneHotBroadcastFillZerosLikeSparseToDenseFakeQuantAndAddNCompile) {
  for (auto dtype : {tflite::TensorType_FLOAT32, tflite::TensorType_INT8,
                     tflite::TensorType_INT32, tflite::TensorType_BOOL}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    Fixture f(tflite::BuiltinOperator_ONE_HOT);
    const int indices = f.Tensor(tflite::TensorType_INT32, {2, 3});
    const int depth = f.Control({}, {4});
    int on_val = 0;
    int off_val = 0;
    if (dtype == tflite::TensorType_FLOAT32) {
      on_val = f.FloatConstant({}, {1.0f});
      off_val = f.FloatConstant({}, {0.0f});
    } else if (dtype == tflite::TensorType_INT8) {
      on_val = f.Constant<int8_t>(dtype, {}, {10}, .125f, 0);
      off_val = f.Constant<int8_t>(dtype, {}, {-2}, .125f, 0);
    } else if (dtype == tflite::TensorType_INT32) {
      on_val = f.Control({}, {1});
      off_val = f.Control({}, {0});
    } else {
      on_val = f.Constant<uint8_t>(dtype, {}, {1});
      off_val = f.Constant<uint8_t>(dtype, {}, {0});
    }
    const int out = f.Tensor(dtype, {2, 3, 4}, .125f, 0);
    f.op->inputs = {indices, depth, on_val, off_val};
    f.op->outputs = {out};
    f.graph->inputs = {indices};
    f.graph->outputs = {out};
    tflite::OneHotOptionsT opts;
    opts.axis = -1;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_ONE_HOT);
    const int indices = f.Tensor(tflite::TensorType_INT32, {});
    const int depth = f.Control({}, {4});
    const int on_val = f.FloatConstant({}, {1.0f});
    const int off_val = f.FloatConstant({}, {0.0f});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {4});
    f.op->inputs = {indices, depth, on_val, off_val};
    f.op->outputs = {out};
    f.graph->inputs = {indices};
    f.graph->outputs = {out};
    tflite::OneHotOptionsT opts;
    opts.axis = 0;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_BROADCAST_TO);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 3});
    const int shape = f.Control({2}, {2, 3});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    f.op->inputs = {in, shape};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  for (auto dtype : {tflite::TensorType_FLOAT32, tflite::TensorType_INT32}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    Fixture f(tflite::BuiltinOperator_ZEROS_LIKE);
    const int in = f.Tensor(dtype, {2, 3});
    const int out = f.Tensor(dtype, {2, 3});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_FILL);
    const int dims = f.Control({2}, {2, 3});
    const int value = f.Tensor(tflite::TensorType_FLOAT32, {});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    f.op->inputs = {dims, value};
    f.op->outputs = {out};
    f.graph->inputs = {value};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_SPARSE_TO_DENSE);
    const int idx = f.Control({2, 1}, {0, 2});
    const int shape = f.Control({1}, {4});
    const int values = f.Tensor(tflite::TensorType_FLOAT32, {});
    const int def = f.FloatConstant({}, {0.0f});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {4});
    f.op->inputs = {idx, shape, values, def};
    f.op->outputs = {out};
    f.graph->inputs = {values};
    f.graph->outputs = {out};
    f.op->builtin_options.Set(tflite::SparseToDenseOptionsT{});
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  for (bool narrow : {false, true}) {
    SCOPED_TRACE(narrow);
    Fixture f(tflite::BuiltinOperator_FAKE_QUANT);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::FakeQuantOptionsT opts;
    opts.min = narrow ? 0.0f : -1.0f;
    opts.max = narrow ? 6.0f : 1.0f;
    opts.num_bits = narrow ? 16 : 8;
    opts.narrow_range = narrow;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_ADD_N);
    const int a = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int b = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int c = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    f.op->inputs = {a, b, c};
    f.op->outputs = {out};
    f.graph->inputs = {a, b, c};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
}

TEST(DecompositionTest,
     ActivationsComplexCastArgMinAndMathDecompositionsCompileAndValidate) {
  // PRELU FP32 and INT8 + unprovable scale rejection.
  for (bool fp : {true, false}) {
    SCOPED_TRACE(fp);
    Fixture f(tflite::BuiltinOperator_PRELU);
    const auto dt = fp ? tflite::TensorType_FLOAT32 : tflite::TensorType_INT8;
    const int in = f.Tensor(dt, {2, 3, 4}, .125f, -5);
    const int alpha = f.Tensor(dt, {1, 1, 4}, .125f, 2);
    const int out = f.Tensor(dt, {2, 3, 4}, .125f, -5);
    f.op->inputs = {in, alpha};
    f.op->outputs = {out};
    f.graph->inputs = {in, alpha};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_PRELU);
    const int in = f.Tensor(tflite::TensorType_INT8, {2, 4}, .125f, 0);
    const int alpha = f.Tensor(tflite::TensorType_INT8, {4}, .125f, 0);
    const int out = f.Tensor(tflite::TensorType_INT8, {2, 4}, 1e-9f, 0);
    f.op->inputs = {in, alpha};
    f.op->outputs = {out};
    f.graph->inputs = {in, alpha};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_FALSE(AnalyzeDecomposition(FirstSubgraph(model).Ops()[0], reason));
  }

  // LEAKY_RELU, RELU_N1_TO_1, RELU_0_TO_1 (FP32 & INT8) + INT8 EXP/LOG/RSQRT/GELU.
  for (auto code :
       {tflite::BuiltinOperator_LEAKY_RELU, tflite::BuiltinOperator_RELU_N1_TO_1,
        tflite::BuiltinOperator_RELU_0_TO_1}) {
    for (bool fp : {true, false}) {
      SCOPED_TRACE(static_cast<int>(code) * 10 + (fp ? 1 : 0));
      Fixture f(code);
      const auto dt = fp ? tflite::TensorType_FLOAT32 : tflite::TensorType_INT8;
      const int in = f.Tensor(dt, {2, 8}, .125f, 0);
      const int out = f.Tensor(dt, {2, 8}, .125f, 0);
      f.op->inputs = {in};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      if (code == tflite::BuiltinOperator_LEAKY_RELU) {
        tflite::LeakyReluOptionsT opts;
        opts.alpha = 0.2f;
        f.op->builtin_options.Set(opts);
      }
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
  }
  {
    Fixture f(tflite::BuiltinOperator_LEAKY_RELU);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::LeakyReluOptionsT opts;
    opts.alpha = std::numeric_limits<float>::quiet_NaN();
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_FALSE(AnalyzeDecomposition(FirstSubgraph(model).Ops()[0], reason));
  }
  for (auto code : {tflite::BuiltinOperator_EXP, tflite::BuiltinOperator_LOG,
                    tflite::BuiltinOperator_RSQRT}) {
    SCOPED_TRACE(static_cast<int>(code));
    Fixture f(code);
    const int in = f.Tensor(tflite::TensorType_INT8, {2, 8}, .05f, -128);
    const int out = f.Tensor(tflite::TensorType_INT8, {2, 8}, .05f, 0);
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto sg = model->Subgraph(0);
    ASSERT_TRUE(sg);
    std::string reason;
    auto plan = AnalyzeDecomposition(sg->Ops()[0], reason);
    ASSERT_TRUE(plan) << reason;
    EXPECT_EQ(plan->table.size(), 256);
    EXPECT_TRUE(CompilePartition(*sg, reason)) << reason;

    if (code != tflite::BuiltinOperator_EXP) {
      f.graph->tensors[in]->quantization->zero_point = {0};
      auto bad = f.Load();
      ASSERT_TRUE(bad);
      EXPECT_FALSE(AnalyzeDecomposition(FirstSubgraph(bad).Ops()[0], reason));
    }
  }
  for (bool approx : {false, true}) {
    SCOPED_TRACE(approx);
    Fixture f(tflite::BuiltinOperator_GELU);
    const int in = f.Tensor(tflite::TensorType_INT8, {2, 8}, .05f, 0);
    const int out = f.Tensor(tflite::TensorType_INT8, {2, 8}, .05f, 0);
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::GeluOptionsT opts;
    opts.approximate = approx;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }

  // REAL, IMAG, and cropped/padded RFFT2D.
  for (auto code : {tflite::BuiltinOperator_REAL, tflite::BuiltinOperator_IMAG}) {
    SCOPED_TRACE(static_cast<int>(code));
    Fixture f(code);
    const int in = f.Tensor(tflite::TensorType_COMPLEX64, {2, 3});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_RFFT2D);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 6, 6});
    const int fft = f.Control({2}, {4, 8});
    const int out = f.Tensor(tflite::TensorType_COMPLEX64, {1, 4, 5});
    f.op->inputs = {in, fft};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    auto compiled = CompilePartition(FirstSubgraph(model), reason);
    ASSERT_TRUE(compiled) << reason;
    EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-FFT"}));
  }

  // CAST type matrix.
  const std::pair<tflite::TensorType, tflite::TensorType> casts[] = {
      {tflite::TensorType_FLOAT32, tflite::TensorType_FLOAT32},
      {tflite::TensorType_FLOAT32, tflite::TensorType_BOOL},
      {tflite::TensorType_INT8, tflite::TensorType_BOOL},
      {tflite::TensorType_INT32, tflite::TensorType_BOOL},
      {tflite::TensorType_BOOL, tflite::TensorType_BOOL},
      {tflite::TensorType_BOOL, tflite::TensorType_FLOAT32},
      {tflite::TensorType_BOOL, tflite::TensorType_INT32},
      {tflite::TensorType_BOOL, tflite::TensorType_INT8},
      {tflite::TensorType_FLOAT32, tflite::TensorType_INT32},
      {tflite::TensorType_FLOAT32, tflite::TensorType_INT8},
      {tflite::TensorType_INT32, tflite::TensorType_INT8},
      {tflite::TensorType_INT8, tflite::TensorType_FLOAT32},
  };
  for (size_t i = 0; i < std::size(casts); ++i) {
    SCOPED_TRACE(i);
    Fixture f(tflite::BuiltinOperator_CAST);
    const int in = f.Tensor(casts[i].first, {2, 3}, .125f, -2);
    const int out = f.Tensor(casts[i].second, {2, 3}, .125f, 3);
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }

  // ARG_MIN (1D -> 0D and 2D -> 1D across FP32, INT8, INT32).
  for (auto dt : {tflite::TensorType_FLOAT32, tflite::TensorType_INT8,
                  tflite::TensorType_INT32}) {
    for (bool rank1 : {true, false}) {
      SCOPED_TRACE(static_cast<int>(dt) * 10 + (rank1 ? 1 : 0));
      Fixture f(tflite::BuiltinOperator_ARG_MIN);
      const int in =
          f.Tensor(dt, rank1 ? std::vector<int32_t>{5} : std::vector<int32_t>{3, 5});
      const int axis = f.Control({}, {rank1 ? 0 : -1});
      const int out = f.Tensor(
          tflite::TensorType_INT32,
          rank1 ? std::vector<int32_t>{} : std::vector<int32_t>{3});
      f.op->inputs = {in, axis};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      tflite::ArgMinOptionsT opts;
      opts.output_type = tflite::TensorType_INT32;
      f.op->builtin_options.Set(opts);
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      auto compiled = CompilePartition(FirstSubgraph(model), reason);
      ASSERT_TRUE(compiled) << reason;
      EXPECT_EQ(compiled->outputs[0].logical_scalar, rank1);
    }
  }

  // SIGN (FP32 & INT32), ROUND, ELU, LOG_SOFTMAX, L2_NORMALIZATION, SQUARED_DIFFERENCE, ATAN2.
  for (auto dt : {tflite::TensorType_FLOAT32, tflite::TensorType_INT32}) {
    SCOPED_TRACE(static_cast<int>(dt));
    Fixture f(tflite::BuiltinOperator_SIGN);
    const int in = f.Tensor(dt, {2, 4});
    const int out = f.Tensor(dt, {2, 4});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  for (auto code : {tflite::BuiltinOperator_ROUND, tflite::BuiltinOperator_ELU,
                    tflite::BuiltinOperator_LOG_SOFTMAX,
                    tflite::BuiltinOperator_L2_NORMALIZATION}) {
    SCOPED_TRACE(static_cast<int>(code));
    Fixture f(code);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 5});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 5});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    if (code == tflite::BuiltinOperator_L2_NORMALIZATION) {
      tflite::L2NormOptionsT opts;
      opts.fused_activation_function = tflite::ActivationFunctionType_NONE;
      f.op->builtin_options.Set(opts);
    }
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_SQUARED_DIFFERENCE);
    const int a = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    const int b = f.Tensor(tflite::TensorType_FLOAT32, {1, 4});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a, b};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_ATAN2);
    const int y = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int x = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    f.op->inputs = {y, x};
    f.op->outputs = {out};
    f.graph->inputs = {y, x};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;

    f.graph->tensors[x]->shape = {1, 3};
    auto bad = f.Load();
    ASSERT_TRUE(bad);
    EXPECT_FALSE(AnalyzeDecomposition(FirstSubgraph(bad).Ops()[0], reason));
  }

  // INT8 comparisons, INT8/INT16 MEAN, INT8 BATCH_MATMUL broadcast, legacy SELECT,
  // and INT32 FLOOR_MOD + direct INT32 FLOOR_DIV EmitDecomposition regression check.
  for (auto code :
       {tflite::BuiltinOperator_GREATER, tflite::BuiltinOperator_GREATER_EQUAL,
        tflite::BuiltinOperator_LESS, tflite::BuiltinOperator_LESS_EQUAL}) {
    SCOPED_TRACE(static_cast<int>(code));
    Fixture f(code);
    const int a = f.Tensor(tflite::TensorType_INT8, {2, 1}, .125f, -3);
    const int b = f.Tensor(tflite::TensorType_INT8, {1, 3}, .125f, -3);
    const int out = f.Tensor(tflite::TensorType_BOOL, {2, 3});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a, b};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_MEAN);
    const int in = f.Tensor(tflite::TensorType_INT8, {2, 3, 4}, .125f, -5);
    const int axes = f.Control({2}, {1, 2});
    const int out = f.Tensor(tflite::TensorType_INT8, {2, 1, 1}, .125f, 2);
    f.op->inputs = {in, axes};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::ReducerOptionsT opts;
    opts.keep_dims = true;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_MEAN);
    const int in = f.Tensor(tflite::TensorType_INT16, {2, 4}, 20000.0f, 0);
    const int axes = f.Control({1}, {1});
    const int out = f.Tensor(tflite::TensorType_INT16, {2, 1}, 1.0f, 0);
    f.op->inputs = {in, axes};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::ReducerOptionsT opts;
    opts.keep_dims = true;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    // Large amplification scale exercises CanEmitNativeRescale multi-step check.
    (void)AnalyzeDecomposition(FirstSubgraph(model).Ops()[0], reason);
  }
  {
    Fixture f(tflite::BuiltinOperator_BATCH_MATMUL);
    const int a = f.Tensor(tflite::TensorType_INT8, {1, 2, 3}, .125f, -1);
    const int b = f.Tensor(tflite::TensorType_INT8, {2, 3, 4}, .125f, 2);
    const int out = f.Tensor(tflite::TensorType_INT8, {2, 2, 4}, .25f, 0);
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a, b};
    f.graph->outputs = {out};
    f.op->builtin_options.Set(tflite::BatchMatMulOptionsT{});
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_SELECT);
    const int cond = f.Tensor(tflite::TensorType_BOOL, {2});
    const int a = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 4});
    const int b = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 4});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 4});
    f.op->inputs = {cond, a, b};
    f.op->outputs = {out};
    f.graph->inputs = {cond, a, b};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    Fixture f(tflite::BuiltinOperator_FLOOR_MOD);
    const int a = f.Tensor(tflite::TensorType_INT32, {2, 3});
    const int b = f.Control({1, 3}, {3, 5, 7});
    const int out = f.Tensor(tflite::TensorType_INT32, {2, 3});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  for (auto code :
       {tflite::BuiltinOperator_FLOOR_DIV, tflite::BuiltinOperator_FLOOR_MOD}) {
    SCOPED_TRACE(static_cast<int>(code));
    Fixture f(code);
    const int a = f.Tensor(tflite::TensorType_FLOAT32, {2, 2});
    const int b = f.FloatConstant({1, 2}, {2.0f, -4.0f});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 2});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
