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

#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "tflite/schema/schema_generated.h"

namespace litert::arm_vulkan_ml {
namespace {

std::vector<uint8_t> AddModel(const std::vector<int32_t>& lhs,
                              const std::vector<int32_t>& rhs,
                              const std::vector<int32_t>& output) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  auto graph = std::make_unique<tflite::SubGraphT>();
  for (const auto& shape : {lhs, rhs, output}) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->shape = shape;
    tensor->type = tflite::TensorType_INT8;
    tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
    tensor->quantization->scale = {0.125f};
    tensor->quantization->zero_point = {-3};
    graph->tensors.push_back(std::move(tensor));
  }
  auto code = std::make_unique<tflite::OperatorCodeT>();
  code->builtin_code = tflite::BuiltinOperator_ADD;
  code->version = 2;
  model.operator_codes.push_back(std::move(code));
  auto op = std::make_unique<tflite::OperatorT>();
  op->inputs = {0, 1};
  op->outputs = {2};
  op->builtin_options.Set(tflite::AddOptionsT());
  graph->operators.push_back(std::move(op));
  graph->inputs = {0, 1};
  graph->outputs = {2};
  model.subgraphs.push_back(std::move(graph));
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

TEST(OperationTest, UnarySimplificationsPreserveRequantizingFallback) {
  for (auto code : {tflite::BuiltinOperator_RELU, tflite::BuiltinOperator_RELU6,
                    tflite::BuiltinOperator_QUANTIZE}) {
    for (int variant = 0; variant < 3; ++variant) {
      auto bytes = AddModel({1, 256}, {1}, {1, 256});
      auto object = tflite::UnPackModel(bytes.data());
      object->operator_codes[0]->builtin_code = code;
      auto& graph = *object->subgraphs[0];
      graph.inputs = {0};
      graph.operators[0]->inputs = {0};
      graph.operators[0]->builtin_options.Reset();
      if (variant == 1) graph.tensors[2]->quantization->scale = {0.25f};
      if (variant == 2) graph.tensors[2]->quantization->zero_point = {7};
      flatbuffers::FlatBufferBuilder builder;
      tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, object.get()));
      auto model = ExtendedModel::CreateFromBuffer(
          BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
      ASSERT_TRUE(model);
      auto subgraph = model->Subgraph(0);
      ASSERT_TRUE(subgraph);
      std::string reason;
      auto plan = AnalyzeOperation(subgraph->Ops()[0], reason);
      ASSERT_TRUE(plan) << reason;
      EXPECT_EQ(plan->table.size(), variant == 0 ? 0 : 256);
      if (variant == 0 && code != tflite::BuiltinOperator_QUANTIZE) {
        EXPECT_EQ(plan->activation.minimum, -3);
        EXPECT_EQ(plan->activation.maximum,
                  code == tflite::BuiltinOperator_RELU6 ? 45 : 127);
      }
      auto compiled = CompilePartition(*subgraph, reason);
      ASSERT_TRUE(compiled) << reason;
      EXPECT_EQ(compiled->constants.size(), variant == 0 ? 0 : 1);
    }
  }
}

TEST(OperationTest, GatherRequiresConstantInRangeIndicesAndMatchingQuantization) {
  for (int variant = 0; variant < 5; ++variant) {
    auto bytes = AddModel({2, 3, 4}, {2}, {2, 2, 4});
    auto object = tflite::UnPackModel(bytes.data());
    object->operator_codes[0]->builtin_code = tflite::BuiltinOperator_GATHER;
    auto& graph = *object->subgraphs[0];
    graph.inputs = variant == 1 ? std::vector<int32_t>{0, 1} : std::vector<int32_t>{0};
    graph.tensors[1]->type = tflite::TensorType_INT32;
    graph.tensors[1]->quantization.reset();
    std::vector<int32_t> indices = {2, variant == 2 ? -1 : 0};
    if (variant != 1) {
      auto buffer = std::make_unique<tflite::BufferT>();
      buffer->data.resize(indices.size() * sizeof(int32_t));
      std::memcpy(buffer->data.data(), indices.data(), buffer->data.size());
      graph.tensors[1]->buffer = 1;
      object->buffers.push_back(std::move(buffer));
    }
    tflite::GatherOptionsT options;
    options.axis = -2;
    options.batch_dims = variant == 3 ? 1 : 0;
    graph.operators[0]->builtin_options.Set(options);
    if (variant == 4) graph.tensors[2]->quantization->scale = {0.25f};
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, object.get()));
    auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)), variant == 0) << reason;
  }
}

TEST(OperationTest, ReductionNormalizesAxesAndChecksOutputShape) {
  for (auto code : {tflite::BuiltinOperator_REDUCE_MAX, tflite::BuiltinOperator_REDUCE_MIN}) {
    for (int variant = 0; variant < 3; ++variant) {
      auto bytes = AddModel({2, 3, 4}, {3}, variant == 2 ? std::vector<int32_t>{3} : std::vector<int32_t>{2});
      auto object = tflite::UnPackModel(bytes.data());
      object->operator_codes[0]->builtin_code = code;
      auto& graph = *object->subgraphs[0];
      graph.inputs = {0};
      graph.tensors[1]->type = tflite::TensorType_INT32;
      graph.tensors[1]->quantization.reset();
      graph.tensors[1]->buffer = 1;
      std::vector<int32_t> axes = {1, variant == 1 ? -4 : -1, 1};
      auto buffer = std::make_unique<tflite::BufferT>();
      buffer->data.resize(axes.size() * sizeof(int32_t));
      std::memcpy(buffer->data.data(), axes.data(), buffer->data.size());
      object->buffers.push_back(std::move(buffer));
      tflite::ReducerOptionsT options;
      options.keep_dims = false;
      graph.operators[0]->builtin_options.Set(options);
      flatbuffers::FlatBufferBuilder builder;
      tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, object.get()));
      auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
      ASSERT_TRUE(model);
      auto subgraph = model->Subgraph(0);
      ASSERT_TRUE(subgraph);
      std::string reason;
      EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)), variant == 0) << reason;
    }
  }
}

TEST(OperationTest, ResizeChecksCoordinatesSizesAndIntegerBilinear) {
  for (int variant = 0; variant < 4; ++variant) {
    auto bytes = AddModel({1, 3, 4, 2}, {2}, {1, 5, 7, 2});
    auto object = tflite::UnPackModel(bytes.data());
    object->operator_codes[0]->builtin_code = variant == 3 ? tflite::BuiltinOperator_RESIZE_BILINEAR : tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR;
    auto& graph = *object->subgraphs[0];
    graph.inputs = {0};
    graph.tensors[1]->type = tflite::TensorType_INT32;
    graph.tensors[1]->quantization.reset();
    graph.tensors[1]->buffer = 1;
    std::vector<int32_t> sizes = {variant == 2 ? 6 : 5, 7};
    auto buffer = std::make_unique<tflite::BufferT>();
    buffer->data.resize(sizes.size() * sizeof(int32_t));
    std::memcpy(buffer->data.data(), sizes.data(), buffer->data.size());
    object->buffers.push_back(std::move(buffer));
    if (variant == 3) graph.operators[0]->builtin_options.Set(tflite::ResizeBilinearOptionsT());
    else {
      tflite::ResizeNearestNeighborOptionsT options;
      options.align_corners = variant == 1;
      options.half_pixel_centers = true;
      graph.operators[0]->builtin_options.Set(options);
    }
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, object.get()));
    auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)), variant == 0) << reason;
  }
}

TEST(OperationTest, TransposeConvolutionUsesDataInputTwoAndRejectsInvalidGeometry) {
  for (int variant = 0; variant < 3; ++variant) {
    auto bytes = AddModel({4}, {1, 3, 3, 1}, {1, 2, 2, 1});
    auto object = tflite::UnPackModel(bytes.data());
    object->operator_codes[0]->builtin_code = tflite::BuiltinOperator_TRANSPOSE_CONV;
    auto& graph = *object->subgraphs[0];
    for (auto& tensor : graph.tensors) {
      tensor->type = tflite::TensorType_FLOAT32;
      tensor->quantization.reset();
    }
    graph.tensors[0]->type = tflite::TensorType_INT32;
    graph.tensors[0]->buffer = 1;
    graph.tensors[1]->buffer = 2;
    auto output = std::make_unique<tflite::TensorT>();
    output->shape = {1, variant == 1 ? 5 : 4, 4, 1};
    output->type = tflite::TensorType_FLOAT32;
    std::vector<int32_t> shape = output->shape;
    graph.tensors.push_back(std::move(output));
    auto shape_buffer = std::make_unique<tflite::BufferT>();
    shape_buffer->data.resize(shape.size() * sizeof(int32_t));
    std::memcpy(shape_buffer->data.data(), shape.data(), shape_buffer->data.size());
    object->buffers.push_back(std::move(shape_buffer));
    std::vector<float> weights(9, 0.25f);
    auto weight_buffer = std::make_unique<tflite::BufferT>();
    weight_buffer->data.resize(weights.size() * sizeof(float));
    std::memcpy(weight_buffer->data.data(), weights.data(), weight_buffer->data.size());
    object->buffers.push_back(std::move(weight_buffer));
    graph.inputs = {2};
    graph.outputs = {3};
    graph.operators[0]->inputs = {0, 1, 2};
    graph.operators[0]->outputs = {3};
    tflite::TransposeConvOptionsT options;
    options.padding = tflite::Padding_SAME;
    options.stride_h = options.stride_w = variant == 2 ? 4 : 2;
    graph.operators[0]->builtin_options.Set(options);
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, object.get()));
    auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    auto plan = AnalyzeOperation(subgraph->Ops()[0], reason);
    EXPECT_EQ(static_cast<bool>(plan), variant == 0) << reason;
    if (plan) EXPECT_EQ(plan->runtime_inputs, std::vector<int>{2});
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)), variant == 0) << reason;
  }
}

TEST(OperationTest, BroadcastAddCompilesWithChannelVectorOnEitherSide) {
  for (bool swap : {false, true}) {
    std::vector<int32_t> full = {1, 4, 5, 8}, channel = {8};
    auto bytes = AddModel(swap ? channel : full, swap ? full : channel, full);
    auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(bytes.data(), bytes.size()));
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    auto plan = AnalyzeOperation(graph->Ops()[0], reason);
    ASSERT_TRUE(plan) << reason;
    EXPECT_EQ(plan->output.shape, (std::vector<uint32_t>{1, 4, 5, 8}));
    auto compiled = CompilePartition(*graph, reason);
    ASSERT_TRUE(compiled) << reason;
  }
}

TEST(OperationTest, RejectsIncompatibleBroadcastAndIncorrectOutput) {
  for (const auto& shapes : std::vector<std::vector<std::vector<int32_t>>>{
           {{1, 4, 5, 8}, {7}, {1, 4, 5, 8}},
           {{1, 4, 5, 8}, {8}, {1, 4, 5, 1}}}) {
    auto bytes = AddModel(shapes[0], shapes[1], shapes[2]);
    auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(bytes.data(), bytes.size()));
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason));
    EXPECT_FALSE(CompilePartition(*graph, reason));
    EXPECT_FALSE(reason.empty());
  }
}

std::vector<uint8_t> LogisticModel(bool floating, float output_scale = 1.0f / 256) {
  auto bytes = AddModel({1, 256}, {1, 256}, {1, 256});
  auto model = tflite::UnPackModel(bytes.data());
  model->operator_codes[0]->builtin_code = tflite::BuiltinOperator_LOGISTIC;
  auto& graph = *model->subgraphs[0];
  graph.inputs = {0};
  graph.operators[0]->inputs = {0};
  graph.operators[0]->builtin_options.Reset();
  graph.tensors[2]->quantization->scale = {output_scale};
  graph.tensors[2]->quantization->zero_point = {-128};
  if (floating) for (auto& tensor : graph.tensors) {
    tensor->type = tflite::TensorType_FLOAT32;
    tensor->quantization.reset();
  }
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, model.get()));
  return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

TEST(OperationTest, LogisticCompilesFloatAndInt8AndRejectsInvalidOutputScale) {
  for (bool floating : {false, true}) {
    auto bytes = LogisticModel(floating);
    auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(bytes.data(), bytes.size()));
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    auto plan = AnalyzeOperation(graph->Ops()[0], reason);
    ASSERT_TRUE(plan) << reason;
    if (!floating) {
      ASSERT_EQ(plan->table.size(), 256);
      EXPECT_EQ(plan->table.front(), -128);
      EXPECT_EQ(plan->table.back(), 127);
      EXPECT_EQ(plan->table[125], 0);  // Input zero point is -3.
      EXPECT_TRUE(std::is_sorted(plan->table.begin(), plan->table.end()));
    }
    EXPECT_TRUE(CompilePartition(*graph, reason)) << reason;
  }
  auto bytes = LogisticModel(false, 0.125f);
  auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(bytes.data(), bytes.size()));
  ASSERT_TRUE(model);
  auto graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  std::string reason;
  EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason));
}

TEST(OperationTest, FloatBroadcastArithmeticChecksShapesAndTypes) {
  for (auto code : {tflite::BuiltinOperator_ADD, tflite::BuiltinOperator_MUL}) {
    for (bool valid : {true, false}) {
      auto bytes = AddModel({1, 4, 5, 8}, {8}, valid ? std::vector<int32_t>{1, 4, 5, 8}
                                                  : std::vector<int32_t>{1, 4, 5, 1});
      auto unpacked = tflite::UnPackModel(bytes.data());
      unpacked->operator_codes[0]->builtin_code = code;
      if (code == tflite::BuiltinOperator_MUL)
        unpacked->subgraphs[0]->operators[0]->builtin_options.Set(tflite::MulOptionsT());
      for (auto& tensor : unpacked->subgraphs[0]->tensors) {
        tensor->type = tflite::TensorType_FLOAT32;
        tensor->quantization.reset();
      }
      flatbuffers::FlatBufferBuilder builder;
      tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, unpacked.get()));
      auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
      ASSERT_TRUE(model);
      auto graph = model->Subgraph(0);
      ASSERT_TRUE(graph);
      std::string reason;
      EXPECT_EQ(static_cast<bool>(CompilePartition(*graph, reason)), valid) << reason;
    }
  }
}

TEST(OperationTest, FloatGlobalAveragePoolRejectsMismatchedGeometryAndMixedTypes) {
  for (int variant = 0; variant < 4; ++variant) {
    auto bytes = AddModel({1, 4, 5, 8}, {1}, {1, 1, 1, 8});
    auto unpacked = tflite::UnPackModel(bytes.data());
    unpacked->operator_codes[0]->builtin_code = tflite::BuiltinOperator_AVERAGE_POOL_2D;
    auto& graph = *unpacked->subgraphs[0];
    graph.inputs = {0};
    graph.operators[0]->inputs = {0};
    tflite::Pool2DOptionsT options;
    options.padding = variant == 1 ? tflite::Padding_SAME : tflite::Padding_VALID;
    options.filter_height = variant == 2 ? 3 : 4;
    options.filter_width = 5;
    options.stride_h = options.stride_w = 1;
    graph.operators[0]->builtin_options.Set(options);
    for (auto& tensor : graph.tensors) {
      tensor->type = tflite::TensorType_FLOAT32;
      tensor->quantization.reset();
    }
    if (variant == 3) {
      graph.tensors[2]->type = tflite::TensorType_INT8;
      graph.tensors[2]->quantization = std::make_unique<tflite::QuantizationParametersT>();
      graph.tensors[2]->quantization->scale = {0.125f};
      graph.tensors[2]->quantization->zero_point = {0};
    }
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, unpacked.get()));
    auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)), variant == 0) << reason;
  }
}

TEST(OperationTest, DisjointInt8AveragePoolAcceptsCroppedEdgesButRejectsOverlap) {
  for (int variant = 0; variant < 3; ++variant) {
    auto bytes = AddModel({1, 5, 5, 3}, {1}, variant == 1 ? std::vector<int32_t>{1, 4, 4, 3}
                                                       : std::vector<int32_t>{1, 2, 2, 3});
    auto unpacked = tflite::UnPackModel(bytes.data());
    unpacked->operator_codes[0]->builtin_code = tflite::BuiltinOperator_AVERAGE_POOL_2D;
    auto& graph = *unpacked->subgraphs[0];
    graph.inputs = {0};
    graph.operators[0]->inputs = {0};
    tflite::Pool2DOptionsT options;
    options.padding = tflite::Padding_VALID;
    options.filter_height = options.filter_width = 2;
    options.stride_h = options.stride_w = variant == 1 ? 1 : 2;
    graph.operators[0]->builtin_options.Set(options);
    if (variant == 2) graph.tensors[2]->quantization->scale = {0.25f};
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, unpacked.get()));
    auto model = ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)), variant == 0) << reason;
  }
}

}  // namespace
}  // namespace litert::arm_vulkan_ml
