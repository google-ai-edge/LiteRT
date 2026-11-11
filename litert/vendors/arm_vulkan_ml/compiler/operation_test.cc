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
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "litert/vendors/arm_vulkan_ml/compiler/graph_ir.h"
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "litert/vendors/arm_vulkan_ml/compiler/softmax.h"
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
  return {builder.GetBufferPointer(),
          builder.GetBufferPointer() + builder.GetSize()};
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
      tflite::FinishModelBuffer(builder,
                                tflite::Model::Pack(builder, object.get()));
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
      EXPECT_EQ(compiled->constants.empty(), variant == 0);
    }
  }
}

TEST(OperationTest,
     QuantizedSoftmaxValidatesOutputQuantizationAndRequirements) {
  for (int variant = 0; variant < 5; ++variant) {
    auto bytes = AddModel({2, variant == 3 ? 4096 : 5}, {1},
                          {2, variant == 3 ? 4096 : 5});
    auto object = tflite::UnPackModel(bytes.data());
    object->operator_codes[0]->builtin_code = tflite::BuiltinOperator_SOFTMAX;
    auto& graph = *object->subgraphs[0];
    graph.inputs = {0};
    graph.operators[0]->inputs = {0};
    tflite::SoftmaxOptionsT options;
    options.beta = variant == 4 ? 0 : .7f;
    graph.operators[0]->builtin_options.Set(options);
    graph.tensors[2]->quantization->scale = {variant == 1 ? .125f : 1.0f / 256};
    graph.tensors[2]->quantization->zero_point = {variant == 2 ? 0 : -128};
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, object.get()));
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    for (auto lut :
         {SoftmaxLutType::kAuto, SoftmaxLutType::kInt8,
          SoftmaxLutType::kInt16}) {
      std::string reason;
      auto compiled =
          CompilePartition(*subgraph, reason, nullptr, nullptr, lut);
      ASSERT_EQ(static_cast<bool>(compiled), variant == 0) << reason;
      if (!compiled) continue;
      EXPECT_EQ(compiled->profiles, (std::vector<std::string>{"PRO-INT"}));
      EXPECT_EQ(compiled->extensions,
                lut == SoftmaxLutType::kInt16
                    ? std::vector<std::string>{"EXT-INT16"}
                    : std::vector<std::string>{});
    }
  }
}

TEST(OperationTest, ScalarConstantActivationIsImportedAndOwned) {
  auto bytes = AddModel({1, 256}, {}, {1, 256});
  auto object = tflite::UnPackModel(bytes.data());
  object->buffers.push_back(std::make_unique<tflite::BufferT>());
  object->buffers[1]->data = {17};
  object->subgraphs[0]->tensors[1]->buffer = 1;
  object->subgraphs[0]->inputs = {0};
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder,
                            tflite::Model::Pack(builder, object.get()));
  auto model = ExtendedModel::CreateFromBuffer(
      BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
  ASSERT_TRUE(model);
  auto subgraph = model->Subgraph(0);
  ASSERT_TRUE(subgraph);
  std::string reason;
  ASSERT_TRUE(CompilePartition(*subgraph, reason)) << reason;
  auto ir = ImportGraph(*subgraph, reason);
  ASSERT_TRUE(ir) << reason;
  auto constant = ir->nodes[0].inputs[1];
  EXPECT_EQ(ir->values[constant].shape, (std::vector<uint32_t>{1}));
  EXPECT_TRUE(ir->values[constant].logical_scalar);
  EXPECT_EQ(ir->values[constant].constant, (std::vector<uint8_t>{17}));
}

TEST(OperationTest, Int8AveragePoolAcceptsOverlappingValidWindows) {
  for (int variant = 0; variant < 6; ++variant) {
    const int stride = variant == 1 ? 2 : 1;
    auto bytes = AddModel(
        {1, 6, 7, 2}, {1},
        {1, variant == 2 ? 5 : (6 - 3) / stride + 1, (7 - 3) / stride + 1, 2});
    auto object = tflite::UnPackModel(bytes.data());
    object->operator_codes[0]->builtin_code =
        tflite::BuiltinOperator_AVERAGE_POOL_2D;
    auto& graph = *object->subgraphs[0];
    graph.inputs = {0};
    graph.operators[0]->inputs = {0};
    tflite::Pool2DOptionsT options;
    options.padding =
        variant == 3 ? tflite::Padding_SAME : tflite::Padding_VALID;
    options.filter_height = options.filter_width = 3;
    options.stride_h = options.stride_w = stride;
    if (variant == 5)
      options.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    graph.operators[0]->builtin_options.Set(options);
    if (variant == 4) graph.tensors[2]->quantization->scale = {0.25f};
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, object.get()));
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    auto plan = AnalyzeOperation(subgraph->Ops()[0], reason);
    const bool supported = variant == 0 || variant == 1 || variant == 5;
    ASSERT_EQ(plan.has_value(), supported) << reason;
    if (!supported) continue;
    EXPECT_EQ(plan->cropped_shape,
              (std::vector<uint32_t>{1, variant == 1 ? 5u : 6u, 7, 2}));
    EXPECT_TRUE(CompilePartition(*subgraph, reason)) << reason;
  }
}

TEST(OperationTest,
     GatherRequiresConstantInRangeIndicesAndMatchingQuantization) {
  for (int variant = 0; variant < 5; ++variant) {
    auto bytes = AddModel({2, 3, 4}, {2}, {2, 2, 4});
    auto object = tflite::UnPackModel(bytes.data());
    object->operator_codes[0]->builtin_code = tflite::BuiltinOperator_GATHER;
    auto& graph = *object->subgraphs[0];
    graph.inputs =
        variant == 1 ? std::vector<int32_t>{0, 1} : std::vector<int32_t>{0};
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
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, object.get()));
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)),
              variant == 0)
        << reason;
  }
}

TEST(OperationTest, ReductionNormalizesAxesAndChecksOutputShape) {
  for (auto code :
       {tflite::BuiltinOperator_REDUCE_MAX, tflite::BuiltinOperator_REDUCE_MIN,
        tflite::BuiltinOperator_SUM}) {
    for (int variant = 0; variant < 3; ++variant) {
      auto bytes = AddModel(
          {2, 3, 4}, {3},
          variant == 2 ? std::vector<int32_t>{3} : std::vector<int32_t>{2});
      auto object = tflite::UnPackModel(bytes.data());
      object->operator_codes[0]->builtin_code = code;
      auto& graph = *object->subgraphs[0];
      if (code == tflite::BuiltinOperator_SUM) {
        for (int index : {0, 2}) {
          graph.tensors[index]->type = tflite::TensorType_FLOAT32;
          graph.tensors[index]->quantization.reset();
        }
      }
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
      tflite::FinishModelBuffer(builder,
                                tflite::Model::Pack(builder, object.get()));
      auto model = ExtendedModel::CreateFromBuffer(
          BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
      ASSERT_TRUE(model);
      auto subgraph = model->Subgraph(0);
      ASSERT_TRUE(subgraph);
      std::string reason;
      EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)),
                variant == 0)
          << reason;
    }
  }
}

TEST(OperationTest, ResizeChecksCoordinatesSizesAndIntegerBilinear) {
  for (int variant = 0; variant < 5; ++variant) {
    auto bytes = AddModel({1, 3, 4, 2}, {2}, {1, 5, 7, 2});
    auto object = tflite::UnPackModel(bytes.data());
    object->operator_codes[0]->builtin_code =
        variant >= 3 ? tflite::BuiltinOperator_RESIZE_BILINEAR
                     : tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR;
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
    if (variant >= 3)
      graph.operators[0]->builtin_options.Set(tflite::ResizeBilinearOptionsT());
    else {
      tflite::ResizeNearestNeighborOptionsT options;
      options.align_corners = variant == 1;
      options.half_pixel_centers = true;
      graph.operators[0]->builtin_options.Set(options);
    }
    if (variant == 4) graph.tensors[2]->quantization->scale = {0.25f};
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, object.get()));
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)),
              (variant == 0 || variant == 3))
        << reason;
  }
}

TEST(OperationTest,
     TransposeConvolutionUsesDataInputTwoAndRejectsInvalidGeometry) {
  for (int variant = 0; variant < 3; ++variant) {
    auto bytes = AddModel({4}, {1, 3, 3, 1}, {1, 2, 2, 1});
    auto object = tflite::UnPackModel(bytes.data());
    object->operator_codes[0]->builtin_code =
        tflite::BuiltinOperator_TRANSPOSE_CONV;
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
    std::memcpy(shape_buffer->data.data(), shape.data(),
                shape_buffer->data.size());
    object->buffers.push_back(std::move(shape_buffer));
    std::vector<float> weights(9, 0.25f);
    auto weight_buffer = std::make_unique<tflite::BufferT>();
    weight_buffer->data.resize(weights.size() * sizeof(float));
    std::memcpy(weight_buffer->data.data(), weights.data(),
                weight_buffer->data.size());
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
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, object.get()));
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    auto plan = AnalyzeOperation(subgraph->Ops()[0], reason);
    EXPECT_EQ(static_cast<bool>(plan), variant == 0) << reason;
    if (plan) EXPECT_EQ(plan->runtime_inputs, std::vector<int>{2});
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)),
              variant == 0)
        << reason;
  }
}

TEST(OperationTest, FloatFullyConnectedChecksConstantsShapeAndOptions) {
  for (int variant = 0; variant < 4; ++variant) {
    auto bytes = AddModel({2, 4}, {3, 4}, {3});
    auto object = tflite::UnPackModel(bytes.data());
    object->operator_codes[0]->builtin_code =
        tflite::BuiltinOperator_FULLY_CONNECTED;
    auto& graph = *object->subgraphs[0];
    for (auto& tensor : graph.tensors) {
      tensor->type = tflite::TensorType_FLOAT32;
      tensor->quantization.reset();
    }
    auto output = std::make_unique<tflite::TensorT>();
    output->type = tflite::TensorType_FLOAT32;
    output->shape = {variant == 2 ? 3 : 2, 3};
    graph.tensors.push_back(std::move(output));
    graph.tensors[1]->buffer = 1;
    graph.tensors[2]->buffer = 2;
    if (variant == 3) graph.tensors[2]->type = tflite::TensorType_INT32;
    for (size_t count : {12, 3}) {
      std::vector<float> values(count, 0.25f);
      auto buffer = std::make_unique<tflite::BufferT>();
      buffer->data.resize(values.size() * sizeof(float));
      std::memcpy(buffer->data.data(), values.data(), buffer->data.size());
      object->buffers.push_back(std::move(buffer));
    }
    graph.inputs = {0};
    graph.outputs = {3};
    graph.operators[0]->inputs = {0, 1, 2};
    graph.operators[0]->outputs = {3};
    tflite::FullyConnectedOptionsT options;
    options.keep_num_dims = variant == 1;
    options.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    graph.operators[0]->builtin_options.Set(options);
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, object.get()));
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)),
              variant == 0)
        << reason;
  }
}

TEST(OperationTest, BroadcastAddCompilesWithChannelVectorOnEitherSide) {
  for (bool swap : {false, true}) {
    std::vector<int32_t> full = {1, 4, 5, 8}, channel = {8};
    auto bytes = AddModel(swap ? channel : full, swap ? full : channel, full);
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(bytes.data(), bytes.size()));
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
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(bytes.data(), bytes.size()));
    ASSERT_TRUE(model);
    auto graph = model->Subgraph(0);
    ASSERT_TRUE(graph);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason));
    EXPECT_FALSE(CompilePartition(*graph, reason));
    EXPECT_FALSE(reason.empty());
  }
}

std::vector<uint8_t> LogisticModel(bool floating,
                                   float output_scale = 1.0f / 256) {
  auto bytes = AddModel({1, 256}, {1, 256}, {1, 256});
  auto model = tflite::UnPackModel(bytes.data());
  model->operator_codes[0]->builtin_code = tflite::BuiltinOperator_LOGISTIC;
  auto& graph = *model->subgraphs[0];
  graph.inputs = {0};
  graph.operators[0]->inputs = {0};
  graph.operators[0]->builtin_options.Reset();
  graph.tensors[2]->quantization->scale = {output_scale};
  graph.tensors[2]->quantization->zero_point = {-128};
  if (floating)
    for (auto& tensor : graph.tensors) {
      tensor->type = tflite::TensorType_FLOAT32;
      tensor->quantization.reset();
    }
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, model.get()));
  return {builder.GetBufferPointer(),
          builder.GetBufferPointer() + builder.GetSize()};
}

TEST(OperationTest, LogisticCompilesFloatAndInt8AndRejectsInvalidOutputScale) {
  for (bool floating : {false, true}) {
    auto bytes = LogisticModel(floating);
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(bytes.data(), bytes.size()));
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
  auto model = ExtendedModel::CreateFromBuffer(
      BufferRef<uint8_t>(bytes.data(), bytes.size()));
  ASSERT_TRUE(model);
  auto graph = model->Subgraph(0);
  ASSERT_TRUE(graph);
  std::string reason;
  EXPECT_FALSE(AnalyzeOperation(graph->Ops()[0], reason));
}

TEST(OperationTest, FloatBroadcastArithmeticChecksShapesAndTypes) {
  for (auto code : {tflite::BuiltinOperator_ADD, tflite::BuiltinOperator_MUL}) {
    for (bool valid : {true, false}) {
      auto bytes = AddModel({1, 4, 5, 8}, {8},
                            valid ? std::vector<int32_t>{1, 4, 5, 8}
                                  : std::vector<int32_t>{1, 4, 5, 1});
      auto unpacked = tflite::UnPackModel(bytes.data());
      unpacked->operator_codes[0]->builtin_code = code;
      if (code == tflite::BuiltinOperator_MUL)
        unpacked->subgraphs[0]->operators[0]->builtin_options.Set(
            tflite::MulOptionsT());
      for (auto& tensor : unpacked->subgraphs[0]->tensors) {
        tensor->type = tflite::TensorType_FLOAT32;
        tensor->quantization.reset();
      }
      flatbuffers::FlatBufferBuilder builder;
      tflite::FinishModelBuffer(builder,
                                tflite::Model::Pack(builder, unpacked.get()));
      auto model = ExtendedModel::CreateFromBuffer(
          BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
      ASSERT_TRUE(model);
      auto graph = model->Subgraph(0);
      ASSERT_TRUE(graph);
      std::string reason;
      EXPECT_EQ(static_cast<bool>(CompilePartition(*graph, reason)), valid)
          << reason;
    }
  }
}

TEST(OperationTest,
     FloatGlobalAveragePoolRejectsMismatchedGeometryAndMixedTypes) {
  for (int variant = 0; variant < 4; ++variant) {
    auto bytes = AddModel({1, 4, 5, 8}, {1}, {1, 1, 1, 8});
    auto unpacked = tflite::UnPackModel(bytes.data());
    unpacked->operator_codes[0]->builtin_code =
        tflite::BuiltinOperator_AVERAGE_POOL_2D;
    auto& graph = *unpacked->subgraphs[0];
    graph.inputs = {0};
    graph.operators[0]->inputs = {0};
    tflite::Pool2DOptionsT options;
    options.padding =
        variant == 1 ? tflite::Padding_SAME : tflite::Padding_VALID;
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
      graph.tensors[2]->quantization =
          std::make_unique<tflite::QuantizationParametersT>();
      graph.tensors[2]->quantization->scale = {0.125f};
      graph.tensors[2]->quantization->zero_point = {0};
    }
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, unpacked.get()));
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)),
              variant == 0)
        << reason;
  }
}

TEST(OperationTest,
     Int8AveragePoolAcceptsCroppedEdgesAndRequiresMatchingQuantization) {
  for (int variant = 0; variant < 3; ++variant) {
    auto bytes = AddModel({1, 5, 5, 3}, {1},
                          variant == 1 ? std::vector<int32_t>{1, 4, 4, 3}
                                       : std::vector<int32_t>{1, 2, 2, 3});
    auto unpacked = tflite::UnPackModel(bytes.data());
    unpacked->operator_codes[0]->builtin_code =
        tflite::BuiltinOperator_AVERAGE_POOL_2D;
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
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, unpacked.get()));
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(CompilePartition(*subgraph, reason)),
              variant != 2)
        << reason;
  }
}

TEST(OperationTest, Int16AddMulAndUnarySupportSymmetryAndFastPaths) {
  // Equal-scale and mismatched-scale INT16 ADD and MUL.
  for (auto code : {tflite::BuiltinOperator_ADD, tflite::BuiltinOperator_MUL}) {
    for (bool equal_scale : {true, false}) {
      auto bytes = AddModel({1, 12, 8}, {1, 1, 8}, {1, 12, 8});
      auto unpacked = tflite::UnPackModel(bytes.data());
      unpacked->operator_codes[0]->builtin_code = code;
      auto& graph = *unpacked->subgraphs[0];
      for (size_t i = 0; i < graph.tensors.size(); ++i) {
        graph.tensors[i]->type = tflite::TensorType_INT16;
        graph.tensors[i]->quantization->scale = {
            (!equal_scale && i == 1) ? 0.25f : 0.125f};
        graph.tensors[i]->quantization->zero_point = {0};
      }
      if (code == tflite::BuiltinOperator_MUL) {
        tflite::MulOptionsT opts;
        opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
        graph.operators[0]->builtin_options.Set(opts);
      } else {
        tflite::AddOptionsT opts;
        opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
        graph.operators[0]->builtin_options.Set(opts);
      }
      flatbuffers::FlatBufferBuilder builder;
      tflite::FinishModelBuffer(builder,
                                tflite::Model::Pack(builder, unpacked.get()));
      auto model = ExtendedModel::CreateFromBuffer(
          BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
      ASSERT_TRUE(model);
      auto subgraph = model->Subgraph(0);
      ASSERT_TRUE(subgraph);
      std::string reason;
      auto plan = AnalyzeOperation(subgraph->Ops()[0], reason);
      ASSERT_TRUE(plan) << reason;
      if (code == tflite::BuiltinOperator_ADD) {
        EXPECT_EQ(plan->equal_scale_fast_path, equal_scale);
      }
      auto compiled = CompilePartition(*subgraph, reason);
      ASSERT_TRUE(compiled) << reason;
      EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-INT16"}));
    }
  }

  // INT16 RELU, RELU6, QUANTIZE (same-scale and rescaled) must never build
  // 256-entry INT8 lookup tables.
  for (auto code : {tflite::BuiltinOperator_RELU, tflite::BuiltinOperator_RELU6,
                    tflite::BuiltinOperator_QUANTIZE}) {
    for (float out_scale : {0.125f, 0.25f}) {
      auto bytes = AddModel({1, 64}, {1}, {1, 64});
      auto unpacked = tflite::UnPackModel(bytes.data());
      unpacked->operator_codes[0]->builtin_code = code;
      auto& graph = *unpacked->subgraphs[0];
      graph.inputs = {0};
      graph.operators[0]->inputs = {0};
      graph.operators[0]->builtin_options.Reset();
      graph.tensors[0]->type = tflite::TensorType_INT16;
      graph.tensors[0]->quantization->scale = {0.125f};
      graph.tensors[0]->quantization->zero_point = {0};
      graph.tensors[2]->type = tflite::TensorType_INT16;
      graph.tensors[2]->quantization->scale = {out_scale};
      graph.tensors[2]->quantization->zero_point = {0};
      flatbuffers::FlatBufferBuilder builder;
      tflite::FinishModelBuffer(builder,
                                tflite::Model::Pack(builder, unpacked.get()));
      auto model = ExtendedModel::CreateFromBuffer(
          BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
      ASSERT_TRUE(model);
      auto subgraph = model->Subgraph(0);
      ASSERT_TRUE(subgraph);
      std::string reason;
      auto plan = AnalyzeOperation(subgraph->Ops()[0], reason);
      ASSERT_TRUE(plan) << reason;
      EXPECT_TRUE(plan->table.empty());
      auto compiled = CompilePartition(*subgraph, reason);
      ASSERT_TRUE(compiled) << reason;
      EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-INT16"}));
    }
  }
}

TEST(OperationTest, Int16LutActivationsSoftmaxLayoutPoolAndSumCompile) {
  // LOGISTIC, TANH, SOFTMAX in symmetric INT16 (out_scale = 1/32768, zp = 0).
  for (auto code :
       {tflite::BuiltinOperator_LOGISTIC, tflite::BuiltinOperator_TANH,
        tflite::BuiltinOperator_SOFTMAX}) {
    auto bytes = AddModel({2, 16}, {1}, {2, 16});
    auto unpacked = tflite::UnPackModel(bytes.data());
    unpacked->operator_codes[0]->builtin_code = code;
    auto& graph = *unpacked->subgraphs[0];
    graph.inputs = {0};
    graph.operators[0]->inputs = {0};
    graph.operators[0]->builtin_options.Reset();
    if (code == tflite::BuiltinOperator_SOFTMAX) {
      tflite::SoftmaxOptionsT opts;
      opts.beta = 1.0f;
      graph.operators[0]->builtin_options.Set(opts);
    }
    graph.tensors[0]->type = tflite::TensorType_INT16;
    graph.tensors[0]->quantization->scale = {0.01f};
    graph.tensors[0]->quantization->zero_point = {0};
    graph.tensors[2]->type = tflite::TensorType_INT16;
    graph.tensors[2]->quantization->scale = {1.0f / 32768.0f};
    graph.tensors[2]->quantization->zero_point = {0};
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder,
                              tflite::Model::Pack(builder, unpacked.get()));
    auto model = ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
    ASSERT_TRUE(model);
    auto subgraph = model->Subgraph(0);
    ASSERT_TRUE(subgraph);
    std::string reason;
    auto plan = AnalyzeOperation(subgraph->Ops()[0], reason);
    ASSERT_TRUE(plan) << reason;
    if (code != tflite::BuiltinOperator_SOFTMAX) {
      EXPECT_EQ(plan->table16.size(), 513);
    } else {
      EXPECT_TRUE(plan->softmax16.has_value());
    }
    auto compiled = CompilePartition(*subgraph, reason);
    ASSERT_TRUE(compiled) << reason;
    EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-INT16"}));
  }

  // INT16 TRANSPOSE, SLICE, AVERAGE_POOL_2D, MAX_POOL_2D, SUM.
  {
    auto bytes = AddModel({1, 4, 4, 3}, {4}, {1, 2, 2, 3});
    auto unpacked = tflite::UnPackModel(bytes.data());
    for (auto code : {tflite::BuiltinOperator_AVERAGE_POOL_2D,
                      tflite::BuiltinOperator_MAX_POOL_2D}) {
      unpacked->operator_codes[0]->builtin_code = code;
      auto& graph = *unpacked->subgraphs[0];
      graph.inputs = {0};
      graph.operators[0]->inputs = {0};
      tflite::Pool2DOptionsT opts;
      opts.padding = tflite::Padding_VALID;
      opts.filter_height = opts.filter_width = 2;
      opts.stride_h = opts.stride_w = 2;
      graph.operators[0]->builtin_options.Set(opts);
      for (int idx : {0, 2}) {
        graph.tensors[idx]->type = tflite::TensorType_INT16;
        graph.tensors[idx]->quantization->scale = {0.125f};
        graph.tensors[idx]->quantization->zero_point = {0};
      }
      flatbuffers::FlatBufferBuilder builder;
      tflite::FinishModelBuffer(builder,
                                tflite::Model::Pack(builder, unpacked.get()));
      auto model = ExtendedModel::CreateFromBuffer(
          BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
      ASSERT_TRUE(model);
      auto subgraph = model->Subgraph(0);
      ASSERT_TRUE(subgraph);
      std::string reason;
      auto compiled = CompilePartition(*subgraph, reason);
      ASSERT_TRUE(compiled) << reason;
      EXPECT_EQ(compiled->extensions, (std::vector<std::string>{"EXT-INT16"}));
    }
  }
}

TEST(OperationTest, CompilePartitionPopulatesTimingReport) {
  std::vector<uint8_t> bytes = AddModel({1, 4}, {1, 4}, {1, 4});
  std::unique_ptr<tflite::ModelT> unpacked = tflite::UnPackModel(bytes.data());
  tflite::SubGraphT& graph = *unpacked->subgraphs[0];
  graph.inputs = {0};
  auto rhs_buffer = std::make_unique<tflite::BufferT>();
  rhs_buffer->data = {1, 2, 3, 4};
  graph.tensors[1]->buffer = unpacked->buffers.size();
  unpacked->buffers.push_back(std::move(rhs_buffer));

  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder,
                            tflite::Model::Pack(builder, unpacked.get()));
  Expected<ExtendedModel> model = ExtendedModel::CreateFromBuffer(
      BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
  ASSERT_TRUE(model);
  Expected<Subgraph> subgraph = model->Subgraph(0);
  ASSERT_TRUE(subgraph);

  std::string reason;
  CompileTimingReport timing;
  std::shared_ptr<const CompiledPartition> compiled =
      CompilePartition(*subgraph, reason, timing);
  ASSERT_NE(compiled, nullptr) << reason;
  EXPECT_GE(timing.import_ms, 0.0);
  EXPECT_GE(timing.optimize_ms, 0.0);
  EXPECT_GE(timing.tfsc_emit_ms, 0.0);
  EXPECT_GE(timing.spirv_serialize_ms, 0.0);
  EXPECT_GT(timing.total_ms, 0.0);
  EXPECT_GT(timing.spirv_bytes, 0u);
  EXPECT_EQ(timing.spirv_bytes, compiled->spirv.size() * sizeof(uint32_t));
  EXPECT_GT(timing.constant_count, 0u);
  EXPECT_EQ(timing.constant_count, compiled->constants.size());
  size_t expected_constant_bytes = 0;
  for (const ConstantTensor& constant : compiled->constants) {
    expected_constant_bytes += constant.bytes.size();
  }
  EXPECT_GT(timing.constant_bytes, 0u);
  EXPECT_EQ(timing.constant_bytes, expected_constant_bytes);
}

template <typename ModelResult>
Subgraph FirstSubgraph(ModelResult& model) {
  auto sg = model->Subgraph(0);
  return std::move(*sg);
}

struct OpFixture {
  tflite::ModelT model;
  tflite::SubGraphT* graph = nullptr;
  tflite::OperatorT* op = nullptr;
  std::vector<uint8_t> bytes;

  explicit OpFixture(tflite::BuiltinOperator code) {
    model.version = 3;
    model.buffers.push_back(std::make_unique<tflite::BufferT>());
    model.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
    graph = model.subgraphs[0].get();
    graph->operators.push_back(std::make_unique<tflite::OperatorT>());
    op = graph->operators[0].get();
    auto opcode = std::make_unique<tflite::OperatorCodeT>();
    opcode->builtin_code = code;
    opcode->version = 2;
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
    return static_cast<int>(graph->tensors.size() - 1);
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
    graph->tensors[index]->buffer = static_cast<uint32_t>(model.buffers.size());
    model.buffers.push_back(std::move(buffer));
    return index;
  }

  int Ints(std::vector<int32_t> shape, const std::vector<int32_t>& values) {
    return Constant<int32_t>(tflite::TensorType_INT32, std::move(shape),
                             values);
  }

  auto Load() {
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
    bytes.assign(builder.GetBufferPointer(),
                 builder.GetBufferPointer() + builder.GetSize());
    return ExtendedModel::CreateFromBuffer(
        BufferRef<uint8_t>(bytes.data(), bytes.size()));
  }
};

TEST(OperationTest,
     FloatAndQuantizedConvolutionsAndDepthwiseConvolutionsCompileAndValidate) {
  // FP32 CONV_2D VALID with trailing crop + RELU6.
  {
    OpFixture f(tflite::BuiltinOperator_CONV_2D);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 5, 5, 2});
    const int w = f.Constant<float>(tflite::TensorType_FLOAT32, {3, 2, 2, 2},
                                    std::vector<float>(24, 0.25f));
    const int b = f.Constant<float>(tflite::TensorType_FLOAT32, {3},
                                    {0.1f, -0.1f, 0.2f});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {1, 2, 2, 3});
    f.op->inputs = {in, w, b};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::Conv2DOptionsT opts;
    opts.padding = tflite::Padding_VALID;
    opts.stride_h = opts.stride_w = 2;
    opts.dilation_h_factor = opts.dilation_w_factor = 1;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto sg = model->Subgraph(0);
    ASSERT_TRUE(sg);
    std::string reason;
    auto plan = AnalyzeOperation(sg->Ops()[0], reason);
    ASSERT_TRUE(plan) << reason;
    EXPECT_EQ(plan->cropped_shape, (std::vector<uint32_t>{1, 4, 4, 2}));
    EXPECT_TRUE(CompilePartition(*sg, reason)) << reason;

    // Non-finite weight rejected.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(f.model.buffers[f.graph->tensors[w]->buffer]->data.data(), &nan,
                sizeof(nan));
    auto bad = f.Load();
    ASSERT_TRUE(bad);
    EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(bad).Ops()[0], reason));
  }
  // FP32 DEPTHWISE_CONV_2D SAME + RELU + invalid multiplier check.
  {
    OpFixture f(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 4, 4, 2});
    const int w = f.Constant<float>(tflite::TensorType_FLOAT32, {1, 3, 3, 4},
                                    std::vector<float>(36, 0.1f));
    const int b = f.Constant<float>(tflite::TensorType_FLOAT32, {4},
                                    {0.0f, 0.1f, -0.1f, 0.2f});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {1, 2, 2, 4});
    f.op->inputs = {in, w, b};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::DepthwiseConv2DOptionsT opts;
    opts.padding = tflite::Padding_SAME;
    opts.stride_h = opts.stride_w = 2;
    opts.depth_multiplier = 2;
    opts.dilation_h_factor = opts.dilation_w_factor = 1;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;

    opts.depth_multiplier = 1;
    f.op->builtin_options.Set(opts);
    auto bad = f.Load();
    ASSERT_TRUE(bad);
    EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(bad).Ops()[0], reason));
  }
  // INT8 DEPTHWISE_CONV_2D (SAME + NONE and RELU6 + adapter rejections).
  for (auto act : {tflite::ActivationFunctionType_NONE,
                   tflite::ActivationFunctionType_RELU6}) {
    SCOPED_TRACE(static_cast<int>(act));
    OpFixture f(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
    const int in = f.Tensor(tflite::TensorType_INT8, {1, 5, 5, 2}, .125f, -3);
    const int w = f.Constant<int8_t>(tflite::TensorType_INT8, {1, 3, 3, 4},
                                     std::vector<int8_t>(36, 1), .25f, 0);
    const int b = f.Constant<int32_t>(tflite::TensorType_INT32, {4},
                                      {0, 2, -2, 4});
    f.graph->tensors[b]->quantization =
        std::make_unique<tflite::QuantizationParametersT>();
    f.graph->tensors[b]->quantization->scale = {.03125f};
    f.graph->tensors[b]->quantization->zero_point = {0};
    const int out = f.Tensor(tflite::TensorType_INT8, {1, 3, 3, 4}, .25f, 0);
    f.op->inputs = {in, w, b};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::DepthwiseConv2DOptionsT opts;
    opts.padding = tflite::Padding_SAME;
    opts.stride_h = opts.stride_w = 2;
    opts.depth_multiplier = 2;
    opts.dilation_h_factor = opts.dilation_w_factor = 1;
    opts.fused_activation_function = act;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;

    if (act == tflite::ActivationFunctionType_NONE) {
      for (int bad = 0; bad < 4; ++bad) {
        SCOPED_TRACE(bad);
        OpFixture bf(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
        const int bi = bf.Tensor(tflite::TensorType_INT8, {1, 5, 5, 2}, .125f, 0);
        const int bw = bf.Constant<int8_t>(
            tflite::TensorType_INT8,
            bad == 0 ? std::vector<int32_t>{2, 1, 3, 4}
                     : std::vector<int32_t>{1, 3, 3, 4},
            std::vector<int8_t>(bad == 0 ? 24 : 36, 1), .25f, bad == 1 ? 1 : 0);
        const int bb = bf.Constant<int32_t>(tflite::TensorType_INT32, {4},
                                            {0, 0, 0, 0});
        bf.graph->tensors[bb]->quantization =
            std::make_unique<tflite::QuantizationParametersT>();
        bf.graph->tensors[bb]->quantization->scale = {.03125f};
        bf.graph->tensors[bb]->quantization->zero_point = {0};
        const int bo = bf.Tensor(tflite::TensorType_INT8,
                                 bad == 2 ? std::vector<int32_t>{1, 2, 2, 4}
                                          : std::vector<int32_t>{1, 3, 3, 4},
                                 .25f, 0);
        bf.op->inputs = {bi, bw, bb};
        bf.op->outputs = {bo};
        bf.graph->inputs = {bi};
        bf.graph->outputs = {bo};
        auto bopts = opts;
        if (bad == 3) bopts.fused_activation_function = tflite::ActivationFunctionType_TANH;
        bf.op->builtin_options.Set(bopts);
        auto bm = bf.Load();
        ASSERT_TRUE(bm);
        EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(bm).Ops()[0], reason));
      }
    }
  }
}

TEST(OperationTest,
     TypedLogicalBitwiseComparisonArgMaxAndFloorDivOperationsCompile) {
  for (auto code : {tflite::BuiltinOperator_REDUCE_ALL,
                    tflite::BuiltinOperator_REDUCE_ANY}) {
    for (bool keep : {true, false}) {
      SCOPED_TRACE(static_cast<int>(code) * 10 + (keep ? 1 : 0));
      OpFixture f(code);
      const int in = f.Tensor(tflite::TensorType_BOOL, {2, 3, 4});
      const int axes = f.Ints({2}, {0, -1});
      const int out = f.Tensor(tflite::TensorType_BOOL,
                               keep ? std::vector<int32_t>{1, 3, 1}
                                    : std::vector<int32_t>{3});
      f.op->inputs = {in, axes};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      tflite::ReducerOptionsT opts;
      opts.keep_dims = keep;
      f.op->builtin_options.Set(opts);
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
  }
  {
    OpFixture f(tflite::BuiltinOperator_ARG_MAX);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 5});
    const int axis = f.Ints({1}, {1});
    const int out = f.Tensor(tflite::TensorType_INT32, {2});
    f.op->inputs = {in, axis};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::ArgMaxOptionsT opts;
    opts.output_type = tflite::TensorType_INT32;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_LOGICAL_NOT);
    const int in = f.Tensor(tflite::TensorType_BOOL, {2, 3});
    const int out = f.Tensor(tflite::TensorType_BOOL, {2, 3});
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
    OpFixture f(tflite::BuiltinOperator_LOGICAL_OR);
    const int a = f.Tensor(tflite::TensorType_BOOL, {2, 3});
    const int b = f.Tensor(tflite::TensorType_BOOL, {1, 3});
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
  for (auto code :
       {tflite::BuiltinOperator_EQUAL, tflite::BuiltinOperator_NOT_EQUAL}) {
    for (auto dt : {tflite::TensorType_BOOL, tflite::TensorType_INT32}) {
      SCOPED_TRACE(static_cast<int>(code) * 10 + static_cast<int>(dt));
      OpFixture f(code);
      const int a = f.Tensor(dt, {2, 3});
      const int b = f.Tensor(dt, {2, 3});
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
  }
  // BITWISE_XOR: runtime RHS, same-shape -1 complement, and broadcasting -1 RHS.
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    OpFixture f(tflite::BuiltinOperator_BITWISE_XOR);
    const int a = f.Tensor(
        tflite::TensorType_INT32,
        mode == 2 ? std::vector<int32_t>{1, 4} : std::vector<int32_t>{2, 4});
    const int b = mode == 0
                      ? f.Tensor(tflite::TensorType_INT32, {2, 4})
                      : f.Ints({2, 4}, std::vector<int32_t>(8, -1));
    const int out = f.Tensor(tflite::TensorType_INT32, {2, 4});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = mode == 0 ? std::vector<int32_t>{a, b}
                                : std::vector<int32_t>{a};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    auto sg = model->Subgraph(0);
    ASSERT_TRUE(sg);
    std::string reason;
    auto plan = AnalyzeOperation(sg->Ops()[0], reason);
    ASSERT_TRUE(plan) << reason;
    // Only non-broadcasting lhs uses unary BitwiseNot complement parameter.
    EXPECT_EQ(!plan->parameters.empty(), mode == 1);
    EXPECT_TRUE(CompilePartition(*sg, reason)) << reason;
  }
  // FLOOR_DIV INT32: same-shape, broadcasting lhs against constant rhs, and <= 0 rejection.
  for (bool bcast_lhs : {false, true}) {
    SCOPED_TRACE(bcast_lhs);
    OpFixture f(tflite::BuiltinOperator_FLOOR_DIV);
    const int a = f.Tensor(
        tflite::TensorType_INT32,
        bcast_lhs ? std::vector<int32_t>{1, 3} : std::vector<int32_t>{2, 3});
    const int b = f.Ints({2, 3}, {2, 3, 5, 7, 11, 13});
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
  {
    OpFixture f(tflite::BuiltinOperator_FLOOR_DIV);
    const int a = f.Tensor(tflite::TensorType_INT32, {2});
    const int b = f.Ints({2}, {2, 0});
    const int out = f.Tensor(tflite::TensorType_INT32, {2});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(model).Ops()[0], reason));
  }
  // RIGHT_SHIFT UINT32 valid + out-of-range shift rejected.
  for (uint32_t shift : {uint32_t{4}, uint32_t{32}}) {
    SCOPED_TRACE(shift);
    OpFixture f(tflite::BuiltinOperator_RIGHT_SHIFT);
    const int a = f.Tensor(tflite::TensorType_UINT32, {2, 2});
    const int b =
        f.Constant<uint32_t>(tflite::TensorType_UINT32, {1, 2}, {1u, shift});
    const int out = f.Tensor(tflite::TensorType_UINT32, {2, 2});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_EQ(static_cast<bool>(
                  AnalyzeOperation(FirstSubgraph(model).Ops()[0], reason)),
              shift <= 31u);
    if (shift <= 31u) {
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
  }
}

TEST(OperationTest,
     QuantizedAndFloatGeneralOperationsAndScalarRejectionsCompileAndValidate) {
  // 0D scalar SOFTMAX and 0D->1D RESHAPE must be rejected cleanly without UB.
  {
    OpFixture f(tflite::BuiltinOperator_SOFTMAX);
    const int in = f.Tensor(tflite::TensorType_INT8, {}, .125f, 0);
    const int out = f.Tensor(tflite::TensorType_INT8, {}, 1.0f / 256.0f, -128);
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::SoftmaxOptionsT opts;
    opts.beta = 1.0f;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(model).Ops()[0], reason));
  }
  {
    OpFixture f(tflite::BuiltinOperator_RESHAPE);
    const int in = f.Tensor(tflite::TensorType_INT8, {}, .125f, 0);
    const int shape = f.Ints({1}, {1});
    const int out = f.Tensor(tflite::TensorType_INT8, {1}, .125f, 0);
    f.op->inputs = {in, shape};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(model).Ops()[0], reason));
  }

  // Quantized SUM (INT8 & INT16) + keep_dims=false & duplicate axis rejections.
  for (auto dt : {tflite::TensorType_INT8, tflite::TensorType_INT16}) {
    SCOPED_TRACE(static_cast<int>(dt));
    OpFixture f(tflite::BuiltinOperator_SUM);
    const int in = f.Tensor(dt, {2, 3, 4}, .125f, dt == tflite::TensorType_INT8 ? -2 : 0);
    const int axes = f.Ints({2}, {1, -1});
    const int out = f.Tensor(dt, {2, 1, 1}, .25f, 0);
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

    opts.keep_dims = false;
    f.op->builtin_options.Set(opts);
    auto bad = f.Load();
    ASSERT_TRUE(bad);
    EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(bad).Ops()[0], reason));
  }
  // RESHAPE, SLICE, CONCATENATION, TRANSPOSE, PAD, PADV2 across INT8 & FP32.
  for (auto dt : {tflite::TensorType_INT8, tflite::TensorType_FLOAT32}) {
    SCOPED_TRACE(static_cast<int>(dt));
    {
      OpFixture f(tflite::BuiltinOperator_RESHAPE);
      const int in = f.Tensor(dt, {2, 6}, .125f, -3);
      const int s = f.Ints({2}, {-1, 3});
      const int out = f.Tensor(dt, {4, 3}, .125f, -3);
      f.op->inputs = {in, s};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
    {
      OpFixture f(tflite::BuiltinOperator_SLICE);
      const int in = f.Tensor(dt, {2, 5, 4}, .125f, -3);
      const int b = f.Ints({3}, {0, 1, 0});
      const int sz = f.Ints({3}, {2, -1, 3});
      const int out = f.Tensor(dt, {2, 4, 3}, .125f, -3);
      f.op->inputs = {in, b, sz};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
    {
      OpFixture f(tflite::BuiltinOperator_CONCATENATION);
      const int a = f.Tensor(dt, {2, 3}, .125f, -3);
      const int b = f.Tensor(dt, {2, 5}, .125f, -3);
      const int out = f.Tensor(dt, {2, 8}, .125f, -3);
      f.op->inputs = {a, b};
      f.op->outputs = {out};
      f.graph->inputs = {a, b};
      f.graph->outputs = {out};
      tflite::ConcatenationOptionsT opts;
      opts.axis = -1;
      opts.fused_activation_function = tflite::ActivationFunctionType_NONE;
      f.op->builtin_options.Set(opts);
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
    {
      OpFixture f(tflite::BuiltinOperator_TRANSPOSE);
      const int in = f.Tensor(dt, {2, 3, 4}, .125f, -3);
      const int p = f.Ints({3}, {0, 2, 1});
      const int out = f.Tensor(dt, {2, 4, 3}, .125f, -3);
      f.op->inputs = {in, p};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
    {
      OpFixture f(tflite::BuiltinOperator_PAD);
      const int in = f.Tensor(dt, {2, 3}, .125f, -3);
      const int p = f.Ints({2, 2}, {1, 1, 0, 2});
      const int out = f.Tensor(dt, {4, 5}, .125f, -3);
      f.op->inputs = {in, p};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
    {
      OpFixture f(tflite::BuiltinOperator_PADV2);
      const int in = f.Tensor(dt, {2, 3}, .125f, -3);
      const int p = f.Ints({2, 2}, {1, 0, 0, 1});
      const int v = dt == tflite::TensorType_INT8
                        ? f.Constant<int8_t>(dt, {}, {5}, .125f, -3)
                        : f.Constant<float>(dt, {}, {2.5f});
      const int out = f.Tensor(dt, {3, 4}, .125f, -3);
      f.op->inputs = {in, p, v};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
  }
  // INT16 PADV2, QUANTIZE/DEQUANTIZE FP32<->INT8, INT8 ABS/TANH/HARD_SWISH/MUL/SUB, FP32 HARD_SWISH/ADD/MUL/AVG_POOL.
  {
    OpFixture f(tflite::BuiltinOperator_PADV2);
    const int in = f.Tensor(tflite::TensorType_INT16, {2, 3}, .125f, 0);
    const int p = f.Ints({2, 2}, {0, 1, 1, 0});
    const int v =
        f.Constant<int16_t>(tflite::TensorType_INT16, {}, {12}, .125f, 0);
    const int out = f.Tensor(tflite::TensorType_INT16, {3, 4}, .125f, 0);
    f.op->inputs = {in, p, v};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  for (bool dequant : {false, true}) {
    SCOPED_TRACE(dequant);
    OpFixture f(dequant ? tflite::BuiltinOperator_DEQUANTIZE
                        : tflite::BuiltinOperator_QUANTIZE);
    const int in = f.Tensor(dequant ? tflite::TensorType_INT8
                                    : tflite::TensorType_FLOAT32,
                            {2, 4}, .125f, -5);
    const int out = f.Tensor(dequant ? tflite::TensorType_FLOAT32
                                     : tflite::TensorType_INT8,
                             {2, 4}, .125f, -5);
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  for (auto code : {tflite::BuiltinOperator_ABS, tflite::BuiltinOperator_TANH,
                    tflite::BuiltinOperator_HARD_SWISH}) {
    SCOPED_TRACE(static_cast<int>(code));
    OpFixture f(code);
    const int in = f.Tensor(tflite::TensorType_INT8, {2, 8}, .125f, -4);
    const int out = f.Tensor(
        tflite::TensorType_INT8, {2, 8},
        code == tflite::BuiltinOperator_TANH ? 1.0f / 128.0f : .125f, 0);
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
    OpFixture f(tflite::BuiltinOperator_HARD_SWISH);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 8});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 8});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  for (auto code :
       {tflite::BuiltinOperator_MUL, tflite::BuiltinOperator_SUB}) {
    SCOPED_TRACE(static_cast<int>(code));
    OpFixture f(code);
    const int a = f.Tensor(tflite::TensorType_INT8, {2, 4}, .125f, -2);
    const int b = f.Tensor(tflite::TensorType_INT8, {1, 4}, .125f, 1);
    const int out = f.Tensor(tflite::TensorType_INT8, {2, 4}, .25f, 0);
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a, b};
    f.graph->outputs = {out};
    if (code == tflite::BuiltinOperator_MUL) {
      tflite::MulOptionsT opts;
      opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
      f.op->builtin_options.Set(opts);
    } else {
      tflite::SubOptionsT opts;
      opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
      f.op->builtin_options.Set(opts);
    }
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  // FP32 ADD with 1D constant rank-broadcasting RHS + RELU6, and FP32 global AVG_POOL_2D.
  {
    OpFixture f(tflite::BuiltinOperator_ADD);
    const int a = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 4});
    const int b = f.Constant<float>(tflite::TensorType_FLOAT32, {4},
                                    {1.0f, 2.0f, 3.0f, 4.0f});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 4});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a};
    f.graph->outputs = {out};
    tflite::AddOptionsT opts;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_AVERAGE_POOL_2D);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 3, 3, 2});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {1, 1, 1, 2});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::Pool2DOptionsT opts;
    opts.padding = tflite::Padding_VALID;
    opts.filter_height = opts.filter_width = 3;
    opts.stride_h = opts.stride_w = 1;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
}

TEST(OperationTest, ExtendedTransposeConvScatterNdReduceAndResizeOperations) {
  // INT8 4-input per-channel SAME RELU6 & 3-input VALID NONE, plus FP32 4-input RELU6.
  {
    OpFixture f(tflite::BuiltinOperator_TRANSPOSE_CONV);
    const int shape = f.Ints({4}, {1, 4, 4, 2});
    const int w = f.Constant<int8_t>(tflite::TensorType_INT8, {2, 3, 3, 2},
                                     std::vector<int8_t>(36, 1));
    f.graph->tensors[w]->quantization->quantized_dimension = 0;
    f.graph->tensors[w]->quantization->scale = {.25f, .5f};
    f.graph->tensors[w]->quantization->zero_point = {0, 0};
    const int in = f.Tensor(tflite::TensorType_INT8, {1, 2, 2, 2}, .125f, -2);
    const int b = f.Constant<int32_t>(tflite::TensorType_INT32, {2}, {4, -4});
    f.graph->tensors[b]->quantization =
        std::make_unique<tflite::QuantizationParametersT>();
    f.graph->tensors[b]->quantization->quantized_dimension = 0;
    f.graph->tensors[b]->quantization->scale = {.03125f, .0625f};
    f.graph->tensors[b]->quantization->zero_point = {0, 0};
    const int out = f.Tensor(tflite::TensorType_INT8, {1, 4, 4, 2}, .25f, 0);
    f.op->inputs = {shape, w, in, b};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::TransposeConvOptionsT opts;
    opts.padding = tflite::Padding_SAME;
    opts.stride_h = opts.stride_w = 2;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_TRANSPOSE_CONV);
    const int shape = f.Ints({4}, {1, 5, 5, 2});
    const int w = f.Constant<int8_t>(tflite::TensorType_INT8, {2, 3, 3, 2},
                                     std::vector<int8_t>(36, 1), .25f, 0);
    const int in = f.Tensor(tflite::TensorType_INT8, {1, 2, 2, 2}, .125f, 0);
    const int out = f.Tensor(tflite::TensorType_INT8, {1, 5, 5, 2}, .25f, 0);
    f.op->inputs = {shape, w, in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::TransposeConvOptionsT opts;
    opts.padding = tflite::Padding_VALID;
    opts.stride_h = opts.stride_w = 2;
    opts.fused_activation_function = tflite::ActivationFunctionType_NONE;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_TRANSPOSE_CONV);
    const int shape = f.Ints({4}, {1, 4, 4, 2});
    const int w = f.Constant<float>(tflite::TensorType_FLOAT32, {2, 3, 3, 2},
                                    std::vector<float>(36, 0.1f));
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 2, 2, 2});
    const int b = f.Constant<float>(tflite::TensorType_FLOAT32, {2}, {0.1f, -0.2f});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {1, 4, 4, 2});
    f.op->inputs = {shape, w, in, b};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::TransposeConvOptionsT opts;
    opts.padding = tflite::Padding_SAME;
    opts.stride_h = opts.stride_w = 2;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;

    opts.stride_h = 4;
    f.op->builtin_options.Set(opts);
    auto bad = f.Load();
    ASSERT_TRUE(bad);
    EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(bad).Ops()[0], reason));
  }

  // SCATTER_ND FP32 valid + duplicate / OOB index rejections.
  for (int variant = 0; variant < 3; ++variant) {
    SCOPED_TRACE(variant);
    OpFixture f(tflite::BuiltinOperator_SCATTER_ND);
    const int idx = f.Ints({2, 1}, variant == 1   ? std::vector<int32_t>{1, 1}
                                   : variant == 2 ? std::vector<int32_t>{4, 0}
                                                  : std::vector<int32_t>{0, 2});
    const int updates = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    const int shape = f.Ints({2}, {4, 4});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {4, 4});
    f.op->inputs = {idx, updates, shape};
    f.op->outputs = {out};
    f.graph->inputs = {updates};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    if (variant == 0) {
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    } else {
      EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(model).Ops()[0], reason));
    }
  }

  // FP32 REDUCE_MAX keep_dims=true and RESIZE_BILINEAR aligned singleton / extreme scale rejections.
  {
    OpFixture f(tflite::BuiltinOperator_REDUCE_MAX);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 4});
    const int axes = f.Ints({1}, {1});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 1, 4});
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
    OpFixture f(tflite::BuiltinOperator_RESIZE_BILINEAR);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 1, 4, 2});
    const int sz = f.Ints({2}, {4, 4});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {1, 4, 4, 2});
    f.op->inputs = {in, sz};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::ResizeBilinearOptionsT opts;
    opts.align_corners = true;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(model).Ops()[0], reason));
  }
}

TEST(OperationTest,
     PortableConv3dStridedSliceBatchMatMulSplitUnpackAndFloatOpsCompile) {
  // FP32 CONV_3D VALID cropped & SAME + RELU6.
  for (bool same : {false, true}) {
    SCOPED_TRACE(same);
    OpFixture f(tflite::BuiltinOperator_CONV_3D);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 5, 5, 5, 2});
    const int w = f.Constant<float>(tflite::TensorType_FLOAT32, {2, 2, 2, 2, 3},
                                    std::vector<float>(48, 0.1f));
    const int b =
        f.Constant<float>(tflite::TensorType_FLOAT32, {3}, {0.1f, 0.0f, -0.1f});
    const int out = f.Tensor(
        tflite::TensorType_FLOAT32,
        same ? std::vector<int32_t>{1, 3, 3, 3, 3}
             : std::vector<int32_t>{1, 2, 2, 2, 3});
    f.op->inputs = {in, w, b};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::Conv3DOptionsT opts;
    opts.padding = same ? tflite::Padding_SAME : tflite::Padding_VALID;
    opts.stride_d = opts.stride_h = opts.stride_w = 2;
    opts.dilation_d_factor = opts.dilation_h_factor = opts.dilation_w_factor = 1;
    opts.fused_activation_function = same
                                         ? tflite::ActivationFunctionType_RELU6
                                         : tflite::ActivationFunctionType_RELU;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }

  // STRIDED_SLICE (FP32 & INT8) with masks and shrink axis.
  for (auto dt : {tflite::TensorType_FLOAT32, tflite::TensorType_INT8}) {
    SCOPED_TRACE(static_cast<int>(dt));
    OpFixture f(tflite::BuiltinOperator_STRIDED_SLICE);
    const int in = f.Tensor(dt, {2, 6, 4}, .125f, -2);
    const int b = f.Ints({3}, {0, -5, 0});
    const int e = f.Ints({3}, {2, 5, 4});
    const int s = f.Ints({3}, {1, 2, 1});
    const int out = f.Tensor(dt, {2, 4}, .125f, -2);
    f.op->inputs = {in, b, e, s};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::StridedSliceOptionsT opts;
    opts.begin_mask = 1;
    opts.end_mask = 4;
    opts.shrink_axis_mask = 1;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }

  // FP32 BATCH_MATMUL (adj_x, adj_y, batch broadcast), MEAN, REDUCE_PROD, SOFTMAX.
  {
    OpFixture f(tflite::BuiltinOperator_BATCH_MATMUL);
    const int a = f.Tensor(tflite::TensorType_FLOAT32, {1, 3, 2});
    const int b = f.Tensor(tflite::TensorType_FLOAT32, {2, 4, 3});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 2, 4});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a, b};
    f.graph->outputs = {out};
    tflite::BatchMatMulOptionsT opts;
    opts.adj_x = true;
    opts.adj_y = true;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  for (auto code : {tflite::BuiltinOperator_MEAN,
                    tflite::BuiltinOperator_REDUCE_PROD}) {
    for (bool keep : {false, true}) {
      SCOPED_TRACE(static_cast<int>(code) * 10 + (keep ? 1 : 0));
      OpFixture f(code);
      const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 4});
      const int axes = f.Ints({1}, {-1});
      const int out = f.Tensor(tflite::TensorType_FLOAT32,
                               keep ? std::vector<int32_t>{2, 3, 1}
                                    : std::vector<int32_t>{2, 3});
      f.op->inputs = {in, axes};
      f.op->outputs = {out};
      f.graph->inputs = {in};
      f.graph->outputs = {out};
      tflite::ReducerOptionsT opts;
      opts.keep_dims = keep;
      f.op->builtin_options.Set(opts);
      auto model = f.Load();
      ASSERT_TRUE(model);
      std::string reason;
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    }
  }
  for (float beta : {1.0f, 0.5f, 0.0f}) {
    SCOPED_TRACE(beta);
    OpFixture f(tflite::BuiltinOperator_SOFTMAX);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 5});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 5});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::SoftmaxOptionsT opts;
    opts.beta = beta;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    if (beta > 0.0f) {
      EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
    } else {
      EXPECT_FALSE(AnalyzeOperation(FirstSubgraph(model).Ops()[0], reason));
    }
  }

  // SQUEEZE, EXPAND_DIMS, TILE, REVERSE_V2, PACK, UNPACK, SPLIT, SPLIT_V.
  {
    OpFixture f(tflite::BuiltinOperator_SQUEEZE);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 3, 1, 4});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {3, 4});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    f.op->builtin_options.Set(tflite::SqueezeOptionsT{});
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_EXPAND_DIMS);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int ax = f.Ints({}, {-1});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 1});
    f.op->inputs = {in, ax};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_TILE);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int m = f.Ints({2}, {2, 1});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {4, 3});
    f.op->inputs = {in, m};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_REVERSE_V2);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 4});
    const int ax = f.Ints({1}, {-1});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3, 4});
    f.op->inputs = {in, ax};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_PACK);
    const int a = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int b = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 2, 3});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a, b};
    f.graph->outputs = {out};
    tflite::PackOptionsT opts;
    opts.values_count = 2;
    opts.axis = 1;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_UNPACK);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int o0 = f.Tensor(tflite::TensorType_FLOAT32, {3});
    const int o1 = f.Tensor(tflite::TensorType_FLOAT32, {3});
    f.op->inputs = {in};
    f.op->outputs = {o0, o1};
    f.graph->inputs = {in};
    f.graph->outputs = {o0, o1};
    tflite::UnpackOptionsT opts;
    opts.num = 2;
    opts.axis = 0;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_SPLIT);
    const int ax = f.Ints({}, {1});
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    const int o0 = f.Tensor(tflite::TensorType_FLOAT32, {2, 2});
    const int o1 = f.Tensor(tflite::TensorType_FLOAT32, {2, 2});
    f.op->inputs = {ax, in};
    f.op->outputs = {o0, o1};
    f.graph->inputs = {in};
    f.graph->outputs = {o0, o1};
    tflite::SplitOptionsT opts;
    opts.num_splits = 2;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_SPLIT_V);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 5});
    const int sz = f.Ints({2}, {2, -1});
    const int ax = f.Ints({}, {1});
    const int o0 = f.Tensor(tflite::TensorType_FLOAT32, {2, 2});
    const int o1 = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    f.op->inputs = {in, sz, ax};
    f.op->outputs = {o0, o1};
    f.graph->inputs = {in};
    f.graph->outputs = {o0, o1};
    tflite::SplitVOptionsT opts;
    opts.num_splits = 2;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }

  // GELU (rank-5 exact + approximate), POW/SUB/DIV/MAX/MIN, cropped MAX_POOL_2D, RELU/RELU6, and 12 FP32 unaries.
  for (bool approx : {false, true}) {
    SCOPED_TRACE(approx);
    OpFixture f(tflite::BuiltinOperator_GELU);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 1, 1, 2, 2});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {1, 1, 1, 2, 2});
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
  for (auto code :
       {tflite::BuiltinOperator_POW, tflite::BuiltinOperator_SUB,
        tflite::BuiltinOperator_DIV, tflite::BuiltinOperator_MAXIMUM,
        tflite::BuiltinOperator_MINIMUM}) {
    SCOPED_TRACE(static_cast<int>(code));
    OpFixture f(code);
    const int a = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    const int b = f.Tensor(tflite::TensorType_FLOAT32, {1, 3});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 3});
    f.op->inputs = {a, b};
    f.op->outputs = {out};
    f.graph->inputs = {a, b};
    f.graph->outputs = {out};
    if (code == tflite::BuiltinOperator_SUB) {
      tflite::SubOptionsT opts;
      opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
      f.op->builtin_options.Set(opts);
    } else if (code == tflite::BuiltinOperator_DIV) {
      tflite::DivOptionsT opts;
      opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
      f.op->builtin_options.Set(opts);
    }
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  {
    OpFixture f(tflite::BuiltinOperator_MAX_POOL_2D);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {1, 5, 5, 2});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {1, 2, 2, 2});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    tflite::Pool2DOptionsT opts;
    opts.padding = tflite::Padding_VALID;
    opts.filter_height = opts.filter_width = 2;
    opts.stride_h = opts.stride_w = 2;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    f.op->builtin_options.Set(opts);
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
  for (auto code :
       {tflite::BuiltinOperator_RELU, tflite::BuiltinOperator_RELU6,
        tflite::BuiltinOperator_SIN, tflite::BuiltinOperator_COS,
        tflite::BuiltinOperator_ABS, tflite::BuiltinOperator_RSQRT,
        tflite::BuiltinOperator_EXP, tflite::BuiltinOperator_LOG,
        tflite::BuiltinOperator_FLOOR, tflite::BuiltinOperator_CEIL,
        tflite::BuiltinOperator_TANH, tflite::BuiltinOperator_NEG,
        tflite::BuiltinOperator_SQRT, tflite::BuiltinOperator_SQUARE}) {
    SCOPED_TRACE(static_cast<int>(code));
    OpFixture f(code);
    const int in = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    const int out = f.Tensor(tflite::TensorType_FLOAT32, {2, 4});
    f.op->inputs = {in};
    f.op->outputs = {out};
    f.graph->inputs = {in};
    f.graph->outputs = {out};
    auto model = f.Load();
    ASSERT_TRUE(model);
    std::string reason;
    EXPECT_TRUE(CompilePartition(FirstSubgraph(model), reason)) << reason;
  }
}

TEST(OperationTest,
     MultiOpSubgraphImportOptimizeReportAndCompositeDumpCoverage) {
  // Build CONV_2D (NONE) -> RELU6 -> SOFTMAX subgraph with OptimizationReport
  // and LoweringRecord diagnostics enabled, plus rescale_multiplier_bits=24 rejection.
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  for (auto code : {tflite::BuiltinOperator_CONV_2D,
                    tflite::BuiltinOperator_RELU6,
                    tflite::BuiltinOperator_SOFTMAX}) {
    auto opcode = std::make_unique<tflite::OperatorCodeT>();
    opcode->builtin_code = code;
    opcode->version = 2;
    model.operator_codes.push_back(std::move(opcode));
  }
  auto g = std::make_unique<tflite::SubGraphT>();
  auto add_q = [&](std::vector<int32_t> shape, float scale, int64_t zp) {
    auto t = std::make_unique<tflite::TensorT>();
    t->type = tflite::TensorType_INT8;
    t->shape = std::move(shape);
    t->quantization = std::make_unique<tflite::QuantizationParametersT>();
    t->quantization->scale = {scale};
    t->quantization->zero_point = {zp};
    g->tensors.push_back(std::move(t));
    return static_cast<int>(g->tensors.size() - 1);
  };
  const int x = add_q({1, 2, 2, 2}, .125f, 0);
  const int w = add_q({2, 1, 1, 2}, .25f, 0);
  {
    auto wb = std::make_unique<tflite::BufferT>();
    wb->data = {1, 1, 1, 1};
    g->tensors[w]->buffer = static_cast<uint32_t>(model.buffers.size());
    model.buffers.push_back(std::move(wb));
  }
  auto bt = std::make_unique<tflite::TensorT>();
  bt->type = tflite::TensorType_INT32;
  bt->shape = {2};
  bt->quantization = std::make_unique<tflite::QuantizationParametersT>();
  bt->quantization->scale = {.03125f};
  bt->quantization->zero_point = {0};
  {
    auto bb = std::make_unique<tflite::BufferT>();
    const int32_t zeros[2] = {0, 0};
    bb->data.resize(sizeof(zeros));
    std::memcpy(bb->data.data(), zeros, sizeof(zeros));
    bt->buffer = static_cast<uint32_t>(model.buffers.size());
    model.buffers.push_back(std::move(bb));
  }
  const int b = static_cast<int>(g->tensors.size());
  g->tensors.push_back(std::move(bt));
  const int c_out = add_q({1, 2, 2, 2}, .25f, 0);
  const int r_out = add_q({1, 2, 2, 2}, .25f, 0);
  const int s_out = add_q({1, 2, 2, 2}, 1.0f / 256.0f, -128);

  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 0;
    op->inputs = {x, w, b};
    op->outputs = {c_out};
    tflite::Conv2DOptionsT opts;
    opts.padding = tflite::Padding_VALID;
    opts.stride_h = opts.stride_w = 1;
    opts.dilation_h_factor = opts.dilation_w_factor = 1;
    opts.fused_activation_function = tflite::ActivationFunctionType_NONE;
    op->builtin_options.Set(opts);
    g->operators.push_back(std::move(op));
  }
  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 1;
    op->inputs = {c_out};
    op->outputs = {r_out};
    g->operators.push_back(std::move(op));
  }
  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 2;
    op->inputs = {r_out};
    op->outputs = {s_out};
    tflite::SoftmaxOptionsT opts;
    opts.beta = 1.0f;
    op->builtin_options.Set(opts);
    g->operators.push_back(std::move(op));
  }
  g->inputs = {x};
  g->outputs = {s_out};
  model.subgraphs.push_back(std::move(g));

  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  auto loaded = ExtendedModel::CreateFromBuffer(
      BufferRef<uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
  ASSERT_TRUE(loaded);
  auto sg = loaded->Subgraph(0);
  ASSERT_TRUE(sg);

  std::string reason;
  EXPECT_EQ(CompilePartition(*sg, reason, nullptr, nullptr,
                             SoftmaxLutType::kAuto, 24),
            nullptr);
  EXPECT_NE(reason.find("Rescale multiplier bits"), std::string::npos);

  std::vector<LoweringRecord> records;
  OptimizationReport report;
  auto compiled = CompilePartition(*sg, reason, &records, &report);
  ASSERT_NE(compiled, nullptr) << reason;
  EXPECT_FALSE(records.empty());
  EXPECT_NE(report.before_ir.find("quantized_convolution"), std::string::npos);
  EXPECT_NE(report.after_ir.find("quantized_softmax"), std::string::npos);
}

}  // namespace
}  // namespace litert::arm_vulkan_ml
