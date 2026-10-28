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

#include "litert/vendors/arm_vulkan_ml/tools/devtool/tool.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <string>

#include "gtest/gtest.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"

namespace litert::arm_vulkan_ml::tool {
namespace {

TEST(Options, RejectsMalformedAndNonfiniteValues) {
  char command[] = "tool";
  char inspect[] = "inspect";
  char flag[] = "--model";
  char* argv[] = {command, inspect, flag};
  EXPECT_FALSE(ParseOptions(3, argv));
  Options options;
  options.values["atol"] = "nan";
  EXPECT_FALSE(options.Number("atol", 0));
  options.values["atol"] = "-1";
  EXPECT_FALSE(options.Number("atol", 0));
  options.values["max_samples"] = "3.1";
  EXPECT_FALSE(options.Integer("max_samples", 3));
}

TEST(Comparison, FloatToleranceAndNonfinite) {
  const float reference[] = {1, 2};
  float candidate[] = {1.0001f, 2};
  Json result;
  ASSERT_TRUE(Compare(candidate, reference, 2, true, {}, result));
  EXPECT_TRUE(result["passed"].get<bool>());
  candidate[0] = 1.1f;
  ASSERT_TRUE(Compare(candidate, reference, 2, true, {}, result));
  EXPECT_FALSE(result["passed"].get<bool>());
  candidate[0] = std::numeric_limits<float>::quiet_NaN();
  ASSERT_TRUE(Compare(candidate, reference, 2, true, {}, result));
  EXPECT_FALSE(result["passed"].get<bool>());
  EXPECT_FALSE(result["finite"].get<bool>());
}

TEST(Comparison, Int8RequiresBothMaximumAndMeanGate) {
  const int8_t reference[] = {0, 0, 0, 0};
  int8_t candidate[] = {2, 0, 0, 0};
  Json result;
  ASSERT_TRUE(Compare(candidate, reference, 4, false, {}, result));
  EXPECT_TRUE(result["passed"].get<bool>());
  candidate[1] = 1;
  ASSERT_TRUE(Compare(candidate, reference, 4, false, {}, result));
  EXPECT_FALSE(result["passed"].get<bool>());
  Options options;
  options.values["mean_code_error"] = "10";
  candidate[0] = 5;
  ASSERT_TRUE(Compare(candidate, reference, 4, false, options, result));
  EXPECT_FALSE(result["passed"].get<bool>());
}

TEST(Capabilities, RequiresRevisionTwoAndEnabledFeatures) {
  DeviceCapabilities device;
  for (const auto* name : {VK_ARM_TENSORS_EXTENSION_NAME,
                          VK_ARM_DATA_GRAPH_EXTENSION_NAME,
                          VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,
                          VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
                          VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                          VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME}) {
    VkExtensionProperties extension{};
    std::strncpy(extension.extensionName, name, sizeof(extension.extensionName) - 1);
    extension.specVersion = 2;
    device.extensions.push_back(extension);
  }
  std::string reason;
  EXPECT_FALSE(CheckDeviceRequirements(device, reason));
  device.tensors_feature = device.graph_feature = device.synchronization2 = true;
  device.maintenance5 = device.replicated_composites = true;
  EXPECT_TRUE(CheckDeviceRequirements(device, reason));
  device.extensions[0].specVersion = 1;
  EXPECT_FALSE(CheckDeviceRequirements(device, reason));
  EXPECT_NE(reason.find(VK_ARM_TENSORS_EXTENSION_NAME), std::string::npos);
}

TEST(Capabilities, EmulationOptInNeverBypassesExtensions) {
  DeviceCapabilities device;
  device.properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_CPU;
  GraphEngine engine{};
  engine.instruction_set = "TOSA.001000.1";
  engine.profiles.push_back({"Emulation Layer", VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM});
  CompiledPartition artifact;
  artifact.instruction_set = engine.instruction_set;
  artifact.profiles = {"PRO-INT"};
  std::string reason;
  EXPECT_FALSE(CheckEngineRequirements(artifact, device, engine, false, reason));
  EXPECT_TRUE(CheckEngineRequirements(artifact, device, engine, true, reason));
  artifact.extensions = {"int16"};
  EXPECT_FALSE(CheckEngineRequirements(artifact, device, engine, true, reason));
  engine.extensions.push_back({"int16", 0});
  EXPECT_FALSE(CheckEngineRequirements(artifact, device, engine, true, reason));
  engine.extensions[0].quality = VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM;
  EXPECT_TRUE(CheckEngineRequirements(artifact, device, engine, true, reason));
}

// Two legal quantized RELUs separated by an unsupported SIN exercise partial
// coverage and cloning of original operator IDs during partitioning.
TEST(Inspection, PreservesOriginsAcrossPartitionCloning) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  for (auto code : {tflite::BuiltinOperator_RELU, tflite::BuiltinOperator_SIN}) {
    auto opcode = std::make_unique<tflite::OperatorCodeT>();
    opcode->builtin_code = code;
    opcode->deprecated_builtin_code = static_cast<int8_t>(code);
    model.operator_codes.push_back(std::move(opcode));
  }
  auto graph = std::make_unique<tflite::SubGraphT>();
  for (int i = 0; i < 4; ++i) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->shape = {1, 2};
    tensor->type = tflite::TensorType_INT8;
    tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
    tensor->quantization->scale = {.1f};
    tensor->quantization->zero_point = {0};
    graph->tensors.push_back(std::move(tensor));
  }
  graph->inputs = {0};
  graph->outputs = {3};
  for (int i = 0; i < 3; ++i) {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = i == 1 ? 1 : 0;
    op->inputs = {i};
    op->outputs = {i + 1};
    graph->operators.push_back(std::move(op));
  }
  model.subgraphs.push_back(std::move(graph));
  auto bytes = PackModel(model);
  ASSERT_TRUE(bytes);
  const std::string path = testing::TempDir() + "/coverage.tflite";
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  file.close();
  Options options;
  options.values["model"] = path;
  Json report;
  ASSERT_TRUE(Inspect(options, report, false));
  EXPECT_FALSE(report["complete_coverage"].get<bool>());
  const auto& partitions = report["subgraphs"][0]["partitions"];
  ASSERT_EQ(partitions.size(), 2);
  const std::set<int> origins = {partitions[0]["operators"][0].get<int>(),
                                 partitions[1]["operators"][0].get<int>()};
  EXPECT_EQ(origins, (std::set<int>{0, 2}));
  EXPECT_FALSE(report["subgraphs"][0]["operators"][1]["reason"].get<std::string>().empty());

  model.subgraphs[0]->operators[1]->opcode_index = 0;
  bytes = PackModel(model);
  ASSERT_TRUE(bytes);
  file.open(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  file.close();
  options.values["output_dir"] = testing::TempDir() + "/export";
  report = Json::object();
  ASSERT_TRUE(Inspect(options, report, true));
  EXPECT_TRUE(report["complete_coverage"].get<bool>());
  const auto& compiled = report["subgraphs"][0]["partitions"][0];
  EXPECT_TRUE(compiled["spirv_valid"].get<bool>());
  ASSERT_EQ(compiled["lowering"].size(), 3);
  EXPECT_EQ(compiled["lowering"][2]["operator"], 2);
  EXPECT_FALSE(compiled["lowering"][2]["created_result_ids"].empty());
}

TEST(ModelLoading, MaterializesAndBoundsChecksExternalConstants) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  model.buffers[0]->offset = 4096;
  model.buffers[0]->size = 4;
  model.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
  auto bytes = PackModel(model);
  ASSERT_TRUE(bytes);
  ASSERT_LT(bytes->size(), 4096);
  bytes->resize(4100);
  (*bytes)[4096] = 42;
  const std::string path = testing::TempDir() + "/external.tflite";
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  file.close();
  auto loaded = LoadModel(path);
  ASSERT_TRUE(loaded);
  EXPECT_EQ((*loaded)->buffers[0]->offset, 0);
  EXPECT_EQ((*loaded)->buffers[0]->data, (std::vector<uint8_t>{42, 0, 0, 0}));
  file.open(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes->data()), 4097);
  file.close();
  EXPECT_FALSE(LoadModel(path));
}

}  // namespace
}  // namespace litert::arm_vulkan_ml::tool
