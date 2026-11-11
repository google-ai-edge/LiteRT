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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/vendors/arm_vulkan_ml/compiler/softmax.h"
#include "litert/c/litert_op_code.h"
#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"
#include "tflite/schema/schema_generated.h"
#include "vulkan/vulkan_core.h"  // from @arm_vulkan_ml_dep_vulkan_headers

namespace litert::arm_vulkan_ml::tool {
namespace {

TEST(Hashing, ComputesExpectedSha256Digests) {
  const auto empty_path = testing::TempDir() + "/empty.bin";
  {
    std::ofstream file(empty_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good());
  }
  auto empty_hash = HashFile(empty_path);
  ASSERT_TRUE(empty_hash);
  EXPECT_EQ(*empty_hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

  const auto abc_path = testing::TempDir() + "/abc.bin";
  {
    std::ofstream file(abc_path, std::ios::binary | std::ios::trunc);
    file.write("abc", 3);
    ASSERT_TRUE(file.good());
  }
  auto abc_hash = HashFile(abc_path);
  ASSERT_TRUE(abc_hash);
  EXPECT_EQ(*abc_hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

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

TEST(Options, RescalePrecisionAndGraphViewsAreValidated) {
  for (const std::string& bits : {"16", "32", "8", "bad"}) {
    std::vector<std::string> args = {"tool", "compile",
                                     "--rescale_multiplier_bits=" + bits};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    EXPECT_EQ(static_cast<bool>(
                  ParseOptions(static_cast<int>(argv.size()), argv.data())),
              bits == "16" || bits == "32");
  }
  for (const std::string& view : {"raw", "effective", "bad"}) {
    std::vector<std::string> args = {"tool", "inspect", "--graph_view=" + view};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    EXPECT_EQ(static_cast<bool>(
                  ParseOptions(static_cast<int>(argv.size()), argv.data())),
              view != "bad");
  }
}

TEST(Options, SoftmaxLutAndAlternateReferencesAreValidated) {
  auto defaults = Options{}.SoftmaxLut();
  ASSERT_TRUE(defaults);
  EXPECT_EQ(*defaults, SoftmaxLutType::kAuto);
  for (const std::string& lut : {"auto", "int8", "int16", "invalid"}) {
    std::vector<std::string> args = {"tool", "compile",
                                     "--softmax_lut_type=" + lut};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    auto parsed = ParseOptions(static_cast<int>(argv.size()), argv.data());
    EXPECT_EQ(static_cast<bool>(parsed), lut != "invalid");
    if (parsed) {
      auto selected = parsed->SoftmaxLut();
      ASSERT_TRUE(selected);
      EXPECT_EQ(SoftmaxLutTypeName(*selected), lut);
    }
  }
  for (const std::string& command :
       {"validate", "compile", "compare", "trace"}) {
    std::vector<std::string> args = {"tool", command, "--reference=fp64"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    EXPECT_EQ(static_cast<bool>(
                  ParseOptions(static_cast<int>(argv.size()), argv.data())),
              command == "validate");
  }
}

TEST(SoftmaxReference, MathematicalQuantizationAndUnsignedRebasing) {
  for (size_t channels :
       {size_t{1}, size_t{2}, size_t{4}, size_t{512}, size_t{1000}}) {
    std::vector<int8_t> input(channels * 2, -17);
    auto fp64 = SoftmaxReference(input, channels, .125, 1, "fp64");
    auto xnn = SoftmaxReference(input, channels, .125, 1, "xnnpack");
    ASSERT_TRUE(fp64);
    ASSERT_TRUE(xnn);
    EXPECT_EQ(*fp64, *xnn);
    const int expected =
        std::min(255, static_cast<int>(std::floor(256.0 / channels + .5))) -
        128;
    EXPECT_EQ((*fp64)[0], expected);
  }
  std::vector<int8_t> input{-128, -37, 0, 42, 127};
  auto fp64 = SoftmaxReference(input, 5, .125, .7, "fp64");
  auto xnn = SoftmaxReference(input, 5, .125, .7, "xnnpack");
  ASSERT_TRUE(fp64);
  ASSERT_TRUE(xnn);
  for (size_t i = 0; i < input.size(); ++i)
    EXPECT_LE(std::abs((*fp64)[i] - (*xnn)[i]), 1);
}
TEST(SoftmaxReference,
     DetectsRoundedXnnpackSumOverflowInsteadOfUsingWrappedResults) {
  for (size_t channels : {size_t{1001}, size_t{1024}, size_t{2048}}) {
    std::vector<int8_t> input(channels, 0);
    EXPECT_TRUE(SoftmaxReference(input, channels, .125, 1, "fp64"));
    auto xnn = SoftmaxReference(input, channels, .125, 1, "xnnpack");
    EXPECT_FALSE(xnn);
    if (!xnn)
      EXPECT_NE(std::string(xnn.Error().Message()).find("exceeds UINT32"),
                std::string::npos);
  }
}

TEST(QuantizedReference,
     SquareAndPerChannelConv3dHaveIndependentExpectedValues) {
  tflite::ModelT model;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  model.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
  model.operator_codes.push_back(std::make_unique<tflite::OperatorCodeT>());
  auto& graph = *model.subgraphs[0];
  graph.operators.push_back(std::make_unique<tflite::OperatorT>());
  graph.inputs = {0};
  graph.outputs = {1};
  auto& operation = *graph.operators[0];
  operation.inputs = {0};
  operation.outputs = {1};
  for (int i = 0; i < 2; ++i) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->type = tflite::TensorType_INT8;
    tensor->shape = {4};
    tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
    tensor->quantization->scale = {i == 0 ? .5f : .25f};
    tensor->quantization->zero_point = {i == 0 ? 0 : -3};
    graph.tensors.push_back(std::move(tensor));
  }
  model.operator_codes[0]->builtin_code = tflite::BuiltinOperator_SQUARE;
  auto square = QuantizedReference(model, {-3, -1, 2, 127});
  ASSERT_TRUE(square);
  EXPECT_EQ(*square, (std::vector<int8_t>{6, -2, 1, 127}));
  EXPECT_FALSE(QuantizedReference(model, {1}));
  graph.tensors[0]->shape = graph.tensors[1]->shape = {1, 1, 1, 1, 2};
  auto weight = std::make_unique<tflite::TensorT>();
  weight->type = tflite::TensorType_INT8;
  weight->shape = {1, 1, 1, 2, 2};
  weight->buffer = 1;
  weight->quantization = std::make_unique<tflite::QuantizationParametersT>();
  weight->quantization->scale = {.25f, .5f};
  weight->quantization->zero_point = {0, 0};
  weight->quantization->quantized_dimension = 4;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  model.buffers[1]->data = {2, static_cast<uint8_t>(-3), 4, 5};
  auto bias = std::make_unique<tflite::TensorT>();
  bias->type = tflite::TensorType_INT32;
  bias->shape = {2};
  bias->buffer = 2;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  const int32_t bias_values[] = {1, -3};
  model.buffers[2]->data.resize(sizeof(bias_values));
  std::memcpy(model.buffers[2]->data.data(), bias_values, sizeof(bias_values));
  graph.tensors.push_back(std::move(weight));
  graph.tensors.push_back(std::move(bias));
  operation.inputs = {0, 2, 3};
  tflite::Conv3DOptionsT options;
  options.padding = tflite::Padding_VALID;
  options.stride_d = options.stride_h = options.stride_w = 1;
  options.dilation_d_factor = options.dilation_h_factor =
      options.dilation_w_factor = 1;
  operation.builtin_options.Set(options);
  model.operator_codes[0]->builtin_code = tflite::BuiltinOperator_CONV_3D;
  auto conv = QuantizedReference(model, {-1, 2});
  ASSERT_TRUE(conv);
  EXPECT_EQ(*conv, (std::vector<int8_t>{1, 7}));
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
  for (const auto* name :
       {VK_ARM_TENSORS_EXTENSION_NAME, VK_ARM_DATA_GRAPH_EXTENSION_NAME,
        VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME}) {
    VkExtensionProperties extension{};
    std::strncpy(extension.extensionName, name,
                 sizeof(extension.extensionName) - 1);
    extension.specVersion = 2;
    device.extensions.push_back(extension);
  }
  std::string reason;
  EXPECT_FALSE(CheckDeviceRequirements(device, reason));
  device.tensors_feature = device.graph_feature = device.synchronization2 =
      true;
  device.maintenance5 = device.replicated_composites = true;
  device.vulkan_memory_model = true;
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
  engine.profiles.push_back(
      {"Emulation Layer", VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM});
  CompiledPartition artifact;
  artifact.instruction_set = engine.instruction_set;
  artifact.profiles = {"PRO-INT"};
  artifact.level_requirements.checked = true;
  std::string reason;
  EXPECT_FALSE(
      CheckEngineRequirements(artifact, device, engine, false, reason));
  EXPECT_TRUE(CheckEngineRequirements(artifact, device, engine, true, reason));
  artifact.extensions = {"EXT-INT16"};
  EXPECT_FALSE(CheckEngineRequirements(artifact, device, engine, true, reason));
  engine.extensions.push_back({"EXT-INT16", 0});
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
  for (auto code :
       {tflite::BuiltinOperator_RELU, tflite::BuiltinOperator_SIN}) {
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
  options.values["graph_view"] = "raw";
  Json report;
  ASSERT_TRUE(Inspect(options, report, false));
  EXPECT_FALSE(report["complete_coverage"].get<bool>());
  const auto& partitions = report["subgraphs"][0]["partitions"];
  ASSERT_EQ(partitions.size(), 2);
  const std::set<int> origins = {partitions[0]["operators"][0].get<int>(),
                                 partitions[1]["operators"][0].get<int>()};
  EXPECT_EQ(origins, (std::set<int>{0, 2}));
  EXPECT_FALSE(report["subgraphs"][0]["operators"][1]["reason"]
                   .get<std::string>()
                   .empty());

  model.subgraphs[0]->operators[1]->opcode_index = 0;
  bytes = PackModel(model);
  ASSERT_TRUE(bytes);
  file.open(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  file.close();
  options.values["output_dir"] = testing::TempDir() + "/export";
  Json optimized;
  ASSERT_TRUE(Inspect(options, optimized, true));
  EXPECT_TRUE(optimized["complete_coverage"].get<bool>());
  const auto& fused = optimized["subgraphs"][0]["partitions"][0];
  EXPECT_TRUE(fused["spirv_valid"].get<bool>());
  ASSERT_EQ(fused["lowering"].size(), 3);
  EXPECT_EQ(fused["lowering"][0]["source_operators"], Json::array({0, 1, 2}));
  EXPECT_FALSE(fused["lowering"][0]["created_result_ids"].empty());
  EXPECT_TRUE(fused["lowering"][1]["created_result_ids"].empty());
  EXPECT_TRUE(fused["lowering"][2]["created_result_ids"].empty());
  EXPECT_EQ(fused["optimization"]["emitted_operators"], 1);
  EXPECT_TRUE(std::filesystem::is_regular_file(
      std::filesystem::path(options.Get("output_dir")) /
      fused["optimization"]["after_ir"].get<std::string>()));
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

TEST(TensorInputs, RejectsUnknownMissingAndWrongSizeInputsBeforeExecution) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  auto graph = std::make_unique<tflite::SubGraphT>();
  auto tensor = std::make_unique<tflite::TensorT>();
  tensor->type = tflite::TensorType_FLOAT32;
  tensor->shape = {2};
  graph->tensors.push_back(std::move(tensor));
  graph->tensors.push_back(
      std::make_unique<tflite::TensorT>(*graph->tensors[0]));
  graph->inputs = {0};
  graph->outputs = {1};
  auto opcode = std::make_unique<tflite::OperatorCodeT>();
  opcode->builtin_code = tflite::BuiltinOperator_RELU;
  opcode->deprecated_builtin_code =
      static_cast<int8_t>(tflite::BuiltinOperator_RELU);
  model.operator_codes.push_back(std::move(opcode));
  auto op = std::make_unique<tflite::OperatorT>();
  op->opcode_index = 0;
  op->inputs = {0};
  op->outputs = {1};
  graph->operators.push_back(std::move(op));
  model.subgraphs.push_back(std::move(graph));
  auto bytes = PackModel(model);
  ASSERT_TRUE(bytes);
  const auto base = testing::TempDir();
  std::ofstream file(base + "/raw.tflite", std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  file.close();
  Options options;
  options.command = "trace";
  options.values["model"] = base + "/raw.tflite";
  options.values["tensor_inputs"] = base + "/inputs.json";
  for (const auto& inputs :
       {Json::array(), Json::array({{{"index", 2}, {"file", "absent.bin"}}}),
        Json::array({{{"index", 0}, {"file", "absent.bin"}}})}) {
    file.open(base + "/inputs.json");
    file << Json({{"cases",
                   Json::array({{{"id", "invalid"}, {"inputs", inputs}}})}})
                .dump();
    file.close();
    Json report;
    EXPECT_FALSE(Trace(options, report));
    EXPECT_FALSE(report.contains("operators"));
  }
  options.command = "validate";
  options.values["start_sample"] =
      std::to_string(std::numeric_limits<int>::max());
  options.values["max_samples"] = "10";
  Json report;
  auto invalid_range = Trace(options, report);
  ASSERT_FALSE(invalid_range);
  EXPECT_NE(std::string(invalid_range.Error().Message()).find("sample range"),
            std::string::npos);
  model.subgraphs[0]->operators[0]->opcode_index = 42;
  bytes = PackModel(model);
  ASSERT_TRUE(bytes);
  file.open(base + "/raw.tflite", std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  file.close();
  options.values["start_sample"] = "0";
  options.values["max_samples"] = "1";
  EXPECT_FALSE(Trace(options, report));
}

TEST(ValidationReports, ErrorsFailAndIncompleteStreamsNeverPass) {
  const auto directory = testing::TempDir() + "/validation-report";
  std::filesystem::create_directories(directory);
  Json report;
  EXPECT_FALSE(CollectValidationLogs(directory, "sync", report));
  EXPECT_EQ(report["status"], "incomplete");
  const auto path = directory + "/instance";
  {
    std::ofstream missing(path);
    missing << Json({{"event", "closed"},
                     {"enabled", false},
                     {"attached", false},
                     {"lost", false}})
                   .dump()
            << '\n';
  }
  EXPECT_FALSE(CollectValidationLogs(directory, "sync", report));
  EXPECT_FALSE(report["baseline_matches"].get<bool>());

  auto write = [&](bool lost, int severity) {
    std::ofstream file(path);
    file << Json({{"event", "instance"},
                  {"api_version", VK_HEADER_VERSION_COMPLETE}})
                .dump()
         << '\n';
    file << Json({{"event", "message"},
                  {"severity", severity},
                  {"id", "VUID-test-09921"}})
                .dump()
         << '\n';
    file << Json({{"event", "closed"},
                  {"enabled", true},
                  {"attached", true},
                  {"lost", lost}})
                .dump()
         << '\n';
  };
  write(false, VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
  EXPECT_FALSE(CollectValidationLogs(directory, "sync", report));
  EXPECT_EQ(report["errors"], 1);
  EXPECT_EQ(report["status"], "failed");
  write(false, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT);
  EXPECT_TRUE(CollectValidationLogs(directory, "sync", report));
  EXPECT_EQ(report["warnings"], 1);
  write(true, 0);
  EXPECT_FALSE(CollectValidationLogs(directory, "sync", report));
  EXPECT_EQ(report["status"], "incomplete");
  EXPECT_TRUE(CollectValidationLogs(directory, "off", report));
  EXPECT_EQ(report["status"], "not_requested");
}

TEST(Inspection, SupportsAndCompilesInt16AttentionSubgraph) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  for (auto code : {tflite::BuiltinOperator_BATCH_MATMUL,
                    tflite::BuiltinOperator_SOFTMAX,
                    tflite::BuiltinOperator_ADD,
                    tflite::BuiltinOperator_GELU}) {
    auto opcode = std::make_unique<tflite::OperatorCodeT>();
    opcode->builtin_code = code;
    opcode->deprecated_builtin_code = static_cast<int8_t>(
        code <= tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
            ? code
            : tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
    model.operator_codes.push_back(std::move(opcode));
  }
  auto graph = std::make_unique<tflite::SubGraphT>();
  auto add_i16 = [&](std::vector<int32_t> shape, float scale) -> int32_t {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->shape = std::move(shape);
    tensor->type = tflite::TensorType_INT16;
    tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
    tensor->quantization->scale = {scale};
    tensor->quantization->zero_point = {0};
    const int32_t id = static_cast<int32_t>(graph->tensors.size());
    graph->tensors.push_back(std::move(tensor));
    return id;
  };
  const int32_t q = add_i16({1, 2, 4, 8}, 0.01f);
  const int32_t k = add_i16({1, 2, 8, 4}, 0.01f);
  const int32_t bias = add_i16({1, 2, 4, 4}, 0.005f);
  const int32_t bmm_out = add_i16({1, 2, 4, 4}, 0.005f);
  const int32_t add_out = add_i16({1, 2, 4, 4}, 0.005f);
  const int32_t sm_out = add_i16({1, 2, 4, 4}, 1.0f / 32768.0f);
  const int32_t gelu_out = add_i16({1, 2, 4, 4}, 0.005f);
  graph->inputs = {q, k, bias};
  graph->outputs = {gelu_out};

  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 0;
    op->inputs = {q, k};
    op->outputs = {bmm_out};
    tflite::BatchMatMulOptionsT opts;
    op->builtin_options.Set(opts);
    graph->operators.push_back(std::move(op));
  }
  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 2;
    op->inputs = {bmm_out, bias};
    op->outputs = {add_out};
    tflite::AddOptionsT opts;
    opts.fused_activation_function = tflite::ActivationFunctionType_NONE;
    op->builtin_options.Set(opts);
    graph->operators.push_back(std::move(op));
  }
  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 1;
    op->inputs = {add_out};
    op->outputs = {sm_out};
    tflite::SoftmaxOptionsT opts;
    opts.beta = 1.0f;
    op->builtin_options.Set(opts);
    graph->operators.push_back(std::move(op));
  }
  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 3;
    op->inputs = {sm_out};
    op->outputs = {gelu_out};
    tflite::GeluOptionsT opts;
    opts.approximate = false;
    op->builtin_options.Set(opts);
    graph->operators.push_back(std::move(op));
  }
  model.subgraphs.push_back(std::move(graph));
  auto bytes = PackModel(model);
  ASSERT_TRUE(bytes);
  const std::string path = testing::TempDir() + "/int16_attn.tflite";
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  file.close();

  Options options;
  options.values["model"] = path;
  options.values["graph_view"] = "raw";
  options.values["output_dir"] = testing::TempDir() + "/int16_export";
  Json report;
  ASSERT_TRUE(Inspect(options, report, true));
  EXPECT_TRUE(report["complete_coverage"].get<bool>());
  const auto& partitions = report["subgraphs"][0]["partitions"];
  ASSERT_EQ(partitions.size(), 1);
  EXPECT_TRUE(partitions[0]["spirv_valid"].get<bool>());
  EXPECT_EQ(partitions[0]["extensions"], Json::array({"EXT-INT16"}));
  std::set<int> opcodes;
  for (const auto& op : partitions[0]["tosa_opcodes"]) {
    opcodes.insert(op.get<int>());
  }
  // MATMUL (6), ADD (14), TABLE (30), CLZ (34), RESCALE (65) must be emitted;
  // INTDIV (19) and CLAMP (10) must not be emitted.
  EXPECT_TRUE(opcodes.count(6));
  EXPECT_TRUE(opcodes.count(14));
  EXPECT_TRUE(opcodes.count(30));
  EXPECT_TRUE(opcodes.count(34));
  EXPECT_TRUE(opcodes.count(65));
  EXPECT_FALSE(opcodes.count(19));
  EXPECT_FALSE(opcodes.count(10));
}

TEST(Inspection, ExtractsMultiSubgraphStablehloCompositeWithoutOutOfBounds) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  auto push_buf = [&](const auto& values) -> uint32_t {
    using Elem = typename std::decay_t<decltype(values)>::value_type;
    auto buf = std::make_unique<tflite::BufferT>();
    buf->data.resize(values.size() * sizeof(Elem));
    std::memcpy(buf->data.data(), values.data(), buf->data.size());
    model.buffers.push_back(std::move(buf));
    return static_cast<uint32_t>(model.buffers.size() - 1);
  };
  for (auto code : {tflite::BuiltinOperator_STABLEHLO_COMPOSITE,
                    tflite::BuiltinOperator_RELU}) {
    auto opcode = std::make_unique<tflite::OperatorCodeT>();
    opcode->builtin_code = code;
    opcode->version = 1;
    opcode->deprecated_builtin_code = static_cast<int8_t>(
        code <= tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
            ? code
            : tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
    model.operator_codes.push_back(std::move(opcode));
  }

  auto main_graph = std::make_unique<tflite::SubGraphT>();
  auto add_tensor = [&](tflite::SubGraphT& g, tflite::TensorType type,
                        std::vector<int32_t> shape, uint32_t buffer,
                        float scale) -> int32_t {
    auto t = std::make_unique<tflite::TensorT>();
    t->type = type;
    t->shape = std::move(shape);
    t->buffer = buffer;
    t->quantization = std::make_unique<tflite::QuantizationParametersT>();
    t->quantization->scale = {scale};
    t->quantization->zero_point = {0};
    const int32_t id = static_cast<int32_t>(g.tensors.size());
    g.tensors.push_back(std::move(t));
    return id;
  };
  const int32_t x =
      add_tensor(*main_graph, tflite::TensorType_INT16, {1, 2, 4}, 0, 0.001f);
  const int32_t gamma =
      add_tensor(*main_graph, tflite::TensorType_INT16, {4},
                 push_buf(std::vector<int16_t>{1000, 1100, 900, 1050}), 0.001f);
  const int32_t beta =
      add_tensor(*main_graph, tflite::TensorType_INT32, {4},
                 push_buf(std::vector<int32_t>{10, -10, 20, -20}), 1e-6f);
  const int32_t y =
      add_tensor(*main_graph, tflite::TensorType_INT16, {1, 2, 4}, 0, 0.002f);
  main_graph->inputs = {x};
  main_graph->outputs = {y};
  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 0;
    op->inputs = {x, gamma, beta};
    op->outputs = {y};
    flexbuffers::Builder attrs;
    attrs.Map([&] {
      attrs.Double("epsilon", 1e-6);
      attrs.Int("num_groups", 1);
      attrs.Int("sub_type", 1);
    });
    attrs.Finish();
    tflite::StableHLOCompositeOptionsT opts;
    opts.name = "odml.group_norm";
    opts.version = 1;
    opts.decomposition_subgraph_index = 1;
    opts.composite_attributes = attrs.GetBuffer();
    op->builtin_options_2.Set(opts);
    main_graph->operators.push_back(std::move(op));
  }
  model.subgraphs.push_back(std::move(main_graph));

  auto decomp_graph = std::make_unique<tflite::SubGraphT>();
  const int32_t dx =
      add_tensor(*decomp_graph, tflite::TensorType_INT16, {1, 2, 4}, 0, 0.001f);
  const int32_t dy =
      add_tensor(*decomp_graph, tflite::TensorType_INT16, {1, 2, 4}, 0, 0.001f);
  decomp_graph->inputs = {dx};
  decomp_graph->outputs = {dy};
  {
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 1;
    op->inputs = {dx};
    op->outputs = {dy};
    decomp_graph->operators.push_back(std::move(op));
  }
  model.subgraphs.push_back(std::move(decomp_graph));

  auto bytes = PackModel(model);
  ASSERT_TRUE(bytes);
  const std::string path = testing::TempDir() + "/composite_ln.tflite";
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  file.close();

  Options options;
  options.values["model"] = path;
  options.values["graph_view"] = "raw";
  options.values["output_dir"] = testing::TempDir() + "/composite_export";
  Json report;
  ASSERT_TRUE(Inspect(options, report, true));
  EXPECT_TRUE(report["complete_coverage"].get<bool>());
  ASSERT_EQ(report["subgraphs"].size(), 2);
  ASSERT_EQ(report["subgraphs"][0]["partitions"].size(), 1);
  EXPECT_TRUE(
      report["subgraphs"][0]["partitions"][0]["spirv_valid"].get<bool>());
}

TEST(DevtoolTest, CoverageIsOfflineAndListsEveryOperator) {
  Json report;
  ASSERT_TRUE(Coverage(report));
  EXPECT_EQ(report["operators"].size(), 66);
  EXPECT_EQ(report["counts"]["emitted"], 64);
  EXPECT_EQ(report["counts"]["unreachable"], 2);
  EXPECT_EQ(report["counts"]["pending"], 0);
  EXPECT_EQ(report["quantized_subset_count"], 26);
  for (const auto& subset : report["quantized_lowerings"]) {
    EXPECT_TRUE(subset["emulator"] == "passed" ||
                subset["emulator"] == "unavailable_EXT_INT16");
    if (subset["emulator"] == "unavailable_EXT_INT16")
      EXPECT_EQ(subset["required_extensions"], Json::array({"EXT-INT16"}));
    EXPECT_EQ(subset["hardware"], "not_qualified");
    EXPECT_FALSE(subset["fixtures"].empty());
  }
  for (const auto& op : report["operators"])
    EXPECT_EQ(op["hardware"], "not_qualified");
}

TEST(DevtoolTest, TfliteCoverageUsesTheCompilerRegistryAndSchema) {
  Json report;
  ASSERT_TRUE(Coverage(report, "tflite"));
  EXPECT_EQ(report["operator_set"], "tflite");
  EXPECT_EQ(report["operators"].size(),
            std::size(tflite::EnumValuesBuiltinOperator()));
  EXPECT_EQ(report["counts"]["implemented_subset"],
            GetOperationRegistrations().size());
  size_t variants = 0, missing = 0, excluded = 0;
  for (const auto& row : report["operators"]) {
    const auto* registration = FindOperationRegistration(
        static_cast<LiteRtOpCode>(row["opcode"].get<int>()));
    if (!registration) {
      EXPECT_NE(row["implementation"], "implemented_subset");
      EXPECT_TRUE(row["variants"].empty());
      missing += row["implementation"] == "not_implemented";
      excluded += row["implementation"] == "excluded";
      continue;
    }
    EXPECT_EQ(row["implementation"], "implemented_subset");
    EXPECT_EQ(row["data_input"], registration->data_input);
    ASSERT_EQ(row["variants"].size(), registration->variants.size());
    for (size_t i = 0; i < registration->variants.size(); ++i) {
      const auto& variant = registration->variants[i];
      EXPECT_EQ(row["variants"][i]["data_input_type"],
                RegisteredElementTypeName(variant.input_type));
      EXPECT_EQ(row["variants"][i]["output_type"],
                RegisteredElementTypeName(variant.output_type));
      EXPECT_EQ(row["variants"][i]["route"], LoweringRouteName(variant.route));
      EXPECT_EQ(row["variants"][i]["constraints"], variant.constraints);
      ++variants;
    }
  }
  EXPECT_EQ(report["counts"]["registered_type_pairs"], variants);
  EXPECT_EQ(report["counts"]["not_implemented"], missing);
  EXPECT_EQ(report["counts"]["excluded"], excluded);
  ASSERT_TRUE(Coverage(report, "all"));
  EXPECT_EQ(report["tosa"]["counts"]["emitted"], 64);
  EXPECT_EQ(report["tflite"]["counts"]["implemented_subset"],
            GetOperationRegistrations().size());
  EXPECT_FALSE(Coverage(report, "invalid"));
}

TEST(Options, CoverageOperatorSetMustBeExplicitAndValid) {
  for (const std::string& command : {"coverage", "inspect"}) {
    for (const std::string& op_set : {"tosa", "tflite", "all", "invalid"}) {
      std::vector<std::string> args = {"tool", command, "--op_set=" + op_set};
      std::vector<char*> argv;
      for (auto& arg : args) argv.push_back(arg.data());
      EXPECT_EQ(static_cast<bool>(
                    ParseOptions(static_cast<int>(argv.size()), argv.data())),
                command == "coverage" && op_set != "invalid");
    }
  }
}

TEST(DevtoolCliTest,
     MainCliCoversHelpCoverageTosaQueryCompileValidateTraceAndReferencePaths) {
  auto invoke_main = [](std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) argv.push_back(arg.data());
    return Main(static_cast<int>(argv.size()), argv.data());
  };

  // 1. Help and CLI validation errors.
  EXPECT_EQ(invoke_main({"tool", "--help"}), 0);
  EXPECT_EQ(invoke_main({"tool"}), 2);
  EXPECT_EQ(invoke_main({"tool", "coverage", "--json=2"}), 2);
  EXPECT_EQ(invoke_main({"tool", "coverage", "--model=x.tflite"}), 2);
  EXPECT_EQ(invoke_main({"tool", "tosa-query", "--model=x.tflite"}), 2);
  EXPECT_EQ(invoke_main({"tool", "tosa-query", "--json=2"}), 2);
  EXPECT_EQ(invoke_main({"tool", "inspect", "--op=ADD"}), 2);
  EXPECT_EQ(invoke_main({"tool", "validate", "--vulkan_validation=bad"}), 2);
  EXPECT_EQ(invoke_main({"tool", "inspect", "--vulkan_validation=core"}), 2);

  // 2. Multi-chunk HashFile (> 4096 bytes with tail rollover).
  const std::string big_bin = testing::TempDir() + "/big_chunk.bin";
  {
    std::ofstream out(big_bin, std::ios::binary | std::ios::trunc);
    const std::string payload(5000, 'A');
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }
  auto big_hash = HashFile(big_bin);
  ASSERT_TRUE(big_hash);
  EXPECT_EQ(big_hash->size(), 64u);

  // 3. Coverage CLI text (op_set=all prints both TOSA and TFLite) and JSON report.
  const std::string cov_json = testing::TempDir() + "/cov_report.json";
  EXPECT_EQ(invoke_main({"tool", "coverage", "--op_set=all",
                         "--report_json=" + cov_json}),
            0);
  auto cov_parsed = ReadJson(cov_json);
  ASSERT_TRUE(cov_parsed);
  EXPECT_EQ((*cov_parsed)["status"], "passed");

  // 4. tosa-query CLI summary, operator query, and missing --database failure.
  EXPECT_EQ(invoke_main({"tool", "tosa-query"}), 1);
  const std::string db_path = testing::TempDir() + "/tosa_db.json";
  {
    const Json db = Json::parse(R"json({
      "schema_version":1,"instruction_set":"TOSA.001000.1","status":"reference",
      "profiles":["PRO-INT","PRO-FP"],"extensions":{"EXT-BF16":{}},"counts":{"operators":1},
      "enums":{"mode_t":{"values":[
        {"name":"BASE","additional_requirements":{"any_of":[{"all_of":[]}]}}
      ]}},
      "operators":[{"name":"ADD",
        "arguments":[{"name":"input1","tosa_type":"tensor_t","rank_range":[1,6],"constant_required":false}],
        "variants":[{"id":"fp32","types":{"input1":{"element_type":"fp32_t"},"input2":{"element_type":"fp32_t"}},
          "parameters":{},"requirements":{"any_of":[{"all_of":["PRO-FP"]}]}}]
      }]
    })json");
    ASSERT_TRUE(WriteJson(db_path, db));
  }
  EXPECT_EQ(invoke_main({"tool", "tosa-query", "--database=" + db_path}), 0);
  EXPECT_EQ(
      invoke_main({"tool", "tosa-query", "--database=" + db_path, "--op=ADD"}),
      0);

  // 5. Build a 2-op INT8 ADD -> RELU6 model and test compile (--dump_vgf=1),
  //    effective inspect fallback, validate missing inputs, and isolated trace.
  tflite::ModelT two_op;
  two_op.version = 3;
  two_op.buffers.push_back(std::make_unique<tflite::BufferT>());
  for (auto code :
       {tflite::BuiltinOperator_ADD, tflite::BuiltinOperator_RELU6}) {
    auto oc = std::make_unique<tflite::OperatorCodeT>();
    oc->builtin_code = code;
    oc->version = 2;
    two_op.operator_codes.push_back(std::move(oc));
  }
  {
    auto g = std::make_unique<tflite::SubGraphT>();
    for (int i = 0; i < 3; ++i) {
      auto t = std::make_unique<tflite::TensorT>();
      t->name = "t" + std::to_string(i);
      t->type = tflite::TensorType_INT8;
      t->shape = {1, 4};
      t->quantization = std::make_unique<tflite::QuantizationParametersT>();
      t->quantization->scale = {0.125f};
      t->quantization->zero_point = {0};
      g->tensors.push_back(std::move(t));
    }
    auto cbuf = std::make_unique<tflite::BufferT>();
    cbuf->data = {1, 2, 3, 4};
    auto ct = std::make_unique<tflite::TensorT>();
    ct->name = "c";
    ct->type = tflite::TensorType_INT8;
    ct->shape = {1, 4};
    ct->buffer = 1;
    ct->quantization = std::make_unique<tflite::QuantizationParametersT>();
    ct->quantization->scale = {0.125f};
    ct->quantization->zero_point = {0};
    two_op.buffers.push_back(std::move(cbuf));
    g->tensors.push_back(std::move(ct));

    auto op0 = std::make_unique<tflite::OperatorT>();
    op0->opcode_index = 0;
    op0->inputs = {0, 3};
    op0->outputs = {1};
    op0->builtin_options.Set(tflite::AddOptionsT{});
    g->operators.push_back(std::move(op0));

    auto op1 = std::make_unique<tflite::OperatorT>();
    op1->opcode_index = 1;
    op1->inputs = {1};
    op1->outputs = {2};
    g->operators.push_back(std::move(op1));

    g->inputs = {0};
    g->outputs = {2};
    two_op.subgraphs.push_back(std::move(g));
  }
  const std::string model_path = testing::TempDir() + "/two_op.tflite";
  {
    auto packed = PackModel(two_op);
    ASSERT_TRUE(packed);
    std::ofstream out(model_path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(packed->data()),
              static_cast<std::streamsize>(packed->size()));
  }

  const std::string vgf_dir = testing::TempDir() + "/cli_vgf_out";
  const std::string compile_report = testing::TempDir() + "/cli_compile.json";
  EXPECT_EQ(invoke_main({"tool", "compile", "--model=" + model_path,
                         "--graph_view=raw", "--dump_vgf=1",
                         "--output_dir=" + vgf_dir,
                         "--report_json=" + compile_report}),
            0);
  auto compiled_json = ReadJson(compile_report);
  ASSERT_TRUE(compiled_json);
  EXPECT_EQ((*compiled_json)["status"], "passed");
  EXPECT_TRUE((*compiled_json)["subgraphs"][0]["partitions"][0].contains(
      "vgf_sha256"));

  // Effective inspect with missing plugin dir exercises CompilerDirectory & RuntimeOptions.
  EXPECT_EQ(invoke_main({"tool", "inspect", "--model=" + model_path,
                         "--graph_view=effective",
                         "--compiler_plugin_library_dir=" +
                             (testing::TempDir() + "/missing_plugin_dir")}),
            1);

  // Validate without --tensor_inputs fails cleanly.
  EXPECT_EQ(invoke_main({"tool", "validate", "--model=" + model_path}), 1);

  // Trace with 2 samples in isolated mode + vulkan_validation=core.
  const std::string in0_path = testing::TempDir() + "/in0.bin";
  {
    std::ofstream out(in0_path, std::ios::binary | std::ios::trunc);
    const int8_t vals[4] = {2, -2, 8, 16};
    out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
  }
  const std::string manifest_path = testing::TempDir() + "/manifest.json";
  ASSERT_TRUE(WriteJson(
      manifest_path,
      Json{{"cases",
            Json::array(
                {{{"id", "c0"},
                  {"inputs", Json::array({{{"index", 0}, {"file", in0_path}}})}},
                 {{"id", "c1"},
                  {"inputs",
                   Json::array({{{"index", 0}, {"file", in0_path}}})}}})}}));
  const std::string trace_report = testing::TempDir() + "/cli_trace.json";
  (void)invoke_main({"tool", "trace", "--model=" + model_path,
                     "--tensor_inputs=" + manifest_path, "--max_samples=1",
                     "--mode=isolated", "--vulkan_validation=core",
                     "--report_json=" + trace_report});
  auto traced_json = ReadJson(trace_report);
  ASSERT_TRUE(traced_json);
  EXPECT_EQ((*traced_json)["command"], "trace");
  EXPECT_TRUE((*traced_json).value("spirv_valid", false));
  EXPECT_TRUE(traced_json->contains("vulkan_validation"));

  // 6. INT8 CONV_3D SAME + RELU6 model exercises QuantizedReference SAME pad & RELU6
  //    plus validate --reference=fp64 --rescale_multiplier_bits=16.
  tflite::ModelT conv3d;
  conv3d.version = 3;
  conv3d.buffers.push_back(std::make_unique<tflite::BufferT>());
  {
    auto oc = std::make_unique<tflite::OperatorCodeT>();
    oc->builtin_code = tflite::BuiltinOperator_CONV_3D;
    oc->version = 1;
    conv3d.operator_codes.push_back(std::move(oc));
    auto g = std::make_unique<tflite::SubGraphT>();
    auto make_t = [&](tflite::TensorType dt, std::vector<int32_t> s,
                      uint32_t buf, float sc) {
      auto t = std::make_unique<tflite::TensorT>();
      t->type = dt;
      t->shape = std::move(s);
      t->buffer = buf;
      t->quantization = std::make_unique<tflite::QuantizationParametersT>();
      t->quantization->scale = {sc};
      t->quantization->zero_point = {0};
      g->tensors.push_back(std::move(t));
    };
    auto wbuf = std::make_unique<tflite::BufferT>();
    wbuf->data.assign(8, 2);
    conv3d.buffers.push_back(std::move(wbuf));
    auto bbuf = std::make_unique<tflite::BufferT>();
    const int32_t bval = 100;
    bbuf->data.resize(sizeof(bval));
    std::memcpy(bbuf->data.data(), &bval, sizeof(bval));
    conv3d.buffers.push_back(std::move(bbuf));

    make_t(tflite::TensorType_INT8, {1, 2, 2, 2, 1}, 0, 0.5f);
    make_t(tflite::TensorType_INT8, {2, 2, 2, 1, 1}, 1, 0.5f);
    make_t(tflite::TensorType_INT32, {1}, 2, 0.25f);
    make_t(tflite::TensorType_INT8, {1, 2, 2, 2, 1}, 0, 0.1f);
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 0;
    op->inputs = {0, 1, 2};
    op->outputs = {3};
    tflite::Conv3DOptionsT opts;
    opts.padding = tflite::Padding_SAME;
    opts.stride_d = opts.stride_h = opts.stride_w = 1;
    opts.dilation_d_factor = opts.dilation_h_factor = opts.dilation_w_factor = 1;
    opts.fused_activation_function = tflite::ActivationFunctionType_RELU6;
    op->builtin_options.Set(opts);
    g->operators.push_back(std::move(op));
    g->inputs = {0};
    g->outputs = {3};
    conv3d.subgraphs.push_back(std::move(g));
  }
  const std::vector<int8_t> c3d_in(8, 10);
  auto ref_out = QuantizedReference(conv3d, c3d_in);
  ASSERT_TRUE(ref_out);
  EXPECT_EQ(ref_out->size(), 8u);
  // Clamped to 6.0 / 0.1 = 60.
  EXPECT_EQ((*ref_out)[0], 60);

  const std::string c3d_model_path = testing::TempDir() + "/conv3d_same.tflite";
  {
    auto packed = PackModel(conv3d);
    ASSERT_TRUE(packed);
    std::ofstream out(c3d_model_path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(packed->data()),
              static_cast<std::streamsize>(packed->size()));
  }
  const std::string c3d_in_path = testing::TempDir() + "/c3d_in.bin";
  {
    std::ofstream out(c3d_in_path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(c3d_in.data()),
              static_cast<std::streamsize>(c3d_in.size()));
  }
  const std::string c3d_manifest = testing::TempDir() + "/c3d_manifest.json";
  ASSERT_TRUE(WriteJson(
      c3d_manifest,
      Json{{"cases",
            Json::array(
                {{{"id", "s0"},
                  {"inputs", Json::array({{{"index", 0}, {"file", c3d_in_path}}})}},
                 {{"id", "s1"},
                  {"inputs",
                   Json::array({{{"index", 0}, {"file", c3d_in_path}}})}}})}}));
  (void)invoke_main({"tool", "validate", "--model=" + c3d_model_path,
                     "--tensor_inputs=" + c3d_manifest, "--max_samples=2",
                     "--reference=fp64", "--rescale_multiplier_bits=16"});

  // 7. Reference and validation-log negative checks.
  EXPECT_FALSE(SoftmaxReference({}, 4, 0.1, 1.0, "fp64"));
  EXPECT_FALSE(SoftmaxReference({1, 2, 3, 4}, 4, 1e-45, 1e-45, "xnnpack"));
  conv3d.subgraphs[0]
      ->operators[0]
      ->builtin_options.AsConv3DOptions()
      ->fused_activation_function = tflite::ActivationFunctionType_TANH;
  EXPECT_FALSE(QuantizedReference(conv3d, c3d_in));

  const std::string bad_vlog_dir = testing::TempDir() + "/bad_vlogs";
  std::filesystem::create_directories(bad_vlog_dir);
  {
    std::ofstream f(bad_vlog_dir + "/bad.jsonl", std::ios::trunc);
    f << "not_json\n";
  }
  Json vlog_res;
  EXPECT_FALSE(CollectValidationLogs(bad_vlog_dir, "core", vlog_res));
}

}  // namespace
}  // namespace litert::arm_vulkan_ml::tool
