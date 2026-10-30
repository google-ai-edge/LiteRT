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
#include <cmath>
#include <fstream>
#include <filesystem>
#include <limits>
#include <iterator>
#include <memory>
#include <set>
#include <string>

#include "gtest/gtest.h"
#include "tflite/interpreter.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"

namespace litert::arm_vulkan_ml::tool {
namespace {

TEST(ImageInput, UsesSharedXnnpackRoundingAndSaturation) {
  tflite::Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(0, kTfLiteInt8, "input",
      std::vector<int>{1, 6}, TfLiteQuantizationParams{1.0f, 0}), kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  const std::vector<float> values{.5f, 1.5f, -.5f, -1.5f, -1000, 1000};
  ASSERT_TRUE(FillImageInput(interpreter, values));
  const auto* bytes = interpreter.typed_input_tensor<int8_t>(0);
  EXPECT_EQ(std::vector<int8_t>(bytes, bytes + values.size()),
            (std::vector<int8_t>{0, 2, 0, -2, -128, 127}));
  EXPECT_EQ(std::vector<int8_t>(bytes, bytes + values.size()),
            imagenet::QuantizePerTensor<int8_t>(values, {1.0f, 0}));
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

TEST(Options, OptimizationModesAreValidated) {
  for (const std::string& mode : {"off", "exact", "qualified", "invalid"}) {
    std::vector<std::string> args = {"tool", "compile", "--optimization_mode=" + mode};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    auto parsed = ParseOptions(static_cast<int>(argv.size()), argv.data());
    EXPECT_EQ(static_cast<bool>(parsed), mode != "invalid");
    if (parsed) { auto selected = parsed->Optimization(); ASSERT_TRUE(selected); EXPECT_EQ(OptimizationModeName(*selected), mode); }
  }
  auto defaults = Options{}.Optimization(); ASSERT_TRUE(defaults);
  EXPECT_EQ(*defaults, kLrtArmVulkanMLOptimizationQualified);
}

TEST(Options, SoftmaxLutAndAlternateReferencesAreValidated) {
  auto defaults = Options{}.SoftmaxLut();
  ASSERT_TRUE(defaults);
  EXPECT_EQ(*defaults, kLrtArmVulkanMLSoftmaxLutAuto);
  for (const std::string& lut : {"auto", "int8", "int16", "invalid"}) {
    std::vector<std::string> args = {"tool", "compile", "--softmax_lut_type=" + lut};
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
  for (const std::string& command : {"validate", "compile", "compare", "trace"}) {
    std::vector<std::string> args = {"tool", command, "--reference=fp64"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    EXPECT_EQ(static_cast<bool>(ParseOptions(static_cast<int>(argv.size()), argv.data())),
              command == "validate");
  }
}

TEST(SoftmaxReference, MathematicalQuantizationAndUnsignedRebasing) {
  for (size_t channels : {size_t{1}, size_t{2}, size_t{4}, size_t{512}, size_t{1000}}) {
    std::vector<int8_t> input(channels * 2, -17);
    auto fp64 = SoftmaxReference(input, channels, .125, 1, "fp64");
    auto xnn = SoftmaxReference(input, channels, .125, 1, "xnnpack");
    ASSERT_TRUE(fp64); ASSERT_TRUE(xnn);
    EXPECT_EQ(*fp64, *xnn);
    const int expected = std::min(255, static_cast<int>(std::floor(256.0 / channels + .5))) - 128;
    EXPECT_EQ((*fp64)[0], expected);
  }
  std::vector<int8_t> input{-128, -37, 0, 42, 127};
  auto fp64 = SoftmaxReference(input, 5, .125, .7, "fp64");
  auto xnn = SoftmaxReference(input, 5, .125, .7, "xnnpack");
  ASSERT_TRUE(fp64); ASSERT_TRUE(xnn);
  for (size_t i = 0; i < input.size(); ++i) EXPECT_LE(std::abs((*fp64)[i] - (*xnn)[i]), 1);
}
TEST(SoftmaxReference, DetectsRoundedXnnpackSumOverflowInsteadOfUsingWrappedResults) {
  for (size_t channels : {size_t{1001}, size_t{1024}, size_t{2048}}) {
    std::vector<int8_t> input(channels, 0);
    EXPECT_TRUE(SoftmaxReference(input, channels, .125, 1, "fp64"));
    auto xnn = SoftmaxReference(input, channels, .125, 1, "xnnpack");
    EXPECT_FALSE(xnn);
    if (!xnn) EXPECT_NE(std::string(xnn.Error().Message()).find("exceeds UINT32"), std::string::npos);
  }
}

TEST(QuantizedReference, SquareAndPerChannelConv3dHaveIndependentExpectedValues) {
  tflite::ModelT model;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  model.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
  model.operator_codes.push_back(std::make_unique<tflite::OperatorCodeT>());
  auto& graph = *model.subgraphs[0];
  graph.operators.push_back(std::make_unique<tflite::OperatorT>());
  graph.inputs = {0}; graph.outputs = {1};
  auto& operation = *graph.operators[0];
  operation.inputs = {0}; operation.outputs = {1};
  for (int i = 0; i < 2; ++i) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->type = tflite::TensorType_INT8; tensor->shape = {4};
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
  weight->type = tflite::TensorType_INT8; weight->shape = {1, 1, 1, 2, 2};
  weight->buffer = 1;
  weight->quantization = std::make_unique<tflite::QuantizationParametersT>();
  weight->quantization->scale = {.25f, .5f};
  weight->quantization->zero_point = {0, 0};
  weight->quantization->quantized_dimension = 4;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  model.buffers[1]->data = {2, static_cast<uint8_t>(-3), 4, 5};
  auto bias = std::make_unique<tflite::TensorT>();
  bias->type = tflite::TensorType_INT32; bias->shape = {2}; bias->buffer = 2;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  const int32_t bias_values[] = {1, -3};
  model.buffers[2]->data.resize(sizeof(bias_values));
  std::memcpy(model.buffers[2]->data.data(), bias_values, sizeof(bias_values));
  graph.tensors.push_back(std::move(weight)); graph.tensors.push_back(std::move(bias));
  operation.inputs = {0, 2, 3};
  tflite::Conv3DOptionsT options;
  options.padding = tflite::Padding_VALID;
  options.stride_d = options.stride_h = options.stride_w = 1;
  options.dilation_d_factor = options.dilation_h_factor = options.dilation_w_factor = 1;
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
  engine.profiles.push_back({"Emulation Layer", VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM});
  CompiledPartition artifact;
  artifact.instruction_set = engine.instruction_set;
  artifact.profiles = {"PRO-INT"};
  artifact.level_requirements.checked = true;
  std::string reason;
  EXPECT_FALSE(CheckEngineRequirements(artifact, device, engine, false, reason));
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
  options.values["optimization_mode"] = "off";
  report = Json::object();
  ASSERT_TRUE(Inspect(options, report, true));
  EXPECT_TRUE(report["complete_coverage"].get<bool>());
  const auto& compiled = report["subgraphs"][0]["partitions"][0];
  EXPECT_TRUE(compiled["spirv_valid"].get<bool>());
  ASSERT_EQ(compiled["lowering"].size(), 3);
  EXPECT_EQ(compiled["lowering"][2]["operator"], 2);
  EXPECT_FALSE(compiled["lowering"][2]["created_result_ids"].empty());
  options.values["optimization_mode"] = "exact";
  Json optimized;
  ASSERT_TRUE(Inspect(options, optimized, true));
  const auto& fused = optimized["subgraphs"][0]["partitions"][0];
  EXPECT_EQ(fused["lowering"][0]["source_operators"], Json::array({0, 1, 2}));
  EXPECT_FALSE(fused["lowering"][0]["created_result_ids"].empty());
  EXPECT_TRUE(fused["lowering"][1]["created_result_ids"].empty());
  EXPECT_TRUE(fused["lowering"][2]["created_result_ids"].empty());
  EXPECT_EQ(fused["optimization"]["emitted_operators"], 1);
  EXPECT_TRUE(std::filesystem::is_regular_file(std::filesystem::path(options.Get("output_dir")) / fused["optimization"]["after_ir"].get<std::string>()));
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
  graph->tensors.push_back(std::make_unique<tflite::TensorT>(*graph->tensors[0]));
  graph->inputs = {0}; graph->outputs = {1};
  auto opcode = std::make_unique<tflite::OperatorCodeT>();
  opcode->builtin_code = tflite::BuiltinOperator_RELU;
  opcode->deprecated_builtin_code = static_cast<int8_t>(tflite::BuiltinOperator_RELU);
  model.operator_codes.push_back(std::move(opcode));
  auto op = std::make_unique<tflite::OperatorT>();
  op->opcode_index = 0; op->inputs = {0}; op->outputs = {1};
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
  for (const auto& inputs : {Json::array(),
                            Json::array({{{"index", 2}, {"file", "absent.bin"}}}),
                            Json::array({{{"index", 0}, {"file", "absent.bin"}}})}) {
    file.open(base + "/inputs.json");
    file << Json({{"cases", Json::array({{{"id", "invalid"}, {"inputs", inputs}}})}}).dump();
    file.close();
    Json report;
    EXPECT_FALSE(Trace(options, report));
    EXPECT_FALSE(report.contains("operators"));
  }
  options.command = "validate";
  options.values["start_sample"] = std::to_string(std::numeric_limits<int>::max());
  options.values["max_samples"] = "10";
  Json report;
  auto invalid_range = Trace(options, report);
  ASSERT_FALSE(invalid_range);
  EXPECT_NE(std::string(invalid_range.Error().Message()).find("sample range"), std::string::npos);
  model.subgraphs[0]->operators[0]->opcode_index = 42;
  bytes = PackModel(model); ASSERT_TRUE(bytes);
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
    missing << Json({{"event", "closed"}, {"enabled", false}, {"attached", false}, {"lost", false}}).dump() << '\n';
  }
  EXPECT_FALSE(CollectValidationLogs(directory, "sync", report));
  EXPECT_FALSE(report["baseline_matches"].get<bool>());

  auto write = [&](bool lost, int severity) {
    std::ofstream file(path);
    file << Json({{"event", "instance"}, {"api_version", VK_HEADER_VERSION_COMPLETE}}).dump() << '\n';
    file << Json({{"event", "message"}, {"severity", severity}, {"id", "VUID-test-09921"}}).dump() << '\n';
    file << Json({{"event", "closed"}, {"enabled", true}, {"attached", true}, {"lost", lost}}).dump() << '\n';
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

TEST(DevtoolTest, CoverageIsOfflineAndListsEveryOperator) {
  Json report;
  ASSERT_TRUE(Coverage(report));
  EXPECT_EQ(report["operators"].size(),66);
  EXPECT_EQ(report["counts"]["emitted"],63);
  EXPECT_EQ(report["counts"]["unreachable"],3);
  EXPECT_EQ(report["counts"]["pending"],0);
  EXPECT_EQ(report["quantized_subset_count"],17);
  for(const auto& subset:report["quantized_lowerings"]){EXPECT_EQ(subset["emulator"],"passed");EXPECT_EQ(subset["hardware"],"not_qualified");EXPECT_FALSE(subset["fixtures"].empty());}
  for(const auto& op:report["operators"])EXPECT_EQ(op["hardware"],"not_qualified");
}

TEST(DevtoolTest, TfliteCoverageUsesTheCompilerRegistryAndSchema) {
  Json report;
  ASSERT_TRUE(Coverage(report, "tflite"));
  EXPECT_EQ(report["operator_set"], "tflite");
  EXPECT_EQ(report["operators"].size(), std::size(tflite::EnumValuesBuiltinOperator()));
  EXPECT_EQ(report["counts"]["implemented_subset"], GetOperationRegistrations().size());
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
      EXPECT_EQ(row["variants"][i]["data_input_type"], RegisteredElementTypeName(variant.input_type));
      EXPECT_EQ(row["variants"][i]["output_type"], RegisteredElementTypeName(variant.output_type));
      EXPECT_EQ(row["variants"][i]["route"], LoweringRouteName(variant.route));
      EXPECT_EQ(row["variants"][i]["constraints"], variant.constraints);
      ++variants;
    }
  }
  EXPECT_EQ(report["counts"]["registered_type_pairs"], variants);
  EXPECT_EQ(report["counts"]["not_implemented"], missing);
  EXPECT_EQ(report["counts"]["excluded"], excluded);
  ASSERT_TRUE(Coverage(report, "all"));
  EXPECT_EQ(report["tosa"]["counts"]["emitted"], 63);
  EXPECT_EQ(report["tflite"]["counts"]["implemented_subset"], GetOperationRegistrations().size());
  EXPECT_FALSE(Coverage(report, "invalid"));
}

TEST(Options, CoverageOperatorSetMustBeExplicitAndValid) {
  for (const std::string& command : {"coverage", "inspect"}) {
    for (const std::string& op_set : {"tosa", "tflite", "all", "invalid"}) {
      std::vector<std::string> args = {"tool", command, "--op_set=" + op_set};
      std::vector<char*> argv;
      for (auto& arg : args) argv.push_back(arg.data());
      EXPECT_EQ(static_cast<bool>(ParseOptions(static_cast<int>(argv.size()), argv.data())),
                command == "coverage" && op_set != "invalid");
    }
  }
}

}  // namespace
}  // namespace litert::arm_vulkan_ml::tool
