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
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "litert/cc/litert_macros.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_invocation.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register_ref.h"
#include "tflite/model_builder.h"
namespace litert::arm_vulkan_ml::tool {
namespace {
Error Failure(const std::string& text) { return Error(kLiteRtStatusErrorRuntimeFailure, text); }
Expected<std::string> ImagePath(const Options& options) {
  LITERT_ASSIGN_OR_RETURN(int start, options.Integer("start_sample", 0));
  if (start < 0) return Failure("Negative start_sample");
  std::ifstream file(options.Get("manifest")); if (!file) return Failure("Cannot open manifest");
  std::string line;
  for (int i = 0; std::getline(file, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream stream(line); std::string id, label, path;
    if (!std::getline(stream, id, '\t') || !std::getline(stream, label, '\t') || !std::getline(stream, path))
      return Failure("Manifest requires ID, label, image path columns");
    if (id == "id") continue;
    if (i++ != start) continue;
    if (!path.empty() && path.back() == '\r') path.pop_back();
    std::filesystem::path image(path);
    if (image.is_relative()) image = std::filesystem::path(options.Get("image_root",
        std::filesystem::path(options.Get("manifest")).parent_path().string())) / image;
    return image.string();
  }
  return Failure("start_sample exceeds manifest");
}
Expected<void> FillInput(tflite::Interpreter& cpu, const std::vector<float>& pixels) {
  if (cpu.inputs().size() != 1) return Failure("Trace requires one image input");
  auto* tensor = cpu.input_tensor(0);
  if (tensor->type == kTfLiteFloat32 && tensor->bytes == pixels.size() * sizeof(float)) {
    std::memcpy(tensor->data.raw, pixels.data(), tensor->bytes); return {};
  }
  if (tensor->type == kTfLiteInt8 && tensor->bytes == pixels.size() &&
      std::isfinite(tensor->params.scale) && tensor->params.scale > 0) {
    for (size_t i = 0; i < pixels.size(); ++i) tensor->data.int8[i] = static_cast<int8_t>(
        std::clamp(std::round(static_cast<double>(pixels[i]) / tensor->params.scale) + tensor->params.zero_point, -128.0, 127.0));
    return {};
  }
  return Failure("Unsupported image tensor type/shape/quantization");
}
}  // namespace
Expected<void> Trace(const Options& options, Json& report) {
  LITERT_ASSIGN_OR_RETURN(auto object, LoadModel(options.Get("model")));
  if (object->subgraphs.size() != 1 || object->subgraphs[0]->inputs.size() != 1)
    return Failure("Trace requires a single-subgraph, single-input model");
  const auto& graph = *object->subgraphs[0];
  const auto& input = *graph.tensors[graph.inputs[0]];
  if (input.shape.size() != 4 || input.shape[0] != 1 || ((input.shape[1] == 3) == (input.shape[3] == 3)))
    return Failure("Trace requires unambiguous NCHW/NHWC batch-one RGB");
  for (auto dim : input.shape_signature) if (dim < 0) return Failure("Dynamic trace input unsupported");
  if (options.Get("manifest").empty() || options.Get("preprocessing_config").empty())
    return Failure("Trace requires manifest and explicit preprocessing");
  LITERT_ASSIGN_OR_RETURN(auto config_json, imagenet::ReadJson(options.Get("preprocessing_config")));
  LITERT_ASSIGN_OR_RETURN(auto preprocessing, imagenet::ParsePreprocessing(config_json));
  LITERT_ASSIGN_OR_RETURN(auto image, ImagePath(options));
  LITERT_ASSIGN_OR_RETURN(auto pixels, imagenet::DecodePreprocess(image, preprocessing, input.shape[1] == 3));
  report["image"] = image;
  report["preprocessing"] = config_json;
  LITERT_ASSIGN_OR_RETURN(report["image_sha256"], imagenet::HashFile(image));
  imagenet::Sha256 input_hash;
  input_hash.Update(pixels.data(), pixels.size() * sizeof(float));
  report["input_sha256"] = input_hash.Hex();
  for (const auto& tolerance : std::vector<std::pair<std::string, double>>{
           {"atol", 1e-4}, {"rtol", 1e-3}, {"max_code_error", 4}, {"mean_code_error", .5}}) {
    LITERT_ASSIGN_OR_RETURN(auto value, options.Number(tolerance.first, tolerance.second));
    report["tolerances"][tolerance.first] = value;
  }
  LITERT_ASSIGN_OR_RETURN(int samples, options.Integer("max_samples", 1));
  if (samples != 1) return Failure("Trace supports exactly one image");
  LITERT_ASSIGN_OR_RETURN(int mb, options.Integer("capture_budget_mb", 256));
  LITERT_ASSIGN_OR_RETURN(int allow, options.Integer("allow_emulation", 0));
  LITERT_ASSIGN_OR_RETURN(int dump, options.Integer("dump_tensors", 0));
  if (mb < 1 || mb > 4096 || (dump != 0 && dump != 1)) return Failure("Invalid capture budget or dump_tensors");
  const size_t budget = static_cast<size_t>(mb) * 1024 * 1024;
  const bool isolated = options.Get("mode", "accumulated") == "isolated";
  if (!isolated && options.Get("mode", "accumulated") != "accumulated") return Failure("Unknown trace mode");
  if (dump && options.Get("output_dir").empty()) return Failure("Tensor dumps require output_dir");
  report["trace_mode"] = isolated ? "isolated_cpu_fed" : "instrumented_prefix";
  report["capture_budget_bytes"] = budget;
  report["budget_scope"] = "captured tensors; excludes model weights and execution workspace";
  report["instrumentation_note"] = "Exposing outputs may change fusion/optimization; not an uninstrumented driver trace";
  report["operators"] = Json::array();
  // Analyze the complete original graph before executing any prefix.
  LITERT_ASSIGN_OR_RETURN(auto original_bytes, PackModel(*object));
  LITERT_ASSIGN_OR_RETURN(auto original, ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(original_bytes.data(), original_bytes.size())));
  LITERT_ASSIGN_OR_RETURN(auto original_graph, original.Subgraph(0));
  for (const auto& op : original_graph.Ops()) {
    std::string reason;
    if (!AnalyzeOperation(op, reason)) return Failure("Trace requires complete coverage: " + reason);
  }
  for (size_t index = 0; index < graph.operators.size(); ++index) {
    report["last_attempted_operator"] = index;
    const auto& op = *graph.operators[index];
    std::vector<int32_t> runtime_inputs;
    for (auto id : op.inputs) if (id >= 0 && object->buffers[graph.tensors[id]->buffer]->data.empty()) {
      if (std::find(runtime_inputs.begin(), runtime_inputs.end(), id) == runtime_inputs.end()) runtime_inputs.push_back(id);
    }
    size_t captured = 0;
    auto account = [&](int32_t id) -> bool {
      if (id < 0 || static_cast<size_t>(id) >= graph.tensors.size()) return false;
      const auto& tensor = *graph.tensors[id];
      size_t bytes = tensor.type == tflite::TensorType_FLOAT32 ? sizeof(float) : 1;
      if (tensor.type != tflite::TensorType_FLOAT32 && tensor.type != tflite::TensorType_INT8) return false;
      for (auto dim : tensor.shape) { if (dim <= 0 || bytes > budget / static_cast<size_t>(dim)) return false; bytes *= dim; }
      if (bytes > (budget - captured) / 2) return false;
      captured += bytes * 2; return true;
    };
    for (auto id : op.outputs) if (!account(id)) return Failure("Captured output exceeds budget or unsupported type/shape");
    const auto& preserved_inputs = isolated ? runtime_inputs : graph.inputs;
    for (auto id : preserved_inputs) if (!account(id)) return Failure("Captured input exceeds budget or unsupported type/shape");
    tflite::ModelT prefix(*object); prefix.signature_defs.clear();
    auto& sg = *prefix.subgraphs[0]; sg.operators.resize(index + 1); sg.outputs = op.outputs;
    for (auto id : preserved_inputs) if (std::find(sg.outputs.begin(), sg.outputs.end(), id) == sg.outputs.end()) sg.outputs.push_back(id);
    LITERT_ASSIGN_OR_RETURN(auto cpu_bytes, PackModel(prefix));
    auto flat = tflite::FlatBufferModel::BuildFromBuffer(reinterpret_cast<const char*>(cpu_bytes.data()), cpu_bytes.size());
    if (!flat) return Failure("Cannot build CPU prefix");
    tflite::ops::builtin::BuiltinRefOpResolver resolver; std::unique_ptr<tflite::Interpreter> cpu;
    if (tflite::InterpreterBuilder(*flat, resolver)(&cpu) != kTfLiteOk || !cpu || cpu->AllocateTensors() != kTfLiteOk)
      return Failure("Cannot allocate CPU reference prefix");
    cpu->SetNumThreads(1);
    LITERT_RETURN_IF_ERROR(FillInput(*cpu, pixels));
    if (cpu->Invoke() != kTfLiteOk) return Failure("CPU reference prefix failed");
    sg.outputs = op.outputs;
    if (isolated) {
      auto selected = std::move(sg.operators[index]); sg.operators.clear(); sg.operators.push_back(std::move(selected));
      sg.inputs = runtime_inputs;
    }
    LITERT_ASSIGN_OR_RETURN(auto vk_bytes, PackModel(prefix));
    LITERT_ASSIGN_OR_RETURN(auto vk_model, ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(vk_bytes.data(), vk_bytes.size())));
    LITERT_ASSIGN_OR_RETURN(auto vk_graph, vk_model.Subgraph(0));
    std::string reason; auto artifact = CompilePartition(vk_graph, reason);
    if (!artifact) return Failure("Trace compilation: " + reason);
    auto invocation = VulkanInvocation::Create(artifact, allow != 0, reason);
    if (!invocation) return Failure("Trace device/pipeline: " + reason);
    std::vector<const void*> inputs;
    for (auto id : sg.inputs) inputs.push_back(cpu->tensor(id)->data.raw);
    std::vector<std::vector<float>> actual(op.outputs.size()); std::vector<void*> outputs;
    for (size_t j = 0; j < op.outputs.size(); ++j) {
      size_t bytes = cpu->tensor(op.outputs[j])->bytes;
      actual[j].resize((bytes + sizeof(float) - 1) / sizeof(float)); outputs.push_back(actual[j].data());
    }
    if (!invocation->Run(inputs, outputs, reason)) return Failure("Trace execution: " + reason);
    Json row = {{"index", index}, {"name", tflite::EnumNameBuiltinOperator(object->operator_codes[op.opcode_index]->builtin_code)},
        {"outputs", Json::array()}, {"captured_bytes", captured}};
    bool passed = true;
    for (size_t j = 0; j < op.outputs.size(); ++j) {
      const auto* expected = cpu->tensor(op.outputs[j]); bool floating = expected->type == kTfLiteFloat32;
      Json comparison;
      LITERT_RETURN_IF_ERROR(Compare(actual[j].data(), expected->data.raw, expected->bytes / (floating ? 4 : 1), floating, options, comparison));
      comparison["tensor"] = op.outputs[j]; passed &= comparison["passed"].get<bool>();
      const auto& tensor = *graph.tensors[op.outputs[j]];
      comparison["shape"] = tensor.shape;
      comparison["type"] = tflite::EnumNameTensorType(tensor.type);
      if (tensor.quantization) {
        comparison["scales"] = tensor.quantization->scale;
        comparison["zero_points"] = tensor.quantization->zero_point;
        comparison["quantized_dimension"] = tensor.quantization->quantized_dimension;
      }
      if (dump && !comparison["passed"].get<bool>()) {
        std::error_code ec; std::filesystem::create_directories(options.Get("output_dir"), ec);
        if (ec) return Failure(ec.message());
        auto base = std::filesystem::path(options.Get("output_dir")) / ("operator_" + std::to_string(index) + "_tensor_" + std::to_string(op.outputs[j]));
        for (bool reference : {false, true}) {
          auto name = base.string() + (reference ? ".cpu.bin" : ".vulkan.bin");
          std::ofstream file(name, std::ios::binary);
          file.write(reference ? expected->data.raw : reinterpret_cast<const char*>(actual[j].data()), expected->bytes); file.flush();
          if (!file) return Failure("Cannot write tensor dump");
        }
      }
      row["outputs"].push_back(std::move(comparison));
    }
    report["operators"].push_back(std::move(row));
    if (!passed) { report["first_divergence"] = index; return Failure("Intermediate output exceeds tolerance"); }
  }
  report["complete"] = true;
  return {};
}
}  // namespace litert::arm_vulkan_ml::tool
