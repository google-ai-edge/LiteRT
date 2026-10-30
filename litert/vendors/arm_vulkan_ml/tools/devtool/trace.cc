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
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>
#include "spirv-tools/libspirv.hpp"
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
size_t TensorWidth(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_INT8:
    case tflite::TensorType_BOOL: return 1;
    case tflite::TensorType_INT32:
    case tflite::TensorType_UINT32:
    case tflite::TensorType_FLOAT32: return 4;
    case tflite::TensorType_COMPLEX64: return 8;
    default: return 0;
  }
}
Expected<std::string> ImagePath(const Options& options) {
  LITERT_ASSIGN_OR_RETURN(int start, options.Integer("start_sample", 0));
  LITERT_ASSIGN_OR_RETURN(auto rows, imagenet::ReadSamples(options.Get("manifest"),
      options.Get("image_root"), start, 1));
  return rows.front().path;
}
}  // namespace
Expected<void> FillImageInput(tflite::Interpreter& cpu, const std::vector<float>& pixels) {
  if (cpu.inputs().size() != 1) return Failure("Trace requires one image input");
  auto* tensor = cpu.input_tensor(0);
  if (tensor->type == kTfLiteFloat32 && tensor->bytes == pixels.size() * sizeof(float)) {
    std::memcpy(tensor->data.raw, pixels.data(), tensor->bytes); return {};
  }
  if (tensor->type == kTfLiteInt8 && tensor->bytes == pixels.size() &&
      std::isfinite(tensor->params.scale) && tensor->params.scale > 0) {
    const auto quantized = imagenet::QuantizePerTensor<int8_t>(pixels,
        {tensor->params.scale, tensor->params.zero_point});
    std::memcpy(tensor->data.raw, quantized.data(), quantized.size());
    return {};
  }
  return Failure("Unsupported image tensor type/shape/quantization");
}
Expected<void> Trace(const Options& options, Json& report) {
  LITERT_ASSIGN_OR_RETURN(auto optimization_mode, options.Optimization());
  LITERT_ASSIGN_OR_RETURN(auto softmax_lut, options.SoftmaxLut());
  const auto reference_kind = options.Get("reference", "tflite");
  const bool external_reference = reference_kind != "tflite";
  if (external_reference && (options.command != "validate" ||
      (reference_kind != "fp64" && reference_kind != "xnnpack"))) return Failure("Invalid alternate reference");
  LITERT_ASSIGN_OR_RETURN(auto object, LoadModel(options.Get("model")));
  if (object->subgraphs.size() != 1 || object->subgraphs[0]->inputs.empty())
    return Failure("Execution requires one subgraph with data inputs");
  report["model"] = options.Get("model");
  LITERT_ASSIGN_OR_RETURN(report["model_sha256"], imagenet::HashFile(options.Get("model")));
  const auto& graph = *object->subgraphs[0];
  if (graph.operators.empty()) return Failure("Empty execution graph unsupported");
  const bool compare = options.command == "compare";
  const bool full_graph = options.command == "validate" || compare;
  if (full_graph && options.values.count("mode")) return Failure("mode is only valid for trace");
  const bool raw = !options.Get("tensor_inputs").empty();
  if (graph.operators[0]->opcode_index >= object->operator_codes.size())
    return Failure("Invalid operator code index");
  const auto first_code = object->operator_codes[graph.operators[0]->opcode_index]->builtin_code;
  const bool mathematical_reference = external_reference && first_code != tflite::BuiltinOperator_SOFTMAX;
  if (external_reference && (!raw || graph.operators.size() != 1 || graph.inputs.size() != 1 || graph.outputs.size() != 1 ||
      (first_code != tflite::BuiltinOperator_SOFTMAX &&
       (reference_kind != "fp64" || (first_code != tflite::BuiltinOperator_SQUARE && first_code != tflite::BuiltinOperator_CONV_3D)))))
    return Failure("Alternate reference requires raw single-op INT8 SOFTMAX, or fp64 SQUARE/CONV_3D");
  report["reference"] = reference_kind;
  std::vector<float> pixels;
  std::vector<std::vector<float>> raw_inputs(graph.inputs.size());
  if (raw) {
    if (!options.Get("manifest").empty() || !options.Get("preprocessing_config").empty())
      return Failure("tensor_inputs and image options are mutually exclusive");
    LITERT_ASSIGN_OR_RETURN(auto manifest, imagenet::ReadJson(options.Get("tensor_inputs")));
    if (!manifest.is_object() || !manifest.contains("cases") || !manifest["cases"].is_array())
      return Failure("Tensor manifest requires cases array");
    LITERT_ASSIGN_OR_RETURN(int start, options.Integer("start_sample", 0));
    LITERT_ASSIGN_OR_RETURN(int samples, options.Integer("max_samples", options.command == "trace" ? 1 : 3));
    if (start < 0 || samples < 1 || samples > 10 || (options.command == "trace" && samples != 1) ||
        static_cast<size_t>(start) > manifest["cases"].size() ||
        static_cast<size_t>(samples) > manifest["cases"].size() - static_cast<size_t>(start))
      return Failure("Invalid tensor sample range");
    if (samples > 1) {
      report["cases"] = Json::array(); bool passed = true;
      for (int i = 0; i < samples; ++i) {
        Options child = options; child.values["max_samples"] = "1";
        child.values["start_sample"] = std::to_string(start + i);
        Json row; auto status = Trace(child, row);
        row["status"] = status ? "passed" : "failed";
        if (!status) { row["error"] = std::string(status.Error().Message()); passed = false; }
        report["cases"].push_back(std::move(row));
      }
      if (!passed) return Failure("Tensor cases failed");
      return {};
    }
    const auto& sample = manifest["cases"][start];
    if (!sample.is_object() || !sample.contains("id") || !sample.contains("inputs") || !sample["inputs"].is_array())
      return Failure("Tensor case requires id and inputs");
    report["case_id"] = sample["id"];
    std::set<int> seen;
    for (const auto& entry : sample["inputs"]) {
      if (!entry.is_object() || !entry.contains("index") || !entry["index"].is_number_integer() ||
          !entry.contains("file") || !entry["file"].is_string()) return Failure("Invalid tensor input entry");
      if (entry["index"] < 0 || entry["index"] >= raw_inputs.size()) return Failure("Unknown input index");
      int slot = entry["index"].get<int>();
      if (slot < 0 || static_cast<size_t>(slot) >= raw_inputs.size() || !seen.insert(slot).second)
        return Failure("Unknown or duplicate input index");
      const auto& tensor = *graph.tensors[graph.inputs[slot]];
      const size_t width = TensorWidth(tensor.type);
      if (!width) return Failure("Raw inputs require FP32, INT8, INT32, UINT32, BOOL or COMPLEX64");
      size_t count = 1;
      for (auto dim : tensor.shape) {
        if (dim <= 0 || count > (256u * 1024u * 1024u / width) / static_cast<size_t>(dim)) return Failure("Invalid/oversized raw shape");
        count *= dim;
      }
      for (auto dim : tensor.shape_signature) if (dim < 0) return Failure("Dynamic raw input unsupported");
      std::filesystem::path path(entry["file"].get<std::string>());
      if (path.is_relative()) path = std::filesystem::path(options.Get("tensor_inputs")).parent_path() / path;
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file || file.tellg() != static_cast<std::streamoff>(count * width)) return Failure("Raw input byte count mismatch");
      raw_inputs[slot].resize((count*width+3)/4); file.seekg(0);
      file.read(reinterpret_cast<char*>(raw_inputs[slot].data()), count * width);
      if (!file) return Failure("Cannot read raw input");
      LITERT_ASSIGN_OR_RETURN(auto hash, imagenet::HashFile(path.string()));
      report["inputs"].push_back({{"index", slot}, {"sha256", hash}});
    }
    if (seen.size() != raw_inputs.size()) return Failure("Missing raw input");
  } else {
    if (graph.inputs.size() != 1) return Failure("Images require one input");
    const auto& input = *graph.tensors[graph.inputs[0]];
    if (input.shape.size() != 4 || input.shape[0] != 1 || ((input.shape[1] == 3) == (input.shape[3] == 3)))
      return Failure("Images require unambiguous NCHW/NHWC RGB batch one");
    for (auto dim : input.shape_signature) if (dim < 0) return Failure("Dynamic input unsupported");
    LITERT_ASSIGN_OR_RETURN(auto config_json, imagenet::ReadJson(options.Get("preprocessing_config")));
    LITERT_ASSIGN_OR_RETURN(auto preprocessing, imagenet::ParsePreprocessing(config_json));
    LITERT_ASSIGN_OR_RETURN(auto image, ImagePath(options));
    LITERT_ASSIGN_OR_RETURN(pixels, imagenet::DecodePreprocess(image, preprocessing, input.shape[1] == 3));
    report["input_sha256"] = imagenet::HashBytes(pixels.data(), pixels.size() * sizeof(float));
    report["image"] = image; report["preprocessing"] = config_json;
    LITERT_ASSIGN_OR_RETURN(report["image_sha256"], imagenet::HashFile(image));
  }
  for (const auto& tolerance : std::vector<std::pair<std::string, double>>{
           {"atol", 1e-4}, {"rtol", 1e-3}, {"max_code_error", 4}, {"mean_code_error", .5}}) {
    LITERT_ASSIGN_OR_RETURN(auto value, options.Number(tolerance.first, tolerance.second));
    report["tolerances"][tolerance.first] = value;
  }
  LITERT_ASSIGN_OR_RETURN(int samples, options.Integer("max_samples", 1));
  if (options.command == "trace" && samples != 1) return Failure("Trace supports exactly one image");
  LITERT_ASSIGN_OR_RETURN(int mb, options.Integer("capture_budget_mb", 256));
  LITERT_ASSIGN_OR_RETURN(int allow, options.Integer("allow_emulation", 0));
  LITERT_ASSIGN_OR_RETURN(int dump, options.Integer("dump_tensors", 0));
  if (mb < 1 || mb > 4096 || (dump != 0 && dump != 1)) return Failure("Invalid capture budget or dump_tensors");
  const size_t budget = static_cast<size_t>(mb) * 1024 * 1024;
  const bool isolated = options.Get("mode", "accumulated") == "isolated";
  if (!isolated && options.Get("mode", "accumulated") != "accumulated") return Failure("Unknown trace mode");
  if (dump && options.Get("output_dir").empty()) return Failure("Tensor dumps require output_dir");
  report["trace_mode"] = full_graph ? "full_graph" : (isolated ? "isolated_cpu_fed" : "instrumented_prefix");
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
    if (full_graph && index + 1 != graph.operators.size()) continue;
    auto op = *graph.operators[index];
    if (full_graph) op.outputs = graph.outputs;
    std::vector<int32_t> runtime_inputs;
    for (auto id : op.inputs) if (id >= 0 && object->buffers[graph.tensors[id]->buffer]->data.empty()) {
      if (std::find(runtime_inputs.begin(), runtime_inputs.end(), id) == runtime_inputs.end()) runtime_inputs.push_back(id);
    }
    size_t captured = 0;
    auto account = [&](int32_t id) -> bool {
      if (id < 0 || static_cast<size_t>(id) >= graph.tensors.size()) return false;
      const auto& tensor = *graph.tensors[id];
      size_t bytes = TensorWidth(tensor.type);
      if (!bytes) return false;
      for (auto dim : tensor.shape) { if (dim <= 0 || bytes > budget / static_cast<size_t>(dim)) return false; bytes *= dim; }
      const size_t copies = compare ? 3 : 2;
      if (bytes > (budget - captured) / copies) return false;
      captured += bytes * copies; return true;
    };
    for (auto id : op.outputs) if (!account(id)) return Failure("Captured output exceeds budget or unsupported type/shape");
    const auto& preserved_inputs = isolated ? runtime_inputs : graph.inputs;
    for (auto id : preserved_inputs) if (!account(id)) return Failure("Captured input exceeds budget or unsupported type/shape");
    tflite::ModelT prefix(*object); prefix.signature_defs.clear();
    auto& sg = *prefix.subgraphs[0]; sg.operators.resize(index + 1); sg.outputs = op.outputs;
    for (auto id : preserved_inputs) if (std::find(sg.outputs.begin(), sg.outputs.end(), id) == sg.outputs.end()) sg.outputs.push_back(id);
    // Use TFLite for owned tensor allocation only when its CPU kernels do not
    // implement the requested types. The original graph still gets compiled.
    tflite::ModelT allocation_model(prefix);
    if (mathematical_reference) allocation_model.subgraphs[0]->operators.clear();
    LITERT_ASSIGN_OR_RETURN(auto cpu_bytes, PackModel(allocation_model));
    auto flat = tflite::FlatBufferModel::BuildFromBuffer(reinterpret_cast<const char*>(cpu_bytes.data()), cpu_bytes.size());
    if (!flat) return Failure("Cannot build CPU prefix");
    tflite::ops::builtin::BuiltinRefOpResolver resolver; std::unique_ptr<tflite::Interpreter> cpu;
    if (tflite::InterpreterBuilder(*flat, resolver)(&cpu) != kTfLiteOk || !cpu || cpu->AllocateTensors() != kTfLiteOk)
      return Failure("Cannot allocate CPU reference prefix");
    cpu->SetNumThreads(1);
    if (raw) {
      for (size_t i = 0; i < raw_inputs.size(); ++i) {
        auto* tensor = cpu->tensor(graph.inputs[i]);
        if ((tensor->bytes+3)/4 != raw_inputs[i].size()) return Failure("CPU input shape mismatch");
        std::memcpy(tensor->data.raw, raw_inputs[i].data(), tensor->bytes);
      }
    } else { LITERT_RETURN_IF_ERROR(FillImageInput(*cpu, pixels)); }
    std::vector<int8_t> mathematical_output;
    if (external_reference) {
      auto* input = cpu->tensor(graph.inputs[0]); auto* output = cpu->tensor(graph.outputs[0]);
      auto* softmax = graph.operators[0]->builtin_options.AsSoftmaxOptions();
      if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
          (!mathematical_reference && (!softmax ||
          output->params.scale != 1.0f / 256 || output->params.zero_point != -128 ||
          !input->dims || input->dims->size < 1))) return Failure("Invalid alternate-reference tensor types/quantization");
      const auto* bytes = input->data.int8;
      std::vector<int8_t> codes(bytes, bytes + input->bytes);
      auto reference_output = mathematical_reference ? QuantizedReference(*object, codes) : SoftmaxReference(
          codes, static_cast<size_t>(input->dims->data[input->dims->size - 1]),
          input->params.scale, softmax->beta, reference_kind);
      if (!reference_output) {
        report["reference_status"] =
            reference_output.Error().Status() == kLiteRtStatusErrorUnsupported
                ? "unavailable" : "failed";
        return reference_output.Error();
      }
      report["reference_status"] = "available";
      if (reference_output->size() != output->bytes)
        return Failure("Reference output byte count mismatch");
      // An operator-free TFLite graph does not allocate unproduced outputs.
      // Keep the independent oracle's output in owned storage instead.
      if (mathematical_reference) mathematical_output = std::move(*reference_output);
      else std::memcpy(output->data.raw, reference_output->data(), reference_output->size());
    }
    if (!compare && !external_reference && cpu->Invoke() != kTfLiteOk) return Failure("CPU reference prefix failed");
    sg.outputs = op.outputs;
    if (isolated) {
      auto selected = std::move(sg.operators[index]); sg.operators.clear(); sg.operators.push_back(std::move(selected));
      sg.inputs = runtime_inputs;
    }
    LITERT_ASSIGN_OR_RETURN(auto vk_bytes, PackModel(prefix));
    LITERT_ASSIGN_OR_RETURN(auto vk_model, ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(vk_bytes.data(), vk_bytes.size())));
    LITERT_ASSIGN_OR_RETURN(auto vk_graph, vk_model.Subgraph(0));
    std::string reason; OptimizationReport optimized_stats;
    auto artifact = CompilePartition(vk_graph, reason, nullptr, optimization_mode, &optimized_stats, softmax_lut);
    if (!artifact) return Failure("Trace compilation: " + reason);
    spvtools::SpirvTools validator(SPV_ENV_UNIVERSAL_1_6);
    if (!validator.Validate(artifact->spirv)) return Failure("SPIR-V validation failed");
    report["spirv_valid"] = true;
    auto invocation = VulkanInvocation::Create(artifact, allow != 0, reason);
    if (!invocation) return Failure("Trace device/pipeline: " + reason);
    std::vector<const void*> inputs;
    for (auto id : sg.inputs) inputs.push_back(cpu->tensor(id)->data.raw);
    std::vector<std::vector<float>> actual(op.outputs.size()); std::vector<void*> outputs;
    for (size_t j = 0; j < op.outputs.size(); ++j) {
      size_t bytes = cpu->tensor(op.outputs[j])->bytes;
      actual[j].resize((bytes + sizeof(float) - 1) / sizeof(float)); outputs.push_back(actual[j].data());
    }
    auto inference_start = std::chrono::steady_clock::now();
    if (!invocation->Run(inputs, outputs, reason)) return Failure("Trace execution: " + reason);
    double inference_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - inference_start).count();
    std::vector<std::vector<float>> off_outputs;
    if (compare) {
      OptimizationReport off_stats;
      auto off = CompilePartition(vk_graph, reason, nullptr, kLrtArmVulkanMLOptimizationOff, &off_stats, softmax_lut);
      if (!off || !validator.Validate(off->spirv)) return Failure("Off-mode preflight: " + reason);
      auto baseline = VulkanInvocation::Create(off, allow != 0, reason);
      if (!baseline) return Failure("Off-mode pipeline: " + reason);
      std::vector<void*> pointers;
      off_outputs.resize(actual.size());
      for (size_t j = 0; j < actual.size(); ++j) {
        off_outputs[j].resize(actual[j].size()); pointers.push_back(off_outputs[j].data());
      }
      auto off_start = std::chrono::steady_clock::now();
      if (!baseline->Run(inputs, pointers, reason)) return Failure("Off-mode execution: " + reason);
      report["comparison_scope"] = "same backend: off versus selected optimization mode";
      report["inference_ms"] = {{"off", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - off_start).count()}, {"optimized", inference_ms}};
      report["graphs"] = {{"off", {{"emitted_operators", off_stats.emitted_operators}, {"constant_count", off_stats.constant_count}, {"constant_bytes", off_stats.constant_bytes}, {"compile_ms", off_stats.compile_ms}}},
          {"optimized", {{"emitted_operators", optimized_stats.emitted_operators}, {"constant_count", optimized_stats.constant_count}, {"constant_bytes", optimized_stats.constant_bytes}, {"compile_ms", optimized_stats.compile_ms}}}};
    }
    auto copy_stats=invocation->Statistics();
    Json row = {{"index", index}, {"name", tflite::EnumNameBuiltinOperator(object->operator_codes[op.opcode_index]->builtin_code)},
        {"outputs", Json::array()}, {"captured_bytes", captured}};
    row["buffer_io"]={{"input_copy_bytes",copy_stats.input_bytes},{"output_copy_bytes",copy_stats.output_bytes},{"imported_slots",copy_stats.imported_slots}};
    bool passed = true;
    for (size_t j = 0; j < op.outputs.size(); ++j) {
      const auto* expected = cpu->tensor(op.outputs[j]); bool floating = expected->type == kTfLiteFloat32 || expected->type == kTfLiteComplex64;
      Json comparison;
      const void* reference = compare ? static_cast<const void*>(off_outputs[j].data()) :
          mathematical_reference ? static_cast<const void*>(mathematical_output.data()) : expected->data.raw;
      Options gates = options;
      if (compare) {
        gates.values["max_code_error"] = optimization_mode == kLrtArmVulkanMLOptimizationExact ? "0" : "2";
        gates.values["mean_code_error"] = optimization_mode == kLrtArmVulkanMLOptimizationExact ? "0" : "0.5";
      }
      if(expected->type==kTfLiteInt32 || expected->type==kTfLiteUInt32 || expected->type==kTfLiteBool){
        comparison={{"passed",std::memcmp(actual[j].data(),reference,expected->bytes)==0},{"comparison","exact"}};
      } else LITERT_RETURN_IF_ERROR(Compare(actual[j].data(), reference, expected->bytes / (floating ? 4 : 1), floating, gates, comparison));
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
        for (bool dump_reference : {false, true}) {
          auto name = base.string() + (dump_reference ? (compare ? ".off.bin" : ".cpu.bin") : ".vulkan.bin");
          std::ofstream file(name, std::ios::binary);
          file.write(dump_reference ? static_cast<const char*>(reference) : reinterpret_cast<const char*>(actual[j].data()), expected->bytes); file.flush();
          if (!file) return Failure("Cannot write tensor dump");
        }
      }
      row["outputs"].push_back(std::move(comparison));
    }
    report["operators"].push_back(std::move(row));
    if (!passed) { if (options.command == "trace") report["first_divergence"] = index; return Failure("Output exceeds tolerance"); }
  }
  report["complete"] = true;
  return {};
}
}  // namespace litert::arm_vulkan_ml::tool
