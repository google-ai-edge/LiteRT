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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <set>
#include <vector>
#include <utility>
#include <string>
#include <system_error>

#include "litert/cc/internal/litert_extended_model.h"
#include "litert/c/litert_common.h"
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/vendors/arm_vulkan_ml/compiler/graph_ir.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "litert/vendors/arm_vulkan_ml/compiler/softmax.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_invocation.h"
#include "litert/vendors/arm_vulkan_ml/tools/devtool/tool.h"
#include "spirv-tools/libspirv.h"  // from @arm_vulkan_ml_tool_spirv_tools
#include "spirv-tools/libspirv.hpp"  // from @arm_vulkan_ml_tool_spirv_tools
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register_ref.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
namespace litert::arm_vulkan_ml::tool {
namespace {
Error Failure(const std::string& text) {
  return Error(kLiteRtStatusErrorRuntimeFailure, text);
}
size_t TensorWidth(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_INT8:
    case tflite::TensorType_BOOL:
      return 1;
    case tflite::TensorType_INT16:
      return 2;
    case tflite::TensorType_INT64:
      return 8;
    case tflite::TensorType_INT32:
    case tflite::TensorType_UINT32:
    case tflite::TensorType_FLOAT32:
      return 4;
    case tflite::TensorType_COMPLEX64:
      return 8;
    default:
      return 0;
  }
}
// Use the application API for whole-model validation, including core composite
// processing and the dispatch plugin's actual tensor-buffer contracts.
Expected<void> ExecuteRuntime(const Options& options, tflite::Interpreter& cpu,
                              const tflite::SubGraphT& graph,
                              std::vector<std::vector<float>>& actual,
                              Json& report) {
  const auto compiler = CompilerDirectory(options);
  const auto default_dispatch =
      std::filesystem::exists(std::filesystem::path(compiler) /
                              "libLiteRtDispatch_ArmVulkanML.so")
          ? compiler
          : (std::filesystem::path(compiler).parent_path() / "dispatch")
                .string();
  const auto dispatch = options.Get("dispatch_library_dir", default_dispatch);
  std::vector<EnvironmentOptions::Option> environment_options{
      {EnvironmentOptions::Tag::kCompilerPluginLibraryDir, compiler},
      {EnvironmentOptions::Tag::kDispatchLibraryDir, dispatch}};
  LITERT_ASSIGN_OR_RETURN(
      auto environment,
      Environment::Create(EnvironmentOptions(environment_options)));
  LITERT_ASSIGN_OR_RETURN(auto runtime_options, RuntimeOptions(options));
  LITERT_ASSIGN_OR_RETURN(
      auto compiled, CompiledModel::Create(environment, options.Get("model"),
                                           runtime_options));
  LITERT_ASSIGN_OR_RETURN(auto fully, compiled.IsFullyAccelerated());
  report["fully_accelerated"] = fully;
  report["execution_path"] =
      "LiteRT CompiledModel with compiler and dispatch plugins";
  LITERT_ASSIGN_OR_RETURN(auto require,
                          options.Integer("require_full_acceleration", 1));
  if (require != 0 && require != 1)
    return Failure("require_full_acceleration must be 0 or 1");
  if (require && !fully) return Failure("Model is not fully accelerated");
  LITERT_ASSIGN_OR_RETURN(auto inputs, compiled.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(auto outputs, compiled.CreateOutputBuffers());
  if (inputs.size() != graph.inputs.size() ||
      outputs.size() != graph.outputs.size())
    return Failure("Runtime interface count mismatch");
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto* tensor = cpu.tensor(graph.inputs[i]);
    LITERT_ASSIGN_OR_RETURN(auto packed, inputs[i].PackedSize());
    if (packed != tensor->bytes)
      return Failure("Runtime input byte count mismatch");
    LITERT_ASSIGN_OR_RETURN(auto lock,
                            TensorBufferScopedLock::Create(
                                inputs[i], TensorBuffer::LockMode::kWrite));
    std::memcpy(lock.second, tensor->data.raw, tensor->bytes);
  }
  LITERT_RETURN_IF_ERROR(compiled.Run(inputs, outputs));
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto* tensor = cpu.tensor(graph.outputs[i]);
    actual[i].resize((tensor->bytes + sizeof(float) - 1) / sizeof(float));
    LITERT_ASSIGN_OR_RETURN(auto packed, outputs[i].PackedSize());
    if (packed != tensor->bytes)
      return Failure("Runtime output byte count mismatch");
    LITERT_ASSIGN_OR_RETURN(auto lock,
                            TensorBufferScopedLock::Create(
                                outputs[i], TensorBuffer::LockMode::kRead));
    std::memcpy(actual[i].data(), lock.second, tensor->bytes);
  }
  return {};
}
}  // namespace
Expected<void> Trace(const Options& options, Json& report) {
  LITERT_ASSIGN_OR_RETURN(auto softmax_lut, options.SoftmaxLut());
  LITERT_ASSIGN_OR_RETURN(auto rescale_bits, options.RescaleMultiplierBits());
  if (options.Get("tensor_inputs").empty())
    return Failure(options.command + " requires --tensor_inputs");
  const auto reference_kind = options.Get("reference", "tflite");
  const bool external_reference = reference_kind != "tflite";
  if (external_reference &&
      (options.command != "validate" ||
       (reference_kind != "fp64" && reference_kind != "xnnpack")))
    return Failure("Invalid alternate reference");
  LITERT_ASSIGN_OR_RETURN(auto object, LoadModel(options.Get("model")));
  if (object->subgraphs.empty() || object->subgraphs[0]->inputs.empty())
    return Failure("Execution requires a main subgraph with data inputs");
  report["model"] = options.Get("model");
  LITERT_ASSIGN_OR_RETURN(report["model_sha256"],
                          HashFile(options.Get("model")));
  const auto& graph = *object->subgraphs[0];
  if (graph.operators.empty())
    return Failure("Empty execution graph unsupported");
  const bool full_graph = options.command == "validate";
  if (full_graph && options.values.count("mode"))
    return Failure("mode is only valid for trace");
  if (graph.operators[0]->opcode_index >= object->operator_codes.size())
    return Failure("Invalid operator code index");
  const auto first_code =
      object->operator_codes[graph.operators[0]->opcode_index]->builtin_code;
  const bool mathematical_reference =
      external_reference && first_code != tflite::BuiltinOperator_SOFTMAX;
  if (external_reference &&
      (graph.operators.size() != 1 || graph.inputs.size() != 1 ||
       graph.outputs.size() != 1 ||
       (first_code != tflite::BuiltinOperator_SOFTMAX &&
        (reference_kind != "fp64" ||
         (first_code != tflite::BuiltinOperator_SQUARE &&
          first_code != tflite::BuiltinOperator_CONV_3D)))))
    return Failure(
        "Alternate reference requires raw single-op INT8 SOFTMAX, or fp64 "
        "SQUARE/CONV_3D");
  report["reference"] = reference_kind;
  std::vector<std::vector<float>> raw_inputs(graph.inputs.size());
  LITERT_ASSIGN_OR_RETURN(auto manifest,
                          ReadJson(options.Get("tensor_inputs")));
  if (!manifest.is_object() || !manifest.contains("cases") ||
      !manifest["cases"].is_array())
    return Failure("Tensor manifest requires cases array");
  LITERT_ASSIGN_OR_RETURN(int start, options.Integer("start_sample", 0));
  LITERT_ASSIGN_OR_RETURN(
      int samples,
      options.Integer("max_samples", options.command == "trace" ? 1 : 3));
  if (start < 0 || samples < 1 || samples > 10 ||
      (options.command == "trace" && samples != 1) ||
      static_cast<size_t>(start) > manifest["cases"].size() ||
      static_cast<size_t>(samples) >
          manifest["cases"].size() - static_cast<size_t>(start))
    return Failure("Invalid tensor sample range");
  if (samples > 1) {
    report["cases"] = Json::array();
    bool passed = true;
    for (int i = 0; i < samples; ++i) {
      Options child = options;
      child.values["max_samples"] = "1";
      child.values["start_sample"] = std::to_string(start + i);
      Json row;
      auto status = Trace(child, row);
      row["status"] = status ? "passed" : "failed";
      if (!status) {
        row["error"] = std::string(status.Error().Message());
        passed = false;
      }
      report["cases"].push_back(std::move(row));
    }
    if (!passed) return Failure("Tensor cases failed");
    return {};
  }
  const auto& sample = manifest["cases"][start];
  if (!sample.is_object() || !sample.contains("id") ||
      !sample.contains("inputs") || !sample["inputs"].is_array())
    return Failure("Tensor case requires id and inputs");
  report["case_id"] = sample["id"];
  std::set<int> seen;
  for (const auto& entry : sample["inputs"]) {
    if (!entry.is_object() || !entry.contains("index") ||
        !entry["index"].is_number_integer() || !entry.contains("file") ||
        !entry["file"].is_string())
      return Failure("Invalid tensor input entry");
    if (entry["index"] < 0 || entry["index"] >= raw_inputs.size())
      return Failure("Unknown input index");
    int slot = entry["index"].get<int>();
    if (slot < 0 || static_cast<size_t>(slot) >= raw_inputs.size() ||
        !seen.insert(slot).second)
      return Failure("Unknown or duplicate input index");
    const auto& tensor = *graph.tensors[graph.inputs[slot]];
    const size_t width = TensorWidth(tensor.type);
    if (!width)
      return Failure(
          "Raw inputs require FP32, INT8, INT16, INT32, INT64, UINT32, BOOL "
          "or COMPLEX64");
    size_t count = 1;
    for (auto dim : tensor.shape) {
      if (dim <= 0 ||
          count > (256u * 1024u * 1024u / width) / static_cast<size_t>(dim))
        return Failure("Invalid/oversized raw shape");
      count *= dim;
    }
    for (auto dim : tensor.shape_signature)
      if (dim < 0) return Failure("Dynamic raw input unsupported");
    std::filesystem::path path(entry["file"].get<std::string>());
    if (path.is_relative())
      path =
          std::filesystem::path(options.Get("tensor_inputs")).parent_path() /
          path;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * width))
      return Failure("Raw input byte count mismatch");
    raw_inputs[slot].resize((count * width + 3) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(raw_inputs[slot].data()), count * width);
    if (!file) return Failure("Cannot read raw input");
    LITERT_ASSIGN_OR_RETURN(auto hash, HashFile(path.string()));
    report["inputs"].push_back({{"index", slot}, {"sha256", hash}});
  }
  if (seen.size() != raw_inputs.size()) return Failure("Missing raw input");
  for (const auto& tolerance :
       std::vector<std::pair<std::string, double>>{{"atol", 1e-4},
                                                   {"rtol", 1e-3},
                                                   {"max_code_error", 4},
                                                   {"mean_code_error", .5}}) {
    LITERT_ASSIGN_OR_RETURN(auto value,
                            options.Number(tolerance.first, tolerance.second));
    report["tolerances"][tolerance.first] = value;
  }
  LITERT_ASSIGN_OR_RETURN(int mb, options.Integer("capture_budget_mb", 256));
  LITERT_ASSIGN_OR_RETURN(int allow, options.Integer("allow_emulation", 0));
  LITERT_ASSIGN_OR_RETURN(int dump, options.Integer("dump_tensors", 0));
  if (mb < 1 || mb > 4096 || (dump != 0 && dump != 1))
    return Failure("Invalid capture budget or dump_tensors");
  const size_t budget = static_cast<size_t>(mb) * 1024 * 1024;
  const bool isolated = options.Get("mode", "accumulated") == "isolated";
  if (!isolated && options.Get("mode", "accumulated") != "accumulated")
    return Failure("Unknown trace mode");
  if (dump && options.Get("output_dir").empty())
    return Failure("Tensor dumps require output_dir");
  report["trace_mode"] =
      full_graph ? "full_graph"
                 : (isolated ? "isolated_cpu_fed" : "instrumented_prefix");
  report["capture_budget_bytes"] = budget;
  report["budget_scope"] =
      "captured tensors; excludes model weights and execution workspace";
  report["instrumentation_note"] =
      "Exposing outputs may change fusion/optimization; not an uninstrumented "
      "driver trace";
  report["operators"] = Json::array();
  // Analyze the complete original graph before executing any prefix.
  LITERT_ASSIGN_OR_RETURN(auto original_bytes, PackModel(*object));
  LITERT_ASSIGN_OR_RETURN(auto original,
                          ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(
                              original_bytes.data(), original_bytes.size())));
  LITERT_ASSIGN_OR_RETURN(auto original_graph, original.Subgraph(0));
  if (options.command != "validate")
    for (const auto& op : original_graph.Ops()) {
      std::string reason;
      if (!AnalyzeOperation(op, reason, softmax_lut))
        return Failure("Trace requires complete coverage: " + reason);
    }
  for (size_t index = 0; index < graph.operators.size(); ++index) {
    report["last_attempted_operator"] = index;
    if (full_graph && index + 1 != graph.operators.size()) continue;
    auto op = *graph.operators[index];
    if (full_graph) op.outputs = graph.outputs;
    std::vector<int32_t> runtime_inputs;
    for (auto id : op.inputs)
      if (id >= 0 && object->buffers[graph.tensors[id]->buffer]->data.empty()) {
        if (absl::c_find(runtime_inputs, id) ==
            runtime_inputs.end())
          runtime_inputs.push_back(id);
      }
    size_t captured = 0;
    auto account = [&](int32_t id) -> bool {
      if (id < 0 || static_cast<size_t>(id) >= graph.tensors.size())
        return false;
      const auto& tensor = *graph.tensors[id];
      size_t bytes = TensorWidth(tensor.type);
      if (!bytes) return false;
      for (auto dim : tensor.shape) {
        if (dim <= 0 || bytes > budget / static_cast<size_t>(dim)) return false;
        bytes *= dim;
      }
      if (bytes > (budget - captured) / 2) return false;
      captured += bytes * 2;
      return true;
    };
    for (auto id : op.outputs)
      if (!account(id))
        return Failure(
            "Captured output exceeds budget or unsupported type/shape");
    const auto& preserved_inputs = isolated ? runtime_inputs : graph.inputs;
    for (auto id : preserved_inputs)
      if (!account(id))
        return Failure(
            "Captured input exceeds budget or unsupported type/shape");
    tflite::ModelT prefix(*object);
    prefix.signature_defs.clear();
    auto& sg = *prefix.subgraphs[0];
    sg.operators.resize(index + 1);
    sg.outputs = op.outputs;
    for (auto id : preserved_inputs)
      if (std::find(sg.outputs.begin(), sg.outputs.end(), id) ==
          sg.outputs.end())
        sg.outputs.push_back(id);
    // Use TFLite for owned tensor allocation only when its CPU kernels do not
    // implement the requested types. The original graph still gets compiled.
    tflite::ModelT allocation_model(prefix);
    if (mathematical_reference)
      allocation_model.subgraphs[0]->operators.clear();
    LITERT_ASSIGN_OR_RETURN(auto cpu_bytes, PackModel(allocation_model));
    auto flat = tflite::FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(cpu_bytes.data()), cpu_bytes.size());
    if (!flat) return Failure("Cannot build CPU prefix");
    tflite::ops::builtin::BuiltinRefOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> cpu;
    if (tflite::InterpreterBuilder(*flat, resolver)(&cpu) != kTfLiteOk ||
        !cpu || cpu->AllocateTensors() != kTfLiteOk)
      return Failure("Cannot allocate CPU reference prefix");
    cpu->SetNumThreads(1);
    for (size_t i = 0; i < raw_inputs.size(); ++i) {
      auto* tensor = cpu->tensor(graph.inputs[i]);
      if ((tensor->bytes + 3) / 4 != raw_inputs[i].size())
        return Failure("CPU input shape mismatch");
      std::memcpy(tensor->data.raw, raw_inputs[i].data(), tensor->bytes);
    }
    std::vector<int8_t> mathematical_output;
    if (external_reference) {
      auto* input = cpu->tensor(graph.inputs[0]);
      auto* output = cpu->tensor(graph.outputs[0]);
      auto* softmax = graph.operators[0]->builtin_options.AsSoftmaxOptions();
      if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
          (!mathematical_reference &&
           (!softmax || output->params.scale != 1.0f / 256 ||
            output->params.zero_point != -128 || !input->dims ||
            input->dims->size < 1)))
        return Failure("Invalid alternate-reference tensor types/quantization");
      const auto* bytes = input->data.int8;
      std::vector<int8_t> codes(bytes, bytes + input->bytes);
      auto reference_output =
          mathematical_reference
              ? QuantizedReference(*object, codes)
              : SoftmaxReference(codes,
                                 static_cast<size_t>(
                                     input->dims->data[input->dims->size - 1]),
                                 input->params.scale, softmax->beta,
                                 reference_kind);
      if (!reference_output) {
        report["reference_status"] =
            reference_output.Error().Status() == kLiteRtStatusErrorUnsupported
                ? "unavailable"
                : "failed";
        return reference_output.Error();
      }
      report["reference_status"] = "available";
      if (reference_output->size() != output->bytes)
        return Failure("Reference output byte count mismatch");
      // An operator-free TFLite graph does not allocate unproduced outputs.
      // Keep the independent oracle's output in owned storage instead.
      if (mathematical_reference)
        mathematical_output = std::move(*reference_output);
      else
        std::memcpy(output->data.raw, reference_output->data(),
                    reference_output->size());
    }
    if (!external_reference && cpu->Invoke() != kTfLiteOk)
      return Failure("CPU reference prefix failed");
    sg.outputs = op.outputs;
    if (isolated) {
      auto selected = std::move(sg.operators[index]);
      sg.operators.clear();
      sg.operators.push_back(std::move(selected));
      sg.inputs = runtime_inputs;
    }
    LITERT_ASSIGN_OR_RETURN(auto vk_bytes, PackModel(prefix));
    LITERT_ASSIGN_OR_RETURN(auto vk_model,
                            ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(
                                vk_bytes.data(), vk_bytes.size())));
    LITERT_ASSIGN_OR_RETURN(auto vk_graph, vk_model.Subgraph(0));
    std::string reason;
    OptimizationReport optimized_stats;
    spvtools::SpirvTools validator(SPV_ENV_UNIVERSAL_1_6);
    std::unique_ptr<VulkanInvocation> invocation;
    std::vector<std::vector<float>> actual(op.outputs.size());
    const bool direct_compile =
        options.command != "validate" ||
        softmax_lut != SoftmaxLutType::kAuto || rescale_bits != 32;
    if (!direct_compile) {
      LITERT_RETURN_IF_ERROR(
          ExecuteRuntime(options, *cpu, graph, actual, report));
    } else {
      auto artifact = CompilePartition(vk_graph, reason, nullptr,
                                       &optimized_stats, softmax_lut,
                                       rescale_bits);
      if (!artifact) return Failure("Trace compilation: " + reason);

      if (!validator.Validate(artifact->spirv))
        return Failure("SPIR-V validation failed");
      report["spirv_valid"] = true;
      if (options.command == "validate") {
        report["fully_accelerated"] = true;
        report["execution_path"] =
            "direct CompilePartition + VulkanInvocation (internal compiler "
            "options)";
      }
      invocation = VulkanInvocation::Create(artifact, allow != 0, reason);
      if (!invocation) return Failure("Trace device/pipeline: " + reason);
      std::vector<const void*> inputs;
      inputs.reserve(sg.inputs.size());
for (auto id : sg.inputs) inputs.push_back(cpu->tensor(id)->data.raw);
      std::vector<void*> outputs;
      for (size_t j = 0; j < op.outputs.size(); ++j) {
        size_t bytes = cpu->tensor(op.outputs[j])->bytes;
        actual[j].resize((bytes + sizeof(float) - 1) / sizeof(float));
        outputs.push_back(actual[j].data());
      }

      if (!invocation->Run(inputs, outputs, reason))
        return Failure("Trace execution: " + reason);
    }

    Json row = {
        {"index", index},
        {"name", tflite::EnumNameBuiltinOperator(
                     object->operator_codes[op.opcode_index]->builtin_code)},
        {"outputs", Json::array()},
        {"captured_bytes", captured}};
    if (invocation) {
      const auto copy_stats = invocation->Statistics();
      row["buffer_io"] = {{"input_copy_bytes", copy_stats.input_bytes},
                          {"output_copy_bytes", copy_stats.output_bytes},
                          {"imported_slots", copy_stats.imported_slots}};
    }
    bool passed = true;
    for (size_t j = 0; j < op.outputs.size(); ++j) {
      const auto* expected = cpu->tensor(op.outputs[j]);
      bool floating = expected->type == kTfLiteFloat32 ||
                      expected->type == kTfLiteComplex64;
      Json comparison;
      const void* reference =
          mathematical_reference
              ? static_cast<const void*>(mathematical_output.data())
              : expected->data.raw;
      if (expected->type == kTfLiteInt64 || expected->type == kTfLiteInt32 ||
          expected->type == kTfLiteUInt32 || expected->type == kTfLiteBool) {
        comparison = {{"passed", std::memcmp(actual[j].data(), reference,
                                             expected->bytes) == 0},
                      {"comparison", "exact"}};
      } else if (expected->type == kTfLiteInt16) {
        const size_t count = expected->bytes / sizeof(int16_t);
        const auto* a = reinterpret_cast<const uint8_t*>(actual[j].data());
        const auto* b = static_cast<const uint8_t*>(reference);
        double maximum = 0, sum = 0;
        for (size_t k = 0; k < count; ++k) {
          int16_t av, bv;
          std::memcpy(&av, a + k * sizeof(av), sizeof(av));
          std::memcpy(&bv, b + k * sizeof(bv), sizeof(bv));
          const double error =
              std::abs(static_cast<int>(av) - static_cast<int>(bv));
          maximum = std::max(maximum, error);
          sum += error;
        }
        LITERT_ASSIGN_OR_RETURN(auto max_code,
                                options.Number("max_code_error", 4));
        LITERT_ASSIGN_OR_RETURN(auto mean_code,
                                options.Number("mean_code_error", .5));
        comparison = {
            {"passed", maximum <= max_code && sum / count <= mean_code},
            {"max_error", maximum},
            {"mean_error", sum / count},
            {"elements", count},
            {"max_dequantized_error", maximum * expected->params.scale}};
      } else
        LITERT_RETURN_IF_ERROR(Compare(actual[j].data(), reference,
                                       expected->bytes / (floating ? 4 : 1),
                                       floating, options, comparison));
      if (options.command == "validate" && floating &&
          expected->dims->size == 2 && expected->dims->data[0] == 1) {
        const auto* a = actual[j].data();
        const auto* b = static_cast<const float*>(reference);
        const size_t count = expected->bytes / sizeof(float);
        const auto ai = std::max_element(a, a + count) - a;
        const auto bi = std::max_element(b, b + count) - b;
        comparison["top1_actual"] = ai;
        comparison["top1_expected"] = bi;
        comparison["top1_agreement"] = ai == bi;
      }
      comparison["tensor"] = op.outputs[j];
      passed &= comparison["passed"].get<bool>();
      const auto& tensor = *graph.tensors[op.outputs[j]];
      comparison["shape"] = tensor.shape;
      comparison["type"] = tflite::EnumNameTensorType(tensor.type);
      if (tensor.quantization) {
        comparison["scales"] = tensor.quantization->scale;
        comparison["zero_points"] = tensor.quantization->zero_point;
        comparison["quantized_dimension"] =
            tensor.quantization->quantized_dimension;
      }
      if (dump && !comparison["passed"].get<bool>()) {
        std::error_code ec;
        std::filesystem::create_directories(options.Get("output_dir"), ec);
        if (ec) return Failure(ec.message());
        auto base = std::filesystem::path(options.Get("output_dir")) /
                    ("operator_" + std::to_string(index) + "_tensor_" +
                     std::to_string(op.outputs[j]));
        for (bool dump_reference : {false, true}) {
          auto name =
              base.string() + (dump_reference ? ".cpu.bin" : ".vulkan.bin");
          std::ofstream file(name, std::ios::binary);
          file.write(dump_reference
                         ? static_cast<const char*>(reference)
                         : reinterpret_cast<const char*>(actual[j].data()),
                     expected->bytes);
          file.flush();
          if (!file) return Failure("Cannot write tensor dump");
        }
      }
      row["outputs"].push_back(std::move(comparison));
    }
    report["operators"].push_back(std::move(row));
    if (!passed) {
      if (options.command == "trace") report["first_divergence"] = index;
      return Failure("Output exceeds tolerance");
    }
  }
  report["complete"] = true;
  return {};
}
}  // namespace litert::arm_vulkan_ml::tool
