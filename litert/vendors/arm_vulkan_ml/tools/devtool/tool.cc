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
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <unordered_map>
#include "absl/strings/numbers.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/tools/imagenet_eval_runner.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"
#include "spirv-tools/libspirv.hpp"
#include "tflite/model_builder.h"
namespace litert::arm_vulkan_ml::tool {
namespace fs = std::filesystem;
namespace {
Error Invalid(const std::string& message) { return Error(kLiteRtStatusErrorInvalidArgument, message); }
Expected<void> Write(const fs::path& path, const void* bytes, size_t size) {
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char*>(bytes), size); file.flush();
  if (!file) return Invalid("Cannot write " + path.string());
  return {};
}
Json TensorInfo(const tflite::TensorT& tensor, int id) {
  Json result = {{"id", id}, {"name", tensor.name}, {"type", tflite::EnumNameTensorType(tensor.type)},
      {"shape", tensor.shape}, {"shape_signature", tensor.shape_signature}};
  if (tensor.quantization) result["quantization"] = {{"scales", tensor.quantization->scale},
      {"zero_points", tensor.quantization->zero_point}, {"dimension", tensor.quantization->quantized_dimension}};
  return result;
}
Expected<void> ImageInterface(const tflite::ModelT& model) {
  if (model.subgraphs.size() != 1) return Invalid("Image execution requires one subgraph");
  const auto& g = *model.subgraphs[0];
  if (g.inputs.size() != 1 || g.outputs.size() != 1) return Invalid("Image execution requires one input and one output");
  auto valid = [&](int id) { return id >= 0 && static_cast<size_t>(id) < g.tensors.size(); };
  if (!valid(g.inputs[0]) || !valid(g.outputs[0])) return Invalid("Invalid model interface");
  const auto& in = *g.tensors[g.inputs[0]]; const auto& out = *g.tensors[g.outputs[0]];
  for (const auto* t : {&in, &out}) {
    if (t->type != tflite::TensorType_FLOAT32 && t->type != tflite::TensorType_INT8)
      return Invalid("Image execution supports FP32/INT8 tensors only");
    for (auto dim : t->shape) if (dim <= 0) return Invalid("Dynamic/empty image tensors unsupported");
    for (auto dim : t->shape_signature) if (dim < 0) return Invalid("Dynamic image signatures unsupported");
  }
  if (in.shape.size() != 4 || in.shape[0] != 1 || ((in.shape[1] == 3) == (in.shape[3] == 3)))
    return Invalid("Image input must be unambiguous NCHW/NHWC RGB with batch one");
  if (out.shape.size() != 2 || out.shape[0] != 1 || out.shape[1] < 5)
    return Invalid("Classification output must be [1, classes] with at least five classes");
  return {};
}
}  // namespace
std::string Options::Get(const std::string& key, const std::string& fallback) const {
  auto it = values.find(key); return it == values.end() ? fallback : it->second;
}
Expected<int> Options::Integer(const std::string& key, int fallback) const {
  auto it = values.find(key); if (it == values.end()) return fallback;
  int value; if (!absl::SimpleAtoi(it->second, &value)) return Invalid("Invalid integer --" + key);
  return value;
}
Expected<double> Options::Number(const std::string& key, double fallback) const {
  auto it = values.find(key); if (it == values.end()) return fallback;
  double value; if (!absl::SimpleAtod(it->second, &value) || !std::isfinite(value) || value < 0)
    return Invalid("Invalid nonnegative number --" + key);
  return value;
}
Expected<Options> ParseOptions(int argc, char** argv) {
  Options options;
  if (argc < 2) return Invalid("Expected inspect, capabilities, compile, validate, or trace; use --help");
  options.command = argv[1];
  const std::set<std::string> commands = {"inspect", "capabilities", "compile", "validate", "trace", "--help"};
  if (!commands.count(options.command)) return Invalid("Unknown command: " + options.command);
  const std::set<std::string> keys = {"model", "report_json", "output_dir", "manifest", "preprocessing_config",
      "image_root", "start_sample", "max_samples", "atol", "rtol", "max_code_error", "mean_code_error",
      "compiler_plugin_library_dir", "dispatch_library_dir", "runtime_library_dir", "allow_emulation",
      "mode", "capture_budget_mb", "suite", "dump_tensors"};
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i]; if (arg.rfind("--", 0) != 0) return Invalid("Expected --option: " + arg);
    auto equals = arg.find('='); std::string key = arg.substr(2, equals == std::string::npos ? equals : equals - 2);
    std::string value;
    if (equals != std::string::npos) value = arg.substr(equals + 1);
    else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) value = argv[++i];
    else return Invalid("Missing value for --" + key);
    if (!keys.count(key) || value.empty() || !options.values.emplace(key, value).second)
      return Invalid("Unknown, empty, or repeated option --" + key);
  }
  return options;
}
Expected<std::unique_ptr<tflite::ModelT>> LoadModel(const std::string& path) {
  auto flat = tflite::FlatBufferModel::VerifyAndBuildFromFile(path.c_str());
  if (!flat) return Invalid("Cannot load TFLite model: " + path);
  std::ifstream file(path, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
  if (file.bad()) return Invalid("Cannot read model: " + path);
  std::unique_ptr<tflite::ModelT> model(flat->GetModel()->UnPack());
  for (auto& buffer : model->buffers) if (buffer->size) {
    if (buffer->offset > bytes.size() || buffer->size > bytes.size() - buffer->offset)
      return Invalid("External constant buffer exceeds model file");
    buffer->data.assign(bytes.begin() + buffer->offset, bytes.begin() + buffer->offset + buffer->size);
    buffer->offset = buffer->size = 0;
  }
  return model;
}
Expected<std::vector<uint8_t>> PackModel(const tflite::ModelT& model) {
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}
Expected<void> Inspect(const Options& options, Json& report, bool compile) {
  LITERT_ASSIGN_OR_RETURN(auto object, LoadModel(options.Get("model")));
  report["subgraphs"] = Json::array(); bool complete = true;
  fs::path output = options.Get("output_dir");
  if (compile) {
    if (output.empty()) return Invalid("compile requires --output_dir");
    std::error_code ec; fs::create_directories(output, ec); if (ec) return Invalid(ec.message());
  }
  for (size_t sg = 0; sg < object->subgraphs.size(); ++sg) {
    tflite::ModelT extracted(*object);
    auto graph = std::move(extracted.subgraphs[sg]); extracted.subgraphs.clear();
    extracted.subgraphs.push_back(std::move(graph)); extracted.signature_defs.clear();
    LITERT_ASSIGN_OR_RETURN(auto bytes, PackModel(extracted));
    LITERT_ASSIGN_OR_RETURN(auto model, ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(bytes.data(), bytes.size())));
    LITERT_ASSIGN_OR_RETURN(auto subgraph, model.Subgraph(0));
    Json entry = {{"index", sg}, {"operators", Json::array()}, {"tensors", Json::array()}, {"partitions", Json::array()}};
    const auto& original = *extracted.subgraphs[0];
    for (size_t i = 0; i < original.tensors.size(); ++i) entry["tensors"].push_back(TensorInfo(*original.tensors[i], i));
    std::vector<LiteRtOpWithPartitionIndex> selected;
    std::unordered_map<std::string, size_t> origins;
    size_t index = 0;
    for (const auto& op : subgraph.Ops()) {
      std::string reason; bool supported = static_cast<bool>(AnalyzeOperation(op, reason));
      entry["operators"].push_back({{"index", index}, {"opcode", static_cast<int>(op.Code())},
          {"name", tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(op.Code()))},
          {"inputs", original.operators[index]->inputs}, {"outputs", original.operators[index]->outputs},
          {"supported", supported}, {"reason", reason}});
      // The partitioner clones ops. Tag copied SSA outputs to retain origin IDs
      // through cloning and topological reordering, without changing the file.
      for (auto* tensor : op.Get()->Outputs()) {
        std::string tag = "litert_devtool_op_" + std::to_string(index);
        tensor->SetName(tag);
        origins.emplace(tag, index);
      }
      ++index;
      if (supported) selected.push_back({op.Get(), 0}); else complete = false;
    }
    entry["selected_operators"] = selected.size();
    if (!selected.empty()) {
      LITERT_ASSIGN_OR_RETURN(auto partitioned, internal::PartitionModelDirect(std::move(selected), *model.Get()));
      size_t pi = 0;
      for (auto* part : partitioned.second.Subgraphs()) {
        Json item = {{"index", pi}, {"operators", Json::array()}};
        for (auto* op : part->Ops()) {
          if (op->Outputs().empty()) return Invalid("Partition operator has no output origin");
          auto origin = origins.find(std::string(op->Outputs().front()->Name()));
          if (origin == origins.end()) return Invalid("Partition lost operator origin");
          item["operators"].push_back(origin->second);
        }
        if (compile) {
          std::string reason; std::vector<LoweringRecord> mapping;
          auto compiled = CompilePartition(Subgraph(part), reason, &mapping);
          if (!compiled) return Invalid("Partition compilation failed: " + reason);
          std::string diagnostics; spvtools::SpirvTools spirv(SPV_ENV_UNIVERSAL_1_6);
          spirv.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& pos, const char* message) {
            diagnostics += std::to_string(pos.index) + ": " + message + "\n";
          });
          if (!spirv.Validate(compiled->spirv)) return Invalid("SPIR-V validation failed: " + diagnostics);
          std::string assembly;
          if (!spirv.Disassemble(compiled->spirv, &assembly, SPV_BINARY_TO_TEXT_OPTION_INDENT))
            return Invalid("SPIR-V disassembly failed: " + diagnostics);
          std::string stem = "subgraph_" + std::to_string(sg) + "_partition_" + std::to_string(pi);
          LITERT_RETURN_IF_ERROR(Write(output / (stem + ".spv"), compiled->spirv.data(), compiled->spirv.size() * sizeof(uint32_t)));
          LITERT_RETURN_IF_ERROR(Write(output / (stem + ".spvasm"), assembly.data(), assembly.size()));
          item["profiles"] = compiled->profiles; item["extensions"] = compiled->extensions;
          item["instruction_set"] = compiled->instruction_set;
          item["entry_point"] = compiled->entry_point;
          item["spirv_file"] = stem + ".spv";
          item["assembly_file"] = stem + ".spvasm";
          auto descriptors = [](const auto& tensors) {
            Json result = Json::array();
            for (const auto& tensor : tensors) {
              result.push_back({{"shape", tensor.shape}, {"binding", tensor.binding},
                  {"element", static_cast<int>(tensor.element)}});
            }
            return result;
          };
          item["inputs"] = descriptors(compiled->inputs);
          item["outputs"] = descriptors(compiled->outputs);
          item["spirv_valid"] = true; item["lowering"] = Json::array(); item["constants"] = Json::array();
          for (const auto& row : mapping) item["lowering"].push_back({{"operator", item["operators"][row.operator_index]}, {"created_result_ids", row.result_ids}});
          size_t ci = 0;
          for (const auto& constant : compiled->constants) {
            auto name = stem + "_constant_" + std::to_string(ci++) + ".bin";
            LITERT_RETURN_IF_ERROR(Write(output / name, constant.bytes.data(), constant.bytes.size()));
            item["constants"].push_back({{"file", name}, {"shape", constant.descriptor.shape},
                {"element", static_cast<int>(constant.descriptor.element)}, {"binding", constant.descriptor.binding}});
          }
          item["artifact_kind"] = "diagnostic_export_not_portable_executable";
          LITERT_RETURN_IF_ERROR(imagenet::WriteJson((output / (stem + ".json")).string(), item));
        }
        entry["partitions"].push_back(std::move(item)); ++pi;
      }
    }
    report["subgraphs"].push_back(std::move(entry));
  }
  report["complete_coverage"] = complete;
  report["coverage_status"] = complete ? "complete" : "partial";
  return {};
}
Expected<void> Capabilities(const Options& options, Json& report) {
  struct Loader {
    void* library = nullptr; VkInstance instance = VK_NULL_HANDLE; PFN_vkDestroyInstance destroy = nullptr;
    ~Loader() { if (instance && destroy) destroy(instance, nullptr); if (library) dlclose(library); }
  } loader;
  report["loader"] = VulkanLibraryName(); report["devices"] = Json::array();
  loader.library = dlopen(VulkanLibraryName(), RTLD_NOW | RTLD_LOCAL);
  if (!loader.library) { const char* detail = dlerror(); return Invalid(detail ? detail : "Cannot load Vulkan"); }
  auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(loader.library, "vkGetInstanceProcAddr"));
  if (!get) return Invalid("Missing vkGetInstanceProcAddr");
  auto create = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
  if (!create) return Invalid("Missing vkCreateInstance");
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; info.pApplicationInfo = &app;
  VkResult result = create(&info, nullptr, &loader.instance);
  if (result != VK_SUCCESS) return Invalid("vkCreateInstance returned " + std::to_string(result));
  loader.destroy = reinterpret_cast<PFN_vkDestroyInstance>(get(loader.instance, "vkDestroyInstance"));
  auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(get(loader.instance, "vkEnumeratePhysicalDevices"));
  if (!enumerate) return Invalid("Missing physical-device enumeration");
  uint32_t count = 0;
  if (enumerate(loader.instance, &count, nullptr) != VK_SUCCESS) return Invalid("Device enumeration failed");
  std::vector<VkPhysicalDevice> devices(count);
  if (enumerate(loader.instance, &count, devices.data()) != VK_SUCCESS) return Invalid("Device enumeration failed");
  std::shared_ptr<const CompiledPartition> artifact;
  if (!options.Get("model").empty()) {
    LITERT_ASSIGN_OR_RETURN(auto object, LoadModel(options.Get("model")));
    if (object->subgraphs.size() != 1) return Invalid("Capability matching requires one subgraph");
    LITERT_ASSIGN_OR_RETURN(auto bytes, PackModel(*object));
    LITERT_ASSIGN_OR_RETURN(auto model, ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(bytes.data(), bytes.size())));
    LITERT_ASSIGN_OR_RETURN(auto graph, model.Subgraph(0));
    std::string error; artifact = CompilePartition(graph, error);
    if (!artifact) return Invalid(error);
  }
  LITERT_ASSIGN_OR_RETURN(int allow, options.Integer("allow_emulation", 0));
  bool any_match = false;
  for (auto physical : devices) {
    DeviceCapabilities caps; std::string error;
    if (!QueryDeviceCapabilities(get, loader.instance, physical, caps, error)) return Invalid(error);
    Json device = {{"name", caps.properties.deviceName}, {"api_version", caps.properties.apiVersion},
        {"driver_version", caps.properties.driverVersion}, {"vendor_id", caps.properties.vendorID},
        {"device_id", caps.properties.deviceID}, {"device_type", caps.properties.deviceType}, {"extensions", Json::array()}, {"engines", Json::array()}, {"formats", Json::array()}};
    for (const auto& e : caps.extensions) device["extensions"].push_back({{"name", e.extensionName}, {"revision", e.specVersion}});
    device["features"] = {{"tensors", caps.tensors_feature}, {"data_graph", caps.graph_feature},
        {"synchronization2", caps.synchronization2}, {"maintenance5", caps.maintenance5},
        {"replicated_composites", caps.replicated_composites}};
    device["tensor_limits"] = {{"rank", caps.tensors.maxTensorDimensionCount}, {"elements", caps.tensors.maxTensorElements},
        {"dimension_elements", caps.tensors.maxPerDimensionTensorElements}, {"bytes", caps.tensors.maxTensorSize},
        {"descriptors", caps.tensors.maxDescriptorSetStorageTensors}};
    for (const auto& f : caps.formats) device["formats"].push_back({{"format", f.first}, {"linear_features", f.second}});
    bool required = CheckDeviceRequirements(caps, error); device["required_extensions"] = required;
    if (!required) device["rejection"] = error;
    for (const auto& engine : caps.engines) {
      Json e = {{"queue_family", engine.queue_family}, {"instruction_set", engine.instruction_set},
          {"level", engine.level}, {"level_constraints", "unchecked"}, {"profiles", Json::array()}, {"extensions", Json::array()}};
      for (const auto& p : engine.profiles) e["profiles"].push_back({{"name", p.name}, {"quality", p.quality}});
      for (const auto& p : engine.extensions) e["extensions"].push_back({{"name", p.name}, {"quality", p.quality}});
      if (artifact) {
        bool matches = required && CheckEngineRequirements(*artifact, caps, engine, allow != 0, error);
        if (matches) {
          auto props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(get(loader.instance, "vkGetPhysicalDeviceProperties2"));
          auto formats = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties2>(get(loader.instance, "vkGetPhysicalDeviceFormatProperties2"));
          matches = props && formats && QueryTensorCapabilities(physical, *artifact, props, formats, error);
        }
        e["reported_requirements_match"] = matches; if (!matches) e["rejection"] = error;
        any_match |= matches;
      }
      device["engines"].push_back(std::move(e));
    }
    report["devices"].push_back(std::move(device));
  }
  report["pipeline_acceptance"] = "not_tested";
  if (artifact && !any_match) return Invalid("No device matches the model's reported requirements");
  return {};
}
Expected<void> Validate(const Options& options, Json& report) {
  if (!options.Get("suite").empty()) {
    LITERT_ASSIGN_OR_RETURN(auto suite, imagenet::ReadJson(options.Get("suite")));
    if (!suite.is_array() || suite.empty()) return Invalid("Suite must be a nonempty array");
    report["models"] = Json::array(); bool passed = true;
    for (const auto& entry : suite) {
      if (!entry.is_object() || !entry.contains("model") || !entry["model"].is_string() ||
          !entry.contains("sha256") || !entry["sha256"].is_string()) return Invalid("Suite entries require model and sha256 strings");
      Options child = options; child.values.erase("suite");
      for (auto it = entry.begin(); it != entry.end(); ++it) {
        if (it.key() == "sha256" || it.key() == "known_failure") continue;
        const std::set<std::string> allowed = {"model", "manifest", "image_root", "preprocessing_config", "start_sample", "max_samples"};
        if (!allowed.count(it.key())) return Invalid("Unknown suite field " + it.key());
        if (!it.value().is_string() && !it.value().is_number()) return Invalid("Invalid suite option " + it.key());
        child.values[it.key()] = it.value().is_string() ? it.value().get<std::string>() : it.value().dump();
      }
      for (const auto* key : {"model", "manifest", "image_root", "preprocessing_config"}) {
        if (!entry.contains(key)) continue;
        auto path = fs::path(child.Get(key));
        if (path.is_relative()) child.values[key] = (fs::path(options.Get("suite")).parent_path() / path).string();
      }
      Json row;
      LITERT_ASSIGN_OR_RETURN(auto hash, imagenet::HashFile(child.Get("model")));
      if (hash != entry["sha256"].get<std::string>()) return Invalid("Suite model hash mismatch");
      auto status = Validate(child, row); row["status"] = status ? "passed" : "failed";
      if (!status) { row["error"] = std::string(status.Error().Message()); passed = false; }
      if (entry.contains("known_failure")) row["known_failure"] = entry["known_failure"];
      report["models"].push_back(std::move(row));
    }
    if (!passed) return Invalid("One or more suite models failed; known failures are not waived");
    return {};
  }
  LITERT_ASSIGN_OR_RETURN(auto object, LoadModel(options.Get("model")));
  LITERT_RETURN_IF_ERROR(ImageInterface(*object));
  LITERT_ASSIGN_OR_RETURN(auto bytes, PackModel(*object));
  LITERT_ASSIGN_OR_RETURN(auto model, ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(bytes.data(), bytes.size())));
  LITERT_ASSIGN_OR_RETURN(auto graph, model.Subgraph(0));
  std::string reason;
  auto artifact = CompilePartition(graph, reason);
  if (!artifact) return Invalid("Full model preflight failed: " + reason);
  spvtools::SpirvTools spirv(SPV_ENV_UNIVERSAL_1_6);
  spirv.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
    reason += std::string(message) + "\n";
  });
  if (!spirv.Validate(artifact->spirv)) return Invalid("SPIR-V preflight failed: " + reason);
  report["spirv_valid"] = true;
  report["selected_operators"] = graph.Ops().size();
  if (options.Get("manifest").empty() || options.Get("preprocessing_config").empty())
    return Invalid("Image validation requires --manifest and --preprocessing_config");
  imagenet::EvaluationConfig config;
  config.model = options.Get("model"); config.baseline_model = config.model;
  config.accelerator = "npu"; config.baseline_accelerator = "cpu"; config.baseline_num_threads = 1;
  config.require_full_acceleration = 1; config.require_output_parity = 1; config.require_prediction_agreement = true;
  config.capture_records = true; config.compare_int8_codes = true; config.arm_vulkan_ml_enable_jit = 1;
  config.manifest = options.Get("manifest"); config.image_root = options.Get("image_root", fs::path(config.manifest).parent_path().string());
  config.preprocessing_config = options.Get("preprocessing_config"); config.out = "/dev/null";
  config.compiler_plugin_library_dir = options.Get("compiler_plugin_library_dir", ".");
  config.dispatch_library_dir = options.Get("dispatch_library_dir", ".");
  config.runtime_library_dir = options.Get("runtime_library_dir");
  LITERT_ASSIGN_OR_RETURN(config.max_samples, options.Integer("max_samples", 3));
  LITERT_ASSIGN_OR_RETURN(config.start_sample, options.Integer("start_sample", 0));
  if (config.max_samples < 1 || config.max_samples > 10 || config.start_sample < 0) return Invalid("Expected 1–10 samples and nonnegative start_sample");
  LITERT_ASSIGN_OR_RETURN(config.atol, options.Number("atol", 1e-4));
  LITERT_ASSIGN_OR_RETURN(config.rtol, options.Number("rtol", 1e-3));
  LITERT_ASSIGN_OR_RETURN(config.max_code_error, options.Number("max_code_error", 4));
  LITERT_ASSIGN_OR_RETURN(config.mean_code_error, options.Number("mean_code_error", .5));
  LITERT_ASSIGN_OR_RETURN(report["model_sha256"], imagenet::HashFile(config.model));
  report["model"] = config.model;
  report["tolerances"] = {{"atol", config.atol}, {"rtol", config.rtol}, {"max_code_error", config.max_code_error}, {"mean_code_error", config.mean_code_error}};
  return imagenet::Evaluate(config, report);
}
Expected<void> Compare(const void* actual, const void* expected, size_t count,
                       bool floating, const Options& options, Json& result) {
  LITERT_ASSIGN_OR_RETURN(auto atol, options.Number("atol", 1e-4));
  LITERT_ASSIGN_OR_RETURN(auto rtol, options.Number("rtol", 1e-3));
  LITERT_ASSIGN_OR_RETURN(auto max_code, options.Number("max_code_error", 4));
  LITERT_ASSIGN_OR_RETURN(auto mean_code, options.Number("mean_code_error", .5));
  if (!count) return Invalid("Cannot compare empty tensor");
  double maximum = 0, sum = 0; bool finite = true, close = true;
  for (size_t i = 0; i < count; ++i) {
    double a = floating ? static_cast<const float*>(actual)[i] : static_cast<const int8_t*>(actual)[i];
    double b = floating ? static_cast<const float*>(expected)[i] : static_cast<const int8_t*>(expected)[i];
    double error = std::abs(a - b); finite &= std::isfinite(a) && std::isfinite(b);
    maximum = std::max(maximum, error); sum += error;
    close &= error <= atol + rtol * std::abs(b);
  }
  bool passed = finite && (floating ? close : maximum <= max_code && sum / count <= mean_code);
  result = {{"passed", passed}, {"max_error", maximum}, {"mean_error", sum / count}, {"finite", finite}, {"elements", count}, {"floating", floating}};
  return {};
}
Expected<void> Run(const Options& options, Json& report) {
  LITERT_ASSIGN_OR_RETURN(int allow, options.Integer("allow_emulation", 0));
  if (allow != 0 && allow != 1) return Invalid("allow_emulation must be 0 or 1");
  // The standalone process explicitly selects this policy for the dispatch plugin.
  setenv("LITERT_ARM_VULKAN_ML_ALLOW_EMULATION", allow ? "1" : "0", 1);
  report["allow_emulation"] = allow != 0;
  if (!options.Get("model").empty()) {
    LITERT_ASSIGN_OR_RETURN(report["model_sha256"], imagenet::HashFile(options.Get("model")));
  }
  if (options.command == "inspect") return Inspect(options, report, false);
  if (options.command == "compile") return Inspect(options, report, true);
  if (options.command == "capabilities") return Capabilities(options, report);
  if (options.command == "validate") return Validate(options, report);
  if (options.command == "trace") return Trace(options, report);
  return Invalid("Unknown command");
}
int Main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "litert_arm_vulkan_ml_tool {inspect|capabilities|compile|validate|trace}\n"
      "  --model FILE --report_json FILE\n"
      "  compile: --output_dir DIR\n"
      "  validate/trace: --manifest FILE --preprocessing_config FILE --start_sample N\n"
      "  validate: --max_samples 3 --suite FILE (optional alternative)\n"
      "  trace: --mode accumulated|isolated --capture_budget_mb 256 --dump_tensors 0|1\n"
      "  --compiler_plugin_library_dir DIR --dispatch_library_dir DIR\n"
      "  --allow_emulation 0|1 --atol 1e-4 --rtol 1e-3 --max_code_error 4 --mean_code_error .5\n";
    return 0;
  }
  Json report = {{"schema_version", 1}, {"status", "failed"}};
  auto parsed = ParseOptions(argc, argv);
  if (!parsed) { std::cerr << parsed.Error().Message() << '\n'; return 2; }
  report["command"] = parsed->command;
  auto binary_hash = imagenet::HashFile("/proc/self/exe");
  if (binary_hash) report["binary_sha256"] = *binary_hash;
  report["instruction_set"] = "TOSA.001000.1";
  report["tensor_api_revision"] = 2;
  auto status = Run(*parsed, report);
  report["status"] = status ? "passed" : (parsed->command == "trace" && !report.contains("first_divergence") ? "incomplete" : "failed");
  if (!status) report["error"] = std::string(status.Error().Message());
  if (!parsed->Get("report_json").empty()) {
    auto written = imagenet::WriteJson(parsed->Get("report_json"), report);
    if (!written) { std::cerr << written.Error().Message() << '\n'; return 1; }
  }
  std::cout << report.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
  return status ? 0 : 1;
}
}  // namespace litert::arm_vulkan_ml::tool
