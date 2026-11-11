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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "litert/vendors/arm_vulkan_ml/compiler/graph_ir.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "litert/vendors/arm_vulkan_ml/compiler/softmax.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/external_buffer.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/tensor_capabilities.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_validation.h"
#include "spirv-tools/libspirv.h"  // from @arm_vulkan_ml_tool_spirv_tools
#include "spirv-tools/libspirv.hpp"  // from @arm_vulkan_ml_tool_spirv_tools
#include "tflite/model_builder.h"
#include "vulkan/vulkan_core.h"  // from @arm_vulkan_ml_dep_vulkan_headers
#include "tflite/schema/schema_generated.h"
namespace litert::arm_vulkan_ml::tool {
namespace fs = std::filesystem;
namespace {
// Numerical evidence is independent of execution/validation errors. A failed
// pipeline must never be described as a measured numerical mismatch.
std::string NumericalStatus(const Json& report) {
  bool measured = false, failed = false;
  if (report.contains("operators")) {
    for (const auto& op : report["operators"])
      if (op.contains("outputs")) {
        for (const auto& output : op["outputs"])
          if (output.contains("passed")) {
            measured = true;
            failed |= !output["passed"].get<bool>();
          }
      }
    // A prefix success followed by a pipeline failure is not full validation.
    if (measured && !failed && !report.value("complete", false))
      return "incomplete";
  }
  if (report.contains("cases")) {
    for (const auto& child : report["cases"]) {
      const auto status = NumericalStatus(child);
      measured |= status == "passed" || status == "failed";
      failed |= status == "failed";
      if (!failed && (status == "not_run" || status == "incomplete"))
        return "incomplete";
    }
  }
  return failed ? "failed" : measured ? "passed" : "not_run";
}
Error Invalid(const std::string& message) {
  return Error(kLiteRtStatusErrorInvalidArgument, message);
}
Expected<void> Write(const fs::path& path, const void* bytes, size_t size) {
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char*>(bytes), size);
  file.flush();
  if (!file) return Invalid("Cannot write " + path.string());
  return {};
}
Json TensorInfo(const tflite::TensorT& tensor, int id) {
  Json result = {{"id", id},
                 {"name", tensor.name},
                 {"type", tflite::EnumNameTensorType(tensor.type)},
                 {"shape", tensor.shape},
                 {"shape_signature", tensor.shape_signature}};
  if (tensor.quantization)
    result["quantization"] = {
        {"scales", tensor.quantization->scale},
        {"zero_points", tensor.quantization->zero_point},
        {"dimension", tensor.quantization->quantized_dimension}};
  return result;
}
uint32_t Rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
void Sha256Block(std::array<uint32_t, 8>& state, const uint8_t block[64]) {
  static constexpr uint32_t kRound[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
           (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
           static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = Rotr32(w[i - 15], 7) ^ Rotr32(w[i - 15], 18) ^
                  (w[i - 15] >> 3);
    uint32_t s1 = Rotr32(w[i - 2], 17) ^ Rotr32(w[i - 2], 19) ^
                  (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    uint32_t s1 = Rotr32(e, 6) ^ Rotr32(e, 11) ^ Rotr32(e, 25);
    uint32_t ch = (e & f) ^ ((~e) & g);
    uint32_t t1 = h + s1 + ch + kRound[i] + w[i];
    uint32_t s0 = Rotr32(a, 2) ^ Rotr32(a, 13) ^ Rotr32(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}
}  // namespace
Expected<Json> ReadJson(const std::string& path) {
  std::ifstream file(path);
  if (!file) return Invalid("Cannot open JSON file: " + path);
  std::string content((std::istreambuf_iterator<char>(file)), {});
  if (file.bad()) return Invalid("Cannot read JSON file: " + path);
  auto parsed = Json::parse(content, nullptr, false);
  if (parsed.is_discarded()) return Invalid("Malformed JSON file: " + path);
  return parsed;
}
Expected<void> WriteJson(const std::string& path, const Json& value) {
  std::ofstream file(path);
  if (!file) return Invalid("Cannot open JSON output: " + path);
  file << value.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
  file.flush();
  if (!file) return Invalid("Cannot write JSON output: " + path);
  return {};
}
Expected<std::string> HashFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return Invalid("Cannot open file for hashing: " + path);
  std::array<uint32_t, 8> state = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                   0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                   0x1f83d9abu, 0x5be0cd19u};
  uint8_t buffer[4096];
  uint8_t tail[64];
  size_t tail_len = 0;
  uint64_t total_bytes = 0;
  while (file) {
    file.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
    const auto read = static_cast<size_t>(file.gcount());
    total_bytes += read;
    size_t offset = 0;
    if (tail_len > 0) {
      const size_t take = std::min(64 - tail_len, read);
      std::memcpy(tail + tail_len, buffer, take);
      tail_len += take;
      offset += take;
      if (tail_len == 64) {
        Sha256Block(state, tail);
        tail_len = 0;
      }
    }
    while (offset + 64 <= read) {
      Sha256Block(state, buffer + offset);
      offset += 64;
    }
    if (offset < read) {
      tail_len = read - offset;
      std::memcpy(tail, buffer + offset, tail_len);
    }
  }
  if (file.bad()) return Invalid("Cannot read file for hashing: " + path);
  tail[tail_len++] = 0x80u;
  if (tail_len > 56) {
    std::memset(tail + tail_len, 0, 64 - tail_len);
    Sha256Block(state, tail);
    tail_len = 0;
  }
  std::memset(tail + tail_len, 0, 56 - tail_len);
  const uint64_t total_bits = total_bytes * 8;
  for (int i = 0; i < 8; ++i) {
    tail[56 + i] = static_cast<uint8_t>((total_bits >> ((7 - i) * 8)) & 0xffu);
  }
  Sha256Block(state, tail);
  static constexpr char kHex[] = "0123456789abcdef";
  std::string digest;
  digest.reserve(64);
  for (uint32_t word : state) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      digest.push_back(kHex[(word >> shift) & 0xfu]);
    }
  }
  return digest;
}
std::string Options::Get(const std::string& key,
                         const std::string& fallback) const {
  auto it = values.find(key);
  return it == values.end() ? fallback : it->second;
}
Expected<SoftmaxLutType> Options::SoftmaxLut() const {
  auto type = ParseSoftmaxLutType(Get("softmax_lut_type", "auto"));
  if (!type) return Invalid("softmax_lut_type must be auto, int8 or int16");
  return *type;
}
Expected<int> Options::Integer(const std::string& key, int fallback) const {
  auto it = values.find(key);
  if (it == values.end()) return fallback;
  int value;
  if (!absl::SimpleAtoi(it->second, &value))
    return Invalid("Invalid integer --" + key);
  return value;
}
Expected<double> Options::Number(const std::string& key,
                                 double fallback) const {
  auto it = values.find(key);
  if (it == values.end()) return fallback;
  double value;
  if (!absl::SimpleAtod(it->second, &value) || !std::isfinite(value) ||
      value < 0)
    return Invalid("Invalid nonnegative number --" + key);
  return value;
}
Expected<int> Options::RescaleMultiplierBits() const {
  LITERT_ASSIGN_OR_RETURN(auto bits, Integer("rescale_multiplier_bits", 32));
  if (bits != 16 && bits != 32)
    return Invalid("rescale_multiplier_bits must be 16 or 32");
  return bits;
}
std::string CompilerDirectory(const Options& options) {
  if (!options.Get("compiler_plugin_library_dir").empty())
    return options.Get("compiler_plugin_library_dir");
  std::error_code error;
  const auto executable = fs::read_symlink("/proc/self/exe", error);
  if (!error) {
    for (const auto& dir :
         {executable.parent_path(),
          executable.parent_path().parent_path() / "compiler"}) {
      if (fs::exists(dir / "libLiteRtCompilerPlugin_ArmVulkanML.so"))
        return dir.string();
    }
  }
  return ".";
}
Expected<litert::Options> RuntimeOptions(const Options&) {
  LITERT_ASSIGN_OR_RETURN(auto result, litert::Options::Create());
  LITERT_RETURN_IF_ERROR(result.SetHardwareAccelerators(HwAccelerators::kNpu));
  LITERT_ASSIGN_OR_RETURN(auto& arm, result.GetOptions<ArmVulkanMLOptions>());
  LITERT_RETURN_IF_ERROR(arm.SetEnableJustInTime(true));
  return result;
}
Expected<Options> ParseOptions(int argc, char** argv) {
  Options options;
  if (argc < 2)
    return Invalid(
        "Expected inspect, capabilities, compile, validate, trace, coverage, "
        "or tosa-query; use --help");
  options.command = argv[1];
  const std::set<std::string> commands = {
      "inspect",    "capabilities", "compile", "validate",
      "trace",      "tosa-query",   "coverage", "--help"};
  if (!commands.count(options.command))
    return Invalid("Unknown command: " + options.command);
  const std::set<std::string> keys = {"model",
                                      "report_json",
                                      "output_dir",
                                      "start_sample",
                                      "max_samples",
                                      "atol",
                                      "rtol",
                                      "max_code_error",
                                      "mean_code_error",
                                      "compiler_plugin_library_dir",
                                      "dispatch_library_dir",
                                      "runtime_library_dir",
                                      "allow_emulation",
                                      "graph_view",
                                      "require_full_acceleration",
                                      "rescale_multiplier_bits",
                                      "mode",
                                      "softmax_lut_type",
                                      "reference",
                                      "capture_budget_mb",
                                      "dump_tensors",
                                      "tensor_inputs",
                                      "dump_vgf",
                                      "vulkan_validation",
                                      "database",
                                      "op",
                                      "type",
                                      "enum",
                                      "capability",
                                      "json",
                                      "op_set"};
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) return Invalid("Expected --option: " + arg);
    auto equals = arg.find('=');
    std::string key =
        arg.substr(2, equals == std::string::npos ? equals : equals - 2);
    std::string value;
    if (arg == "--dump_vgf")
      value = "1";
    else if (arg == "--json" &&
             (options.command == "tosa-query" || options.command == "coverage"))
      value = "1";
    else if (equals != std::string::npos)
      value = arg.substr(equals + 1);
    else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
      value = argv[++i];
    else
      return Invalid("Missing value for --" + key);
    if (key == "dump_vgf" &&
        (options.command != "compile" || (value != "0" && value != "1")))
      return Invalid("dump_vgf requires compile and a value of 0 or 1");
    const bool query_flag =
        key == "type" || key == "enum" || key == "capability";
    if (query_flag && options.command == "tosa-query") {
      if (value.empty()) return Invalid("Empty query option --" + key);
      options.repeated[key].push_back(value);
      continue;
    }
    if ((query_flag || key == "database" || key == "op" || key == "json") &&
        options.command != "tosa-query" &&
        !(key == "json" && options.command == "coverage"))
      return Invalid("Query options require tosa-query");
    if (options.command == "coverage" && key == "json" && value != "0" &&
        value != "1")
      return Invalid("json must be 0 or 1");
    if (key == "op_set" &&
        (options.command != "coverage" ||
         (value != "tosa" && value != "tflite" && value != "all")))
      return Invalid("op_set requires coverage and tosa, tflite or all");
    if (options.command == "coverage" && key != "json" &&
        key != "report_json" && key != "op_set")
      return Invalid("coverage accepts only op_set, json and report_json");
    if (options.command == "tosa-query") {
      const std::set<std::string> allowed = {
          "database", "op", "json", "report_json", "vulkan_validation"};
      if (!allowed.count(key))
        return Invalid("Option is not used by tosa-query: " + key);
      if (key == "json" && value != "0" && value != "1")
        return Invalid("json must be 0 or 1");
    }
    if (!keys.count(key) || value.empty() ||
        !options.values.emplace(key, value).second)
      return Invalid("Unknown, empty, or repeated option --" + key);
  }
  LITERT_RETURN_IF_ERROR(options.SoftmaxLut());
  LITERT_RETURN_IF_ERROR(options.RescaleMultiplierBits());
  if (options.Get("graph_view", "effective") != "raw" &&
      options.Get("graph_view", "effective") != "effective")
    return Invalid("graph_view must be raw or effective");
  if (options.values.count("reference") &&
      (options.command != "validate" ||
       (options.Get("reference") != "tflite" &&
        options.Get("reference") != "fp64" &&
        options.Get("reference") != "xnnpack")))
    return Invalid("reference requires validate and tflite, fp64 or xnnpack");
  return options;
}
Expected<std::unique_ptr<tflite::ModelT>> LoadModel(const std::string& path) {
  auto flat = tflite::FlatBufferModel::VerifyAndBuildFromFile(path.c_str());
  if (!flat) return Invalid("Cannot load TFLite model: " + path);
  std::ifstream file(path, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
  if (file.bad()) return Invalid("Cannot read model: " + path);
  std::unique_ptr<tflite::ModelT> model(flat->GetModel()->UnPack());
  for (auto& buffer : model->buffers)
    if (buffer->size) {
      if (buffer->offset > bytes.size() ||
          buffer->size > bytes.size() - buffer->offset)
        return Invalid("External constant buffer exceeds model file");
      buffer->data.assign(bytes.begin() + buffer->offset,
                          bytes.begin() + buffer->offset + buffer->size);
      buffer->offset = buffer->size = 0;
    }
  return model;
}
Expected<std::vector<uint8_t>> PackModel(const tflite::ModelT& model) {
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}
Expected<void> Inspect(const Options& options, Json& report, bool compile) {
  LITERT_ASSIGN_OR_RETURN(auto softmax_lut, options.SoftmaxLut());
  LITERT_ASSIGN_OR_RETURN(auto rescale_bits, options.RescaleMultiplierBits());
  LITERT_ASSIGN_OR_RETURN(auto object, LoadModel(options.Get("model")));
  const bool effective = options.Get("graph_view", "effective") == "effective";
  report["graph_view"] = effective ? "effective" : "raw";
  report["operator_index_scope"] =
      effective ? "partition-local after LiteRT core composite processing"
                : "original subgraph";
  bool complete = true;
  if (effective) {
    LITERT_ASSIGN_OR_RETURN(auto source_bytes, PackModel(*object));
    LITERT_ASSIGN_OR_RETURN(auto source,
                            ExtendedModel::CreateFromBuffer(BufferRef<uint8_t>(
                                source_bytes.data(), source_bytes.size())));
    LITERT_ASSIGN_OR_RETURN(auto runtime_options, RuntimeOptions(options));
    LITERT_ASSIGN_OR_RETURN(auto environment, Environment::Create({}));
    LITERT_ASSIGN_OR_RETURN(auto built_options,
                            internal::LiteRtOptionsPtrBuilder::Build(
                                runtime_options, environment.GetHolder()));
    const std::string directory = CompilerDirectory(options);
    const absl::string_view paths[] = {directory};
    LITERT_ASSIGN_OR_RETURN(
        auto plugin, internal::CompilerPlugin::FindPlugin(
                         "ArmVulkanML", paths, nullptr, built_options.get()));
    LITERT_ASSIGN_OR_RETURN(auto parts,
                            internal::PartitionModel(plugin, *source.Get()));
    report["residual_operators"] = Json::array();
    const absl::flat_hash_set<LiteRtOp> dispatch(parts.first.begin(),
                                                 parts.first.end());
    for (size_t si = 0; si < source.NumSubgraphs(); ++si) {
      auto* sg = source.Get()->Subgraphs()[si];
      size_t oi = 0;
      for (auto* op : sg->Ops()) {
        if (!dispatch.count(op)) {
          complete = false;
          report["residual_operators"].push_back(
              {{"subgraph", si},
               {"index", oi},
               {"opcode", static_cast<int>(op->OpCode())}});
        }
        ++oi;
      }
    }
    report["effective_partition_count"] = parts.first.size();
    if (parts.second.NumSubgraphs()) {
      LITERT_ASSIGN_OR_RETURN(
          auto partition_bytes,
          internal::SerializeModel(std::move(parts.second)));
      object.reset(tflite::GetModel(partition_bytes.Data())->UnPack());
    } else
      object->subgraphs.clear();
  }
  report["subgraphs"] = Json::array();
  fs::path output = options.Get("output_dir");
  if (compile) {
    if (output.empty()) return Invalid("compile requires --output_dir");
    std::error_code ec;
    fs::create_directories(output, ec);
    if (ec) return Invalid(ec.message());
  }
  for (size_t sg = 0; sg < object->subgraphs.size(); ++sg) {
    tflite::ModelT extracted(*object);
    auto graph = std::move(extracted.subgraphs[sg]);
    for (auto& op : graph->operators) {
      if (auto* composite =
              op->builtin_options_2.AsStableHLOCompositeOptions()) {
        composite->decomposition_subgraph_index = -1;
      }
    }
    extracted.subgraphs.clear();
    extracted.subgraphs.push_back(std::move(graph));
    extracted.signature_defs.clear();
    LITERT_ASSIGN_OR_RETURN(auto bytes, PackModel(extracted));
    LITERT_ASSIGN_OR_RETURN(
        auto model, ExtendedModel::CreateFromBuffer(
                        BufferRef<uint8_t>(bytes.data(), bytes.size())));
    LITERT_ASSIGN_OR_RETURN(auto subgraph, model.Subgraph(0));
    Json entry = {{"index", sg},
                  {"operators", Json::array()},
                  {"tensors", Json::array()},
                  {"partitions", Json::array()}};
    const auto& original = *extracted.subgraphs[0];
    for (size_t i = 0; i < original.tensors.size(); ++i)
      entry["tensors"].push_back(TensorInfo(*original.tensors[i], i));
    std::vector<LiteRtOpWithPartitionIndex> selected;
    std::unordered_map<std::string, size_t> origins;
    size_t index = 0;
    for (const auto& op : subgraph.Ops()) {
      std::string reason;
      bool supported =
          static_cast<bool>(AnalyzeOperation(op, reason, softmax_lut));
      entry["operators"].push_back(
          {{"index", index},
           {"opcode", static_cast<int>(op.Code())},
           {"name", tflite::EnumNameBuiltinOperator(
                        static_cast<tflite::BuiltinOperator>(op.Code()))},
           {"inputs", original.operators[index]->inputs},
           {"outputs", original.operators[index]->outputs},
           {"supported", supported},
           {"reason", reason}});
      entry["operators"].back()["output_names"] = Json::array();
      for (const auto& tensor : op.Outputs())
        entry["operators"].back()["output_names"].push_back(
            std::string(tensor.Name()));
      if (const auto* composite =
              original.operators[index]
                  ->builtin_options_2.AsStableHLOCompositeOptions()) {
        entry["operators"].back()["composite_name"] = composite->name;
        entry["operators"].back()["composite_version"] = composite->version;
      }
      // The partitioner clones ops. Tag copied SSA outputs to retain origin IDs
      // through cloning and topological reordering, without changing the file.
      for (auto* tensor : op.Get()->Outputs()) {
        std::string tag = "litert_devtool_op_" + std::to_string(index);
        tensor->SetName(tag);
        origins.emplace(tag, index);
      }
      ++index;
      if (supported)
        selected.push_back({op.Get(), 0});
      else
        complete = false;
    }
    entry["selected_operators"] = selected.size();
    if (!selected.empty()) {
      LITERT_ASSIGN_OR_RETURN(
          auto partitioned,
          internal::PartitionModelDirect(std::move(selected), *model.Get()));
      size_t pi = 0;
      for (auto* part : partitioned.second.Subgraphs()) {
        Json item = {{"index", pi}, {"operators", Json::array()}};
        for (auto* op : part->Ops()) {
          if (op->Outputs().empty())
            return Invalid("Partition operator has no output origin");
          auto origin =
              origins.find(std::string(op->Outputs().front()->Name()));
          if (origin == origins.end())
            return Invalid("Partition lost operator origin");
          item["operators"].push_back(origin->second);
        }
        if (compile) {
          std::string reason;
          std::vector<LoweringRecord> mapping;
          OptimizationReport optimization;
          auto compiled =
              CompilePartition(Subgraph(part), reason, &mapping, &optimization,
                               softmax_lut, rescale_bits);
          if (!compiled)
            return Invalid("Partition compilation failed: " + reason);
          std::string diagnostics;
          spvtools::SpirvTools spirv(SPV_ENV_UNIVERSAL_1_6);
          spirv.SetMessageConsumer([&](spv_message_level_t, const char*,
                                       const spv_position_t& pos,
                                       const char* message) {
            diagnostics += std::to_string(pos.index) + ": " + message + "\n";
          });
          if (!spirv.Validate(compiled->spirv))
            return Invalid("SPIR-V validation failed: " + diagnostics);
          std::string assembly;
          if (!spirv.Disassemble(compiled->spirv, &assembly,
                                 SPV_BINARY_TO_TEXT_OPTION_INDENT))
            return Invalid("SPIR-V disassembly failed: " + diagnostics);
          std::string stem = "subgraph_" + std::to_string(sg) + "_partition_" +
                             std::to_string(pi);
          LITERT_RETURN_IF_ERROR(
              Write(output / (stem + ".spv"), compiled->spirv.data(),
                    compiled->spirv.size() * sizeof(uint32_t)));
          LITERT_RETURN_IF_ERROR(Write(output / (stem + ".spvasm"),
                                       assembly.data(), assembly.size()));
          auto before_name = stem + ".before.ir";
          auto after_name = stem + ".after.ir";
          LITERT_RETURN_IF_ERROR(Write(output / before_name,
                                       optimization.before_ir.data(),
                                       optimization.before_ir.size()));
          LITERT_RETURN_IF_ERROR(Write(output / after_name,
                                       optimization.after_ir.data(),
                                       optimization.after_ir.size()));
          item["optimization"] = {
              {"before_ir", before_name},
              {"after_ir", after_name},
              {"emitted_operators", optimization.emitted_operators},
              {"constant_count", optimization.constant_count},
              {"constant_bytes", optimization.constant_bytes},
              {"compile_ms", optimization.compile_ms},
              {"decisions", Json::array()}};
          for (const auto& decision : optimization.decisions) {
            Json sources = Json::array();
            for (auto source : decision.source_operators)
              sources.push_back(item["operators"][source]);
            item["optimization"]["decisions"].push_back(
                {{"pass", decision.pass},
                 {"operators", sources},
                 {"applied", decision.applied},
                 {"reason", decision.reason}});
          }
          item["softmax_lut_type"] = SoftmaxLutTypeName(softmax_lut);
          item["rescale_multiplier_bits"] = rescale_bits;
          item["profiles"] = compiled->profiles;
          item["extensions"] = compiled->extensions;
          item["instruction_set"] = compiled->instruction_set;
          item["entry_point"] = compiled->entry_point;
          item["spirv_file"] = stem + ".spv";
          item["assembly_file"] = stem + ".spvasm";
          auto descriptors = [](const auto& tensors) {
            Json result = Json::array();
            for (const auto& tensor : tensors) {
              result.push_back({{"shape", tensor.shape},
                                {"binding", tensor.binding},
                                {"element", static_cast<int>(tensor.element)},
                                {"logical_element",
                                 static_cast<int>(tensor.logical_element)}});
              result.back()["logical_scalar"] = tensor.logical_scalar;
            }
            return result;
          };
          item["inputs"] = descriptors(compiled->inputs);
          item["outputs"] = descriptors(compiled->outputs);
          const auto& limits = compiled->level_requirements;
          item["level_requirements"] = {
              {"checked", limits.checked},
              {"max_rank", limits.max_rank},
              {"max_tensor_bytes", limits.max_tensor_bytes},
              {"max_kernel", limits.max_kernel},
              {"max_stride", limits.max_stride},
              {"max_scale", limits.max_scale},
              {"max_tensor_list_size", limits.max_tensor_list_size},
              {"max_nesting", limits.max_nesting}};
          item["tosa_opcodes"] = Json::array();
          std::set<uint32_t> tosa_sets, emitted;
          const auto& words = compiled->spirv;
          for (size_t i = 5; i < words.size();) {
            const uint32_t count = words[i] >> 16, opcode = words[i] & 0xffff;
            if (!count || count > words.size() - i)
              return Invalid("Malformed SPIR-V instruction");
            if (opcode == 11 && count >= 3) {
              const char* text = reinterpret_cast<const char*>(&words[i + 2]);
              size_t limit = (count - 2) * 4;
              const void* end = std::memchr(text, 0, limit);
              if (end && std::string(text, static_cast<const char*>(end) -
                                               text) == "TOSA.001000.1")
                tosa_sets.insert(words[i + 1]);
            }
            if (opcode == 12 && count >= 5 && tosa_sets.count(words[i + 3]))
              emitted.insert(words[i + 4]);
            i += count;
          }
          for (auto opcode : emitted) item["tosa_opcodes"].push_back(opcode);
          Json coverage;
          LITERT_RETURN_IF_ERROR(Coverage(coverage));
          Json histogram = Json::array();
          for (const auto& row : coverage["operators"]) {
            auto found =
                compiled->metrics.operators.find(row["opcode"].get<uint32_t>());
            if (found != compiled->metrics.operators.end())
              histogram.push_back({{"name", row["name"]},
                                   {"opcode", found->first},
                                   {"count", found->second}});
          }
          item["graph_metrics"] = {
              {"operators", histogram},
              {"identity_reshapes", compiled->metrics.identity_reshapes},
              {"identity_reshape_logical_bytes",
               compiled->metrics.identity_reshape_bytes},
              {"all_reshape_logical_bytes", compiled->metrics.reshape_bytes},
              {"largest_result_tensor_bytes",
               compiled->metrics.largest_result_bytes},
              {"note",
               "Static logical storage volumes; reshapes may be views; "
               "excludes peak workspace and measured allocation sizes"}};
          item["spirv_valid"] = true;
          item["lowering"] = Json::array();
          item["constants"] = Json::array();
          for (const auto& row : mapping) {
            Json sources = Json::array();
            for (auto index : row.source_operators)
              sources.push_back(item["operators"][index]);
            item["lowering"].push_back(
                {{"operator", item["operators"][row.operator_index]},
                 {"created_result_ids", row.result_ids},
                 {"source_operators", sources},
                 {"composite", row.composite}});
          }
          size_t ci = 0;
          for (const auto& constant : compiled->constants) {
            auto name = stem + "_constant_" + std::to_string(ci++) + ".bin";
            LITERT_RETURN_IF_ERROR(Write(output / name, constant.bytes.data(),
                                         constant.bytes.size()));
            item["constants"].push_back(
                {{"file", name},
                 {"shape", constant.descriptor.shape},
                 {"element", static_cast<int>(constant.descriptor.element)},
                 {"binding", constant.descriptor.binding}});
          }
          if (options.Get("dump_vgf", "0") == "1") {
            LITERT_RETURN_IF_ERROR(ExportVgf(
                *compiled, stem, (output / (stem + ".vgf")).string()));
            item["vgf_file"] = stem + ".vgf";
            item["vgf_vulkan_header_version"] = VK_HEADER_VERSION;
            LITERT_ASSIGN_OR_RETURN(
                item["vgf_sha256"],
                HashFile((output / (stem + ".vgf")).string()));
            item["vgf_scope"] =
                "one accelerated partition; excludes CPU fallback and LiteRT "
                "orchestration";
          }
          item["artifact_kind"] = "diagnostic_export_not_portable_executable";
          LITERT_RETURN_IF_ERROR(
              WriteJson((output / (stem + ".json")).string(), item));
        }
        entry["partitions"].push_back(std::move(item));
        ++pi;
      }
    }
    report["subgraphs"].push_back(std::move(entry));
  }
  report["complete_coverage"] = complete;
  report["coverage_status"] = complete ? "complete" : "partial";
  return {};
}
Expected<void> Capabilities(const Options& options, Json& report) {
  LITERT_ASSIGN_OR_RETURN(auto softmax_lut, options.SoftmaxLut());
  LITERT_ASSIGN_OR_RETURN(auto rescale_bits, options.RescaleMultiplierBits());
  struct Loader {
    VulkanValidation validation;
    void* library = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkDestroyInstance destroy = nullptr;
    ~Loader() {
      validation.Close();
      if (instance && destroy) destroy(instance, nullptr);
      if (library) dlclose(library);
    }
  } loader;
  report["loader"] = VulkanLibraryName();
  report["devices"] = Json::array();
  loader.library = dlopen(VulkanLibraryName(), RTLD_NOW | RTLD_LOCAL);
  if (!loader.library) {
    const char* detail = dlerror();
    return Invalid(detail ? detail : "Cannot load Vulkan");
  }
  auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dlsym(loader.library, "vkGetInstanceProcAddr"));
  if (!get) return Invalid("Missing vkGetInstanceProcAddr");
  auto create =
      reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
  if (!create) return Invalid("Missing vkCreateInstance");
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  info.pApplicationInfo = &app;
  std::string validation_error;
  if (!loader.validation.Prepare(get, info, validation_error))
    return Invalid(validation_error);
  VkResult result = create(&info, nullptr, &loader.instance);
  if (result != VK_SUCCESS)
    return Invalid("vkCreateInstance returned " + std::to_string(result));
  loader.destroy = reinterpret_cast<PFN_vkDestroyInstance>(
      get(loader.instance, "vkDestroyInstance"));
  if (!loader.validation.Attach(get, loader.instance, validation_error))
    return Invalid(validation_error);
  auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
      get(loader.instance, "vkEnumeratePhysicalDevices"));
  if (!enumerate) return Invalid("Missing physical-device enumeration");
  uint32_t count = 0;
  if (enumerate(loader.instance, &count, nullptr) != VK_SUCCESS)
    return Invalid("Device enumeration failed");
  std::vector<VkPhysicalDevice> devices(count);
  if (enumerate(loader.instance, &count, devices.data()) != VK_SUCCESS)
    return Invalid("Device enumeration failed");
  std::shared_ptr<const CompiledPartition> artifact;
  if (!options.Get("model").empty()) {
    LITERT_ASSIGN_OR_RETURN(auto object, LoadModel(options.Get("model")));
    if (object->subgraphs.empty())
      return Invalid("Capability matching requires a main subgraph");
    LITERT_ASSIGN_OR_RETURN(auto bytes, PackModel(*object));
    LITERT_ASSIGN_OR_RETURN(
        auto model, ExtendedModel::CreateFromBuffer(
                        BufferRef<uint8_t>(bytes.data(), bytes.size())));
    LITERT_ASSIGN_OR_RETURN(auto graph, model.Subgraph(0));
    std::string error;
    artifact = CompilePartition(graph, error, nullptr, nullptr, softmax_lut,
                                rescale_bits);
    if (!artifact) return Invalid(error);
  }
  LITERT_ASSIGN_OR_RETURN(int allow, options.Integer("allow_emulation", 0));
  bool any_match = false;
  for (auto physical : devices) {
    DeviceCapabilities caps;
    std::string error;
    if (!QueryDeviceCapabilities(get, loader.instance, physical, caps, error))
      return Invalid(error);
    Json device = {{"name", caps.properties.deviceName},
                   {"api_version", caps.properties.apiVersion},
                   {"driver_version", caps.properties.driverVersion},
                   {"vendor_id", caps.properties.vendorID},
                   {"device_id", caps.properties.deviceID},
                   {"device_type", caps.properties.deviceType},
                   {"extensions", Json::array()},
                   {"engines", Json::array()},
                   {"formats", Json::array()}};
    for (const auto& e : caps.extensions)
      device["extensions"].push_back(
          {{"name", e.extensionName}, {"revision", e.specVersion}});
    device["features"] = {{"tensors", caps.tensors_feature},
                          {"data_graph", caps.graph_feature},
                          {"vulkan_memory_model", caps.vulkan_memory_model},
                          {"synchronization2", caps.synchronization2},
                          {"maintenance5", caps.maintenance5},
                          {"replicated_composites", caps.replicated_composites},
                          {"shader_int16", caps.shader_int16},
                          {"shader_int64", caps.shader_int64}};
    device["tensor_limits"] = {
        {"rank", caps.tensors.maxTensorDimensionCount},
        {"elements", caps.tensors.maxTensorElements},
        {"dimension_elements", caps.tensors.maxPerDimensionTensorElements},
        {"bytes", caps.tensors.maxTensorSize},
        {"descriptors", caps.tensors.maxDescriptorSetStorageTensors}};
    for (const auto& f : caps.formats)
      device["formats"].push_back(
          {{"format", f.first}, {"linear_features", f.second}});
    device["external_tensors"] = Json::array();
    if (artifact) {
      auto query =
          reinterpret_cast<PFN_vkGetPhysicalDeviceExternalTensorPropertiesARM>(
              get(loader.instance,
                  "vkGetPhysicalDeviceExternalTensorPropertiesARM"));
      std::vector<TensorDescriptor> io = artifact->inputs;
      io.insert(io.end(), artifact->outputs.begin(), artifact->outputs.end());
      auto extension = [&](const char* n) {
        return std::any_of(
            caps.extensions.begin(), caps.extensions.end(),
            [&](const auto& e) { return !std::strcmp(e.extensionName, n); });
      };
      for (size_t slot = 0; slot < io.size(); ++slot) {
        Json row = {
            {"slot", slot},
            {"direction", slot < artifact->inputs.size() ? "input" : "output"},
            {"handles", Json::array()}};
        for (
            auto handle :
            {VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
             VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID}) {
          bool platform =
              handle == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                  ? extension(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) &&
                        extension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)
                  : extension(
                        "VK_ANDROID_external_memory_android_hardware_buffer") &&
                        extension(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME) &&
                        static_cast<bool>(AndroidHardwareBufferApi());
          platform = platform && extension(VK_ARM_TENSORS_EXTENSION_NAME);
          VkExternalTensorPropertiesARM props{
              VK_STRUCTURE_TYPE_EXTERNAL_TENSOR_PROPERTIES_ARM};
          if (platform && query) {
            std::vector<int64_t> dims(io[slot].shape.begin(),
                                      io[slot].shape.end());
            VkTensorDescriptionARM d{VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM};
            d.tiling = VK_TENSOR_TILING_LINEAR_ARM;
            d.format = TensorFormat(io[slot].element);
            d.dimensionCount = static_cast<uint32_t>(dims.size());
            d.pDimensions = dims.data();
            d.usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
            VkPhysicalDeviceExternalTensorInfoARM info{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_TENSOR_INFO_ARM};
            info.pDescription = &d;
            info.handleType = handle;
            query(physical, &info, &props);
          }
          bool supported =
              platform && query &&
              (props.externalMemoryProperties.externalMemoryFeatures &
               VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) &&
              (props.externalMemoryProperties.compatibleHandleTypes & handle);
          row["handles"].push_back(
              {{"type", handle == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                            ? "dma_buf"
                            : "ahardware_buffer"},
               {"importable", supported},
               {"dedicated_only",
                static_cast<bool>(
                    props.externalMemoryProperties.externalMemoryFeatures &
                    VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT)},
               {"reason", supported ? "configuration_importable; concrete "
                                      "allocation still checked"
                          : !platform ? "platform_or_extension_unavailable"
                          : !query ? "query_entry_point_unavailable"
                                   : "tensor_configuration_not_importable"}});
        }
        device["external_tensors"].push_back(std::move(row));
      }
    }
    bool required = CheckDeviceRequirements(caps, error);
    device["required_extensions"] = required;
    if (!required) device["rejection"] = error;
    for (const auto& engine : caps.engines) {
      Json e = {{"queue_family", engine.queue_family},
                {"instruction_set", engine.instruction_set},
                {"level", engine.level},
                {"level_constraints",
                 artifact ? "enforced_for_engine_match" : "model_not_supplied"},
                {"profiles", Json::array()},
                {"extensions", Json::array()}};
      for (const auto& p : engine.profiles)
        e["profiles"].push_back({{"name", p.name}, {"quality", p.quality}});
      for (const auto& p : engine.extensions)
        e["extensions"].push_back({{"name", p.name}, {"quality", p.quality}});
      if (artifact) {
        bool matches =
            required &&
            CheckEngineRequirements(*artifact, caps, engine, allow != 0, error);
        if (matches) {
          auto props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
              get(loader.instance, "vkGetPhysicalDeviceProperties2"));
          auto formats =
              reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties2>(
                  get(loader.instance, "vkGetPhysicalDeviceFormatProperties2"));
          matches = props && formats &&
                    QueryTensorCapabilities(physical, *artifact, props, formats,
                                            error);
        }
        e["reported_requirements_match"] = matches;
        if (!matches) e["rejection"] = error;
        any_match |= matches;
      }
      device["engines"].push_back(std::move(e));
    }
    report["devices"].push_back(std::move(device));
  }
  report["pipeline_acceptance"] = "not_tested";
  if (artifact && !any_match)
    return Invalid("No device matches the model's reported requirements");
  return {};
}
Expected<void> Validate(const Options& options, Json& report) {
  if (options.Get("tensor_inputs").empty()) {
    return Invalid(
        "validate requires --tensor_inputs (use litert_vision_eval for "
        "dataset/image evaluation)");
  }
  return Trace(options, report);
}
Expected<void> Compare(const void* actual, const void* expected, size_t count,
                       bool floating, const Options& options, Json& result) {
  LITERT_ASSIGN_OR_RETURN(auto atol, options.Number("atol", 1e-4));
  LITERT_ASSIGN_OR_RETURN(auto rtol, options.Number("rtol", 1e-3));
  LITERT_ASSIGN_OR_RETURN(auto max_code, options.Number("max_code_error", 4));
  LITERT_ASSIGN_OR_RETURN(auto mean_code,
                          options.Number("mean_code_error", .5));
  bool allclose = true;
  bool finite = true;
  double max_abs = 0.0;
  double sum_abs = 0.0;
  if (floating) {
    const auto* a = static_cast<const float*>(actual);
    const auto* b = static_cast<const float*>(expected);
    for (size_t i = 0; i < count; ++i) {
      if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
        finite = false;
        allclose = false;
        continue;
      }
      const double diff =
          std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
      max_abs = std::max(max_abs, diff);
      sum_abs += diff;
      if (diff > atol + rtol * std::abs(static_cast<double>(b[i]))) {
        allclose = false;
      }
    }
  } else {
    const auto* a = static_cast<const int8_t*>(actual);
    const auto* b = static_cast<const int8_t*>(expected);
    for (size_t i = 0; i < count; ++i) {
      const double diff =
          std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
      max_abs = std::max(max_abs, diff);
      sum_abs += diff;
    }
    const double mean = count ? sum_abs / static_cast<double>(count) : 0.0;
    allclose = max_abs <= max_code && mean <= mean_code;
  }
  const double mean_abs = count ? sum_abs / static_cast<double>(count) : 0.0;
  result = {{"passed", allclose && finite},
            {"max_error", max_abs},
            {"mean_error", mean_abs},
            {"finite", finite},
            {"elements", count},
            {"floating", floating}};
  return {};
}
Expected<void> Run(const Options& options, Json& report) {
  if (options.command == "coverage")
    return Coverage(report, options.Get("op_set", "tosa"));
  if (options.command == "tosa-query") return TosaQuery(options, report);
  LITERT_ASSIGN_OR_RETURN(int allow, options.Integer("allow_emulation", 0));
  if (allow != 0 && allow != 1)
    return Invalid("allow_emulation must be 0 or 1");
  // The standalone process explicitly selects this policy for the dispatch
  // plugin.
  setenv("LITERT_ARM_VULKAN_ML_ALLOW_EMULATION", allow ? "1" : "0", 1);
  report["allow_emulation"] = allow != 0;
  if (!options.Get("model").empty()) {
    LITERT_ASSIGN_OR_RETURN(report["model_sha256"],
                            HashFile(options.Get("model")));
  }
  if (options.command == "inspect") return Inspect(options, report, false);
  if (options.command == "compile") return Inspect(options, report, true);
  if (options.command == "capabilities") return Capabilities(options, report);
  if (options.command == "validate") return Validate(options, report);
  if (options.command == "trace") return Trace(options, report);
  return Invalid("Unknown command");
}
Expected<void> CollectValidationLogs(const std::string& directory,
                                     const std::string& mode, Json& result) {
  Expected<void> status;
  Json diagnostics = {{"requested", mode},
                      {"baseline_api_version", VK_HEADER_VERSION_COMPLETE},
                      {"status", "not_requested"},
                      {"errors", 0},
                      {"warnings", 0},
                      {"instances", Json::array()},
                      {"messages", Json::array()}};
  if (mode != "off") {
    bool complete = true, baseline_matches = true;
    size_t instances = 0;
    int errors = 0, warnings = 0;
    std::error_code ec;
    for (const auto& file : fs::directory_iterator(directory, ec)) {
      std::ifstream stream(file.path());
      std::string line;
      bool opened = false, closed = false;
      while (std::getline(stream, line)) {
        auto event = Json::parse(line, nullptr, false);
        if (event.is_discarded() || !event.is_object()) {
          complete = false;
          continue;
        }
        if (event.value("event", "") == "instance") {
          opened = true;
          baseline_matches &=
              event.value("api_version", 0u) == VK_HEADER_VERSION_COMPLETE;
          diagnostics["instances"].push_back(event);
        } else if (event.value("event", "") == "closed") {
          closed = true;
          complete &= event.value("enabled", false) &&
                      event.value("attached", false) &&
                      !event.value("lost", true);
        } else if (event.value("event", "") == "message") {
          int severity = event.value("severity", 0);
          errors +=
              (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
          warnings +=
              (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0;
          event["instance_log"] = file.path().filename().string();
          diagnostics["messages"].push_back(event);
        }
      }
      complete &= opened && closed && !stream.bad();
      ++instances;
    }
    complete &= !ec && instances > 0;
    diagnostics["baseline_matches"] =
        baseline_matches && !diagnostics["instances"].empty();
    diagnostics["errors"] = errors;
    diagnostics["warnings"] = warnings;
    diagnostics["status"] = !complete ? "incomplete"
                            : errors  ? "failed"
                                      : "passed";
    if (!complete || errors) {
      status = Invalid(!complete ? "Vulkan validation incomplete"
                                 : "Vulkan validation errors");
    }
  }
  result = std::move(diagnostics);
  return status;
}
int Main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout
        << "litert_arm_vulkan_ml_tool "
           "{inspect|capabilities|compile|validate|trace|tosa-query|"
           "coverage}\n"
           "  --model FILE --report_json FILE\n"
           "  coverage: [--op_set tosa|tflite|all] [--json] [--report_json "
           "FILE]\n"
           "  compile: --output_dir DIR [--dump_vgf[=0|1]]\n"
           "  tosa-query: --database FILE [--op NAME] [--type column=type] "
           "[--enum argument=value] [--capability NAME] [--json]\n"
           "  validate/trace: --tensor_inputs FILE (raw model-typed tensors) "
           "[--start_sample N] [--max_samples N]\n"
           "  trace: --mode accumulated|isolated --capture_budget_mb 256 "
           "--dump_tensors 0|1\n"
           "  --compiler_plugin_library_dir DIR --dispatch_library_dir DIR\n"
           "  --softmax_lut_type auto|int8|int16 (auto uses int8)\n"
           "  --rescale_multiplier_bits 16|32 (default 32)\n"
           "  --graph_view raw|effective (inspect/compile; default effective)\n"
           "  --require_full_acceleration 0|1 (validate; default 1)\n"
           "  validate --tensor_inputs FILE --reference tflite|fp64|xnnpack "
           "(raw single INT8 Softmax; fp64 also supports SQUARE/CONV_3D)\n"
           "  --vulkan_validation off|core|sync\n"
           "  --allow_emulation 0|1 --atol 1e-4 --rtol 1e-3 --max_code_error 4 "
           "--mean_code_error .5\n";
    return 0;
  }
  Json report = {{"schema_version", 1}, {"status", "failed"}};
  auto parsed = ParseOptions(argc, argv);
  if (!parsed) {
    std::cerr << parsed.Error().Message() << '\n';
    return 2;
  }
  report["command"] = parsed->command;
  report["softmax_lut_type"] = parsed->Get("softmax_lut_type", "auto");
  report["rescale_multiplier_bits"] =
      parsed->Get("rescale_multiplier_bits", "32");
  auto binary_hash = HashFile("/proc/self/exe");
  if (binary_hash) report["binary_sha256"] = *binary_hash;
  report["instruction_set"] = "TOSA.001000.1";
  report["tensor_api_revision"] = 2;
  const auto validation_mode = parsed->Get("vulkan_validation", "off");
  if (validation_mode != "off" && validation_mode != "core" &&
      validation_mode != "sync") {
    std::cerr << "vulkan_validation must be off, core or sync\n";
    return 2;
  }
  if (validation_mode != "off" &&
      (parsed->command == "inspect" || parsed->command == "compile" ||
       parsed->command == "tosa-query")) {
    std::cerr
        << "Vulkan validation requires an execution or capabilities command\n";
    return 2;
  }
  struct ValidationScope {
    std::map<std::string, std::string> previous;
    std::set<std::string> absent;
    std::string directory;
    void Set(const std::string& key, const std::string& value) {
      const char* old = std::getenv(key.c_str());
      if (old)
        previous[key] = old;
      else
        absent.insert(key);
      setenv(key.c_str(), value.c_str(), 1);
    }
    ~ValidationScope() {
      for (const auto& entry : previous)
        setenv(entry.first.c_str(), entry.second.c_str(), 1);
      for (const auto& key : absent) unsetenv(key.c_str());
      std::error_code ec;
      if (!directory.empty()) fs::remove_all(directory, ec);
    }
  } validation;
  if (validation_mode != "off") {
#if defined(__ANDROID__)
    char android_pattern[] = "/data/local/tmp/litert-vulkan-validation-XXXXXX";
    char* directory = mkdtemp(android_pattern);
#else
    char pattern[] = "/tmp/litert-vulkan-validation-XXXXXX";
    char* directory = mkdtemp(pattern);
#endif
    if (!directory) {
      std::cerr << "Cannot create validation directory\n";
      return 1;
    }
    validation.directory = directory;
    validation.Set("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR", directory);
  }
  validation.Set("LITERT_ARM_VULKAN_ML_VALIDATION", validation_mode);
  auto status = Run(*parsed, report);
  report["execution_status"] = status ? "passed" : "failed";
  report["numerical_status"] = NumericalStatus(report);
  auto validation_status = CollectValidationLogs(
      validation.directory, validation_mode, report["vulkan_validation"]);
  if (!validation_status && status) status = validation_status.Error();
  report["status"] = status ? "passed"
                            : (parsed->command == "trace" &&
                                       !report.contains("first_divergence")
                                   ? "incomplete"
                                   : "failed");
  if (report["vulkan_validation"].value("status", "") == "failed")
    report["status"] = "failed";
  if (!status) report["error"] = std::string(status.Error().Message());
  if (!parsed->Get("report_json").empty()) {
    auto written = WriteJson(parsed->Get("report_json"), report);
    if (!written) {
      std::cerr << written.Error().Message() << '\n';
      return 1;
    }
  }
  if (status && parsed->command == "coverage" && parsed->Get("json") != "1") {
    PrintCoverage(report);
    return 0;
  }
  if (status && parsed->command == "tosa-query" && parsed->Get("json") != "1") {
    PrintTosaQuery(report);
    return 0;
  }
  std::cout << report.dump(2, ' ', false, Json::error_handler_t::replace)
            << '\n';
  return status ? 0 : 1;
}
}  // namespace litert::arm_vulkan_ml::tool
