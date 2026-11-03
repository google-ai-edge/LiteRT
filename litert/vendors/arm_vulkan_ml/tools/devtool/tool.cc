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
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
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
#include "litert/vendors/arm_vulkan_ml/tools/devtool/devtool.pb.h"
#include "google/protobuf/text_format.h"  // from @com_google_protobuf
#include "spirv-tools/libspirv.h"  // from @arm_vulkan_ml_tool_spirv_tools
#include "spirv-tools/libspirv.hpp"  // from @arm_vulkan_ml_tool_spirv_tools
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
#include "vulkan/vulkan_core.h"  // from @arm_vulkan_ml_dep_vulkan_headers
#include "tosa/OperatorEnum.hpp"
namespace litert::arm_vulkan_ml::tool {
namespace fs = std::filesystem;
namespace {
Error Invalid(const std::string& message) {
  return Error(kLiteRtStatusErrorInvalidArgument, message);
}
struct JsonScalar {
  enum Kind { kString, kBool, kNumber };
  Kind kind = kString;
  std::string str;
  bool bool_val = false;
  int64_t int_val = 0;
  uint64_t uint_val = 0;
  bool is_negative = false;
};
bool ParseJsonString(absl::string_view input, size_t& pos, std::string& out) {
  if (pos >= input.size() || input[pos] != '"') return false;
  ++pos;
  out.clear();
  while (pos < input.size()) {
    const char c = input[pos++];
    if (c == '"') return true;
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (pos >= input.size()) return false;
    const char esc = input[pos++];
    switch (esc) {
      case '"':
      case '\\':
      case '/':
        out.push_back(esc);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'u': {
        if (pos + 4 > input.size()) return false;
        uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
          const char h = input[pos++];
          cp <<= 4;
          if (h >= '0' && h <= '9') {
            cp |= static_cast<uint32_t>(h - '0');
          } else if (h >= 'a' && h <= 'f') {
            cp |= static_cast<uint32_t>(h - 'a' + 10);
          } else if (h >= 'A' && h <= 'F') {
            cp |= static_cast<uint32_t>(h - 'A' + 10);
          } else {
            return false;
          }
        }
        if (cp < 0x80) {
          out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
          out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
          out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
          out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
          out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
          out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
        break;
      }
      default:
        return false;
    }
  }
  return false;
}
void SkipWhitespace(absl::string_view input, size_t& pos) {
  while (pos < input.size() &&
         std::isspace(static_cast<unsigned char>(input[pos]))) {
    ++pos;
  }
}
bool ParseFlatJsonObject(absl::string_view line,
                         std::map<std::string, JsonScalar>& fields) {
  fields.clear();
  size_t pos = 0;
  SkipWhitespace(line, pos);
  if (pos >= line.size() || line[pos] != '{') return false;
  ++pos;
  SkipWhitespace(line, pos);
  if (pos < line.size() && line[pos] == '}') {
    ++pos;
    SkipWhitespace(line, pos);
    return pos == line.size();
  }
  while (pos < line.size()) {
    SkipWhitespace(line, pos);
    std::string key;
    if (!ParseJsonString(line, pos, key)) return false;
    SkipWhitespace(line, pos);
    if (pos >= line.size() || line[pos] != ':') return false;
    ++pos;
    SkipWhitespace(line, pos);
    if (pos >= line.size()) return false;
    JsonScalar val;
    if (line[pos] == '"') {
      val.kind = JsonScalar::kString;
      if (!ParseJsonString(line, pos, val.str)) return false;
    } else if (line.substr(pos, 4) == "true") {
      val.kind = JsonScalar::kBool;
      val.bool_val = true;
      pos += 4;
    } else if (line.substr(pos, 5) == "false") {
      val.kind = JsonScalar::kBool;
      val.bool_val = false;
      pos += 5;
    } else if (line[pos] == '-' ||
               std::isdigit(static_cast<unsigned char>(line[pos]))) {
      val.kind = JsonScalar::kNumber;
      const size_t start = pos;
      if (line[pos] == '-') {
        val.is_negative = true;
        ++pos;
      }
      if (pos >= line.size() ||
          !std::isdigit(static_cast<unsigned char>(line[pos]))) {
        return false;
      }
      while (pos < line.size() &&
             std::isdigit(static_cast<unsigned char>(line[pos]))) {
        ++pos;
      }
      const absl::string_view token = line.substr(start, pos - start);
      if (val.is_negative) {
        if (!absl::SimpleAtoi(token, &val.int_val)) return false;
      } else {
        if (!absl::SimpleAtoi(token, &val.uint_val)) return false;
        if (val.uint_val <=
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          val.int_val = static_cast<int64_t>(val.uint_val);
        }
      }
    } else {
      return false;
    }
    fields[key] = std::move(val);
    SkipWhitespace(line, pos);
    if (pos < line.size() && line[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < line.size() && line[pos] == '}') {
      ++pos;
      SkipWhitespace(line, pos);
      return pos == line.size();
    }
    return false;
  }
  return false;
}
Expected<void> Write(const fs::path& path, const void* bytes, size_t size) {
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char*>(bytes), size);
  file.flush();
  if (!file) return Invalid("Cannot write " + path.string());
  return {};
}
TensorInfoReport TensorInfo(const tflite::TensorT& tensor, int id) {
  TensorInfoReport result;
  result.set_id(id);
  result.set_name(tensor.name);
  result.set_type(tflite::EnumNameTensorType(tensor.type));
  for (int32_t d : tensor.shape) result.add_shape(d);
  for (int32_t d : tensor.shape_signature) result.add_shape_signature(d);
  if (tensor.quantization) {
    auto* quant = result.mutable_quantization();
    for (float s : tensor.quantization->scale) quant->add_scales(s);
    for (int64_t zp : tensor.quantization->zero_point) {
      quant->add_zero_points(zp);
    }
    quant->set_dimension(tensor.quantization->quantized_dimension);
  }
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
    uint32_t s0 =
        Rotr32(w[i - 15], 7) ^ Rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 =
        Rotr32(w[i - 2], 17) ^ Rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
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
std::string TosaOperatorName(uint32_t opcode) {
  static constexpr const char* kTosaOperatorNames[] = {
      "ARGMAX",         "AVG_POOL2D",   "CONV2D",
      "CONV3D",         "DEPTHWISE_CONV2D", "FFT2D",
      "MATMUL",         "MAX_POOL2D",   "RFFT2D",
      "TRANSPOSE_CONV2D", "CLAMP",      "ERF",
      "SIGMOID",        "TANH",         "ADD",
      "ARITHMETIC_RIGHT_SHIFT", "BITWISE_AND", "BITWISE_OR",
      "BITWISE_XOR",    "INTDIV",       "LOGICAL_AND",
      "LOGICAL_LEFT_SHIFT", "LOGICAL_RIGHT_SHIFT", "LOGICAL_OR",
      "LOGICAL_XOR",    "MAXIMUM",      "MINIMUM",
      "MUL",            "POW",          "SUB",
      "TABLE",          "ABS",          "BITWISE_NOT",
      "CEIL",           "CLZ",          "COS",
      "EXP",            "FLOOR",        "LOG",
      "LOGICAL_NOT",    "NEGATE",       "RECIPROCAL",
      "RSQRT",          "SIN",          "SELECT",
      "EQUAL",          "GREATER",      "GREATER_EQUAL",
      "REDUCE_ALL",     "REDUCE_ANY",   "REDUCE_MAX",
      "REDUCE_MIN",     "REDUCE_PRODUCT", "REDUCE_SUM",
      "CONCAT",         "PAD",          "RESHAPE",
      "REVERSE",        "SLICE",        "TILE",
      "TRANSPOSE",      "GATHER",       "SCATTER",
      "RESIZE",         "CAST",         "RESCALE",
  };
  static_assert(sizeof(kTosaOperatorNames) / sizeof(kTosaOperatorNames[0]) ==
                    static_cast<size_t>(tfsc::tosa::OperatorEnum::Rescale) + 1,
                "TOSA operator name table out of sync");
  if (opcode < sizeof(kTosaOperatorNames) / sizeof(kTosaOperatorNames[0])) {
    return kTosaOperatorNames[opcode];
  }
  return "OPCODE_" + std::to_string(opcode);
}
std::string NumericalStatus(const ToolReport& report) {
  bool measured = false;
  bool failed = false;
  if (report.operators_size() > 0) {
    for (const auto& op : report.operators()) {
      for (const auto& output : op.outputs()) {
        if (output.has_passed()) {
          measured = true;
          failed |= !output.passed();
        }
      }
    }
    // A prefix success followed by a pipeline failure is not full validation.
    if (measured && !failed && !report.complete()) return "incomplete";
  }
  for (const auto& child : report.cases()) {
    const auto status = NumericalStatus(child);
    measured |= status == "passed" || status == "failed";
    failed |= status == "failed";
    if (!failed && (status == "not_run" || status == "incomplete")) {
      return "incomplete";
    }
  }
  return failed ? "failed" : (measured ? "passed" : "not_run");
}
Expected<void> ReadTextProto(const std::string& path,
                               google::protobuf::Message& message) {
  std::ifstream file(path);
  if (!file) return Invalid("Cannot open textproto file: " + path);
  std::string content((std::istreambuf_iterator<char>(file)), {});
  if (file.bad()) return Invalid("Cannot read textproto file: " + path);
  if (!google::protobuf::TextFormat::ParseFromString(content, &message)) {
    return Invalid("Malformed textproto file: " + path);
  }
  return {};
}
Expected<std::string> FormatTextProto(const google::protobuf::Message& message) {
  std::string output;
  if (!google::protobuf::TextFormat::PrintToString(message, &output)) {
    return Invalid("Cannot serialize textproto output");
  }
  return output;
}
Expected<void> WriteTextProto(const std::string& path,
                              const google::protobuf::Message& message) {
  LITERT_ASSIGN_OR_RETURN(const std::string serialized,
                          FormatTextProto(message));
  std::ofstream file(path);
  if (!file) return Invalid("Cannot open textproto output: " + path);
  file << serialized;
  file.flush();
  if (!file) return Invalid("Cannot write textproto output: " + path);
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
        "Expected inspect, capabilities, compile, validate, trace, or "
        "coverage; use --help");
  options.command = argv[1];
  const std::set<std::string> commands = {
      "inspect", "capabilities", "compile",  "validate",
      "trace",   "coverage",     "--help"};
  if (!commands.count(options.command))
    return Invalid("Unknown command: " + options.command);
  const std::set<std::string> keys = {"model",
                                      "report_txtpb",
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
                                      "txtpb"};
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) return Invalid("Expected --option: " + arg);
    auto equals = arg.find('=');
    std::string key =
        arg.substr(2, equals == std::string::npos ? equals : equals - 2);
    std::string value;
    if (arg == "--dump_vgf")
      value = "1";
    else if (arg == "--txtpb" && options.command == "coverage")
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
    if (key == "txtpb" && options.command != "coverage")
      return Invalid("--txtpb requires coverage");
    if (options.command == "coverage" && key == "txtpb" && value != "0" &&
        value != "1")
      return Invalid("txtpb must be 0 or 1");
    if (options.command == "coverage" && key != "txtpb" &&
        key != "report_txtpb")
      return Invalid("coverage accepts only txtpb and report_txtpb");
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
Expected<void> Inspect(const Options& options, ToolReport& report,
                       bool compile) {
  LITERT_ASSIGN_OR_RETURN(auto softmax_lut, options.SoftmaxLut());
  LITERT_ASSIGN_OR_RETURN(auto rescale_bits, options.RescaleMultiplierBits());
  LITERT_ASSIGN_OR_RETURN(auto object, LoadModel(options.Get("model")));
  const bool effective = options.Get("graph_view", "effective") == "effective";
  report.set_graph_view(effective ? "effective" : "raw");
  report.set_operator_index_scope(
      effective ? "partition-local after LiteRT core composite processing"
                : "original subgraph");
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
    const absl::flat_hash_set<LiteRtOp> dispatch(parts.first.begin(),
                                                 parts.first.end());
    for (size_t si = 0; si < source.NumSubgraphs(); ++si) {
      auto* sg = source.Get()->Subgraphs()[si];
      size_t oi = 0;
      for (auto* op : sg->Ops()) {
        if (!dispatch.count(op)) {
          complete = false;
          auto* residual = report.add_residual_operators();
          residual->set_subgraph(si);
          residual->set_index(oi);
          residual->set_opcode(static_cast<int32_t>(op->OpCode()));
        }
        ++oi;
      }
    }
    report.set_effective_partition_count(parts.first.size());
    if (parts.second.NumSubgraphs()) {
      LITERT_ASSIGN_OR_RETURN(
          auto partition_bytes,
          internal::SerializeModel(std::move(parts.second)));
      object.reset(tflite::GetModel(partition_bytes.Data())->UnPack());
    } else {
      object->subgraphs.clear();
    }
  }
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
    auto* entry = report.add_subgraphs();
    entry->set_index(sg);
    const auto& original = *extracted.subgraphs[0];
    for (size_t i = 0; i < original.tensors.size(); ++i)
      *entry->add_tensors() = TensorInfo(*original.tensors[i], i);
    std::vector<LiteRtOpWithPartitionIndex> selected;
    std::unordered_map<std::string, size_t> origins;
    size_t index = 0;
    for (const auto& op : subgraph.Ops()) {
      std::string reason;
      bool supported =
          static_cast<bool>(AnalyzeOperation(op, reason, softmax_lut));
      auto* op_info = entry->add_operators();
      op_info->set_index(index);
      op_info->set_opcode(static_cast<int32_t>(op.Code()));
      op_info->set_name(tflite::EnumNameBuiltinOperator(
          static_cast<tflite::BuiltinOperator>(op.Code())));
      for (int32_t id : original.operators[index]->inputs)
        op_info->add_inputs(id);
      for (int32_t id : original.operators[index]->outputs)
        op_info->add_outputs(id);
      op_info->set_supported(supported);
      op_info->set_reason(reason);
      for (const auto& tensor : op.Outputs())
        op_info->add_output_names(std::string(tensor.Name()));
      if (const auto* composite =
              original.operators[index]
                  ->builtin_options_2.AsStableHLOCompositeOptions()) {
        op_info->set_composite_name(composite->name);
        op_info->set_composite_version(composite->version);
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
    entry->set_selected_operators(selected.size());
    if (!selected.empty()) {
      LITERT_ASSIGN_OR_RETURN(
          auto partitioned,
          internal::PartitionModelDirect(std::move(selected), *model.Get()));
      size_t pi = 0;
      for (auto* part : partitioned.second.Subgraphs()) {
        auto* item = entry->add_partitions();
        item->set_index(pi);
        for (auto* op : part->Ops()) {
          if (op->Outputs().empty())
            return Invalid("Partition operator has no output origin");
          auto origin =
              origins.find(std::string(op->Outputs().front()->Name()));
          if (origin == origins.end())
            return Invalid("Partition lost operator origin");
          item->add_operators(origin->second);
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
          auto* opt_report = item->mutable_optimization();
          opt_report->set_before_ir(before_name);
          opt_report->set_after_ir(after_name);
          opt_report->set_emitted_operators(optimization.emitted_operators);
          opt_report->set_constant_count(optimization.constant_count);
          opt_report->set_constant_bytes(optimization.constant_bytes);
          opt_report->set_compile_ms(optimization.compile_ms);
          for (const auto& decision : optimization.decisions) {
            auto* dec = opt_report->add_decisions();
            dec->set_pass(decision.pass);
            for (auto source : decision.source_operators) {
              dec->add_operators(item->operators(source));
            }
            dec->set_applied(decision.applied);
            dec->set_reason(decision.reason);
          }
          item->set_softmax_lut_type(SoftmaxLutTypeName(softmax_lut));
          item->set_rescale_multiplier_bits(rescale_bits);
          for (const auto& profile : compiled->profiles)
            item->add_profiles(profile);
          for (const auto& ext : compiled->extensions)
            item->add_extensions(ext);
          item->set_instruction_set(compiled->instruction_set);
          item->set_entry_point(compiled->entry_point);
          item->set_spirv_file(stem + ".spv");
          item->set_assembly_file(stem + ".spvasm");
          auto add_descriptors = [](const auto& tensors, auto* out_list) {
            for (const auto& tensor : tensors) {
              auto* desc = out_list->Add();
              for (int64_t d : tensor.shape) desc->add_shape(d);
              desc->set_binding(tensor.binding);
              desc->set_element(static_cast<int32_t>(tensor.element));
              desc->set_logical_element(
                  static_cast<int32_t>(tensor.logical_element));
              desc->set_logical_scalar(tensor.logical_scalar);
            }
          };
          add_descriptors(compiled->inputs, item->mutable_inputs());
          add_descriptors(compiled->outputs, item->mutable_outputs());
          const auto& limits = compiled->level_requirements;
          auto* req = item->mutable_level_requirements();
          req->set_checked(limits.checked);
          req->set_max_rank(limits.max_rank);
          req->set_max_tensor_bytes(limits.max_tensor_bytes);
          req->set_max_kernel(limits.max_kernel);
          req->set_max_stride(limits.max_stride);
          req->set_max_scale(limits.max_scale);
          req->set_max_tensor_list_size(limits.max_tensor_list_size);
          req->set_max_nesting(limits.max_nesting);
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
          for (auto opcode : emitted) item->add_tosa_opcodes(opcode);
          auto* metrics = item->mutable_graph_metrics();
          for (const auto& [opcode, count] : compiled->metrics.operators) {
            auto* op_count = metrics->add_operators();
            op_count->set_name(TosaOperatorName(opcode));
            op_count->set_opcode(opcode);
            op_count->set_count(count);
          }
          metrics->set_identity_reshapes(compiled->metrics.identity_reshapes);
          metrics->set_identity_reshape_logical_bytes(
              compiled->metrics.identity_reshape_bytes);
          metrics->set_all_reshape_logical_bytes(
              compiled->metrics.reshape_bytes);
          metrics->set_largest_result_tensor_bytes(
              compiled->metrics.largest_result_bytes);
          metrics->set_note(
              "Static logical storage volumes; reshapes may be views; "
              "excludes peak workspace and measured allocation sizes");
          item->set_spirv_valid(true);
          for (const auto& row : mapping) {
            auto* rec = item->add_lowering();
            rec->set_operator_(item->operators(row.operator_index));
            for (uint32_t rid : row.result_ids)
              rec->add_created_result_ids(rid);
            for (auto src_idx : row.source_operators)
              rec->add_source_operators(item->operators(src_idx));
            rec->set_composite(row.composite);
          }
          size_t ci = 0;
          for (const auto& constant : compiled->constants) {
            auto name = stem + "_constant_" + std::to_string(ci++) + ".bin";
            LITERT_RETURN_IF_ERROR(Write(output / name, constant.bytes.data(),
                                         constant.bytes.size()));
            auto* crep = item->add_constants();
            crep->set_file(name);
            for (int64_t d : constant.descriptor.shape) crep->add_shape(d);
            crep->set_element(
                static_cast<int32_t>(constant.descriptor.element));
            crep->set_binding(constant.descriptor.binding);
          }
          if (options.Get("dump_vgf", "0") == "1") {
            LITERT_RETURN_IF_ERROR(ExportVgf(
                *compiled, stem, (output / (stem + ".vgf")).string()));
            item->set_vgf_file(stem + ".vgf");
            item->set_vgf_vulkan_header_version(VK_HEADER_VERSION);
            LITERT_ASSIGN_OR_RETURN(
                const std::string vgf_hash,
                HashFile((output / (stem + ".vgf")).string()));
            item->set_vgf_sha256(vgf_hash);
            item->set_vgf_scope(
                "one accelerated partition; excludes CPU fallback and LiteRT "
                "orchestration");
          }
          item->set_artifact_kind("diagnostic_export_not_portable_executable");
          LITERT_RETURN_IF_ERROR(
              WriteTextProto((output / (stem + ".txtpb")).string(), *item));
        }
        ++pi;
      }
    }
  }
  report.set_complete_coverage(complete);
  report.set_coverage_status(complete ? "complete" : "partial");
  return {};
}
Expected<void> Capabilities(const Options& options, ToolReport& report) {
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
  report.set_loader(VulkanLibraryName());
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
    auto* device = report.add_devices();
    device->set_name(caps.properties.deviceName);
    device->set_api_version(caps.properties.apiVersion);
    device->set_driver_version(caps.properties.driverVersion);
    device->set_vendor_id(caps.properties.vendorID);
    device->set_device_id(caps.properties.deviceID);
    device->set_device_type(static_cast<int32_t>(caps.properties.deviceType));
    for (const auto& e : caps.extensions) {
      auto* ext = device->add_extensions();
      ext->set_name(e.extensionName);
      ext->set_revision(e.specVersion);
    }
    auto* features = device->mutable_features();
    features->set_tensors(caps.tensors_feature);
    features->set_data_graph(caps.graph_feature);
    features->set_vulkan_memory_model(caps.vulkan_memory_model);
    features->set_synchronization2(caps.synchronization2);
    features->set_maintenance5(caps.maintenance5);
    features->set_replicated_composites(caps.replicated_composites);
    features->set_shader_int16(caps.shader_int16);
    features->set_shader_int64(caps.shader_int64);
    auto* tensor_limits = device->mutable_tensor_limits();
    tensor_limits->set_rank(caps.tensors.maxTensorDimensionCount);
    tensor_limits->set_elements(caps.tensors.maxTensorElements);
    tensor_limits->set_dimension_elements(
        caps.tensors.maxPerDimensionTensorElements);
    tensor_limits->set_bytes(caps.tensors.maxTensorSize);
    tensor_limits->set_descriptors(caps.tensors.maxDescriptorSetStorageTensors);
    for (const auto& f : caps.formats) {
      auto* fmt = device->add_formats();
      fmt->set_format(static_cast<int32_t>(f.first));
      fmt->set_linear_features(f.second);
    }
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
        auto* row = device->add_external_tensors();
        row->set_slot(slot);
        row->set_direction(slot < artifact->inputs.size() ? "input" : "output");
        const VkExternalMemoryHandleTypeFlagBits kHandles[] = {
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
        };
        for (auto handle : kHandles) {
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
          auto* h = row->add_handles();
          h->set_type(handle == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                          ? "dma_buf"
                          : "ahardware_buffer");
          h->set_importable(supported);
          h->set_dedicated_only(static_cast<bool>(
              props.externalMemoryProperties.externalMemoryFeatures &
              VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT));
          h->set_reason(supported   ? "configuration_importable; concrete "
                                      "allocation still checked"
                        : !platform ? "platform_or_extension_unavailable"
                        : !query    ? "query_entry_point_unavailable"
                                    : "tensor_configuration_not_importable");
        }
      }
    }
    bool required = CheckDeviceRequirements(caps, error);
    device->set_required_extensions(required);
    if (!required) device->set_rejection(error);
    for (const auto& engine : caps.engines) {
      auto* e = device->add_engines();
      e->set_queue_family(engine.queue_family);
      e->set_instruction_set(engine.instruction_set);
      e->set_level(static_cast<int32_t>(engine.level));
      e->set_level_constraints(
          artifact ? "enforced_for_engine_match" : "model_not_supplied");
      for (const auto& p : engine.profiles) {
        auto* out_p = e->add_profiles();
        out_p->set_name(p.name);
        out_p->set_quality(p.quality);
      }
      for (const auto& p : engine.extensions) {
        auto* out_ext = e->add_extensions();
        out_ext->set_name(p.name);
        out_ext->set_quality(p.quality);
      }
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
        e->set_reported_requirements_match(matches);
        if (!matches) e->set_rejection(error);
        any_match |= matches;
      }
    }
  }
  report.set_pipeline_acceptance("not_tested");
  if (artifact && !any_match)
    return Invalid("No device matches the model's reported requirements");
  return {};
}
Expected<void> Validate(const Options& options, ToolReport& report) {
  if (options.Get("tensor_inputs").empty()) {
    return Invalid(
        "validate requires --tensor_inputs (use litert_vision_eval for "
        "dataset/image evaluation)");
  }
  return Trace(options, report);
}
Expected<void> Compare(const void* actual, const void* expected, size_t count,
                       bool floating, const Options& options,
                       OutputComparisonReport& result) {
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
  result.set_passed(allclose && finite);
  result.set_max_error(max_abs);
  result.set_mean_error(mean_abs);
  result.set_finite(finite);
  result.set_elements(count);
  result.set_floating(floating);
  return {};
}
Expected<void> Run(const Options& options, ToolReport& report) {
  if (options.command == "coverage") {
    return Coverage(*report.mutable_coverage());
  }
  LITERT_ASSIGN_OR_RETURN(int allow, options.Integer("allow_emulation", 0));
  if (allow != 0 && allow != 1)
    return Invalid("allow_emulation must be 0 or 1");
  // The standalone process explicitly selects this policy for the dispatch
  // plugin.
  setenv("LITERT_ARM_VULKAN_ML_ALLOW_EMULATION", allow ? "1" : "0", 1);
  report.set_allow_emulation(allow != 0);
  if (!options.Get("model").empty()) {
    LITERT_ASSIGN_OR_RETURN(const std::string hash,
                            HashFile(options.Get("model")));
    report.set_model_sha256(hash);
  }
  if (options.command == "inspect") return Inspect(options, report, false);
  if (options.command == "compile") return Inspect(options, report, true);
  if (options.command == "capabilities") return Capabilities(options, report);
  if (options.command == "validate") return Validate(options, report);
  if (options.command == "trace") return Trace(options, report);
  return Invalid("Unknown command");
}
Expected<void> CollectValidationLogs(const std::string& directory,
                                     const std::string& mode,
                                     VulkanValidationReport& result) {
  Expected<void> status;
  VulkanValidationReport diagnostics;
  diagnostics.set_requested(mode);
  diagnostics.set_baseline_api_version(VK_HEADER_VERSION_COMPLETE);
  diagnostics.set_status("not_requested");
  diagnostics.set_errors(0);
  diagnostics.set_warnings(0);
  if (mode != "off") {
    bool complete = true;
    bool baseline_matches = true;
    size_t instances = 0;
    int errors = 0;
    int warnings = 0;
    std::error_code ec;
    for (const auto& file : fs::directory_iterator(directory, ec)) {
      std::ifstream stream(file.path());
      std::string line;
      bool opened = false;
      bool closed = false;
      while (std::getline(stream, line)) {
        std::map<std::string, JsonScalar> fields;
        auto event_it = fields.end();
        if (!ParseFlatJsonObject(line, fields) ||
            (event_it = fields.find("event")) == fields.end() ||
            event_it->second.kind != JsonScalar::kString) {
          complete = false;
          continue;
        }
        const std::string& event = event_it->second.str;
        if (event == "instance") {
          opened = true;
          auto* rec = diagnostics.add_instances();
          rec->set_event(event);
          if (auto it = fields.find("mode");
              it != fields.end() && it->second.kind == JsonScalar::kString) {
            rec->set_mode(it->second.str);
          }
          if (auto it = fields.find("layer");
              it != fields.end() && it->second.kind == JsonScalar::kString) {
            rec->set_layer(it->second.str);
          }
          uint32_t api_ver = 0;
          if (auto it = fields.find("api_version");
              it != fields.end() && it->second.kind == JsonScalar::kNumber &&
              !it->second.is_negative) {
            api_ver = static_cast<uint32_t>(it->second.uint_val);
            rec->set_api_version(api_ver);
          }
          if (auto it = fields.find("implementation_version");
              it != fields.end() && it->second.kind == JsonScalar::kNumber &&
              !it->second.is_negative) {
            rec->set_implementation_version(
                static_cast<uint32_t>(it->second.uint_val));
          }
          baseline_matches &= api_ver == VK_HEADER_VERSION_COMPLETE;
        } else if (event == "closed") {
          closed = true;
          auto bool_field = [&](const char* key, bool fallback) {
            auto it = fields.find(key);
            return (it != fields.end() && it->second.kind == JsonScalar::kBool)
                       ? it->second.bool_val
                       : fallback;
          };
          complete &= bool_field("enabled", false) &&
                      bool_field("attached", false) &&
                      !bool_field("lost", true);
        } else if (event == "message") {
          int severity = 0;
          if (auto it = fields.find("severity");
              it != fields.end() && it->second.kind == JsonScalar::kNumber) {
            severity = static_cast<int>(it->second.int_val);
          }
          errors +=
              (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
          warnings +=
              (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0;
          auto* msg = diagnostics.add_messages();
          msg->set_event(event);
          msg->set_severity(severity);
          if (auto it = fields.find("type");
              it != fields.end() && it->second.kind == JsonScalar::kNumber &&
              !it->second.is_negative) {
            msg->set_type(static_cast<uint32_t>(it->second.uint_val));
          }
          if (auto it = fields.find("id");
              it != fields.end() && it->second.kind == JsonScalar::kString) {
            msg->set_id(it->second.str);
          }
          if (auto it = fields.find("message");
              it != fields.end() && it->second.kind == JsonScalar::kString) {
            msg->set_message(it->second.str);
          }
          msg->set_instance_log(file.path().filename().string());
        }
      }
      complete &= opened && closed && !stream.bad();
      ++instances;
    }
    complete &= !ec && instances > 0;
    diagnostics.set_baseline_matches(baseline_matches &&
                                     diagnostics.instances_size() > 0);
    diagnostics.set_errors(errors);
    diagnostics.set_warnings(warnings);
    diagnostics.set_status(!complete ? "incomplete"
                           : errors  ? "failed"
                                     : "passed");
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
           "{inspect|capabilities|compile|validate|trace|coverage}\n"
           "  --model FILE --report_txtpb FILE\n"
           "  coverage: [--txtpb] [--report_txtpb FILE]\n"
           "  compile: --output_dir DIR [--dump_vgf[=0|1]]\n"
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
  ToolReport report;
  report.set_schema_version(1);
  report.set_status("failed");
  auto parsed = ParseOptions(argc, argv);
  if (!parsed) {
    std::cerr << parsed.Error().Message() << '\n';
    return 2;
  }
  report.set_command(parsed->command);
  report.set_softmax_lut_type(parsed->Get("softmax_lut_type", "auto"));
  report.set_rescale_multiplier_bits(
      parsed->Get("rescale_multiplier_bits", "32"));
  auto binary_hash = HashFile("/proc/self/exe");
  if (binary_hash) report.set_binary_sha256(*binary_hash);
  report.set_instruction_set("TOSA.001000.1");
  report.set_tensor_api_revision(2);
  const auto validation_mode = parsed->Get("vulkan_validation", "off");
  if (validation_mode != "off" && validation_mode != "core" &&
      validation_mode != "sync") {
    std::cerr << "vulkan_validation must be off, core or sync\n";
    return 2;
  }
  if (validation_mode != "off" &&
      (parsed->command == "inspect" || parsed->command == "compile")) {
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
  report.set_execution_status(status ? "passed" : "failed");
  report.set_numerical_status(NumericalStatus(report));
  auto validation_status = CollectValidationLogs(
      validation.directory, validation_mode,
      *report.mutable_vulkan_validation());
  if (!validation_status && status) status = validation_status.Error();
  report.set_status(status ? "passed"
                           : (parsed->command == "trace" &&
                                      !report.has_first_divergence()
                                  ? "incomplete"
                                  : "failed"));
  if (report.vulkan_validation().status() == "failed")
    report.set_status("failed");
  if (!status) report.set_error(status.Error().Message());
  if (!parsed->Get("report_txtpb").empty()) {
    auto written = WriteTextProto(parsed->Get("report_txtpb"), report);
    if (!written) {
      std::cerr << written.Error().Message() << '\n';
      return 1;
    }
  }
  if (status && parsed->command == "coverage" &&
      parsed->Get("txtpb") != "1") {
    PrintCoverage(report.coverage());
    return 0;
  }
  auto formatted = FormatTextProto(report);
  if (!formatted) {
    std::cerr << formatted.Error().Message() << '\n';
    return 1;
  }
  std::cout << *formatted;
  return status ? 0 : 1;
}
}  // namespace litert::arm_vulkan_ml::tool
