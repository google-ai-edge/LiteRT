// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"         // from @com_google_absl
#include "absl/strings/str_split.h"    // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"           // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/tools/imagenet_eval_data.h"

namespace {
namespace im = litert::imagenet;
using im::Json;

struct Config {
  std::string model;
  std::string baseline_model, summary_json, runtime_library_dir;
  std::string gpu_use_ir_model = "default";
  std::string interpolation = "bilinear", preprocessing_config;
  int require_full_acceleration = 0, require_output_parity = 0;
  double atol = 1e-4, rtol = 1e-3;
  std::string manifest;
  std::string image_root;
  std::string out;
  std::string scores_out;
  std::string inputs_out;
  int num_threads = 0;
  int xnnpack_only = 0;
  int warmup_runs = 0;
  int start_sample = 0;
  std::string accelerator = "cpu";
  std::string compiler_plugin_library_dir;
  int arm_vulkan_ml_enable_jit = 0;
  std::string baseline_accelerator = "inherit";
  int baseline_num_threads = 1;
  std::string dispatch_library_dir;
  std::string openvino_device;
  std::string gpu_backend = "automatic";
  std::string gpu_precision = "default";
  int gpu_benchmark_mode = 0;
  int resize = 232;
  int crop_h = 224;
  int crop_w = 224;
  int antialias = 1;
  std::string mean = "0.485,0.456,0.406";
  std::string stddev = "0.229,0.224,0.225";
  int max_samples = 0;
};

struct Row {
  int id;
  int label;
  std::string path;
};

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --model=MODEL --manifest=MANIFEST --image_root=DIR [options]\n"
      << "Options:\n"
      << "  --baseline_model=PATH   Compare a baseline with --model on "
         "identical inputs\n"
      << "  --summary_json=PATH     Structured completion, provenance and "
         "metrics\n"
      << "  --runtime_library_dir=DIR  Dynamic runtime directory\n"
      << "  --gpu_use_ir_model=default|true|false\n"
      << "  --require_full_acceleration=0|1\n"
      << "  --require_output_parity=0|1 --atol=1e-4 --rtol=1e-3\n"
      << "  --interpolation=bilinear|bicubic\n"
      << "  --preprocessing_config=PATH  Checkpoint preprocessing JSON\n"
      << "  --out=PATH              Output JSONL path\n"
      << "  --scores_out=PATH       Raw float32 scores, in manifest order\n"
      << "  --inputs_out=PATH       Raw preprocessed float32 inputs\n"
      << "  --num_threads=0         CPU thread budget; 0 keeps default\n"
      << "  --xnnpack_only=0|1      Disable YNNPACK; enable signed INT8 "
         "XNNPACK\n"
      << "  --warmup_runs=0         Untimed runs on the first selected input\n"
      << "  --start_sample=0        Skip this many manifest rows\n"
      << "  --accelerator=cpu       Comma-delimited cpu,gpu,npu\n"
      << "  --dispatch_library_dir=DIR\n"
      << "  --compiler_plugin_library_dir=DIR\n"
      << "  --arm_vulkan_ml_enable_jit=0|1  Requires accelerator=npu\n"
      << "  --baseline_accelerator=inherit|cpu  CPU uses reference kernels\n"
      << "  --baseline_num_threads=1  Explicit CPU baseline threads\n"
      << "  --openvino_device=cpu|gpu|npu|auto\n"
      << "  --gpu_backend=automatic|webgpu|opengl|opencl\n"
      << "  --gpu_precision=default|fp32|fp16\n"
      << "  --gpu_benchmark_mode=0|1\n"
      << "  --resize=232            Resize shorter side\n"
      << "  --crop_h=224 --crop_w=224\n"
      << "  --antialias=1           Antialias downsampling\n"
      << "  --mean=0.485,0.456,0.406 --std=0.229,0.224,0.225\n"
      << "  --max_samples=0         0 means all rows\n";
}

void ValidateBackendOptions(const Config& cfg) {
  bool npu = false;
  for (absl::string_view accelerator : absl::StrSplit(cfg.accelerator, ',')) {
    if (accelerator != "cpu" && accelerator != "gpu" && accelerator != "npu")
      throw std::runtime_error("Unknown accelerator: " + std::string(accelerator));
    npu |= accelerator == "npu";
  }
  if (cfg.arm_vulkan_ml_enable_jit != 0 && cfg.arm_vulkan_ml_enable_jit != 1)
    throw std::runtime_error("arm_vulkan_ml_enable_jit must be 0 or 1");
  if (cfg.arm_vulkan_ml_enable_jit && !npu)
    throw std::runtime_error("Arm Vulkan ML JIT requires accelerator selection to include npu");
  if (cfg.baseline_accelerator != "inherit" && cfg.baseline_accelerator != "cpu")
    throw std::runtime_error("baseline_accelerator must be inherit or cpu");
  if (cfg.baseline_accelerator == "cpu" && cfg.baseline_model.empty())
    throw std::runtime_error("CPU comparison requires baseline_model");
  if (cfg.baseline_num_threads <= 0)
    throw std::runtime_error("baseline_num_threads must be positive");
}

Config BaselineConfig(const Config& candidate) {
  Config baseline = candidate;
  baseline.model = candidate.baseline_model;
  if (candidate.baseline_accelerator == "cpu") {
    baseline.accelerator = "cpu";
    baseline.num_threads = candidate.baseline_num_threads;
    baseline.xnnpack_only = 0;
    baseline.arm_vulkan_ml_enable_jit = 0;
    baseline.compiler_plugin_library_dir.clear();
    baseline.dispatch_library_dir.clear();
    baseline.openvino_device.clear();
    baseline.gpu_backend = "automatic";
    baseline.gpu_precision = "default";
    baseline.gpu_use_ir_model = "default";
    baseline.gpu_benchmark_mode = 0;
    baseline.require_full_acceleration = 0;
  }
  return baseline;
}

void ParseArgs(int argc, char** argv, Config& cfg) {
  std::unordered_map<std::string, std::string*> strings = {
      {"model", &cfg.model},
      {"baseline_model", &cfg.baseline_model},
      {"summary_json", &cfg.summary_json},
      {"runtime_library_dir", &cfg.runtime_library_dir},
      {"gpu_use_ir_model", &cfg.gpu_use_ir_model},
      {"interpolation", &cfg.interpolation},
      {"preprocessing_config", &cfg.preprocessing_config},
      {"manifest", &cfg.manifest},
      {"image_root", &cfg.image_root},
      {"out", &cfg.out},
      {"scores_out", &cfg.scores_out},
      {"inputs_out", &cfg.inputs_out},
      {"accelerator", &cfg.accelerator},
      {"dispatch_library_dir", &cfg.dispatch_library_dir},
      {"compiler_plugin_library_dir", &cfg.compiler_plugin_library_dir},
      {"baseline_accelerator", &cfg.baseline_accelerator},
      {"openvino_device", &cfg.openvino_device},
      {"gpu_backend", &cfg.gpu_backend},
      {"gpu_precision", &cfg.gpu_precision},
      {"mean", &cfg.mean},
      {"std", &cfg.stddev},
  };
  std::unordered_map<std::string, int*> ints = {
      {"require_full_acceleration", &cfg.require_full_acceleration},
      {"arm_vulkan_ml_enable_jit", &cfg.arm_vulkan_ml_enable_jit},
      {"baseline_num_threads", &cfg.baseline_num_threads},
      {"require_output_parity", &cfg.require_output_parity},
      {"resize", &cfg.resize},
      {"crop_h", &cfg.crop_h},
      {"crop_w", &cfg.crop_w},
      {"antialias", &cfg.antialias},
      {"gpu_benchmark_mode", &cfg.gpu_benchmark_mode},
      {"max_samples", &cfg.max_samples},
      {"num_threads", &cfg.num_threads},
      {"xnnpack_only", &cfg.xnnpack_only},
      {"warmup_runs", &cfg.warmup_runs},
      {"start_sample", &cfg.start_sample},
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (arg.rfind("--", 0) != 0) {
      std::cerr << "Unexpected positional argument: " << arg << "\n";
      std::exit(EXIT_FAILURE);
    }
    size_t eq = arg.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Expected --name=value argument: " << arg << "\n";
      std::exit(EXIT_FAILURE);
    }
    std::string name = arg.substr(2, eq - 2);
    std::string value = arg.substr(eq + 1);
    if (name == "atol" || name == "rtol") {
      (name == "atol" ? cfg.atol : cfg.rtol) = std::stod(value);
    } else if (auto it = strings.find(name); it != strings.end()) {
      *it->second = value;
    } else if (auto int_it = ints.find(name); int_it != ints.end()) {
      *int_it->second = std::stoi(value);
    } else {
      std::cerr << "Unknown argument: --" << name << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if (cfg.model.empty() || cfg.manifest.empty() || cfg.image_root.empty()) {
    PrintUsage(argv[0]);
    std::exit(EXIT_FAILURE);
  }
  if (cfg.num_threads < 0 || cfg.warmup_runs < 0 || cfg.start_sample < 0 ||
      cfg.max_samples < 0) {
    std::cerr << "Thread, warmup, and sample limits must be nonnegative\n";
    std::exit(EXIT_FAILURE);
  }
  ValidateBackendOptions(cfg);
  if (cfg.require_output_parity && cfg.baseline_model.empty())
    throw std::runtime_error("Output parity requires a baseline model");
  if (cfg.gpu_use_ir_model != "default" && cfg.gpu_use_ir_model != "true" &&
      cfg.gpu_use_ir_model != "false")
    throw std::runtime_error("gpu_use_ir_model must be default, true or false");
  if (!std::isfinite(cfg.atol) || !std::isfinite(cfg.rtol) || cfg.atol < 0 ||
      cfg.rtol < 0)
    throw std::runtime_error("Invalid comparison tolerances");
}

// Record libraries actually mapped by the process, rather than assuming that a
// requested search directory selected the core runtime. Linux and Android use
// this for reproducible comparisons with a Python runtime wheel.
void RecordRuntimeLibraries(Json& summary) {
#if defined(__ANDROID__)
  for (const auto& property : {"ro.product.model", "ro.serialno", "ro.build.version.release", "ro.soc.model"}) {
    char value[PROP_VALUE_MAX] = {};
    __system_property_get(property, value);
    summary["android_device"][property] = value;
  }
#endif
#if defined(__linux__)
  size_t thread_count = 0;
  for (const auto& entry :
       std::filesystem::directory_iterator("/proc/self/task")) {
    (void)entry;
    ++thread_count;
  }
  summary["process_threads_after_compile"] = thread_count;
  std::ifstream maps("/proc/self/maps");
  std::set<std::string> paths;
  std::string line;
  while (std::getline(maps, line)) {
    auto start = line.find('/');
    if (start == std::string::npos) continue;
    auto path = line.substr(start);
    if (std::filesystem::path(path).filename().string().rfind("libLiteRt", 0) ==
            0 &&
        std::filesystem::is_regular_file(path))
      paths.insert(path);
  }
  summary["loaded_runtime_libraries"] = Json::array();
  for (const auto& path : paths)
    summary["loaded_runtime_libraries"].push_back(
        {{"path", path}, {"sha256", im::HashFile(path)}});
#endif
}

// Before October 2026 the runtime and XNNPACK delegate looked up CPU options
// under different names. Keep the current typed options and add independently
// owned aliases so a pinned older runtime does not silently ignore threads or
// kernel settings. A build action allocates fresh payloads for each model.
litert::Expected<void> AddLegacyCpuOptionNames(litert::Options& options,
                                               const litert::CpuOptions& cpu) {
  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*deleter)(void*) = nullptr;
  LITERT_RETURN_IF_ERROR(
      cpu.GetOpaqueOptionsData(&identifier, &payload, &deleter));
  std::string serialized(static_cast<const char*>(payload));
  deleter(payload);
  return options.AddBuildAction(
      [serialized = std::move(serialized)](auto* runtime, LiteRtOptions built) {
        for (const char* name : {"xnnpack", "cpu_delegate"}) {
          auto data = std::make_unique<char[]>(serialized.size() + 1);
          std::memcpy(data.get(), serialized.c_str(), serialized.size() + 1);
          LiteRtOpaqueOptions opaque;
          auto status = runtime->CreateOpaqueOptions(
              name, data.get(), [](void* p) { delete[] static_cast<char*>(p); },
              &opaque);
          if (status != kLiteRtStatusOk) return status;
          data.release();
          status = runtime->AddOpaqueOptions(built, opaque);
          if (status != kLiteRtStatusOk) {
            runtime->DestroyOpaqueOptions(opaque);
            return status;
          }
        }
        return kLiteRtStatusOk;
      });
}

bool RequestsGpu(const Config& cfg) {
  for (absl::string_view accelerator : absl::StrSplit(cfg.accelerator, ',')) {
    if (accelerator == "gpu") return true;
  }
  return false;
}

std::vector<float> ParseTriple(const std::string& text) {
  std::vector<float> out;
  for (absl::string_view part : absl::StrSplit(text, ',')) {
    out.push_back(std::stof(std::string(part)));
  }
  if (out.size() != 3) {
    std::cerr << "Expected 3 comma-delimited values: " << text << "\n";
    std::exit(EXIT_FAILURE);
  }
  return out;
}

litert::HwAcceleratorSet GetAccelerators(const Config& cfg) {
  litert::HwAcceleratorSet accelerators(litert::HwAccelerators::kNone);
  for (absl::string_view accelerator : absl::StrSplit(cfg.accelerator, ',')) {
    if (accelerator == "cpu") {
      accelerators |= litert::HwAccelerators::kCpu;
    } else if (accelerator == "gpu") {
      accelerators |= litert::HwAccelerators::kGpu;
    } else if (accelerator == "npu") {
      accelerators |= litert::HwAccelerators::kNpu;
    }
  }
  return accelerators;
}

litert::Expected<void> SetOpenVinoDevice(litert::Options& options,
                                         const std::string& device) {
  if (device.empty()) {
    return {};
  }
  LITERT_ASSIGN_OR_RETURN(auto& openvino_options,
                          options.GetIntelOpenVinoOptions());
  if (device == "cpu") {
    openvino_options.SetGraphBackend(/*graph_index=*/-1,
                                     kLiteRtIntelOpenVinoGraphBackendCPU);
  } else if (device == "gpu") {
    openvino_options.SetGraphBackend(/*graph_index=*/-1,
                                     kLiteRtIntelOpenVinoGraphBackendGPU);
  } else if (device == "npu") {
    openvino_options.SetGraphBackend(/*graph_index=*/-1,
                                     kLiteRtIntelOpenVinoGraphBackendNPU);
  } else if (device == "auto") {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "openvino_device=auto is not supported");
  } else {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Unknown openvino_device");
  }
  return {};
}

std::vector<Row> ReadManifest(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open manifest: " + path);
  }
  std::vector<Row> rows;
  std::string line;
  bool first = true;
  std::set<int> ids;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    if (first && line == "id\tlabel\tpath") {
      first = false;
      continue;
    }
    first = false;
    std::vector<std::string> cols;
    for (absl::string_view col : absl::StrSplit(line, '\t')) {
      cols.emplace_back(col);
    }
    if (cols.size() != 3) {
      throw std::runtime_error("Bad manifest row: " + line);
    }
    Row row{std::stoi(cols[0]), std::stoi(cols[1]), cols[2]};
    if (row.path.empty() || !ids.insert(row.id).second)
      throw std::runtime_error("Empty image path or duplicate sample ID");
    rows.push_back(std::move(row));
  }
  return rows;
}

bool IsSupportedIoElementType(litert::ElementType element_type) {
  return element_type == litert::ElementType::Float32 ||
         element_type == litert::ElementType::Int8 ||
         element_type == litert::ElementType::UInt8;
}

litert::Expected<LiteRtQuantizationPerTensor> GetPerTensorQuantization(
    const litert::SimpleTensor& tensor, const char* tensor_name) {
  if (tensor.QTypeId() != kLiteRtQuantizationPerTensor) {
    return litert::Error(
        kLiteRtStatusErrorInvalidArgument,
        std::string(tensor_name) + " must use per-tensor quantization");
  }
  LiteRtQuantizationPerTensor quantization = tensor.PerTensorQuantization();
  if (!std::isfinite(quantization.scale) || quantization.scale <= 0.0f) {
    return litert::Error(
        kLiteRtStatusErrorInvalidArgument,
        std::string(tensor_name) + " has invalid quantization scale");
  }
  return quantization;
}

template <typename T>
std::vector<T> QuantizePerTensor(
    const std::vector<float>& values,
    const LiteRtQuantizationPerTensor& quantization) {
  std::vector<T> out(values.size());
  const int64_t q_min = static_cast<int64_t>(std::numeric_limits<T>::min());
  const int64_t q_max = static_cast<int64_t>(std::numeric_limits<T>::max());
  for (size_t i = 0; i < values.size(); ++i) {
    double scaled =
        double(values[i]) / quantization.scale + quantization.zero_point;
    int64_t q = std::llround(std::clamp(scaled, double(q_min), double(q_max)));
    out[i] = static_cast<T>(q);
  }
  return out;
}

template <typename T>
std::vector<float> DequantizePerTensor(
    absl::Span<const T> values,
    const LiteRtQuantizationPerTensor& quantization) {
  std::vector<float> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = (static_cast<int64_t>(values[i]) - quantization.zero_point) *
             quantization.scale;
  }
  return out;
}

litert::Expected<void> WriteInputTensor(
    litert::TensorBuffer& input_buffer, const std::vector<float>& input,
    const litert::RankedTensorType& input_type,
    const litert::SimpleTensor& input_tensor) {
  switch (input_type.ElementType()) {
    case litert::ElementType::Float32:
      LITERT_RETURN_IF_ERROR(input_buffer.Write<float>(absl::MakeSpan(input)));
      return {};
    case litert::ElementType::Int8: {
      LITERT_ASSIGN_OR_RETURN(auto quantization,
                              GetPerTensorQuantization(input_tensor, "input"));
      std::vector<int8_t> quantized =
          QuantizePerTensor<int8_t>(input, quantization);
      LITERT_RETURN_IF_ERROR(
          input_buffer.Write<int8_t>(absl::MakeSpan(quantized)));
      return {};
    }
    case litert::ElementType::UInt8: {
      LITERT_ASSIGN_OR_RETURN(auto quantization,
                              GetPerTensorQuantization(input_tensor, "input"));
      std::vector<uint8_t> quantized =
          QuantizePerTensor<uint8_t>(input, quantization);
      LITERT_RETURN_IF_ERROR(
          input_buffer.Write<uint8_t>(absl::MakeSpan(quantized)));
      return {};
    }
    default:
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Unsupported input element type");
  }
}

litert::Expected<std::vector<float>> ReadOutputTensor(
    litert::TensorBuffer& output_buffer,
    const litert::RankedTensorType& output_type,
    const litert::SimpleTensor& output_tensor, size_t output_size) {
  switch (output_type.ElementType()) {
    case litert::ElementType::Float32: {
      std::vector<float> scores(output_size);
      LITERT_RETURN_IF_ERROR(output_buffer.Read<float>(absl::MakeSpan(scores)));
      return scores;
    }
    case litert::ElementType::Int8: {
      LITERT_ASSIGN_OR_RETURN(
          auto quantization, GetPerTensorQuantization(output_tensor, "output"));
      std::vector<int8_t> quantized(output_size);
      LITERT_RETURN_IF_ERROR(
          output_buffer.Read<int8_t>(absl::MakeSpan(quantized)));
      return DequantizePerTensor<int8_t>(absl::MakeConstSpan(quantized),
                                         quantization);
    }
    case litert::ElementType::UInt8: {
      LITERT_ASSIGN_OR_RETURN(
          auto quantization, GetPerTensorQuantization(output_tensor, "output"));
      std::vector<uint8_t> quantized(output_size);
      LITERT_RETURN_IF_ERROR(
          output_buffer.Read<uint8_t>(absl::MakeSpan(quantized)));
      return DequantizePerTensor<uint8_t>(absl::MakeConstSpan(quantized),
                                          quantization);
    }
    default:
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Unsupported output element type");
  }
}

// Field order keeps buffers alive only while their compiled model is alive.
struct Execution {
  litert::CompiledModel model;
  std::vector<litert::TensorBuffer> inputs, outputs;
  std::vector<int32_t> shape;
  size_t classes;
  bool fully_accelerated;

  litert::Expected<std::vector<float>> Invoke(const std::vector<float>& input,
                                              int warmups, double& ms) {
    LITERT_ASSIGN_OR_RETURN(auto signature, model.GetSignature(0));
    LITERT_ASSIGN_OR_RETURN(const auto& in, signature.InputTensor(0));
    LITERT_ASSIGN_OR_RETURN(const auto& out, signature.OutputTensor(0));
    LITERT_ASSIGN_OR_RETURN(auto it, inputs[0].TensorType());
    LITERT_ASSIGN_OR_RETURN(auto ot, outputs[0].TensorType());
    LITERT_RETURN_IF_ERROR(WriteInputTensor(inputs[0], input, it, in));
    for (int i = 0; i < warmups; ++i)
      LITERT_RETURN_IF_ERROR(model.Run(size_t(0), inputs, outputs));
    auto start = std::chrono::steady_clock::now();
    LITERT_RETURN_IF_ERROR(model.Run(size_t(0), inputs, outputs));
    ms = std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
             .count();
    return ReadOutputTensor(outputs[0], ot, out, classes);
  }
};

litert::Expected<std::unique_ptr<Execution>> CreateExecution(
    litert::Environment& env, litert::Options& options, const std::string& path,
    bool require_full) {
  LITERT_ASSIGN_OR_RETURN(auto model,
                          litert::CompiledModel::Create(env, path, options));
  LITERT_ASSIGN_OR_RETURN(bool full, model.IsFullyAccelerated());
  if (require_full && !full)
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Incomplete acceleration: " + path);
  LITERT_ASSIGN_OR_RETURN(auto inputs, model.CreateInputBuffers(0));
  LITERT_ASSIGN_OR_RETURN(auto outputs, model.CreateOutputBuffers(0));
  if (inputs.size() != 1 || outputs.size() != 1)
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Expected one input and one classification output");
  LITERT_ASSIGN_OR_RETURN(auto it, inputs[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto ot, outputs[0].TensorType());
  if (!IsSupportedIoElementType(it.ElementType()) ||
      !IsSupportedIoElementType(ot.ElementType()))
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Unsupported image/classification element type");
  auto dims = it.Layout().Dimensions();
  std::vector<int32_t> shape(dims.begin(), dims.end());
  if (shape.size() != 4 || shape[0] != 1 || (shape[1] != 3 && shape[3] != 3) ||
      std::any_of(shape.begin(), shape.end(), [](int d) { return d <= 0; }))
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Expected static batch-one three-channel image input");
  auto output_dims = ot.Layout().Dimensions();
  if (output_dims.empty() || (output_dims.size() > 1 && output_dims[0] != 1))
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Expected batch-one classification output");
  size_t classes = 1;
  for (int d : output_dims) {
    if (d <= 0 || classes > 1048576 / size_t(d))
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Invalid classification output shape");
    classes *= d;
  }
  LITERT_ASSIGN_OR_RETURN(auto signature, model.GetSignature(0));
  LITERT_ASSIGN_OR_RETURN(const auto& in, signature.InputTensor(0));
  LITERT_ASSIGN_OR_RETURN(const auto& out, signature.OutputTensor(0));
  if (it.ElementType() != litert::ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(auto q, GetPerTensorQuantization(in, "input"));
    (void)q;
  }
  if (ot.ElementType() != litert::ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(auto q, GetPerTensorQuantization(out, "output"));
    (void)q;
  }
  return std::make_unique<Execution>(
      Execution{std::move(model), std::move(inputs), std::move(outputs),
                std::move(shape), classes, full});
}

litert::Expected<litert::Options> CreateOptions(const Config& cfg, bool reference_cpu = false) {
  LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(GetAccelerators(cfg)));
  if (cfg.num_threads > 0 || cfg.xnnpack_only || reference_cpu) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    if (cfg.num_threads > 0) {
      LITERT_RETURN_IF_ERROR(cpu_options.SetNumThreads(cfg.num_threads));
    }
    if (reference_cpu) {
      LITERT_RETURN_IF_ERROR(cpu_options.SetEnableYNNPack(false));
      LITERT_RETURN_IF_ERROR(cpu_options.SetKernelMode(kLiteRtCpuKernelModeReference));
    } else if (cfg.xnnpack_only) {
      LITERT_RETURN_IF_ERROR(cpu_options.SetEnableYNNPack(false));
      LITERT_RETURN_IF_ERROR(
          cpu_options.SetKernelMode(kLiteRtCpuKernelModeDelegate));
      LITERT_RETURN_IF_ERROR(cpu_options.SetXNNPackFlags(0x00000001));
    }
    LITERT_RETURN_IF_ERROR(AddLegacyCpuOptionNames(options, cpu_options));
  }
  LITERT_RETURN_IF_ERROR(SetOpenVinoDevice(options, cfg.openvino_device));
  if (RequestsGpu(cfg)) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    if (cfg.gpu_backend == "webgpu") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetBackend(litert::GpuOptions::Backend::kWebGpu));
    } else if (cfg.gpu_backend == "opengl" || cfg.gpu_backend == "gl") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetBackend(litert::GpuOptions::Backend::kOpenGl));
    } else if (cfg.gpu_backend == "opencl" || cfg.gpu_backend == "cl") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetBackend(litert::GpuOptions::Backend::kOpenCl));
    } else if (cfg.gpu_backend != "automatic") {
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Unknown gpu_backend");
    }
    if (cfg.gpu_precision == "fp32") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp32));
    } else if (cfg.gpu_precision == "fp16") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp16));
    } else if (cfg.gpu_precision != "default") {
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Unknown gpu_precision");
    }
    if (cfg.gpu_use_ir_model != "default")
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetUseIrModel(cfg.gpu_use_ir_model == "true"));
    LITERT_RETURN_IF_ERROR(
        gpu_options.EnableBenchmarkMode(cfg.gpu_benchmark_mode != 0));
  }
  if (cfg.arm_vulkan_ml_enable_jit) {
    LITERT_ASSIGN_OR_RETURN(auto& arm, options.GetOptions<litert::arm_vulkan_ml::ArmVulkanMLOptions>());
    LITERT_RETURN_IF_ERROR(arm.SetEnableJustInTime(true));
  }
  return options;
}

Json DescribeExecution(const Config& cfg, bool reference_cpu) {
  return {{"accelerator", cfg.accelerator}, {"num_threads", cfg.num_threads},
          {"cpu_kernel", reference_cpu ? "reference" : (cfg.xnnpack_only ? "xnnpack" : "default")},
          {"gpu_backend", cfg.gpu_backend}, {"gpu_precision", cfg.gpu_precision},
          {"gpu_use_ir_model", cfg.gpu_use_ir_model},
          {"arm_vulkan_ml_enable_jit", bool(cfg.arm_vulkan_ml_enable_jit)},
          {"compiler_plugin_library_dir", cfg.compiler_plugin_library_dir},
          {"dispatch_library_dir", cfg.dispatch_library_dir},
          {"require_full_acceleration", bool(cfg.require_full_acceleration)}};
}

litert::Expected<Json> DelegationSummary(litert::CompiledModel& model) {
  LITERT_ASSIGN_OR_RETURN(auto metrics, model.GetDelegationMetrics());
  Json result = {{"total_runtime_ops", metrics.total_node_count},
                 {"accelerators", Json::array()}};
  for (const auto& accelerator : metrics.accelerators) {
    result["accelerators"].push_back({
        {"name", accelerator.accelerator_name},
        {"hardware_type", accelerator.hardware_type},
        {"delegated_ops", accelerator.delegated_node_count},
        {"partitions", accelerator.partition_count}});
  }
  return result;
}

litert::Expected<void> Run(const Config& cfg, Json& summary) {
  std::vector<litert::EnvironmentOptions::Option> env_options;
  if (!cfg.runtime_library_dir.empty())
    env_options.push_back({litert::EnvironmentOptions::Tag::kRuntimeLibraryDir,
                           cfg.runtime_library_dir.c_str()});
  if (!cfg.compiler_plugin_library_dir.empty())
    env_options.push_back({litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
                           cfg.compiler_plugin_library_dir.c_str()});
  if (!cfg.dispatch_library_dir.empty()) {
    env_options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
        cfg.dispatch_library_dir.c_str()});
  }
  LITERT_ASSIGN_OR_RETURN(
      auto env,
      litert::Environment::Create(litert::EnvironmentOptions(env_options)));
  LITERT_ASSIGN_OR_RETURN(auto options, CreateOptions(cfg));
  LITERT_ASSIGN_OR_RETURN(
      auto candidate,
      CreateExecution(env, options, cfg.model, cfg.require_full_acceleration));
  summary["candidate_execution"] = DescribeExecution(cfg, false);
  LITERT_ASSIGN_OR_RETURN(summary["candidate_delegation"], DelegationSummary(candidate->model));
  if (cfg.num_threads > 0 || cfg.xnnpack_only)
    summary["cpu_options_identifiers"] = {"cpu_options", "xnnpack", "cpu_delegate"};
  std::unique_ptr<Execution> baseline;
  if (!cfg.baseline_model.empty()) {
    const auto baseline_cfg = BaselineConfig(cfg);
    const bool reference_cpu = cfg.baseline_accelerator == "cpu";
    LITERT_ASSIGN_OR_RETURN(auto baseline_options, CreateOptions(baseline_cfg, reference_cpu));
    summary["baseline_execution"] = DescribeExecution(baseline_cfg, reference_cpu);
    LITERT_ASSIGN_OR_RETURN(baseline,
                            CreateExecution(env, baseline_options, cfg.baseline_model,
                                            baseline_cfg.require_full_acceleration));
    LITERT_ASSIGN_OR_RETURN(summary["baseline_delegation"], DelegationSummary(baseline->model));
    if (baseline->shape != candidate->shape ||
        baseline->classes != candidate->classes)
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Baseline/candidate interface mismatch");
  }
  RecordRuntimeLibraries(summary);
  bool nchw = candidate->shape[1] == 3;
  im::Preprocessing preprocessing;
  preprocessing.interpolation = cfg.interpolation;
  preprocessing.resize = cfg.resize;
  preprocessing.crop_h =
      cfg.crop_h > 0 ? cfg.crop_h : candidate->shape[nchw ? 2 : 1];
  preprocessing.crop_w =
      cfg.crop_w > 0 ? cfg.crop_w : candidate->shape[nchw ? 3 : 2];
  preprocessing.antialias = cfg.antialias != 0;
  auto mean = ParseTriple(cfg.mean), stddev = ParseTriple(cfg.stddev);
  std::copy(mean.begin(), mean.end(), preprocessing.mean.begin());
  std::copy(stddev.begin(), stddev.end(), preprocessing.stddev.begin());
  if (!cfg.preprocessing_config.empty())
    preprocessing =
        im::ParsePreprocessing(im::ReadJson(cfg.preprocessing_config));
  im::ValidatePreprocessing(preprocessing);
  if (preprocessing.crop_h != candidate->shape[nchw ? 2 : 1] ||
      preprocessing.crop_w != candidate->shape[nchw ? 3 : 2])
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Crop/model shape mismatch");
  auto rows = ReadManifest(cfg.manifest);
  for (auto& row : rows)
    row.path = (std::filesystem::path(cfg.image_root) / row.path).string();
  summary["preprocessing"] = im::DescribePreprocessing(preprocessing);
  summary["image_manifest_sha256"] = im::HashFile(cfg.manifest);
  if (size_t(cfg.start_sample) > rows.size())
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "start_sample exceeds manifest size");
  rows.erase(rows.begin(), rows.begin() + cfg.start_sample);
  if (cfg.max_samples > 0 && size_t(cfg.max_samples) < rows.size())
    rows.resize(cfg.max_samples);
  if (rows.empty())
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "No selected samples");
  for (const auto& row : rows)
    if (row.label < 0 || size_t(row.label) >= candidate->classes)
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Label outside classification output range");
  std::ofstream out_file, scores_file, inputs_file;
  if (!cfg.out.empty()) {
    out_file.open(cfg.out);
    if (!out_file) throw std::runtime_error("Cannot open JSONL output");
  }
  if (!cfg.scores_out.empty()) {
    scores_file.open(cfg.scores_out, std::ios::binary);
    if (!scores_file) throw std::runtime_error("Cannot open scores output");
  }
  if (!cfg.inputs_out.empty()) {
    inputs_file.open(cfg.inputs_out, std::ios::binary);
    if (!inputs_file) throw std::runtime_error("Cannot open inputs output");
  }
  auto& out = out_file.is_open() ? out_file : std::cout;
  out << std::setprecision(9);
  summary["candidate_fully_accelerated"] = candidate->fully_accelerated;
  if (baseline)
    summary["baseline_fully_accelerated"] = baseline->fully_accelerated;
  summary["samples_requested"] = rows.size();
  summary["samples"] = 0;
  summary["differing_rows"] = Json::array();
  summary["labels"] = Json::array();
  summary["candidate_predictions"] = Json::array();
  summary["baseline_predictions"] = Json::array();
  size_t top1 = 0, top5_count = 0, base_top1 = 0, base_top5 = 0, agreements = 0,
         allclose = 0;
  double max_abs = 0, mean_abs = 0, total_ms = 0, baseline_ms = 0;
  im::Sha256 input_hash;
  for (size_t index = 0; index < rows.size(); ++index) {
    const auto& row = rows[index];
    auto input = im::DecodePreprocess(row.path, preprocessing, nchw);
    input_hash.Update(input.data(), input.size() * sizeof(float));
    uint64_t fingerprint = UINT64_C(14695981039346656037);
    auto* bytes = reinterpret_cast<const uint8_t*>(input.data());
    for (size_t i = 0; i < input.size() * sizeof(float); ++i)
      fingerprint = (fingerprint ^ bytes[i]) * UINT64_C(1099511628211);
    if (inputs_file.is_open())
      inputs_file.write(reinterpret_cast<const char*>(input.data()),
                        input.size() * sizeof(float));
    double ms = 0, bms = 0;
    std::vector<float> reference;
    if (baseline) {
      LITERT_ASSIGN_OR_RETURN(
          reference,
          baseline->Invoke(input, index == 0 ? cfg.warmup_runs : 0, bms));
      baseline_ms += bms;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto scores,
        candidate->Invoke(input, index == 0 ? cfg.warmup_runs : 0, ms));
    total_ms += ms;
    auto ranking = im::TopK(scores, 5);
    int pred = ranking[0];
    top1 += pred == row.label;
    top5_count +=
        std::find(ranking.begin(), ranking.end(), row.label) != ranking.end();
    Json record = {{"id", row.id},
                   {"label", row.label},
                   {"top1", pred},
                   {"top5", ranking},
                   {"input_fingerprint", std::to_string(fingerprint)},
                   {"inference_ms", ms},
                   {"output_size", scores.size()}};
    if (baseline) {
      auto br = im::TopK(reference, 5);
      auto comparison =
          im::CompareScores(scores, reference, cfg.atol, cfg.rtol);
      base_top1 += br[0] == row.label;
      base_top5 += std::find(br.begin(), br.end(), row.label) != br.end();
      agreements += br[0] == pred;
      allclose += comparison.allclose;
      max_abs = std::max(max_abs, comparison.max_abs);
      mean_abs += comparison.mean_abs;
      record["baseline_top1"] = br[0];
      record["baseline_top5"] = br;
      record["baseline_inference_ms"] = bms;
      record["allclose"] = comparison.allclose;
      record["max_abs"] = comparison.max_abs;
      record["mean_abs"] = comparison.mean_abs;
      summary["baseline_predictions"].push_back(br[0]);
      if (br[0] != pred) summary["differing_rows"].push_back(row.id);
    }
    if (scores_file.is_open())
      scores_file.write(reinterpret_cast<const char*>(scores.data()),
                        scores.size() * sizeof(float));
    out << record.dump() << '\n';
    summary["labels"].push_back(row.label);
    summary["candidate_predictions"].push_back(pred);
    summary["samples"] = index + 1;
  }
  out.flush();
  if (inputs_file.is_open()) inputs_file.flush();
  if (scores_file.is_open()) scores_file.flush();
  if (!out || (inputs_file.is_open() && !inputs_file) ||
      (scores_file.is_open() && !scores_file))
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to write evaluation output");
  double n = rows.size();
  summary["input_sha256"] = input_hash.Hex();
  summary["candidate_accuracy"] = top1 / n;
  summary["candidate_top5"] = top5_count / n;
  summary["inference_ms"] = total_ms;
  if (baseline) {
    summary["baseline_accuracy"] = base_top1 / n;
    summary["baseline_top5"] = base_top5 / n;
    summary["accuracy_delta"] = double(top1) / n - double(base_top1) / n;
    summary["top5_delta"] = double(top5_count) / n - double(base_top5) / n;
    summary["prediction_agreement"] = agreements / n;
    summary["output_allclose_rows"] = allclose;
    summary["output_max_abs"] = max_abs;
    summary["output_mean_abs"] = mean_abs / n;
    summary["baseline_inference_ms"] = baseline_ms;
    if (cfg.require_output_parity && allclose != rows.size())
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "Output parity tolerance exceeded");
  }
  std::cerr << "samples=" << rows.size() << " top1=" << top1 / n
            << " top5=" << top5_count / n << " inference_ms=" << total_ms
            << '\n';
  return {};
}
}  // namespace

#ifndef LITERT_IMAGENET_EVAL_TEST
int main(int argc, char** argv) {
  Config cfg;
  Json summary = {{"schema_version", 1}, {"status", "failed"}, {"samples", 0}};
  try {
    ParseArgs(argc, argv, cfg);
    summary.update(
        {{"status", "running"},
         {"model", cfg.model},
         {"model_sha256", im::HashFile(cfg.model)},
         {"baseline_model", cfg.baseline_model},
         {"input_mode", "images"},
         {"runtime_library_dir", cfg.runtime_library_dir},
         {"dispatch_library_dir", cfg.dispatch_library_dir},
         {"compiler_plugin_library_dir", cfg.compiler_plugin_library_dir},
         {"arm_vulkan_ml_enable_jit", bool(cfg.arm_vulkan_ml_enable_jit)},
         {"baseline_accelerator", cfg.baseline_accelerator},
         {"accelerator", cfg.accelerator},
         {"gpu_backend", cfg.gpu_backend},
         {"gpu_precision", cfg.gpu_precision},
         {"gpu_use_ir_model", cfg.gpu_use_ir_model},
         {"require_full_acceleration", bool(cfg.require_full_acceleration)},
         {"num_threads", cfg.num_threads},
         {"xnnpack_only", bool(cfg.xnnpack_only)},
         {"output_tolerance", {{"atol", cfg.atol}, {"rtol", cfg.rtol}}}});
    if (!cfg.baseline_model.empty())
      summary["baseline_sha256"] = im::HashFile(cfg.baseline_model);
    if (std::filesystem::is_regular_file(argv[0]))
      summary["binary_sha256"] = im::HashFile(argv[0]);
    if (!cfg.runtime_library_dir.empty()) {
      auto library =
          std::filesystem::path(cfg.runtime_library_dir) / "libLiteRt.so";
      if (std::filesystem::is_regular_file(library))
        summary["requested_runtime_sha256"] = im::HashFile(library.string());
    }
    if (!cfg.summary_json.empty()) im::WriteJson(cfg.summary_json, summary);
    auto start = std::chrono::steady_clock::now();
    auto status = Run(cfg, summary);
    summary["elapsed_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    summary["status"] = status ? "passed" : "failed";
    if (!status) summary["error"] = std::string(status.Error().Message());
    if (!cfg.summary_json.empty()) im::WriteJson(cfg.summary_json, summary);
    if (!status) {
      std::cerr << summary["error"] << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    summary["status"] = "failed";
    summary["error"] = error.what();
    if (!cfg.summary_json.empty()) {
      try {
        im::WriteJson(cfg.summary_json, summary);
      } catch (...) {
      }
    }
    return EXIT_FAILURE;
  }
}

#endif  // LITERT_IMAGENET_EVAL_TEST
