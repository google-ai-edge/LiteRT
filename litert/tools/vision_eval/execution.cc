// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/vision_eval/execution.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <memory>
#include <ratio>  // NOLINT(build/c++11)
#include <set>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/image.h"

namespace litert::vision {
Expected<void> ValidateBackendOptions(const BackendOptions& backend) {
  if (backend.num_threads() < 0) {
    return Invalid("num_threads must be nonnegative");
  }
  if (backend.gpu_use_ir_model() != "default" &&
      backend.gpu_use_ir_model() != "true" &&
      backend.gpu_use_ir_model() != "false") {
    return Invalid("gpu_use_ir_model must be default, true or false");
  }
  if (backend.gpu_backend() != "automatic" &&
      backend.gpu_backend() != "webgpu" && backend.gpu_backend() != "opencl" &&
      backend.gpu_backend() != "cl" && backend.gpu_backend() != "opengl" &&
      backend.gpu_backend() != "gl") {
    return Invalid("Unknown gpu_backend");
  }
  if (backend.gpu_precision() != "default" &&
      backend.gpu_precision() != "fp32" && backend.gpu_precision() != "fp16") {
    return Invalid("Unknown gpu_precision");
  }
  bool npu = false;
  for (absl::string_view accelerator :
       absl::StrSplit(backend.accelerator(), ',')) {
    if (accelerator != "cpu" && accelerator != "gpu" && accelerator != "npu") {
      return Invalid("Unknown accelerator: " + std::string(accelerator));
    }
    npu |= accelerator == "npu";
  }
  if (backend.arm_vulkan_ml_enable_jit() && !npu) {
    return Invalid("Arm Vulkan ML JIT requires accelerator selection to include npu");
  }
  return {};
}

namespace {
Expected<void> ConfigureCpuOptions(CpuOptions& cpu_options,
                                   const BackendOptions& backend,
                                   bool reference_cpu) {
  if (backend.num_threads() > 0) {
    LITERT_RETURN_IF_ERROR(cpu_options.SetNumThreads(backend.num_threads()));
  }
  if (reference_cpu) {
    LITERT_RETURN_IF_ERROR(cpu_options.SetEnableYNNPack(false));
    LITERT_RETURN_IF_ERROR(
        cpu_options.SetKernelMode(kLiteRtCpuKernelModeReference));
  } else if (backend.xnnpack_only()) {
    LITERT_RETURN_IF_ERROR(cpu_options.SetEnableYNNPack(false));
    LITERT_RETURN_IF_ERROR(
        cpu_options.SetKernelMode(kLiteRtCpuKernelModeDelegate));
    LITERT_RETURN_IF_ERROR(cpu_options.SetXNNPackFlags(0x00000001));
  }
  return {};
}

Expected<void> AddLegacyCpuOptionNames(Options& options,
                                       const BackendOptions& backend,
                                       bool reference_cpu) {
  LITERT_ASSIGN_OR_RETURN(auto legacy_cpu, CpuOptions::Create());
  LITERT_RETURN_IF_ERROR(
      ConfigureCpuOptions(legacy_cpu, backend, reference_cpu));
  auto shared_cpu = std::make_shared<CpuOptions>(std::move(legacy_cpu));
  return options.AddBuildAction([shared_cpu](auto* runtime, LiteRtOptions built) {
    for (const char* name : {"xnnpack", "cpu_delegate"}) {
      const char* identifier = nullptr;
      void* payload = nullptr;
      void (*deleter)(void*) = nullptr;
      LiteRtStatus status =
          shared_cpu->GetOpaqueOptionsData(&identifier, &payload, &deleter);
      if (status != kLiteRtStatusOk) return status;
      LiteRtOpaqueOptions opaque;
      status = runtime->CreateOpaqueOptions(name, payload, deleter, &opaque);
      if (status != kLiteRtStatusOk) {
        deleter(payload);
        return status;
      }
      status = runtime->AddOpaqueOptions(built, opaque);
      if (status != kLiteRtStatusOk) {
        runtime->DestroyOpaqueOptions(opaque);
        return status;
      }
    }
    return kLiteRtStatusOk;
  });
}

bool RequestsGpu(const BackendOptions& cfg) {
  for (absl::string_view accelerator : absl::StrSplit(cfg.accelerator(), ',')) {
    if (accelerator == "gpu") return true;
  }
  return false;
}

HwAcceleratorSet GetAccelerators(const BackendOptions& cfg) {
  HwAcceleratorSet accelerators(HwAccelerators::kNone);
  for (absl::string_view accelerator : absl::StrSplit(cfg.accelerator(), ',')) {
    if (accelerator == "cpu") {
      accelerators |= HwAccelerators::kCpu;
    } else if (accelerator == "gpu") {
      accelerators |= HwAccelerators::kGpu;
    } else if (accelerator == "npu") {
      accelerators |= HwAccelerators::kNpu;
    }
  }
  return accelerators;
}

Expected<void> SetOpenVinoDevice(Options& options,
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
    return Error(kLiteRtStatusErrorInvalidArgument,
                         "openvino_device=auto is not supported");
  } else {
    return Error(kLiteRtStatusErrorInvalidArgument,
                         "Unknown openvino_device");
  }
  return {};
}

}  // namespace

Expected<Options> CreateOptions(const BackendOptions& backend,
                                bool reference_cpu) {
  LITERT_RETURN_IF_ERROR(ValidateBackendOptions(backend));
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  LITERT_RETURN_IF_ERROR(
      options.SetHardwareAccelerators(GetAccelerators(backend)));
  if (backend.num_threads() > 0 || backend.xnnpack_only() || reference_cpu) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    LITERT_RETURN_IF_ERROR(
        ConfigureCpuOptions(cpu_options, backend, reference_cpu));
    LITERT_RETURN_IF_ERROR(
        AddLegacyCpuOptionNames(options, backend, reference_cpu));
  }
  LITERT_RETURN_IF_ERROR(SetOpenVinoDevice(options, backend.openvino_device()));
  if (RequestsGpu(backend)) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    if (backend.gpu_backend() == "webgpu") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetBackend(GpuOptions::Backend::kWebGpu));
    } else if (backend.gpu_backend() == "opengl" ||
               backend.gpu_backend() == "gl") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetBackend(GpuOptions::Backend::kOpenGl));
    } else if (backend.gpu_backend() == "opencl" ||
               backend.gpu_backend() == "cl") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetBackend(GpuOptions::Backend::kOpenCl));
    } else if (backend.gpu_backend() != "automatic") {
      return Error(kLiteRtStatusErrorInvalidArgument,
                           "Unknown gpu_backend");
    }
    if (backend.gpu_precision() == "fp32") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetPrecision(GpuOptions::Precision::kFp32));
    } else if (backend.gpu_precision() == "fp16") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetPrecision(GpuOptions::Precision::kFp16));
    } else if (backend.gpu_precision() != "default") {
      return Error(kLiteRtStatusErrorInvalidArgument,
                           "Unknown gpu_precision");
    }
    if (backend.gpu_use_ir_model() != "default")
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetUseIrModel(backend.gpu_use_ir_model() == "true"));
    LITERT_RETURN_IF_ERROR(
        gpu_options.EnableBenchmarkMode(backend.gpu_benchmark_mode() != 0));
  }
  if (backend.arm_vulkan_ml_enable_jit()) {
    LITERT_ASSIGN_OR_RETURN(
        auto& arm,
        options.GetOptions<litert::arm_vulkan_ml::ArmVulkanMLOptions>());
    LITERT_RETURN_IF_ERROR(arm.SetEnableJustInTime(true));
  }
  return options;
}

BackendOptions BaselineOptions(const EvaluationConfig& config) {
  auto out = config.backend();
  if (config.baseline_accelerator() == "cpu") {
    out.set_accelerator("cpu");
    out.set_num_threads(config.baseline_num_threads());
    out.set_xnnpack_only(false);
    out.set_arm_vulkan_ml_enable_jit(false);
    out.clear_compiler_plugin_library_dir();
    out.clear_dispatch_library_dir();
    out.clear_openvino_device();
    out.clear_gpu_backend();
    out.clear_gpu_precision();
    out.clear_gpu_use_ir_model();
    out.clear_gpu_benchmark_mode();
    out.set_require_full_acceleration(false);
  }
  return out;
}

namespace {
Expected<TensorSpec> Describe(TensorBuffer& buffer, const SimpleTensor& tensor,
                              const std::string& name) {
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  TensorSpec result;
  result.set_name(name);
  switch (type.ElementType()) {
    case litert::ElementType::Float32:
      result.set_type(FLOAT32);
      break;
    case litert::ElementType::Int8:
      result.set_type(INT8);
      break;
    case litert::ElementType::UInt8:
      result.set_type(UINT8);
      break;
    default:
      return Invalid("Model interfaces support FP32, INT8 and UINT8");
  }
  for (auto dim : type.Layout().Dimensions()) result.add_shape(dim);
  if (result.type() != FLOAT32) {
    if (tensor.QTypeId() != kLiteRtQuantizationPerTensor) {
      return Invalid("Model IO requires per-tensor quantization");
    }
    const auto q = tensor.PerTensorQuantization();
    result.set_scale(q.scale);
    result.set_zero_point(q.zero_point);
  }
  LITERT_RETURN_IF_ERROR(TensorBytes(result));
  return result;
}
}  // namespace

Expected<std::unique_ptr<Execution>> Execution::Create(
    Environment& environment, const BackendOptions& backend,
    const std::filesystem::path& path, int signature, bool reference_cpu,
    ExecutionIdentity& identity) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateOptions(backend, reference_cpu));
  const auto start = std::chrono::steady_clock::now();
  std::vector<uint8_t> model_bytes;
#if defined(_WIN32)
  // Open via wide `std::filesystem::path` stream on Windows so Unicode and
  // extended-length paths work regardless of the narrow ANSI codepage.
  std::ifstream model_stream(path, std::ios::binary | std::ios::ate);
  const auto end = model_stream ? model_stream.tellg() : std::streampos(-1);
  if (!model_stream || end <= 0) {
    return Invalid("Cannot read model file: " + PathToUtf8(path));
  }
  model_bytes.resize(static_cast<size_t>(end));
  model_stream.seekg(0);
  if (!model_stream.read(reinterpret_cast<char*>(model_bytes.data()),
                         static_cast<std::streamsize>(model_bytes.size()))) {
    return Invalid("Cannot read model file: " + PathToUtf8(path));
  }
  LITERT_ASSIGN_OR_RETURN(
      auto compiled,
      CompiledModel::Create(environment,
                            BufferRef<uint8_t>(model_bytes.data(),
                                               model_bytes.size()),
                            options));
#else
  (void)sizeof(BufferRef<uint8_t>);
  LITERT_ASSIGN_OR_RETURN(
      auto compiled,
      CompiledModel::Create(environment, PathToUtf8(path), options));
#endif
  LITERT_ASSIGN_OR_RETURN(auto full, compiled.IsFullyAccelerated());
  if (backend.require_full_acceleration() && !full) {
    return Invalid("Incomplete acceleration: " + PathToUtf8(path));
  }
  auto result = std::make_unique<Execution>(Execution::Passkey{},
                                            std::move(model_bytes),
                                            std::move(compiled), signature);
  LITERT_ASSIGN_OR_RETURN(result->inputs_, result->model_.CreateInputBuffers(signature));
  LITERT_ASSIGN_OR_RETURN(result->outputs_, result->model_.CreateOutputBuffers(signature));
  LITERT_ASSIGN_OR_RETURN(auto metadata, result->model_.GetSignature(signature));
  LITERT_ASSIGN_OR_RETURN(auto input_names, result->model_.GetSignatureInputNames(signature));
  LITERT_ASSIGN_OR_RETURN(auto output_names, result->model_.GetSignatureOutputNames(signature));
  for (size_t i = 0; i < result->inputs_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(const auto& tensor, metadata.InputTensor(i));
    LITERT_ASSIGN_OR_RETURN(auto spec, Describe(result->inputs_[i], tensor,
                                               std::string(input_names[i])));
    result->input_specs_.push_back(spec);
    *identity.add_inputs() = std::move(spec);
  }
  for (size_t i = 0; i < result->outputs_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(const auto& tensor, metadata.OutputTensor(i));
    LITERT_ASSIGN_OR_RETURN(auto spec, Describe(result->outputs_[i], tensor,
                                               std::string(output_names[i])));
    result->output_specs_.push_back(spec);
    *identity.add_outputs() = std::move(spec);
  }
  LITERT_ASSIGN_OR_RETURN(auto model_file, Identify(path));
  *identity.mutable_model() = std::move(model_file);
  *identity.mutable_options() = backend;
  identity.set_fully_accelerated(full);
  identity.set_setup_ms(std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start).count());
  LITERT_ASSIGN_OR_RETURN(auto metrics, result->model_.GetDelegationMetrics());
  identity.set_total_runtime_ops(metrics.total_node_count);
  for (const auto& accelerator : metrics.accelerators) {
    auto* record = identity.add_accelerators();
    record->set_name(accelerator.accelerator_name);
    record->set_hardware_type(accelerator.hardware_type);
    record->set_delegated_ops(accelerator.delegated_node_count);
    record->set_partitions(accelerator.partition_count);
  }
  return result;
}

Expected<std::vector<uint8_t>> EncodeImage(const std::vector<float>& input,
                                           const TensorSpec& spec) {
  for (float value : input) {
    if (!std::isfinite(value)) return Invalid("Nonfinite preprocessed input");
  }
  LITERT_ASSIGN_OR_RETURN(auto size, TensorBytes(spec));
  if (input.size() != size / (spec.type() == FLOAT32 ? sizeof(float) : 1)) {
    return Invalid("Preprocessed image byte count mismatch");
  }
  std::vector<uint8_t> bytes(size);
  if (spec.type() == FLOAT32) {
    std::memcpy(bytes.data(), input.data(), bytes.size());
  } else if ((spec.type() == INT8 || spec.type() == UINT8) && spec.has_scale()) {
    const LiteRtQuantizationPerTensor q = {spec.scale(), spec.zero_point()};
    if (spec.type() == INT8) {
      auto values = QuantizePerTensor<int8_t>(input, q);
      std::memcpy(bytes.data(), values.data(), bytes.size());
    } else {
      bytes = QuantizePerTensor<uint8_t>(input, q);
    }
  } else {
    return Invalid("Unsupported image input type or missing quantization");
  }
  return bytes;
}

Expected<std::vector<std::vector<float>>> Execution::Invoke(
    const std::vector<std::vector<uint8_t>>& inputs, int warmups,
    double& milliseconds) {
  if (inputs.size() != inputs_.size()) return Invalid("Runtime input count mismatch");
  for (size_t i = 0; i < inputs.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto size, inputs_[i].PackedSize());
    if (size != inputs[i].size()) return Invalid("Runtime input byte count mismatch");
    LITERT_RETURN_IF_ERROR(inputs_[i].Write<uint8_t>(
        Span<const uint8_t>(inputs[i].data(), inputs[i].size())));
  }
  for (int i = 0; i < warmups; ++i) {
    LITERT_RETURN_IF_ERROR(model_.Run(static_cast<size_t>(signature_), inputs_, outputs_));
  }
  const auto start = std::chrono::steady_clock::now();
  LITERT_RETURN_IF_ERROR(model_.Run(static_cast<size_t>(signature_), inputs_, outputs_));
  milliseconds = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - start)
                     .count();
  std::vector<std::vector<float>> results;
  for (size_t i = 0; i < outputs_.size(); ++i) {
    const auto& spec = output_specs_[i];
    LITERT_ASSIGN_OR_RETURN(auto bytes, TensorBytes(spec));
    std::vector<float> values(bytes / (spec.type() == FLOAT32 ? sizeof(float) : 1));
    if (spec.type() == FLOAT32) {
      LITERT_RETURN_IF_ERROR(outputs_[i].Read<float>(absl::MakeSpan(values)));
    } else {
      std::vector<uint8_t> raw(bytes);
      LITERT_RETURN_IF_ERROR(outputs_[i].Read<uint8_t>(absl::MakeSpan(raw)));
      for (size_t j = 0; j < raw.size(); ++j) {
        const int code = spec.type() == INT8 && raw[j] >= 128
                             ? static_cast<int>(raw[j]) - 256 : raw[j];
        values[j] = (code - spec.zero_point()) * spec.scale();
      }
    }
    for (float value : values) {
      if (!std::isfinite(value)) return Invalid("Nonfinite runtime output: " + spec.name());
    }
    results.push_back(std::move(values));
  }
  return results;
}

DeviceIdentity DeviceInfo() {
  DeviceIdentity result;
#if defined(__unix__) || defined(__APPLE__)
  struct utsname system;
  if (uname(&system) == 0) {
    result.set_operating_system(std::string(system.sysname) + " " + system.release);
    result.set_architecture(system.machine);
  }
#endif
#if defined(__ANDROID__)
  auto property = [](const char* name) {
    char value[PROP_VALUE_MAX] = {};
    __system_property_get(name, value);
    return std::string(value);
  };
  result.set_model(property("ro.product.model"));
  result.set_build_fingerprint(property("ro.build.fingerprint"));
  result.set_hardware(property("ro.hardware"));
#elif defined(__linux__)
  std::ifstream cpu("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpu, line)) {
    if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0) {
      const auto colon = line.find(':');
      if (colon != std::string::npos) result.set_hardware(line.substr(colon + 1));
      break;
    }
  }
#endif
  return result;
}

Expected<std::vector<FileIdentity>> RuntimeLibraries(
    const EvaluationConfig& config) {
  std::set<std::filesystem::path> paths;
#if defined(__linux__)
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    const auto start = line.find('/');
    if (start == std::string::npos) continue;
    const auto raw_path = line.substr(start);
    if (absl::StrContains(raw_path, ".so")) {
      const auto path = Utf8ToPath(raw_path);
      std::error_code ec;
      if (std::filesystem::is_regular_file(path, ec) && !ec) {
        paths.insert(path);
      }
    }
  }
#endif
  // Requested directories also cover plugins that load lazily on Run.
  for (const auto& raw_dir : {config.backend().runtime_library_dir(),
                              config.backend().dispatch_library_dir(),
                              config.backend().compiler_plugin_library_dir()}) {
    if (raw_dir.empty()) continue;
    const auto dir = Utf8ToPath(raw_dir);
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end;
         it.increment(ec)) {
      const auto entry_path = it->path();
      const auto filename = PathToUtf8(entry_path.filename());
      if (absl::StrContains(filename, ".so") ||
          absl::EndsWithIgnoreCase(filename, ".dll") ||
          absl::EndsWithIgnoreCase(filename, ".dylib")) {
        const auto absolute = std::filesystem::absolute(entry_path, ec);
        if (ec) {
          return Invalid("Cannot resolve library path: " +
                         PathToUtf8(entry_path));
        }
        paths.insert(absolute);
      }
    }
    if (ec) {
      return Invalid("Cannot list library directory: " + raw_dir);
    }
  }
  std::vector<FileIdentity> result;
  for (const auto& path : paths) {
    LITERT_ASSIGN_OR_RETURN(auto identity, Identify(path));
    result.push_back(std::move(identity));
  }
  return result;
}
}  // namespace litert::vision
