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
#include <memory>
#include <ratio>  // NOLINT(build/c++11)
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>
#if defined(_WIN32)
#include <limits>
#endif
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

#include "absl/container/btree_set.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/image.h"
#if defined(_WIN32)
#include "litert/cc/litert_buffer_ref.h"
#endif

namespace litert::vision {
namespace {
Expected<void> ConfigureCpuOptions(CpuOptions& cpu_options,
                                   const BackendOptions& backend,
                                   CpuKernelPreference cpu_kernels) {
  if (backend.num_threads() > 0) {
    LITERT_RETURN_IF_ERROR(cpu_options.SetNumThreads(backend.num_threads()));
  }
  if (cpu_kernels == CpuKernelPreference::kReference) {
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
    } else {
      accelerators |= HwAccelerators::kNpu;
    }
  }
  return accelerators;
}

Expected<void> SetOpenVinoDevice(Options& options, absl::string_view device) {
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

Expected<TensorSpec> Describe(TensorBuffer& buffer, const SimpleTensor& tensor,
                              absl::string_view name) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType type, buffer.TensorType());
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
  for (int32_t dim : type.Layout().Dimensions()) result.add_shape(dim);
  if (result.type() != FLOAT32) {
    if (tensor.QTypeId() != kLiteRtQuantizationPerTensor) {
      return Invalid("Model IO requires per-tensor quantization");
    }
    const LiteRtQuantizationPerTensor quant_params =
        tensor.PerTensorQuantization();
    result.set_scale(quant_params.scale);
    result.set_zero_point(quant_params.zero_point);
  }
  LITERT_RETURN_IF_ERROR(TensorBytes(result));
  return result;
}

}  // namespace

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
      return Invalid(absl::StrCat("Unknown accelerator: ", accelerator));
    }
    npu |= accelerator == "npu";
  }
  if (backend.arm_vulkan_ml_enable_jit() && !npu) {
    return Invalid(
        "Arm Vulkan ML JIT requires accelerator selection to include npu");
  }
  return {};
}

EnvironmentOptions BuildEnvironmentOptions(const BackendOptions& backend) {
  std::vector<EnvironmentOptions::Option> options;
  if (!backend.runtime_library_dir().empty()) {
    options.push_back({EnvironmentOptions::Tag::kRuntimeLibraryDir,
                       backend.runtime_library_dir()});
  }
  if (!backend.dispatch_library_dir().empty()) {
    options.push_back({EnvironmentOptions::Tag::kDispatchLibraryDir,
                       backend.dispatch_library_dir()});
  }
  if (!backend.compiler_plugin_library_dir().empty()) {
    options.push_back({EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
                       backend.compiler_plugin_library_dir()});
  }
  return EnvironmentOptions(options);
}

Expected<Options> CreateOptions(const BackendOptions& backend,
                                CpuKernelPreference cpu_kernels) {
  LITERT_RETURN_IF_ERROR(ValidateBackendOptions(backend));
  LITERT_ASSIGN_OR_RETURN(Options options, Options::Create());
  LITERT_RETURN_IF_ERROR(
      options.SetHardwareAccelerators(GetAccelerators(backend)));
  if (backend.num_threads() > 0 || backend.xnnpack_only() ||
      cpu_kernels == CpuKernelPreference::kReference) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    LITERT_RETURN_IF_ERROR(
        ConfigureCpuOptions(cpu_options, backend, cpu_kernels));
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
    }
    if (backend.gpu_precision() == "fp32") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetPrecision(GpuOptions::Precision::kFp32));
    } else if (backend.gpu_precision() == "fp16") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetPrecision(GpuOptions::Precision::kFp16));
    }
    if (backend.gpu_use_ir_model() != "default") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetUseIrModel(backend.gpu_use_ir_model() == "true"));
    }
    LITERT_RETURN_IF_ERROR(
        gpu_options.EnableBenchmarkMode(backend.gpu_benchmark_mode()));
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
  BackendOptions out = config.backend();
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

Expected<std::unique_ptr<Execution>> Execution::Create(
    Environment& environment, const BackendOptions& backend,
    const std::filesystem::path& path, int signature_index,
    CpuKernelPreference cpu_kernels) {
  LITERT_ASSIGN_OR_RETURN(Options options, CreateOptions(backend, cpu_kernels));
  const auto start = std::chrono::steady_clock::now();
  std::vector<uint8_t> model_bytes;
#if defined(_WIN32)
  // Open via wide `std::filesystem::path` handle on Windows so Unicode and
  // extended-length paths work regardless of the narrow ANSI codepage.
  auto loaded = ReadFileBytes(path, std::numeric_limits<size_t>::max());
  if (!loaded || loaded->empty()) {
    return Invalid(absl::StrCat("Cannot read model file: ", PathToUtf8(path)));
  }
  model_bytes = std::move(*loaded);
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel compiled,
      CompiledModel::Create(environment,
                            BufferRef<uint8_t>(model_bytes.data(),
                                               model_bytes.size()),
                            options));
#else
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel compiled,
      CompiledModel::Create(environment, PathToUtf8(path), options));
#endif
  LITERT_ASSIGN_OR_RETURN(bool is_fully_accelerated,
                          compiled.IsFullyAccelerated());
  if (backend.require_full_acceleration() && !is_fully_accelerated) {
    return Invalid(absl::StrCat("Incomplete acceleration: ", PathToUtf8(path)));
  }
  std::unique_ptr<Execution> result(new Execution(
      std::move(model_bytes), std::move(compiled), signature_index));
  LITERT_ASSIGN_OR_RETURN(result->inputs_,
                          result->model_.CreateInputBuffers(signature_index));
  LITERT_ASSIGN_OR_RETURN(result->outputs_,
                          result->model_.CreateOutputBuffers(signature_index));
  LITERT_ASSIGN_OR_RETURN(SimpleSignature metadata,
                          result->model_.GetSignature(signature_index));
  LITERT_ASSIGN_OR_RETURN(
      std::vector<absl::string_view> input_names,
      result->model_.GetSignatureInputNames(signature_index));
  LITERT_ASSIGN_OR_RETURN(
      std::vector<absl::string_view> output_names,
      result->model_.GetSignatureOutputNames(signature_index));
  for (size_t i = 0; i < result->inputs_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(const SimpleTensor& tensor,
                            metadata.InputTensor(i));
    LITERT_ASSIGN_OR_RETURN(
        TensorSpec spec, Describe(result->inputs_[i], tensor, input_names[i]));
    result->input_specs_.push_back(spec);
    *result->identity_.add_inputs() = std::move(spec);
  }
  for (size_t i = 0; i < result->outputs_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(const SimpleTensor& tensor,
                            metadata.OutputTensor(i));
    LITERT_ASSIGN_OR_RETURN(
        TensorSpec spec,
        Describe(result->outputs_[i], tensor, output_names[i]));
    result->output_specs_.push_back(spec);
    *result->identity_.add_outputs() = std::move(spec);
  }
  LITERT_ASSIGN_OR_RETURN(FileIdentity model_file, Identify(path));
  *result->identity_.mutable_model() = std::move(model_file);
  *result->identity_.mutable_options() = backend;
  result->identity_.set_fully_accelerated(is_fully_accelerated);
  result->identity_.set_setup_ms(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count());
  LITERT_ASSIGN_OR_RETURN(CompiledModel::DelegationMetrics metrics,
                          result->model_.GetDelegationMetrics());
  result->identity_.set_total_runtime_ops(metrics.total_node_count);
  for (const CompiledModel::AcceleratorDelegationMetrics& accelerator :
       metrics.accelerators) {
    AcceleratorMetrics* record = result->identity_.add_accelerators();
    record->set_name(accelerator.accelerator_name);
    record->set_hardware_type(accelerator.hardware_type);
    record->set_delegated_ops(accelerator.delegated_node_count);
    record->set_partitions(accelerator.partition_count);
  }
  return result;
}

Expected<std::vector<uint8_t>> EncodeImage(absl::Span<const float> input,
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
  } else if (spec.has_scale()) {
    const LiteRtQuantizationPerTensor quant_params = {spec.scale(),
                                                      spec.zero_point()};
    if (spec.type() == INT8) {
      auto values = QuantizePerTensor<int8_t>(input, quant_params);
      std::memcpy(bytes.data(), values.data(), bytes.size());
    } else {
      bytes = QuantizePerTensor<uint8_t>(input, quant_params);
    }
  } else {
    return Invalid("Unsupported image input type or missing quantization");
  }
  return bytes;
}

Expected<InvocationResult> Execution::Invoke(
    absl::Span<const std::vector<uint8_t>> inputs, int warmups) {
  if (inputs.size() != inputs_.size()) {
    return Invalid("Runtime input count mismatch");
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(size_t size, inputs_[i].PackedSize());
    if (size != inputs[i].size()) {
      return Invalid("Runtime input byte count mismatch");
    }
    LITERT_RETURN_IF_ERROR(
        inputs_[i].Write<uint8_t>(absl::MakeConstSpan(inputs[i])));
  }
  for (int i = 0; i < warmups; ++i) {
    LITERT_RETURN_IF_ERROR(
        model_.Run(static_cast<size_t>(signature_), inputs_, outputs_));
  }
  const auto start = std::chrono::steady_clock::now();
  LITERT_RETURN_IF_ERROR(
      model_.Run(static_cast<size_t>(signature_), inputs_, outputs_));
  InvocationResult result;
  result.latency_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  for (size_t i = 0; i < outputs_.size(); ++i) {
    const auto& spec = output_specs_[i];
    LITERT_ASSIGN_OR_RETURN(auto bytes, TensorBytes(spec));
    std::vector<float> values(
        bytes / (spec.type() == FLOAT32 ? sizeof(float) : 1));
    if (spec.type() == FLOAT32) {
      LITERT_RETURN_IF_ERROR(outputs_[i].Read<float>(absl::MakeSpan(values)));
    } else {
      std::vector<uint8_t> raw(bytes);
      LITERT_RETURN_IF_ERROR(outputs_[i].Read<uint8_t>(absl::MakeSpan(raw)));
      for (size_t j = 0; j < raw.size(); ++j) {
        const int code = spec.type() == INT8 && raw[j] >= 128
                             ? static_cast<int>(raw[j]) - 256
                             : raw[j];
        values[j] = (code - spec.zero_point()) * spec.scale();
      }
    }
    for (float value : values) {
      if (!std::isfinite(value)) {
        return Invalid(absl::StrCat("Nonfinite runtime output: ", spec.name()));
      }
    }
    result.outputs.push_back(std::move(values));
  }
  return result;
}

namespace internal {

DeviceIdentity ParseCpuInfoHardware(absl::string_view content) {
  DeviceIdentity result;
  for (absl::string_view line : absl::StrSplit(content, '\n')) {
    if (absl::StartsWith(line, "model name") ||
        absl::StartsWith(line, "Hardware")) {
      const auto colon = line.find(':');
      if (colon != absl::string_view::npos) {
        result.set_hardware(line.substr(colon + 1));
      }
      break;
    }
  }
  return result;
}

absl::btree_set<std::string> ParseProcMapsLibraries(absl::string_view maps) {
  absl::btree_set<std::string> paths;
  for (absl::string_view line : absl::StrSplit(maps, '\n')) {
    const auto start = line.find('/');
    if (start == absl::string_view::npos) continue;
    const absl::string_view raw_path = line.substr(start);
    if (absl::StrContains(raw_path, ".so")) {
      const auto path = Utf8ToPath(raw_path);
      std::error_code ec;
      if (std::filesystem::is_regular_file(path, ec)) {
        paths.insert(PathToUtf8(path));
      }
    }
  }
  return paths;
}

}  // namespace internal

DeviceIdentity DeviceInfo() {
  DeviceIdentity result;
#if defined(__unix__) || defined(__APPLE__)
  struct utsname system = {};
  uname(&system);
  result.set_operating_system(
      absl::StrCat(system.sysname, " ", system.release));
  result.set_architecture(system.machine);
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
  if (auto cpu = ReadFileUntilEof("/proc/cpuinfo"); cpu) {
    result.MergeFrom(internal::ParseCpuInfoHardware(*cpu));
  }
#endif
  return result;
}

Expected<std::vector<FileIdentity>> RuntimeLibraries(
    const EvaluationConfig& config) {
  absl::btree_set<std::string> paths;
#if defined(__linux__)
  if (auto maps = ReadFileUntilEof("/proc/self/maps"); maps) {
    paths = internal::ParseProcMapsLibraries(*maps);
  }
#endif
  // Requested directories also cover plugins that load lazily on Run.
  for (const auto& raw_dir : {config.backend().runtime_library_dir(),
                              config.backend().dispatch_library_dir(),
                              config.backend().compiler_plugin_library_dir()}) {
    if (raw_dir.empty()) continue;
    const auto dir = Utf8ToPath(raw_dir);
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; it != end;
         it.increment(ec)) {
      const auto entry_path = it->path();
      const auto filename = PathToUtf8(entry_path.filename());
      if (absl::StrContains(filename, ".so") ||
          absl::EndsWithIgnoreCase(filename, ".dll") ||
          absl::EndsWithIgnoreCase(filename, ".dylib")) {
        const auto absolute = std::filesystem::absolute(entry_path, ec);
        if (ec) {
          return Invalid(absl::StrCat("Cannot resolve library path: ",
                                      PathToUtf8(entry_path)));
        }
        paths.insert(PathToUtf8(absolute));
      }
    }
    if (ec) {
      return Invalid(absl::StrCat("Cannot list library directory: ", raw_dir));
    }
  }
  std::vector<FileIdentity> result;
  for (const auto& raw_path : paths) {
    LITERT_ASSIGN_OR_RETURN(auto identity, Identify(Utf8ToPath(raw_path)));
    result.push_back(std::move(identity));
  }
  return result;
}
}  // namespace litert::vision
