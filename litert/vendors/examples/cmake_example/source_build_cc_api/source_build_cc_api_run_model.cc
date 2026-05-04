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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_gpu_options.h"

#ifndef LITERT_EXAMPLE_DEFAULT_MODEL
#define LITERT_EXAMPLE_DEFAULT_MODEL ""
#endif

#ifndef LITERT_EXAMPLE_DEFAULT_GPU_MODEL
#define LITERT_EXAMPLE_DEFAULT_GPU_MODEL ""
#endif

#ifndef LITERT_EXAMPLE_DEFAULT_RUNTIME_LIBRARY_DIR
#define LITERT_EXAMPLE_DEFAULT_RUNTIME_LIBRARY_DIR ""
#endif

namespace {

using TensorBufferMap =
    absl::flat_hash_map<absl::string_view, litert::TensorBuffer>;

struct Config {
  std::string model_path;
  std::string runtime_library_dir = LITERT_EXAMPLE_DEFAULT_RUNTIME_LIBRARY_DIR;
  std::string dispatch_library_dir;
  std::string compiler_plugin_library_dir;
  std::string compiler_cache_dir;
  std::string accelerator = "cpu";
  std::string cpu_kernel_mode = "xnnpack";
  int cpu_threads = 0;
  std::string gpu_precision = "default";
  std::string gpu_buffer_storage = "default";
  std::string gpu_backend = "automatic";
  std::string gpu_priority = "default";
  int gpu_kernel_batch_size = -1;
  std::optional<bool> use_metal_argument_buffers;
  size_t signature_index = 0;
  size_t iterations = 1;
  size_t sample_size = 8;
  bool print_tensors = false;
  bool use_async = false;
  bool use_named_maps = false;
  bool non_strict_resize = false;
  bool resize_input_dims_explicit = false;
  std::vector<int> resize_input_dims;
  std::string input_dir;
};

std::string ToString(absl::string_view value) {
  return std::string(value.data(), value.size());
}

litert::Unexpected InvalidArgument(std::string message) {
  return litert::Unexpected(litert::Status::kErrorInvalidArgument,
                            std::move(message));
}

template <typename T>
std::string SpanToString(absl::Span<const T> values) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << "]";
  return os.str();
}

std::vector<std::string> SplitComma(std::string_view value) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start <= value.size()) {
    const size_t comma = value.find(',', start);
    const size_t end = comma == std::string_view::npos ? value.size() : comma;
    parts.emplace_back(value.substr(start, end - start));
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  return parts;
}

litert::Expected<size_t> ParseSize(std::string_view value,
                                   std::string_view flag_name) {
  try {
    std::string owned(value);
    size_t parsed_chars = 0;
    unsigned long long parsed = std::stoull(owned, &parsed_chars);
    if (parsed_chars != owned.size() ||
        parsed > std::numeric_limits<size_t>::max()) {
      return InvalidArgument("Invalid value for " + std::string(flag_name));
    }
    return static_cast<size_t>(parsed);
  } catch (...) {
    return InvalidArgument("Invalid value for " + std::string(flag_name));
  }
}

litert::Expected<int> ParseInt(std::string_view value,
                               std::string_view flag_name) {
  try {
    std::string owned(value);
    size_t parsed_chars = 0;
    int parsed = std::stoi(owned, &parsed_chars);
    if (parsed_chars != owned.size()) {
      return InvalidArgument("Invalid value for " + std::string(flag_name));
    }
    return parsed;
  } catch (...) {
    return InvalidArgument("Invalid value for " + std::string(flag_name));
  }
}

litert::Expected<bool> ParseBool(std::string_view value,
                                 std::string_view flag_name) {
  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    return false;
  }
  return InvalidArgument("Invalid boolean value for " + std::string(flag_name));
}

litert::Expected<std::vector<int>> ParseDims(std::string_view value) {
  if (value.empty() || value == "none") {
    return std::vector<int>();
  }

  std::vector<int> dims;
  for (const std::string& part : SplitComma(value)) {
    LITERT_ASSIGN_OR_RETURN(int dim, ParseInt(part, "--resize_inputs"));
    if (dim <= 0) {
      return InvalidArgument("--resize_inputs dimensions must be positive");
    }
    dims.push_back(dim);
  }
  if (dims.empty()) {
    return InvalidArgument("--resize_inputs must be a comma-delimited shape");
  }
  return dims;
}

bool AcceleratorListMentionsGpu(std::string_view accelerator_list) {
  for (const std::string& accelerator : SplitComma(accelerator_list)) {
    if (accelerator == "gpu") {
      return true;
    }
  }
  return false;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [--model=/path/model.tflite] [options]\n\n"
      << "Default CPU model:\n"
      << "  " << LITERT_EXAMPLE_DEFAULT_MODEL << "\n"
      << "Default GPU model:\n"
      << "  " << LITERT_EXAMPLE_DEFAULT_GPU_MODEL << "\n\n"
      << "Core options:\n"
      << "  --accelerator=cpu|gpu|npu|cpu,gpu   Default: cpu\n"
      << "  --signature_index=N                 Default: 0\n"
      << "  --iterations=N                      Default: 1\n"
      << "  --print_tensors[=true|false]        Print input/output samples\n"
      << "  --sample_size=N                     Default: 8\n"
      << "  --use_async[=true|false]            Call RunAsync\n"
      << "  --use_named_maps[=true|false]       Run with name->TensorBuffer "
         "maps\n"
      << "  --resize_inputs=d0,d1,...|none      Default: 1,128,4 for the "
         "CPU default model; none otherwise\n"
      << "  --non_strict_resize[=true|false]    Use non-strict resize\n"
      << "  --input_dir=DIR                     Read <input_name>.raw files\n\n"
      << "Runtime lookup options:\n"
      << "  --runtime_library_dir=DIR|none      Directory for accelerator "
         "dylibs\n"
      << "  --dispatch_library_dir=DIR\n"
      << "  --compiler_plugin_library_dir=DIR\n"
      << "  --compiler_cache_dir=DIR\n\n"
      << "CPU options:\n"
      << "  --cpu_kernel_mode=xnnpack|builtin|reference\n"
      << "  --cpu_threads=N\n\n"
      << "GPU options:\n"
      << "  --gpu_precision=default|fp16|fp32\n"
      << "  --gpu_buffer_storage=default|buffer|texture2d\n"
      << "  --gpu_backend=automatic|opencl|opengl|webgpu|metal\n"
      << "  --gpu_priority=default|low|normal|high\n"
      << "  --gpu_kernel_batch_size=N\n"
      << "  --use_metal_argument_buffers=true|false\n";
}

litert::Expected<Config> ParseArgs(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (arg.rfind("--", 0) != 0) {
      config.model_path = arg;
      continue;
    }

    const size_t equal = arg.find('=');
    const std::string key =
        equal == std::string::npos ? arg.substr(2) : arg.substr(2, equal - 2);
    const bool has_inline_value = equal != std::string::npos;
    const std::string inline_value =
        has_inline_value ? arg.substr(equal + 1) : std::string();

    auto require_value = [&]() -> litert::Expected<std::string> {
      if (has_inline_value) {
        return inline_value;
      }
      if (i + 1 >= argc) {
        return InvalidArgument("--" + key + " requires a value");
      }
      return std::string(argv[++i]);
    };
    auto optional_bool_value = [&]() -> litert::Expected<bool> {
      if (!has_inline_value) {
        return true;
      }
      return ParseBool(inline_value, "--" + key);
    };

    if (key == "model" || key == "graph") {
      LITERT_ASSIGN_OR_RETURN(config.model_path, require_value());
    } else if (key == "runtime_library_dir") {
      LITERT_ASSIGN_OR_RETURN(config.runtime_library_dir, require_value());
      if (config.runtime_library_dir == "none") {
        config.runtime_library_dir.clear();
      }
    } else if (key == "dispatch_library_dir") {
      LITERT_ASSIGN_OR_RETURN(config.dispatch_library_dir, require_value());
    } else if (key == "compiler_plugin_library_dir") {
      LITERT_ASSIGN_OR_RETURN(config.compiler_plugin_library_dir,
                              require_value());
    } else if (key == "compiler_cache_dir") {
      LITERT_ASSIGN_OR_RETURN(config.compiler_cache_dir, require_value());
    } else if (key == "accelerator") {
      LITERT_ASSIGN_OR_RETURN(config.accelerator, require_value());
    } else if (key == "signature_index") {
      std::string value;
      LITERT_ASSIGN_OR_RETURN(value, require_value());
      LITERT_ASSIGN_OR_RETURN(config.signature_index,
                              ParseSize(value, "--signature_index"));
    } else if (key == "iterations") {
      std::string value;
      LITERT_ASSIGN_OR_RETURN(value, require_value());
      LITERT_ASSIGN_OR_RETURN(config.iterations,
                              ParseSize(value, "--iterations"));
    } else if (key == "sample_size") {
      std::string value;
      LITERT_ASSIGN_OR_RETURN(value, require_value());
      LITERT_ASSIGN_OR_RETURN(config.sample_size,
                              ParseSize(value, "--sample_size"));
    } else if (key == "print_tensors") {
      LITERT_ASSIGN_OR_RETURN(config.print_tensors, optional_bool_value());
    } else if (key == "use_async") {
      LITERT_ASSIGN_OR_RETURN(config.use_async, optional_bool_value());
    } else if (key == "use_named_maps") {
      LITERT_ASSIGN_OR_RETURN(config.use_named_maps, optional_bool_value());
    } else if (key == "resize_inputs") {
      std::string value;
      LITERT_ASSIGN_OR_RETURN(value, require_value());
      LITERT_ASSIGN_OR_RETURN(config.resize_input_dims, ParseDims(value));
      config.resize_input_dims_explicit = true;
    } else if (key == "non_strict_resize") {
      LITERT_ASSIGN_OR_RETURN(config.non_strict_resize, optional_bool_value());
    } else if (key == "input_dir") {
      LITERT_ASSIGN_OR_RETURN(config.input_dir, require_value());
    } else if (key == "cpu_kernel_mode") {
      LITERT_ASSIGN_OR_RETURN(config.cpu_kernel_mode, require_value());
    } else if (key == "cpu_threads") {
      std::string value;
      LITERT_ASSIGN_OR_RETURN(value, require_value());
      LITERT_ASSIGN_OR_RETURN(config.cpu_threads,
                              ParseInt(value, "--cpu_threads"));
    } else if (key == "gpu_precision") {
      LITERT_ASSIGN_OR_RETURN(config.gpu_precision, require_value());
    } else if (key == "gpu_buffer_storage") {
      LITERT_ASSIGN_OR_RETURN(config.gpu_buffer_storage, require_value());
    } else if (key == "gpu_backend") {
      LITERT_ASSIGN_OR_RETURN(config.gpu_backend, require_value());
    } else if (key == "gpu_priority") {
      LITERT_ASSIGN_OR_RETURN(config.gpu_priority, require_value());
    } else if (key == "gpu_kernel_batch_size") {
      std::string value;
      LITERT_ASSIGN_OR_RETURN(value, require_value());
      LITERT_ASSIGN_OR_RETURN(config.gpu_kernel_batch_size,
                              ParseInt(value, "--gpu_kernel_batch_size"));
    } else if (key == "use_metal_argument_buffers") {
      LITERT_ASSIGN_OR_RETURN(config.use_metal_argument_buffers,
                              optional_bool_value());
    } else {
      return InvalidArgument("Unknown option --" + key);
    }
  }

  if (config.model_path.empty()) {
    if (AcceleratorListMentionsGpu(config.accelerator) &&
        std::string_view(LITERT_EXAMPLE_DEFAULT_GPU_MODEL).size() > 0) {
      config.model_path = LITERT_EXAMPLE_DEFAULT_GPU_MODEL;
    } else {
      config.model_path = LITERT_EXAMPLE_DEFAULT_MODEL;
      if (!config.resize_input_dims_explicit) {
        config.resize_input_dims = {1, 128, 4};
      }
    }
  }

  if (config.model_path.empty()) {
    return InvalidArgument("No model path was provided");
  }
  if (config.iterations == 0) {
    return InvalidArgument("--iterations must be greater than zero");
  }
  return config;
}

std::string ElementTypeToString(litert::ElementType type) {
  switch (type) {
    case litert::ElementType::None:
      return "none";
    case litert::ElementType::Bool:
      return "bool";
    case litert::ElementType::Int2:
      return "int2";
    case litert::ElementType::Int4:
      return "int4";
    case litert::ElementType::Int8:
      return "int8";
    case litert::ElementType::Int16:
      return "int16";
    case litert::ElementType::Int32:
      return "int32";
    case litert::ElementType::Int64:
      return "int64";
    case litert::ElementType::UInt8:
      return "uint8";
    case litert::ElementType::UInt16:
      return "uint16";
    case litert::ElementType::UInt32:
      return "uint32";
    case litert::ElementType::UInt64:
      return "uint64";
    case litert::ElementType::Float16:
      return "float16";
    case litert::ElementType::BFloat16:
      return "bfloat16";
    case litert::ElementType::Float32:
      return "float32";
    case litert::ElementType::Float64:
      return "float64";
    case litert::ElementType::Complex64:
      return "complex64";
    case litert::ElementType::Complex128:
      return "complex128";
    case litert::ElementType::TfResource:
      return "tf_resource";
    case litert::ElementType::TfString:
      return "tf_string";
    case litert::ElementType::TfVariant:
      return "tf_variant";
  }
  return "unknown";
}

void PrintRankedTensorType(const litert::RankedTensorType& type) {
  const litert::Layout& layout = type.Layout();
  auto num_elements = layout.NumElements();
  std::cout << ElementTypeToString(type.ElementType()) << " "
            << SpanToString(layout.Dimensions());
  if (layout.HasStrides()) {
    std::cout << " strides=" << SpanToString(layout.Strides());
  }
  if (num_elements) {
    std::cout << " elements=" << *num_elements;
  } else {
    std::cout << " elements=<dynamic>";
  }
}

void PrintSimpleTensor(const litert::SimpleTensor& tensor,
                       std::string_view indent) {
  std::cout << indent << "tensor_index=" << tensor.TensorIndex()
            << " name=" << tensor.Name()
            << " element_type=" << ElementTypeToString(tensor.ElementType());
  if (tensor.HasQuantization()) {
    std::cout << " quantization_type=" << tensor.QTypeId();
  }
  auto ranked = tensor.RankedTensorType();
  if (ranked) {
    std::cout << " ranked_type=";
    PrintRankedTensorType(*ranked);
  }
  std::cout << "\n";
}

litert::Expected<void> PrintSignatures(
    const litert::CompiledModel& compiled_model) {
  LITERT_ASSIGN_OR_RETURN(auto signatures, compiled_model.GetSignatures());
  std::cout << "Signatures: " << signatures.size() << "\n";
  for (size_t i = 0; i < signatures.size(); ++i) {
    const litert::SimpleSignature& signature = signatures[i];
    std::cout << "  [" << i << "] key=" << signature.Key() << "\n";
    for (size_t input_index = 0; input_index < signature.InputNames().size();
         ++input_index) {
      LITERT_ASSIGN_OR_RETURN(const auto& tensor,
                              signature.InputTensor(input_index));
      std::cout << "    input[" << input_index << "] ";
      PrintSimpleTensor(tensor, "");
    }
    for (size_t output_index = 0; output_index < signature.OutputNames().size();
         ++output_index) {
      LITERT_ASSIGN_OR_RETURN(const auto& tensor,
                              signature.OutputTensor(output_index));
      std::cout << "    output[" << output_index << "] ";
      PrintSimpleTensor(tensor, "");
    }
  }
  return {};
}

litert::Expected<litert::HwAcceleratorSet> ParseAccelerators(
    std::string_view accelerator_list) {
  litert::HwAcceleratorSet accelerators(
      static_cast<int>(litert::HwAccelerators::kNone));
  for (const std::string& accelerator : SplitComma(accelerator_list)) {
    if (accelerator == "none") {
      continue;
    }
    if (accelerator == "cpu") {
      accelerators |= litert::HwAccelerators::kCpu;
    } else if (accelerator == "gpu") {
      accelerators |= litert::HwAccelerators::kGpu;
    } else if (accelerator == "npu") {
      accelerators |= litert::HwAccelerators::kNpu;
    } else {
      return InvalidArgument("Unknown accelerator: " + accelerator);
    }
  }
  return accelerators;
}

litert::Expected<litert::Environment> CreateEnvironment(const Config& config) {
  std::vector<litert::EnvironmentOptions::Option> env_options;
  if (!config.runtime_library_dir.empty()) {
    env_options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kRuntimeLibraryDir,
        absl::string_view(config.runtime_library_dir)});
  }
  if (!config.dispatch_library_dir.empty()) {
    env_options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
        absl::string_view(config.dispatch_library_dir)});
  }
  if (!config.compiler_plugin_library_dir.empty()) {
    env_options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
        absl::string_view(config.compiler_plugin_library_dir)});
  }
  if (!config.compiler_cache_dir.empty()) {
    env_options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kCompilerCacheDir,
        absl::string_view(config.compiler_cache_dir)});
  }

  return litert::Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(env_options)));
}

litert::Expected<void> ConfigureCpuOptions(const Config& config,
                                           litert::Options& options) {
  if (config.cpu_kernel_mode.empty() && config.cpu_threads <= 0) {
    return {};
  }
  LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());

  if (config.cpu_threads > 0) {
    LITERT_RETURN_IF_ERROR(cpu_options.SetNumThreads(config.cpu_threads));
  }

  if (config.cpu_kernel_mode == "xnnpack") {
    LITERT_RETURN_IF_ERROR(
        cpu_options.SetKernelMode(kLiteRtCpuKernelModeXnnpack));
  } else if (config.cpu_kernel_mode == "builtin") {
    LITERT_RETURN_IF_ERROR(
        cpu_options.SetKernelMode(kLiteRtCpuKernelModeBuiltin));
  } else if (config.cpu_kernel_mode == "reference") {
    LITERT_RETURN_IF_ERROR(
        cpu_options.SetKernelMode(kLiteRtCpuKernelModeReference));
  } else if (!config.cpu_kernel_mode.empty()) {
    return InvalidArgument("Invalid --cpu_kernel_mode value");
  }
  return {};
}

litert::Expected<void> ConfigureGpuOptions(const Config& config,
                                           litert::Options& options) {
  LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());

  if (config.gpu_precision == "default") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetPrecision(litert::GpuOptions::Precision::kDefault));
  } else if (config.gpu_precision == "fp16") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp16));
  } else if (config.gpu_precision == "fp32") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp32));
  } else {
    return InvalidArgument("Invalid --gpu_precision value");
  }

  if (config.gpu_buffer_storage == "default") {
    LITERT_RETURN_IF_ERROR(gpu_options.SetBufferStorageType(
        litert::GpuOptions::BufferStorageType::kDefault));
  } else if (config.gpu_buffer_storage == "buffer") {
    LITERT_RETURN_IF_ERROR(gpu_options.SetBufferStorageType(
        litert::GpuOptions::BufferStorageType::kBuffer));
  } else if (config.gpu_buffer_storage == "texture2d") {
    LITERT_RETURN_IF_ERROR(gpu_options.SetBufferStorageType(
        litert::GpuOptions::BufferStorageType::kTexture2D));
  } else {
    return InvalidArgument("Invalid --gpu_buffer_storage value");
  }

  if (config.gpu_backend == "automatic" || config.gpu_backend == "metal") {
    if (config.gpu_backend == "metal") {
      std::cout << "GPU backend 'metal' maps to automatic GPU backend "
                << "selection; on Apple, LiteRT selects Metal by loading "
                << "libLiteRtMetalAccelerator.dylib.\n";
    }
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetBackend(litert::GpuOptions::Backend::kAutomatic));
  } else if (config.gpu_backend == "opencl") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetBackend(litert::GpuOptions::Backend::kOpenCl));
  } else if (config.gpu_backend == "opengl") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetBackend(litert::GpuOptions::Backend::kOpenGl));
  } else if (config.gpu_backend == "webgpu") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetBackend(litert::GpuOptions::Backend::kWebGpu));
  } else {
    return InvalidArgument("Invalid --gpu_backend value");
  }

  if (config.gpu_priority == "default") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetPriority(litert::GpuOptions::Priority::kDefault));
  } else if (config.gpu_priority == "low") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetPriority(litert::GpuOptions::Priority::kLow));
  } else if (config.gpu_priority == "normal") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetPriority(litert::GpuOptions::Priority::kNormal));
  } else if (config.gpu_priority == "high") {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetPriority(litert::GpuOptions::Priority::kHigh));
  } else {
    return InvalidArgument("Invalid --gpu_priority value");
  }

  if (config.gpu_kernel_batch_size > 0) {
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetKernelBatchSize(config.gpu_kernel_batch_size));
  }

#ifdef __APPLE__
  if (config.use_metal_argument_buffers.has_value()) {
    LITERT_RETURN_IF_ERROR(gpu_options.SetUseMetalArgumentBuffers(
        *config.use_metal_argument_buffers));
  }
#else
  if (config.use_metal_argument_buffers.has_value()) {
    return InvalidArgument(
        "--use_metal_argument_buffers is only available on Apple platforms");
  }
#endif

  return {};
}

litert::Expected<litert::Options> CreateOptions(const Config& config) {
  LITERT_ASSIGN_OR_RETURN(litert::HwAcceleratorSet accelerators,
                          ParseAccelerators(config.accelerator));
  LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(accelerators));
  LITERT_RETURN_IF_ERROR(ConfigureCpuOptions(config, options));
  if (accelerators & litert::HwAccelerators::kGpu) {
    LITERT_RETURN_IF_ERROR(ConfigureGpuOptions(config, options));
  }
  return options;
}

litert::Expected<size_t> GetNumElements(const litert::TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  return type.Layout().NumElements();
}

template <typename T>
T SampleValue(size_t element_index, size_t input_index) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>((input_index + 1) * 0.25 +
                          (element_index % 17) * 0.125);
  } else if constexpr (std::is_signed_v<T>) {
    return static_cast<T>((static_cast<int>(element_index % 19) - 9) +
                          static_cast<int>(input_index));
  } else {
    return static_cast<T>((element_index + input_index) % 251);
  }
}

template <typename T>
litert::Expected<void> FillTypedInput(litert::TensorBuffer& buffer,
                                      size_t input_index) {
  LITERT_ASSIGN_OR_RETURN(size_t num_elements, GetNumElements(buffer));
  std::vector<T> data(num_elements);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = SampleValue<T>(i, input_index);
  }
  return buffer.Write<T>(absl::MakeConstSpan(data));
}

litert::Expected<void> FillInputBuffer(litert::TensorBuffer& buffer,
                                       size_t input_index) {
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  switch (type.ElementType()) {
    case litert::ElementType::Float32:
      return FillTypedInput<float>(buffer, input_index);
    case litert::ElementType::Float64:
      return FillTypedInput<double>(buffer, input_index);
    case litert::ElementType::Int8:
      return FillTypedInput<int8_t>(buffer, input_index);
    case litert::ElementType::Int16:
      return FillTypedInput<int16_t>(buffer, input_index);
    case litert::ElementType::Int32:
      return FillTypedInput<int32_t>(buffer, input_index);
    case litert::ElementType::Int64:
      return FillTypedInput<int64_t>(buffer, input_index);
    case litert::ElementType::UInt8:
    case litert::ElementType::Bool:
      return FillTypedInput<uint8_t>(buffer, input_index);
    case litert::ElementType::UInt16:
    case litert::ElementType::Float16:
    case litert::ElementType::BFloat16:
      return FillTypedInput<uint16_t>(buffer, input_index);
    case litert::ElementType::UInt32:
      return FillTypedInput<uint32_t>(buffer, input_index);
    case litert::ElementType::UInt64:
      return FillTypedInput<uint64_t>(buffer, input_index);
    default:
      return buffer.Clear();
  }
}

litert::Expected<std::vector<uint8_t>> ReadFileBytes(
    const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return litert::Unexpected(litert::Status::kErrorFileIO,
                              "Failed to open " + path.string());
  }
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  if (size < 0) {
    return litert::Unexpected(litert::Status::kErrorFileIO,
                              "Failed to size " + path.string());
  }
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!data.empty()) {
    stream.read(reinterpret_cast<char*>(data.data()), data.size());
  }
  if (!stream) {
    return litert::Unexpected(litert::Status::kErrorFileIO,
                              "Failed to read " + path.string());
  }
  return data;
}

litert::Expected<void> FillInputBuffers(
    const Config& config, absl::Span<const absl::string_view> input_names,
    std::vector<litert::TensorBuffer>& input_buffers) {
  if (!config.input_dir.empty()) {
    for (size_t i = 0; i < input_buffers.size(); ++i) {
      const std::filesystem::path raw_path =
          std::filesystem::path(config.input_dir) /
          (ToString(input_names[i]) + ".raw");
      LITERT_ASSIGN_OR_RETURN(std::vector<uint8_t> data,
                              ReadFileBytes(raw_path));
      LITERT_RETURN_IF_ERROR(input_buffers[i].Clear());
      LITERT_RETURN_IF_ERROR(
          input_buffers[i].Write<uint8_t>(absl::MakeConstSpan(data)));
    }
    return {};
  }

  for (size_t i = 0; i < input_buffers.size(); ++i) {
    LITERT_RETURN_IF_ERROR(FillInputBuffer(input_buffers[i], i));
  }
  return {};
}

template <typename T>
void PrintValue(T value) {
  if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
    std::cout << static_cast<int>(value);
  } else {
    std::cout << value;
  }
}

template <typename T>
litert::Expected<void> PrintTypedSamples(litert::TensorBuffer& buffer,
                                         size_t sample_size) {
  LITERT_ASSIGN_OR_RETURN(size_t num_elements, GetNumElements(buffer));
  std::vector<T> data(num_elements);
  LITERT_RETURN_IF_ERROR(buffer.Read<T>(absl::MakeSpan(data)));
  if (data.empty()) {
    std::cout << "    values=[]\n";
    return {};
  }

  const auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
  long double sum = 0;
  for (T value : data) {
    sum += static_cast<long double>(value);
  }

  std::cout << "    stats count=" << data.size() << " min=";
  PrintValue(*min_it);
  std::cout << " max=";
  PrintValue(*max_it);
  std::cout << " avg=" << std::fixed << std::setprecision(6)
            << static_cast<double>(sum / data.size()) << "\n";

  const size_t to_print = std::min(sample_size, data.size());
  std::cout << "    values=[";
  for (size_t i = 0; i < to_print; ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    PrintValue(data[i]);
  }
  if (to_print < data.size()) {
    std::cout << ", ...";
  }
  std::cout << "]\n";
  return {};
}

litert::Expected<void> PrintTensorBuffer(litert::TensorBuffer& buffer,
                                         std::string_view role,
                                         absl::string_view name,
                                         bool print_values,
                                         size_t sample_size) {
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t size, buffer.Size());
  LITERT_ASSIGN_OR_RETURN(size_t packed_size, buffer.PackedSize());
  LITERT_ASSIGN_OR_RETURN(size_t offset, buffer.Offset());
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, buffer.BufferType());

  std::cout << "  " << role << " " << name << "\n"
            << "    tensor_type=";
  PrintRankedTensorType(type);
  std::cout << "\n"
            << "    buffer_type=" << litert::BufferTypeToStringCC(buffer_type)
            << " size=" << size << " packed_size=" << packed_size
            << " offset=" << offset << " gpu_memory="
            << (buffer.IsOpenClMemory() || buffer.IsWebGpuMemory() ||
                        buffer.IsMetalMemory() || buffer.IsVulkanMemory()
                    ? "yes"
                    : "no")
            << "\n";

  if (!print_values) {
    return {};
  }

  switch (type.ElementType()) {
    case litert::ElementType::Float32:
      return PrintTypedSamples<float>(buffer, sample_size);
    case litert::ElementType::Float64:
      return PrintTypedSamples<double>(buffer, sample_size);
    case litert::ElementType::Int8:
      return PrintTypedSamples<int8_t>(buffer, sample_size);
    case litert::ElementType::Int16:
      return PrintTypedSamples<int16_t>(buffer, sample_size);
    case litert::ElementType::Int32:
      return PrintTypedSamples<int32_t>(buffer, sample_size);
    case litert::ElementType::Int64:
      return PrintTypedSamples<int64_t>(buffer, sample_size);
    case litert::ElementType::UInt8:
    case litert::ElementType::Bool:
      return PrintTypedSamples<uint8_t>(buffer, sample_size);
    case litert::ElementType::UInt16:
    case litert::ElementType::Float16:
    case litert::ElementType::BFloat16:
      return PrintTypedSamples<uint16_t>(buffer, sample_size);
    case litert::ElementType::UInt32:
      return PrintTypedSamples<uint32_t>(buffer, sample_size);
    case litert::ElementType::UInt64:
      return PrintTypedSamples<uint64_t>(buffer, sample_size);
    default:
      std::cout << "    value printing is not implemented for "
                << ElementTypeToString(type.ElementType()) << "\n";
      return {};
  }
}

litert::Expected<void> PrintRequirements(
    const litert::TensorBufferRequirements& requirements) {
  LITERT_ASSIGN_OR_RETURN(std::vector<litert::TensorBufferType> types,
                          requirements.SupportedTypes());
  LITERT_ASSIGN_OR_RETURN(size_t size, requirements.BufferSize());
  LITERT_ASSIGN_OR_RETURN(size_t alignment, requirements.Alignment());
  LITERT_ASSIGN_OR_RETURN(auto strides, requirements.Strides());
  std::cout << "    requirements size=" << size << " alignment=" << alignment
            << " supported_types=[";
  for (size_t i = 0; i < types.size(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << litert::BufferTypeToStringCC(types[i]);
  }
  std::cout << "]";
  if (!strides.empty()) {
    std::cout << " strides=" << SpanToString(strides);
  }
  std::cout << "\n";
  return {};
}

litert::Expected<TensorBufferMap> DuplicateNamedBuffers(
    absl::Span<const absl::string_view> names,
    const std::vector<litert::TensorBuffer>& buffers) {
  TensorBufferMap map;
  for (size_t i = 0; i < buffers.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(litert::TensorBuffer duplicate,
                            buffers[i].Duplicate());
    map.emplace(names[i], std::move(duplicate));
  }
  return map;
}

litert::Expected<void> PrintVectorBuffers(
    absl::Span<const absl::string_view> names,
    std::vector<litert::TensorBuffer>& buffers, std::string_view role,
    bool print_values, size_t sample_size) {
  for (size_t i = 0; i < buffers.size(); ++i) {
    LITERT_RETURN_IF_ERROR(PrintTensorBuffer(buffers[i], role, names[i],
                                             print_values, sample_size));
  }
  return {};
}

litert::Expected<void> PrintMapBuffers(
    absl::Span<const absl::string_view> names, TensorBufferMap& buffers,
    std::string_view role, bool print_values, size_t sample_size) {
  for (absl::string_view name : names) {
    auto it = buffers.find(name);
    if (it == buffers.end()) {
      return InvalidArgument("Missing named tensor buffer: " + ToString(name));
    }
    LITERT_RETURN_IF_ERROR(
        PrintTensorBuffer(it->second, role, name, print_values, sample_size));
  }
  return {};
}

litert::Expected<void> ApplyInputResize(
    const Config& config, litert::CompiledModel& compiled_model,
    absl::Span<const absl::string_view> input_names) {
  if (config.resize_input_dims.empty()) {
    return {};
  }
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (config.non_strict_resize) {
      LITERT_RETURN_IF_ERROR(compiled_model.ResizeInputTensorNonStrict(
          config.signature_index, i,
          absl::MakeConstSpan(config.resize_input_dims)));
    } else {
      LITERT_RETURN_IF_ERROR(compiled_model.ResizeInputTensor(
          config.signature_index, i,
          absl::MakeConstSpan(config.resize_input_dims)));
    }
    std::cout << "Resized input " << input_names[i] << " to "
              << SpanToString(absl::MakeConstSpan(config.resize_input_dims))
              << "\n";
  }
  return {};
}

litert::Expected<void> PrintTensorMetadata(
    litert::CompiledModel& compiled_model, size_t signature_index,
    absl::Span<const absl::string_view> input_names,
    absl::Span<const absl::string_view> output_names) {
  std::cout << "Input tensor metadata:\n";
  for (size_t i = 0; i < input_names.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto type, compiled_model.GetInputTensorType(
                                           signature_index, input_names[i]));
    LITERT_ASSIGN_OR_RETURN(
        auto layout, compiled_model.GetInputTensorLayout(signature_index, i));
    LITERT_ASSIGN_OR_RETURN(auto requirements,
                            compiled_model.GetInputBufferRequirements(
                                signature_index, input_names[i]));
    std::cout << "  input[" << i << "] " << input_names[i] << " type=";
    PrintRankedTensorType(type);
    std::cout << " runtime_layout=" << SpanToString(layout.Dimensions())
              << "\n";
    LITERT_RETURN_IF_ERROR(PrintRequirements(requirements));
  }

  std::cout << "Output tensor metadata:\n";
  LITERT_ASSIGN_OR_RETURN(std::vector<litert::Layout> output_layouts,
                          compiled_model.GetOutputTensorLayouts(
                              signature_index, /*update_allocation=*/true));
  for (size_t i = 0; i < output_names.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto type, compiled_model.GetOutputTensorType(
                                           signature_index, output_names[i]));
    LITERT_ASSIGN_OR_RETURN(auto requirements,
                            compiled_model.GetOutputBufferRequirements(
                                signature_index, output_names[i]));
    std::cout << "  output[" << i << "] " << output_names[i] << " type=";
    PrintRankedTensorType(type);
    if (i < output_layouts.size()) {
      std::cout << " runtime_layout="
                << SpanToString(output_layouts[i].Dimensions());
    }
    std::cout << "\n";
    LITERT_RETURN_IF_ERROR(PrintRequirements(requirements));
  }
  return {};
}

litert::Expected<void> ExerciseDispatchAnnotations(
    litert::CompiledModel& compiled_model, size_t signature_index) {
  constexpr absl::string_view kKey = "source_build_cc_api_example";
  constexpr absl::string_view kValue = "cmake";
  auto set =
      compiled_model.SetDispatchAnnotation(signature_index, kKey, kValue);
  if (!set) {
    std::cout << "Dispatch annotations are not available: " << set.Error()
              << "\n";
    return {};
  }
  LITERT_ASSIGN_OR_RETURN(
      auto value, compiled_model.GetDispatchAnnotation(signature_index, kKey));
  std::cout << "Dispatch annotation " << kKey << "="
            << (value.has_value() ? *value : "<unset>") << "\n";
  LITERT_RETURN_IF_ERROR(
      compiled_model.RemoveDispatchAnnotation(signature_index, kKey));
  return {};
}

litert::Expected<void> RunIterations(
    const Config& config, litert::CompiledModel& compiled_model,
    absl::string_view signature_key,
    const std::vector<litert::TensorBuffer>& input_buffers,
    const std::vector<litert::TensorBuffer>& output_buffers,
    TensorBufferMap* input_map, TensorBufferMap* output_map) {
  std::vector<double> timings_us;
  timings_us.reserve(config.iterations);
  bool last_async = false;
  for (size_t i = 0; i < config.iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    if (config.use_named_maps) {
      if (config.use_async) {
        LITERT_RETURN_IF_ERROR(compiled_model.RunAsync(
            signature_key, *input_map, *output_map, last_async));
      } else {
        LITERT_RETURN_IF_ERROR(
            compiled_model.Run(signature_key, *input_map, *output_map));
      }
    } else {
      if (config.use_async) {
        LITERT_RETURN_IF_ERROR(compiled_model.RunAsync(
            config.signature_index, input_buffers, output_buffers, last_async));
      } else {
        LITERT_RETURN_IF_ERROR(compiled_model.Run(
            config.signature_index, input_buffers, output_buffers));
      }
    }
    const auto end = std::chrono::steady_clock::now();
    timings_us.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  const auto [fastest, slowest] =
      std::minmax_element(timings_us.begin(), timings_us.end());
  const double total =
      std::accumulate(timings_us.begin(), timings_us.end(), 0.0);
  std::cout << "Run mode: "
            << (config.use_named_maps ? "named maps" : "ordered vectors")
            << (config.use_async ? " + RunAsync" : " + Run") << "\n"
            << "  async_selected_by_runtime=" << (last_async ? "true" : "false")
            << "\n"
            << "  first_us=" << timings_us.front() << " fastest_us=" << *fastest
            << " slowest_us=" << *slowest
            << " average_us=" << (total / timings_us.size()) << "\n";
  return {};
}

litert::Expected<void> RunExample(const Config& config) {
  if (!std::filesystem::exists(config.model_path)) {
    return litert::Unexpected(litert::Status::kErrorFileIO,
                              "Model does not exist: " + config.model_path);
  }

  std::cout << "Model: " << config.model_path << "\n"
            << "Accelerators: " << config.accelerator << "\n";
  if (!config.runtime_library_dir.empty()) {
    std::cout << "Runtime library dir: " << config.runtime_library_dir << "\n";
  }

  LITERT_ASSIGN_OR_RETURN(litert::Environment env, CreateEnvironment(config));
  std::cout << "Environment support: fp16="
            << (env.SupportsFP16() ? "yes" : "no")
            << " cl_gl_interop=" << (env.SupportsClGlInterop() ? "yes" : "no")
            << " ahwb_cl_interop="
            << (env.SupportsAhwbClInterop() ? "yes" : "no")
            << " ahwb_gl_interop="
            << (env.SupportsAhwbGlInterop() ? "yes" : "no") << "\n";

  LITERT_ASSIGN_OR_RETURN(litert::Options options, CreateOptions(config));
  LITERT_ASSIGN_OR_RETURN(
      litert::CompiledModel compiled_model,
      litert::CompiledModel::Create(env, config.model_path, options));
  compiled_model.SetCancellationFunction([] { return false; });

  LITERT_RETURN_IF_ERROR(PrintSignatures(compiled_model));
  LITERT_ASSIGN_OR_RETURN(auto signature_keys,
                          compiled_model.GetSignatureKeys());
  if (config.signature_index >= signature_keys.size()) {
    return InvalidArgument("--signature_index is out of range");
  }
  const absl::string_view signature_key =
      signature_keys[config.signature_index];
  LITERT_ASSIGN_OR_RETURN(
      auto input_names,
      compiled_model.GetSignatureInputNames(config.signature_index));
  LITERT_ASSIGN_OR_RETURN(
      auto output_names,
      compiled_model.GetSignatureOutputNames(config.signature_index));
  std::cout << "Selected signature[" << config.signature_index
            << "] key=" << signature_key << "\n";

  LITERT_RETURN_IF_ERROR(ApplyInputResize(config, compiled_model,
                                          absl::MakeConstSpan(input_names)));
  LITERT_RETURN_IF_ERROR(PrintTensorMetadata(
      compiled_model, config.signature_index, absl::MakeConstSpan(input_names),
      absl::MakeConstSpan(output_names)));
  LITERT_RETURN_IF_ERROR(
      ExerciseDispatchAnnotations(compiled_model, config.signature_index));

  auto fully_accelerated = compiled_model.IsFullyAccelerated();
  if (fully_accelerated) {
    std::cout << "Compiled model fully accelerated: "
              << (*fully_accelerated ? "yes" : "no") << "\n";
  } else {
    std::cout << "Compiled model fully accelerated: unavailable ("
              << fully_accelerated.Error() << ")\n";
  }

  LITERT_ASSIGN_OR_RETURN(
      std::vector<litert::TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(config.signature_index));
  LITERT_ASSIGN_OR_RETURN(
      std::vector<litert::TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(config.signature_index));

  LITERT_RETURN_IF_ERROR(FillInputBuffers(
      config, absl::MakeConstSpan(input_names), input_buffers));

  std::optional<TensorBufferMap> input_map;
  std::optional<TensorBufferMap> output_map;
  if (config.use_named_maps) {
    LITERT_ASSIGN_OR_RETURN(
        input_map,
        DuplicateNamedBuffers(absl::MakeConstSpan(input_names), input_buffers));
    LITERT_ASSIGN_OR_RETURN(
        output_map, DuplicateNamedBuffers(absl::MakeConstSpan(output_names),
                                          output_buffers));
  }

  std::cout << "Input buffers:\n";
  if (config.use_named_maps) {
    LITERT_RETURN_IF_ERROR(
        PrintMapBuffers(absl::MakeConstSpan(input_names), *input_map, "input",
                        config.print_tensors, config.sample_size));
  } else {
    LITERT_RETURN_IF_ERROR(
        PrintVectorBuffers(absl::MakeConstSpan(input_names), input_buffers,
                           "input", config.print_tensors, config.sample_size));
  }

  LITERT_RETURN_IF_ERROR(RunIterations(
      config, compiled_model, signature_key, input_buffers, output_buffers,
      input_map ? &*input_map : nullptr, output_map ? &*output_map : nullptr));

  std::cout << "Output buffers:\n";
  if (config.use_named_maps) {
    LITERT_RETURN_IF_ERROR(
        PrintMapBuffers(absl::MakeConstSpan(output_names), *output_map,
                        "output", config.print_tensors, config.sample_size));
  } else {
    LITERT_RETURN_IF_ERROR(
        PrintVectorBuffers(absl::MakeConstSpan(output_names), output_buffers,
                           "output", config.print_tensors, config.sample_size));
  }

  return {};
}

int PrintError(std::string_view step, const litert::Error& error) {
  std::cerr << step << " failed: " << error << "\n";
  return EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
  auto config = ParseArgs(argc, argv);
  if (!config) {
    return PrintError("ParseArgs", config.Error());
  }

  auto run = RunExample(*config);
  if (!run) {
    return PrintError("RunExample", run.Error());
  }

  return EXIT_SUCCESS;
}
