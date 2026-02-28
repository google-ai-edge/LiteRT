// Copyright 2025 Google LLC.
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

#include "litert/c/options/litert_gpu_options.h"

#include <string.h>  // NOLINT: To use strdup in some environments.

#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

using ::litert::ErrorStatusBuilder;
using ::litert::internal::ParseToml;
using ::litert::internal::ParseTomlBool;
using ::litert::internal::ParseTomlInt;
using ::litert::internal::ParseTomlString;
using ::litert::internal::ParseTomlStringArray;

struct LrtGpuOptions {
  // Increment the minor version every time a field is added.
  static constexpr const absl::string_view kIdentifier = "gpu_options";

  std::optional<bool> enable_constant_tensors_sharing;
  std::optional<bool> enable_infinite_float_capping;
  std::optional<bool> benchmark_mode;

  // Added in version 1.2.0.
  std::optional<bool> allow_src_quantized_fc_conv_ops;
  std::optional<LiteRtDelegatePrecision> precision;
  std::optional<LiteRtDelegateBufferStorageType> buffer_storage_type;

  // If true, the delegate will prefer to use textures rather than buffers for
  // weights. Use option when weights in texture has better performance.
  std::optional<bool> prefer_texture_weights;

  // The null-terminated directory to use for serialization.
  // If program_cache_fd is set, this field is ignored for the program cache.
  std::optional<std::string> serialization_dir;

  // The unique null-terminated token string that acts as a 'namespace' for
  // all serialization entries.
  std::optional<std::string> model_cache_key;

  // When set to true AND the serialization_dir and model_cache_key are also
  // set, the delegate will serialize the program cache.
  std::optional<bool> serialize_program_cache;

  // Set to true to serialize immutable external tensors. By default only the
  // non-external tensors are serialized.
  std::optional<bool> serialize_external_tensors;

  // Set to true to run in no external tensors mode. This enables GPU
  // Accelerator using external tensors (PHWC4 format) directly as inputs and
  // outputs.
  std::optional<bool> external_tensors_mode;

  // List of external tensor patterns which are not affected by the no immutable
  // external tensors mode.
  std::vector<std::string> external_tensor_patterns;

  // Added in version 1.4.0.
  // GPU backend to use.
  std::optional<LiteRtGpuBackend> backend;

  // Added in version 2.0.2a1.
  // GPU priority to use.
  std::optional<LiteRtGpuPriority> priority;

  // Added in version 2.0.2a1.
  // Set to true to madvise the original shared tensors after use.
  std::optional<bool> madvise_original_shared_tensors;

  // Added in version 2.0.2a1.
  // Number of steps to prepare WebGPU or Vulkan command buffers in advance.
  std::optional<int> num_steps_of_command_buffer_preparations;

  // Set to true to use Metal argument buffers.
  std::optional<bool> use_metal_argument_buffers;

  // Added in version 2.0.2a1.
  std::optional<LiteRtGpuWaitType> wait_type;

  // Added in version 2.0.2a1.
  // Preferred WebGPU device name substring, case-insensitive.
  // If not empty, the adapter which the device name contains the substring will
  // be chosen.
  // If empty, the device will be determined by other factors.
  std::optional<std::string> preferred_device_substr;

  // Added in version 2.0.2a1.
  // Set to true to hint that the delegate is fully delegated to a single
  // delegate.
  std::optional<bool> hint_fully_delegated_to_single_delegate;

  // Added in version 2.0.2a1.
  // Number of threads for WebGPU upload.
  std::optional<int> num_threads_to_upload;

  // Added in version 2.0.2a1.
  // Number of threads for WebGPU kernel shader compilation.
  std::optional<int> num_threads_to_compile;

  // Added in version 2.0.2a1.
  // Whether to convert weights on GPU.
  // It is not supported by the all backends so this flag is ignored when using
  // non-OpenCL and non-WebGPU backends.
  std::optional<bool> convert_weights_on_gpu;

  // Added in version 2.1.0.
  // Whether to wait for weights conversion on GPU complete.
  // It is not supported by the all backends so this flag is ignored when using
  // non-OpenCL and non-WebGPU backends.
  std::optional<bool> wait_for_weights_conversion_complete;

  // Added in version 2.1.0.
  // Whether to disable Vulkan kernel shader optimization to reduce init time.
  std::optional<bool> disable_shader_optimization;

  // The file descriptor to use for program caching. If set, it overrides the
  // serialization_dir.
  std::optional<int> program_cache_fd;

  // Added in version 2.1.0.
  // If true, only the compiled programs will be cached. If false, gpu graph
  // info including work group sizes (and all compiled programs depending on
  // backend) will be cached.
  std::optional<bool> cache_only_compiled_programs;

  // Added in version 2.1.0.
  // List of prefix patterns of the tensor name that is used for buffer storage
  // type. When it matches, those tensors will use buffer storage type.
  //
  // WARNING: This option is experimental and subject to change.
  std::vector<std::string> buffer_storage_tensor_patterns;
};

LiteRtStatus LrtCreateGpuOptions(LrtGpuOptions** options) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  *options = new LrtGpuOptions();
  if (!*options) return kLiteRtStatusErrorMemoryAllocationFailure;
  return kLiteRtStatusOk;
}

void LrtDestroyGpuOptions(LrtGpuOptions* options) {
  if (options) delete options;
}

const char* LrtGetGpuOptionsIdentifier() { return "gpu_options"; }

LiteRtStatus LrtGetOpaqueGpuOptionsData(const LrtGpuOptions* options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*)) {
  if (!options || !identifier || !payload || !payload_deleter) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  std::stringstream ss;
  if (options->enable_constant_tensors_sharing.has_value()) {
    ss << "enable_constant_tensors_sharing = "
       << (options->enable_constant_tensors_sharing.value() ? "true" : "false")
       << "\n";
  }
  if (options->enable_infinite_float_capping.has_value()) {
    ss << "enable_infinite_float_capping = "
       << (options->enable_infinite_float_capping.value() ? "true" : "false")
       << "\n";
  }
  if (options->benchmark_mode.has_value()) {
    ss << "benchmark_mode = "
       << (options->benchmark_mode.value() ? "true" : "false") << "\n";
  }
  if (options->allow_src_quantized_fc_conv_ops.has_value()) {
    ss << "allow_src_quantized_fc_conv_ops = "
       << (options->allow_src_quantized_fc_conv_ops.value() ? "true" : "false")
       << "\n";
  }
  if (options->precision.has_value()) {
    ss << "precision = " << static_cast<int>(options->precision.value())
       << "\n";
  }
  if (options->buffer_storage_type.has_value()) {
    ss << "buffer_storage_type = "
       << static_cast<int>(options->buffer_storage_type.value()) << "\n";
  }
  if (options->prefer_texture_weights.has_value()) {
    ss << "prefer_texture_weights = "
       << (options->prefer_texture_weights.value() ? "true" : "false") << "\n";
  }
  if (options->serialization_dir.has_value()) {
    ss << "serialization_dir = \"" << options->serialization_dir.value()
       << "\"\n";
  }
  if (options->model_cache_key.has_value()) {
    ss << "model_cache_key = \"" << options->model_cache_key.value() << "\"\n";
  }
  if (options->serialize_program_cache.has_value()) {
    ss << "serialize_program_cache = "
       << (options->serialize_program_cache.value() ? "true" : "false") << "\n";
  }
  if (options->serialize_external_tensors.has_value()) {
    ss << "serialize_external_tensors = "
       << (options->serialize_external_tensors.value() ? "true" : "false")
       << "\n";
  }
  if (options->external_tensors_mode.has_value()) {
    ss << "external_tensors_mode = "
       << (options->external_tensors_mode.value() ? "true" : "false") << "\n";
  }
  if (!options->external_tensor_patterns.empty()) {
    ss << "external_tensor_patterns = [";
    for (size_t i = 0; i < options->external_tensor_patterns.size(); ++i) {
      ss << "\"" << options->external_tensor_patterns[i] << "\"";
      if (i != options->external_tensor_patterns.size() - 1) ss << ", ";
    }
    ss << "]\n";
  }
  if (options->backend.has_value()) {
    ss << "backend = " << static_cast<int>(options->backend.value()) << "\n";
  }
  if (options->priority.has_value()) {
    ss << "priority = " << static_cast<int>(options->priority.value()) << "\n";
  }
  if (options->madvise_original_shared_tensors.has_value()) {
    ss << "madvise_original_shared_tensors = "
       << (options->madvise_original_shared_tensors.value() ? "true" : "false")
       << "\n";
  }
  if (options->num_steps_of_command_buffer_preparations.has_value()) {
    ss << "num_steps_of_command_buffer_preparations = "
       << static_cast<int>(
              options->num_steps_of_command_buffer_preparations.value())
       << "\n";
  }
  if (options->use_metal_argument_buffers.has_value()) {
    ss << "use_metal_argument_buffers = "
       << (options->use_metal_argument_buffers.value() ? "true" : "false")
       << "\n";
  }
  if (options->wait_type.has_value()) {
    ss << "wait_type = " << static_cast<int>(options->wait_type.value())
       << "\n";
  }
  if (options->preferred_device_substr.has_value()) {
    ss << "preferred_device_substr = \""
       << options->preferred_device_substr.value() << "\"\n";
  }
  if (options->hint_fully_delegated_to_single_delegate.has_value()) {
    ss << "hint_fully_delegated_to_single_delegate = "
       << (options->hint_fully_delegated_to_single_delegate.value() ? "true"
                                                                    : "false")
       << "\n";
  }
  if (options->num_threads_to_upload.has_value()) {
    ss << "num_threads_to_upload = "
       << static_cast<int>(options->num_threads_to_upload.value()) << "\n";
  }
  if (options->num_threads_to_compile.has_value()) {
    ss << "num_threads_to_compile = "
       << static_cast<int>(options->num_threads_to_compile.value()) << "\n";
  }
  if (options->convert_weights_on_gpu.has_value()) {
    ss << "convert_weights_on_gpu = "
       << (options->convert_weights_on_gpu.value() ? "true" : "false") << "\n";
  }
  if (options->wait_for_weights_conversion_complete.has_value()) {
    ss << "wait_for_weights_conversion_complete = "
       << (options->wait_for_weights_conversion_complete.value() ? "true"
                                                                 : "false")
       << "\n";
  }
  if (options->disable_shader_optimization.has_value()) {
    ss << "disable_shader_optimization = "
       << (options->disable_shader_optimization.value() ? "true" : "false")
       << "\n";
  }
  if (options->program_cache_fd.has_value()) {
    ss << "program_cache_fd = "
       << static_cast<int>(options->program_cache_fd.value()) << "\n";
  }
  if (options->cache_only_compiled_programs.has_value()) {
    ss << "cache_only_compiled_programs = "
       << (options->cache_only_compiled_programs.value() ? "true" : "false")
       << "\n";
  }
  if (!options->buffer_storage_tensor_patterns.empty()) {
    ss << "buffer_storage_tensor_patterns = [";
    for (size_t i = 0; i < options->buffer_storage_tensor_patterns.size();
         ++i) {
      ss << "\"" << options->buffer_storage_tensor_patterns[i] << "\"";
      if (i != options->buffer_storage_tensor_patterns.size() - 1) ss << ", ";
    }
    ss << "]\n";
  }
  *identifier = LrtGetGpuOptionsIdentifier();
  *payload = strdup(ss.str().c_str());
  *payload_deleter = [](void* p) { free(p); };
  return kLiteRtStatusOk;
}

LiteRtStatus LrtCreateGpuOptionsFromToml(const char* toml_string,
                                         LrtGpuOptions** options) {
  if (!toml_string || !options) return kLiteRtStatusErrorInvalidArgument;
  *options = new LrtGpuOptions();
  if (!*options) return kLiteRtStatusErrorMemoryAllocationFailure;
  absl::string_view toml_view(toml_string);
  if (toml_view.empty()) return kLiteRtStatusOk;
  auto status = ParseToml(
      toml_view,
      [&](absl::string_view key, absl::string_view value) -> LiteRtStatus {
        if (key == "enable_constant_tensors_sharing") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->enable_constant_tensors_sharing = *res;
        } else if (key == "enable_infinite_float_capping") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->enable_infinite_float_capping = *res;
        } else if (key == "benchmark_mode") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->benchmark_mode = *res;
        } else if (key == "allow_src_quantized_fc_conv_ops") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->allow_src_quantized_fc_conv_ops = *res;
        } else if (key == "precision") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->precision = static_cast<LiteRtDelegatePrecision>(*res);
        } else if (key == "buffer_storage_type") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->buffer_storage_type =
              static_cast<LiteRtDelegateBufferStorageType>(*res);
        } else if (key == "prefer_texture_weights") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->prefer_texture_weights = *res;
        } else if (key == "serialization_dir") {
          auto res = ParseTomlString(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->serialization_dir = *res;
        } else if (key == "model_cache_key") {
          auto res = ParseTomlString(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->model_cache_key = *res;
        } else if (key == "serialize_program_cache") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->serialize_program_cache = *res;
        } else if (key == "serialize_external_tensors") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->serialize_external_tensors = *res;
        } else if (key == "external_tensors_mode") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->external_tensors_mode = *res;
        } else if (key == "external_tensor_patterns") {
          auto res = ParseTomlStringArray(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->external_tensor_patterns = *res;
        } else if (key == "backend") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->backend = static_cast<LiteRtGpuBackend>(*res);
        } else if (key == "priority") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->priority = static_cast<LiteRtGpuPriority>(*res);
        } else if (key == "madvise_original_shared_tensors") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->madvise_original_shared_tensors = *res;
        } else if (key == "num_steps_of_command_buffer_preparations") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->num_steps_of_command_buffer_preparations = *res;
        } else if (key == "use_metal_argument_buffers") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->use_metal_argument_buffers = *res;
        } else if (key == "wait_type") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->wait_type = static_cast<LiteRtGpuWaitType>(*res);
        } else if (key == "preferred_device_substr") {
          auto res = ParseTomlString(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->preferred_device_substr = *res;
        } else if (key == "hint_fully_delegated_to_single_delegate") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->hint_fully_delegated_to_single_delegate = *res;
        } else if (key == "num_threads_to_upload") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->num_threads_to_upload = *res;
        } else if (key == "num_threads_to_compile") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->num_threads_to_compile = *res;
        } else if (key == "convert_weights_on_gpu") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->convert_weights_on_gpu = *res;
        } else if (key == "wait_for_weights_conversion_complete") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->wait_for_weights_conversion_complete = *res;
        } else if (key == "disable_shader_optimization") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->disable_shader_optimization = *res;
        } else if (key == "program_cache_fd") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->program_cache_fd = *res;
        } else if (key == "cache_only_compiled_programs") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->cache_only_compiled_programs = *res;
        } else if (key == "buffer_storage_tensor_patterns") {
          auto res = ParseTomlStringArray(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->buffer_storage_tensor_patterns = *res;
        }
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    delete *options;
    *options = nullptr;
    return status;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsConstantTensorsSharing(LrtGpuOptions* gpu_options,
                                                    bool enable) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->enable_constant_tensors_sharing = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsInfiniteFloatCapping(LrtGpuOptions* gpu_options,
                                                  bool enable) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->enable_infinite_float_capping = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsBenchmarkMode(LrtGpuOptions* gpu_options,
                                           bool enable) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->benchmark_mode = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsGpuBackend(LrtGpuOptions* gpu_options,
                                        LiteRtGpuBackend backend) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->backend = backend;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsExternalTensorsMode(LrtGpuOptions* gpu_options,
                                                 bool enable) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->external_tensors_mode = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAddGpuOptionsExternalTensorPattern(LrtGpuOptions* gpu_options,
                                                   const char* pattern) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->external_tensor_patterns.push_back(std::string(pattern));
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAddGpuOptionsBufferStorageTensorPattern(
    LrtGpuOptions* gpu_options, const char* pattern) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->buffer_storage_tensor_patterns.push_back(std::string(pattern));
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsGpuPriority(LrtGpuOptions* gpu_options,
                                         LiteRtGpuPriority priority) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->priority = priority;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    LrtGpuOptions* gpu_options, bool enable) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->allow_src_quantized_fc_conv_ops = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsPrecision(
    LrtGpuOptions* gpu_options, LiteRtDelegatePrecision precision) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->precision = precision;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
    LrtGpuOptions* gpu_options,
    LiteRtDelegateBufferStorageType buffer_storage_type) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->buffer_storage_type = buffer_storage_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsPreferTextureWeights(
    LrtGpuOptions* gpu_options, bool prefer_texture_weights) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->prefer_texture_weights = prefer_texture_weights;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsSerializationDir(
    LrtGpuOptions* gpu_options, const char* serialization_dir) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  if (serialization_dir) {
    gpu_options->serialization_dir = std::string(serialization_dir);
  } else {
    gpu_options->serialization_dir = std::nullopt;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsModelCacheKey(
    LrtGpuOptions* gpu_options, const char* model_cache_key) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  if (model_cache_key) {
    gpu_options->model_cache_key = std::string(model_cache_key);
  } else {
    gpu_options->model_cache_key = std::nullopt;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsProgramCacheFd(
    LrtGpuOptions* gpu_options, int program_cache_fd) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->program_cache_fd = program_cache_fd;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsSerializeProgramCache(
    LrtGpuOptions* gpu_options, bool serialize_program_cache) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->serialize_program_cache = serialize_program_cache;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly(
    LrtGpuOptions* gpu_options, bool cache_only_compiled_programs) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->cache_only_compiled_programs = cache_only_compiled_programs;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
    LrtGpuOptions* gpu_options, bool serialize_external_tensors) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->serialize_external_tensors = serialize_external_tensors;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
    LrtGpuOptions* gpu_options, bool madvise_original_shared_tensors) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->madvise_original_shared_tensors =
      madvise_original_shared_tensors;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
    LrtGpuOptions* gpu_options, bool disable_shader_optimization) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->disable_shader_optimization = disable_shader_optimization;
  return kLiteRtStatusOk;
}

LiteRtStatus
LrtSetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
    LrtGpuOptions* gpu_options, int num_steps_of_command_buffer_preparations) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->num_steps_of_command_buffer_preparations =
      num_steps_of_command_buffer_preparations;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsUseMetalArgumentBuffers(
    LrtGpuOptions* gpu_options, bool use_metal_argument_buffers) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->use_metal_argument_buffers = use_metal_argument_buffers;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsWaitType(
    LrtGpuOptions* gpu_options, LiteRtGpuWaitType wait_type) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->wait_type = wait_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
    LrtGpuOptions* gpu_options, const char* preferred_device_substr) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  if (preferred_device_substr) {
    gpu_options->preferred_device_substr = std::string(preferred_device_substr);
  } else {
    gpu_options->preferred_device_substr = std::nullopt;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
    LrtGpuOptions* gpu_options, int num_threads_to_upload) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->num_threads_to_upload = num_threads_to_upload;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
    LrtGpuOptions* gpu_options, int num_threads_to_compile) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->num_threads_to_compile = num_threads_to_compile;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
    LrtGpuOptions* gpu_options, bool convert_weights_on_gpu) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->convert_weights_on_gpu = convert_weights_on_gpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsWaitForWeightsConversionComplete(
    LrtGpuOptions* gpu_options, bool wait_for_weights_conversion_complete) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->wait_for_weights_conversion_complete =
      wait_for_weights_conversion_complete;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsHintFullyDelegatedToSingleDelegate(
    LrtGpuOptions* gpu_options, bool hint_fully_delegated_to_single_delegate) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->hint_fully_delegated_to_single_delegate =
      hint_fully_delegated_to_single_delegate;
  return kLiteRtStatusOk;
}

const char* LrtGetGpuOptionsPayloadIdentifier() {
  return LrtGpuOptions::kIdentifier.data();
}

LiteRtStatus LrtGetGpuOptionsConstantTensorsSharing(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *enabled = options->enable_constant_tensors_sharing.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsInfiniteFloatCapping(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *enabled = options->enable_infinite_float_capping.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsGpuBackend(LiteRtGpuBackend* backend,
                                        const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(backend, ErrorStatusBuilder::InvalidArgument())
      << "`backend` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *backend = options->backend.value_or(kLiteRtGpuBackendAutomatic);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsBenchmarkMode(bool* enabled,
                                           const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *enabled = options->benchmark_mode.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsExternalTensorsMode(bool* enabled,
                                                 const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *enabled = options->external_tensors_mode.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsGpuPriority(LiteRtGpuPriority* priority,
                                         const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(priority, ErrorStatusBuilder::InvalidArgument())
      << "`priority` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *priority = options->priority.value_or(kLiteRtGpuPriorityDefault);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *enabled = options->allow_src_quantized_fc_conv_ops.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtDelegatePrecision* precision, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(precision, ErrorStatusBuilder::InvalidArgument())
      << "`precision` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *precision = options->precision.value_or(kLiteRtDelegatePrecisionDefault);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsBufferStorageType(
    LiteRtDelegateBufferStorageType* buffer_storage_type,
    const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(buffer_storage_type,
                         ErrorStatusBuilder::InvalidArgument())
      << "`use_buffer_storage_type` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *buffer_storage_type = options->buffer_storage_type.value_or(
      kLiteRtDelegateBufferStorageTypeDefault);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
    bool* prefer_texture_weights, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(prefer_texture_weights,
                         ErrorStatusBuilder::InvalidArgument())
      << "`prefer_texture_weights` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *prefer_texture_weights = options->prefer_texture_weights.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
    const char** serialization_dir, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(serialization_dir,
                         ErrorStatusBuilder::InvalidArgument())
      << "`serialization_dir` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *serialization_dir = options->serialization_dir.has_value()
                           ? options->serialization_dir->c_str()
                           : nullptr;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
    const char** model_cache_key, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(model_cache_key, ErrorStatusBuilder::InvalidArgument())
      << "`model_cache_key` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *model_cache_key = options->model_cache_key.has_value()
                         ? options->model_cache_key->c_str()
                         : nullptr;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
    int* program_cache_fd, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(program_cache_fd,
                         ErrorStatusBuilder::InvalidArgument())
      << "`program_cache_fd` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *program_cache_fd = options->program_cache_fd.value_or(-1);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
    bool* serialize_program_cache, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(serialize_program_cache,
                         ErrorStatusBuilder::InvalidArgument())
      << "`serialize_program_cache` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *serialize_program_cache = options->serialize_program_cache.value_or(true);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly(
    bool* cache_only_compiled_programs, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(cache_only_compiled_programs,
                         ErrorStatusBuilder::InvalidArgument())
      << "`cache_only_compiled_programs` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *cache_only_compiled_programs =
      options->cache_only_compiled_programs.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
    bool* serialize_external_tensors, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(serialize_external_tensors,
                         ErrorStatusBuilder::InvalidArgument())
      << "`serialize_external_tensors` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *serialize_external_tensors =
      options->serialize_external_tensors.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetNumGpuAcceleratorCompilationOptionsExternalTensorPatterns(
    int* num_patterns, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(num_patterns, ErrorStatusBuilder::InvalidArgument())
      << "`num_patterns` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *num_patterns = options->external_tensor_patterns.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsExternalTensorPattern(
    const char** external_tensor_pattern, int pattern_index,
    const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(external_tensor_pattern,
                         ErrorStatusBuilder::InvalidArgument())
      << "`external_tensor_pattern` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (pattern_index >= options->external_tensor_patterns.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *external_tensor_pattern =
      options->external_tensor_patterns[pattern_index].c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus
LrtGetNumGpuAcceleratorCompilationOptionsBufferStorageTensorPatterns(
    int* num_patterns, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(num_patterns, ErrorStatusBuilder::InvalidArgument())
      << "`num_patterns` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *num_patterns = options->buffer_storage_tensor_patterns.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsBufferStorageTensorPattern(
    const char** buffer_storage_tensor_pattern, int pattern_index,
    const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(buffer_storage_tensor_pattern,
                         ErrorStatusBuilder::InvalidArgument())
      << "`buffer_storage_tensor_pattern` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (pattern_index >= options->buffer_storage_tensor_patterns.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *buffer_storage_tensor_pattern =
      options->buffer_storage_tensor_patterns[pattern_index].c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
    bool* madvise_original_shared_tensors, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(madvise_original_shared_tensors,
                         ErrorStatusBuilder::InvalidArgument())
      << "`madvise_original_shared_tensors` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *madvise_original_shared_tensors =
      options->madvise_original_shared_tensors.value_or(0);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
    bool* disable_shader_optimization, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(disable_shader_optimization,
                         ErrorStatusBuilder::InvalidArgument())
      << "`disable_shader_compilation_optimization` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *disable_shader_optimization =
      options->disable_shader_optimization.value_or(0);
  return kLiteRtStatusOk;
}

LiteRtStatus
LrtGetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
    int* num_steps_of_command_buffer_preparations,
    const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(num_steps_of_command_buffer_preparations,
                         ErrorStatusBuilder::InvalidArgument())
      << "`num_steps_of_command_buffer_preparations` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *num_steps_of_command_buffer_preparations =
      options->num_steps_of_command_buffer_preparations.value_or(0);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsUseMetalArgumentBuffers(
    const LrtGpuOptions* options, bool* use_metal_argument_buffers) {
  LITERT_RETURN_IF_ERROR(use_metal_argument_buffers,
                         ErrorStatusBuilder::InvalidArgument())
      << "`use_metal_argument_buffers` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *use_metal_argument_buffers =
      options->use_metal_argument_buffers.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsWaitType(
    LiteRtGpuWaitType* wait_type, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(wait_type, ErrorStatusBuilder::InvalidArgument())
      << "`wait_type` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *wait_type = options->wait_type.value_or(kLiteRtGpuWaitTypeDefault);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
    const char** preferred_device_substr, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(preferred_device_substr,
                         ErrorStatusBuilder::InvalidArgument())
      << "`preferred_device_substr` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *preferred_device_substr = options->preferred_device_substr.has_value()
                                 ? options->preferred_device_substr->c_str()
                                 : "";
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
    int* num_threads_to_upload, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(num_threads_to_upload,
                         ErrorStatusBuilder::InvalidArgument())
      << "`num_threads_to_upload` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *num_threads_to_upload = options->num_threads_to_upload.value_or(0);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
    int* num_threads_to_compile, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(num_threads_to_compile,
                         ErrorStatusBuilder::InvalidArgument())
      << "`num_threads_to_compile` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *num_threads_to_compile = options->num_threads_to_compile.value_or(0);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
    bool* convert_weights_on_gpu, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(convert_weights_on_gpu,
                         ErrorStatusBuilder::InvalidArgument())
      << "`convert_weights_on_gpu` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *convert_weights_on_gpu = options->convert_weights_on_gpu.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsWaitForWeightsConversionComplete(
    bool* wait_for_weights_conversion_complete, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(wait_for_weights_conversion_complete,
                         ErrorStatusBuilder::InvalidArgument())
      << "`wait_for_weights_conversion_complete` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *wait_for_weights_conversion_complete =
      options->wait_for_weights_conversion_complete.value_or(false);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
    bool* hint_fully_delegated_to_single_delegate,
    const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(hint_fully_delegated_to_single_delegate,
                         ErrorStatusBuilder::InvalidArgument())
      << "`hint_fully_delegated_to_single_delegate` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  *hint_fully_delegated_to_single_delegate =
      options->hint_fully_delegated_to_single_delegate.value_or(false);
  return kLiteRtStatusOk;
}
