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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_options_encoder.h"
#include "litert/core/litert_toml_parser.h"

using ::litert::ErrorStatusBuilder;
using ::litert::internal::ParseToml;
using ::litert::internal::ParseTomlBool;
using ::litert::internal::ParseTomlInt;
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

  // Set to true to use Metal residency sets to prevent memory swapping.
  std::optional<bool> enable_metal_residency_set;

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

  // The file descriptor to use for weight caching. If set, it overrides the
  // serialization_dir.
  std::optional<int> weight_cache_fd;

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

  // Added in version 2.1.0.
  // If true, the delegate will wait for gpu completion after each invoke.
  // Only for specific model and specific chipset, e.g. Open source models on
  // OpenCL on AMD and Mali GPUs, when lower quality is observed.
  std::optional<bool> hint_waiting_for_completion;
  // Added in version 2.1.5.
  // If > 0, specifies the kernel (op) batch size, for a flush.
  // Only for OpenCL backend.
  std::optional<int> kernel_batch_size;

  // Added in version 2.1.7.
  // Pointer to SharedTensorMaps to share pre-loaded weights between models.
  std::optional<void*> shared_tensor_maps;

  // Added in version 2.2.1.
  // If true, the delegate initializes via the IrModel pipeline instead of the
  // legacy GraphFloat32 pipeline.
  std::optional<bool> use_ir_model;

  // If true, tuning is faster but the resulting inference can be slower.
  std::optional<bool> enable_fast_tuning;

  // Per-op profiling and its detailed report.
  std::optional<bool> enable_op_profiling;
  std::optional<bool> enable_op_profiling_detailed_report;

  // Debugging: restrict delegation to a node range.
  std::optional<bool> debug_delegate_partition;
  std::optional<int> debug_first_delegate_node_index;
  std::optional<int> debug_last_delegate_node_index;
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
  litert::internal::OptionsEncoder encoder;
  if (options->enable_constant_tensors_sharing.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("enable_constant_tensors_sharing",
                           options->enable_constant_tensors_sharing.value()));
  }
  if (options->enable_infinite_float_capping.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("enable_infinite_float_capping",
                           options->enable_infinite_float_capping.value()));
  }
  if (options->benchmark_mode.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("benchmark_mode", options->benchmark_mode.value()));
  }
  if (options->allow_src_quantized_fc_conv_ops.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("allow_src_quantized_fc_conv_ops",
                           options->allow_src_quantized_fc_conv_ops.value()));
  }
  if (options->hint_waiting_for_completion.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("hint_waiting_for_completion",
                           options->hint_waiting_for_completion.value()));
  }
  if (options->kernel_batch_size.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "kernel_batch_size", options->kernel_batch_size.value()));
  }
  if (options->shared_tensor_maps.has_value() &&
      *(options->shared_tensor_maps) != nullptr) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "shared_tensor_maps",
        reinterpret_cast<int64_t>(*(options->shared_tensor_maps))));
  }
  if (options->precision.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "precision", static_cast<int>(options->precision.value())));
  }
  if (options->buffer_storage_type.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "buffer_storage_type",
        static_cast<int>(options->buffer_storage_type.value())));
  }
  if (options->prefer_texture_weights.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "prefer_texture_weights", options->prefer_texture_weights.value()));
  }
  if (options->serialization_dir.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeString(
        "serialization_dir", options->serialization_dir.value()));
  }
  if (options->model_cache_key.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeString(
        "model_cache_key", options->model_cache_key.value()));
  }
  if (options->serialize_program_cache.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "serialize_program_cache", options->serialize_program_cache.value()));
  }
  if (options->serialize_external_tensors.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("serialize_external_tensors",
                           options->serialize_external_tensors.value()));
  }
  if (options->external_tensors_mode.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "external_tensors_mode", options->external_tensors_mode.value()));
  }
  if (!options->external_tensor_patterns.empty()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeStringArray(
        "external_tensor_patterns", options->external_tensor_patterns));
  }
  if (options->backend.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "backend", static_cast<int>(options->backend.value())));
  }
  if (options->priority.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "priority", static_cast<int>(options->priority.value())));
  }
  if (options->madvise_original_shared_tensors.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("madvise_original_shared_tensors",
                           options->madvise_original_shared_tensors.value()));
  }
  if (options->num_steps_of_command_buffer_preparations.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "num_steps_of_command_buffer_preparations",
        options->num_steps_of_command_buffer_preparations.value()));
  }
  if (options->use_metal_argument_buffers.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("use_metal_argument_buffers",
                           options->use_metal_argument_buffers.value()));
  }
  if (options->enable_metal_residency_set.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("enable_metal_residency_set",
                           options->enable_metal_residency_set.value()));
  }
  if (options->wait_type.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "wait_type", static_cast<int>(options->wait_type.value())));
  }
  if (options->preferred_device_substr.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeString(
        "preferred_device_substr", options->preferred_device_substr.value()));
  }
  if (options->hint_fully_delegated_to_single_delegate.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "hint_fully_delegated_to_single_delegate",
        options->hint_fully_delegated_to_single_delegate.value()));
  }
  if (options->num_threads_to_upload.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "num_threads_to_upload", options->num_threads_to_upload.value()));
  }
  if (options->num_threads_to_compile.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "num_threads_to_compile", options->num_threads_to_compile.value()));
  }
  if (options->convert_weights_on_gpu.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "convert_weights_on_gpu", options->convert_weights_on_gpu.value()));
  }
  if (options->wait_for_weights_conversion_complete.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "wait_for_weights_conversion_complete",
        options->wait_for_weights_conversion_complete.value()));
  }
  if (options->disable_shader_optimization.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("disable_shader_optimization",
                           options->disable_shader_optimization.value()));
  }
  if (options->program_cache_fd.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeInt(
        "program_cache_fd", options->program_cache_fd.value()));
  }
  if (options->weight_cache_fd.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeInt("weight_cache_fd", options->weight_cache_fd.value()));
  }
  if (options->cache_only_compiled_programs.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("cache_only_compiled_programs",
                           options->cache_only_compiled_programs.value()));
  }
  if (!options->buffer_storage_tensor_patterns.empty()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeStringArray("buffer_storage_tensor_patterns",
                                  options->buffer_storage_tensor_patterns));
  }
  if (options->use_ir_model.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeBool("use_ir_model", options->use_ir_model.value()));
  }
  if (options->enable_fast_tuning.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "enable_fast_tuning", options->enable_fast_tuning.value()));
  }
  if (options->enable_op_profiling.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "enable_op_profiling", options->enable_op_profiling.value()));
  }
  if (options->enable_op_profiling_detailed_report.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "enable_op_profiling_detailed_report",
        options->enable_op_profiling_detailed_report.value()));
  }
  if (options->debug_delegate_partition.has_value()) {
    LITERT_RETURN_IF_ERROR(encoder.EncodeBool(
        "debug_delegate_partition", options->debug_delegate_partition.value()));
  }
  if (options->debug_first_delegate_node_index.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeInt("debug_first_delegate_node_index",
                          options->debug_first_delegate_node_index.value()));
  }
  if (options->debug_last_delegate_node_index.has_value()) {
    LITERT_RETURN_IF_ERROR(
        encoder.EncodeInt("debug_last_delegate_node_index",
                          options->debug_last_delegate_node_index.value()));
  }
  *identifier = LrtGetGpuOptionsIdentifier();
  litert::internal::MakeCStringPayload(encoder.Str(), payload, payload_deleter);
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
        } else if (key == "hint_waiting_for_completion") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->hint_waiting_for_completion = *res;
        } else if (key == "kernel_batch_size") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->kernel_batch_size = *res;
        } else if (key == "shared_tensor_maps") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->shared_tensor_maps = reinterpret_cast<void*>(*res);
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
          (*options)->serialization_dir = std::string(value);
        } else if (key == "model_cache_key") {
          (*options)->model_cache_key = std::string(value);
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
        } else if (key == "enable_metal_residency_set") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->enable_metal_residency_set = *res;
        } else if (key == "wait_type") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->wait_type = static_cast<LiteRtGpuWaitType>(*res);
        } else if (key == "preferred_device_substr") {
          (*options)->preferred_device_substr = std::string(value);
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
        } else if (key == "weight_cache_fd") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->weight_cache_fd = *res;
        } else if (key == "cache_only_compiled_programs") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->cache_only_compiled_programs = *res;
        } else if (key == "buffer_storage_tensor_patterns") {
          auto res = ParseTomlStringArray(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->buffer_storage_tensor_patterns = *res;
        } else if (key == "use_ir_model") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->use_ir_model = *res;
        } else if (key == "enable_fast_tuning") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->enable_fast_tuning = *res;
        } else if (key == "enable_op_profiling") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->enable_op_profiling = *res;
        } else if (key == "enable_op_profiling_detailed_report") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->enable_op_profiling_detailed_report = *res;
        } else if (key == "debug_delegate_partition") {
          auto res = ParseTomlBool(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->debug_delegate_partition = *res;
        } else if (key == "debug_first_delegate_node_index") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->debug_first_delegate_node_index = *res;
        } else if (key == "debug_last_delegate_node_index") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->debug_last_delegate_node_index = *res;
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

LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsHintWaitingForCompletion(
    LrtGpuOptions* gpu_options, bool enable) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->hint_waiting_for_completion = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsKernelBatchSize(
    LrtGpuOptions* gpu_options, int kernel_batch_size) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->kernel_batch_size = kernel_batch_size;
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

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsWeightCacheFd(
    LrtGpuOptions* gpu_options, int weight_cache_fd) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->weight_cache_fd = weight_cache_fd;
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

LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsSharedTensorMaps(
    LrtGpuOptions* gpu_options, void* shared_tensor_maps) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->shared_tensor_maps = shared_tensor_maps;
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

LiteRtStatus LrtSetGpuOptionsMetalResidencySet(LrtGpuOptions* gpu_options,
                                               bool enable) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->enable_metal_residency_set = enable;
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

LiteRtStatus LrtSetGpuOptionsUseIrModel(LrtGpuOptions* gpu_options,
                                        bool enable) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->use_ir_model = enable;
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
  if (!options->enable_constant_tensors_sharing.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->enable_constant_tensors_sharing;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsInfiniteFloatCapping(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->enable_infinite_float_capping.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->enable_infinite_float_capping;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsGpuBackend(LiteRtGpuBackend* backend,
                                        const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(backend, ErrorStatusBuilder::InvalidArgument())
      << "`backend` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->backend.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *backend = *options->backend;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsBenchmarkMode(bool* enabled,
                                           const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->benchmark_mode.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->benchmark_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsExternalTensorsMode(bool* enabled,
                                                 const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->external_tensors_mode.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->external_tensors_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsGpuPriority(LiteRtGpuPriority* priority,
                                         const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(priority, ErrorStatusBuilder::InvalidArgument())
      << "`priority` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->priority.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *priority = *options->priority;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->allow_src_quantized_fc_conv_ops.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->allow_src_quantized_fc_conv_ops;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsHintWaitingForCompletion(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->hint_waiting_for_completion.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->hint_waiting_for_completion;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsKernelBatchSize(
    int* kernel_batch_size, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(kernel_batch_size,
                         ErrorStatusBuilder::InvalidArgument())
      << "`kernel_batch_size` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->kernel_batch_size.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *kernel_batch_size = *options->kernel_batch_size;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtDelegatePrecision* precision, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(precision, ErrorStatusBuilder::InvalidArgument())
      << "`precision` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->precision.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *precision = *options->precision;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsBufferStorageType(
    LiteRtDelegateBufferStorageType* buffer_storage_type,
    const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(buffer_storage_type,
                         ErrorStatusBuilder::InvalidArgument())
      << "`buffer_storage_type` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->buffer_storage_type.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *buffer_storage_type = *options->buffer_storage_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->prefer_texture_weights.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->prefer_texture_weights;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
    const char** serialization_dir, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(serialization_dir,
                         ErrorStatusBuilder::InvalidArgument())
      << "`serialization_dir` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->serialization_dir.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *serialization_dir = options->serialization_dir->c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
    const char** model_cache_key, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(model_cache_key, ErrorStatusBuilder::InvalidArgument())
      << "`model_cache_key` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->model_cache_key.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *model_cache_key = options->model_cache_key->c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
    int* program_cache_fd, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(program_cache_fd,
                         ErrorStatusBuilder::InvalidArgument())
      << "`program_cache_fd` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->program_cache_fd.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *program_cache_fd = *options->program_cache_fd;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsWeightCacheFd(
    int* weight_cache_fd, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(weight_cache_fd, ErrorStatusBuilder::InvalidArgument())
      << "`weight_cache_fd` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->weight_cache_fd.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *weight_cache_fd = *options->weight_cache_fd;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->serialize_program_cache.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->serialize_program_cache;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->cache_only_compiled_programs.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->cache_only_compiled_programs;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->serialize_external_tensors.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->serialize_external_tensors;
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
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->madvise_original_shared_tensors.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->madvise_original_shared_tensors;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->disable_shader_optimization.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->disable_shader_optimization;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSharedTensorMaps(
    void** shared_tensor_maps, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(shared_tensor_maps,
                         ErrorStatusBuilder::InvalidArgument())
      << "`shared_tensor_maps` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->shared_tensor_maps.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *shared_tensor_maps = *options->shared_tensor_maps;
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
  if (!options->num_steps_of_command_buffer_preparations.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *num_steps_of_command_buffer_preparations =
      *options->num_steps_of_command_buffer_preparations;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsUseMetalArgumentBuffers(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->use_metal_argument_buffers.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->use_metal_argument_buffers;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsMetalResidencySet(bool* enabled,
                                               const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->enable_metal_residency_set.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->enable_metal_residency_set;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsWaitType(
    LiteRtGpuWaitType* wait_type, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(wait_type, ErrorStatusBuilder::InvalidArgument())
      << "`wait_type` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->wait_type.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *wait_type = *options->wait_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
    const char** preferred_device_substr, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(preferred_device_substr,
                         ErrorStatusBuilder::InvalidArgument())
      << "`preferred_device_substr` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->preferred_device_substr.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *preferred_device_substr = options->preferred_device_substr->c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
    int* num_threads_to_upload, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(num_threads_to_upload,
                         ErrorStatusBuilder::InvalidArgument())
      << "`num_threads_to_upload` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->num_threads_to_upload.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *num_threads_to_upload = *options->num_threads_to_upload;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
    int* num_threads_to_compile, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(num_threads_to_compile,
                         ErrorStatusBuilder::InvalidArgument())
      << "`num_threads_to_compile` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->num_threads_to_compile.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *num_threads_to_compile = *options->num_threads_to_compile;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->convert_weights_on_gpu.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->convert_weights_on_gpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsWaitForWeightsConversionComplete(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->wait_for_weights_conversion_complete.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->wait_for_weights_conversion_complete;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->hint_fully_delegated_to_single_delegate.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->hint_fully_delegated_to_single_delegate;
  return kLiteRtStatusOk;
}
LiteRtStatus LrtGetGpuOptionsUseIrModel(bool* enabled,
                                        const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
#if defined(LITERT_GPU_USE_IR_MODEL)
  *enabled = options->use_ir_model.value_or(LITERT_GPU_USE_IR_MODEL);
  return kLiteRtStatusOk;
#else
  if (!options->use_ir_model.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->use_ir_model;
  return kLiteRtStatusOk;
#endif
}

LiteRtStatus LrtSetGpuOptionsEnableFastTuning(LrtGpuOptions* gpu_options,
                                              bool enable_fast_tuning) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->enable_fast_tuning = enable_fast_tuning;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsEnableFastTuning(bool* enabled,
                                              const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->enable_fast_tuning.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->enable_fast_tuning;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsEnableOpProfiling(LrtGpuOptions* gpu_options,
                                               bool enable_op_profiling) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->enable_op_profiling = enable_op_profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsEnableOpProfiling(bool* enabled,
                                               const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->enable_op_profiling.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->enable_op_profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsEnableOpProfilingDetailedReport(
    LrtGpuOptions* gpu_options, bool enable_op_profiling_detailed_report) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->enable_op_profiling_detailed_report =
      enable_op_profiling_detailed_report;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsEnableOpProfilingDetailedReport(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->enable_op_profiling_detailed_report.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->enable_op_profiling_detailed_report;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsDebugDelegatePartition(
    LrtGpuOptions* gpu_options, bool debug_delegate_partition) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->debug_delegate_partition = debug_delegate_partition;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsDebugDelegatePartition(
    bool* enabled, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->debug_delegate_partition.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enabled = *options->debug_delegate_partition;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsDebugFirstDelegateNodeIndex(
    LrtGpuOptions* gpu_options, int debug_first_delegate_node_index) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->debug_first_delegate_node_index =
      debug_first_delegate_node_index;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsDebugFirstDelegateNodeIndex(
    int* debug_first_delegate_node_index, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(debug_first_delegate_node_index,
                         ErrorStatusBuilder::InvalidArgument())
      << "`debug_first_delegate_node_index` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->debug_first_delegate_node_index.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *debug_first_delegate_node_index =
      *options->debug_first_delegate_node_index;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetGpuOptionsDebugLastDelegateNodeIndex(
    LrtGpuOptions* gpu_options, int debug_last_delegate_node_index) {
  if (!gpu_options) return kLiteRtStatusErrorInvalidArgument;

  gpu_options->debug_last_delegate_node_index = debug_last_delegate_node_index;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetGpuOptionsDebugLastDelegateNodeIndex(
    int* debug_last_delegate_node_index, const LrtGpuOptions* options) {
  LITERT_RETURN_IF_ERROR(debug_last_delegate_node_index,
                         ErrorStatusBuilder::InvalidArgument())
      << "`debug_last_delegate_node_index` cannot be null.";
  LITERT_RETURN_IF_ERROR(options, ErrorStatusBuilder::InvalidArgument())
      << "`options` cannot be null.";
  if (!options->debug_last_delegate_node_index.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *debug_last_delegate_node_index =
      *options->debug_last_delegate_node_index;
  return kLiteRtStatusOk;
}

