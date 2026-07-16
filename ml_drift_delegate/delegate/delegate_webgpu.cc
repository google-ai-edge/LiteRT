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

#include "ml_drift_delegate/delegate/delegate_webgpu.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#if !defined(__EMSCRIPTEN__)
#include "dawn/dawn_proc.h"  // from @dawn
#include "dawn/dawn_proc_table.h"  // from @dawn
#include "webgpu/webgpu.h"  // from @dawn
#endif  // !defined(__EMSCRIPTEN__)
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/webgpu/environment.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/instance.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_api_util.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/cache/simple_cache.h"
#include "ml_drift_delegate/delegate/cache/webgpu_pipeline_cache.h"
#include "ml_drift_delegate/delegate/composite/custom_parsers.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_kernel_litert.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/delegate_types.h"
#include "ml_drift_delegate/delegate/delegate_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend_webgpu_litert.h"
#include "ml_drift_delegate/delegate/precision.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/task_executor.h"
#include "ml_drift_delegate/tflite/model_builder.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_types.h"
#include "tflite/core/c/common.h"
#include "util/hash/farmhash_fingerprint.h"

using ::litert::ml_drift::DelegateKernelLiteRt;
using ::litert::ml_drift::MlDriftDelegateData;

namespace {

// Max number of entries in the WebGPU pipeline cache.
constexpr size_t kMaxNumEntriesInWebGpuPipelineCache = 1024;

// A heuristic to destroy the WebGPU pipeline cache after a few invokes.
constexpr int kInvokeCountToDestroyWebGpuPipelineCache = 5;

struct DelegateEnvironment {
  std::unique_ptr<ml_drift::webgpu::ExecutionEnvironment> webgpu_env;
  std::unique_ptr<litert::ml_drift::WebGpuPipelineCache> pipeline_cache;
};

// Shell environment variables to debug tflite on GPU with WebGPU delegate.
constexpr char kEnvDebugEndNode[] = "LITERT_GPU_DEBUG_END_NODE";
constexpr char kEnvDebugExcludeNodes[] = "LITERT_GPU_DEBUG_EXCLUDE_NODES";

void DestroyDelegateEnvironment(void* user_data) {
  delete reinterpret_cast<DelegateEnvironment*>(user_data);
  ABSL_VLOG(1) << "Destroyed delegate environment.";
}

// Callback called by Dawn native to load cached data if any.
size_t CacheLoad(std::span<const std::byte> key,
                 std::span<std::byte> value,
                 void* userdata) {
  auto* delegate_env = reinterpret_cast<DelegateEnvironment*>(userdata);
  if (delegate_env == nullptr || delegate_env->pipeline_cache == nullptr) {
    return 0;
  }

  uint64_t key_hash = farmhash::Fingerprint64(
      reinterpret_cast<const char*>(key.data()), key.size());
  return delegate_env->pipeline_cache->Load(
      key_hash,
      absl::MakeSpan(reinterpret_cast<uint8_t*>(value.data()), value.size()));
}

// Callback called by Dawn native to store cached data.
void CacheStore(std::span<const std::byte> key,
                std::span<const std::byte> data,
                void* userdata) {
  auto* delegate_env = reinterpret_cast<DelegateEnvironment*>(userdata);
  if (delegate_env == nullptr || delegate_env->pipeline_cache == nullptr) {
    return;
  }

  uint64_t key_hash = farmhash::Fingerprint64(
      reinterpret_cast<const char*>(key.data()), key.size());
  delegate_env->pipeline_cache->Store(
      key_hash,
      absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(data.data()),
                          data.size()));
}

void CacheDetach() ABSL_NO_THREAD_SAFETY_ANALYSIS {
  // No-op: pipeline_cache is owned by DelegateEnvironment and destroyed on env
  // close.
}

std::unique_ptr<ml_drift::webgpu::ExecutionEnvironment> CreateWebGpuEnvironment(
    LiteRtEnvironment litert_env,
    GpuPriority gpu_priority,
    litert::ml_drift::SimpleCache&& pipeline_cache,
    const LiteRtRuntimeContext* runtime_context,
    DelegateEnvironment* delegate_env) {
  auto webgpu_env = std::make_unique<ml_drift::webgpu::ExecutionEnvironment>(
#if defined(__APPLE__)
      wgpu::BackendType::Metal
#elif defined(_WIN32)
      wgpu::BackendType::D3D12
#elif defined(__EMSCRIPTEN__)
      wgpu::BackendType::WebGPU
#else
      wgpu::BackendType::Vulkan
#endif
  );
  LiteRtEnvironmentOptions env_options;
  runtime_context->get_environment_options(litert_env, &env_options);
  LiteRtAny wegpu_device_id;
  auto wgpu_device_id_status = runtime_context->get_environment_options_value(
      env_options, kLiteRtEnvOptionTagWebGpuDevice, &wegpu_device_id);

#if !defined(__EMSCRIPTEN__)
  LiteRtAny wegpu_procs;
  if (runtime_context->get_environment_options_value(
          env_options, kLiteRtEnvOptionTagWebGpuProcs, &wegpu_procs) ==
          kLiteRtStatusOk &&
      wegpu_procs.int_value != 0) {
    dawnProcSetProcs(
        reinterpret_cast<const DawnProcTable*>(wegpu_procs.int_value));
  }
#endif  // !defined(__EMSCRIPTEN__)
  LiteRtAny wegpu_instance;
  if (runtime_context->get_environment_options_value(
          env_options, kLiteRtEnvOptionTagWebGpuInstance, &wegpu_instance) ==
          kLiteRtStatusOk &&
      wegpu_instance.int_value != 0) {
    WGPUInstance wgpu_inst =
        reinterpret_cast<WGPUInstance>(wegpu_instance.int_value);
    wgpu::Instance inst = wgpu_inst;
    (void)::ml_drift::webgpu::Instance::Set(inst);
  }
  LiteRtAny wegpu_flush;
  if (runtime_context->get_environment_options_value(
          env_options, kLiteRtEnvOptionTagWebGpuFlushCallback, &wegpu_flush) ==
          kLiteRtStatusOk &&
      wegpu_flush.int_value != 0) {
    ::ml_drift::webgpu::SetFlushCallback(
        reinterpret_cast<::ml_drift::webgpu::WebGpuFlushCallback>(
            wegpu_flush.int_value));
  }

  absl::Status webgpu_init_status;
  std::string success_message;
  if (wgpu_device_id_status == kLiteRtStatusOk) {
    // Use the WebGPU device id provided by the client.
    WGPUDevice wgpu_device =
        reinterpret_cast<WGPUDevice>(wegpu_device_id.int_value);
    wgpu::Device device = wgpu_device;
    wgpu::AdapterInfo adapter_info;
    device.GetAdapterInfo(&adapter_info);
    webgpu_init_status = webgpu_env->Initialize(device, adapter_info);
    success_message = "Created a WebGPU environment with provided device.";
  } else {
#ifdef __EMSCRIPTEN__
    WGPUDevice ems_device = emscripten_webgpu_get_device();
    auto device = wgpu::Device::Acquire(ems_device);
    wgpu::AdapterInfo adapter_info;
    device.GetAdapterInfo(&adapter_info);
    webgpu_init_status = webgpu_env->Initialize(device, adapter_info);
    success_message =
        "Created a WebGPU environment with emscripten_webgpu_get_device().";
#else
    const bool use_low_power = (gpu_priority == kGpuLowPriority);
    const bool enable_host_mapped_pointer = true;
    ABSL_LOG(INFO) << "Create WebGPU environment (use_low_power="
                   << use_low_power << ", enable_host_mapped_pointer="
                   << enable_host_mapped_pointer << ")";
    // Create ExecutionEnvironment with InitParams.
    ml_drift::webgpu::Environment::InitParams init_params{
        .use_low_power = use_low_power,
        .enable_host_mapped_pointer = enable_host_mapped_pointer};
    wgpu::DawnCacheDeviceDescriptor cache_desc;
    cache_desc.SetDawnLoadCacheDataCallback(&CacheLoad, delegate_env);
    cache_desc.SetDawnStoreCacheDataCallback(&CacheStore, delegate_env);
    if (pipeline_cache.IsValid() && delegate_env != nullptr) {
      delegate_env->pipeline_cache =
          std::make_unique<litert::ml_drift::WebGpuPipelineCache>(
              std::move(pipeline_cache), kMaxNumEntriesInWebGpuPipelineCache);
      init_params.cache_descriptor = &cache_desc;
    }
    webgpu_init_status = webgpu_env->Initialize(init_params);
    success_message = "Created a WebGPU environment.";
#endif
  }

  if (!webgpu_init_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to initialize WebGPU environment: "
                    << webgpu_init_status;
    return nullptr;
  }

  ABSL_VLOG(1) << success_message;
  return webgpu_env;
};

litert::Expected<ml_drift::webgpu::ExecutionEnvironment*>
GetSingletonWebGpuEnvironment(LiteRtEnvironment litert_env,
                              GpuPriority gpu_priority,
                              litert::ml_drift::SimpleCache&& pipeline_cache,
                              const LiteRtRuntimeContext* runtime_context) {
  auto resources = std::make_unique<DelegateEnvironment>();

  bool has_gpu_environment = false;
  runtime_context->environment_has_gpu_environment(litert_env,
                                                   &has_gpu_environment);
  if (has_gpu_environment) {
    LiteRtEnvironmentOptions env_options;
    LITERT_RETURN_IF_ERROR(
        runtime_context->get_environment_options(litert_env, &env_options));
    LiteRtAny user_data;
    auto status = runtime_context->get_environment_options_value(
        env_options, kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
        &user_data);
    if (status == kLiteRtStatusOk && user_data.ptr_value != nullptr) {
      return reinterpret_cast<DelegateEnvironment*>(
                 const_cast<void*>(user_data.ptr_value))
          ->webgpu_env.get();
    }
  }

  resources->webgpu_env = CreateWebGpuEnvironment(
      litert_env, gpu_priority, std::move(pipeline_cache), runtime_context,
      resources.get());
  if (!resources->webgpu_env) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create WebGPU environment");
  }

  LITERT_ASSIGN_OR_RETURN(LiteRtAny callback,
                          litert::ToLiteRtAny(reinterpret_cast<const void*>(
                              &DestroyDelegateEnvironment)));
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny user_data,
      litert::ToLiteRtAny(reinterpret_cast<const void*>(resources.get())));

  if (has_gpu_environment) {
    std::array<LiteRtEnvOption, 2> options = {
        LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy,
                        .value = callback},
        LiteRtEnvOption{
            .tag = kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
            .value = user_data},
    };
    LITERT_RETURN_IF_ERROR(runtime_context->add_environment_options(
        litert_env, options.size(), options.data(), /*overwrite=*/true));
    return resources.release()->webgpu_env.get();
  }

  LITERT_ASSIGN_OR_RETURN(LiteRtAny device_id,
                          litert::ToLiteRtAny(reinterpret_cast<int64_t>(
                              resources->webgpu_env->device().Get())));
  LITERT_ASSIGN_OR_RETURN(LiteRtAny command_queue,
                          litert::ToLiteRtAny(reinterpret_cast<int64_t>(
                              resources->webgpu_env->queue().Get())));
  LITERT_ASSIGN_OR_RETURN(LiteRtAny wgpu_instance,
                          litert::ToLiteRtAny(reinterpret_cast<int64_t>(
                              &ml_drift::webgpu::Instance::Get())));
  std::array<LiteRtEnvOption, 5> options = {
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagWebGpuDevice,
                      .value = device_id},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagWebGpuQueue,
                      .value = command_queue},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy,
                      .value = callback},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
                      .value = user_data},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagWebGpuInstance,
                      .value = wgpu_instance},
  };

  LITERT_RETURN_IF_ERROR(runtime_context->gpu_environment_create(
      litert_env, options.size(), options.data()));

  return resources.release()->webgpu_env.get();
}

void* Init(TfLiteContext* context, const char* buffer, size_t) {
  auto kernel = litert::ml_drift::DelegateKernelLiteRt::Create(
      context, reinterpret_cast<const TfLiteDelegateParams*>(buffer));
  if (!kernel.ok()) {
    ABSL_LOG(ERROR)
        << "Failed to create litert::ml_drift::DelegateKernelLiteRt: "
        << kernel.status();
    return TfLiteKernelInitFailed();
  }
  return *kernel;
}

void Free(TfLiteContext*, void* buffer) {
  delete reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* delegate_kernel =
      reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(
          node->user_data);
  if (delegate_kernel == nullptr ||
      delegate_kernel == TfLiteKernelInitFailed()) {
    ABSL_LOG(ERROR) << "Delegate kernel initialization failed.";
    return kTfLiteError;
  }
  if (absl::Status s = delegate_kernel->GetRequiredTemporaries(
          context, node, &node->temporaries);
      !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
  CacheDetach();

  auto* delegate_kernel =
      reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(
          node->user_data);
  if (delegate_kernel->HasQuantizedTensors()) {
    if (absl::Status s = delegate_kernel->DequantizeInputs(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  }
  if (delegate_kernel->NoExternalTensorsMode()) {
    if (absl::Status s = delegate_kernel->UploadOrBindTensorBuffer(context);
        !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  } else {
    if (absl::Status s = delegate_kernel->BindTensorBuffers(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  }

  if (absl::Status s = delegate_kernel->HandleInputEvents(context); !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }

  if (absl::Status s = delegate_kernel->Dispatch(context); !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }

  if (absl::Status s = delegate_kernel->HandleOutputEvents(
          context, litert::ml_drift::IsAsyncExecutionMode(
                       context, delegate_kernel->runtime_context()));
      !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }

  if (delegate_kernel->NoExternalTensorsMode()) {
    // Download internal output GPU memory to output TensorBufferGPU memory.
    if (absl::Status s =
            delegate_kernel->DownloadGpuMemoryToTensorBufferGpuMemory(context);
        !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  }

  if (delegate_kernel->IsBenchmarkMode()) {
    // In benchmark mode, call WaitForCompletion() to wait for all the
    // enqueued commands to be completed.
    if (auto s = delegate_kernel->backend()->WaitForCompletion(); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  }

  if (absl::Status s = delegate_kernel->DownloadGpuMemoryToCpuMemory(context);
      !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }
  if (delegate_kernel->HasQuantizedTensors()) {
    if (absl::Status s = delegate_kernel->QuantizeOutputs(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  // Check ML Drift op compatibility.
  const absl::flat_hash_set<TfLiteBuiltinOperator> kExcludedOps = {};
  const auto& delegate_options =
      reinterpret_cast<litert::ml_drift::MlDriftDelegateData*>(delegate->data_)
          ->options;

  int start_node_index = 0;
  int end_node_index = std::numeric_limits<int>::max();
  if (delegate_options->debug_delegate_partition) {
    start_node_index = delegate_options->debug_first_delegate_node_index;
    end_node_index = delegate_options->debug_last_delegate_node_index;
#if defined(__linux__)
  } else if (auto* env_debug_end_node = std::getenv(kEnvDebugEndNode)) {
    TfLiteNode* node = nullptr;
    TfLiteRegistration* reg = nullptr;
    int end_node_index_from_env = 0;
    if (absl::SimpleAtoi(env_debug_end_node, &end_node_index_from_env) &&
        context->GetNodeAndRegistration(context, end_node_index_from_env,
                                        &node, &reg) == kTfLiteOk &&
        reg != nullptr) {
      end_node_index = end_node_index_from_env;
      ABSL_LOG(INFO) << kEnvDebugEndNode << " set to " << end_node_index
                     << ". Restricting WebGPU delegation from node 0 to node "
                     << end_node_index << ": code=" << reg->builtin_code;
    }
#endif  // defined(__linux__)
  }
  litert::ml_drift::CustomOperationParserFactory custom_parser_factory;
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(
      context, /*allow_quant_ops=*/true, /*max_delegated_partitions=*/1,
      &kExcludedOps, start_node_index, end_node_index, &custom_parser_factory);

#if defined(__linux__)
  if (auto* env_debug_exclude_nodes = std::getenv(kEnvDebugExcludeNodes)) {
    absl::flat_hash_set<int> excluded_nodes;
    for (absl::string_view s : absl::StrSplit(env_debug_exclude_nodes, ',')) {
      int node_idx;
      if (absl::SimpleAtoi(s, &node_idx)) {
        excluded_nodes.insert(node_idx);
      }
    }
    ABSL_LOG(INFO) << kEnvDebugExcludeNodes << " set to "
                   << absl::StrJoin(excluded_nodes, ",");

    int new_size = 0;
    for (int i = 0; i < ops_to_replace->size; ++i) {
      int node_idx = ops_to_replace->data[i];
      if (excluded_nodes.contains(node_idx)) {
        ABSL_LOG(INFO) << "Excluding node " << node_idx << " (" << i
                       << " in ops_to_replace) from WebGPU delegation.";
      } else {
        ops_to_replace->data[new_size++] = node_idx;
      }
    }
    ops_to_replace->size = new_size;
  }
#endif  // defined(__linux__)

  // Replace the ops with delegate kernel.
  const TfLiteRegistration kRegistration = {
      .init = Init,
      .free = Free,
      .prepare = Prepare,
      .invoke = Invoke,
      .profiling_string = nullptr,
      .builtin_code = 0,
      .custom_name = "LITERT_WEBGPU",
      .version = 1,
      .registration_external = nullptr,
      .async_kernel = nullptr,
  };
  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace

extern "C" {

void LiteRtDeleteMlDriftWebGpuDelegate(TfLiteDelegate* delegate) {
  if (!delegate) {
    return;
  }

  auto* delegate_data =
      reinterpret_cast<litert::ml_drift::MlDriftDelegateData*>(delegate->data_);
  if (delegate_data->weights_conversion_counter) {
    delegate_data->weights_conversion_counter->Wait();
  }
  delete delegate_data;
  delete delegate;
}

}  // extern "C"

namespace litert::ml_drift {

// Returns default options for ML Drift WebGpu delegate.
//
// This calls `MlDriftClDelegateDefaultOptions()` add return the result in an
// RAII wrapper.
MlDriftDelegateOptionsPtr MlDriftWebGpuDelegateDefaultOptionsPtr() {
  return std::make_unique<MlDriftDelegateOptions>(MlDriftDelegateOptions{
      .precision = MlDriftDelegatePrecision::kDefault,
      .debug_last_delegate_node_index = std::numeric_limits<int>::max(),
      .enable_fast_tuning = true,
      // Note that the program cache is not serialized unless serialization_dir
      // and model_token are also set.
      .serialize_program_cache = true,
      .madvise_original_shared_tensors = true,
      // Don't wait for GPU completion on synchronous execution mode for
      // backward compatibility.
      .wait_type = kGpuDelegateWaitTypeDoNotWait,
  });
}

// Creates a new ML Drift WebGpu delegate object.
TfLiteDelegatePtr CreateMlDriftWebGpuDelegate(MlDriftDelegateOptionsPtr options,
                                              LiteRtEnvironment litert_env) {
  if (!options) {
    ABSL_LOG(ERROR) << "Missing MLDrift delegate options";
    return {nullptr, LiteRtDeleteMlDriftWebGpuDelegate};
  }

  if (options->litert_benchmark_mode) {
    ABSL_LOG(INFO) << "Benchmark mode is enabled.";
  }
  if (options->litert_external_tensors_mode) {
    ABSL_LOG(INFO) << "External tensors mode is enabled.";
  }

  // Initialize delegate_data.
  auto delegate_data =
      std::make_unique<litert::ml_drift::MlDriftDelegateData>();
  delegate_data->options = std::move(options);
  delegate_data->weight_loader = delegate_data->options->weight_loader;

  // Copy serialization options since they are not owned by the delegate.
  if (delegate_data->options->serialization_dir) {
    delegate_data->serialization_dir =
        delegate_data->options->serialization_dir;
  }
  if (delegate_data->options->model_token) {
    delegate_data->model_token = delegate_data->options->model_token;
  }

  litert::ml_drift::SimpleCache compiled_cache;
  if (delegate_data->options->cache_compiled_programs_only) {
    if (delegate_data->options->program_cache_fd > 0) {
      compiled_cache = litert::ml_drift::SimpleCache(
          ::ml_drift::FileDescriptor(delegate_data->options->program_cache_fd));
    } else if (delegate_data->options->serialize_program_cache) {
      compiled_cache = litert::ml_drift::SimpleCache(
          delegate_data->serialization_dir, delegate_data->model_token);
    }
  }

  const LiteRtRuntimeContext* runtime_context =
      delegate_data->options->runtime_context;
  if (runtime_context == nullptr) {
    ABSL_LOG(ERROR) << "Missing LiteRT runtime context.";
    return {nullptr, LiteRtDeleteMlDriftWebGpuDelegate};
  }

  // Use the shared WebGPU environment in LiteRT runtime.
  auto webgpu_env = GetSingletonWebGpuEnvironment(
      litert_env, delegate_data->options->gpu_priority,
      std::move(compiled_cache), runtime_context);
  if (!webgpu_env) {
    ABSL_LOG(ERROR) << "Failed to get WebGPU environment: "
                    << webgpu_env.Error();
    return {nullptr, LiteRtDeleteMlDriftWebGpuDelegate};
  }

  auto backend = std::make_shared<GpuBackendWebGpuLitert>(
      *webgpu_env,
      /*strict_error_handling=*/delegate_data->options->litert_benchmark_mode,
      runtime_context);
  backend->set_num_steps_of_command_buffer_preparations(
      delegate_data->options->num_steps_of_command_buffer_preparations);
  delegate_data->shared_backend = std::move(backend);

#if !defined(__EMSCRIPTEN__)
  // Set up the executors.
  ABSL_LOG(INFO) << "# of threads to upload weights = "
                 << delegate_data->options->num_threads_to_upload;
  if (delegate_data->options->num_threads_to_upload > 0) {
    delegate_data->upload_executor =
        std::make_shared<TaskExecutor>(
            "WGPU_Upload", delegate_data->options->num_threads_to_upload);
  }

  ABSL_LOG(INFO) << "# of threads to compile kernels = "
                 << delegate_data->options->num_threads_to_compile;
  if (delegate_data->options->num_threads_to_compile > 0) {
    (*webgpu_env)->GetComputePipelineCache()->set_executor(
        std::make_unique<TaskExecutor>(
            "WGPU_Compile", delegate_data->options->num_threads_to_compile));
  }
#endif  // !defined(__EMSCRIPTEN__)

  switch (delegate_data->options->precision) {
    case kDefault:
      delegate_data->calculation_precision =
          (*webgpu_env)->GetInfo().SupportsFP16()
              ? ::ml_drift::CalculationsPrecision::F16
              : ::ml_drift::CalculationsPrecision::F32;
      break;
    case kFp16:
      delegate_data->calculation_precision =
          ::ml_drift::CalculationsPrecision::F16;
      break;
    case kFp32:
      delegate_data->calculation_precision =
          ::ml_drift::CalculationsPrecision::F32;
      break;
  }
  bool hint_fully_delegated_to_single_delegate =
      delegate_data->options->hint_fully_delegated_to_single_delegate;

  // Initialize the ml_drift WebGpu delegate.
  TfLiteDelegatePtr delegate(new TfLiteDelegate(TfLiteDelegateCreate()),
                             LiteRtDeleteMlDriftWebGpuDelegate);
  delegate->data_ = delegate_data.release();
  delegate->Prepare = DelegatePrepare;
  if (hint_fully_delegated_to_single_delegate) {
    delegate->flags |= kTfLiteDelegateFlagsHintFullyDelegatedToSingleDelegate;
  }
  return delegate;
}

}  // namespace litert::ml_drift
