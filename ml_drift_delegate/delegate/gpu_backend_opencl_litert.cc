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

#include "ml_drift_delegate/delegate/gpu_backend_opencl_litert.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/log/die_if_null.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/cl_event.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/memory_manager.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/serialization_generated.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_generated.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/serialization_base_generated.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_opencl_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/cache/simple_cache.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/gpu_backend_opencl.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/mmap_handle.h"
#include <CL/cl_platform.h>

#if LITERT_HAS_OPENGL_SUPPORT
#include <EGL/egl.h>

#include "ml_drift/cl/gl_interop.h"  // from @ml_drift
#endif

namespace litert::ml_drift {
namespace {

// A heuristic to flush the compiled shader program cache after a few invokes.
constexpr int kInvokeCountToFlushCompiledCache = 5;

uint64_t SaturatingAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a) {
    return std::numeric_limits<uint64_t>::max();
  }
  return a + b;
}

uint64_t SumFlops(absl::Span<const uint64_t> node_flops) {
  uint64_t total = 0;
  for (uint64_t flops : node_flops) {
    total = SaturatingAdd(total, flops);
  }
  return total;
}

std::vector<uint64_t> CollectNodeFlops(const ::ml_drift::GpuModel& gpu_model) {
  std::vector<uint64_t> node_flops;
  node_flops.reserve(gpu_model.nodes.size());
  for (const auto& node : gpu_model.nodes) {
    node_flops.push_back(node.gpu_operation ? node.gpu_operation->flops_ : 0);
  }
  return node_flops;
}

std::vector<uint64_t> CollectSerializedNodeFlops(
    absl::Span<const uint8_t> serialized_model) {
  if (serialized_model.empty()) {
    return {};
  }
  const auto* decoded_fb =
      ::ml_drift::cl::data::GetInferenceContext(serialized_model.data());
  if (decoded_fb == nullptr || decoded_fb->gpu_model() == nullptr ||
      decoded_fb->gpu_model()->nodes() == nullptr) {
    return {};
  }
  const auto* nodes = decoded_fb->gpu_model()->nodes();
  std::vector<uint64_t> node_flops;
  node_flops.reserve(nodes->size());
  for (const auto* node : *nodes) {
    node_flops.push_back(node != nullptr && node->gpu_op() != nullptr
                             ? node->gpu_op()->flops()
                             : 0);
  }
  return node_flops;
}

}  // namespace

uint64_t GpuBackendOpenClLitert::ResolveAutoBatchFlopsBudget(
    ::ml_drift::CalculationsPrecision precision) const {
  if (kernel_batch_size_ > 0) {
    return 0;
  }
  return ComputeMaxFlopsPerKernelBatch(cl_env()->device().GetInfo(), precision);
}

void GpuBackendOpenClLitert::MaybeSetAutoKernelBatchSizes(
    absl::Span<const uint64_t> node_flops, uint64_t max_flops_per_batch,
    GpuInferenceContextOpenClLitert* ctx, const char* log_prefix) {
  if (max_flops_per_batch == 0 || node_flops.empty()) {
    return;
  }
  std::vector<int> auto_batch_sizes =
      PartitionNodesByFlops(node_flops, max_flops_per_batch);
  if (auto_batch_sizes.empty()) {
    return;
  }
  LITERT_LOG(LITERT_INFO,
             "%s into %zu FLOPs-bounded batches "
             "(nodes=%zu, total_flops=%llu, budget=%llu).",
             log_prefix, auto_batch_sizes.size(), node_flops.size(),
             static_cast<unsigned long long>(SumFlops(node_flops)),
             static_cast<unsigned long long>(max_flops_per_batch));
  ctx->set_auto_kernel_batch_sizes(std::move(auto_batch_sizes));
}

uint64_t GpuBackendOpenClLitert::ComputeMaxFlopsPerKernelBatch(
    const ::ml_drift::GpuInfo& gpu_info,
    ::ml_drift::CalculationsPrecision precision) {
  if (!gpu_info.IsAdreno() || !gpu_info.adreno_info.IsAdreno8xx()) {
    return 0;
  }
  // On Adreno Gen 8 (8xx), `cl_qcom_recordable_queues` packs the entire graph
  // into a single IB1 submission (`ts`), which can trip the GMU long-IB
  // hardware watchdog (~9-10 s, `Fault id: 2`) on compute-heavy graphs such as
  // high-resolution ViT encoders (`vision_1120`, ~8.16 TFLOPs) in both FP32
  // and thermally throttled FP16 (`902 MHz` -> ~70 GFLOP/s/CU effective on
  // 12-CU Adreno 840). A per-CU budget of 50 GFLOPs (FP16) / 20 GFLOPs (FP32 /
  // F32F16) targets ~0.5-0.7 s per flushed batch, keeping small and decode
  // subgraphs (e.g. multi-thousand-node LLM decode) on the single-recording
  // fast path.
  constexpr uint64_t kFp16FlopsPerCu = 50ULL * 1000 * 1000 * 1000;
  constexpr uint64_t kFp32FlopsPerCu = 20ULL * 1000 * 1000 * 1000;
  int raw_cu_count = gpu_info.GetComputeUnitsCount();
  if (raw_cu_count <= 0) {
    raw_cu_count = gpu_info.adreno_info.compute_units;
  }
  const uint64_t cu_count = static_cast<uint64_t>(std::max(1, raw_cu_count));
  const uint64_t flops_per_cu =
      (precision == ::ml_drift::CalculationsPrecision::kF16) ? kFp16FlopsPerCu
                                                             : kFp32FlopsPerCu;
  return cu_count * flops_per_cu;
}

std::vector<int> GpuBackendOpenClLitert::PartitionNodesByFlops(
    absl::Span<const uint64_t> node_flops, uint64_t max_flops_per_batch) {
  if (max_flops_per_batch == 0 || node_flops.empty()) {
    return {};
  }
  std::vector<int> batch_sizes;
  int current_count = 0;
  uint64_t current_flops = 0;
  for (uint64_t flops : node_flops) {
    if (current_count > 0 && (current_flops >= max_flops_per_batch ||
                              flops > max_flops_per_batch - current_flops)) {
      batch_sizes.push_back(current_count);
      current_count = 0;
      current_flops = 0;
    }
    ++current_count;
    current_flops = SaturatingAdd(current_flops, flops);
  }
  if (batch_sizes.empty()) {
    // Entire graph fits in one submission; stay on the single command-buffer
    // fast path.
    return {};
  }
  batch_sizes.push_back(current_count);
  return batch_sizes;
}

#if LITERT_HAS_OPENGL_SUPPORT
namespace internal {

// Encapsulates all GL-CL synchronization logic for LiteRT Tensor Buffers.
class GlInteropFabricLiteRt {
 public:
  GlInteropFabricLiteRt() = default;
  GlInteropFabricLiteRt(LiteRtEglDisplay egl_display, cl_context context,
                        cl_command_queue queue,
                        bool egl_to_cl_mapping_supported)
      : egl_display_(egl_display),
        context_(context),
        queue_(queue),
        egl_to_cl_mapping_supported_(egl_to_cl_mapping_supported) {};

  // Registers a memory object with the GL interop fabric.
  absl::Status RegisterMemory(cl_mem memory);

  bool HasRegisteredMemory() const { return !mem_objects_.empty(); }

  // Starts acquisition of all the registered memory objects.
  // Note: GlInteropFabric retains ownership over cl_event.
  absl::StatusOr<cl_event> Start();
  // Finishes release of all the registered memory objects back to OpenGL
  // context.
  // Note: GlInteropFabric retains ownership over cl_event.
  absl::StatusOr<cl_event> Finish();

 private:
  // A list of memory objects that are registered for GL-CL interop. These are
  // accumulated by RegisterTensorBuffer().
  std::vector<cl_mem> mem_objects_;
  // Holds the acquired GL objects. This is valid only between a call to Start()
  // and a call to Finish().
  ::ml_drift::cl::AcquiredGlObjects acquired_gl_objects_;
  LiteRtEglDisplay egl_display_;
  cl_context context_;
  cl_command_queue queue_;
  bool egl_to_cl_mapping_supported_;
  // Owns the start event from Start().
  ::ml_drift::cl::CLEvent start_event_;
  // Owns the finish event from Finish().
  ::ml_drift::cl::CLEvent finish_event_;
};

absl::Status GlInteropFabricLiteRt::RegisterMemory(cl_mem memory) {
  // Register the memory object.
  for (const auto& mem : mem_objects_) {
    if (mem == memory) {
      return absl::OkStatus();
    }
  }
  mem_objects_.push_back(memory);
  return absl::OkStatus();
}

absl::StatusOr<cl_event> GlInteropFabricLiteRt::Start() {
  ::ml_drift::cl::CLEvent inbound_event;
  std::vector<cl_event> inbound_events;
  ::ml_drift::cl::EglSync sync;
  ABSL_RETURN_IF_ERROR(::ml_drift::cl::EglSync::NewFence(egl_display_, &sync));
  if (egl_to_cl_mapping_supported_) {
    glFlush();
    ABSL_RETURN_IF_ERROR(::ml_drift::cl::CreateClEventFromEglSync(
        context_, sync, &inbound_event));
    inbound_events.push_back(inbound_event.event());
  } else {
    if (auto status = sync.ClientWait(); !status.ok()) {
      return status;
    }
  }
  ABSL_RETURN_IF_ERROR(::ml_drift::cl::AcquiredGlObjects::Acquire(
      mem_objects_, queue_, inbound_events, &start_event_,
      &acquired_gl_objects_));
  return start_event_.event();
}

absl::StatusOr<cl_event> GlInteropFabricLiteRt::Finish() {
  ABSL_RETURN_IF_ERROR(acquired_gl_objects_.Release({}, &finish_event_));
  mem_objects_.clear();
  return finish_event_.event();
}

}  // namespace internal
#endif  // LITERT_HAS_OPENGL_SUPPORT

GpuBackendOpenClLitert::GpuBackendOpenClLitert(
    ::ml_drift::cl::Environment* env, LiteRtEglDisplay egl_display,
    SimpleCache&& compiled_cache, const LiteRtRuntimeContext* runtime_context)
    : GpuBackendOpenCl(env),
      egl_display_(egl_display),
#if LITERT_HAS_OPENGL_SUPPORT
      gl_interop_fabric_(
          ::ml_drift::cl::IsGlSharingSupported(env->device())
              ? std::make_unique<internal::GlInteropFabricLiteRt>(
                    egl_display, env->context().context(),
                    env->queue()->queue(),
                    ::ml_drift::cl::IsClEventFromEglSyncSupported(
                        env->device()))
              : nullptr),
#endif
      compiled_cache_(std::move(compiled_cache)),
      runtime_context_(ABSL_DIE_IF_NULL(runtime_context)),
      num_compiled_programs_(0),
      invoke_count_to_flush_compiled_cache_(0) {
  if (compiled_cache_.IsValid()) {
    auto status = UploadCompiledCache();
    if (!status.ok()) {
      LITERT_LOG(LITERT_WARNING, "Failed to upload compiled cache: %s",
                 status.message());
    }
  }
}

GpuBackendOpenClLitert::~GpuBackendOpenClLitert() {
  // Flush the compiled cache if it hasn't been flushed yet.
  if (compiled_cache_.IsValid() && invoke_count_to_flush_compiled_cache_ > 0) {
    invoke_count_to_flush_compiled_cache_ = 1;
    FlushCompiledCacheIfNeeded();
  }
}

absl::StatusOr<GpuMemoryHandle> GpuBackendOpenClLitert::GetGpuMemoryAllocated(
    const GpuTensorBufferPtr& tensor_buffer) {
  LiteRtClMem cl_mem_addr;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_opencl_memory(
      tensor_buffer.get(), &cl_mem_addr));
#if LITERT_HAS_OPENGL_SUPPORT
  if (gl_interop_fabric_) {
    LiteRtTensorBufferType buffer_type;
    LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_type(
        tensor_buffer.get(), &buffer_type));
    if (buffer_type == kLiteRtTensorBufferTypeGlBuffer) {
      ABSL_RETURN_IF_ERROR(gl_interop_fabric_->RegisterMemory(cl_mem_addr));
    }
  }
#endif  // LITERT_HAS_OPENGL_SUPPORT
  return cl_mem_addr;
}

absl::StatusOr<GpuEventHandle> GpuBackendOpenClLitert::GetGpuEventAssociated(
    const GpuTensorBufferPtr& tensor_buffer) {
  bool has_event;
  LITERT_RETURN_IF_ERROR(runtime_context_->has_tensor_buffer_event(
      tensor_buffer.get(), &has_event));
  if (!has_event) {
    return absl::NotFoundError("Tensor buffer does not have an event.");
  }

  LiteRtEvent event;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_event(tensor_buffer.get(), &event));
  LiteRtEventType event_type;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_event_event_type(event, &event_type));
  if (event_type == LiteRtEventTypeOpenCl) {
#if LITERT_HAS_OPENCL_SUPPORT
    LiteRtClEvent cl_event;
    LITERT_RETURN_IF_ERROR(
        runtime_context_->get_event_opencl_event(event, &cl_event));
    return cl_event;
#else
    return absl::InternalError("OpenCL is not supported on this platform.");
#endif  // LITERT_HAS_OPENCL_SUPPORT
  }

  if (event_type == LiteRtEventTypeEglSyncFence) {
    return absl::InternalError(
        "Attaching EGLSyncFence event to TensorBuffer is not needed. GL-CL "
        "event synchronization is handled internally within LiteRT OpenCL "
        "backend.");
  }

  LITERT_LOG(LITERT_WARNING, "Non-CL event. Waiting for it on CPU.");
  // Wait for the non OpenCL input event to be signaled.
  // TODO - b/404273554: Convert all the input events to CL events.
  LITERT_RETURN_IF_ERROR(runtime_context_->wait_event(event, -1));

  return absl::NotFoundError("Non-CL event has been waited for.");
}

absl::Status GpuBackendOpenClLitert::AssociateGpuEvent(
    GpuEventHandle event, LiteRtEnvironment env,
    GpuTensorBufferPtr& tensor_buffer) {
  if (runtime_context_ == nullptr) {
    return absl::InternalError("Runtime context is not set.");
  }
  LiteRtEvent litert_event;
  LITERT_RETURN_IF_ERROR(runtime_context_->create_event_from_opencl_event(
      env, reinterpret_cast<cl_event>(event), &litert_event));
  LITERT_RETURN_IF_ERROR(runtime_context_->set_tensor_buffer_event(
      tensor_buffer.get(), litert_event));
  return absl::OkStatus();
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendOpenClLitert::GetGpuBufferRequirements(
    ::ml_drift::TensorStorageType used_storage_type,
    ::ml_drift::DataType data_type) {
  GpuBufferRequirements requirements;
  if (used_storage_type == ::ml_drift::TensorStorageType::kTexture2D) {
    if (data_type == ::ml_drift::DataType::kFloat16) {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeOpenClTextureFp16);
    } else {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeOpenClTexture);
    }
  } else if (used_storage_type == ::ml_drift::TensorStorageType::kImageBuffer) {
    if (data_type == ::ml_drift::DataType::kFloat16) {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeOpenClImageBufferFp16);
    } else {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeOpenClImageBuffer);
    }
  } else {
    if (data_type == ::ml_drift::DataType::kFloat16) {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeOpenClBufferFp16);
    } else {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeOpenClBuffer);
    }
  }
  // MLD uses PHWC4, 16 bytes strides.
  requirements.strides = {16};

#if LITERT_HAS_OPENGL_SUPPORT
  if (gl_interop_fabric_) {
    requirements.buffer_types.push_back(kLiteRtTensorBufferTypeGlBuffer);
    requirements.strides.push_back(0);
  }
#endif
  return std::move(requirements);
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendOpenClLitert::GetGpuBufferRequirementsForNonExternalTensors() {
  GpuBufferRequirements requirements{
      .buffer_types = {kLiteRtTensorBufferTypeOpenClBufferPacked},
      // No strides for packed buffer.
      .strides = {0},
  };
#if LITERT_HAS_OPENGL_SUPPORT
  if (gl_interop_fabric_) {
    requirements.buffer_types.push_back(kLiteRtTensorBufferTypeGlBuffer);
    requirements.strides.push_back(0);
  }
#endif
#if LITERT_HAS_AHWB_SUPPORT
  // `cl_arm_import_memory` alone only guarantees the base import API (e.g.
  // host / dma_buf import types). Importing with
  // CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM additionally requires
  // `cl_arm_import_memory_android_hardware_buffer`; drivers such as PowerVR
  // DXT on Pixel 10 advertise the former but not the latter, and fail
  // clImportMemoryARM with CL_INVALID_PROPERTY.
  if (cl_env()->device().GetInfo().SupportsExtension(
          "cl_arm_import_memory_android_hardware_buffer") &&
      ::ml_drift::cl::clImportMemoryARM != nullptr) {
    requirements.buffer_types.push_back(kLiteRtTensorBufferTypeAhwb);
    requirements.strides.push_back(0);
  }
#endif
  return requirements;
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendOpenClLitert::CreateInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
    bool may_share_memory_manager) {
  void* gl_interop_fabric = nullptr;
#if LITERT_HAS_OPENGL_SUPPORT
  gl_interop_fabric = gl_interop_fabric_.get();
#endif
  auto ctx = std::make_unique<GpuInferenceContextOpenClLitert>(
      this, may_share_memory_manager ? &memory_manager() : nullptr,
      gl_interop_fabric);
  const uint64_t max_flops_per_batch =
      ResolveAutoBatchFlopsBudget(create_info.precision);
  // Collect per-node FLOPs before InitFromGpuModel() because it calls
  // std::move() on each `gpu_model.nodes[i].gpu_operation` (leaving them
  // null), and ml_drift::cl::InferenceContext does not expose node FLOPs.
  const std::vector<uint64_t> node_flops = max_flops_per_batch > 0
                                               ? CollectNodeFlops(gpu_model)
                                               : std::vector<uint64_t>{};
  ABSL_RETURN_IF_ERROR(ctx->cl_ctx().InitFromGpuModel(
      create_info, &gpu_model, cl_env(), serialized_model));
  MaybeSetAutoKernelBatchSizes(node_flops, max_flops_per_batch, ctx.get(),
                               "Splitting OpenCL dispatch");
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendOpenClLitert::RestoreInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const absl::Span<const uint8_t> serialized_model) {
  void* gl_interop_fabric = nullptr;
#if LITERT_HAS_OPENGL_SUPPORT
  gl_interop_fabric = gl_interop_fabric_.get();
#endif
  auto ctx = std::make_unique<GpuInferenceContextOpenClLitert>(
      this, &memory_manager(), gl_interop_fabric);
  ABSL_RETURN_IF_ERROR(ctx->cl_ctx().RestoreDeserialized(
      serialized_model, cl_env(), &create_info));
  const uint64_t max_flops_per_batch =
      ResolveAutoBatchFlopsBudget(create_info.precision);
  if (max_flops_per_batch > 0) {
    MaybeSetAutoKernelBatchSizes(CollectSerializedNodeFlops(serialized_model),
                                 max_flops_per_batch, ctx.get(),
                                 "Splitting restored OpenCL dispatch");
  }
  return std::move(ctx);
}

absl::Status GpuBackendOpenClLitert::UploadCompiledCache() {
  // Set the invoke count first to flush the cache later when the compiled cache
  // file doesn't exist now.
  invoke_count_to_flush_compiled_cache_ = kInvokeCountToFlushCompiledCache;

  int num_compiled_programs_before =
      cl_env()->program_cache()->GetProgramCount();

  ABSL_RETURN_IF_ERROR(compiled_cache_.Load(
      [cl_env = cl_env()](absl::Span<const uint8_t> data,
                          ::ml_drift::MMapHandle& mmap_handle) {
        return cl_env->program_cache()->AddSerializedCache(
            cl_env->context(), cl_env->device(), data);
      }));

  num_compiled_programs_ = cl_env()->program_cache()->GetProgramCount();

  LITERT_LOG(LITERT_VERBOSE, "Loaded compiled cache from %s: num_entries=%d",
             compiled_cache_.ToString().c_str(),
             num_compiled_programs_ - num_compiled_programs_before);
  return absl::OkStatus();
}

void GpuBackendOpenClLitert::FlushCompiledCacheIfNeeded() {
  if (invoke_count_to_flush_compiled_cache_ <= 0 ||
      !compiled_cache_.IsValid()) {
    return;
  }

  if (--invoke_count_to_flush_compiled_cache_ > 0) {
    return;
  }

  // If no programs are added to the cache, no need to flush the cache file.
  int num_compiled_programs_after =
      cl_env()->program_cache()->GetProgramCount();
  if (num_compiled_programs_ == num_compiled_programs_after) {
    LITERT_LOG(LITERT_VERBOSE, "No compiled programs to flush to cache file.");
    return;
  }

  std::vector<uint8_t> serialized_cache;
  if (auto s = cl_env()->program_cache()->GetSerializedCache(cl_env()->device(),
                                                             &serialized_cache);
      !s.ok()) {
    LITERT_LOG(LITERT_WARNING, "Failed to get serialized cache: %s",
               s.message());
    return;
  }

  if (auto s = compiled_cache_.Store(serialized_cache); !s.ok()) {
    LITERT_LOG(LITERT_WARNING, "Failed to write cache file: %s, status=%s",
               compiled_cache_.ToString().c_str(), s.message());
    return;
  }

  LITERT_LOG(LITERT_VERBOSE,
             "Flushed compiled cache file: %s, new=%d, total=%d",
             compiled_cache_.ToString().c_str(),
             num_compiled_programs_after - num_compiled_programs_,
             num_compiled_programs_after);
  num_compiled_programs_ = num_compiled_programs_after;
}

GpuInferenceContextOpenClLitert::GpuInferenceContextOpenClLitert(
    GpuBackendOpenClLitert* backend,
    ::ml_drift::cl::MemoryManager* memory_manager, void* gl_interop_fabric)
    : GpuInferenceContextOpenCl(backend, memory_manager)
#if LITERT_HAS_OPENGL_SUPPORT
      ,
      gl_interop_fabric_(
          reinterpret_cast<internal::GlInteropFabricLiteRt*>(gl_interop_fabric))
#endif
{
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI
absl::Status GpuInferenceContextOpenClLitert::Dispatch() {
  GpuBackendOpenClLitert* current_backend =
      static_cast<GpuBackendOpenClLitert*>(backend());
  current_backend->FlushCompiledCacheIfNeeded();

  // 1. Explicit caller-configured batch size always takes precedence.
  if (current_backend->kernel_batch_size() > 0) {
    int offset = 0;
    do {
      ABSL_ASSIGN_OR_RETURN(
          offset, cl_ctx().AddToQueue(cl_env()->queue(), offset,
                                      current_backend->kernel_batch_size()));
      ::ml_drift::cl::clFlush(cl_env()->queue()->queue());
    } while (offset != -1);
    return absl::OkStatus();
  }

  // 2. Per-context automatic FLOPs-bounded batching (Adreno 8xx heavy graphs).
  if (!auto_kernel_batch_sizes_.empty()) {
    int offset = 0;
    for (int count : auto_kernel_batch_sizes_) {
      ABSL_ASSIGN_OR_RETURN(
          offset, cl_ctx().AddToQueue(cl_env()->queue(), offset, count));
      ::ml_drift::cl::clFlush(cl_env()->queue()->queue());
      if (offset == -1) {
        break;
      }
    }
    return absl::OkStatus();
  }

  // 3. Default fast path: dispatch the whole graph (uses recorded command
  // buffers when supported).
  return cl_ctx().AddToQueue(cl_env()->queue());
}

absl::Status GpuInferenceContextOpenClLitert::PreConvert(bool input) {
  if (!input) {
    // We only acquire GL buffers at the very beginning (input=true).
    return absl::OkStatus();
  }
#if LITERT_HAS_OPENGL_SUPPORT
  if (gl_interop_fabric_ && gl_interop_fabric_->HasRegisteredMemory()) {
    LITERT_LOG(LITERT_DEBUG,
               "Enqueuing acquisition of CL memory objects from GL buffers.");
    ABSL_RETURN_IF_ERROR(gl_interop_fabric_->Start().status());
  }
#endif  // LITERT_HAS_OPENGL_SUPPORT
  return absl::OkStatus();
}

absl::StatusOr<GpuEventHandle>
GpuInferenceContextOpenClLitert::GetPreDispatchEvent() {
  // Async is not currently supported for GL-CL interop, so we always return
  // NotFoundError. Acquisition was taken care of in PreConvert().
  return absl::NotFoundError("No pre-dispatch event.");
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI
absl::StatusOr<GpuEventHandle>
GpuInferenceContextOpenClLitert::GetPostDispatchEvent(
    bool is_async_execution_mode) {
  if (!is_async_execution_mode) {
    return absl::NotFoundError("No post-dispatch event.");
  }

  // Adreno and PowerVR don't seem to support async execution properly
  // when OpenGL is not involved. See b/467546752.
  // TODO: b/403337563 - Investigate other ways to enable async on Adreno and
  // PowerVR. Either Mark or Barrier doesn't work as expected.
  auto gpu_info = cl_env()->GetDevicePtr()->GetInfo();
  if (gpu_info.IsAdreno() || gpu_info.IsPowerVR()) {
    return absl::NotFoundError(
        "Async execution is not supported on Adreno and PowerVR.");
  }

  cl_event output_event;
  // Create a marker event to notify for all the output events to be
  // completed.
  cl_int enqueue_status = ::ml_drift::cl::clEnqueueMarkerWithWaitList(
      cl_env()->queue()->queue(), 0, nullptr, &output_event);
  if (enqueue_status != CL_SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "Failed to enqueue marker with wait list: ", enqueue_status));
  }

  // Ensures the marker event is not released before enqueue.
  post_dispatch_event_ = ::ml_drift::cl::CLEvent(output_event);
  return output_event;
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI
absl::Status GpuInferenceContextOpenClLitert::WaitForEventsCompleted(
    absl::Span<GpuEventHandle> events, bool force_sync) {
  if (events.empty()) {
    return absl::InvalidArgumentError("No events to wait for.");
  }

  LITERT_LOG(LITERT_DEBUG, "Waiting for %zu OpenCL events", events.size());
  auto* events_ptr = reinterpret_cast<cl_event*>(events.data());
  cl_int enqueue_status =
      force_sync
          ? ::ml_drift::cl::clWaitForEvents(events.size(), events_ptr)
          : ::ml_drift::cl::clEnqueueBarrierWithWaitList(
                cl_env()->queue()->queue(), events.size(), events_ptr, nullptr);
  if (enqueue_status != CL_SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "Failed to enqueue barrier with wait list: ", enqueue_status));
  }
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenClLitert::PostConvert(bool input) {
#if LITERT_HAS_OPENGL_SUPPORT
  if (input) {
    // We only release GL buffers at the very end (input=false).
    return absl::OkStatus();
  }

  if (gl_interop_fabric_ && gl_interop_fabric_->HasRegisteredMemory()) {
    LITERT_LOG(LITERT_DEBUG,
               "Enqueuing release of CL memory objects to GL buffers.");
    ABSL_ASSIGN_OR_RETURN(cl_event interop_release_event,
                          gl_interop_fabric_->Finish());
    std::vector<GpuEventHandle> events = {interop_release_event};
    return WaitForEventsCompleted(absl::MakeSpan(events), /*force_sync=*/true);
  }
#endif  // LITERT_HAS_OPENGL_SUPPORT
  return absl::OkStatus();
}

}  // namespace litert::ml_drift
