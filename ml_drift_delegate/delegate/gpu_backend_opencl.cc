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

#include "ml_drift_delegate/delegate/gpu_backend_opencl.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <malloc.h>
#include <windows.h>
#else  // defined(_WIN32)
#include <unistd.h>
#endif  // defined(_WIN32)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "ml_drift/cl/converter.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/cl/memory_manager.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/cl/util.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include <CL/cl.h>
#ifdef ML_DRIFT_MEM_STATS
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_cl_benchmark_util.h"  // IWYU pragma: keep
#endif
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_cl.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr absl::string_view kBackendName = "OpenCL";

// Serialization data prefix. FYI, Jet uses "gpuv2_data_".
constexpr absl::string_view kSerializedDataPrefix = "gpuv3_data_";

}  // namespace

GpuBackendOpenCl::GpuBackendOpenCl(::ml_drift::cl::Environment* env)
    : env_(env) {}

GpuBackendOpenCl::GpuBackendOpenCl(
    std::unique_ptr<::ml_drift::cl::Environment> env)
    : env_owned_(std::move(env)), env_(env_owned_.get()) {}

absl::string_view GpuBackendOpenCl::GetBackendName() { return kBackendName; }

absl::string_view GpuBackendOpenCl::GetSerializedDataPrefix() {
  return kSerializedDataPrefix;
}

absl::StatusOr<::ml_drift::GpuInfo> GpuBackendOpenCl::GetInfo() {
  return env_->GetDevicePtr()->GetInfo();
}

absl::StatusOr<::ml_drift::TensorStorageType>
GpuBackendOpenCl::GetFastestStorageType() {
  return ::ml_drift::cl::GetFastestStorageType(env_->GetDevicePtr()->GetInfo());
}

absl::StatusOr<GpuMemoryHandle> GpuBackendOpenCl::GetGpuMemoryAllocated(
    const GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("GetGpuMemoryAllocated is not implemented.");
}

absl::StatusOr<GpuEventHandle> GpuBackendOpenCl::GetGpuEventAssociated(
    const GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("GetGpuEventAssociated is not implemented.");
}

absl::Status GpuBackendOpenCl::AssociateGpuEvent(
    GpuEventHandle event, LiteRtEnvironment env,
    GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("AssociateGpuEvent is not implemented.");
}

absl::Status GpuBackendOpenCl::WaitForCompletion() {
  return env_->queue()->WaitForCompletion();
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendOpenCl::GetGpuBufferRequirements(
    ::ml_drift::TensorStorageType used_storage_type,
    ::ml_drift::DataType data_type) {
  return absl::UnimplementedError(
      "GetGpuBufferRequirements is not implemented.");
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendOpenCl::GetGpuBufferRequirementsForNonExternalTensors() {
  return absl::UnimplementedError(
      "GetGpuBufferRequirementsForNonExternalTensors is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendOpenCl::CreateInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
    bool may_share_memory_manager) {
  auto ctx = std::make_unique<GpuInferenceContextOpenCl>(
      this, may_share_memory_manager ? &memory_manager_ : nullptr);
  ABSL_RETURN_IF_ERROR(ctx->cl_ctx().InitFromGpuModel(create_info, &gpu_model,
                                                      env_, serialized_model));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendOpenCl::RestoreInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const absl::Span<const uint8_t> serialized_model) {
  auto ctx =
      std::make_unique<GpuInferenceContextOpenCl>(this, &memory_manager_);
  ABSL_RETURN_IF_ERROR(
      ctx->cl_ctx().RestoreDeserialized(serialized_model, env_, &create_info));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
GpuBackendOpenCl::CreateSharedMemoryManager(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    std::unique_ptr<::ml_drift::GraphAdapter> graph_adapter,
    TfLiteContext* context, MlDriftDelegateData& delegate_data,
    ::ml_drift::SerializationWeightCache* serialization_cache) {
  return ::ml_drift::MakeSharedMemoryManagerCl(
      *env_, create_info, std::move(graph_adapter), context,
      GetBufferIdToSpatialTensorMap(delegate_data),
      GetQuantParamIdToSpatialTensorMap(delegate_data),
      delegate_data.options->has_prepacked_external_tflite_tensors,
      serialization_cache,
      delegate_data.options->madvise_original_shared_tensors);
}

absl::StatusOr<std::shared_ptr<::ml_drift::WeightsManager>>
GpuBackendOpenCl::CreateWeightsManager() {
  return std::make_shared<::ml_drift::WeightsManager>();
}

absl::StatusOr<std::vector<
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
GpuBackendOpenCl::GetBatchesForWeightsPreparation(
    ::ml_drift::WeightsManager* weights_manager,
    size_t total_shared_tensor_size) {
  return absl::UnimplementedError(
      "GetBatchesForWeightsPreparation is not implemented.");
}

absl::StatusOr<absl::flat_hash_map<
    ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendOpenCl::PrepareWeightsInBatch(
    ::ml_drift::WeightsManager* weights_manager,
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
        op_infos) {
  return absl::UnimplementedError("PrepareWeightsInBatch is not implemented.");
}

absl::StatusOr<absl::flat_hash_map<
    ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendOpenCl::PrepareWeightsInBatches(
    ::ml_drift::WeightsManager* weights_manager,
    size_t total_shared_tensor_size) {
  return absl::UnimplementedError(
      "PrepareWeightsInBatches is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuTensorWrapper>>
GpuBackendOpenCl::CreateTensorWrapper(const ::ml_drift::TensorDescriptor& desc,
                                      GpuMemoryHandle gpu_memory) {
  auto cl_tensor = std::make_unique<GpuTensorWrapperOpenCl>();
  ABSL_RETURN_IF_ERROR(::ml_drift::cl::CreateTensorShared(
      env_->context(), static_cast<cl_mem>(gpu_memory), desc,
      &cl_tensor->cl_tensor()));
  return std::move(cl_tensor);
}

absl::Status GpuBackendOpenCl::ReadSpatialTensorToDescriptor(
    ::ml_drift::GpuSpatialTensor& tensor, ::ml_drift::TensorDescriptor& desc) {
  return absl::UnimplementedError(
      "ReadSpatialTensorToDescriptor is not implemented");
}

absl::Status GpuBackendOpenCl::UpdateSpatialTensor(
    ::ml_drift::GpuSpatialTensor* tensor,
    const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
    ReleaseDataCallback release_data_callback) {
  return absl::UnimplementedError("UpdateSpatialTensor is not implemented.");
}

absl::Status GpuBackendOpenCl::ReleaseSpatialTensorMemory(
    ::ml_drift::GpuSpatialTensor* tensor) {
  return absl::UnimplementedError(
      "ReleaseSpatialTensorMemory is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>> GpuBackendOpenCl::CreateIOBuffer(
    GpuMemoryHandle gpu_memory) {
  return std::make_unique<GpuIOBufferOpenCl>(
      env_,
      ::ml_drift::cl::CreateBufferShared(static_cast<cl_mem>(gpu_memory)));
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
GpuBackendOpenCl::CreateIOBufferWithSize(::ml_drift::DataType data_type,
                                         size_t size, bool input) {
  ::ml_drift::cl::Buffer cl_buffer;
  // On PowerVR architectures, force CPU-cached allocation on output readback
  // buffers to avoid uncached bus readback stalls.
  if (!input && env_->GetDevicePtr()->GetInfo().IsPowerVR()) {
    cl_int err = CL_SUCCESS;
    cl_mem mem = ::ml_drift::cl::clCreateBuffer(
        env_->context().context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        size, nullptr, &err);
    if (err != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to allocate device memory with clCreateBuffer: ", err));
    }
    cl_buffer = ::ml_drift::cl::Buffer(mem, size);  // takes ownership
  } else {
    ABSL_RETURN_IF_ERROR(::ml_drift::cl::CreateReadWriteBuffer(
        size, &env_->context(), &cl_buffer));
  }
  return std::make_unique<GpuIOBufferOpenCl>(env_, std::move(cl_buffer));
}

// Answers the allocation policy question: returns true if the device
// architecture permits zero-copy host pointer wrapping (CL_MEM_USE_HOST_PTR).
// On verified PowerVR architectures (D-Series DXT on Tensor G5, and C-Series
// CXTP on Tensor G6), CL_MEM_USE_HOST_PTR is supported and required to
// eliminate uncached bus readback stalls.
bool GpuBackendOpenCl::SupportsZeroCopyHostPtr(bool input) const {
  if (force_host_ptr_io_buffer_) {
    return true;
  }
  if (env_ == nullptr || env_->GetDevicePtr() == nullptr) {
    return false;
  }
  const auto& info = env_->GetDevicePtr()->GetInfo();
  // Both D-Series (DXT, Tensor G5) and C-Series (CXTP, Tensor G6) have been
  // verified to support zero-copy host pointer wrapping with CPU-GPU cache
  // coherency upon clFinish.
  if (info.IsPowerVR() &&
      (info.powervr_info.IsImgDxx() || info.powervr_info.IsImgCxx())) {
    return true;
  }
  // Future GPU architectures can be added here after cache coherency
  // verification.
  return false;
}

// Answers the hardware memory coherence capability question: returns true if
// the device requires explicit cache flushing/invalidation (via
// clEnqueueMapBuffer/clEnqueueUnmapMemObject) between host CPU and GPU accesses
// to CL_MEM_USE_HOST_PTR memory.
//
// This is an intrinsic hardware capability property kept independent of the
// allocation policy (force_host_ptr_io_buffer_):
// - On PowerVR D-Series (DXT, Tensor G5) and C-Series (CXTP, Tensor G6),
//   hardware cache coherency is guaranteed across CPU and GPU caches upon
//   clFinish, so explicit host sync is not required.
// - For non-coherent devices (e.g. Nvidia/AMD discrete GPUs or unverified
//   PowerVR generations), standard operation uses normal device allocations
//   with clEnqueueReadBuffer or clEnqueueWriteBuffer. If host pointers are
//   wrapped (such as during host workstation unit testing), this map/unmap
//   synchronization ensures OpenCL specification compliance and prevents
//   cache divergence.
bool GpuBackendOpenCl::RequiresExplicitHostSync() const {
  if (env_ == nullptr || env_->GetDevicePtr() == nullptr) {
    return false;
  }
  const auto& info = env_->GetDevicePtr()->GetInfo();
  // Both D-Series (DXT, Tensor G5) and C-Series (CXTP, Tensor G6) have been
  // verified to be cache coherent after clFinish. Other PowerVR generations
  // fall back to explicit map/unmap.
  return !(info.IsPowerVR() &&
           (info.powervr_info.IsImgDxx() || info.powervr_info.IsImgCxx()));
}

bool GpuBackendOpenCl::SupportsHostPtrIOBuffer(bool input) const {
  return SupportsZeroCopyHostPtr(input);
}

// OpenCL CL_MEM_USE_HOST_PTR requires host memory to be aligned to the OS
// physical page boundary (typically 4096 bytes). On PowerVR kernel drivers
// (PVRSRVWrapExtMemMIW), passing an unaligned host address fails with
// CL_INVALID_HOST_PTR (-37).
size_t GpuBackendOpenCl::GetHostPtrAlignment() const {
  if (!SupportsZeroCopyHostPtr(/*input=*/true) &&
      !SupportsZeroCopyHostPtr(/*input=*/false)) {
    return 0;
  }
  size_t alignment = 4096;
#if defined(_WIN32)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  if (sys_info.dwPageSize > 0) {
    alignment = std::max(alignment, static_cast<size_t>(sys_info.dwPageSize));
  }
#elif defined(_SC_PAGESIZE)
  const int64_t page_size = sysconf(_SC_PAGESIZE);
  if (page_size > 0) {
    alignment = std::max(alignment, static_cast<size_t>(page_size));
  }
#endif
  return alignment;
}

// Allocates page-aligned internal host memory and wraps it via
// CL_MEM_USE_HOST_PTR.
// The returned buffer owns this memory (OwnsHostPtr() == true) and releases it
// in the destructor.
absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
GpuBackendOpenCl::CreateInternalHostBuffer(::ml_drift::DataType data_type,
                                           size_t size, bool input) {
  if (!SupportsZeroCopyHostPtr(input)) {
    return absl::FailedPreconditionError(
        "Host pointer I/O buffers are not supported for this direction.");
  }
  if (size == 0) {
    return absl::InvalidArgumentError("Buffer size must be greater than 0.");
  }
  const size_t alignment = GetHostPtrAlignment();
  if (alignment == 0) {
    return absl::FailedPreconditionError(
        "Host pointer alignment is 0; host pointers not supported.");
  }

  void* host_ptr = nullptr;
  const size_t alloc_size = (size + alignment - 1) & ~(alignment - 1);
#if defined(_WIN32)
  host_ptr = _aligned_malloc(alloc_size, alignment);
  if (host_ptr == nullptr) {
    return absl::ResourceExhaustedError(
        "Failed to allocate aligned host memory.");
  }
#else
  if (posix_memalign(&host_ptr, alignment, alloc_size) != 0 ||
      host_ptr == nullptr) {
    return absl::ResourceExhaustedError(
        "Failed to allocate aligned host memory.");
  }
#endif
  std::memset(host_ptr, 0, alloc_size);

  cl_int err = CL_SUCCESS;
  cl_mem mem = ::ml_drift::cl::clCreateBuffer(
      env_->context().context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
      alloc_size, host_ptr, &err);
  if (err != CL_SUCCESS) {
#if defined(_WIN32)
    _aligned_free(host_ptr);
#else
    free(host_ptr);
#endif
    return absl::InternalError(absl::StrCat(
        "Failed to create CL_MEM_USE_HOST_PTR buffer with clCreateBuffer: ",
        err));
  }

  ::ml_drift::cl::Buffer cl_buffer(mem, size);
  return std::make_unique<GpuIOBufferOpenCl>(
      env_, std::move(cl_buffer), /*is_host_ptr=*/true, host_ptr,
      /*owns_host_ptr=*/true, RequiresExplicitHostSync());
}

// Strictly attempts to wrap caller's pointer with CL_MEM_USE_HOST_PTR.
// Returns an error status immediately if host_ptr is null, unaligned, or if
// clCreateBuffer fails, without silently falling back to internal allocation.
absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
GpuBackendOpenCl::WrapExternalHostBuffer(::ml_drift::DataType data_type,
                                         size_t size, bool input,
                                         void* host_ptr) {
  if (!SupportsZeroCopyHostPtr(input)) {
    return absl::FailedPreconditionError(
        "Host pointer I/O buffers are not supported for this direction.");
  }
  if (host_ptr == nullptr) {
    return absl::InvalidArgumentError("host_ptr cannot be null.");
  }
  if (size == 0) {
    return absl::InvalidArgumentError("Buffer size must be greater than 0.");
  }
  const size_t alignment = GetHostPtrAlignment();
  if (reinterpret_cast<uintptr_t>(host_ptr) % alignment != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "host_ptr ", absl::Hex(reinterpret_cast<uintptr_t>(host_ptr)),
        " does not meet required alignment: ", alignment));
  }

  cl_int err = CL_SUCCESS;
  cl_mem mem = ::ml_drift::cl::clCreateBuffer(
      env_->context().context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size,
      host_ptr, &err);
  if (err != CL_SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "Failed to wrap external host pointer with clCreateBuffer: ", err));
  }

  ::ml_drift::cl::Buffer cl_buffer(mem, size);
  return std::make_unique<GpuIOBufferOpenCl>(
      env_, std::move(cl_buffer), /*is_host_ptr=*/true, host_ptr,
      /*owns_host_ptr=*/false, RequiresExplicitHostSync());
}

absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
GpuBackendOpenCl::CreateTensor2BufferConverter(
    const ::ml_drift::TensorDescriptor& src_desc,
    const ::ml_drift::BufferDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::cl::TensorToBHWCBufferConverter>();
  ABSL_RETURN_IF_ERROR(
      converter->InitExplicit(env_->GetDevicePtr(), &env_->context(),
                              env_->program_cache(), src_desc, dst_desc));
  return std::make_unique<Tensor2BufferConverterOpenCl>(env_,
                                                        std::move(converter));
}

absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
GpuBackendOpenCl::CreateBuffer2TensorConverter(
    const ::ml_drift::BufferDescriptor& src_desc,
    const ::ml_drift::TensorDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::cl::BHWCBufferToTensorConverter>();
  ABSL_RETURN_IF_ERROR(
      converter->InitExplicit(env_->GetDevicePtr(), &env_->context(),
                              env_->program_cache(), src_desc, dst_desc));
  return std::make_unique<Buffer2TensorConverterOpenCl>(env_,
                                                        std::move(converter));
}

GpuInferenceContextOpenCl::GpuInferenceContextOpenCl(
    GpuBackendOpenCl* backend, ::ml_drift::cl::MemoryManager* memory_manager)
    : backend_(backend),
      ctx_(memory_manager != nullptr
               ? std::make_unique<::ml_drift::cl::InferenceContext>(
                     memory_manager)
               : std::make_unique<::ml_drift::cl::InferenceContext>()) {}

absl::StatusOr<::ml_drift::GpuSpatialTensor*>
GpuInferenceContextOpenCl::GetSpatialTensor(::ml_drift::ValueId id) {
  auto* tensor = ctx_->GetTensor(id);
  if (tensor == nullptr) {
    // Returning OK(nullptr) here makes callers dereference null instead of
    // failing, which turns a recoverable setup error into a SIGSEGV.
    return absl::NotFoundError(
        absl::StrCat("No GPU tensor allocated for tensor id ", id, "."));
  }
  return tensor;
}

absl::Status GpuInferenceContextOpenCl::BindSpatialTensor(
    ::ml_drift::ValueId id, ::ml_drift::GpuSpatialTensor* tensor) {
  return ctx_->SetTensor(id, static_cast<::ml_drift::cl::Tensor*>(tensor));
}

absl::Status GpuInferenceContextOpenCl::WriteDataToWeightTensor(
    ::ml_drift::ValueId id, absl::Span<const uint8_t> data) {
  auto* cl_tensor = ctx_->GetTensor(id);
  // Mali GPU returns incorrect results when utilizing staging buffer to upload
  // data to GPU.
  ABSL_ASSIGN_OR_RETURN(auto gpu_info, backend_->GetInfo());
  if (gpu_info.IsMali()) {
    return cl_tensor->WriteData(data.data(), cl_env()->queue(),
                                /*async=*/false);
  } else {
    return cl_tensor->WriteDataViaStaging(data.data(), cl_env()->queue(),
                                          &cl_env()->context());
  }
}

absl::Status GpuInferenceContextOpenCl::ReadWeightTensorToDescriptor(
    ::ml_drift::ValueId id, ::ml_drift::TensorDescriptor& desc) {
  auto* cl_tensor = ctx_->GetTensor(id);
  return cl_tensor->ToDescriptor(&desc, cl_env()->queue());
}

absl::Status GpuInferenceContextOpenCl::Dispatch() {
  return ctx_->AddToQueue(cl_env()->queue());
}

absl::StatusOr<GpuEventHandle>
GpuInferenceContextOpenCl::GetPreDispatchEvent() {
  return absl::UnimplementedError("GetPreDispatchEvent is not implemented.");
}

absl::StatusOr<GpuEventHandle> GpuInferenceContextOpenCl::GetPostDispatchEvent(
    bool is_async_execution_mode) {
  return absl::UnimplementedError("GetPostDispatchEvent is not implemented.");
}

absl::Status GpuInferenceContextOpenCl::WaitForEventsCompleted(
    absl::Span<GpuEventHandle> events, bool force_sync) {
  return absl::UnimplementedError("WaitForEventsCompleted is not implemented.");
}

absl::Status GpuInferenceContextOpenCl::PreConvert(bool input) {
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenCl::PostConvert(bool input) {
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenCl::Profile(
    ::ml_drift::ProfilingInfo& profiling_info) {
  return ctx_->Profile(cl_env()->profiling_queue(), &profiling_info);
}

absl::StatusOr<size_t>
GpuInferenceContextOpenCl::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  return ctx_->GetSizeOfMemoryAllocatedForIntermediateTensors();
}

absl::StatusOr<size_t>
GpuInferenceContextOpenCl::GetSizeOfMemoryAllocatedForConstantTensors() const {
  return ctx_->GetConstantTensorsSize();
}

absl::StatusOr<size_t>
GpuInferenceContextOpenCl::GetSizeOfMemoryAllocatedForExternalTensors() const {
  return ctx_->GetExternalTensorsSize();
}

absl::Status GpuInferenceContextOpenCl::ReportMemoryBenchmarkIfEnabled(
    const ::ml_drift::CreateGpuModelInfo& create_info) {
#ifdef ML_DRIFT_MEM_STATS
  std::cout << ml_drift_delegate::GetMemoryBenchmarkReport(*ctx_, create_info)
            << std::endl;
#endif
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenCl::SetCommandBufferHint(
    int num_nodes_per_command_encoder) {
  return absl::UnimplementedError("SetCommandBufferHint is not implemented.");
}

GpuIOBufferOpenCl::GpuIOBufferOpenCl(::ml_drift::cl::Environment* env,
                                     ::ml_drift::cl::Buffer&& buffer,
                                     bool is_host_ptr, void* host_ptr,
                                     bool owns_host_ptr,
                                     bool requires_explicit_host_sync)
    : env_(env),
      buffer_(std::move(buffer)),
      is_host_ptr_(is_host_ptr),
      host_ptr_(host_ptr),
      owns_host_ptr_(owns_host_ptr),
      requires_explicit_host_sync_(is_host_ptr && requires_explicit_host_sync) {
}

GpuIOBufferOpenCl::~GpuIOBufferOpenCl() {
  buffer_ = ::ml_drift::cl::Buffer();
  if (owns_host_ptr_ && host_ptr_ != nullptr) {
#if defined(_WIN32)
    _aligned_free(host_ptr_);
#else
    free(host_ptr_);
#endif
  }
}

absl::Status GpuIOBufferOpenCl::SyncToDevice() {
  if (requires_explicit_host_sync_) {
    const size_t size = buffer_.GetMemorySizeInBytes();
    if (size == 0) {
      return absl::OkStatus();
    }
    // Save host memory contents into staging storage before mapping. This
    // fallback ensures OpenCL specification compliance and supports unit
    // testing on non-coherent host workstation GPUs (e.g. Nvidia). This path
    // is test/compat-only (requires_explicit_host_sync_ is false on PowerVR
    // D-Series/C-Series) and is not perf-critical. On such non-coherent
    // platforms,
    // clEnqueueMapBuffer with CL_MAP_WRITE_INVALIDATE_REGION may invalidate
    // host CPU caches, discard dirty cache lines, or return a divergent driver
    // bounce buffer. A reusable staging buffer avoids heap allocations per
    // inference on workstation GPUs.
    if (host_ptr_ != nullptr) {
      if (staging_buffer_.size() < size) {
        staging_buffer_.resize(size);
      }
      std::memcpy(staging_buffer_.data(), host_ptr_, size);
    }
    cl_int err = CL_SUCCESS;
    void* mapped_ptr = ::ml_drift::cl::clEnqueueMapBuffer(
        env_->queue()->queue(), buffer_.GetMemoryPtr(),
        /*blocking_map=*/CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, /*offset=*/0,
        size, /*num_events_in_wait_list=*/0, /*event_wait_list=*/nullptr,
        /*event=*/nullptr, &err);
    if (err != CL_SUCCESS || mapped_ptr == nullptr) {
      return absl::InternalError(
          absl::StrCat("clEnqueueMapBuffer failed in SyncToDevice: ",
                       ::ml_drift::cl::CLErrorCodeToString(err)));
    }
    if (host_ptr_ != nullptr) {
      // NOMUTANTS -- Driver bounce buffer copy: on unified virtual memory
      // drivers mapped_ptr == host_ptr_, but OpenCL spec compliance requires
      // copying staging buffer on non-coherent drivers (b/376383693).
      std::memcpy(mapped_ptr, staging_buffer_.data(), size);
    }
    err = ::ml_drift::cl::clEnqueueUnmapMemObject(
        env_->queue()->queue(), buffer_.GetMemoryPtr(), mapped_ptr,
        /*num_events_in_wait_list=*/0, /*event_wait_list=*/nullptr,
        /*event=*/nullptr);
    if (err != CL_SUCCESS) {
      return absl::InternalError(
          absl::StrCat("clEnqueueUnmapMemObject failed in SyncToDevice: ",
                       ::ml_drift::cl::CLErrorCodeToString(err)));
    }
    return env_->queue()->WaitForCompletion();
  }
  return absl::OkStatus();
}

absl::Status GpuIOBufferOpenCl::SyncToHost() {
  // NOMUTANTS -- Only host pointer buffers have explicit host sync enabled.
  if (requires_explicit_host_sync_) {
    const size_t size = buffer_.GetMemorySizeInBytes();
    if (size == 0) {
      return absl::OkStatus();
    }
    // Map/unmap read synchronization fallback for OpenCL specification
    // compliance and unit testing on non-coherent workstation GPUs. If the
    // driver allocates a divergent bounce buffer (mapped_ptr != host_ptr_),
    // copy the results back to host_ptr_.
    cl_int err = CL_SUCCESS;
    void* mapped_ptr = ::ml_drift::cl::clEnqueueMapBuffer(
        env_->queue()->queue(), buffer_.GetMemoryPtr(),
        /*blocking_map=*/CL_TRUE, CL_MAP_READ, /*offset=*/0, size,
        /*num_events_in_wait_list=*/0,
        /*event_wait_list=*/nullptr, /*event=*/nullptr, &err);
    if (err != CL_SUCCESS || mapped_ptr == nullptr) {
      return absl::InternalError(
          absl::StrCat("clEnqueueMapBuffer failed in SyncToHost: ",
                       ::ml_drift::cl::CLErrorCodeToString(err)));
    }
    if (mapped_ptr != host_ptr_ && host_ptr_ != nullptr) {
      std::memcpy(host_ptr_, mapped_ptr, size);
    }
    err = ::ml_drift::cl::clEnqueueUnmapMemObject(
        env_->queue()->queue(), buffer_.GetMemoryPtr(), mapped_ptr,
        /*num_events_in_wait_list=*/0, /*event_wait_list=*/nullptr,
        /*event=*/nullptr);
    if (err != CL_SUCCESS) {
      return absl::InternalError(
          absl::StrCat("clEnqueueUnmapMemObject failed in SyncToHost: ",
                       ::ml_drift::cl::CLErrorCodeToString(err)));
    }
    return env_->queue()->WaitForCompletion();
  }
  return absl::OkStatus();
}

absl::Status GpuIOBufferOpenCl::Read(absl::Span<uint8_t> data) {
  if (is_host_ptr_) {
    ABSL_RETURN_IF_ERROR(SyncToHost());
    const size_t copy_size = std::min(
        data.size(), static_cast<size_t>(buffer_.GetMemorySizeInBytes()));
    if (data.data() != host_ptr_ && copy_size > 0) {
      std::memcpy(data.data(), host_ptr_, copy_size);
    }
    return absl::OkStatus();
  }
  return env_->queue()->EnqueueReadBuffer(buffer_.GetMemoryPtr(), data.size(),
                                          data.data(), /*async=*/true);
}

absl::Status GpuIOBufferOpenCl::Write(absl::Span<const uint8_t> data) {
  if (is_host_ptr_) {
    const size_t copy_size = std::min(
        data.size(), static_cast<size_t>(buffer_.GetMemorySizeInBytes()));
    if (data.data() != host_ptr_ && copy_size > 0) {
      std::memcpy(host_ptr_, data.data(), copy_size);
    }
    return SyncToDevice();
  }
  return env_->queue()->EnqueueWriteBuffer(buffer_.GetMemoryPtr(), data.size(),
                                           data.data(), /*async=*/true);
}

Tensor2BufferConverterOpenCl::Tensor2BufferConverterOpenCl(
    ::ml_drift::cl::Environment* env,
    std::unique_ptr<::ml_drift::cl::TensorToBHWCBufferConverter> converter)
    : env_(env), converter_(std::move(converter)) {}

absl::Status Tensor2BufferConverterOpenCl::Convert(
    ::ml_drift::GpuSpatialTensor& src_tensor, GpuIOBuffer& dst_buffer) {
  return converter_->ConvertExplicit(
      env_->queue(), static_cast<::ml_drift::cl::Tensor*>(&src_tensor),
      &(static_cast<GpuIOBufferOpenCl&>(dst_buffer).cl_buffer()));
}

Buffer2TensorConverterOpenCl::Buffer2TensorConverterOpenCl(
    ::ml_drift::cl::Environment* env,
    std::unique_ptr<::ml_drift::cl::BHWCBufferToTensorConverter> converter)
    : env_(env), converter_(std::move(converter)) {}

absl::Status Buffer2TensorConverterOpenCl::Convert(
    GpuIOBuffer& src_buffer, ::ml_drift::GpuSpatialTensor& dst_tensor) {
  return converter_->ConvertExplicit(
      env_->queue(), &(static_cast<GpuIOBufferOpenCl&>(src_buffer).cl_buffer()),
      static_cast<::ml_drift::cl::Tensor*>(&dst_tensor));
}

}  // namespace litert::ml_drift
