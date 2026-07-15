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

#include "ml_drift_delegate/delegate/gpu_backend_metal.h"

#import <Metal/Metal.h>

#include <memory>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl

#include "ml_drift/common/convert.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/metal/buffer.h"  // from @ml_drift
#include "ml_drift/metal/converter.h"  // from @ml_drift
#include "ml_drift/metal/inference_context.h"  // from @ml_drift
#include "ml_drift/metal/metal_device.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "ml_drift/metal/metal_weights_manager.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/delegate_data_util.h"
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_metal_benchmark_util.h"
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "third_party/odml/infra/ml_drift_delegate/shared_memory_manager_metal.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"

namespace litert::ml_drift {
namespace {
constexpr absl::string_view kBackendName = "Metal";
constexpr absl::string_view kSerializedDataPrefix = "gpuv3_metal_data_";
constexpr int kFlushPeriodDuringDispatch = 256;
}  // namespace

GpuBackendMetal::GpuBackendMetal(GpuDelegateWaitType wait_type)
    : device_owned_(std::make_unique<::ml_drift::metal::MetalDevice>()),
      device_(device_owned_.get()),
      wait_type_(wait_type) {
  command_queue_ = [device_->device() newCommandQueue];
}

GpuBackendMetal::GpuBackendMetal(::ml_drift::metal::MetalDevice* device,
                                 GpuDelegateWaitType wait_type)
    : device_(device), wait_type_(wait_type) {
  command_queue_ = [device_->device() newCommandQueue];
}

GpuBackendMetal::GpuBackendMetal(::ml_drift::metal::MetalDevice* device,
                                 GpuDelegateWaitType wait_type, id<MTLCommandQueue> command_queue,
                                 bool enable_residency_set)
    : device_(device), wait_type_(wait_type), command_queue_(command_queue) {
  if (@available(macOS 15.0, iOS 18.0, *)) {
    InitResidencySet();
    bool start_thread = false;
    {
      absl::MutexLock lock(&residency_mutex_);
      residency_runtime_enabled_ = enable_residency_set;
      if (residency_runtime_enabled_) {
        residency_active_ = true;
        start_thread = true;
      } else {
        residency_active_ = false;
      }
    }
    if (start_thread) {
      StartHeartbeat();
    }
  }
}

GpuBackendMetal::~GpuBackendMetal() {
  StopHeartbeat();
  @autoreleasepool {
    absl::MutexLock lock(&residency_mutex_);
    ReleaseResidencyLocked();
  }
}

void GpuBackendMetal::InitResidencySet() {
  if (@available(macOS 15.0, iOS 18.0, *)) {
    @autoreleasepool {
      MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
      desc.label = @"LiteRT Model Residency Set";
      NSError *error = nil;
      residency_set_ = [device_->device() newResidencySetWithDescriptor:desc error:&error];
      if (residency_set_ != nil) {
        [command_queue_ addResidencySet:residency_set_];
        ABSL_LOG(INFO) << "Metal Residency Set successfully enabled and added to Command Queue.";
      } else {
        ABSL_LOG(ERROR) << "Failed to create Metal Residency Set: "
                        << [error.localizedDescription UTF8String];
      }
    }
  }
}

absl::string_view GpuBackendMetal::GetBackendName() { return kBackendName; }

void GpuBackendMetal::SetResidencyRuntimeEnabled(bool enabled) {
  if (residency_set_ == nil) return;

  bool start_thread = false;
  bool stop_thread = false;

  {
    absl::MutexLock lock(&residency_mutex_);
    if (residency_runtime_enabled_ == enabled) return;
    residency_runtime_enabled_ = enabled;
    if (enabled) {
      if (!residency_active_) {
        if (@available(macOS 15.0, iOS 18.0, *)) {
          [residency_set_ requestResidency];
          residency_active_ = true;
          start_thread = true;
          ABSL_LOG(INFO) << "Metal residency set dynamically enabled.";
        }
      }
    } else {
      if (residency_active_) {
        ReleaseResidencyLocked();
        stop_thread = true;
        ABSL_LOG(INFO) << "Metal residency set dynamically disabled.";
      }
    }
  }

  if (start_thread) {
    StartHeartbeat();
  }
  if (stop_thread) {
    StopHeartbeat();
  }
}

absl::string_view GpuBackendMetal::GetSerializedDataPrefix() { return kSerializedDataPrefix; }

absl::StatusOr<::ml_drift::GpuInfo> GpuBackendMetal::GetInfo() { return device_->GetInfo(); }

absl::StatusOr<::ml_drift::TensorStorageType> GpuBackendMetal::GetFastestStorageType() {
  return ::ml_drift::metal::GetFastestStorageType(device_->GetInfo());
}

absl::StatusOr<GpuMemoryHandle> GpuBackendMetal::GetGpuMemoryAllocated(
    const GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("GetGpuMemoryAllocated is not implemented.");
}

absl::StatusOr<GpuEventHandle> GpuBackendMetal::GetGpuEventAssociated(
    const GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("GetGpuEventAssociated is not implemented.");
}

absl::Status GpuBackendMetal::AssociateGpuEvent(GpuEventHandle event, LiteRtEnvironment env,
                                                GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("AssociateGpuEvent is not implemented.");
}

absl::Status GpuBackendMetal::WaitForCompletion() {
  if (command_buffer_ == nullptr) {
    // No pending GPU operations. No need to wait.
    return absl::OkStatus();
  }

  // We need only synchronization so volatile works better than atomic which reads from global
  // memory each time.
  __block volatile bool buffer_completed = false;

  if (wait_type_ == kGpuDelegateWaitTypeActive) {
    [command_buffer_ addCompletedHandler:^(id<MTLCommandBuffer>) {
      buffer_completed = true;
    }];
  }

  [command_buffer_ commit];

  if (wait_type_ == kGpuDelegateWaitTypeActive) {
    while (!buffer_completed) {
      // Busy wait. Use local variable. Volatile uses RAM access all the time.
      for (volatile int i = 0; i < 100; ++i) {
      }
    }
  } else if (wait_type_ == kGpuDelegateWaitTypePassive) {
    // Passive wait: this thread sleeps until GPU finishes.
    [command_buffer_ waitUntilCompleted];
  }

  // Reset the command buffer as it's committed above.
  command_buffer_ = nullptr;

  return absl::OkStatus();
}

absl::StatusOr<GpuBackend::GpuBufferRequirements> GpuBackendMetal::GetGpuBufferRequirements(
    ::ml_drift::TensorStorageType used_storage_type, ::ml_drift::DataType data_type) {
  return absl::UnimplementedError("GetGpuBufferRequirements is not implemented.");
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendMetal::GetGpuBufferRequirementsForNonExternalTensors() {
  return absl::UnimplementedError(
      "GetGpuBufferRequirementsForNonExternalTensors is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>> GpuBackendMetal::CreateInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info, ::ml_drift::GpuModel& gpu_model,
    std::vector<uint8_t>* serialized_model, bool may_share_memory_manager) {
  auto ctx = std::make_unique<GpuInferenceContextMetal>(
      this, may_share_memory_manager ? &memory_manager_ : nullptr);
  RETURN_IF_ERROR(ctx->metal_ctx().InitFromGpuModel(create_info, &gpu_model, device_->device(),
                                                    serialized_model));
  if (@available(macOS 15.0, iOS 18.0, *)) {
    PopulateResidencySet(create_info, ctx->metal_ctx());
  }
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>> GpuBackendMetal::RestoreInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const absl::Span<const uint8_t> serialized_model) {
  auto ctx = std::make_unique<GpuInferenceContextMetal>(this, &memory_manager_);
  RETURN_IF_ERROR(
      ctx->metal_ctx().RestoreDeserialized(serialized_model, device_->device(), &create_info));
  if (@available(macOS 15.0, iOS 18.0, *)) {
    PopulateResidencySet(create_info, ctx->metal_ctx());
  }
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
GpuBackendMetal::CreateSharedMemoryManager(
    const ::ml_drift::CreateGpuModelInfo& create_info, ::ml_drift::GraphFloat32& graph,
    TfLiteContext* context, MlDriftDelegateData& delegate_data,
    ::ml_drift::SerializationWeightCache* serialization_cache) {
  return ::ml_drift::MakeSharedMemoryManagerMetal(
      device_, create_info, graph, context, GetBufferIdToSpatialTensorMap(delegate_data),
      GetQuantParamIdToSpatialTensorMap(delegate_data),
      delegate_data.options->has_prepacked_external_tflite_tensors, serialization_cache,
      delegate_data.options->madvise_original_shared_tensors);
}

absl::StatusOr<std::shared_ptr<::ml_drift::WeightsManager>>
GpuBackendMetal::CreateWeightsManager() {
  return std::make_shared<::ml_drift::metal::MetalWeightsManager>();
}

absl::StatusOr<std::vector<std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
GpuBackendMetal::GetBatchesForWeightsPreparation(::ml_drift::WeightsManager* weights_manager) {
  auto* metal_weights_manager =
      static_cast<::ml_drift::metal::MetalWeightsManager*>(weights_manager);
  return metal_weights_manager->GetBatchesForWeightsPreparation(
      *device_, ::ml_drift::WeightsManager::ScheduleStrategy::kBatchByMaxWeightSize);
}

absl::StatusOr<
    absl::flat_hash_map<::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendMetal::PrepareWeightsInBatch(
    ::ml_drift::WeightsManager* weights_manager,
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>& op_infos) {
  auto* metal_weights_manager =
      static_cast<::ml_drift::metal::MetalWeightsManager*>(weights_manager);
  return metal_weights_manager->PrepareWeightsInBatch(*device_, op_infos);
}

absl::StatusOr<
    absl::flat_hash_map<::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendMetal::PrepareWeightsInBatches(::ml_drift::WeightsManager* weights_manager) {
  auto* metal_weights_manager =
      static_cast<::ml_drift::metal::MetalWeightsManager*>(weights_manager);
  return metal_weights_manager->PrepareWeightsInBatches(
      *device_, ::ml_drift::WeightsManager::ScheduleStrategy::kBatchByMaxWeightSize);
}

absl::StatusOr<std::unique_ptr<GpuTensorWrapper>> GpuBackendMetal::CreateTensorWrapper(
    const ::ml_drift::TensorDescriptor& desc, GpuMemoryHandle gpu_memory) {
  auto metal_tensor = std::make_unique<GpuTensorWrapperMetal>();
  const ::ml_drift::metal::MetalSpatialTensor* metal_spatial_tensor =
      reinterpret_cast<::ml_drift::metal::MetalSpatialTensor*>(gpu_memory);
  if (metal_spatial_tensor->GetBufferHandle() != nullptr) {
    RETURN_IF_ERROR(::ml_drift::metal::CreateTensorSharedBuffer(
        metal_spatial_tensor->GetBufferHandle(), desc, &metal_tensor->metal_tensor()));
  } else if (metal_spatial_tensor->GetTextureHandle() != nullptr) {
    RETURN_IF_ERROR(::ml_drift::metal::CreateTensorSharedTexture(
        metal_spatial_tensor->GetTextureHandle(), desc, &metal_tensor->metal_tensor()));
  } else {
    return absl::InvalidArgumentError("MetalSpatialTensor has no buffer and texture handle.");
  }

  return std::move(metal_tensor);
}

absl::Status GpuBackendMetal::ReadSpatialTensorToDescriptor(::ml_drift::GpuSpatialTensor& tensor,
                                                            ::ml_drift::TensorDescriptor& desc) {
  auto* metal_tensor = static_cast<::ml_drift::metal::MetalSpatialTensor*>(&tensor);
  return metal_tensor->ToDescriptor(&desc, device_->device());
}

absl::Status GpuBackendMetal::UpdateSpatialTensor(
    ::ml_drift::GpuSpatialTensor* tensor, const ::ml_drift::TensorDescriptor& desc,
    size_t page_adjusted_offset, ::ml_drift_delegate::ReleaseDataCallback release_data_callback) {
  RETURN_IF_ERROR(ReleaseSpatialTensorMemory(tensor));

  auto* metal_tensor = static_cast<::ml_drift::metal::MetalSpatialTensor*>(tensor);

  if (release_data_callback && desc.GetStorageType() == ::ml_drift::TensorStorageType::BUFFER &&
      (desc.GetDataType() == ::ml_drift::DataType::UINT8 ||
       desc.GetDataType() == ::ml_drift::DataType::UINT4 ||
       desc.GetDataType() == ::ml_drift::DataType::UINT2)) {
    const uint8_t* data = desc.GetData().data();
    size_t size = desc.GetData().size();
    const uint8_t* raw_data = data - page_adjusted_offset;
    size_t raw_size = size + page_adjusted_offset;

    auto* release_cb_ptr =
        new ::ml_drift_delegate::ReleaseDataCallback(std::move(release_data_callback));
    id<MTLBuffer> buffer =
        [device_->device() newBufferWithBytesNoCopy:const_cast<uint8_t*>(raw_data)
                                             length:raw_size
                                            options:MTLResourceStorageModeShared
                                        deallocator:^(void* pointer, NSUInteger length) {
                                          (**release_cb_ptr)();
                                          delete release_cb_ptr;
                                        }];
    if (buffer == nil) {
      (**release_cb_ptr)();
      delete release_cb_ptr;
      return absl::InternalError("Failed to create Metal buffer with no copy.");
    }
    ::ml_drift::TensorDescriptor desc_no_data;
    desc.CopyWithoutData(&desc_no_data);
    return ::ml_drift::metal::CreateTensorSharedBuffer(buffer, desc_no_data, metal_tensor,
                                                       page_adjusted_offset);
  }

  auto status = metal_tensor->CreateFromDescriptor(desc, device_->device());
  if (release_data_callback) {
    (*release_data_callback)();
  }
  return status;
}

absl::Status GpuBackendMetal::ReleaseSpatialTensorMemory(::ml_drift::GpuSpatialTensor* tensor) {
  auto* metal_tensor = static_cast<::ml_drift::metal::MetalSpatialTensor*>(tensor);
  metal_tensor->Release();
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>> GpuBackendMetal::CreateIOBuffer(
    GpuMemoryHandle gpu_memory) {
  return absl::UnimplementedError("CreateIOBuffer is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>> GpuBackendMetal::CreateIOBufferWithSize(
    ::ml_drift::DataType data_type, size_t size, bool input) {
  ::ml_drift::metal::Buffer buffer;
  RETURN_IF_ERROR(
      ::ml_drift::metal::CreateBuffer(size, /*data=*/nullptr, device_->device(), &buffer));
  return std::make_unique<GpuIOBufferMetal>(this, std::move(buffer));
}

absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
GpuBackendMetal::CreateTensor2BufferConverter(const ::ml_drift::TensorDescriptor& src_desc,
                                              const ::ml_drift::BufferDescriptor& dst_desc) {
  auto converter = std::make_unique<::ml_drift::metal::TensorToBHWCBufferConverter>();
  RETURN_IF_ERROR(converter->Init(device_, src_desc, dst_desc.element_type));
  return std::make_unique<Tensor2BufferConverterMetal>(this, std::move(converter));
}

absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
GpuBackendMetal::CreateBuffer2TensorConverter(const ::ml_drift::BufferDescriptor& src_desc,
                                              const ::ml_drift::TensorDescriptor& dst_desc) {
  auto converter = std::make_unique<::ml_drift::metal::BHWCBufferToTensorConverter>();
  RETURN_IF_ERROR(converter->Init(device_, src_desc.element_type, dst_desc));
  return std::make_unique<Buffer2TensorConverterMetal>(this, std::move(converter));
}

GpuInferenceContextMetal::GpuInferenceContextMetal(GpuBackendMetal* backend,
                                                   ::ml_drift::metal::MemoryManager* memory_manager)
    : backend_(backend),
      ctx_(memory_manager != nullptr
               ? std::make_unique<::ml_drift::metal::InferenceContext>(memory_manager)
               : std::make_unique<::ml_drift::metal::InferenceContext>()) {}

absl::StatusOr<::ml_drift::GpuSpatialTensor*> GpuInferenceContextMetal::GetSpatialTensor(
    ::ml_drift::ValueId id) {
  return ctx_->GetTensor(id);
}

absl::Status GpuInferenceContextMetal::BindSpatialTensor(::ml_drift::ValueId tensor_id,
                                                         ::ml_drift::GpuSpatialTensor* tensor) {
  RETURN_IF_ERROR(
      ctx_->SetTensor(tensor_id, static_cast<::ml_drift::metal::MetalSpatialTensor*>(tensor)));
  if (@available(macOS 15.0, iOS 18.0, *)) {
    id<MTLResidencySet> res_set = backend_->residency_set();
    if (res_set != nil) {
      @autoreleasepool {
        auto* metal_tensor = static_cast<::ml_drift::metal::MetalSpatialTensor*>(tensor);
        if (metal_tensor != nullptr) {
          bool added = false;
          if (metal_tensor->GetBufferHandle() != nil) {
            [res_set addAllocation:metal_tensor->GetBufferHandle()];
            added = true;
          }
          if (metal_tensor->GetTextureHandle() != nil) {
            [res_set addAllocation:metal_tensor->GetTextureHandle()];
            added = true;
          }
          if (added) {
            [res_set commit];
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status GpuInferenceContextMetal::WriteDataToWeightTensor(::ml_drift::ValueId id,
                                                               absl::Span<const uint8_t> data) {
  auto* metal_tensor = ctx_->GetTensor(id);
  return metal_tensor->WriteData(backend_->command_queue(), data.data(),
                                 /*wait_for_completion=*/false);
}

absl::Status GpuInferenceContextMetal::ReadWeightTensorToDescriptor(
    ::ml_drift::ValueId id, ::ml_drift::TensorDescriptor& desc) {
  auto* metal_tensor = ctx_->GetTensor(id);
  return metal_tensor->ToDescriptor(&desc, backend_->metal_device()->device());
}

absl::Status GpuInferenceContextMetal::Dispatch() {
  // If it will be waiting actively, enqueue commands into multiple command buffers.
  if (backend_->wait_type() == kGpuDelegateWaitTypeActive) {
    // Commit the previous command buffer if any.
    if (backend_->command_buffer() != nullptr) {
      [backend_->command_buffer() commit];
    }

    // This drains all temporary Metal objects created during the graph encoding immediately.
    @autoreleasepool {
      ctx_->EncodeWithCommandQueue(backend_->command_queue(), kFlushPeriodDuringDispatch);
    }
    // Need a new command buffer for conversion and waiting for completion.
    backend_->set_command_buffer([backend_->command_queue() commandBuffer]);
    return absl::OkStatus();
  }

  if (backend_->command_buffer() == nullptr) {
    backend_->set_command_buffer([backend_->command_queue() commandBuffer]);
  } else if ([backend_->command_buffer() status] >= MTLCommandBufferStatusCommitted) {
    return absl::FailedPreconditionError("Command buffer is already committed.");
  }

  ctx_->EncodeWithCommandBuffer(backend_->command_buffer());
  return absl::OkStatus();
}

absl::StatusOr<GpuEventHandle> GpuInferenceContextMetal::GetPreDispatchEvent() {
  return absl::UnimplementedError("GetPreDispatchEvent is not implemented.");
}

absl::StatusOr<GpuEventHandle> GpuInferenceContextMetal::GetPostDispatchEvent(
    bool is_async_execution_mode) {
  return absl::UnimplementedError("GetPostDispatchEvent is not implemented.");
}

absl::Status GpuInferenceContextMetal::WaitForEventsCompleted(absl::Span<GpuEventHandle> events,
                                                              bool force_sync) {
  return absl::UnimplementedError("WaitForEventsCompleted is not implemented.");
}

absl::Status GpuInferenceContextMetal::PreConvert(bool input) {
  if (backend_->command_buffer() == nullptr) {
    backend_->set_command_buffer([backend_->command_queue() commandBuffer]);
  } else if ([backend_->command_buffer() status] >= MTLCommandBufferStatusCommitted) {
    return absl::FailedPreconditionError("Command buffer is already committed.");
  }
  backend_->set_compute_command_encoder([backend_->command_buffer() computeCommandEncoder]);
  return absl::OkStatus();
}

absl::Status GpuInferenceContextMetal::PostConvert(bool input) {
  [backend_->compute_command_encoder() endEncoding];
  backend_->set_compute_command_encoder(nullptr);

  // If output conversion is done, waits for conversion to finish.
  if (!input) {
    RETURN_IF_ERROR(backend_->WaitForCompletion());
  }

  return absl::OkStatus();
}

absl::Status GpuInferenceContextMetal::Profile(::ml_drift::ProfilingInfo& profiling_info) {
  return absl::UnimplementedError("Profile is not implemented.");
}

absl::StatusOr<size_t> GpuInferenceContextMetal::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  return absl::UnimplementedError(
      "GetSizeOfMemoryAllocatedForIntermediateTensors is not implemented.");
}

absl::StatusOr<size_t> GpuInferenceContextMetal::GetSizeOfMemoryAllocatedForConstantTensors()
    const {
  return absl::UnimplementedError("GetSizeOfMemoryAllocatedForConstantTensors is not implemented.");
}

absl::StatusOr<size_t> GpuInferenceContextMetal::GetSizeOfMemoryAllocatedForExternalTensors()
    const {
  return absl::UnimplementedError("GetSizeOfMemoryAllocatedForExternalTensors is not implemented.");
}

absl::Status GpuInferenceContextMetal::ReportMemoryBenchmarkIfEnabled(
    const ::ml_drift::CreateGpuModelInfo& create_info) {
#ifdef ML_DRIFT_MEM_STATS
  std::cout << ml_drift_delegate::GetMemoryBenchmarkReport(*ctx_, create_info) << std::endl;
#endif
  return absl::OkStatus();
}

absl::Status GpuInferenceContextMetal::SetCommandBufferHint(int num_nodes_per_command_encoder) {
  return absl::UnimplementedError("SetCommandBufferHint is not implemented.");
}

GpuIOBufferMetal::GpuIOBufferMetal(GpuBackendMetal* backend, ::ml_drift::metal::Buffer buffer)
    : backend_(backend), buffer_(std::move(buffer)) {}

absl::Status GpuIOBufferMetal::Read(absl::Span<uint8_t> data) {
  std::memcpy(data.data(), [buffer_.GetMemoryPtr() contents], data.size());
  return absl::OkStatus();
}

absl::Status GpuIOBufferMetal::Write(absl::Span<const uint8_t> data) {
  std::memcpy([buffer_.GetMemoryPtr() contents], data.data(), data.size());
  return absl::OkStatus();
}

Tensor2BufferConverterMetal::Tensor2BufferConverterMetal(
    GpuBackendMetal* backend,
    std::unique_ptr<::ml_drift::metal::TensorToBHWCBufferConverter> converter)
    : backend_(backend), converter_(std::move(converter)) {}

absl::Status Tensor2BufferConverterMetal::Convert(::ml_drift::GpuSpatialTensor& src_tensor,
                                                  GpuIOBuffer& dst_buffer) {
  if (backend_->compute_command_encoder() == nullptr) {
    return absl::FailedPreconditionError("Compute command encoder is not set.");
  }

  return converter_->Encode(
      backend_->compute_command_encoder(),
      static_cast<::ml_drift::metal::MetalSpatialTensor*>(&src_tensor),
      static_cast<GpuIOBufferMetal&>(dst_buffer).metal_buffer().GetMemoryPtr());
}

Buffer2TensorConverterMetal::Buffer2TensorConverterMetal(
    GpuBackendMetal* backend,
    std::unique_ptr<::ml_drift::metal::BHWCBufferToTensorConverter> converter)
    : backend_(backend), converter_(std::move(converter)) {}

absl::Status Buffer2TensorConverterMetal::Convert(GpuIOBuffer& src_buffer,
                                                  ::ml_drift::GpuSpatialTensor& dst_tensor) {
  if (backend_->compute_command_encoder() == nullptr) {
    return absl::FailedPreconditionError("Compute command encoder is not set.");
  }

  return converter_->Encode(
      backend_->compute_command_encoder(),
      static_cast<GpuIOBufferMetal&>(src_buffer).metal_buffer().GetMemoryPtr(),
      static_cast<::ml_drift::metal::MetalSpatialTensor*>(&dst_tensor));
}

void GpuBackendMetal::PopulateResidencySet(const ::ml_drift::CreateGpuModelInfo& create_info,
                                           ::ml_drift::metal::InferenceContext& metal_ctx) {
  if (residency_set_ == nil) return;
  if (@available(macOS 15.0, iOS 18.0, *)) {
    @autoreleasepool {
      metal_ctx.AddConstantsToResidencySet(residency_set_);
      for (const auto& [id, tensor] : create_info.external_immutable_tensors) {
        auto* metal_tensor = static_cast<::ml_drift::metal::MetalSpatialTensor*>(tensor);
        if (metal_tensor != nullptr) {
          if (metal_tensor->GetBufferHandle() != nil) {
            [residency_set_ addAllocation:metal_tensor->GetBufferHandle()];
          }
          if (metal_tensor->GetTextureHandle() != nil) {
            [residency_set_ addAllocation:metal_tensor->GetTextureHandle()];
          }
        }
      }
      [residency_set_ commit];
      bool req_residency = false;
      {
        absl::MutexLock lock(&residency_mutex_);
        req_residency = residency_runtime_enabled_;
        if (req_residency) {
          residency_active_ = true;
        }
      }
      if (req_residency) {
        [residency_set_ requestResidency];
      }
    }
  }
}

void GpuBackendMetal::StartHeartbeat() {
  absl::MutexLock lock(&residency_mutex_);
  if (!residency_runtime_enabled_) return;
  if (heartbeat_thread_.joinable()) return;
  stop_heartbeat_ = std::make_unique<absl::Notification>();
  absl::Notification* stop_notif = stop_heartbeat_.get();
  heartbeat_thread_ = std::thread([this, stop_notif]() {
    while (true) {
      if (stop_notif->WaitForNotificationWithTimeout(absl::Milliseconds(100))) {
        break;
      }
      @autoreleasepool {
        if (@available(macOS 15.0, iOS 18.0, *)) {
          absl::MutexLock lock(&residency_mutex_);
          if (residency_active_ && residency_set_ != nil && residency_runtime_enabled_) {
            [residency_set_ requestResidency];
          }
        }
      }
    }
  });
}

void GpuBackendMetal::StopHeartbeat() {
  std::thread t_to_join;
  {
    absl::MutexLock lock(&residency_mutex_);
    if (stop_heartbeat_) {
      stop_heartbeat_->Notify();
    }
    if (heartbeat_thread_.joinable()) {
      t_to_join = std::move(heartbeat_thread_);
    }
  }
  if (t_to_join.joinable()) {
    t_to_join.join();
  }
  {
    absl::MutexLock lock(&residency_mutex_);
    stop_heartbeat_.reset();
  }
}

void GpuBackendMetal::ReleaseResidencyLocked() {
  if (residency_active_) {
    if (@available(macOS 15.0, iOS 18.0, *)) {
      [residency_set_ endResidency];
      residency_active_ = false;
    }
  }
}

}  // namespace litert::ml_drift
