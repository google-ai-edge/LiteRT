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

#include "ml_drift_delegate/delegate/gpu_backend_webgpu.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#ifdef __EMSCRIPTEN__
#include <algorithm>
#include <optional>
#endif  // __EMSCRIPTEN__
#include <thread>  // NOLINT (Open source code)
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift/webgpu/buffer.h"  // from @ml_drift
#include "ml_drift/webgpu/converter.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/inference_context.h"  // from @ml_drift
#include "ml_drift/webgpu/memory_manager.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_api_util.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_weights_manager.h"  // from @ml_drift
#ifdef ML_DRIFT_MEM_STATS
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_webgpu_benchmark_util.h"  // IWYU pragma: keep
#endif
#include "ml_drift_delegate/delegate/delegate_utils.h"
// clang-format off
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
// clang-format on
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_webgpu.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/c/common.h"

#ifdef __EMSCRIPTEN__
#include "weight_loader/external_weight_loader_litert.h"
#endif  // __EMSCRIPTEN__

#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>  // IWYU pragma: export
#else
#include "webgpu/webgpu.h"  // from @dawn
#endif  // __EMSCRIPTEN__

namespace litert::ml_drift {
namespace {

constexpr absl::string_view kBackendName = "WebGPU";
// Serialization data prefix. FYI, Jet uses "gpuv2_data_".
constexpr absl::string_view kSerializedDataPrefix = "gpuv3_wgpu_data_";

// At least Adreno 730 and some Intel GPUs fail sometimes with 512(or more)
// nodes per encoder.
// TODO: b/403337563 - Find a better way to set this value.
#if defined(__ANDROID__)
constexpr int kNumNodesPerCommandEncoder = 64;
#else
constexpr int kNumNodesPerCommandEncoder = 256;
#endif

#ifdef __EMSCRIPTEN__
// Traverses the weights conversion graph forward from a raw weight input tensor
// to find its corresponding processed output tensor mapped to the main model.
// Used to map WebGPU buffers with external weight buffer IDs for streaming.
std::optional<::ml_drift::ValueId> FindMappedOutput(
    const ::ml_drift::GpuModel& model, ::ml_drift::ValueId input_id,
    const absl::flat_hash_map<::ml_drift::ValueId, ::ml_drift::ValueId>&
        io_mapping) {
  absl::flat_hash_set<::ml_drift::ValueId> visited;
  std::vector<::ml_drift::ValueId> stack = {input_id};
  while (!stack.empty()) {
    auto current = stack.back();
    stack.pop_back();
    if (!visited.insert(current).second) {
      continue;
    }

    if (io_mapping.contains(current)) {
      return current;
    }

    for (const auto& node : model.nodes) {
      if (std::find(node.inputs.begin(), node.inputs.end(), current) !=
          node.inputs.end()) {
        for (auto output : node.outputs) {
          stack.push_back(output);
        }
      }
    }
  }
  return std::nullopt;
}
#endif  // __EMSCRIPTEN__

std::vector<std::shared_ptr<absl::Notification>> GetAsyncUploadNotifications(
    const ::ml_drift::CreateGpuModelInfo& create_info) {
  std::vector<std::shared_ptr<absl::Notification>> notifications;
#ifndef __EMSCRIPTEN__
  for (const auto& [_, tensor] : create_info.external_immutable_tensors) {
    auto* notified_tensor =
        dynamic_cast<const ::ml_drift::webgpu::NotifiedTensor*>(tensor);
    if (notified_tensor && notified_tensor->upload_notification()) {
      notifications.push_back(notified_tensor->upload_notification());
    }
  }
#endif  // __EMSCRIPTEN__
  return notifications;
}

}  // namespace

GpuBackendWebGpu::GpuBackendWebGpu()
    : env_owned_(std::make_unique<::ml_drift::webgpu::ExecutionEnvironment>(
#if defined(__APPLE__)
          wgpu::BackendType::Metal
#elif defined(_WIN32)
          wgpu::BackendType::D3D12
#elif defined(__EMSCRIPTEN__)
          wgpu::BackendType::WebGPU
#else
          wgpu::BackendType::Vulkan
#endif
          )),
      env_(env_owned_.get()) {
}

GpuBackendWebGpu::GpuBackendWebGpu(
    ::ml_drift::webgpu::ExecutionEnvironment* env)
    : env_(env) {}

absl::string_view GpuBackendWebGpu::GetBackendName() { return kBackendName; }

absl::string_view GpuBackendWebGpu::GetSerializedDataPrefix() {
  return kSerializedDataPrefix;
}

absl::StatusOr<::ml_drift::GpuInfo> GpuBackendWebGpu::GetInfo() {
  return env_->GetInfo();
}

absl::StatusOr<::ml_drift::TensorStorageType>
GpuBackendWebGpu::GetFastestStorageType() {
  return ::ml_drift::webgpu::GetFastestStorageType(env_->GetInfo());
}

absl::StatusOr<GpuMemoryHandle> GpuBackendWebGpu::GetGpuMemoryAllocated(
    const GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("GetGpuMemoryAllocated is not implemented.");
}

absl::StatusOr<GpuEventHandle> GpuBackendWebGpu::GetGpuEventAssociated(
    const GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("GetGpuEventAssociated is not implemented.");
}

absl::Status GpuBackendWebGpu::AssociateGpuEvent(
    GpuEventHandle event, LiteRtEnvironment env,
    GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("AssociateGpuEvent is not implemented.");
}

absl::Status GpuBackendWebGpu::WaitForCompletion() {
  return ::ml_drift::webgpu::WaitUntilCompleted(env_->queue(), env_->device(),
                                                absl::Seconds(10));
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendWebGpu::GetGpuBufferRequirements(
    ::ml_drift::TensorStorageType used_storage_type,
    ::ml_drift::DataType data_type) {
  return absl::UnimplementedError(
      "GetGpuBufferRequirements is not implemented.");
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendWebGpu::GetGpuBufferRequirementsForNonExternalTensors() {
  return absl::UnimplementedError(
      "GetGpuBufferRequirementsForNonExternalTensors is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendWebGpu::CreateInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
    bool may_share_memory_manager) {
  auto ctx = std::make_unique<GpuInferenceContextWebGpu>(
      this, may_share_memory_manager ? &memory_manager_ : nullptr, create_info,
      num_steps_of_command_buffer_preparations_);
  RETURN_IF_ERROR(ctx->wgpu_ctx().InitFromGpuModel(
      *env_, create_info, &gpu_model, serialized_model));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendWebGpu::RestoreInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const absl::Span<const uint8_t> serialized_model) {
  auto ctx = std::make_unique<GpuInferenceContextWebGpu>(
      this, &memory_manager_, create_info,
      num_steps_of_command_buffer_preparations_);
  RETURN_IF_ERROR(ctx->wgpu_ctx().RestoreDeserialized(serialized_model, *env_,
                                                      &create_info));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
GpuBackendWebGpu::CreateSharedMemoryManager(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GraphFloat32& graph, TfLiteContext* context,
    MlDriftDelegateData& delegate_data,
    ::ml_drift::SerializationWeightCache* serialization_cache) {
  return ::ml_drift::MakeSharedMemoryManagerWebgpu(
      *env_, create_info, graph, context,
      GetBufferIdToSpatialTensorMap(delegate_data),
      GetQuantParamIdToSpatialTensorMap(delegate_data),
      delegate_data.options->has_prepacked_external_tflite_tensors,
      serialization_cache, delegate_data.upload_executor,
      delegate_data.options->madvise_original_shared_tensors);
}

absl::StatusOr<std::shared_ptr<::ml_drift::WeightsManager>>
GpuBackendWebGpu::CreateWeightsManager() {
  return std::make_shared<::ml_drift::webgpu::WebGpuWeightsManager>();
}

absl::StatusOr<std::vector<
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
GpuBackendWebGpu::GetBatchesForWeightsPreparation(
    ::ml_drift::WeightsManager* weights_manager) {
  auto* webgpu_weights_manager =
      static_cast<::ml_drift::webgpu::WebGpuWeightsManager*>(weights_manager);
  return webgpu_weights_manager->GetBatchesForWeightsPreparation(
      *env_,
      ::ml_drift::WeightsManager::ScheduleStrategy::kBatchByMaxWeightSize);
}

absl::StatusOr<absl::flat_hash_map<
    ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendWebGpu::PrepareWeightsInBatch(
    ::ml_drift::WeightsManager* weights_manager,
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
        op_infos) {
  auto* webgpu_weights_manager =
      static_cast<::ml_drift::webgpu::WebGpuWeightsManager*>(weights_manager);
  return webgpu_weights_manager->PrepareWeightsInBatch(*env_, op_infos);
}

absl::StatusOr<absl::flat_hash_map<
    ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendWebGpu::PrepareWeightsInBatches(
    ::ml_drift::WeightsManager* weights_manager) {
  return absl::UnimplementedError(
      "PrepareWeightsInBatches is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuTensorWrapper>>
GpuBackendWebGpu::CreateTensorWrapper(const ::ml_drift::TensorDescriptor& desc,
                                      GpuMemoryHandle gpu_memory) {
  return std::make_unique<GpuTensorWrapperWebGpu>(
      reinterpret_cast<::ml_drift::webgpu::SpatialTensor*>(gpu_memory));
}

absl::Status GpuBackendWebGpu::ReadSpatialTensorToDescriptor(
    ::ml_drift::GpuSpatialTensor& tensor, ::ml_drift::TensorDescriptor& desc) {
  auto* wgpu_tensor = static_cast<::ml_drift::webgpu::SpatialTensor*>(&tensor);
  return wgpu_tensor->ToDescriptor(env_->device(), &desc);
}

absl::Status GpuBackendWebGpu::UpdateSpatialTensor(
    ::ml_drift::GpuSpatialTensor* tensor,
    const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
    ReleaseDataCallback release_data_callback) {
  // If there is any tensor memory, release it before we update the tensor.
  RETURN_IF_ERROR(ReleaseSpatialTensorMemory(tensor));

  auto* wgpu_tensor = static_cast<::ml_drift::webgpu::SpatialTensor*>(tensor);

#if defined(__APPLE__)
  // If the device is Apple and it meets the right storage & data types, use
  // an optimized path.
  if (desc.GetStorageType() == ::ml_drift::TensorStorageType::BUFFER &&
      (desc.GetDataType() == ::ml_drift::DataType::UINT8 ||
       desc.GetDataType() == ::ml_drift::DataType::UINT4 ||
       desc.GetDataType() == ::ml_drift::DataType::UINT2)) {
    ::ml_drift::TensorDescriptor desc_no_data;
    desc.CopyWithoutData(&desc_no_data);
    RETURN_IF_ERROR(
        wgpu_tensor->CreateFromDescriptor(env_->device(), desc_no_data));

    return ::ml_drift::CopyBufferToBuffer(env_, desc, page_adjusted_offset,
                                          std::move(release_data_callback),
                                          wgpu_tensor);
  }
#endif  // __APPLE__

  // By default, use the generic path.
  return wgpu_tensor->CreateFromDescriptor(env_->device(), desc);
}

absl::Status GpuBackendWebGpu::ReleaseSpatialTensorMemory(
    ::ml_drift::GpuSpatialTensor* tensor) {
  auto* wtensor = static_cast<::ml_drift::webgpu::SpatialTensor*>(tensor);
  wtensor->Release();
#if !defined(__EMSCRIPTEN__)
  this->env_->device().Tick();
#endif  // !__EMSCRIPTEN__
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>> GpuBackendWebGpu::CreateIOBuffer(
    GpuMemoryHandle gpu_memory) {
  return absl::UnimplementedError("CreateIOBuffer is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
GpuBackendWebGpu::CreateIOBufferWithSize(::ml_drift::DataType data_type,
                                         size_t size, bool input) {
  size_t size_aligned =
      ::ml_drift::AlignByN(size, ::ml_drift::SizeOf(data_type) * 4);
  auto wgpu_buffer =
      ::ml_drift::webgpu::CreateBufferStorage(env_->device(), size_aligned);
  if (input) {
    return std::make_unique<GpuIOBufferWebGpu>(this, std::move(wgpu_buffer));
  }

  wgpu::BufferDescriptor mappable_buffer_desc{
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = size_aligned,
  };
  auto buffer = std::make_unique<GpuIOBufferWebGpu>(
      this, std::move(wgpu_buffer),
      std::make_unique<::ml_drift::webgpu::Buffer>(
          env_->device().CreateBuffer(&mappable_buffer_desc), size_aligned));
  return std::move(buffer);
}

absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
GpuBackendWebGpu::CreateTensor2BufferConverter(
    const ::ml_drift::TensorDescriptor& src_desc,
    const ::ml_drift::BufferDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::webgpu::TensorToBHWCBufferConverter>();
  RETURN_IF_ERROR(converter->Init(*env_, src_desc, dst_desc.element_type));
  return std::make_unique<Tensor2BufferConverterWebGpu>(this,
                                                        std::move(converter));
}

absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
GpuBackendWebGpu::CreateBuffer2TensorConverter(
    const ::ml_drift::BufferDescriptor& src_desc,
    const ::ml_drift::TensorDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::webgpu::BHWCBufferToTensorConverter>();
  RETURN_IF_ERROR(converter->Init(*env_, src_desc.element_type, dst_desc));
  return std::make_unique<Buffer2TensorConverterWebGpu>(this,
                                                        std::move(converter));
}

GpuInferenceContextWebGpu::GpuInferenceContextWebGpu(
    GpuBackendWebGpu* backend,
    ::ml_drift::webgpu::MemoryManager* memory_manager,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    int num_steps_of_command_buffer_preparations)
    : backend_(backend),
      ctx_(memory_manager != nullptr
               ? std::make_unique<::ml_drift::webgpu::InferenceContext>(
                     memory_manager)
               : std::make_unique<::ml_drift::webgpu::InferenceContext>()),
      async_upload_notifications_(GetAsyncUploadNotifications(create_info)),
      next_command_buffers_(num_steps_of_command_buffer_preparations),
      num_nodes_per_command_encoder_(kNumNodesPerCommandEncoder) {}

GpuInferenceContextWebGpu::~GpuInferenceContextWebGpu() {
  if (next_command_buffers_thread_) {
    next_command_buffers_thread_->join();
  }
}

absl::StatusOr<::ml_drift::GpuSpatialTensor*>
GpuInferenceContextWebGpu::GetSpatialTensor(::ml_drift::ValueId id) {
  return ctx_->GetTensor(id);
}

absl::Status GpuInferenceContextWebGpu::BindSpatialTensor(
    ::ml_drift::ValueId id, ::ml_drift::GpuSpatialTensor* tensor) {
  return ctx_->SetTensor(
      id, static_cast<::ml_drift::webgpu::SpatialTensor*>(tensor));
}

absl::Status GpuInferenceContextWebGpu::WriteDataToWeightTensor(
    ::ml_drift::ValueId id, absl::Span<const uint8_t> data) {
  auto* wgpu_tensor = ctx_->GetTensor(id);
  return wgpu_tensor->WriteDataViaStaging(backend_->wgpu_env(), data.data());
}

absl::Status GpuInferenceContextWebGpu::ReadWeightTensorToDescriptor(
    ::ml_drift::ValueId id, ::ml_drift::TensorDescriptor& desc) {
  auto* wgpu_tensor = ctx_->GetTensor(id);
  return wgpu_tensor->ToDescriptor(backend_->wgpu_env().device(), &desc);
}

absl::Status GpuInferenceContextWebGpu::UploadWeightsOnWeb(
    weight_loader::WeightLoader* weight_loader,
    const ::ml_drift::GpuModel& gpu_model,
    const absl::flat_hash_map<::ml_drift::ValueId, ::ml_drift::ValueId>&
        io_mapping,
    const absl::flat_hash_map<::ml_drift::ValueId, uint32_t>&
        weight_id_to_external_buffer_id,
    std::vector<::ml_drift::WeightsManager::UploadWeightsInfo>& upload_infos) {
#ifdef __EMSCRIPTEN__
  absl::flat_hash_map<int, wgpu::Buffer> tfl_id_to_wgpu_buffer;
  for (const auto& upload_info : upload_infos) {
    ASSIGN_OR_RETURN(auto* tensor, GetSpatialTensor(upload_info.input_id));
    auto* wgpu_tensor = static_cast<::ml_drift::webgpu::SpatialTensor*>(tensor);

    auto mapped_output_opt =
        FindMappedOutput(gpu_model, upload_info.input_id, io_mapping);
    if (!mapped_output_opt.has_value()) {
      return absl::InternalError(
          "Could not find mapped output for raw weight.");
    }
    auto main_model_weight_id = io_mapping.at(*mapped_output_opt);

    auto it = weight_id_to_external_buffer_id.find(main_model_weight_id);
    if (it == weight_id_to_external_buffer_id.end()) {
      return absl::InternalError("Could not find external buffer ID.");
    }

    tfl_id_to_wgpu_buffer[it->second] = wgpu_tensor->GetBufferHandle();
  }

  RETURN_IF_ERROR(weight_loader->UploadWeightsOnWeb(tfl_id_to_wgpu_buffer));

  return absl::OkStatus();
#else
  return absl::UnimplementedError("UploadWeightsOnWeb is only supported on Emscripten.");
#endif  // __EMSCRIPTEN__
}

absl::Status GpuInferenceContextWebGpu::PrepareCommandBuffers(
    std::vector<wgpu::CommandBuffer>& command_buffers,
    bool submit_command_buffers) {
  ASSIGN_OR_RETURN(command_buffers,
                   ctx_->CreateCommandBuffers(backend_->wgpu_env(),
                                              num_nodes_per_command_encoder_,
                                              /*command_buffer_infos=*/nullptr,
                                              submit_command_buffers));
  if (submit_command_buffers) {
    ::ml_drift::webgpu::FlushIfCallbackSet();
  }
  return absl::OkStatus();
}

GpuInferenceContextWebGpu* GpuInferenceContextWebGpu::last_ctx_ = nullptr;

absl::Status GpuInferenceContextWebGpu::PrepareCommandBuffersFromCached(
    std::vector<wgpu::CommandBuffer>& command_buffers) {
  if (next_command_buffers_.empty()) {
    return PrepareCommandBuffers(command_buffers,
                                 /*submit_command_buffers=*/true);
  }

  if (next_command_buffers_thread_) {
    next_command_buffers_thread_->join();
  }

  if (last_ctx_ != this) {
    last_ctx_ = this;
    // The next_command_buffers_ must be invalidated since the last Dispatch()
    // call was done by a different context. next_command_buffers_ is valid only
    // when Dispatch() is called by the same context continuously.
    next_command_buffers_index_ = 0;
    next_command_buffers_ready_ = false;
    for (auto& buffers : next_command_buffers_) {
      buffers.clear();
    }
  }

  if (next_command_buffers_ready_) {
    // Use cached command buffers since it's ready to use.
    command_buffers.swap(next_command_buffers_[next_command_buffers_index_]);
  } else {
    // Prepare and use command buffers immediately since cached ones are not
    // ready to use yet.
    RETURN_IF_ERROR(PrepareCommandBuffers(command_buffers,
                                          /*submit_command_buffers=*/true));
  }

#ifndef __EMSCRIPTEN__
  // Schedule another next_command_buffers_thread_ to prepare command buffers
  // in parallel with the current Dispatch() call.
  next_command_buffers_thread_ = std::make_unique<std::thread>([this]() {
    if (auto s = PrepareCommandBuffers(
            next_command_buffers_[next_command_buffers_index_],
            /*submit_command_buffers=*/false);
        !s.ok()) {
      ABSL_LOG(ERROR) << "Failed to prepare next command buffers: " << s;
    } else {
      next_command_buffers_index_ =
          (next_command_buffers_index_ + 1) % next_command_buffers_.size();
      if (!next_command_buffers_ready_ && next_command_buffers_index_ == 0) {
        next_command_buffers_ready_ = true;
      }
    }
  });
#endif  // !__EMSCRIPTEN__

  return absl::OkStatus();
}

absl::Status GpuInferenceContextWebGpu::Dispatch() {
  // Waits for async upload notifications notified.
  for (auto& notification : async_upload_notifications_) {
    notification->WaitForNotification();
  }
  async_upload_notifications_.clear();

  std::vector<wgpu::CommandBuffer> command_buffers;
  RETURN_IF_ERROR(PrepareCommandBuffersFromCached(command_buffers));

  for (int i = 0; i < command_buffers.size(); ++i) {
    // Submit every buffer separately to avoid hangs
    backend_->wgpu_env().queue().Submit(1, &command_buffers[i]);
    ::ml_drift::webgpu::FlushIfCallbackSet();
  }
  return absl::OkStatus();
}

absl::StatusOr<GpuEventHandle>
GpuInferenceContextWebGpu::GetPreDispatchEvent() {
  return absl::UnimplementedError("GetPreDispatchEvent is not implemented.");
}

absl::StatusOr<GpuEventHandle> GpuInferenceContextWebGpu::GetPostDispatchEvent(
    bool is_async_execution_mode) {
  return absl::UnimplementedError("GetPostDispatchEvent is not implemented.");
}

absl::Status GpuInferenceContextWebGpu::WaitForEventsCompleted(
    absl::Span<GpuEventHandle> events, bool force_sync) {
  return absl::UnimplementedError("WaitForEventsCompleted is not implemented.");
}

absl::Status GpuInferenceContextWebGpu::PreConvert(bool input) {
  backend_->set_command_encoder(std::make_unique<wgpu::CommandEncoder>(
      backend_->wgpu_env().device().CreateCommandEncoder()));
  backend_->set_compute_pass_encoder(std::make_unique<wgpu::ComputePassEncoder>(
      backend_->command_encoder()->BeginComputePass()));
  return absl::OkStatus();
}

absl::Status GpuInferenceContextWebGpu::PostConvert(bool input) {
  backend_->compute_pass_encoder()->End();
  backend_->set_compute_pass_encoder(nullptr);

  // If output conversion is done, do PreRead on output buffers.
  if (!input) {
    for (auto* buffer : backend_->output_buffers()) {
      RETURN_IF_ERROR(buffer->PreRead(*backend_->command_encoder()));
    }
  }

  auto cb = backend_->command_encoder()->Finish();
  backend_->wgpu_env().queue().Submit(1, &cb);
  ::ml_drift::webgpu::FlushIfCallbackSet();
  backend_->set_command_encoder(nullptr);
  return absl::OkStatus();
}

absl::Status GpuInferenceContextWebGpu::Profile(
    ::ml_drift::ProfilingInfo& profiling_info) {
  return absl::UnimplementedError("Profile is not implemented.");
}

absl::StatusOr<size_t>
GpuInferenceContextWebGpu::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  return ctx_->GetSizeOfMemoryAllocatedForIntermediateTensors();
}

absl::StatusOr<size_t>
GpuInferenceContextWebGpu::GetSizeOfMemoryAllocatedForConstantTensors() const {
  return absl::UnimplementedError(
      "GetSizeOfMemoryAllocatedForConstantTensors is not implemented.");
}

absl::StatusOr<size_t>
GpuInferenceContextWebGpu::GetSizeOfMemoryAllocatedForExternalTensors() const {
  return absl::UnimplementedError(
      "GetSizeOfMemoryAllocatedForExternalTensors is not implemented.");
}

absl::Status GpuInferenceContextWebGpu::ReportMemoryBenchmarkIfEnabled(
    const ::ml_drift::CreateGpuModelInfo& create_info) {
#ifdef ML_DRIFT_MEM_STATS
  std::cout << ml_drift_delegate::GetMemoryBenchmarkReport(*ctx_, create_info)
            << std::endl;
#endif
  return absl::OkStatus();
}

absl::Status GpuInferenceContextWebGpu::SetCommandBufferHint(
    int num_nodes_per_command_encoder) {
  num_nodes_per_command_encoder_ = num_nodes_per_command_encoder;
  return absl::OkStatus();
}

GpuIOBufferWebGpu::GpuIOBufferWebGpu(
    GpuBackendWebGpu* backend, ::ml_drift::webgpu::Buffer&& buffer,
    std::unique_ptr<::ml_drift::webgpu::Buffer> mappable_buffer)
    : backend_(backend),
      buffer_(std::move(buffer)),
      mappable_buffer_(std::move(mappable_buffer)) {
  // Register this as an output buffer in the backend, so PostConvert() can do
  // PreRead on it when output conversion is done.
  if (mappable_buffer_ != nullptr) {
    backend_->output_buffers().insert(this);
  }
}

GpuIOBufferWebGpu::~GpuIOBufferWebGpu() {
  if (mappable_buffer_ != nullptr) {
    backend_->output_buffers().erase(this);
  }
}

absl::Status GpuIOBufferWebGpu::Read(absl::Span<uint8_t> data) {
  if (mappable_buffer_ == nullptr) {
    return absl::InvalidArgumentError("Buffer is not for read.");
  }
  return ::ml_drift::webgpu::ReadDataFromBuffer(
      backend_->wgpu_env().device(), backend_->wgpu_env().queue(),
      mappable_buffer_->GetMemoryHandle(), data.size(), data.data());
}

absl::Status GpuIOBufferWebGpu::Write(absl::Span<const uint8_t> data) {
  ::ml_drift::webgpu::WriteDataToBuffer(backend_->wgpu_env().queue(),
                                        buffer_.GetMemoryHandle(), data.size(),
                                        data.data());
  return absl::OkStatus();
}

absl::Status GpuIOBufferWebGpu::PreRead(wgpu::CommandEncoder& command_encoder) {
  if (mappable_buffer_ == nullptr) {
    return absl::InvalidArgumentError("Buffer is not for read.");
  }
  command_encoder.CopyBufferToBuffer(buffer_.GetMemoryHandle(), 0,
                                     mappable_buffer_->GetMemoryHandle(), 0,
                                     mappable_buffer_->GetMemorySizeInBytes());
  return absl::OkStatus();
}

Tensor2BufferConverterWebGpu::Tensor2BufferConverterWebGpu(
    GpuBackendWebGpu* backend,
    std::unique_ptr<::ml_drift::webgpu::TensorToBHWCBufferConverter> converter)
    : backend_(backend), converter_(std::move(converter)) {}

absl::Status Tensor2BufferConverterWebGpu::Convert(
    ::ml_drift::GpuSpatialTensor& src_tensor, GpuIOBuffer& dst_buffer) {
  if (backend_->compute_pass_encoder() == nullptr) {
    return absl::InvalidArgumentError("Compute pass encoder is not set.");
  }
  return converter_->Convert(
      backend_->wgpu_env().device(), *backend_->compute_pass_encoder(),
      static_cast<::ml_drift::webgpu::SpatialTensor*>(&src_tensor),
      &(static_cast<GpuIOBufferWebGpu&>(dst_buffer).wgpu_buffer()));
}

Buffer2TensorConverterWebGpu::Buffer2TensorConverterWebGpu(
    GpuBackendWebGpu* backend,
    std::unique_ptr<::ml_drift::webgpu::BHWCBufferToTensorConverter> converter)
    : backend_(backend), converter_(std::move(converter)) {}

absl::Status Buffer2TensorConverterWebGpu::Convert(
    GpuIOBuffer& src_buffer, ::ml_drift::GpuSpatialTensor& dst_tensor) {
  if (backend_->compute_pass_encoder() == nullptr) {
    return absl::InvalidArgumentError("Compute pass encoder is not set.");
  }
  return converter_->Convert(
      backend_->wgpu_env().device(), *backend_->compute_pass_encoder(),
      &(static_cast<GpuIOBufferWebGpu&>(src_buffer).wgpu_buffer()),
      static_cast<::ml_drift::webgpu::SpatialTensor*>(&dst_tensor));
}

}  // namespace litert::ml_drift
