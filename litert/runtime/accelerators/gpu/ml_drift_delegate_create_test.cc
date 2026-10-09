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

#include "litert/runtime/accelerators/gpu/ml_drift_delegate_create.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/internal/litert_accelerator_registration.h"
#include "litert/c/internal/litert_delegate_wrapper.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/delegate_types.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

namespace {

void DtorHelper(void*) {}

}  // namespace

extern "C" void LiteRtDeleteMockGpuDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  delete delegate;
}

litert::TfLiteDelegatePtr CreateMockGpuDelegate(
    litert::ml_drift::MlDriftDelegateOptionsPtr options,
    LiteRtEnvironment litert_env) {
  litert::TfLiteDelegatePtr delegate(new TfLiteDelegate(TfLiteDelegateCreate()),
                                     LiteRtDeleteMockGpuDelegate);
  return delegate;
}

TEST(MlDriftDelegateCreateTest,
     CreateDelegateNoDelegateOptionsNoGpuOptionsPayload) {
  LiteRtAccelerator accelerator;

  ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);

  LiteRtOptions compilation_options = nullptr;
  ASSERT_EQ(LiteRtCreateOptions(&compilation_options), kLiteRtStatusOk);
  // Has opaque_options but no gpu options.
  int dummy = 0;
  LiteRtOpaqueOptions opaque_options = nullptr;
  ASSERT_EQ(
      LiteRtCreateOpaqueOptions("my key", &dummy, DtorHelper, &opaque_options),
      kLiteRtStatusOk);
  ASSERT_EQ(LiteRtAddOpaqueOptions(compilation_options, opaque_options),
            kLiteRtStatusOk);
  litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};
  LiteRtRuntimeContext* runtime_context = LrtGetRuntimeContext();
  ASSERT_EQ(litert::ml_drift::CreateDelegate(
                runtime_context, nullptr, accelerator,
                litert::ml_drift::GetGpuOptionsPayload(runtime_context,
                                                       compilation_options),
                nullptr, CreateMockGpuDelegate, delegate_ptr),
            kLiteRtStatusOk);
  LiteRtDestroyOptions(compilation_options);
  LiteRtDestroyAccelerator(accelerator);
}

TEST(MlDriftDelegateCreateTest, CreateDelegateNoGpuOptionsPayload) {
  LiteRtAccelerator accelerator;

  ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);

  LiteRtOptions compilation_options = nullptr;
  ASSERT_EQ(LiteRtCreateOptions(&compilation_options), kLiteRtStatusOk);
  // Has opaque_options but no gpu options.
  int dummy = 0;
  LiteRtOpaqueOptions opaque_options = nullptr;
  ASSERT_EQ(
      LiteRtCreateOpaqueOptions("my key", &dummy, DtorHelper, &opaque_options),
      kLiteRtStatusOk);
  ASSERT_EQ(LiteRtAddOpaqueOptions(compilation_options, opaque_options),
            kLiteRtStatusOk);

  auto gpu_delegate_options = std::make_unique<MlDriftDelegateOptions>();
  litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};

  LiteRtRuntimeContext* runtime_context = LrtGetRuntimeContext();
  ASSERT_EQ(
      litert::ml_drift::CreateDelegate(
          runtime_context, nullptr, accelerator,
          litert::ml_drift::GetGpuOptionsPayload(runtime_context,
                                                 compilation_options),
          std::move(gpu_delegate_options), CreateMockGpuDelegate, delegate_ptr),
      kLiteRtStatusOk);

  LiteRtDestroyOptions(compilation_options);
  LiteRtDestroyAccelerator(accelerator);
}

TEST(MlDriftDelegateCreateTest, CreateDelegateNoDelegateOptionsNoPayload) {
  LiteRtAccelerator accelerator;

  ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);

  litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};
  LiteRtRuntimeContext* runtime_context = LrtGetRuntimeContext();
  ASSERT_EQ(litert::ml_drift::CreateDelegate(
                runtime_context, nullptr, accelerator, nullptr, nullptr,
                CreateMockGpuDelegate, delegate_ptr),
            kLiteRtStatusOk);

  LiteRtDestroyAccelerator(accelerator);
}

TEST(MlDriftDelegateCreateTest, CreateDelegateNoPayload) {
  LiteRtAccelerator accelerator;

  ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);
  auto gpu_delegate_options = std::make_unique<MlDriftDelegateOptions>();
  litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};
  LiteRtRuntimeContext* runtime_context = LrtGetRuntimeContext();
  ASSERT_EQ(
      litert::ml_drift::CreateDelegate(runtime_context, nullptr, accelerator,
                                       nullptr, std::move(gpu_delegate_options),
                                       CreateMockGpuDelegate, delegate_ptr),
      kLiteRtStatusOk);

  LiteRtDestroyAccelerator(accelerator);
}

namespace litert::ml_drift {
namespace {

class DummyGpuBackend : public GpuBackend {
 public:
  absl::string_view GetBackendName() override { return "dummy"; }
  absl::string_view GetSerializedDataPrefix() override { return "dummy"; }
  absl::StatusOr<::ml_drift::GpuInfo> GetInfo() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<::ml_drift::TensorStorageType> GetFastestStorageType()
      override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<GpuMemoryHandle> GetGpuMemoryAllocated(
      const GpuTensorBufferPtr& tensor_buffer) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<GpuEventHandle> GetGpuEventAssociated(
      const GpuTensorBufferPtr& tensor_buffer) override {
    return absl::UnimplementedError("");
  }
  absl::Status AssociateGpuEvent(GpuEventHandle event, LiteRtEnvironment env,
                                 GpuTensorBufferPtr& tensor_buffer) override {
    return absl::UnimplementedError("");
  }
  absl::Status WaitForCompletion() override { return absl::OkStatus(); }
  absl::StatusOr<GpuBufferRequirements> GetGpuBufferRequirements(
      ::ml_drift::TensorStorageType used_storage_type,
      ::ml_drift::DataType data_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<GpuBufferRequirements>
  GetGpuBufferRequirementsForNonExternalTensors() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> CreateInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
      bool may_share_memory_manager) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> RestoreInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      absl::Span<const uint8_t> serialized_model) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::unique_ptr<
      ::ml_drift::SharedMemoryManager>>  // NOLINT(misc-include-cleaner)
  CreateSharedMemoryManager(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      std::unique_ptr<::ml_drift::GraphAdapter> graph_adapter,
      TfLiteContext* context, MlDriftDelegateData& delegate_data,
      // NOLINTNEXTLINE(misc-include-cleaner)
      ::ml_drift::SerializationWeightCache* serialization_cache) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::vector<
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
  GetBatchesForWeightsPreparation(::ml_drift::WeightsManager* weights_manager,
                                  size_t total_shared_tensor_size) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatch(
      ::ml_drift::WeightsManager* weights_manager,
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
          op_infos) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatches(::ml_drift::WeightsManager* weights_manager,
                          size_t total_shared_tensor_size) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::unique_ptr<GpuTensorWrapper>> CreateTensorWrapper(
      const ::ml_drift::TensorDescriptor& desc,
      GpuMemoryHandle gpu_memory) override {
    return absl::UnimplementedError("");
  }
  absl::Status ReadSpatialTensorToDescriptor(
      ::ml_drift::GpuSpatialTensor& tensor,
      ::ml_drift::TensorDescriptor& desc) override {
    return absl::UnimplementedError("");
  }
  absl::Status UpdateSpatialTensor(
      ::ml_drift::GpuSpatialTensor* tensor,
      const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
      // NOLINTNEXTLINE(misc-include-cleaner)
      ReleaseDataCallback release_data_callback) override {
    return absl::UnimplementedError("");
  }
  absl::Status ReleaseSpatialTensorMemory(
      ::ml_drift::GpuSpatialTensor* tensor) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBuffer(
      GpuMemoryHandle gpu_memory) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBufferWithSize(
      ::ml_drift::DataType data_type, size_t size, bool input) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
  CreateTensor2BufferConverter(
      const ::ml_drift::TensorDescriptor& src_desc,
      const ::ml_drift::BufferDescriptor& dst_desc) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
  CreateBuffer2TensorConverter(
      const ::ml_drift::BufferDescriptor& src_desc,
      const ::ml_drift::TensorDescriptor& dst_desc) override {
    return absl::UnimplementedError("");
  }

  absl::StatusOr<uint64_t>
  GetSizeOfMemoryAllocatedForIntermediateTensors() const override {
    return 1024 * 512;
  }
  absl::StatusOr<uint64_t>
  GetSizeOfMemoryAllocatedForConstantTensors() const override {
    return 1024 * 256;
  }
};

}  // namespace
}  // namespace litert::ml_drift

TEST(MlDriftDelegateCreateTest, CollectMetricsSuccess) {
  LiteRtRuntimeContext* runtime_context = LrtGetRuntimeContext();
  auto delegate_data =
      std::make_unique<litert::ml_drift::MlDriftDelegateData>();
  delegate_data->backend =
      std::make_unique<litert::ml_drift::DummyGpuBackend>();

  auto* delegate = new TfLiteDelegate(TfLiteDelegateCreate());
  delegate->data_ = delegate_data.get();

  LiteRtDelegateWrapper delegate_wrapper = nullptr;
  auto deleter = [](TfLiteOpaqueDelegate* d) {
    delete reinterpret_cast<TfLiteDelegate*>(d);
  };
  ASSERT_EQ(runtime_context->wrap_delegate(
                reinterpret_cast<TfLiteOpaqueDelegate*>(delegate), deleter,
                &delegate_wrapper),
            kLiteRtStatusOk);

  ASSERT_EQ(litert::ml_drift::StartMetricsCollection(runtime_context,
                                                     delegate_wrapper, 0),
            kLiteRtStatusOk);

  LiteRtMetrics metrics = nullptr;
  ASSERT_EQ(LiteRtCreateMetrics(&metrics), kLiteRtStatusOk);
  ASSERT_EQ(litert::ml_drift::StopMetricsCollection(runtime_context,
                                                    delegate_wrapper, metrics),
            kLiteRtStatusOk);

  int num_metrics = 0;
  ASSERT_EQ(LiteRtGetNumMetrics(metrics, &num_metrics), kLiteRtStatusOk);
  ASSERT_EQ(num_metrics, 2);

  LiteRtMetric metric0;
  ASSERT_EQ(LiteRtGetMetric(metrics, 0, &metric0), kLiteRtStatusOk);
  EXPECT_STREQ(metric0.name, "gpu_intermediate_memory_bytes");
  EXPECT_EQ(metric0.value.type, kLiteRtAnyTypeInt);
  EXPECT_EQ(metric0.value.int_value, 1024 * 512);

  LiteRtMetric metric1;
  ASSERT_EQ(LiteRtGetMetric(metrics, 1, &metric1), kLiteRtStatusOk);
  EXPECT_STREQ(metric1.name, "gpu_constant_memory_bytes");
  EXPECT_EQ(metric1.value.type, kLiteRtAnyTypeInt);
  EXPECT_EQ(metric1.value.int_value, 1024 * 256);

  LiteRtDestroyMetrics(metrics);
  LiteRtDestroyDelegateWrapper(delegate_wrapper);
}
