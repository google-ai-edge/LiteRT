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
#include "ml_drift_delegate/delegate/delegate_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "weight_loader/external_weight_loader_litert.h"
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

namespace {

class FakeWeightLoader : public weight_loader::WeightLoader {
 public:
  explicit FakeWeightLoader(std::vector<weight_loader::WeightInfo> infos)
      : infos_(std::move(infos)) {}

  absl::Span<const weight_loader::WeightInfo> GetWeightInfo() const override {
    return infos_;
  }
  LiteRtStatus PrepareAccess(const weight_loader::WeightAccessRequest&,
                             LiteRtEnvironmentT*) override {
    return kLiteRtStatusOk;
  }
  LiteRtStatus PrepareAccessForBuffer(uint32_t,
                                      const weight_loader::WeightAccessRequest&,
                                      LiteRtEnvironmentT*) override {
    return kLiteRtStatusOk;
  }
  const weight_loader::WeightInfo* FindWeightInfoByBuffer(
      uint32_t external_buffer_id) const override {
    for (const auto& info : infos_) {
      if (info.external_buffer_id == external_buffer_id) {
        return &info;
      }
    }
    return nullptr;
  }
  uint32_t GetCanonicalExternalBufferId(
      uint32_t external_buffer_id) const override {
    return external_buffer_id;
  }
  LiteRtStatus SetExternalWeightByBuffer(uint32_t,
                                         weight_loader::WeightAccess) override {
    return kLiteRtStatusOk;
  }
  const weight_loader::WeightAccess* GetExternalWeightByBuffer(
      uint32_t) const override {
    return nullptr;
  }
  LiteRtStatus DiscardExternalWeightByBuffer(uint32_t) override {
    return kLiteRtStatusOk;
  }
  LiteRtStatus ReleaseExternalWeightByBuffer(uint32_t) override {
    return kLiteRtStatusOk;
  }

 private:
  std::vector<weight_loader::WeightInfo> infos_;
};

TfLiteExternalContext* GetFakeLiteRtExternalContext(
    struct TfLiteContext* context, TfLiteExternalContextType type) {
  static TfLiteExternalContext fake_buffer_context{};
  if (type == kTfLiteLiteRtBufferContext) {
    return &fake_buffer_context;
  }
  return nullptr;
}

}  // namespace

TEST(MlDriftDelegateCreateTest,
     GetTensorBufferIdentifiersInLiteRtContextDoesNotDereferenceImpl) {
  litert::ml_drift::MlDriftDelegateData delegate_data;
  delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
  FakeWeightLoader fake_loader({
      weight_loader::WeightInfo{
          .external_buffer_id = 77,
          .subgraph_index = 0,
          .tensor_index = 1,
          .packing = "",
      },
  });
  delegate_data.weight_loader = &fake_loader;

  const uint8_t weight_a[4] = {1, 2, 3, 4};
  const uint8_t weight_b[4] = {5, 6, 7, 8};
  TfLiteTensor tensors[4] = {};
  // Tensor 0: inline constant weight_a.
  tensors[0].allocation_type = kTfLiteMmapRo;
  tensors[0].data.raw_const = reinterpret_cast<const char*>(weight_a);
  // Tensor 1: external weight (should be in external buffer IDs, not inline).
  tensors[1].allocation_type = kTfLiteMmapRo;
  tensors[1].data.raw_const = reinterpret_cast<const char*>(weight_b);
  // Tensor 2: inline constant sharing weight_a pointer.
  tensors[2].allocation_type = kTfLiteMmapRo;
  tensors[2].data.raw_const = reinterpret_cast<const char*>(weight_a);
  // Tensor 3: dynamic activation tensor.
  tensors[3].allocation_type = kTfLiteArenaRw;
  tensors[3].data.raw_const = nullptr;

  TfLiteContext context{};
  // Poison impl_ so any attempt to cast and dereference tflite::Subgraph*
  // across a mismatched C++ standard library ABI immediately crashes.
  context.impl_ = reinterpret_cast<void*>(static_cast<uintptr_t>(0x100000000));
  context.tensors = tensors;
  context.tensors_size = 4;
  context.GetExternalContext = GetFakeLiteRtExternalContext;

  const auto& ext_ids = litert::ml_drift::GetExternalTensorBufferIdentifiers(
      &context, delegate_data);
  ASSERT_EQ(ext_ids.size(), 1);
  EXPECT_EQ(ext_ids.at(1), 77);

  const auto& inline_ids =
      litert::ml_drift::GetTensorBufferIdentifiers(&context, delegate_data);
  ASSERT_EQ(inline_ids.size(), 2);
  EXPECT_EQ(inline_ids.at(0), 1);
  EXPECT_EQ(inline_ids.at(2), 1);
  EXPECT_EQ(inline_ids.count(1), 0);
  EXPECT_EQ(inline_ids.count(3), 0);

  // Verify reference stability across insertions of many new contexts (forcing
  // outer map rehashes).
  const auto* ext_ids_ptr = &ext_ids;
  const auto* inline_ids_ptr = &inline_ids;
  TfLiteContext extra_contexts[32] = {};
  for (auto& extra_ctx : extra_contexts) {
    extra_ctx.GetExternalContext = GetFakeLiteRtExternalContext;
    (void)litert::ml_drift::GetExternalTensorBufferIdentifiers(&extra_ctx,
                                                               delegate_data);
    (void)litert::ml_drift::GetTensorBufferIdentifiers(&extra_ctx,
                                                       delegate_data);
  }
  EXPECT_EQ(&litert::ml_drift::GetExternalTensorBufferIdentifiers(
                &context, delegate_data),
            ext_ids_ptr);
  EXPECT_EQ(
      &litert::ml_drift::GetTensorBufferIdentifiers(&context, delegate_data),
      inline_ids_ptr);
  EXPECT_EQ(ext_ids.at(1), 77);
  EXPECT_EQ(inline_ids.at(0), 1);
}
