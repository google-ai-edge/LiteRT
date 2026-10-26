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

#include "ml_drift_delegate/delegate/host_buffer_arbitrator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
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
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

class FakeGpuIOBuffer : public GpuIOBuffer {
 public:
  FakeGpuIOBuffer(size_t size, bool is_host_ptr, bool owns_host_ptr = false,
                  void* host_ptr = nullptr)
      : size_(size),
        is_host_ptr_(is_host_ptr),
        owns_host_ptr_(owns_host_ptr),
        host_ptr_(host_ptr) {}

  absl::Status Read(absl::Span<uint8_t> data) override {
    return absl::OkStatus();
  }
  absl::Status Write(absl::Span<const uint8_t> data) override {
    return absl::OkStatus();
  }
  bool IsHostPtr() const override { return is_host_ptr_; }
  void* GetHostPtr() const override { return host_ptr_; }
  bool OwnsHostPtr() const override { return owns_host_ptr_; }
  size_t GetSize() const override { return size_; }

 private:
  size_t size_ = 0;
  bool is_host_ptr_ = false;
  bool owns_host_ptr_ = false;
  void* host_ptr_ = nullptr;
};

class FakeGpuBackend : public GpuBackend {
 public:
  absl::string_view GetBackendName() override { return "FakeBackend"; }
  absl::string_view GetSerializedDataPrefix() override { return "FAKE"; }
  absl::StatusOr<::ml_drift::GpuInfo> GetInfo() override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<::ml_drift::TensorStorageType> GetFastestStorageType()
      override {
    return ::ml_drift::TensorStorageType::kBuffer;
  }
  absl::StatusOr<GpuMemoryHandle> GetGpuMemoryAllocated(
      const GpuTensorBufferPtr& tensor_buffer) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<GpuEventHandle> GetGpuEventAssociated(
      const GpuTensorBufferPtr& tensor_buffer) override {
    return absl::UnimplementedError("Unused");
  }
  absl::Status AssociateGpuEvent(GpuEventHandle event, LiteRtEnvironment env,
                                 GpuTensorBufferPtr& tensor_buffer) override {
    return absl::UnimplementedError("Unused");
  }
  absl::Status WaitForCompletion() override { return absl::OkStatus(); }
  absl::StatusOr<GpuBufferRequirements> GetGpuBufferRequirements(
      ::ml_drift::TensorStorageType used_storage_type,
      ::ml_drift::DataType data_type) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<GpuBufferRequirements>
  GetGpuBufferRequirementsForNonExternalTensors() override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> CreateInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
      bool may_share_memory_manager) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> RestoreInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      absl::Span<const uint8_t> serialized_model) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
  CreateSharedMemoryManager(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      std::unique_ptr<::ml_drift::GraphAdapter> graph_adapter,
      TfLiteContext* context, MlDriftDelegateData& delegate_data,
      ::ml_drift::SerializationWeightCache* serialization_cache) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<std::vector<
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
  GetBatchesForWeightsPreparation(::ml_drift::WeightsManager* weights_manager,
                                  size_t total_shared_tensor_size) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatch(
      ::ml_drift::WeightsManager* weights_manager,
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
          op_infos) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatches(::ml_drift::WeightsManager* weights_manager,
                          size_t total_shared_tensor_size) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<std::unique_ptr<GpuTensorWrapper>> CreateTensorWrapper(
      const ::ml_drift::TensorDescriptor& desc,
      GpuMemoryHandle gpu_memory) override {
    return absl::UnimplementedError("Unused");
  }
  absl::Status ReadSpatialTensorToDescriptor(
      ::ml_drift::GpuSpatialTensor& tensor,
      ::ml_drift::TensorDescriptor& desc) override {
    return absl::UnimplementedError("Unused");
  }
  absl::Status UpdateSpatialTensor(
      ::ml_drift::GpuSpatialTensor* tensor,
      const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
      ReleaseDataCallback release_data_callback) override {
    return absl::UnimplementedError("Unused");
  }
  absl::Status ReleaseSpatialTensorMemory(
      ::ml_drift::GpuSpatialTensor* tensor) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBuffer(
      GpuMemoryHandle gpu_memory) override {
    return absl::UnimplementedError("Unused");
  }

  absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
  CreateTensor2BufferConverter(
      const ::ml_drift::TensorDescriptor& src_desc,
      const ::ml_drift::BufferDescriptor& dst_desc) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
  CreateBuffer2TensorConverter(
      const ::ml_drift::BufferDescriptor& src_desc,
      const ::ml_drift::TensorDescriptor& dst_desc) override {
    return absl::UnimplementedError("Unused");
  }
  absl::StatusOr<uint64_t> GetSizeOfMemoryAllocatedForIntermediateTensors()
      const override {
    return 0;
  }
  absl::StatusOr<uint64_t> GetSizeOfMemoryAllocatedForConstantTensors()
      const override {
    return 0;
  }

  // Host pointer related overrides.
  bool SupportsHostPtrIOBuffer(bool input) const override {
    if (force_host_ptr_) {
      return true;
    }
    return input ? supports_host_ptr_input_ : supports_host_ptr_output_;
  }
  bool SupportsZeroCopyHostPtr(bool input) const override {
    return SupportsHostPtrIOBuffer(input);
  }
  size_t GetHostPtrAlignment() const override { return alignment_; }
  void SetForceHostPtrIOBuffer(bool force) override {
    force_host_ptr_ = force;
    force_host_ptr_called_ = true;
  }

  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBufferWithSize(
      ::ml_drift::DataType data_type, size_t size, bool input) override {
    create_io_buffer_size_called_ = true;
    last_io_buffer_size_ = size;
    return std::make_unique<FakeGpuIOBuffer>(size, /*is_host_ptr=*/false);
  }

  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateInternalHostBuffer(
      ::ml_drift::DataType data_type, size_t size, bool input) override {
    create_internal_host_buffer_called_ = true;
    if (fail_internal_host_buffer_) {
      return absl::InternalError("Simulated internal host buffer failure.");
    }
    if (return_null_internal_host_buffer_) {
      return nullptr;
    }
    return std::make_unique<FakeGpuIOBuffer>(size, /*is_host_ptr=*/true,
                                             /*owns_host_ptr=*/true);
  }

  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> WrapExternalHostBuffer(
      ::ml_drift::DataType data_type, size_t size, bool input,
      void* host_ptr) override {
    wrap_external_host_buffer_called_ = true;
    if (fail_wrap_external_host_buffer_) {
      return absl::InternalError("Simulated wrap external buffer failure.");
    }
    if (return_null_wrap_external_host_buffer_) {
      return nullptr;
    }
    return std::make_unique<FakeGpuIOBuffer>(size, /*is_host_ptr=*/true,
                                             /*owns_host_ptr=*/false, host_ptr);
  }

  bool supports_host_ptr_input_ = true;
  bool supports_host_ptr_output_ = true;
  bool force_host_ptr_ = false;
  bool force_host_ptr_called_ = false;
  size_t alignment_ = 4096;

  bool fail_internal_host_buffer_ = false;
  bool fail_wrap_external_host_buffer_ = false;
  bool return_null_internal_host_buffer_ = false;
  bool return_null_wrap_external_host_buffer_ = false;
  bool create_io_buffer_size_called_ = false;
  bool create_internal_host_buffer_called_ = false;
  bool wrap_external_host_buffer_called_ = false;
  size_t last_io_buffer_size_ = 0;
};

TEST(HostBufferDescriptorTest, DefaultConstruction) {
  HostBufferDescriptor desc;
  EXPECT_EQ(desc.host_ptr, nullptr);
  EXPECT_EQ(desc.bytes, 0);
  EXPECT_FALSE(desc.is_custom_allocation);
  EXPECT_EQ(desc.data_type, ::ml_drift::DataType::kUnknown);
}

TEST(HostBufferDescriptorTest, ParametrizedConstruction) {
  void* dummy_ptr = reinterpret_cast<void*>(0x1234);
  HostBufferDescriptor desc(dummy_ptr, 1024, /*is_custom_allocation=*/true,
                            ::ml_drift::DataType::kFloat32);
  EXPECT_EQ(desc.host_ptr, dummy_ptr);
  EXPECT_EQ(desc.bytes, 1024);
  EXPECT_TRUE(desc.is_custom_allocation);
  EXPECT_EQ(desc.data_type, ::ml_drift::DataType::kFloat32);
}

TEST(HostBufferDescriptorTest, FromTfLiteTensorCustomAllocation) {
  TfLiteTensor tensor{};
  tensor.data.raw = reinterpret_cast<char*>(0x4000);
  tensor.bytes = 512;
  tensor.allocation_type = kTfLiteCustom;

  auto desc = HostBufferDescriptor::FromTfLiteTensor(
      tensor, ::ml_drift::DataType::kInt32);
  EXPECT_EQ(desc.host_ptr, reinterpret_cast<void*>(0x4000));
  EXPECT_EQ(desc.bytes, 512);
  EXPECT_TRUE(desc.is_custom_allocation);
  EXPECT_EQ(desc.data_type, ::ml_drift::DataType::kInt32);
}

TEST(HostBufferDescriptorTest, FromTfLiteTensorArenaAllocation) {
  TfLiteTensor tensor{};
  tensor.data.raw = reinterpret_cast<char*>(0x4000);
  tensor.bytes = 256;
  tensor.allocation_type = kTfLiteArenaRw;

  auto desc = HostBufferDescriptor::FromTfLiteTensor(tensor);
  EXPECT_EQ(desc.host_ptr, reinterpret_cast<void*>(0x4000));
  EXPECT_EQ(desc.bytes, 256);
  EXPECT_FALSE(desc.is_custom_allocation);
  EXPECT_EQ(desc.data_type, ::ml_drift::DataType::kUnknown);
}

TEST(HostBufferArbitratorTest, CanZeroCopyValidAlignment) {
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x2000), 1024,
                            /*is_custom_allocation=*/true);
  EXPECT_TRUE(HostBufferArbitrator::CanZeroCopy(desc, /*alignment=*/4096));
}

TEST(HostBufferArbitratorTest, CanZeroCopyUnalignedPointer) {
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x2040), 1024,
                            /*is_custom_allocation=*/true);
  EXPECT_FALSE(HostBufferArbitrator::CanZeroCopy(desc, /*alignment=*/4096));
}

TEST(HostBufferArbitratorTest, CanZeroCopyZeroAlignmentReturnsFalse) {
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x2000), 1024,
                            /*is_custom_allocation=*/true);
  EXPECT_FALSE(HostBufferArbitrator::CanZeroCopy(desc, /*alignment=*/0));
}

TEST(HostBufferArbitratorTest, CanZeroCopyNullPointerReturnsFalse) {
  HostBufferDescriptor desc(nullptr, 1024, /*is_custom_allocation=*/true);
  EXPECT_FALSE(HostBufferArbitrator::CanZeroCopy(desc, /*alignment=*/4096));
}

TEST(HostBufferArbitratorTest, CanZeroCopyNonCustomAllocationReturnsFalse) {
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x2000), 1024,
                            /*is_custom_allocation=*/false);
  EXPECT_FALSE(HostBufferArbitrator::CanZeroCopy(desc, /*alignment=*/4096));
}

TEST(HostBufferArbitratorTest, CanZeroCopyViaTfLiteTensor) {
  TfLiteTensor tensor{};
  tensor.data.raw = reinterpret_cast<char*>(0x3000);
  tensor.allocation_type = kTfLiteCustom;
  EXPECT_TRUE(HostBufferArbitrator::CanZeroCopy(tensor, /*alignment=*/4096));

  tensor.allocation_type = kTfLiteArenaRw;
  EXPECT_FALSE(HostBufferArbitrator::CanZeroCopy(tensor, /*alignment=*/4096));
}

TEST(HostBufferArbitratorTest, IsHostPtrSupportedNullBackend) {
  HostBufferArbitrator arbitrator(/*backend=*/nullptr);
  EXPECT_FALSE(arbitrator.IsHostPtrSupported(/*is_input=*/true));
  EXPECT_FALSE(arbitrator.IsHostPtrSupported(/*is_input=*/false));
}

TEST(HostBufferArbitratorTest, IsHostPtrSupportedDisabledInOptions) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = false;
  HostBufferArbitrator arbitrator(&backend, options);

  EXPECT_FALSE(arbitrator.IsHostPtrSupported(/*is_input=*/true));
  EXPECT_FALSE(arbitrator.IsHostPtrSupported(/*is_input=*/false));
}

TEST(HostBufferArbitratorTest, IsHostPtrSupportedEnabledAndSupported) {
  FakeGpuBackend backend;
  backend.supports_host_ptr_input_ = true;
  backend.supports_host_ptr_output_ = false;

  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  EXPECT_TRUE(arbitrator.IsHostPtrSupported(/*is_input=*/true));
  EXPECT_FALSE(arbitrator.IsHostPtrSupported(/*is_input=*/false));
}

TEST(HostBufferArbitratorTest, IsHostPtrSupportedForceOption) {
  FakeGpuBackend backend;
  backend.supports_host_ptr_input_ = false;

  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = false;
  options.force_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  // Force flag should be applied to backend during construction.
  EXPECT_TRUE(backend.force_host_ptr_called_);
  EXPECT_TRUE(arbitrator.IsHostPtrSupported(/*is_input=*/true));

  // Reset spy flag and verify IsHostPtrSupported is side-effect-free.
  backend.force_host_ptr_called_ = false;
  EXPECT_TRUE(arbitrator.IsHostPtrSupported(/*is_input=*/true));
  EXPECT_FALSE(backend.force_host_ptr_called_);
}

TEST(HostBufferArbitratorTest, ApplyBackendOptionsOnSetBackendAndSetOptions) {
  FakeGpuBackend backend;
  backend.supports_host_ptr_input_ = false;

  HostBufferArbitrator arbitrator;
  EXPECT_FALSE(backend.force_host_ptr_called_);

  HostBufferArbitrator::Options options;
  options.force_host_ptr_io_buffers = true;
  arbitrator.SetOptions(options);

  arbitrator.SetBackend(&backend);
  EXPECT_TRUE(backend.force_host_ptr_called_);
  EXPECT_TRUE(arbitrator.IsHostPtrSupported(/*is_input=*/true));

  backend.force_host_ptr_called_ = false;
  arbitrator.SetOptions(options);
  EXPECT_TRUE(backend.force_host_ptr_called_);
}

TEST(HostBufferArbitratorTest, NeedsRecreateBufferNullBufferReturnsTrue) {
  FakeGpuBackend backend;
  HostBufferArbitrator arbitrator(&backend);
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x1000), 1024,
                            /*is_custom_allocation=*/true);

  EXPECT_TRUE(arbitrator.NeedsRecreateBuffer(
      /*buffer=*/nullptr, desc, /*is_input=*/true, /*last_seen_ptr=*/nullptr));
}

TEST(HostBufferArbitratorTest, NeedsRecreateBufferDynamicResizeReturnsFalse) {
  FakeGpuBackend backend;
  HostBufferArbitrator arbitrator(&backend);
  FakeGpuIOBuffer buffer(512, /*is_host_ptr=*/false);
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x1000), 1024,
                            /*is_custom_allocation=*/false);

  EXPECT_FALSE(arbitrator.NeedsRecreateBuffer(&buffer, desc, /*is_input=*/true,
                                              /*last_seen_ptr=*/nullptr));
}

TEST(HostBufferArbitratorTest, NeedsRecreateBufferZeroBytesDeviceBuffer) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  FakeGpuIOBuffer device_buffer(0, /*is_host_ptr=*/false);
  HostBufferDescriptor desc(nullptr, 0, /*is_custom_allocation=*/false);

  EXPECT_FALSE(arbitrator.NeedsRecreateBuffer(&device_buffer, desc,
                                              /*is_input=*/true, nullptr));
}

TEST(HostBufferArbitratorTest, NeedsRecreateBufferZeroBytesHostPtrBuffer) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  FakeGpuIOBuffer host_buffer(0, /*is_host_ptr=*/true);
  HostBufferDescriptor desc(nullptr, 0, /*is_custom_allocation=*/false);

  // Downgrade host buffer to device buffer when size is 0.
  EXPECT_TRUE(arbitrator.NeedsRecreateBuffer(&host_buffer, desc,
                                             /*is_input=*/true, nullptr));
}

TEST(HostBufferArbitratorTest, NeedsRecreateBufferHostPtrDisabledRevertsHost) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = false;
  HostBufferArbitrator arbitrator(&backend, options);

  FakeGpuIOBuffer host_buffer(64, /*is_host_ptr=*/true);
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x1000), 64,
                            /*is_custom_allocation=*/false);

  EXPECT_TRUE(arbitrator.NeedsRecreateBuffer(&host_buffer, desc,
                                             /*is_input=*/true, nullptr));

  FakeGpuIOBuffer device_buffer(64, /*is_host_ptr=*/false);
  EXPECT_FALSE(arbitrator.NeedsRecreateBuffer(&device_buffer, desc,
                                              /*is_input=*/true, nullptr));
}

TEST(HostBufferArbitratorTest, NeedsRecreateBufferUpgradeDeviceToHost) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  FakeGpuIOBuffer device_buffer(64, /*is_host_ptr=*/false);
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x1000), 64,
                            /*is_custom_allocation=*/false);

  EXPECT_TRUE(arbitrator.NeedsRecreateBuffer(&device_buffer, desc,
                                             /*is_input=*/true, nullptr));
}

TEST(HostBufferArbitratorTest, NeedsRecreateBufferSamePointerZeroCopyReused) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* ptr = reinterpret_cast<void*>(0x2000);
  FakeGpuIOBuffer zero_copy_buffer(64, /*is_host_ptr=*/true,
                                   /*owns_host_ptr=*/false, ptr);
  HostBufferDescriptor desc(ptr, 64, /*is_custom_allocation=*/true);

  EXPECT_FALSE(arbitrator.NeedsRecreateBuffer(&zero_copy_buffer, desc,
                                              /*is_input=*/true, ptr));
}

TEST(HostBufferArbitratorTest,
     NeedsRecreateBufferSamePointerZeroCopyReusedNullLastSeenPtr) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* ptr = reinterpret_cast<void*>(0x2000);
  FakeGpuIOBuffer zero_copy_buffer(64, /*is_host_ptr=*/true,
                                   /*owns_host_ptr=*/false, ptr);
  HostBufferDescriptor desc(ptr, 64, /*is_custom_allocation=*/true);

  // Verifies that when a zero-copy buffer already wraps the incoming host
  // pointer, NeedsRecreateBuffer returns false even if last_seen_ptr is null
  // (the state immediately following buffer recreation), preventing
  // per-invoke recreation thrashing.
  EXPECT_FALSE(arbitrator.NeedsRecreateBuffer(&zero_copy_buffer, desc,
                                              /*is_input=*/true,
                                              /*last_seen_ptr=*/nullptr));
}

TEST(HostBufferArbitratorTest,
     NeedsRecreateBufferNonZeroCopyOwnedFallbackReused) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* internal_ptr = reinterpret_cast<void*>(0x5000);
  FakeGpuIOBuffer owned_buffer(64, /*is_host_ptr=*/true,
                               /*owns_host_ptr=*/true, internal_ptr);
  // Unaligned custom pointer does not qualify for zero-copy.
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x2040), 64,
                            /*is_custom_allocation=*/true);

  EXPECT_FALSE(arbitrator.NeedsRecreateBuffer(&owned_buffer, desc,
                                              /*is_input=*/true, nullptr));
}

TEST(HostBufferArbitratorTest,
     NeedsRecreateBufferUnchangedPointerSuppressesRecreation) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* ptr = reinterpret_cast<void*>(0x2000);
  FakeGpuIOBuffer zero_copy_buffer(64, /*is_host_ptr=*/true,
                                   /*owns_host_ptr=*/false, ptr);
  HostBufferDescriptor desc(ptr, 64, /*is_custom_allocation=*/true);

  EXPECT_FALSE(arbitrator.NeedsRecreateBuffer(&zero_copy_buffer, desc,
                                              /*is_input=*/true,
                                              /*last_seen_ptr=*/ptr));
}

TEST(HostBufferArbitratorTest,
     NeedsRecreateBufferNewZeroCopyPointerTriggersRecreate) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* ptr_a = reinterpret_cast<void*>(0x2000);
  void* ptr_b = reinterpret_cast<void*>(0x3000);
  FakeGpuIOBuffer zero_copy_buffer(64, /*is_host_ptr=*/true,
                                   /*owns_host_ptr=*/false, ptr_a);
  HostBufferDescriptor desc(ptr_b, 64, /*is_custom_allocation=*/true);

  EXPECT_TRUE(arbitrator.NeedsRecreateBuffer(&zero_copy_buffer, desc,
                                             /*is_input=*/true,
                                             /*last_seen_ptr=*/ptr_a));
}

TEST(HostBufferArbitratorTest,
     NeedsRecreateBufferTransitionFromZeroCopyToFallback) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* ptr_a = reinterpret_cast<void*>(0x2000);
  void* unaligned_ptr = reinterpret_cast<void*>(0x3001);
  FakeGpuIOBuffer zero_copy_buffer(64, /*is_host_ptr=*/true,
                                   /*owns_host_ptr=*/false, ptr_a);
  HostBufferDescriptor desc(unaligned_ptr, 64, /*is_custom_allocation=*/true);

  EXPECT_TRUE(arbitrator.NeedsRecreateBuffer(&zero_copy_buffer, desc,
                                             /*is_input=*/true,
                                             /*last_seen_ptr=*/ptr_a));
}

TEST(HostBufferArbitratorTest, CreateOrWrapBufferNullBackendReturnsError) {
  HostBufferArbitrator arbitrator(/*backend=*/nullptr);
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x1000), 1024,
                            /*is_custom_allocation=*/true,
                            ::ml_drift::DataType::kFloat32);

  auto result = arbitrator.CreateOrWrapBuffer(desc, /*is_input=*/true);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInternal));
}

TEST(HostBufferArbitratorTest, CreateOrWrapBufferZeroBytesDeviceBuffer) {
  FakeGpuBackend backend;
  HostBufferArbitrator arbitrator(&backend);
  HostBufferDescriptor desc(nullptr, 0, /*is_custom_allocation=*/false,
                            ::ml_drift::DataType::kFloat32);

  auto result = arbitrator.CreateOrWrapBuffer(desc, /*is_input=*/true);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(backend.create_io_buffer_size_called_);
  EXPECT_EQ(backend.last_io_buffer_size_, 0);
  EXPECT_FALSE((*result)->IsHostPtr());
}

TEST(HostBufferArbitratorTest, CreateOrWrapBufferZeroCopySuccess) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* page_aligned_ptr = reinterpret_cast<void*>(0x4000);
  HostBufferDescriptor desc(page_aligned_ptr, 1024,
                            /*is_custom_allocation=*/true,
                            ::ml_drift::DataType::kFloat32);

  auto result = arbitrator.CreateOrWrapBuffer(desc, /*is_input=*/true);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(backend.wrap_external_host_buffer_called_);
  EXPECT_FALSE((*result)->OwnsHostPtr());
  EXPECT_EQ((*result)->GetHostPtr(), page_aligned_ptr);
}

TEST(HostBufferArbitratorTest, CreateOrWrapBufferWrapFailsFallbackToInternal) {
  FakeGpuBackend backend;
  backend.fail_wrap_external_host_buffer_ = true;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* page_aligned_ptr = reinterpret_cast<void*>(0x4000);
  HostBufferDescriptor desc(page_aligned_ptr, 1024,
                            /*is_custom_allocation=*/true,
                            ::ml_drift::DataType::kFloat32);

  auto result = arbitrator.CreateOrWrapBuffer(desc, /*is_input=*/true);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(backend.wrap_external_host_buffer_called_);
  EXPECT_TRUE(backend.create_internal_host_buffer_called_);
  EXPECT_TRUE((*result)->OwnsHostPtr());
}

TEST(HostBufferArbitratorTest,
     CreateOrWrapBufferInternalFailsFallbackToDevice) {
  FakeGpuBackend backend;
  backend.fail_internal_host_buffer_ = true;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  HostBufferDescriptor desc(reinterpret_cast<void*>(0x4040), 1024,
                            /*is_custom_allocation=*/false,
                            ::ml_drift::DataType::kFloat32);

  auto result = arbitrator.CreateOrWrapBuffer(desc, /*is_input=*/true);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(backend.create_internal_host_buffer_called_);
  EXPECT_TRUE(backend.create_io_buffer_size_called_);
  EXPECT_FALSE((*result)->IsHostPtr());
}

TEST(HostBufferArbitratorTest, CreateOrWrapBufferHostPtrDisabledReturnsDevice) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = false;
  HostBufferArbitrator arbitrator(&backend, options);

  void* page_aligned_ptr = reinterpret_cast<void*>(0x4000);
  HostBufferDescriptor desc(page_aligned_ptr, 1024,
                            /*is_custom_allocation=*/true,
                            ::ml_drift::DataType::kFloat32);

  auto result = arbitrator.CreateOrWrapBuffer(desc, /*is_input=*/true);
  ASSERT_THAT(result, IsOk());
  EXPECT_FALSE(backend.wrap_external_host_buffer_called_);
  EXPECT_FALSE(backend.create_internal_host_buffer_called_);
  EXPECT_TRUE(backend.create_io_buffer_size_called_);
  EXPECT_FALSE((*result)->IsHostPtr());
}

TEST(HostBufferArbitratorTest, CreateOrWrapBufferViaTfLiteTensor) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* page_aligned_ptr = reinterpret_cast<void*>(0x4000);
  TfLiteTensor tensor{};
  tensor.data.raw = static_cast<char*>(page_aligned_ptr);
  tensor.bytes = 1024;
  tensor.allocation_type = kTfLiteCustom;

  auto result = arbitrator.CreateOrWrapBuffer(
      tensor, ::ml_drift::DataType::kFloat32, /*is_input=*/true);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(backend.wrap_external_host_buffer_called_);
  EXPECT_FALSE((*result)->OwnsHostPtr());
  EXPECT_EQ((*result)->GetHostPtr(), page_aligned_ptr);
}

TEST(HostBufferArbitratorTest,
     NeedsRecreateBufferZeroSizeBufferExpandedBytesDevice) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = false;
  HostBufferArbitrator arbitrator(&backend, options);

  FakeGpuIOBuffer zero_size_buffer(0, /*is_host_ptr=*/false);
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x1000), 1024,
                            /*is_custom_allocation=*/false);

  EXPECT_TRUE(arbitrator.NeedsRecreateBuffer(&zero_size_buffer, desc,
                                             /*is_input=*/true, nullptr));
}

TEST(HostBufferArbitratorTest,
     NeedsRecreateBufferZeroSizeBufferExpandedBytesHost) {
  FakeGpuBackend backend;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  FakeGpuIOBuffer zero_size_buffer(0, /*is_host_ptr=*/true,
                                   /*owns_host_ptr=*/true);
  HostBufferDescriptor desc(reinterpret_cast<void*>(0x1000), 1024,
                            /*is_custom_allocation=*/false);

  EXPECT_TRUE(arbitrator.NeedsRecreateBuffer(&zero_size_buffer, desc,
                                             /*is_input=*/true, nullptr));
}

TEST(HostBufferArbitratorTest,
     CreateOrWrapBufferWrapReturnsNullFallbackToInternal) {
  FakeGpuBackend backend;
  backend.return_null_wrap_external_host_buffer_ = true;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  void* page_aligned_ptr = reinterpret_cast<void*>(0x4000);
  HostBufferDescriptor desc(page_aligned_ptr, 1024,
                            /*is_custom_allocation=*/true,
                            ::ml_drift::DataType::kFloat32);

  auto result = arbitrator.CreateOrWrapBuffer(desc, /*is_input=*/true);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(backend.wrap_external_host_buffer_called_);
  EXPECT_TRUE(backend.create_internal_host_buffer_called_);
  EXPECT_TRUE((*result)->OwnsHostPtr());
}

TEST(HostBufferArbitratorTest,
     CreateOrWrapBufferInternalReturnsNullFallbackToDevice) {
  FakeGpuBackend backend;
  backend.return_null_internal_host_buffer_ = true;
  HostBufferArbitrator::Options options;
  options.enable_host_ptr_io_buffers = true;
  HostBufferArbitrator arbitrator(&backend, options);

  HostBufferDescriptor desc(reinterpret_cast<void*>(0x4040), 1024,
                            /*is_custom_allocation=*/false,
                            ::ml_drift::DataType::kFloat32);

  auto result = arbitrator.CreateOrWrapBuffer(desc, /*is_input=*/true);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(backend.create_internal_host_buffer_called_);
  EXPECT_TRUE(backend.create_io_buffer_size_called_);
  EXPECT_FALSE((*result)->IsHostPtr());
}

TEST(HostBufferArbitratorTest, NeedsRecreateBufferViaTfLiteTensor) {
  FakeGpuBackend backend;
  HostBufferArbitrator arbitrator(&backend);
  TfLiteTensor tensor{};
  tensor.bytes = 1024;
  tensor.data.raw = reinterpret_cast<char*>(0x4000);
  tensor.allocation_type = kTfLiteCustom;

  EXPECT_TRUE(arbitrator.NeedsRecreateBuffer(
      /*buffer=*/nullptr, tensor, /*is_input=*/true,
      /*last_seen_ptr=*/nullptr));
}

}  // namespace
}  // namespace litert::ml_drift
