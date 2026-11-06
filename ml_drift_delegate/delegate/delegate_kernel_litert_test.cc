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

#include "ml_drift_delegate/delegate/delegate_kernel_litert.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "weight_loader/external_weight_loader_litert.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

class DelegateKernelLiteRtTest;
DelegateKernelLiteRtTest* g_test_instance = nullptr;

class DelegateKernelLiteRtTest : public ::testing::Test {
 public:
  const TfLiteTensor* last_queried_tensor_ = nullptr;
  std::vector<const TfLiteTensor*> registered_requirement_tensors_;

 protected:
  void SetUp() override {
    g_test_instance = this;
    last_queried_tensor_ = nullptr;
    registered_requirement_tensors_.clear();
    tensors_.resize(6);
    for (size_t i = 0; i < tensors_.size(); ++i) {
      tensors_[i] = {};
      tensors_[i].type = kTfLiteFloat32;
      tensors_[i].allocation_type = kTfLiteArenaRw;
      tensors_[i].bytes = 16;
    }
    context_.tensors = tensors_.data();
    context_.tensors_size = tensors_.size();
  }

  void TearDown() override { g_test_instance = nullptr; }

  TfLiteContext context_ = {};
  std::vector<TfLiteTensor> tensors_;
};

TEST_F(DelegateKernelLiteRtTest,
       GetEffectiveTensorNoQuantizedTensorsReturnsOriginal) {
  DelegateKernelLiteRt kernel(nullptr);
  EXPECT_FALSE(kernel.HasQuantizedTensors());
  EXPECT_TRUE(kernel.quant_conversion_map().empty());

  for (int i = 0; i < context_.tensors_size; ++i) {
    EXPECT_EQ(kernel.GetEffectiveTensor(&context_, i), &context_.tensors[i]);
  }
}

TEST_F(DelegateKernelLiteRtTest,
       GetEffectiveTensorWithQuantizedMappingResolvesTensors) {
  DelegateKernelLiteRt kernel(nullptr);
  // Map synthetic temporary tensor IDs (4, 5) to signature tensor IDs (0, 1),
  // and vice-versa for bidirectional lookup.
  absl::flat_hash_map<int, int> quant_map = {
      {4, 0},
      {0, 4},
      {5, 1},
      {1, 5},
  };
  kernel.set_quant_conversion_map(std::move(quant_map));
  EXPECT_TRUE(kernel.HasQuantizedTensors());

  // Synthetic tensors 4 and 5 resolve to signature tensors 0 and 1.
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 4), &context_.tensors[0]);
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 5), &context_.tensors[1]);

  // Signature tensors 0 and 1 resolve to synthetic tensors 4 and 5.
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 0), &context_.tensors[4]);
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 1), &context_.tensors[5]);

  // Unmapped tensors (e.g. index 2, 3) fall back to returning themselves.
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 2), &context_.tensors[2]);
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 3), &context_.tensors[3]);
}

class FakeGpuInferenceContext : public GpuInferenceContext {
 public:
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> GetSpatialTensor(
      ::ml_drift::ValueId) override {
    return nullptr;
  }
  absl::Status BindSpatialTensor(::ml_drift::ValueId,
                                 ::ml_drift::GpuSpatialTensor*) override {
    return absl::OkStatus();
  }
  absl::Status UploadWeightsOnWeb(
      weight_loader::WeightLoader*, const ::ml_drift::GpuModel&,
      const absl::flat_hash_map<::ml_drift::ValueId, ::ml_drift::ValueId>&,
      const absl::flat_hash_map<::ml_drift::ValueId, uint32_t>&,
      std::vector<::ml_drift::WeightsManager::UploadWeightsInfo>&) override {
    return absl::OkStatus();
  }
  absl::Status WriteDataToWeightTensor(::ml_drift::ValueId,
                                       absl::Span<const uint8_t>) override {
    return absl::OkStatus();
  }
  absl::Status ReadWeightTensorToDescriptor(
      ::ml_drift::ValueId, ::ml_drift::TensorDescriptor&) override {
    return absl::OkStatus();
  }
  absl::Status Dispatch() override { return absl::OkStatus(); }
  absl::StatusOr<GpuEventHandle> GetPreDispatchEvent() override {
    return absl::NotFoundError("No pre-dispatch event");
  }
  absl::StatusOr<GpuEventHandle> GetPostDispatchEvent(
      bool is_async_execution_mode) override {
    if (is_async_execution_mode) {
      return GpuEventHandle(reinterpret_cast<void*>(0x1234));
    }
    return absl::NotFoundError("Sync mode has no post-dispatch event");
  }
  absl::Status WaitForEventsCompleted(absl::Span<GpuEventHandle>,
                                      bool) override {
    return absl::OkStatus();
  }
  absl::Status PreConvert(bool input) override { return absl::OkStatus(); }
  absl::Status PostConvert(bool input) override { return absl::OkStatus(); }
  absl::Status Profile(::ml_drift::ProfilingInfo& profiling_info) override {
    return absl::OkStatus();
  }
  absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForIntermediateTensors()
      const override {
    return 0;
  }
  absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForConstantTensors()
      const override {
    return 0;
  }
  absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForExternalTensors()
      const override {
    return 0;
  }
  absl::Status ReportMemoryBenchmarkIfEnabled(
      const ::ml_drift::CreateGpuModelInfo& create_info) override {
    return absl::OkStatus();
  }
  absl::Status SetCommandBufferHint(
      int num_nodes_per_command_encoder) override {
    return absl::OkStatus();
  }
};

class FakeGpuBackend : public GpuBackend {
 public:
  absl::string_view GetBackendName() override { return "FakeGpuBackend"; }
  absl::string_view GetSerializedDataPrefix() override { return ""; }
  absl::StatusOr<::ml_drift::GpuInfo> GetInfo() override {
    return ::ml_drift::GpuInfo();
  }
  absl::StatusOr<::ml_drift::TensorStorageType> GetFastestStorageType()
      override {
    return ::ml_drift::TensorStorageType::kBuffer;
  }
  absl::StatusOr<GpuMemoryHandle> GetGpuMemoryAllocated(
      const GpuTensorBufferPtr& tensor_buffer) override {
    return reinterpret_cast<GpuMemoryHandle>(0x5678);
  }
  absl::StatusOr<GpuEventHandle> GetGpuEventAssociated(
      const GpuTensorBufferPtr& tensor_buffer) override {
    return nullptr;
  }
  absl::Status AssociateGpuEvent(GpuEventHandle event, LiteRtEnvironment env,
                                 GpuTensorBufferPtr& tensor_buffer) override {
    return absl::OkStatus();
  }
  absl::Status WaitForCompletion() override { return absl::OkStatus(); }
  absl::StatusOr<GpuBufferRequirements> GetGpuBufferRequirements(
      ::ml_drift::TensorStorageType used_storage_type,
      ::ml_drift::DataType data_type) override {
    return GpuBufferRequirements{};
  }
  absl::StatusOr<GpuBufferRequirements>
  GetGpuBufferRequirementsForNonExternalTensors() override {
    return GpuBufferRequirements{};
  }
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> CreateInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
      bool may_share_memory_manager) override {
    return nullptr;
  }
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> RestoreInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      absl::Span<const uint8_t> serialized_model) override {
    return nullptr;
  }
  absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
  CreateSharedMemoryManager(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      std::unique_ptr<::ml_drift::GraphAdapter> graph_adapter,
      TfLiteContext* context, MlDriftDelegateData& delegate_data,
      ::ml_drift::SerializationWeightCache* serialization_cache) override {
    return nullptr;
  }
  absl::StatusOr<std::vector<
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
  GetBatchesForWeightsPreparation(::ml_drift::WeightsManager* weights_manager,
                                  size_t total_shared_tensor_size) override {
    return std::vector<
        std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>{};
  }
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatch(
      ::ml_drift::WeightsManager* weights_manager,
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
          op_infos) override {
    return absl::flat_hash_map<::ml_drift::ValueId,
                               std::unique_ptr<::ml_drift::GpuSpatialTensor>>{};
  }
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatches(::ml_drift::WeightsManager* weights_manager,
                          size_t total_shared_tensor_size) override {
    return absl::flat_hash_map<::ml_drift::ValueId,
                               std::unique_ptr<::ml_drift::GpuSpatialTensor>>{};
  }
  absl::StatusOr<std::unique_ptr<GpuTensorWrapper>> CreateTensorWrapper(
      const ::ml_drift::TensorDescriptor& desc,
      GpuMemoryHandle gpu_memory) override {
    return nullptr;
  }
  absl::Status ReadSpatialTensorToDescriptor(
      ::ml_drift::GpuSpatialTensor& tensor,
      ::ml_drift::TensorDescriptor& desc) override {
    return absl::OkStatus();
  }
  absl::Status UpdateSpatialTensor(
      ::ml_drift::GpuSpatialTensor* tensor,
      const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
      ReleaseDataCallback release_data_callback) override {
    return absl::OkStatus();
  }
  absl::Status ReleaseSpatialTensorMemory(
      ::ml_drift::GpuSpatialTensor* tensor) override {
    return absl::OkStatus();
  }
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBuffer(
      GpuMemoryHandle gpu_memory) override {
    return nullptr;
  }
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBufferWithSize(
      ::ml_drift::DataType data_type, size_t size, bool input) override {
    return nullptr;
  }
  absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
  CreateTensor2BufferConverter(
      const ::ml_drift::TensorDescriptor& src_desc,
      const ::ml_drift::BufferDescriptor& dst_desc) override {
    return nullptr;
  }
  absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
  CreateBuffer2TensorConverter(
      const ::ml_drift::BufferDescriptor& src_desc,
      const ::ml_drift::TensorDescriptor& dst_desc) override {
    return nullptr;
  }
  absl::StatusOr<uint64_t> GetSizeOfMemoryAllocatedForIntermediateTensors()
      const override {
    return 0;
  }
  absl::StatusOr<uint64_t> GetSizeOfMemoryAllocatedForConstantTensors()
      const override {
    return 0;
  }
};

class TestableDelegateKernelLiteRt : public DelegateKernelLiteRt {
 public:
  explicit TestableDelegateKernelLiteRt(
      const LiteRtRuntimeContext* runtime_context)
      : DelegateKernelLiteRt(runtime_context) {}

  void SetGpuInferenceContext(std::unique_ptr<GpuInferenceContext> ctx) {
    ctx_ = std::move(ctx);
  }

  void SetBufferContext(LiteRtExternalLiteRtBufferContext buffer_context) {
    buffer_context_ = buffer_context;
  }

  void SetBackend(GpuBackend* backend) { backend_ = backend; }

  void SetDelegateData(MlDriftDelegateData* delegate_data) {
    delegate_data_ = delegate_data;
  }

  void SetOutputIndices(std::vector<int64_t> output_indices) {
    output_indices_ = std::move(output_indices);
  }

  void SetInputIndices(std::vector<int64_t> input_indices) {
    input_indices_ = std::move(input_indices);
    input_ids_.resize(input_indices_.size());
  }

  void SetOutputIds(std::vector<::ml_drift::ValueId> ids) {
    output_ids_ = std::move(ids);
  }

  void SetExternalTensorIds(absl::flat_hash_set<::ml_drift::ValueId> ids) {
    external_tensor_ids_ = std::move(ids);
  }

  absl::Status CallUpdateCreateInfoWithExternalTensors(
      TfLiteContext* context, ::ml_drift::CreateGpuModelInfo& create_info,
      const std::vector<::ml_drift::TensorRef<::ml_drift::BHWC>>&
          input_tensor_refs = {},
      const std::vector<::ml_drift::TensorRef<::ml_drift::BHWC>>&
          output_tensor_refs = {},
      const absl::flat_hash_set<::ml_drift::ValueId>& preserved_fp32_values =
          {}) {
    return UpdateCreateInfoWithExternalTensors(
        context, create_info, input_tensor_refs, output_tensor_refs,
        preserved_fp32_values);
  }
};

TEST_F(DelegateKernelLiteRtTest,
       UploadIntermediateCpuTensorsResolvesEffectiveTensor) {
  TestableDelegateKernelLiteRt kernel(nullptr);
  kernel.SetInputIndices({5});
  kernel.set_quant_conversion_map({{5, 0}, {0, 5}});
  kernel.SetExternalTensorIds({0});

  // Tensors are arena allocated (not kTfLiteCustom), so after resolving the
  // effective tensor it skips upload and returns OK.
  EXPECT_TRUE(kernel.UploadIntermediateCpuTensorsToGpuMemory(&context_).ok());
}

TEST_F(DelegateKernelLiteRtTest,
       DownloadGpuMemoryToIntermediateCpuTensorsResolvesEffectiveTensor) {
  TestableDelegateKernelLiteRt kernel(nullptr);
  kernel.SetOutputIndices({4});
  kernel.SetOutputIds({1});
  kernel.set_quant_conversion_map({{4, 1}, {1, 4}});
  kernel.SetExternalTensorIds({1});

  // Tensors are arena allocated (not kTfLiteCustom), so after resolving the
  // effective tensor it skips download and returns OK.
  EXPECT_TRUE(kernel.DownloadGpuMemoryToIntermediateCpuTensors(&context_).ok());
}

TEST_F(DelegateKernelLiteRtTest,
       HandleOutputEventsResolvesEffectiveTensorInAsyncMode) {
  LiteRtRuntimeContext runtime_context = {};
  runtime_context.external_litert_buffer_context_get_environment =
      [](LiteRtExternalLiteRtBufferContext,
         LiteRtEnvironment* env) -> LiteRtStatus {
    *env = reinterpret_cast<LiteRtEnvironment>(0x5678);
    return kLiteRtStatusOk;
  };
  runtime_context.get_external_litert_buffer_context_tensor_buffer =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor*,
         LiteRtTensorBuffer*) -> LiteRtStatus {
    return kLiteRtStatusErrorNotFound;
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetGpuInferenceContext(std::make_unique<FakeGpuInferenceContext>());
  kernel.SetBufferContext(
      reinterpret_cast<LiteRtExternalLiteRtBufferContext>(0x9abc));
  kernel.SetOutputIndices({4});
  kernel.set_quant_conversion_map({{4, 1}, {1, 4}});

  // In async mode, HandleOutputEvents resolves the effective tensor for output
  // index 4 (mapping to signature tensor 1) and queries the buffer context.
  // Since the tensor buffer is not registered, it returns an error verifying
  // that effective tensor resolution and lookup occurred.
  auto status =
      kernel.HandleOutputEvents(&context_, /*is_async_execution_mode=*/true);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("TensorBuffer is not registered for output#0 tensor T#4."));
}

TEST_F(DelegateKernelLiteRtTest, HandleInputEventsResolvesEffectiveTensor) {
  LiteRtRuntimeContext runtime_context = {};
  runtime_context.get_external_litert_buffer_context_tensor_buffer =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor*,
         LiteRtTensorBuffer*) -> LiteRtStatus {
    return kLiteRtStatusErrorNotFound;
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetBufferContext(
      reinterpret_cast<LiteRtExternalLiteRtBufferContext>(0x9abc));
  kernel.SetInputIndices({5});
  kernel.set_quant_conversion_map({{5, 0}, {0, 5}});

  auto status = kernel.HandleInputEvents(&context_);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("TensorBuffer is not registered for input#0 tensor T#5."));
}

TEST_F(DelegateKernelLiteRtTest,
       InitTensorConvertersResolvesEffectiveInputTensor) {
  MlDriftDelegateData delegate_data;
  delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
  delegate_data.options->litert_external_tensors_mode = false;
  FakeGpuBackend backend;

  TestableDelegateKernelLiteRt kernel(nullptr);
  kernel.SetBackend(&backend);
  kernel.SetDelegateData(&delegate_data);
  kernel.SetGpuInferenceContext(std::make_unique<FakeGpuInferenceContext>());
  kernel.SetInputIndices({5});
  kernel.set_quant_conversion_map({{5, 0}, {0, 5}});

  tensors_[5].type = kTfLiteFloat32;
  tensors_[0].type = kTfLiteNoType;

  auto status = kernel.InitTensorConverters(&context_);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("model input type is not supported."));
}

TEST_F(DelegateKernelLiteRtTest,
       InitTensorConvertersResolvesEffectiveOutputTensor) {
  MlDriftDelegateData delegate_data;
  delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
  delegate_data.options->litert_external_tensors_mode = false;
  FakeGpuBackend backend;

  TestableDelegateKernelLiteRt kernel(nullptr);
  kernel.SetBackend(&backend);
  kernel.SetDelegateData(&delegate_data);
  kernel.SetGpuInferenceContext(std::make_unique<FakeGpuInferenceContext>());
  kernel.SetInputIndices({});
  kernel.SetOutputIndices({4});
  kernel.SetOutputIds({1});
  kernel.set_quant_conversion_map({{4, 1}, {1, 4}});

  tensors_[4].type = kTfLiteFloat32;
  tensors_[1].type = kTfLiteNoType;

  auto status = kernel.InitTensorConverters(&context_);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("model output type is not supported."));
}

TEST_F(DelegateKernelLiteRtTest,
       BindExternalTensorBuffersResolvesEffectiveInputTensor) {
  MlDriftDelegateData delegate_data;
  delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
  delegate_data.options->litert_external_tensors_mode = false;
  FakeGpuBackend backend;

  LiteRtRuntimeContext runtime_context = {};
  runtime_context.get_external_litert_buffer_context_tensor_buffer =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor* tensor,
         LiteRtTensorBuffer* buffer) -> LiteRtStatus {
    if (g_test_instance) {
      g_test_instance->last_queried_tensor_ = tensor;
    }
    *buffer = reinterpret_cast<LiteRtTensorBuffer>(0x1234);
    return kLiteRtStatusOk;
  };
  runtime_context.destroy_tensor_buffer = [](LiteRtTensorBuffer) {};
  runtime_context.get_tensor_buffer_type =
      [](LiteRtTensorBuffer,
         LiteRtTensorBufferType* buffer_type) -> LiteRtStatus {
    *buffer_type = kLiteRtTensorBufferTypeHostMemory;
    return kLiteRtStatusOk;
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetBackend(&backend);
  kernel.SetDelegateData(&delegate_data);
  kernel.SetBufferContext(
      reinterpret_cast<LiteRtExternalLiteRtBufferContext>(0x9abc));
  kernel.SetInputIndices({5});
  kernel.set_quant_conversion_map({{5, 0}, {0, 5}});

  EXPECT_TRUE(kernel.BindExternalTensorBuffers(&context_).ok());
  EXPECT_EQ(last_queried_tensor_, &context_.tensors[0]);
}

TEST_F(DelegateKernelLiteRtTest,
       BindExternalTensorBuffersResolvesEffectiveOutputTensor) {
  MlDriftDelegateData delegate_data;
  delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
  delegate_data.options->litert_external_tensors_mode = false;
  FakeGpuBackend backend;

  LiteRtRuntimeContext runtime_context = {};
  runtime_context.get_external_litert_buffer_context_tensor_buffer =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor* tensor,
         LiteRtTensorBuffer* buffer) -> LiteRtStatus {
    if (g_test_instance) {
      g_test_instance->last_queried_tensor_ = tensor;
    }
    *buffer = reinterpret_cast<LiteRtTensorBuffer>(0x1234);
    return kLiteRtStatusOk;
  };
  runtime_context.destroy_tensor_buffer = [](LiteRtTensorBuffer) {};
  runtime_context.get_tensor_buffer_type =
      [](LiteRtTensorBuffer,
         LiteRtTensorBufferType* buffer_type) -> LiteRtStatus {
    *buffer_type = kLiteRtTensorBufferTypeHostMemory;
    return kLiteRtStatusOk;
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetBackend(&backend);
  kernel.SetDelegateData(&delegate_data);
  kernel.SetBufferContext(
      reinterpret_cast<LiteRtExternalLiteRtBufferContext>(0x9abc));
  kernel.SetInputIndices({});
  kernel.SetOutputIndices({4});
  kernel.SetOutputIds({1});
  kernel.set_quant_conversion_map({{4, 1}, {1, 4}});

  EXPECT_TRUE(kernel.BindExternalTensorBuffers(&context_).ok());
  EXPECT_EQ(last_queried_tensor_, &context_.tensors[1]);
}

TEST_F(DelegateKernelLiteRtTest,
       ConvertNonExternalInputTensorsResolvesEffectiveTensor) {
  MlDriftDelegateData delegate_data;
  delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
  delegate_data.options->litert_external_tensors_mode = false;
  FakeGpuBackend backend;

  LiteRtRuntimeContext runtime_context = {};
  runtime_context.get_external_litert_buffer_context_tensor_buffer =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor* tensor,
         LiteRtTensorBuffer*) -> LiteRtStatus {
    if (g_test_instance) {
      g_test_instance->last_queried_tensor_ = tensor;
    }
    return kLiteRtStatusErrorNotFound;
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetBackend(&backend);
  kernel.SetDelegateData(&delegate_data);
  kernel.SetGpuInferenceContext(std::make_unique<FakeGpuInferenceContext>());
  kernel.SetBufferContext(
      reinterpret_cast<LiteRtExternalLiteRtBufferContext>(0x9abc));
  kernel.SetInputIndices({5});
  kernel.set_quant_conversion_map({{5, 0}, {0, 5}});

  auto status = kernel.ConvertNonExternalInputTensorsToGpuMemory(&context_);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Input#0 doesn't have a tensor buffer."));
  EXPECT_EQ(last_queried_tensor_, &context_.tensors[0]);
}

TEST_F(DelegateKernelLiteRtTest,
       ConvertGpuMemoryToNonExternalOutputTensorsResolvesEffectiveTensor) {
  MlDriftDelegateData delegate_data;
  delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
  delegate_data.options->litert_external_tensors_mode = false;
  FakeGpuBackend backend;

  LiteRtRuntimeContext runtime_context = {};
  runtime_context.get_external_litert_buffer_context_tensor_buffer =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor* tensor,
         LiteRtTensorBuffer*) -> LiteRtStatus {
    if (g_test_instance) {
      g_test_instance->last_queried_tensor_ = tensor;
    }
    return kLiteRtStatusErrorNotFound;
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetBackend(&backend);
  kernel.SetDelegateData(&delegate_data);
  kernel.SetGpuInferenceContext(std::make_unique<FakeGpuInferenceContext>());
  kernel.SetBufferContext(
      reinterpret_cast<LiteRtExternalLiteRtBufferContext>(0x9abc));
  kernel.SetOutputIndices({4});
  kernel.SetOutputIds({1});
  kernel.set_quant_conversion_map({{4, 1}, {1, 4}});

  auto status = kernel.ConvertGpuMemoryToNonExternalOutputTensors(&context_);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Output#0 doesn't have a tensor buffer."));
  EXPECT_EQ(last_queried_tensor_, &context_.tensors[1]);
}

TEST_F(DelegateKernelLiteRtTest,
       UpdateCreateInfoRegistersRequirementsForQuantizedTensors) {
  MlDriftDelegateData delegate_data;
  delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
  delegate_data.options->litert_external_tensors_mode = false;
  FakeGpuBackend backend;

  LiteRtRuntimeContext runtime_context = {};
  runtime_context.create_tensor_buffer_requirements =
      [](int, const LiteRtTensorBufferType*, size_t, int, const uint32_t*,
         LiteRtTensorBufferRequirements* reqs) -> LiteRtStatus {
    *reqs = reinterpret_cast<LiteRtTensorBufferRequirements>(0x1234);
    return kLiteRtStatusOk;
  };
  runtime_context.external_litert_buffer_context_register_buffer_requirements =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor* tensor,
         LiteRtTensorBufferRequirements) -> LiteRtStatus {
    if (g_test_instance) {
      g_test_instance->registered_requirement_tensors_.push_back(tensor);
    }
    return kLiteRtStatusOk;
  };

  context_.GetExternalContext =
      [](TfLiteContext*, TfLiteExternalContextType) -> TfLiteExternalContext* {
    return reinterpret_cast<TfLiteExternalContext*>(0x9abc);
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetBackend(&backend);
  kernel.SetDelegateData(&delegate_data);
  kernel.SetInputIndices({5});
  kernel.SetOutputIndices({4});
  kernel.SetOutputIds({1});
  kernel.set_quant_conversion_map({{5, 0}, {0, 5}, {4, 1}, {1, 4}});

  ::ml_drift::CreateGpuModelInfo create_info;
  auto status =
      kernel.CallUpdateCreateInfoWithExternalTensors(&context_, create_info);
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_THAT(registered_requirement_tensors_,
              ElementsAre(&context_.tensors[5], &context_.tensors[0],
                          &context_.tensors[4], &context_.tensors[1]));
}

TEST_F(DelegateKernelLiteRtTest,
       UpdateCreateInfoWithExternalTensorsNoQuantizedTensorsRegistersOnce) {
  MlDriftDelegateData delegate_data;
  delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
  delegate_data.options->litert_external_tensors_mode = false;
  FakeGpuBackend backend;

  LiteRtRuntimeContext runtime_context = {};
  runtime_context.create_tensor_buffer_requirements =
      [](int, const LiteRtTensorBufferType*, size_t, int, const uint32_t*,
         LiteRtTensorBufferRequirements* reqs) -> LiteRtStatus {
    *reqs = reinterpret_cast<LiteRtTensorBufferRequirements>(0x1234);
    return kLiteRtStatusOk;
  };
  runtime_context.external_litert_buffer_context_register_buffer_requirements =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor* tensor,
         LiteRtTensorBufferRequirements) -> LiteRtStatus {
    if (g_test_instance) {
      g_test_instance->registered_requirement_tensors_.push_back(tensor);
    }
    return kLiteRtStatusOk;
  };

  context_.GetExternalContext =
      [](TfLiteContext*, TfLiteExternalContextType) -> TfLiteExternalContext* {
    return reinterpret_cast<TfLiteExternalContext*>(0x9abc);
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetBackend(&backend);
  kernel.SetDelegateData(&delegate_data);
  kernel.SetInputIndices({5});
  kernel.SetOutputIndices({4});
  kernel.SetOutputIds({1});

  ::ml_drift::CreateGpuModelInfo create_info;
  auto status =
      kernel.CallUpdateCreateInfoWithExternalTensors(&context_, create_info);
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_THAT(registered_requirement_tensors_,
              ElementsAre(&context_.tensors[5], &context_.tensors[4]));
}

}  // namespace
}  // namespace litert::ml_drift
