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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
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
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_kernel_litert.h"
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

using ::absl_testing::IsOk;

class FakeGpuSpatialTensor : public ::ml_drift::GpuSpatialTensor {
 public:
  int Width() const override { return 1; }
  int Height() const override { return 1; }
  int Depth() const override { return 1; }
  int Channels() const override { return 4; }
  int Slices() const override { return 1; }
  int Batch() const override { return 1; }
  ::ml_drift::TensorDescriptor GetDescriptor() const override {
    return ::ml_drift::TensorDescriptor(::ml_drift::DataType::kFloat32,
                                        ::ml_drift::TensorStorageType::kBuffer,
                                        ::ml_drift::Layout::kBHWC);
  }
};

class FakeGpuIOBuffer : public GpuIOBuffer {
 public:
  absl::Status Read(absl::Span<uint8_t>) override { return absl::OkStatus(); }
  absl::Status Write(absl::Span<const uint8_t>) override {
    return absl::OkStatus();
  }
  size_t GetSize() const override { return 16; }
};

class CountingBuffer2TensorConverter : public Buffer2TensorConverter {
 public:
  explicit CountingBuffer2TensorConverter(int* convert_count)
      : convert_count_(convert_count) {}
  absl::Status Convert(GpuIOBuffer&, ::ml_drift::GpuSpatialTensor&) override {
    ++(*convert_count_);
    return absl::OkStatus();
  }

 private:
  int* convert_count_;
};

class CountingTensor2BufferConverter : public Tensor2BufferConverter {
 public:
  explicit CountingTensor2BufferConverter(int* convert_count)
      : convert_count_(convert_count) {}
  absl::Status Convert(::ml_drift::GpuSpatialTensor&, GpuIOBuffer&) override {
    ++(*convert_count_);
    return absl::OkStatus();
  }

 private:
  int* convert_count_;
};

class CountingPackBatch : public GpuPackBatch {
 public:
  explicit CountingPackBatch(int* dispatch_count)
      : dispatch_count_(dispatch_count) {}
  absl::Status Dispatch() override {
    ++(*dispatch_count_);
    return absl::OkStatus();
  }

 private:
  int* dispatch_count_;
};

class CountingUnpackBatch : public GpuUnpackBatch {
 public:
  explicit CountingUnpackBatch(int* dispatch_count)
      : dispatch_count_(dispatch_count) {}
  absl::Status Dispatch() override {
    ++(*dispatch_count_);
    return absl::OkStatus();
  }

 private:
  int* dispatch_count_;
};

class FakeGpuInferenceContext : public GpuInferenceContext {
 public:
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> GetSpatialTensor(
      ::ml_drift::ValueId id) override {
    auto& tensor = tensors_[id];
    if (!tensor) {
      tensor = std::make_unique<FakeGpuSpatialTensor>();
    }
    return tensor.get();
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
    return absl::NotFoundError("no event");
  }
  absl::StatusOr<GpuEventHandle> GetPostDispatchEvent(bool) override {
    return absl::NotFoundError("no event");
  }
  absl::Status WaitForEventsCompleted(absl::Span<GpuEventHandle>,
                                      bool) override {
    return absl::OkStatus();
  }
  absl::Status PreConvert(bool) override { return absl::OkStatus(); }
  absl::Status PostConvert(bool) override { return absl::OkStatus(); }
  absl::Status Profile(::ml_drift::ProfilingInfo&) override {
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
      const ::ml_drift::CreateGpuModelInfo&) override {
    return absl::OkStatus();
  }
  absl::Status SetCommandBufferHint(int) override { return absl::OkStatus(); }

 private:
  absl::flat_hash_map<::ml_drift::ValueId,
                      std::unique_ptr<FakeGpuSpatialTensor>>
      tensors_;
};

class FakeGpuBackend : public GpuBackend {
 public:
  bool supports_command_buffer_ = true;
  bool force_convert_command_buffer_ = false;
  bool return_null_pack_batch_ = false;
  bool return_error_unpack_batch_ = false;
  bool fail_create_io_buffer_ = false;
  int create_pack_batch_calls_ = 0;
  int create_unpack_batch_calls_ = 0;
  size_t last_pack_batch_size_ = 0;
  size_t last_unpack_batch_size_ = 0;
  int pack_dispatch_calls_ = 0;
  int unpack_dispatch_calls_ = 0;
  int input_convert_calls_ = 0;
  int output_convert_calls_ = 0;

  absl::string_view GetBackendName() override { return "Fake"; }
  absl::string_view GetSerializedDataPrefix() override { return "fake"; }
  absl::StatusOr<::ml_drift::GpuInfo> GetInfo() override {
    return ::ml_drift::GpuInfo{};
  }
  absl::StatusOr<::ml_drift::TensorStorageType> GetFastestStorageType()
      override {
    return ::ml_drift::TensorStorageType::kBuffer;
  }
  absl::StatusOr<GpuMemoryHandle> GetGpuMemoryAllocated(
      const GpuTensorBufferPtr& tensor_buffer) override {
    return reinterpret_cast<GpuMemoryHandle>(tensor_buffer.get());
  }
  absl::StatusOr<GpuEventHandle> GetGpuEventAssociated(
      const GpuTensorBufferPtr&) override {
    return absl::NotFoundError("no event");
  }
  absl::Status AssociateGpuEvent(GpuEventHandle, LiteRtEnvironment,
                                 GpuTensorBufferPtr&) override {
    return absl::OkStatus();
  }
  absl::Status WaitForCompletion() override { return absl::OkStatus(); }
  absl::StatusOr<GpuBufferRequirements> GetGpuBufferRequirements(
      ::ml_drift::TensorStorageType, ::ml_drift::DataType) override {
    return GpuBufferRequirements{};
  }
  absl::StatusOr<GpuBufferRequirements>
  GetGpuBufferRequirementsForNonExternalTensors() override {
    return GpuBufferRequirements{};
  }
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> CreateInferenceContext(
      const ::ml_drift::CreateGpuModelInfo&, ::ml_drift::GpuModel&,
      std::vector<uint8_t>*, bool) override {
    return absl::UnimplementedError("unused");
  }
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> RestoreInferenceContext(
      const ::ml_drift::CreateGpuModelInfo&,
      absl::Span<const uint8_t>) override {
    return absl::UnimplementedError("unused");
  }
  absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
  CreateSharedMemoryManager(const ::ml_drift::CreateGpuModelInfo&,
                            std::unique_ptr<::ml_drift::GraphAdapter>,
                            TfLiteContext*, MlDriftDelegateData&,
                            ::ml_drift::SerializationWeightCache*) override {
    return absl::UnimplementedError("unused");
  }
  absl::StatusOr<std::vector<
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
  GetBatchesForWeightsPreparation(::ml_drift::WeightsManager*,
                                  size_t) override {
    return absl::UnimplementedError("unused");
  }
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatch(
      ::ml_drift::WeightsManager*,
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&)
      override {
    return absl::UnimplementedError("unused");
  }
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatches(::ml_drift::WeightsManager*, size_t) override {
    return absl::UnimplementedError("unused");
  }
  absl::StatusOr<std::unique_ptr<GpuTensorWrapper>> CreateTensorWrapper(
      const ::ml_drift::TensorDescriptor&, GpuMemoryHandle) override {
    return absl::UnimplementedError("unused");
  }
  absl::Status ReadSpatialTensorToDescriptor(
      ::ml_drift::GpuSpatialTensor&, ::ml_drift::TensorDescriptor&) override {
    return absl::UnimplementedError("unused");
  }
  absl::Status ReadSpatialTensorData(::ml_drift::GpuSpatialTensor&,
                                     absl::Span<uint8_t>) override {
    return absl::UnimplementedError("unused");
  }
  absl::Status UpdateSpatialTensor(::ml_drift::GpuSpatialTensor*,
                                   const ::ml_drift::TensorDescriptor&, size_t,
                                   ReleaseDataCallback) override {
    return absl::UnimplementedError("unused");
  }
  absl::Status ReleaseSpatialTensorMemory(
      ::ml_drift::GpuSpatialTensor*) override {
    return absl::OkStatus();
  }
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBuffer(
      GpuMemoryHandle) override {
    if (fail_create_io_buffer_) {
      return absl::InternalError("simulated CreateIOBuffer failure");
    }
    return std::make_unique<FakeGpuIOBuffer>();
  }
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBufferWithSize(
      ::ml_drift::DataType, size_t, bool) override {
    return std::make_unique<FakeGpuIOBuffer>();
  }
  absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
  CreateTensor2BufferConverter(const ::ml_drift::TensorDescriptor&,
                               const ::ml_drift::BufferDescriptor&) override {
    return std::make_unique<CountingTensor2BufferConverter>(
        &output_convert_calls_);
  }
  absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
  CreateBuffer2TensorConverter(const ::ml_drift::BufferDescriptor&,
                               const ::ml_drift::TensorDescriptor&) override {
    return std::make_unique<CountingBuffer2TensorConverter>(
        &input_convert_calls_);
  }
  absl::StatusOr<uint64_t> GetSizeOfMemoryAllocatedForIntermediateTensors()
      const override {
    return 0;
  }
  absl::StatusOr<uint64_t> GetSizeOfMemoryAllocatedForConstantTensors()
      const override {
    return 0;
  }

  bool SupportsConvertCommandBuffer() const override {
    return supports_command_buffer_ || force_convert_command_buffer_;
  }
  void SetForceConvertCommandBuffer(bool force) override {
    force_convert_command_buffer_ = force;
  }

  absl::StatusOr<std::unique_ptr<GpuPackBatch>> CreatePackBatch(
      absl::Span<const InputPackItem> items) override {
    ++create_pack_batch_calls_;
    last_pack_batch_size_ = items.size();
    if (return_null_pack_batch_) {
      return nullptr;
    }
    return std::make_unique<CountingPackBatch>(&pack_dispatch_calls_);
  }

  absl::StatusOr<std::unique_ptr<GpuUnpackBatch>> CreateUnpackBatch(
      absl::Span<const OutputUnpackItem> items) override {
    ++create_unpack_batch_calls_;
    last_unpack_batch_size_ = items.size();
    if (return_error_unpack_batch_) {
      return absl::UnavailableError("simulated command buffer failure");
    }
    return std::make_unique<CountingUnpackBatch>(&unpack_dispatch_calls_);
  }
};

class TestDelegateKernelLiteRt : public DelegateKernelLiteRt {
 public:
  explicit TestDelegateKernelLiteRt(const LiteRtRuntimeContext* runtime_context)
      : DelegateKernelLiteRt(runtime_context) {}

  void ConfigureForTest(MlDriftDelegateData* data, GpuBackend* backend,
                        std::unique_ptr<GpuInferenceContext> ctx,
                        std::vector<int64_t> input_indices,
                        std::vector<::ml_drift::ValueId> input_ids,
                        std::vector<int64_t> output_indices,
                        std::vector<::ml_drift::ValueId> output_ids,
                        bool is_opencl = false) {
    delegate_data_ = data;
    backend_ = backend;
    ctx_ = std::move(ctx);
    input_indices_ = std::move(input_indices);
    input_ids_ = std::move(input_ids);
    output_indices_ = std::move(output_indices);
    output_ids_ = std::move(output_ids);
    is_opencl_backend_ = is_opencl;
  }
};

struct FakeBufferContextState {
  absl::flat_hash_map<const TfLiteTensor*, LiteRtTensorBuffer> tensor_buffers;
  LiteRtTensorBufferType buffer_type = kLiteRtTensorBufferTypeOpenClBuffer;
  int unregister_calls = 0;
  int fail_unregister_after = -1;
};

FakeBufferContextState* g_buffer_state = nullptr;

LiteRtStatus FakeGetTensorBuffer(LiteRtExternalLiteRtBufferContext,
                                 const TfLiteTensor* tensor,
                                 LiteRtTensorBuffer* out_buffer) {
  auto it = g_buffer_state->tensor_buffers.find(tensor);
  if (it == g_buffer_state->tensor_buffers.end()) {
    return kLiteRtStatusErrorNotFound;
  }
  *out_buffer = it->second;
  return kLiteRtStatusOk;
}

void FakeDestroyTensorBuffer(LiteRtTensorBuffer) {}

LiteRtStatus FakeGetTensorBufferType(LiteRtTensorBuffer,
                                     LiteRtTensorBufferType* out_type) {
  *out_type = g_buffer_state->buffer_type;
  return kLiteRtStatusOk;
}

LiteRtStatus FakeUnregisterTensorBuffer(LiteRtExternalLiteRtBufferContext,
                                        const TfLiteTensor*) {
  ++g_buffer_state->unregister_calls;
  if (g_buffer_state->fail_unregister_after >= 0 &&
      g_buffer_state->unregister_calls >
          g_buffer_state->fail_unregister_after) {
    return kLiteRtStatusErrorUnknown;
  }
  return kLiteRtStatusOk;
}

struct TestHarness {
  FakeBufferContextState buffer_state;
  LiteRtRuntimeContext runtime_context = {};
  MlDriftDelegateData delegate_data;
  FakeGpuBackend backend;
  std::vector<TfLiteTensor> tensors;
  TfLiteContext tflite_context = {};
  std::unique_ptr<TestDelegateKernelLiteRt> kernel;

  void Init(int num_inputs, int num_outputs, bool is_opencl = false) {
    g_buffer_state = &buffer_state;
    runtime_context.get_external_litert_buffer_context_tensor_buffer =
        &FakeGetTensorBuffer;
    runtime_context.destroy_tensor_buffer = &FakeDestroyTensorBuffer;
    runtime_context.get_tensor_buffer_type = &FakeGetTensorBufferType;
    runtime_context.external_litert_buffer_context_unregister_tensor_buffer =
        &FakeUnregisterTensorBuffer;

    delegate_data.options = std::make_unique<MlDriftDelegateOptions>();
    delegate_data.options->litert_external_tensors_mode = false;

    const int total = num_inputs + num_outputs;
    tensors.resize(total);
    std::vector<int64_t> input_indices;
    std::vector<::ml_drift::ValueId> input_ids;
    std::vector<int64_t> output_indices;
    std::vector<::ml_drift::ValueId> output_ids;

    for (int i = 0; i < total; ++i) {
      tensors[i] = {};
      tensors[i].type = kTfLiteFloat32;
      tensors[i].allocation_type = kTfLiteArenaRw;
      tensors[i].bytes = 16;
      buffer_state.tensor_buffers[&tensors[i]] =
          reinterpret_cast<LiteRtTensorBuffer>(
              static_cast<uintptr_t>(0x1000 + i * 0x100));
      if (i < num_inputs) {
        input_indices.push_back(i);
        input_ids.push_back(i);
      } else {
        output_indices.push_back(i);
        output_ids.push_back(i);
      }
    }
    tflite_context.tensors = tensors.data();
    tflite_context.tensors_size = tensors.size();

    kernel = std::make_unique<TestDelegateKernelLiteRt>(&runtime_context);
    kernel->ConfigureForTest(
        &delegate_data, &backend, std::make_unique<FakeGpuInferenceContext>(),
        std::move(input_indices), std::move(input_ids),
        std::move(output_indices), std::move(output_ids), is_opencl);
    ASSERT_THAT(kernel->InitTensorConverters(&tflite_context), IsOk());
  }

  ~TestHarness() { g_buffer_state = nullptr; }
};

TEST(DelegateKernelLiteRtBatchTest,
     MultiTensorInputAndOutputUseAndReuseCommandBufferBatches) {
  TestHarness h;
  h.Init(/*num_inputs=*/2, /*num_outputs=*/2);

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());

  EXPECT_EQ(h.backend.create_pack_batch_calls_, 1);
  EXPECT_EQ(h.backend.last_pack_batch_size_, 2);
  EXPECT_EQ(h.backend.pack_dispatch_calls_, 1);
  EXPECT_EQ(h.backend.input_convert_calls_, 0);

  EXPECT_EQ(h.backend.create_unpack_batch_calls_, 1);
  EXPECT_EQ(h.backend.last_unpack_batch_size_, 2);
  EXPECT_EQ(h.backend.unpack_dispatch_calls_, 1);
  EXPECT_EQ(h.backend.output_convert_calls_, 0);

  // Empty FlushBufferCacheIfNeeded must not invalidate cached command buffers.
  ASSERT_THAT(h.kernel->FlushBufferCacheIfNeeded(&h.tflite_context), IsOk());
  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());

  EXPECT_EQ(h.backend.create_pack_batch_calls_, 1);
  EXPECT_EQ(h.backend.pack_dispatch_calls_, 2);
  EXPECT_EQ(h.backend.create_unpack_batch_calls_, 1);
  EXPECT_EQ(h.backend.unpack_dispatch_calls_, 2);
}

TEST(DelegateKernelLiteRtBatchTest, SingleTensorSkipsCommandBufferBatching) {
  TestHarness h;
  h.Init(/*num_inputs=*/1, /*num_outputs=*/1);

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());

  EXPECT_EQ(h.backend.create_pack_batch_calls_, 0);
  EXPECT_EQ(h.backend.create_unpack_batch_calls_, 0);
  EXPECT_EQ(h.backend.input_convert_calls_, 1);
  EXPECT_EQ(h.backend.output_convert_calls_, 1);
}

TEST(DelegateKernelLiteRtBatchTest,
     MemoryHandleChangeRecreatesCommandBufferBatches) {
  TestHarness h;
  h.Init(/*num_inputs=*/2, /*num_outputs=*/2);

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());
  ASSERT_EQ(h.backend.create_pack_batch_calls_, 1);
  ASSERT_EQ(h.backend.create_unpack_batch_calls_, 1);

  // Swap bound tensor buffer handles for one input and one output.
  h.buffer_state.tensor_buffers[&h.tensors[0]] =
      reinterpret_cast<LiteRtTensorBuffer>(static_cast<uintptr_t>(0x9000));
  h.buffer_state.tensor_buffers[&h.tensors[2]] =
      reinterpret_cast<LiteRtTensorBuffer>(static_cast<uintptr_t>(0xA000));

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());

  EXPECT_EQ(h.backend.create_pack_batch_calls_, 2);
  EXPECT_EQ(h.backend.pack_dispatch_calls_, 2);
  EXPECT_EQ(h.backend.create_unpack_batch_calls_, 2);
  EXPECT_EQ(h.backend.unpack_dispatch_calls_, 2);
}

TEST(DelegateKernelLiteRtBatchTest,
     FlushBufferCacheWithGlBufferInvalidatesBatches) {
  TestHarness h;
  h.Init(/*num_inputs=*/2, /*num_outputs=*/2, /*is_opencl=*/true);
  h.buffer_state.buffer_type = kLiteRtTensorBufferTypeGlBuffer;

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());
  ASSERT_EQ(h.backend.create_pack_batch_calls_, 1);
  ASSERT_EQ(h.backend.create_unpack_batch_calls_, 1);

  // Binding GL buffers marks them in tensors_to_flush_.
  ASSERT_THAT(h.kernel->BindExternalTensorBuffers(&h.tflite_context), IsOk());
  ASSERT_THAT(h.kernel->FlushBufferCacheIfNeeded(&h.tflite_context), IsOk());
  EXPECT_EQ(h.buffer_state.unregister_calls, 4);

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());

  EXPECT_EQ(h.backend.create_pack_batch_calls_, 2);
  EXPECT_EQ(h.backend.create_unpack_batch_calls_, 2);
}

TEST(DelegateKernelLiteRtBatchTest,
     NullOrErrorBatchCreationFallsBackToIndividualConverters) {
  TestHarness h;
  h.Init(/*num_inputs=*/2, /*num_outputs=*/2);
  h.backend.return_null_pack_batch_ = true;
  h.backend.return_error_unpack_batch_ = true;

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());

  EXPECT_EQ(h.backend.create_pack_batch_calls_, 1);
  EXPECT_EQ(h.backend.pack_dispatch_calls_, 0);
  EXPECT_EQ(h.backend.input_convert_calls_, 2);

  EXPECT_EQ(h.backend.create_unpack_batch_calls_, 1);
  EXPECT_EQ(h.backend.unpack_dispatch_calls_, 0);
  EXPECT_EQ(h.backend.output_convert_calls_, 2);
}

TEST(DelegateKernelLiteRtBatchTest, OptionsDisableCommandBufferBatching) {
  TestHarness h;
  h.Init(/*num_inputs=*/2, /*num_outputs=*/2);
  h.delegate_data.options->enable_convert_command_buffer = false;

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());

  EXPECT_EQ(h.backend.create_pack_batch_calls_, 0);
  EXPECT_EQ(h.backend.create_unpack_batch_calls_, 0);
  EXPECT_EQ(h.backend.input_convert_calls_, 2);
  EXPECT_EQ(h.backend.output_convert_calls_, 2);
}

TEST(DelegateKernelLiteRtBatchTest, OptionsForceCommandBufferBatching) {
  TestHarness h;
  h.Init(/*num_inputs=*/2, /*num_outputs=*/2);
  h.backend.supports_command_buffer_ = false;
  h.delegate_data.options->force_convert_command_buffer = true;
  ASSERT_THAT(h.kernel->InitTensorConverters(&h.tflite_context), IsOk());

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());

  EXPECT_TRUE(h.backend.force_convert_command_buffer_);
  EXPECT_EQ(h.backend.create_pack_batch_calls_, 1);
  EXPECT_EQ(h.backend.pack_dispatch_calls_, 1);
  EXPECT_EQ(h.backend.create_unpack_batch_calls_, 1);
  EXPECT_EQ(h.backend.unpack_dispatch_calls_, 1);
}

TEST(DelegateKernelLiteRtBatchTest,
     ReRecordFailureClearsCachedBatchAndMemories) {
  TestHarness h;
  h.Init(/*num_inputs=*/2, /*num_outputs=*/2);

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_EQ(h.backend.create_pack_batch_calls_, 1);
  ASSERT_EQ(h.backend.pack_dispatch_calls_, 1);

  // Swap memory handles and inject an error during CreateIOBuffer.
  h.buffer_state.tensor_buffers[&h.tensors[0]] =
      reinterpret_cast<LiteRtTensorBuffer>(static_cast<uintptr_t>(0x3000));
  h.backend.fail_create_io_buffer_ = true;
  EXPECT_FALSE(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context)
          .ok());
  EXPECT_EQ(h.backend.pack_dispatch_calls_, 1);

  // Subsequent invoke with the same new handles must re-record rather than
  // dispatching the stale batch for the old handles.
  h.backend.fail_create_io_buffer_ = false;
  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  EXPECT_EQ(h.backend.create_pack_batch_calls_, 2);
  EXPECT_EQ(h.backend.pack_dispatch_calls_, 2);
}

TEST(DelegateKernelLiteRtBatchTest,
     PartialFlushUnregisterFailureStillInvalidatesBatches) {
  TestHarness h;
  h.Init(/*num_inputs=*/2, /*num_outputs=*/2, /*is_opencl=*/true);
  h.buffer_state.buffer_type = kLiteRtTensorBufferTypeGlBuffer;

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());
  ASSERT_EQ(h.backend.create_pack_batch_calls_, 1);
  ASSERT_EQ(h.backend.create_unpack_batch_calls_, 1);

  ASSERT_THAT(h.kernel->BindExternalTensorBuffers(&h.tflite_context), IsOk());
  h.buffer_state.fail_unregister_after = 1;
  EXPECT_FALSE(h.kernel->FlushBufferCacheIfNeeded(&h.tflite_context).ok());

  ASSERT_THAT(
      h.kernel->ConvertNonExternalInputTensorsToGpuMemory(&h.tflite_context),
      IsOk());
  ASSERT_THAT(
      h.kernel->ConvertGpuMemoryToNonExternalOutputTensors(&h.tflite_context),
      IsOk());
  EXPECT_EQ(h.backend.create_pack_batch_calls_, 2);
  EXPECT_EQ(h.backend.create_unpack_batch_calls_, 2);
}

}  // namespace
}  // namespace litert::ml_drift
