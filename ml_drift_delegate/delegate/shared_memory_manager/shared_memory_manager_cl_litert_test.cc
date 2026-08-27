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

// Tests for TryCreateTensorViaAhwb and the create_tensor_func contract in
// MakeSharedMemoryManagerClLitert.
//
// This file contains two groups of tests:
//
//   1. Android-only direct unit tests (#ifdef __ANDROID__):
//      These call TryCreateTensorViaAhwb directly and require a device with
//      a Mali GPU that supports the cl_arm_import_memory extension. They will
//      GTEST_SKIP on devices without the extension.
//
//   2. Cross-platform integration tests:
//      These test the create_tensor_func path through
//      MakeSharedMemoryManagerClLitert and work on any OpenCL-capable device
//      (host GPU or Android).
//
// === Running on host (NVIDIA GPU) ===
//
//   bazel test //third_party/odml/litert/ml_drift/delegate/\
//       shared_memory_manager:shared_memory_manager_cl_litert_test
//
//   Only the cross-platform integration tests will run; the Android-only
//   tests are compiled out.
//
// === Running on Android (Mali GPU) ===
//
//   Step 1: Cross-compile for Android ARM64:
//
//     bazel build --config=android_arm64 //third_party/odml/litert/ml_drift/\
//         delegate/shared_memory_manager:shared_memory_manager_cl_litert_test
//
//   Step 2: Push the test binary to the device:
//
//     adb push bazel-bin/third_party/odml/litert/ml_drift/delegate/\
//         shared_memory_manager/shared_memory_manager_cl_litert_test \
//         /data/local/tmp/
//
//   Step 3: Run the test on the device:
//
//     adb shell /data/local/tmp/shared_memory_manager_cl_litert_test
//

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_cl_litert.h"

#include <memory>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/shared_memory_manager/gf32_graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/c/common.h"

#ifdef __ANDROID__
#include "ml_drift/common/data_type.h"  // from @ml_drift
#endif  // __ANDROID__

namespace ml_drift {
namespace {

class SharedMemoryManagerClLitertTest : public ::testing::Test {
 public:
  void SetUp() override {
    ASSERT_OK(cl::LoadOpenCL());
    ASSERT_OK(CreateEnvironment(&env_));
  }

 protected:
  cl::Environment env_;
};

// ---------------------------------------------------------------------------
// Direct unit tests for TryCreateTensorViaAhwb (Android-only)
// ---------------------------------------------------------------------------

#ifdef __ANDROID__

TEST_F(SharedMemoryManagerClLitertTest,
       AhwbRejectsNonTexture2DStorageTypes) {
  // BUFFER should be rejected.
  {
    TensorDescriptor desc(DataType::FLOAT16, TensorStorageType::BUFFER,
                          Layout::LINEAR);
    desc.SetBHWCShape(BHWC(1, 1, 1, 64));
    std::unique_ptr<GpuSpatialTensor> tensor;
    EXPECT_FALSE(internal::TryCreateTensorViaAhwb(env_, desc, tensor));
    EXPECT_EQ(tensor, nullptr);
  }
  // IMAGE_BUFFER should be rejected.
  {
    TensorDescriptor desc(DataType::FLOAT16, TensorStorageType::IMAGE_BUFFER,
                          Layout::LINEAR);
    desc.SetBHWCShape(BHWC(1, 1, 1, 64));
    std::unique_ptr<GpuSpatialTensor> tensor;
    EXPECT_FALSE(internal::TryCreateTensorViaAhwb(env_, desc, tensor));
    EXPECT_EQ(tensor, nullptr);
  }
  // SINGLE_TEXTURE_2D should be rejected.
  {
    TensorDescriptor desc(DataType::FLOAT16,
                          TensorStorageType::SINGLE_TEXTURE_2D, Layout::HW);
    desc.SetBHWCShape(BHWC(1, 1, 1, 4));
    std::unique_ptr<GpuSpatialTensor> tensor;
    EXPECT_FALSE(internal::TryCreateTensorViaAhwb(env_, desc, tensor));
    EXPECT_EQ(tensor, nullptr);
  }
  // TEXTURE_ARRAY should be rejected.
  {
    TensorDescriptor desc(DataType::FLOAT16, TensorStorageType::TEXTURE_ARRAY,
                          Layout::HW);
    desc.SetBHWCShape(BHWC(1, 1, 1, 4));
    std::unique_ptr<GpuSpatialTensor> tensor;
    EXPECT_FALSE(internal::TryCreateTensorViaAhwb(env_, desc, tensor));
    EXPECT_EQ(tensor, nullptr);
  }
}

TEST_F(SharedMemoryManagerClLitertTest,
       AhwbCreatesTexture2DTensorWithData) {
  // Create a TEXTURE_2D tensor descriptor with weight data.
  TensorDescriptor desc(DataType::FLOAT32, TensorStorageType::TEXTURE_2D,
                        Layout::HW);
  desc.SetBHWCShape(BHWC(1, 4, 4, 4));  // 4x4x4 = 16 RGBA pixels = 4x4 image

  // Fill with known data pattern.
  const size_t num_elements = 4 * 4 * 4;
  std::vector<float> data(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = static_cast<float>(i);
  }
  desc.SetData(std::vector<uint8_t>(
      reinterpret_cast<const uint8_t*>(data.data()),
      reinterpret_cast<const uint8_t*>(data.data() + data.size())));

  std::unique_ptr<GpuSpatialTensor> tensor;
  bool success = internal::TryCreateTensorViaAhwb(env_, desc, tensor);

  if (!success) {
    GTEST_SKIP() << "AHWB allocation not supported on this device";
  }

  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetDescriptor().GetStorageType(),
            TensorStorageType::TEXTURE_2D);
  EXPECT_EQ(tensor->GetDescriptor().GetDataType(), DataType::FLOAT32);
}

TEST_F(SharedMemoryManagerClLitertTest,
       AhwbCreatesTexture2DTensorWithoutData) {
  // Create a TEXTURE_2D tensor with no initial data (e.g., an intermediate
  // activation tensor).
  TensorDescriptor desc(DataType::FLOAT32, TensorStorageType::TEXTURE_2D,
                        Layout::HW);
  desc.SetBHWCShape(BHWC(1, 4, 4, 4));

  std::unique_ptr<GpuSpatialTensor> tensor;
  bool success = internal::TryCreateTensorViaAhwb(env_, desc, tensor);

  if (!success) {
    GTEST_SKIP() << "AHWB allocation not supported on this device";
  }

  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetDescriptor().GetStorageType(),
            TensorStorageType::TEXTURE_2D);
}

TEST_F(SharedMemoryManagerClLitertTest,
       AhwbPreservesDescriptorMetadata) {
  TensorDescriptor desc(DataType::FLOAT32, TensorStorageType::TEXTURE_2D,
                        Layout::HW);
  BHWC shape(1, 8, 8, 4);
  desc.SetBHWCShape(shape);

  std::unique_ptr<GpuSpatialTensor> tensor;
  bool success = internal::TryCreateTensorViaAhwb(env_, desc, tensor);

  if (!success) {
    GTEST_SKIP() << "AHWB allocation not supported on this device";
  }

  ASSERT_NE(tensor, nullptr);
  // Verify the tensor descriptor preserves the metadata.
  const auto& result_desc = tensor->GetDescriptor();
  EXPECT_EQ(result_desc.GetStorageType(), TensorStorageType::TEXTURE_2D);
  EXPECT_EQ(result_desc.GetDataType(), DataType::FLOAT32);
  EXPECT_EQ(result_desc.GetBHWCShape(), shape);
  // The descriptor should not carry the data (CopyWithoutData was used).
  EXPECT_TRUE(result_desc.GetData().empty());
}

TEST_F(SharedMemoryManagerClLitertTest,
       AhwbDoesNotModifyTensorOnFailure) {
  // Use an unsupported storage type. The function should return false
  // and leave the tensor pointer unchanged.
  TensorDescriptor desc(DataType::FLOAT16, TensorStorageType::BUFFER,
                        Layout::LINEAR);
  desc.SetBHWCShape(BHWC(1, 1, 1, 64));

  std::unique_ptr<GpuSpatialTensor> tensor;
  EXPECT_FALSE(internal::TryCreateTensorViaAhwb(env_, desc, tensor));
  EXPECT_EQ(tensor, nullptr);
}

TEST_F(SharedMemoryManagerClLitertTest,
       AhwbReturnsFalseWhenClImportMemoryArmIsNull) {
  // Temporarily null out clImportMemoryARM to simulate a device without
  // the ARM import extension.
  auto* original = cl::clImportMemoryARM;
  cl::clImportMemoryARM = nullptr;

  TensorDescriptor desc(DataType::FLOAT16, TensorStorageType::TEXTURE_2D,
                        Layout::HW);
  desc.SetBHWCShape(BHWC(1, 2, 2, 4));

  std::unique_ptr<GpuSpatialTensor> tensor;
  EXPECT_FALSE(internal::TryCreateTensorViaAhwb(env_, desc, tensor));
  EXPECT_EQ(tensor, nullptr);

  // Restore.
  cl::clImportMemoryARM = original;
}

#endif  // __ANDROID__

// ---------------------------------------------------------------------------
// Integration tests that work on both host and Android.
// These test the create_tensor_func contract through
// MakeSharedMemoryManagerClLitert.
// ---------------------------------------------------------------------------

TEST_F(SharedMemoryManagerClLitertTest,
       CreateTensorFuncProducesValidTensor) {
  // Verify that the create_tensor_func (which internally tries AHWB
  // on Android, falling through to CreateTensor on host) produces a
  // valid tensor.
  ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ValueIdToSharedTensorMap quant_param_tensors;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;
  create_info.storage_type = TensorStorageType::TEXTURE_2D;

  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* input = graph.NewValue();
  Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = BHWC(1, 2, 2, 4);

  TfLiteContext context;
  auto manager = MakeSharedMemoryManagerClLitert(
      env_, /*runtime_context=*/nullptr, create_info,
      std::make_unique<GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_external_tensors=*/false,
      /*serialization_cache=*/nullptr,
      /*madvise_original_tensors=*/false,
      /*weight_loader=*/nullptr);

  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  std::vector<float> dummy(2 * 2 * 4);
  tflite_tensor.dims = TfLiteIntArrayCreate(4);
  tflite_tensor.dims->data[0] = 1;
  tflite_tensor.dims->data[1] = 2;
  tflite_tensor.dims->data[2] = 2;
  tflite_tensor.dims->data[3] = 4;
  tflite_tensor.data.f = dummy.data();
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = 0;

  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 0);
  absl::flat_hash_map<ml_drift::ValueId,
                      ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  ASSERT_OK(manager->RegisterExternalConstantTensors(
      input->id, shared_tflite_tensor, local_to_global_id_map));
  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 1);

  ASSERT_EQ(local_to_global_id_map.size(), 1);
  for (auto& [_, global_id] : local_to_global_id_map) {
    ASSERT_OK_AND_ASSIGN(ml_drift::GpuSpatialTensor * external_tensor,
                         manager->GetExternalConstantTensor(global_id));
    EXPECT_NE(external_tensor, nullptr);
    EXPECT_EQ(external_tensor->GetDescriptor().GetStorageType(),
              TensorStorageType::TEXTURE_2D);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

TEST_F(SharedMemoryManagerClLitertTest,
       CreateTensorFuncFallsBackForBufferStorageType) {
  // BUFFER storage type should work via standard CreateTensor (the
  // AHWB path rejects non-TEXTURE_2D on Android, and is not compiled
  // on host).
  ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ValueIdToSharedTensorMap quant_param_tensors;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;
  create_info.storage_type = TensorStorageType::BUFFER;

  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* input = graph.NewValue();
  Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = BHWC(1, 1, 1, 64);

  TfLiteContext context;
  auto manager = MakeSharedMemoryManagerClLitert(
      env_, /*runtime_context=*/nullptr, create_info,
      std::make_unique<GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_external_tensors=*/false,
      /*serialization_cache=*/nullptr,
      /*madvise_original_tensors=*/false,
      /*weight_loader=*/nullptr);

  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  std::vector<float> dummy(64);
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 64;
  tflite_tensor.data.f = dummy.data();
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = 0;

  absl::flat_hash_map<ml_drift::ValueId,
                      ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  ASSERT_OK(manager->RegisterExternalConstantTensors(
      input->id, shared_tflite_tensor, local_to_global_id_map));
  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 1);

  for (auto& [_, global_id] : local_to_global_id_map) {
    ASSERT_OK_AND_ASSIGN(ml_drift::GpuSpatialTensor * external_tensor,
                         manager->GetExternalConstantTensor(global_id));
    EXPECT_NE(external_tensor, nullptr);
    EXPECT_EQ(external_tensor->GetDescriptor().GetStorageType(),
              TensorStorageType::BUFFER);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

}  // namespace
}  // namespace ml_drift
