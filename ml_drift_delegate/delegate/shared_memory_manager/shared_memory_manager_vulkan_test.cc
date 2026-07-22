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

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_vulkan.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/syrtis/testing/vulkan_test.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/shared_memory_manager/gf32_graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/c/common.h"

namespace ml_drift {

using ::testing::status::StatusIs;

class SharedMemoryManagerTest : public syrtis::VulkanOperationTest {
 protected:
  void SetUp() override {
    syrtis::VulkanOperationTest::SetUp();
    shared_vulkan_env_.vulkan_env() = std::move(*exec_env_.GetEnv());
  }

  ::litert::ml_drift::SharedVulkanEnv shared_vulkan_env_;
};

TEST_F(SharedMemoryManagerTest, GetNonExistingExternalTensor) {
  ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ValueIdToSharedTensorMap quant_param_tensors;
  CreateGpuModelInfo create_info;
  GraphFloat32 graph;

  TfLiteContext context;
  auto manager = MakeSharedMemoryManagerVulkan(
      &shared_vulkan_env_, create_info,
      std::make_unique<GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_external_tensors=*/false, /*serialization_cache=*/nullptr,
      /*madvise_original_tensors=*/false);
  SharedMemoryManager::GlobalId non_existing_global_id =
      SharedMemoryManager::GlobalId::BuildSourceId(0);
  EXPECT_THAT(manager->GetExternalConstantTensor(non_existing_global_id),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(SharedMemoryManagerTest, CreateExternalFloatTensorF16FromF32) {
  ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ValueIdToSharedTensorMap param_tensors;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;
  create_info.storage_type = TensorStorageType::BUFFER;

  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* input = graph.NewValue();
  Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = BHWC(1, 1, 1, 10);

  TfLiteContext context;
  auto manager = MakeSharedMemoryManagerVulkan(
      &shared_vulkan_env_, create_info,
      std::make_unique<GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, param_tensors,
      /*has_prepacked_external_tensors=*/false, /*serialization_cache=*/nullptr,
      /*madvise_original_tensors=*/false);
  int global_tensor_id = 0;
  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  std::vector<float> dummy(10);
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 10;
  tflite_tensor.data.f = dummy.data();
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = global_tensor_id;

  // Check that the non-existing global id is not found.
  SharedMemoryManager::GlobalId non_existing_global_id =
      SharedMemoryManager::GlobalId::BuildSourceId(1234);
  EXPECT_THAT(manager->GetExternalConstantTensor(non_existing_global_id),
              StatusIs(absl::StatusCode::kInternal));

  // `buffer_id_to_spatial_tensor` stores the Gpu tensors that are shared
  // between different signatures of the same tflite model.
  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 0);
  absl::flat_hash_map<ml_drift::ValueId,
                      ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  ASSERT_OK(manager->RegisterExternalConstantTensors(
      input->id, shared_tflite_tensor, local_to_global_id_map));
  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 1);
  // Emulate the call from the second signature
  local_to_global_id_map.clear();
  ASSERT_OK(manager->RegisterExternalConstantTensors(
      input->id, shared_tflite_tensor, local_to_global_id_map));
  // For running the second signature, the Gpu tensors should be reused, so the
  // number of the shared Gpu tensors should remain the same.
  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 1);

  ASSERT_EQ(local_to_global_id_map.size(), 1);
  for (auto& [_, global_id] : local_to_global_id_map) {
    ASSERT_OK_AND_ASSIGN(ml_drift::GpuSpatialTensor * external_tensor,
                         manager->GetExternalConstantTensor(global_id));
    EXPECT_NE(external_tensor, nullptr);
    EXPECT_EQ(external_tensor->GetDescriptor().GetDataType(),
              DataType::FLOAT16);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

TEST_F(SharedMemoryManagerTest, CreateExternalFloatTensorF16FromF16) {
  ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ValueIdToSharedTensorMap param_tensors;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;
  create_info.storage_type = TensorStorageType::BUFFER;

  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* input = graph.NewValue();
  Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = BHWC(1, 1, 1, 10);

  TfLiteContext context;
  auto manager = MakeSharedMemoryManagerVulkan(
      &shared_vulkan_env_, create_info,
      std::make_unique<GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, param_tensors,
      /*has_prepacked_external_tensors=*/false, /*serialization_cache=*/nullptr,
      /*madvise_original_tensors=*/false);
  int global_tensor_id = 0;
  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  std::vector<TfLiteFloat16> dummy(10);
  tflite_tensor.type = TfLiteType::kTfLiteFloat16;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 10;
  tflite_tensor.data.f16 = dummy.data();
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = global_tensor_id;

  // `buffer_id_to_spatial_tensor` stores the Gpu tensors that are shared
  // between different signatures of the same tflite model.
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
    EXPECT_EQ(external_tensor->GetDescriptor().GetDataType(),
              DataType::FLOAT16);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

TEST_F(SharedMemoryManagerTest, CreateExternalFloatTensorF32FromF16) {
  ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ValueIdToSharedTensorMap param_tensors;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;
  create_info.storage_type = TensorStorageType::BUFFER;

  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* input = graph.NewValue();
  Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = BHWC(1, 1, 1, 10);

  TfLiteContext context;
  auto manager = MakeSharedMemoryManagerVulkan(
      &shared_vulkan_env_, create_info,
      std::make_unique<GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, param_tensors,
      /*has_prepacked_external_tensors=*/false, /*serialization_cache=*/nullptr,
      /*madvise_original_tensors=*/false);
  int global_tensor_id = 0;
  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  std::vector<TfLiteFloat16> dummy(10);
  tflite_tensor.type = TfLiteType::kTfLiteFloat16;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 10;
  tflite_tensor.data.f16 = dummy.data();
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = global_tensor_id;

  // `buffer_id_to_spatial_tensor` stores the Gpu tensors that are shared
  // between different signatures of the same tflite model.
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
    EXPECT_EQ(external_tensor->GetDescriptor().GetDataType(),
              DataType::FLOAT16);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

TEST_F(SharedMemoryManagerTest, CreateExternalFloatTensorF32FromF32) {
  ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ValueIdToSharedTensorMap param_tensors;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F32;
  create_info.storage_type = TensorStorageType::BUFFER;

  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* input = graph.NewValue();
  Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = BHWC(1, 1, 1, 10);
  input->tensor.type = DataType::FLOAT32;

  TfLiteContext context;
  auto manager = MakeSharedMemoryManagerVulkan(
      &shared_vulkan_env_, create_info,
      std::make_unique<GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, param_tensors,
      /*has_prepacked_external_tensors=*/false, /*serialization_cache=*/nullptr,
      /*madvise_original_tensors=*/false);
  int global_tensor_id = 0;
  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  std::vector<float> dummy(10);
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 10;
  tflite_tensor.data.f = dummy.data();
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = global_tensor_id;

  // `buffer_id_to_spatial_tensor` stores the Gpu tensors that are shared
  // between different signatures of the same tflite model.
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
    EXPECT_EQ(external_tensor->GetDescriptor().GetDataType(),
              DataType::FLOAT32);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

TEST_F(SharedMemoryManagerTest, CreateExternalQuantizedTensor) {
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
  input->tensor.shape = BHWC(1, 1, 1, 10);

  TfLiteContext context;
  auto manager = MakeSharedMemoryManagerVulkan(
      &shared_vulkan_env_, create_info,
      std::make_unique<GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_external_tensors=*/false, /*serialization_cache=*/nullptr,
      /*madvise_original_tensors=*/false);
  int global_tensor_id = 0;
  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteAffineQuantization;
  tflite_tensor.type = TfLiteType::kTfLiteInt8;
  std::vector<int8_t> dummy(10);
  tflite_tensor.data.int8 = dummy.data();
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 10;

  tflite_tensor.quantization.params = new TfLiteAffineQuantization();
  TfLiteAffineQuantization* quant_params =
      static_cast<TfLiteAffineQuantization*>(tflite_tensor.quantization.params);
  quant_params->quantized_dimension = 0;
  quant_params->scale = TfLiteFloatArrayCreate(1);
  quant_params->scale->data[0] = 1.0;
  quant_params->zero_point = TfLiteIntArrayCreate(1);
  quant_params->zero_point->data[0] = 0;
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = global_tensor_id;

  // Check that the non-existing global id is not found.
  SharedMemoryManager::GlobalId non_existing_global_id =
      SharedMemoryManager::GlobalId::BuildSourceId(1234);
  EXPECT_THAT(manager->GetExternalConstantTensor(non_existing_global_id),
              StatusIs(absl::StatusCode::kInternal));

  // `buffer_id_to_spatial_tensor` and `quant_param_tensors` store the Gpu
  // tensors that are shared between different signatures of the same tflite
  // model.
  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 0);
  EXPECT_EQ(quant_param_tensors.size(), 0);
  absl::flat_hash_map<ml_drift::ValueId,
                      ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  ASSERT_OK(manager->RegisterExternalConstantTensors(
      input->id, shared_tflite_tensor, local_to_global_id_map));
  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 1);
  EXPECT_EQ(quant_param_tensors.size(), 2);
  // Emulate the call from the second signature
  local_to_global_id_map.clear();
  ASSERT_OK(manager->RegisterExternalConstantTensors(
      input->id, shared_tflite_tensor, local_to_global_id_map));
  // For running the second signature, the Gpu tensors should be reused, so the
  // number of the shared Gpu tensors should remain the same.
  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 1);
  EXPECT_EQ(quant_param_tensors.size(), 2);

  ASSERT_EQ(local_to_global_id_map.size(), 3);
  for (auto& [_, global_id] : local_to_global_id_map) {
    ASSERT_OK_AND_ASSIGN(ml_drift::GpuSpatialTensor * external_tensor,
                         manager->GetExternalConstantTensor(global_id));
    EXPECT_NE(external_tensor, nullptr);
  }
  TfLiteIntArrayFree(quant_params->zero_point);
  TfLiteFloatArrayFree(quant_params->scale);
  delete quant_params;
  TfLiteIntArrayFree(tflite_tensor.dims);
}

void RunBlockwiseQuantizationTest(::litert::ml_drift::SharedVulkanEnv* env,
                                  TfLiteType zp_type, int zp_size,
                                  const std::vector<int>& expected_zp_values,
                                  bool use_invalid_zp = false,
                                  bool expect_failure = false) {
  ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ValueIdToSharedTensorMap quant_param_tensors;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;
  create_info.storage_type = TensorStorageType::BUFFER;

  GraphFloat32 graph;
  Node* node = graph.NewNode();
  node->operation.type = "fully_connected";
  Value* weights_val = graph.NewValue();
  graph.AddConsumer(node->id, weights_val->id);
  weights_val->tensor.shape = BHWC(100, 1, 1, 64);

  TfLiteContext context;
  std::vector<TfLiteTensor> tensors(3);
  context.tensors = tensors.data();
  context.tensors_size = 3;

  // 0: weights (INT8)
  tensors[0].type = TfLiteType::kTfLiteInt8;
  tensors[0].dims = TfLiteIntArrayCreate(2);
  tensors[0].dims->data[0] = 100;
  tensors[0].dims->data[1] = 64;
  std::vector<int8_t> weights_data(100 * 64, 1);
  tensors[0].data.int8 = weights_data.data();

  TfLiteBlockwiseQuantization* quant_params = new TfLiteBlockwiseQuantization();
  quant_params->scale = 1;
  quant_params->zero_point = use_invalid_zp ? -1 : 2;
  quant_params->blocksize = 32;
  tensors[0].quantization.type = kTfLiteBlockwiseQuantization;
  tensors[0].quantization.params = quant_params;

  // 1: scale (FP32)
  tensors[1].type = TfLiteType::kTfLiteFloat32;
  tensors[1].dims = TfLiteIntArrayCreate(2);
  tensors[1].dims->data[0] = 100;
  tensors[1].dims->data[1] = 2;
  std::vector<float> scale_data(100 * 2, 0.5f);
  tensors[1].data.f = scale_data.data();

  // 2: zero point
  std::vector<int32_t> zp_data_i32;
  std::vector<int64_t> zp_data_i64;
  std::vector<int8_t> zp_data_i8;

  tensors[2].type = zp_type;
  tensors[2].dims = TfLiteIntArrayCreate(zp_size == 1 ? 1 : 2);
  if (zp_size == 1) {
    tensors[2].dims->data[0] = 1;
  } else {
    tensors[2].dims->data[0] = 100;
    tensors[2].dims->data[1] = 2;
  }

  if (zp_type == kTfLiteInt32) {
    zp_data_i32.resize(zp_size, expected_zp_values[0]);
    if (zp_size > 1) {
      for (size_t i = 0; i < expected_zp_values.size(); ++i) {
        zp_data_i32[i] = expected_zp_values[i];
      }
    }
    tensors[2].data.i32 = zp_data_i32.data();
    tensors[2].bytes = zp_size * sizeof(int32_t);
  } else if (zp_type == kTfLiteInt64) {
    zp_data_i64.resize(zp_size, expected_zp_values[0]);
    if (zp_size > 1) {
      for (size_t i = 0; i < expected_zp_values.size(); ++i) {
        zp_data_i64[i] = expected_zp_values[i];
      }
    }
    tensors[2].data.i64 = zp_data_i64.data();
    tensors[2].bytes = zp_size * sizeof(int64_t);
  } else if (zp_type == kTfLiteInt8) {
    zp_data_i8.resize(zp_size, expected_zp_values[0]);
    if (zp_size > 1) {
      for (size_t i = 0; i < expected_zp_values.size(); ++i) {
        zp_data_i8[i] = expected_zp_values[i];
      }
    }
    tensors[2].data.int8 = zp_data_i8.data();
    tensors[2].bytes = zp_size * sizeof(int8_t);
  }

  auto manager = MakeSharedMemoryManagerVulkan(
      env, create_info, std::make_unique<GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_external_tensors=*/false,
      /*serialization_cache=*/nullptr,
      /*madvise_original_tensors=*/false);

  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = 0;

  absl::flat_hash_map<ml_drift::ValueId,
                      ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  absl::Status status = manager->RegisterExternalConstantTensors(
      weights_val->id, shared_tflite_tensor, local_to_global_id_map);

  if (expect_failure) {
    EXPECT_FALSE(status.ok());
    TfLiteIntArrayFree(tensors[0].dims);
    delete quant_params;
    TfLiteIntArrayFree(tensors[1].dims);
    TfLiteIntArrayFree(tensors[2].dims);
    return;
  }

  ASSERT_OK(status);

  EXPECT_EQ(buffer_id_to_spatial_tensor.size(), 1);
  EXPECT_EQ(quant_param_tensors.size(), 2);

  int params_found = 0;
  for (auto& [_, global_id] : local_to_global_id_map) {
    if (global_id.IsParamId()) {
      params_found++;
      ASSERT_OK_AND_ASSIGN(ml_drift::GpuSpatialTensor * external_tensor,
                           manager->GetExternalConstantTensor(global_id));
      EXPECT_NE(external_tensor, nullptr);
      EXPECT_EQ(external_tensor->GetDescriptor().GetDataType(),
                DataType::FLOAT16);
      EXPECT_EQ(external_tensor->GetDescriptor().GetBHWCShape(),
                BHWC(1, 1, 25, 8));
    }
  }
  EXPECT_EQ(params_found, 2);

  TfLiteIntArrayFree(tensors[0].dims);
  delete quant_params;
  TfLiteIntArrayFree(tensors[1].dims);
  TfLiteIntArrayFree(tensors[2].dims);
}

TEST_F(SharedMemoryManagerTest,
       CreateBlockwiseQuantizedTensorWithInt32ZeroPoints) {
  std::vector<int> expected_zp(200, 5);
  RunBlockwiseQuantizationTest(&shared_vulkan_env_, kTfLiteInt32, 200,
                               expected_zp);
}

TEST_F(SharedMemoryManagerTest,
       CreateBlockwiseQuantizedTensorFailsWithInt64ZeroPoints) {
  std::vector<int> expected_zp(200, 5);
  RunBlockwiseQuantizationTest(&shared_vulkan_env_, kTfLiteInt64, 200,
                               expected_zp,
                               /*use_invalid_zp=*/false,
                               /*expect_failure=*/true);
}

TEST_F(SharedMemoryManagerTest,
       CreateBlockwiseQuantizedTensorFailsWithInt8ZeroPoints) {
  std::vector<int> expected_zp(200, 5);
  RunBlockwiseQuantizationTest(&shared_vulkan_env_, kTfLiteInt8, 200,
                               expected_zp,
                               /*use_invalid_zp=*/false,
                               /*expect_failure=*/true);
}

TEST_F(SharedMemoryManagerTest,
       CreateBlockwiseQuantizedTensorWithBroadcastZeroPoints) {
  std::vector<int> expected_zp(1, 5);
  RunBlockwiseQuantizationTest(&shared_vulkan_env_, kTfLiteInt32, 1,
                               expected_zp);
}

TEST_F(SharedMemoryManagerTest, CreateBlockwiseQuantizedTensorSymmetric) {
  std::vector<int> expected_zp(200, 0);
  RunBlockwiseQuantizationTest(&shared_vulkan_env_, kTfLiteInt32, 200,
                               expected_zp, /*use_invalid_zp=*/true);
}

}  // namespace ml_drift
