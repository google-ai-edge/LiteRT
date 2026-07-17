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

#import "third_party/odml/litert/ml_drift/delegate/shared_memory_manager/shared_memory_manager_metal.h"

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>  // IWYU pragma: keep

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/metal/metal_device.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/shared_memory_manager/gf32_graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/c/common.h"

@interface SharedMemoryManagerMetalTest : XCTestCase
@end

@implementation SharedMemoryManagerMetalTest

- (void)testSharedMemoryManagerMetalGetNonExistingExternalTensor {
  self.continueAfterFailure = NO;
  ml_drift::ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ml_drift::ValueIdToSharedTensorMap quant_param_tensors;
  ml_drift::CreateGpuModelInfo create_info;
  ml_drift::GraphFloat32 graph;
  TfLiteContext context;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ml_drift::metal::MetalDevice metal_device(device);
  // Create the SharedMemoryManagerMetal
  auto metal_shared_memory_manager = ml_drift::MakeSharedMemoryManagerMetal(
      &metal_device, create_info, std::make_unique<ml_drift::GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_external_tensors*/ false,
      /*serialized_external_tensors=*/nullptr, /*madvise_original_tensors*/ false);

  ml_drift::SharedMemoryManager::GlobalId non_existing_global_id =
      ml_drift::SharedMemoryManager::GlobalId::BuildSourceId(0);
  XCTAssertEqual(metal_shared_memory_manager->GetExternalConstantTensor(non_existing_global_id)
                     .status()
                     .code(),
                 absl::StatusCode::kInternal);
}

- (void)testCreateExternalFloatTensorF16FromF32 {
  self.continueAfterFailure = NO;
  ml_drift::ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ml_drift::ValueIdToSharedTensorMap quant_param_tensors;

  ml_drift::CreateGpuModelInfo create_info;
  create_info.precision = ml_drift::CalculationsPrecision::F16;
  create_info.storage_type = ml_drift::TensorStorageType::BUFFER;

  ml_drift::GraphFloat32 graph;
  ml_drift::Node* node = graph.NewNode();
  ml_drift::Value* input = graph.NewValue();
  ml_drift::Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = ml_drift::BHWC(1, 1, 1, 10);

  TfLiteContext context;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ml_drift::metal::MetalDevice metal_device(device);
  auto manager = ml_drift::MakeSharedMemoryManagerMetal(
      &metal_device, create_info, std::make_unique<ml_drift::GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_tflite_tensors=*/false,
      /*serialized_external_tensors=*/nullptr,
      /*madvise_original_tensors=*/false);

  int global_tensor_id = 0;
  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  std::vector<float> dummy(10);
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 10;
  tflite_tensor.data.f = dummy.data();
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = global_tensor_id;

  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 0);
  absl::flat_hash_map<ml_drift::ValueId, ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  XCTAssertTrue(
      manager
          ->RegisterExternalConstantTensors(input->id, shared_tflite_tensor, local_to_global_id_map)
          .ok());
  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 1);

  XCTAssertEqual(local_to_global_id_map.size(), 1);
  for (auto& [_, global_id] : local_to_global_id_map) {
    auto status_or_tensor = manager->GetExternalConstantTensor(global_id);
    XCTAssertTrue(status_or_tensor.ok());
    ml_drift::GpuSpatialTensor* external_tensor = *status_or_tensor;
    XCTAssertNotEqual(external_tensor, nullptr);
    XCTAssertEqual(external_tensor->GetDescriptor().GetDataType(), ml_drift::DataType::FLOAT16);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

- (void)testCreateExternalFloatTensorF16FromF16 {
  self.continueAfterFailure = NO;
  ml_drift::ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ml_drift::ValueIdToSharedTensorMap quant_param_tensors;

  ml_drift::CreateGpuModelInfo create_info;
  create_info.precision = ml_drift::CalculationsPrecision::F16;
  create_info.storage_type = ml_drift::TensorStorageType::BUFFER;

  ml_drift::GraphFloat32 graph;
  ml_drift::Node* node = graph.NewNode();
  ml_drift::Value* input = graph.NewValue();
  ml_drift::Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = ml_drift::BHWC(1, 1, 1, 10);

  TfLiteContext context;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ml_drift::metal::MetalDevice metal_device(device);
  auto manager = ml_drift::MakeSharedMemoryManagerMetal(
      &metal_device, create_info, std::make_unique<ml_drift::GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_tflite_tensors=*/false,
      /*serialized_external_tensors=*/nullptr,
      /*madvise_original_tensors=*/false);

  int global_tensor_id = 0;
  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  tflite_tensor.type = TfLiteType::kTfLiteFloat16;
  std::vector<TfLiteFloat16> dummy(10);
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 10;
  tflite_tensor.data.f16 = dummy.data();
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = global_tensor_id;

  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 0);
  absl::flat_hash_map<ml_drift::ValueId, ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  XCTAssertTrue(
      manager
          ->RegisterExternalConstantTensors(input->id, shared_tflite_tensor, local_to_global_id_map)
          .ok());
  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 1);

  XCTAssertEqual(local_to_global_id_map.size(), 1);
  for (auto& [_, global_id] : local_to_global_id_map) {
    auto status_or_tensor = manager->GetExternalConstantTensor(global_id);
    XCTAssertTrue(status_or_tensor.ok());
    ml_drift::GpuSpatialTensor* external_tensor = *status_or_tensor;
    XCTAssertNotEqual(external_tensor, nullptr);
    XCTAssertEqual(external_tensor->GetDescriptor().GetDataType(), ml_drift::DataType::FLOAT16);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

- (void)testCreateExternalFloatTensorF32FromF16 {
  self.continueAfterFailure = NO;
  ml_drift::ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ml_drift::ValueIdToSharedTensorMap quant_param_tensors;

  ml_drift::CreateGpuModelInfo create_info;
  create_info.precision = ml_drift::CalculationsPrecision::F32;
  create_info.storage_type = ml_drift::TensorStorageType::BUFFER;

  ml_drift::GraphFloat32 graph;
  ml_drift::Node* node = graph.NewNode();
  ml_drift::Value* input = graph.NewValue();
  ml_drift::Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = ml_drift::BHWC(1, 1, 1, 10);
  input->tensor.type = ml_drift::DataType::FLOAT16;

  TfLiteContext context;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ml_drift::metal::MetalDevice metal_device(device);
  auto manager = ml_drift::MakeSharedMemoryManagerMetal(
      &metal_device, create_info, std::make_unique<ml_drift::GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_tflite_tensors=*/false,
      /*serialized_external_tensors=*/nullptr,
      /*madvise_original_tensors=*/false);

  int global_tensor_id = 0;
  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  tflite_tensor.type = TfLiteType::kTfLiteFloat16;
  std::vector<TfLiteFloat16> dummy(10);
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 10;
  tflite_tensor.data.f16 = dummy.data();
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = global_tensor_id;

  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 0);
  absl::flat_hash_map<ml_drift::ValueId, ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  XCTAssertTrue(
      manager
          ->RegisterExternalConstantTensors(input->id, shared_tflite_tensor, local_to_global_id_map)
          .ok());
  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 1);

  XCTAssertEqual(local_to_global_id_map.size(), 1);
  for (auto& [_, global_id] : local_to_global_id_map) {
    auto status_or_tensor = manager->GetExternalConstantTensor(global_id);
    XCTAssertTrue(status_or_tensor.ok());
    ml_drift::GpuSpatialTensor* external_tensor = *status_or_tensor;
    XCTAssertNotEqual(external_tensor, nullptr);
    XCTAssertEqual(external_tensor->GetDescriptor().GetDataType(), ml_drift::DataType::FLOAT16);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

- (void)testCreateExternalFloatTensorF32FromF32 {
  self.continueAfterFailure = NO;
  ml_drift::ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ml_drift::ValueIdToSharedTensorMap quant_param_tensors;

  ml_drift::CreateGpuModelInfo create_info;
  create_info.precision = ml_drift::CalculationsPrecision::F32;
  create_info.storage_type = ml_drift::TensorStorageType::BUFFER;

  ml_drift::GraphFloat32 graph;
  ml_drift::Node* node = graph.NewNode();
  ml_drift::Value* input = graph.NewValue();
  ml_drift::Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = ml_drift::BHWC(1, 1, 1, 10);
  input->tensor.type = ml_drift::DataType::FLOAT32;

  TfLiteContext context;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ml_drift::metal::MetalDevice metal_device(device);
  auto manager = ml_drift::MakeSharedMemoryManagerMetal(
      &metal_device, create_info, std::make_unique<ml_drift::GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_tflite_tensors=*/false,
      /*serialized_external_tensors=*/nullptr,
      /*madvise_original_tensors=*/false);

  int global_tensor_id = 0;
  TfLiteTensor tflite_tensor;
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  std::vector<float> dummy(10);
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 10;
  tflite_tensor.data.f = dummy.data();
  tflite_tensor.quantization.type = kTfLiteNoQuantization;
  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  context.tensors_size = 1;
  context.tensors = &tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = global_tensor_id;

  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 0);
  absl::flat_hash_map<ml_drift::ValueId, ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  XCTAssertTrue(
      manager
          ->RegisterExternalConstantTensors(input->id, shared_tflite_tensor, local_to_global_id_map)
          .ok());
  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 1);

  XCTAssertEqual(local_to_global_id_map.size(), 1);
  for (auto& [_, global_id] : local_to_global_id_map) {
    auto status_or_tensor = manager->GetExternalConstantTensor(global_id);
    XCTAssertTrue(status_or_tensor.ok());
    ml_drift::GpuSpatialTensor* external_tensor = *status_or_tensor;
    XCTAssertNotEqual(external_tensor, nullptr);
    XCTAssertEqual(external_tensor->GetDescriptor().GetDataType(), ml_drift::DataType::FLOAT32);
  }
  TfLiteIntArrayFree(tflite_tensor.dims);
}

- (void)testCreateQuantizedTensorMixedPrecision {
  self.continueAfterFailure = NO;
  ml_drift::ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ml_drift::ValueIdToSharedTensorMap quant_param_tensors;

  ml_drift::CreateGpuModelInfo create_info;
  create_info.precision = ml_drift::CalculationsPrecision::F32;
  create_info.storage_type = ml_drift::TensorStorageType::BUFFER;

  ml_drift::GraphFloat32 graph;
  ml_drift::Node* node = graph.NewNode();
  ml_drift::Value* input = graph.NewValue();
  ml_drift::Value* output = graph.NewValue();
  graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  input->tensor.shape = ml_drift::BHWC(1, 1, 1, 10);
  input->tensor.type = ml_drift::DataType::FLOAT16;

  TfLiteContext context;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ml_drift::metal::MetalDevice metal_device(device);
  auto manager = ml_drift::MakeSharedMemoryManagerMetal(
      &metal_device, create_info, std::make_unique<ml_drift::GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_tflite_tensors=*/false,
      /*serialized_external_tensors=*/nullptr,
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

  absl::flat_hash_map<ml_drift::ValueId, ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  XCTAssertTrue(
      manager
          ->RegisterExternalConstantTensors(input->id, shared_tflite_tensor, local_to_global_id_map)
          .ok());
  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 1);
  XCTAssertEqual(quant_param_tensors.size(), 2);

  bool scale_zp_found = false;
  for (auto& [_, global_id] : local_to_global_id_map) {
    if (global_id.IsParamId()) {
      scale_zp_found = true;
      auto status_or_tensor = manager->GetExternalConstantTensor(global_id);
      XCTAssertTrue(status_or_tensor.ok());
      ml_drift::GpuSpatialTensor* external_tensor = *status_or_tensor;
      XCTAssertNotEqual(external_tensor, nullptr);
      XCTAssertEqual(external_tensor->GetDescriptor().GetDataType(), ml_drift::DataType::FLOAT16);
    }
  }
  XCTAssertTrue(scale_zp_found);

  TfLiteIntArrayFree(quant_params->zero_point);
  TfLiteFloatArrayFree(quant_params->scale);
  delete quant_params;
  TfLiteIntArrayFree(tflite_tensor.dims);
}

- (void)runBlockwiseQuantizationTestWithZpType:(TfLiteType)zpType
                                        zpSize:(int)zpSize
                              expectedZpValues:(const std::vector<int>&)expectedZpValues
                                  useInvalidZp:(bool)useInvalidZp
                                 expectFailure:(bool)expectFailure {
  ml_drift::ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  ml_drift::ValueIdToSharedTensorMap quant_param_tensors;

  ml_drift::CreateGpuModelInfo create_info;
  create_info.precision = ml_drift::CalculationsPrecision::F16;
  create_info.storage_type = ml_drift::TensorStorageType::BUFFER;

  ml_drift::GraphFloat32 graph;
  ml_drift::Node* node = graph.NewNode();
  node->operation.type = "fully_connected";
  ml_drift::Value* weights_val = graph.NewValue();
  graph.AddConsumer(node->id, weights_val->id);
  weights_val->tensor.shape = ml_drift::BHWC(100, 1, 1, 64);

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
  quant_params->zero_point = useInvalidZp ? -1 : 2;
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

  tensors[2].type = zpType;
  tensors[2].dims = TfLiteIntArrayCreate(zpSize == 1 ? 1 : 2);
  if (zpSize == 1) {
    tensors[2].dims->data[0] = 1;
  } else {
    tensors[2].dims->data[0] = 100;
    tensors[2].dims->data[1] = 2;
  }

  if (zpType == kTfLiteInt32) {
    zp_data_i32.resize(zpSize, expectedZpValues[0]);
    if (zpSize > 1) {
      for (size_t i = 0; i < expectedZpValues.size(); ++i) {
        zp_data_i32[i] = expectedZpValues[i];
      }
    }
    tensors[2].data.i32 = zp_data_i32.data();
    tensors[2].bytes = zpSize * sizeof(int32_t);
  } else if (zpType == kTfLiteInt64) {
    zp_data_i64.resize(zpSize, expectedZpValues[0]);
    if (zpSize > 1) {
      for (size_t i = 0; i < expectedZpValues.size(); ++i) {
        zp_data_i64[i] = expectedZpValues[i];
      }
    }
    tensors[2].data.i64 = zp_data_i64.data();
    tensors[2].bytes = zpSize * sizeof(int64_t);
  } else if (zpType == kTfLiteInt8) {
    zp_data_i8.resize(zpSize, expectedZpValues[0]);
    if (zpSize > 1) {
      for (size_t i = 0; i < expectedZpValues.size(); ++i) {
        zp_data_i8[i] = expectedZpValues[i];
      }
    }
    tensors[2].data.int8 = zp_data_i8.data();
    tensors[2].bytes = zpSize * sizeof(int8_t);
  }

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ml_drift::metal::MetalDevice metal_device(device);
  auto manager = ml_drift::MakeSharedMemoryManagerMetal(
      &metal_device, create_info, std::make_unique<ml_drift::GraphFloat32Adapter>(graph), &context,
      buffer_id_to_spatial_tensor, quant_param_tensors,
      /*has_prepacked_external_tensors=*/false,
      /*serialized_external_tensors=*/nullptr,
      /*madvise_original_tensors=*/false);

  ::litert::ml_drift::SharedTfliteTensor shared_tflite_tensor;
  shared_tflite_tensor.tflite_tensor_id = 0;
  shared_tflite_tensor.global_id = 0;

  absl::flat_hash_map<ml_drift::ValueId, ml_drift::SharedMemoryManager::GlobalId>
      local_to_global_id_map;
  absl::Status status = manager->RegisterExternalConstantTensors(
      weights_val->id, shared_tflite_tensor, local_to_global_id_map);

  if (expectFailure) {
    XCTAssertFalse(status.ok());
    TfLiteIntArrayFree(tensors[0].dims);
    delete quant_params;
    TfLiteIntArrayFree(tensors[1].dims);
    TfLiteIntArrayFree(tensors[2].dims);
    return;
  }

  XCTAssertTrue(status.ok());

  XCTAssertEqual(buffer_id_to_spatial_tensor.size(), 1);
  XCTAssertEqual(quant_param_tensors.size(), 2);

  int params_found = 0;
  for (auto& [_, global_id] : local_to_global_id_map) {
    if (global_id.IsParamId()) {
      params_found++;
      auto status_or_tensor = manager->GetExternalConstantTensor(global_id);
      XCTAssertTrue(status_or_tensor.ok());
      ml_drift::GpuSpatialTensor* external_tensor = *status_or_tensor;
      XCTAssertNotEqual(external_tensor, nullptr);
      XCTAssertEqual(external_tensor->GetDescriptor().GetDataType(), ml_drift::DataType::FLOAT16);
      XCTAssertEqual(external_tensor->GetDescriptor().GetBHWCShape(), ml_drift::BHWC(1, 1, 25, 8));
    }
  }
  XCTAssertEqual(params_found, 2);

  TfLiteIntArrayFree(tensors[0].dims);
  delete quant_params;
  TfLiteIntArrayFree(tensors[1].dims);
  TfLiteIntArrayFree(tensors[2].dims);
}

- (void)testCreateBlockwiseQuantizedTensorWithInt32ZeroPoints {
  std::vector<int> expected_zp(200, 5);
  [self runBlockwiseQuantizationTestWithZpType:kTfLiteInt32
                                        zpSize:200
                              expectedZpValues:expected_zp
                                  useInvalidZp:false
                                 expectFailure:false];
}

- (void)testCreateBlockwiseQuantizedTensorFailsWithInt64ZeroPoints {
  std::vector<int> expected_zp(200, 5);
  [self runBlockwiseQuantizationTestWithZpType:kTfLiteInt64
                                        zpSize:200
                              expectedZpValues:expected_zp
                                  useInvalidZp:false
                                 expectFailure:true];
}

- (void)testCreateBlockwiseQuantizedTensorFailsWithInt8ZeroPoints {
  std::vector<int> expected_zp(200, 5);
  [self runBlockwiseQuantizationTestWithZpType:kTfLiteInt8
                                        zpSize:200
                              expectedZpValues:expected_zp
                                  useInvalidZp:false
                                 expectFailure:true];
}

- (void)testCreateBlockwiseQuantizedTensorWithBroadcastZeroPoints {
  std::vector<int> expected_zp(1, 5);
  [self runBlockwiseQuantizationTestWithZpType:kTfLiteInt32
                                        zpSize:1
                              expectedZpValues:expected_zp
                                  useInvalidZp:false
                                 expectFailure:false];
}

- (void)testCreateBlockwiseQuantizedTensorSymmetric {
  std::vector<int> expected_zp(200, 0);
  [self runBlockwiseQuantizationTestWithZpType:kTfLiteInt32
                                        zpSize:200
                              expectedZpValues:expected_zp
                                  useInvalidZp:true
                                 expectFailure:false];
}

@end
