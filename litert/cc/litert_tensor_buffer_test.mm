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

#import "third_party/odml/litert/litert/cc/litert_tensor_buffer.h"

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>
#import <XCTest/XCTestAssertions.h>

#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_platform_support.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#import "third_party/odml/litert/litert/test/metal_test_helper.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#import "third_party/tensorflow/lite/delegates/gpu/metal/metal_device.h"

@interface LitertTensorBufferTest : XCTestCase
@end

@implementation LitertTensorBufferTest

constexpr const float kTensorData[] = {10, 20, 30, 40};
constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) / sizeof(kTensorData[0])};
constexpr const LiteRtRankedTensorType kTestTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32, ::litert::BuildLayout(kTensorDimensions)};
const float kTolerance = 1e-5;


// Create LiteRt environment with metal options.
- (litert::Environment)createEnvironmentWithMetalDevice:
    (tflite::gpu::metal::MetalDevice *)metal_device {
  std::vector<litert::Environment::Option> environment_options;
  environment_options.push_back({.tag = litert::Environment::OptionTag::MetalDevice,
                                 .value = (__bridge const void *)(metal_device->device())});

  id<MTLCommandQueue> command_queue = [metal_device->device() newCommandQueue];
  environment_options.push_back({.tag = litert::Environment::OptionTag::MetalCommandQueue,
                                 .value = (__bridge const void *)(command_queue)});
  auto env = litert::Environment::Create(environment_options);
  XCTAssertTrue(env);
  return std::move(*env);
}

- (void)testTensorBufferMetalMemory {
  XCTAssertTrue(litert::HasMetalSupport());

  auto metal_device = tflite::gpu::metal::MetalDevice();
  litert::Environment env = [self createEnvironmentWithMetalDevice:&metal_device];

  const litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = litert::TensorBufferType::kMetalBuffer;

  auto tensor_buffer =
      litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType, sizeof(kTensorData));

  auto tensor_buffer_type = tensor_buffer->BufferType();
  XCTAssertTrue(tensor_buffer_type);
  XCTAssertEqual(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  XCTAssertTrue(tensor_type);

  XCTAssertEqual(tensor_type->ElementType(), litert::ElementType::Float32);
  XCTAssertEqual(tensor_type->Layout().Rank(), 1);
  XCTAssertEqual(tensor_type->Layout().Dimensions()[0], kTensorType.Layout().Dimensions()[0]);
  XCTAssertFalse(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  XCTAssertTrue(size);
  XCTAssertEqual(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  XCTAssertTrue(offset);
  XCTAssertEqual(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(
        *tensor_buffer, litert::TensorBuffer::LockMode::kWrite);
    XCTAssertTrue(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(
        *tensor_buffer, litert::TensorBuffer::LockMode::kRead);
    XCTAssertTrue(lock_and_addr);
    XCTAssertEqual(std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)), 0);
  }
}

// Create a TensorBuffer from the native Metal buffer for the given input index.
//
// @param input_index The index of the input tensor in the compiled_model.
// @param env The LiteRt environment.
// @param metal_device The Metal device.
// @param compiled_model The compiled model.
// @return The created TensorBuffer.
- (litert::TensorBuffer)createManagedTensorBufferForInput:(int)input_index
                                          withEnvironment:(litert::Environment *)env
                                          withMetalDevice:
                                              (tflite::gpu::metal::MetalDevice *)metal_device
                                        withCompiledModel:(litert::CompiledModel *)compiled_model {
  auto input_tensor_type = compiled_model->GetInputTensorType(
      /*signature_index=*/0, input_index);
  auto bytes = input_tensor_type->Bytes();
  // Create a native Metal buffer.
  id<MTLBuffer> metal_buffer =
      [metal_device->device() newBufferWithLength:*bytes options:MTLResourceStorageModeShared];
  // Create a TensorBuffer from the native Metal buffer.
  auto tensor_buffer = litert::TensorBuffer::CreateFromMetalBuffer(
      *env, *input_tensor_type, litert::TensorBufferType::kMetalBufferPacked,
      (__bridge void *)metal_buffer, *bytes);
  XCTAssertTrue(tensor_buffer);
  return std::move(*tensor_buffer);
}

- (void)testTensorBufferCreateFromMetalBuffer {
  XCTAssertTrue(litert::HasMetalSupport());

  auto metal_device = tflite::gpu::metal::MetalDevice();
  litert::Environment env = [self createEnvironmentWithMetalDevice:&metal_device];

  NSString *modelFilePath = [MetalTestHelper pathForModelName:@"simple_model"];
  XCTAssertNotNil(modelFilePath);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(env, modelFilePath.UTF8String, litert::HwAccelerators::kGpu));

  litert::TensorBuffer tensor_buffer0 = [self createManagedTensorBufferForInput:0
                                                                withEnvironment:&env
                                                                withMetalDevice:&metal_device
                                                              withCompiledModel:&compiled_model];
  litert::TensorBuffer tensor_buffer1 = [self createManagedTensorBufferForInput:1
                                                                withEnvironment:&env
                                                                withMetalDevice:&metal_device
                                                              withCompiledModel:&compiled_model];

  XCTAssertTrue(tensor_buffer0.IsMetalMemory());
  XCTAssertTrue(
      tensor_buffer0.Write<float>(absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));

  XCTAssertTrue(tensor_buffer0.IsMetalMemory());
  XCTAssertTrue(
      tensor_buffer1.Write<float>(absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  std::vector<litert::TensorBuffer> input_buffers;
  input_buffers.push_back(std::move(tensor_buffer0));
  input_buffers.push_back(std::move(tensor_buffer1));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers, compiled_model.CreateOutputBuffers());

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  XCTAssertTrue(output_buffers[0].IsMetalMemory());
  [MetalTestHelper checkTensorBufferFloatOutput:&output_buffers[0]
                             withExpectedOutput:kTestOutputTensor
                               withElementCount:kTestOutputSize
                                  withTolerance:kTolerance];
}

@end
