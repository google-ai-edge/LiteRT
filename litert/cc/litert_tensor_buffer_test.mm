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
#import <XCTest/XCTest.h>
#import <XCTest/XCTestAssertions.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_platform_support.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/runtime/tensor_buffer.h"
#import "third_party/tensorflow/lite/delegates/gpu/metal/metal_device.h"

@interface LitertTensorBufferTest : XCTestCase
@end

@implementation LitertTensorBufferTest

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) / sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTestTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32, ::litert::BuildLayout(kTensorDimensions)};

- (void)testTensorBufferMetalMemory {
  XCTAssertTrue(litert::HasMetalSupport());

  auto metal_device = tflite::gpu::metal::MetalDevice();
  // Create LiteRt environment from metal options.
  std::vector<litert::Environment::Option> environment_options;
  environment_options.push_back({.tag = litert::Environment::OptionTag::MetalDevice,
                                 .value = (__bridge const void *)(metal_device.device())});

  id<MTLCommandQueue> command_queue = [metal_device.device() newCommandQueue];

  environment_options.push_back({.tag = litert::Environment::OptionTag::MetalCommandQueue,
                                 .value = (__bridge const void *)(command_queue)});
  auto env = litert::Environment::Create(environment_options);

  const litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = litert::TensorBufferType::kMetalBuffer;

  auto tensor_buffer = litert::TensorBuffer::CreateManaged(*env, kTensorBufferType, kTensorType,
                                                           sizeof(kTensorData));

  auto tensor_buffer_type = tensor_buffer->BufferTypeCC();
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

- (void)testTensorBufferCreateFromMetalMemory {
  XCTAssertTrue(litert::HasMetalSupport());

  auto metal_device = tflite::gpu::metal::MetalDevice();
  // Create LiteRt environment from metal options.
  std::vector<litert::Environment::Option> environment_options;
  environment_options.push_back({.tag = litert::Environment::OptionTag::MetalDevice,
                                 .value = (__bridge const void *)(metal_device.device())});

  id<MTLCommandQueue> command_queue = [metal_device.device() newCommandQueue];

  environment_options.push_back({.tag = litert::Environment::OptionTag::MetalCommandQueue,
                                 .value = (__bridge const void *)(command_queue)});
  auto env = litert::Environment::Create(environment_options);

  const litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = litert::TensorBufferType::kMetalBuffer;

  // Create a managed buffer
  auto tensor_buffer = litert::TensorBuffer::CreateManaged(*env, kTensorBufferType, kTensorType,
                                                           sizeof(kTensorData));
  XCTAssertTrue(tensor_buffer);

  // Get the native handle from the managed buffer.
  auto metal_buffer = tensor_buffer->GetMetalBuffer();
  XCTAssertTrue(metal_buffer);

  // Create a tensor buffer from the existing metal buffer.
  auto metal_buffer_created = litert::TensorBuffer::CreateFromMetalBuffer(
      *env, kTensorType, kTensorBufferType, *metal_buffer, sizeof(kTensorData));
  XCTAssertTrue(metal_buffer_created);

  // Check properties of the wrapped buffer
  auto tensor_buffer_type = metal_buffer_created->BufferTypeCC();
  XCTAssertTrue(tensor_buffer_type);
  XCTAssertEqual(*tensor_buffer_type, kTensorBufferType);

  // Check that the wrapped buffer has the same native handle as the original buffer.
  auto metal_buffer_new_ptr = metal_buffer_created->GetMetalBuffer();
  XCTAssertTrue(metal_buffer_new_ptr);
  XCTAssertEqual(*metal_buffer_new_ptr, *metal_buffer);
}

@end
