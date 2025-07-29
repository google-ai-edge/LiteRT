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
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_platform_support.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/runtime/metal_memory.h"
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
  XCTAssertTrue(litert::internal::MetalMemory::IsSupported());

  // auto metal_device = std::make_unique<tflite::gpu::metal::MetalDevice>();
  auto metal_device = tflite::gpu::metal::MetalDevice();
  litert::Environment::Option options = {.tag = litert::Environment::OptionTag::MetalDevice,
                                         .value = (__bridge const void *)(metal_device.device())};
  // Create LiteRt environment from metal options.
  std::vector<litert::Environment::Option> options_vector = {
      {.tag = litert::Environment::OptionTag::MetalDevice,
       .value = (__bridge const void *)(metal_device.device())}};
  auto env = litert::Environment::Create(options_vector);

  const litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeMetalBuffer;

  auto tensor_buffer = litert::TensorBuffer::CreateManaged(env->Get(), kTensorBufferType,
                                                           kTensorType, sizeof(kTensorData));

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

@end
