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

#import "third_party/odml/litert/litert/c/litert_tensor_buffer.h"
#import <Metal/Metal.h>
#import <XCTest/XCTest.h>
#import <XCTest/XCTestAssertions.h>
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_layout.h"

@interface LitertTensorBufferTest : XCTestCase
@end

@implementation LitertTensorBufferTest

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) / sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32, ::litert::BuildLayout(kTensorDimensions)};

- (void)testMetalBuffer {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  const void* kTestPtr = (__bridge void*)(device);

  id<MTLCommandQueue> commandQueue = [device newCommandQueue];
  const void* kCommandQueuePtr = (__bridge void*)(commandQueue);

  auto status = litert::ToLiteRtAny(litert::LiteRtVariant(kTestPtr));
  auto null_deivce_id = &status;

  auto command_queue_status = litert::ToLiteRtAny(litert::LiteRtVariant(kCommandQueuePtr));
  auto null_command_queue_id = &command_queue_status;

  const std::array<LiteRtEnvOption, 2> environment_options = {
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagMetalDevice,
          /*.value=*/null_deivce_id->Value(),
      },
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagMetalCommandQueue,
          /*.value=*/null_command_queue_id->Value(),
      },
  };
  LiteRtEnvironment env;
  XCTAssertTrue(LiteRtCreateEnvironment(environment_options.size(), environment_options.data(),
                                        &env) == kLiteRtStatusOk);

  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeMetalBuffer;
  LiteRtTensorBuffer tensor_buffer;
  XCTAssertTrue(LiteRtCreateManagedTensorBuffer(env, kTensorBufferType, &kTensorType,
                                                sizeof(kTensorData),
                                                &tensor_buffer) == kLiteRtStatusOk);
  LiteRtTensorBufferType buffer_type;
  XCTAssertTrue(LiteRtGetTensorBufferType(tensor_buffer, &buffer_type) == kLiteRtStatusOk);
  LITERT_LOG(LITERT_INFO, "buffer_type: %d", buffer_type);
  XCTAssertTrue(buffer_type == kLiteRtTensorBufferTypeMetalBuffer);

  LiteRtRankedTensorType tensor_type;
  XCTAssertTrue(LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type) == kLiteRtStatusOk);
  XCTAssertEqual(tensor_type.element_type, kLiteRtElementTypeFloat32);
  XCTAssertEqual(tensor_type.layout.rank, 1);
  XCTAssertEqual(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  XCTAssertEqual(tensor_type.layout.has_strides, false);

  size_t size;
  XCTAssertTrue(LiteRtGetTensorBufferSize(tensor_buffer, &size) == kLiteRtStatusOk);
  XCTAssertEqual(size, sizeof(kTensorData));

  size_t offset;
  XCTAssertTrue(LiteRtGetTensorBufferOffset(tensor_buffer, &offset) == kLiteRtStatusOk);
  XCTAssertEqual(offset, 0);

  void* host_memory_ptr;
  XCTAssertTrue(LiteRtLockTensorBuffer(tensor_buffer, &host_memory_ptr,
                                       kLiteRtTensorBufferLockModeWrite) == kLiteRtStatusOk);
  std::memcpy(host_memory_ptr, kTensorData, sizeof(kTensorData));
  XCTAssertTrue(LiteRtUnlockTensorBuffer(tensor_buffer) == kLiteRtStatusOk);

  XCTAssertTrue(LiteRtLockTensorBuffer(tensor_buffer, &host_memory_ptr,
                                       kLiteRtTensorBufferLockModeRead) == kLiteRtStatusOk);

  XCTAssertEqual(std::memcmp(host_memory_ptr, kTensorData, sizeof(kTensorData)), 0);
  XCTAssertTrue(LiteRtUnlockTensorBuffer(tensor_buffer) == kLiteRtStatusOk);

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

@end
