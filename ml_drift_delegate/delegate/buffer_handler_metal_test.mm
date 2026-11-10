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

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>

#include <malloc/malloc.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "ml_drift_delegate/delegate/buffer_handler_metal.h"
#include "ml_drift_delegate/delegate/custom_event_metal.h"
#include "ml_drift_delegate/delegate/kv_cache_metal.h"

@interface BufferHandlerMetalTest : XCTestCase
@end

@implementation BufferHandlerMetalTest {
  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;
  LiteRtEnvironment _environment;
}

- (void)setUp {
  [super setUp];
  _device = MTLCreateSystemDefaultDevice();
  if (!_device) {
    XCTSkip(@"Metal is not supported on this device");
    return;
  }
  _commandQueue = [_device newCommandQueue];

  LiteRtEnvOption options[2] = {};
  options[0].tag = kLiteRtEnvOptionTagMetalDevice;
  options[0].value.type = kLiteRtAnyTypeVoidPtr;
  options[0].value.ptr_value = (__bridge void*)_device;
  options[1].tag = kLiteRtEnvOptionTagMetalCommandQueue;
  options[1].value.type = kLiteRtAnyTypeVoidPtr;
  options[1].value.ptr_value = (__bridge void*)_commandQueue;

  LiteRtStatus status = LiteRtCreateEnvironment(2, options, &_environment);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)tearDown {
  if (_environment) {
    LiteRtDestroyEnvironment(_environment);
  }
  [super tearDown];
}

- (void)testCreateMetalMemory {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 2;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.dimensions[1] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(memoryInfo != nullptr);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testCreateMetalMemoryRequiresCommandQueue {
  XCTAssertNotNil(_device, @"Metal device should be initialized in setUp");

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  // Passing nullptr for queue_id should return an invalid argument error.
  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, nullptr, &tensorType,
                              kLiteRtTensorBufferTypeMetalBufferPacked, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusErrorInvalidArgument);
  XCTAssertTrue(memoryInfo == nullptr);
}

- (void)testImportMetalMemoryRequiresCommandQueue {
  XCTAssertNotNil(_device, @"Metal device should be initialized in setUp");

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  id<MTLBuffer> buffer = [_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  HwMemoryInfoPtr memoryInfo = nullptr;

  // Passing nullptr for queue_id should return an invalid argument error.
  LiteRtStatus status = LiteRtImportMetalMemory((__bridge void*)_device, nullptr, &tensorType,
                                                kLiteRtTensorBufferTypeMetalBuffer,
                                                (__bridge void*)buffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusErrorInvalidArgument);
  XCTAssertTrue(memoryInfo == nullptr);
}

- (void)testCustomEventMetalSignalingAndTimeout {
  XCTAssertNotNil(_device, @"Metal device should be initialized in setUp");
  XCTAssertNotNil(_commandQueue, @"Metal command queue should be initialized in setUp");

  auto event = std::make_unique<litert::ml_drift::CustomEventMetal>(_device);
  XCTAssertEqual(event->IsSignaled(event.get()), 1);

  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
  event->EncodeSignal(commandBuffer);
  // Before commit, event must not be reported as signaled.
  XCTAssertEqual(event->IsSignaled(event.get()), 0);

  // Wait with 0ms timeout on unsignaled event should return without clearing requirement.
  event->Wait(event.get(), /*timeout_in_ms=*/0);
  XCTAssertEqual(event->IsSignaled(event.get()), 0);

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  // Now wait should observe the signaled value and reset requirement.
  event->Wait(event.get(), /*timeout_in_ms=*/1000);
  XCTAssertEqual(event->IsSignaled(event.get()), 1);

  // Verify indefinite wait (-1) after signaling.
  id<MTLCommandBuffer> commandBuffer2 = [_commandQueue commandBuffer];
  event->EncodeSignal(commandBuffer2);
  XCTAssertEqual(event->IsSignaled(event.get()), 0);
  [commandBuffer2 commit];
  [commandBuffer2 waitUntilCompleted];
  event->Wait(event.get(), /*timeout_in_ms=*/-1);
  XCTAssertEqual(event->IsSignaled(event.get()), 1);
}

- (void)testCustomEventMetalEncodeWaitOrdersCommandBuffers {
  XCTAssertNotNil(_device, @"Metal device should be initialized in setUp");
  XCTAssertNotNil(_commandQueue, @"Metal command queue should be initialized in setUp");

  constexpr NSUInteger kBufferSize = 64 * 1024 * 1024;
  constexpr NSUInteger kResultSize = 16;
  constexpr uint8_t kFillValue = 0xAB;
  id<MTLBuffer> buffer = [_device newBufferWithLength:kBufferSize
                                              options:MTLResourceStorageModePrivate];
  id<MTLBuffer> result = [_device newBufferWithLength:kResultSize
                                              options:MTLResourceStorageModeShared];
  std::memset([result contents], 0, kResultSize);
  auto event = std::make_unique<litert::ml_drift::CustomEventMetal>(_device);

  // The producer fills the buffer on one queue and signals the event.
  id<MTLCommandBuffer> producer = [_commandQueue commandBuffer];
  id<MTLBlitCommandEncoder> fill = [producer blitCommandEncoder];
  [fill fillBuffer:buffer range:NSMakeRange(0, kBufferSize) value:kFillValue];
  [fill endEncoding];
  event->EncodeSignal(producer);
  [producer commit];

  // The consumer on another queue waits for the event before reading the end of the buffer.
  id<MTLCommandQueue> other_queue = [_device newCommandQueue];
  id<MTLCommandBuffer> consumer = [other_queue commandBuffer];
  event->EncodeWait(consumer);
  id<MTLBlitCommandEncoder> copy = [consumer blitCommandEncoder];
  [copy copyFromBuffer:buffer
           sourceOffset:kBufferSize - kResultSize
               toBuffer:result
      destinationOffset:0
                   size:kResultSize];
  [copy endEncoding];
  [consumer commit];
  [consumer waitUntilCompleted];

  XCTAssertEqual(consumer.status, MTLCommandBufferStatusCompleted);
  const uint8_t* data = static_cast<const uint8_t*>([result contents]);
  for (NSUInteger i = 0; i < kResultSize; ++i) {
    XCTAssertEqual(data[i], kFillValue);
  }
  event->Wait(event.get(), /*timeout_in_ms=*/-1);
  XCTAssertEqual(event->IsSignaled(event.get()), 1);
}

- (void)testLockUnlockMetalMemory {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(hostMemory != nullptr);

  float* floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    floatMemory[i] = (float)i;
  }

  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Read back.
  hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(hostMemory != nullptr);
  floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqualWithAccuracy(floatMemory[i], (float)i, 0.0001);
  }

  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testDoubleLockMetalMemoryShouldFail {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  void* hostMemory1 = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory1);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(hostMemory1 != nullptr);

  // Try to lock again.
  void* hostMemory2 = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory2);
  XCTAssertNotEqual(status, kLiteRtStatusOk);

  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testDoubleUnlockMetalMemoryShouldFail {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Try to unlock without locking first.
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertNotEqual(status, kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testImportMetalMemory {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;
  size_t bytes = 4 * sizeof(float);

  id<MTLBuffer> buffer = [_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  HwMemoryInfoPtr memoryInfo = nullptr;

  LiteRtStatus status = LiteRtImportMetalMemory(
      (__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
      kLiteRtTensorBufferTypeMetalBuffer, (__bridge void*)buffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(memoryInfo != nullptr);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testClearMetalMemoryBuffer {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Write some data.
  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  float* floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    floatMemory[i] = (float)(i + 1);
  }
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Clear memory.
  status = LiteRtClearMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Read back and verify it is zero.
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqual(floatMemory[i], 0.0f);
  }
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testClearMetalMemoryTexture {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalTexture, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Write some data.
  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  float* floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    floatMemory[i] = (float)(i + 1);
  }
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Clear memory.
  status = LiteRtClearMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Read back and verify it is zero.
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqual(floatMemory[i], 0.0f);
  }
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testImportMetalMemoryWithPrivateStorageMode {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;
  size_t bytes = 4 * sizeof(float);

  // Create a private buffer.
  id<MTLBuffer> buffer = [_device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  HwMemoryInfoPtr memoryInfo = nullptr;

  LiteRtStatus status = LiteRtImportMetalMemory(
      (__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
      kLiteRtTensorBufferTypeMetalBufferPacked, (__bridge void*)buffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(memoryInfo != nullptr);

  // Write data to the private buffer via Lock/Unlock (which triggers WriteDataToBuffer).
  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(hostMemory != nullptr);

  float* floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    floatMemory[i] = (float)(i + 10);
  }

  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Read back.
  hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(hostMemory != nullptr);
  floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqualWithAccuracy(floatMemory[i], (float)(i + 10), 0.0001);
  }

  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

#if TARGET_OS_OSX
- (void)testImportMetalMemoryWithManagedStorageMode {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;
  size_t bytes = 4 * sizeof(float);

  // Create a managed buffer.
  id<MTLBuffer> buffer = [_device newBufferWithLength:bytes options:MTLResourceStorageModeManaged];
  HwMemoryInfoPtr memoryInfo = nullptr;

  LiteRtStatus status = LiteRtImportMetalMemory(
      (__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
      kLiteRtTensorBufferTypeMetalBufferPacked, (__bridge void*)buffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(memoryInfo != nullptr);

  // Write data.
  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(hostMemory != nullptr);

  float* floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    floatMemory[i] = (float)(i + 20);
  }

  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Read back.
  hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(hostMemory != nullptr);
  floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqualWithAccuracy(floatMemory[i], (float)(i + 20), 0.0001);
  }

  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}
#endif

- (void)testCopyKvCacheMetalReduce {
  if (!_device) return;

  constexpr int kSrcBatchSize = 2;
  constexpr int kDstBatchSize = 1;
  constexpr int kElementsPerBatch = 4;

  LiteRtRankedTensorType srcTensorType;
  srcTensorType.element_type = kLiteRtElementTypeFloat32;
  srcTensorType.layout.rank = 2;
  srcTensorType.layout.dimensions[0] = kSrcBatchSize;
  srcTensorType.layout.dimensions[1] = kElementsPerBatch;
  srcTensorType.layout.has_strides = false;

  LiteRtRankedTensorType dstTensorType;
  dstTensorType.element_type = kLiteRtElementTypeFloat32;
  dstTensorType.layout.rank = 2;
  dstTensorType.layout.dimensions[0] = kDstBatchSize;
  dstTensorType.layout.dimensions[1] = kElementsPerBatch;
  dstTensorType.layout.has_strides = false;

  size_t srcBytes = kSrcBatchSize * kElementsPerBatch * sizeof(float);
  size_t dstBytes = kDstBatchSize * kElementsPerBatch * sizeof(float);

  HwMemoryInfoPtr srcMemoryInfo = nullptr;
  HwMemoryInfoPtr dstMemoryInfo = nullptr;

  LiteRtStatus status = LiteRtCreateMetalMemory(
      (__bridge void*)_device, (__bridge void*)_commandQueue, &srcTensorType,
      kLiteRtTensorBufferTypeMetalBuffer, srcBytes, srcBytes, &srcMemoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue,
                                   &dstTensorType, kLiteRtTensorBufferTypeMetalBuffer, dstBytes,
                                   dstBytes, &dstMemoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Write data to source.
  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(srcMemoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  float* floatMemory = (float*)hostMemory;
  for (int i = 0; i < 8; ++i) {
    floatMemory[i] = (float)(i + 1);  // 1, 2, 3, 4, 5, 6, 7, 8
  }
  status = LiteRtUnlockMetalMemory(srcMemoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Copy index 1 (second batch: 5, 6, 7, 8).
  status = LiteRtCopyKvCacheMetal(srcMemoryInfo->raw_handle, dstMemoryInfo->raw_handle,
                                  /*src_index_to_copy_on_prefill=*/1,
                                  /*decode_batch_size=*/2, srcBytes, dstBytes,
                                  (__bridge void*)_commandQueue);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Wait for the command queue to finish the copy.
  id<MTLCommandBuffer> command_buffer = [_commandQueue commandBuffer];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  // Read back destination.
  hostMemory = nullptr;
  status = LiteRtLockMetalMemory(dstMemoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  floatMemory = (float*)hostMemory;
  XCTAssertEqual(floatMemory[0], 5.0f);
  XCTAssertEqual(floatMemory[1], 6.0f);
  XCTAssertEqual(floatMemory[2], 7.0f);
  XCTAssertEqual(floatMemory[3], 8.0f);

  status = LiteRtUnlockMetalMemory(dstMemoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  LiteRtDestroyMetalMemory(srcMemoryInfo);
  LiteRtDestroyMetalMemory(dstMemoryInfo);
}

- (void)testCopyKvCacheMetalBroadcast {
  if (!_device) return;

  constexpr int kSrcBatchSize = 1;
  constexpr int kDstBatchSize = 2;
  constexpr int kElementsPerBatch = 4;

  LiteRtRankedTensorType srcTensorType;
  srcTensorType.element_type = kLiteRtElementTypeFloat32;
  srcTensorType.layout.rank = 2;
  srcTensorType.layout.dimensions[0] = kSrcBatchSize;
  srcTensorType.layout.dimensions[1] = kElementsPerBatch;
  srcTensorType.layout.has_strides = false;

  LiteRtRankedTensorType dstTensorType;
  dstTensorType.element_type = kLiteRtElementTypeFloat32;
  dstTensorType.layout.rank = 2;
  dstTensorType.layout.dimensions[0] = kDstBatchSize;
  dstTensorType.layout.dimensions[1] = kElementsPerBatch;
  dstTensorType.layout.has_strides = false;

  size_t srcBytes = kSrcBatchSize * kElementsPerBatch * sizeof(float);
  size_t dstBytes = kDstBatchSize * kElementsPerBatch * sizeof(float);

  HwMemoryInfoPtr srcMemoryInfo = nullptr;
  HwMemoryInfoPtr dstMemoryInfo = nullptr;

  LiteRtStatus status = LiteRtCreateMetalMemory(
      (__bridge void*)_device, (__bridge void*)_commandQueue, &srcTensorType,
      kLiteRtTensorBufferTypeMetalBuffer, srcBytes, srcBytes, &srcMemoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue,
                                   &dstTensorType, kLiteRtTensorBufferTypeMetalBuffer, dstBytes,
                                   dstBytes, &dstMemoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Write data to source.
  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(srcMemoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  float* floatMemory = (float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    floatMemory[i] = (float)(i + 10);  // 10, 11, 12, 13
  }
  status = LiteRtUnlockMetalMemory(srcMemoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Broadcast (src_index = -1).
  status = LiteRtCopyKvCacheMetal(srcMemoryInfo->raw_handle, dstMemoryInfo->raw_handle,
                                  /*src_index_to_copy_on_prefill=*/-1,
                                  /*decode_batch_size=*/2, srcBytes, dstBytes,
                                  (__bridge void*)_commandQueue);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // Wait for the command queue to finish.
  id<MTLCommandBuffer> command_buffer = [_commandQueue commandBuffer];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  // Read back destination.
  hostMemory = nullptr;
  status = LiteRtLockMetalMemory(dstMemoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  floatMemory = (float*)hostMemory;
  XCTAssertEqual(floatMemory[0], 10.0f);
  XCTAssertEqual(floatMemory[1], 11.0f);
  XCTAssertEqual(floatMemory[2], 12.0f);
  XCTAssertEqual(floatMemory[3], 13.0f);
  XCTAssertEqual(floatMemory[4], 10.0f);
  XCTAssertEqual(floatMemory[5], 11.0f);
  XCTAssertEqual(floatMemory[6], 12.0f);
  XCTAssertEqual(floatMemory[7], 13.0f);

  status = LiteRtUnlockMetalMemory(dstMemoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  LiteRtDestroyMetalMemory(srcMemoryInfo);
  LiteRtDestroyMetalMemory(dstMemoryInfo);
}

- (void)testCreateDestroyMetalMemoryLeak {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);

  for (int i = 0; i < 500; ++i) {
    HwMemoryInfoPtr memoryInfo = nullptr;
    LiteRtStatus status =
        LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                                kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
    XCTAssertEqual(status, kLiteRtStatusOk);
    LiteRtDestroyMetalMemory(memoryInfo);
  }

#if TARGET_OS_OSX
  pid_t pid = getpid();
  std::string cmd = "leaks " + std::to_string(pid) + " > /dev/null";
  int exit_code = std::system(cmd.c_str());
  XCTAssertEqual(exit_code, 0, @"leaks tool detected memory leaks!");
#endif
}

// Copies `bytes` bytes of `src` into a new shared buffer on the GPU and waits,
// so the result reflects what GPU work observes.
- (id<MTLBuffer>)gpuCopyOfBuffer:(id<MTLBuffer>)src bytes:(size_t)bytes {
  id<MTLBuffer> staging = [_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
  [blit copyFromBuffer:src sourceOffset:0 toBuffer:staging destinationOffset:0 size:bytes];
  [blit endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  return staging;
}

- (void)testUnlockWritesSharedMetalBufferInPlace {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 4;
  tensorType.layout.dimensions[0] = 1;
  tensorType.layout.dimensions[1] = 1;
  tensorType.layout.dimensions[2] = 1;
  tensorType.layout.dimensions[3] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;
  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)memoryInfo->raw_handle;
  XCTAssertNotNil(buffer);
  XCTAssertEqual(buffer.storageMode, MTLStorageModeShared);

  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  const float values[4] = {1.0f, -2.5f, 3.0f, 4.25f};
  std::memcpy(hostMemory, values, bytes);
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // The data is in the buffer as soon as Unlock returns...
  const float* contents = (const float*)[buffer contents];
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqual(contents[i], values[i]);
  }
  // ...and visible to GPU work submitted afterwards.
  id<MTLBuffer> gpuCopy = [self gpuCopyOfBuffer:buffer bytes:bytes];
  const float* copied = (const float*)[gpuCopy contents];
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqual(copied[i], values[i]);
  }

  // A second write overwrites the first one.
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  const float values2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  std::memcpy(hostMemory, values2, bytes);
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqual(contents[i], values2[i]);
  }

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testUnlockConvertsToFp16MetalBufferInPlace {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 4;
  tensorType.layout.dimensions[0] = 1;
  tensorType.layout.dimensions[1] = 1;
  tensorType.layout.dimensions[2] = 1;
  tensorType.layout.dimensions[3] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;
  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBufferFp16, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)memoryInfo->raw_handle;
  XCTAssertNotNil(buffer);

  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  const float values[4] = {0.5f, -1.25f, 2.0f, 3.75f};
  std::memcpy(hostMemory, values, bytes);
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // IEEE half bit patterns of `values`, as seen by the GPU.
  const uint16_t expected_halfs[4] = {0x3800, 0xBD00, 0x4000, 0x4380};
  id<MTLBuffer> gpuCopy = [self gpuCopyOfBuffer:buffer bytes:4 * sizeof(uint16_t)];
  const uint16_t* halfs = (const uint16_t*)[gpuCopy contents];
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqual(halfs[i], expected_halfs[i]);
  }

  hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  const float* floatMemory = (const float*)hostMemory;
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqual(floatMemory[i], values[i]);
  }
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testUnlockPadsChannelsInPlace {
  if (!_device) return;

  // 3 channels are padded to a 4-channel slice on the GPU.
  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 4;
  tensorType.layout.dimensions[0] = 1;
  tensorType.layout.dimensions[1] = 1;
  tensorType.layout.dimensions[2] = 2;
  tensorType.layout.dimensions[3] = 3;
  tensorType.layout.has_strides = false;

  size_t bytes = 6 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;
  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  float* floatMemory = (float*)hostMemory;
  for (int i = 0; i < 6; ++i) {
    floatMemory[i] = (float)(i + 7);
  }
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  floatMemory = (float*)hostMemory;
  for (int i = 0; i < 6; ++i) {
    XCTAssertEqual(floatMemory[i], (float)(i + 7));
  }
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testUnlockUploadsToPrivateMetalBuffer {
  if (!_device) return;

  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 4;
  tensorType.layout.dimensions[0] = 1;
  tensorType.layout.dimensions[1] = 1;
  tensorType.layout.dimensions[2] = 1;
  tensorType.layout.dimensions[3] = 4;
  tensorType.layout.has_strides = false;
  size_t bytes = 4 * sizeof(float);

  // A private buffer is not CPU-visible, so Unlock must upload through the GPU.
  id<MTLBuffer> buffer = [_device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  HwMemoryInfoPtr memoryInfo = nullptr;
  LiteRtStatus status = LiteRtImportMetalMemory(
      (__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
      kLiteRtTensorBufferTypeMetalBuffer, (__bridge void*)buffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  const float values[4] = {10.0f, 11.0f, 12.0f, 13.0f};
  std::memcpy(hostMemory, values, bytes);
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  id<MTLBuffer> gpuCopy = [self gpuCopyOfBuffer:buffer bytes:bytes];
  const float* copied = (const float*)[gpuCopy contents];
  for (int i = 0; i < 4; ++i) {
    XCTAssertEqual(copied[i], values[i]);
  }

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testPartialWriteLockPreservesUnwrittenContents {
  if (!_device) return;

  constexpr int kNumElements = 8;
  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = kNumElements;
  tensorType.layout.has_strides = false;

  const size_t bytes = kNumElements * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;
  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  // First write lock initializes the whole buffer (like InitializeAttentionMask()).
  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  float* floatMemory = static_cast<float*>(hostMemory);
  for (int i = 0; i < kNumElements; ++i) {
    floatMemory[i] = -1.0f;
  }
  XCTAssertEqual(LiteRtUnlockMetalMemory(memoryInfo), kLiteRtStatusOk);

  // Second write lock updates only the first element (like FillAttentionMask()).
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  static_cast<float*>(hostMemory)[0] = 0.0f;
  XCTAssertEqual(LiteRtUnlockMetalMemory(memoryInfo), kLiteRtStatusOk);

  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeRead, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  floatMemory = static_cast<float*>(hostMemory);
  XCTAssertEqual(floatMemory[0], 0.0f);
  for (int i = 1; i < kNumElements; ++i) {
    XCTAssertEqual(floatMemory[i], -1.0f);
  }
  XCTAssertEqual(LiteRtUnlockMetalMemory(memoryInfo), kLiteRtStatusOk);

  status = LiteRtDestroyMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
}

- (void)testLockUnlockFreesShadowHostMemory {
  if (!_device) return;

  constexpr int kNumElements = 4 * 1024 * 1024;  // 16 MB
  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = kNumElements;
  tensorType.layout.has_strides = false;

  const size_t bytes = kNumElements * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;
  LiteRtStatus status =
      LiteRtCreateMetalMemory((__bridge void*)_device, (__bridge void*)_commandQueue, &tensorType,
                              kLiteRtTensorBufferTypeMetalBuffer, bytes, bytes, &memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(memoryInfo != nullptr);

  malloc_statistics_t statsBefore = {};
  malloc_zone_statistics(nullptr, &statsBefore);

  void* hostMemory = nullptr;
  status = LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite, &hostMemory);
  XCTAssertEqual(status, kLiteRtStatusOk);
  XCTAssertTrue(hostMemory != nullptr);
  std::memset(hostMemory, 0xAB, bytes);

  malloc_statistics_t statsLocked = {};
  malloc_zone_statistics(nullptr, &statsLocked);
  XCTAssertGreaterThanOrEqual(statsLocked.size_in_use, statsBefore.size_in_use + bytes / 2);

  // Unlocking must free the aligned host shadow buffer while memoryInfo remains alive.
  status = LiteRtUnlockMetalMemory(memoryInfo);
  XCTAssertEqual(status, kLiteRtStatusOk);

  malloc_statistics_t statsUnlocked = {};
  malloc_zone_statistics(nullptr, &statsUnlocked);
  XCTAssertLessThanOrEqual(statsUnlocked.size_in_use + bytes / 2, statsLocked.size_in_use);

  // Repeated lock/unlock cycles on a live buffer should not accumulate host allocations.
  for (int i = 0; i < 10; ++i) {
    void* ptr = nullptr;
    XCTAssertEqual(LiteRtLockMetalMemory(memoryInfo, kLiteRtTensorBufferLockModeRead, &ptr),
                   kLiteRtStatusOk);
    XCTAssertTrue(ptr != nullptr);
    XCTAssertEqual(LiteRtUnlockMetalMemory(memoryInfo), kLiteRtStatusOk);
  }

  malloc_statistics_t statsAfterLoop = {};
  malloc_zone_statistics(nullptr, &statsAfterLoop);
  XCTAssertLessThanOrEqual(statsAfterLoop.size_in_use, statsBefore.size_in_use + bytes / 2);

  XCTAssertEqual(LiteRtDestroyMetalMemory(memoryInfo), kLiteRtStatusOk);
}
@end
