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

#include <unistd.h>
#include <cstdlib>
#include <string>

#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_metal.h"
#include "third_party/odml/litert/ml_drift/delegate/kv_cache_metal.h"

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

@end
