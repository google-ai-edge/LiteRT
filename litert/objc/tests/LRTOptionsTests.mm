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

#import <XCTest/XCTest.h>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_gpu_options.h"
#import "third_party/odml/litert/litert/objc/apis/LRTOptions.h"
#import "third_party/odml/litert/litert/objc/sources/LRTOptions+Internal.h"

#if defined(__APPLE__)

namespace {

/**
 * Reads the Metal flags carried by the C++ options backing @c options.
 *
 * @return Whether the GPU options exist and both flags could be read.
 */
bool ReadMetalFlags(LRTOptions *options, bool *usesMetalArgumentBuffers,
                    bool *enablesMetalResidencySet) {
  auto gpuOptions = [options cppOptions]->GetGpuOptions();
  if (!gpuOptions.HasValue()) {
    return false;
  }
  return LrtGetGpuOptionsUseMetalArgumentBuffers(gpuOptions->Get(), usesMetalArgumentBuffers) ==
             kLiteRtStatusOk &&
         LrtGetGpuOptionsMetalResidencySet(gpuOptions->Get(), enablesMetalResidencySet) ==
             kLiteRtStatusOk;
}

}  // namespace

#endif  // defined(__APPLE__)

@interface LRTOptionsTests : XCTestCase
@end

@implementation LRTOptionsTests

- (void)testDefaultHardwareAcceleratorsIsNone {
  LRTOptions *options = [[LRTOptions alloc] init];
  XCTAssertNotNil(options);
  XCTAssertEqual(options.hardwareAccelerators, LRTHardwareAcceleratorNone);
}

- (void)testInitWithHardwareAccelerators {
  LRTHardwareAccelerators expectedFlags =
      LRTHardwareAcceleratorCPU | LRTHardwareAcceleratorNPU;
  LRTOptions *options =
      [[LRTOptions alloc] initWithHardwareAccelerators:expectedFlags];
  XCTAssertNotNil(options);
  XCTAssertEqual(options.hardwareAccelerators, expectedFlags);
}

- (void)testDefaultMetalOptionsAreDisabled {
  LRTOptions *options = [[LRTOptions alloc] init];
  XCTAssertNotNil(options);
  XCTAssertFalse(options.usesMetalArgumentBuffers);
  XCTAssertFalse(options.enablesMetalResidencySet);
}

- (void)testSetMetalOptions {
  LRTOptions *options = [[LRTOptions alloc] initWithHardwareAccelerators:LRTHardwareAcceleratorGPU];
  XCTAssertNotNil(options);
  options.usesMetalArgumentBuffers = YES;
  options.enablesMetalResidencySet = YES;
  XCTAssertTrue(options.usesMetalArgumentBuffers);
  XCTAssertTrue(options.enablesMetalResidencySet);
}

#if defined(__APPLE__)

- (void)testSetMetalOptionsUpdatesCppOptions {
  LRTOptions *options = [[LRTOptions alloc] initWithHardwareAccelerators:LRTHardwareAcceleratorGPU];
  options.usesMetalArgumentBuffers = YES;
  options.enablesMetalResidencySet = YES;

  bool usesMetalArgumentBuffers = false;
  bool enablesMetalResidencySet = false;
  XCTAssertTrue(ReadMetalFlags(options, &usesMetalArgumentBuffers, &enablesMetalResidencySet));
  XCTAssertTrue(usesMetalArgumentBuffers);
  XCTAssertTrue(enablesMetalResidencySet);
}

- (void)testResetMetalOptionsUpdatesCppOptions {
  LRTOptions *options = [[LRTOptions alloc] initWithHardwareAccelerators:LRTHardwareAcceleratorGPU];
  options.usesMetalArgumentBuffers = YES;
  options.enablesMetalResidencySet = YES;
  options.usesMetalArgumentBuffers = NO;
  options.enablesMetalResidencySet = NO;

  bool usesMetalArgumentBuffers = true;
  bool enablesMetalResidencySet = true;
  XCTAssertTrue(ReadMetalFlags(options, &usesMetalArgumentBuffers, &enablesMetalResidencySet));
  XCTAssertFalse(usesMetalArgumentBuffers);
  XCTAssertFalse(enablesMetalResidencySet);
}

#endif  // defined(__APPLE__)

@end
