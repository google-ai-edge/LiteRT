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

import XCTest
@testable import LiteRT

#if canImport(Metal)
import Metal
#endif

final class EnvironmentTests: XCTestCase {
  func testEnvironmentWithOptions() throws {
    let env = try Environment(options: [
      .compilerCacheMaxConfigsPerModel(5),
      .compilerCacheMaxTotalSize(Int64(1024 * 1024)),
      .autoRegisterAccelerators(1)
    ])
    XCTAssertNotNil(env)
  }

  func testEnvironmentCapabilities() throws {
    let env = try Environment()
    XCTAssertNotNil(env)
    let _ = try env.supportsFP16()
    let _ = env.hasGpuEnvironment()
    let _ = try env.supportsClGlInterop()
    let _ = try env.supportsAhwbClInterop()
    let _ = try env.supportsAhwbGlInterop()
  }

#if canImport(Metal)
  func testEnvironmentWithMetalOptions() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      print("Metal is not supported on this device/environment, skipping test.")
      return
    }

    guard let queue = device.makeCommandQueue() else {
      XCTFail("Failed to create MTLCommandQueue")
      return
    }

    let devicePtr = Unmanaged.passUnretained(device).toOpaque()
    let queuePtr = Unmanaged.passUnretained(queue).toOpaque()

    let env = try Environment(options: [
      .metalDevice(devicePtr),
      .metalCommandQueue(queuePtr)
    ])
    XCTAssertNotNil(env)
  }
#endif
}
