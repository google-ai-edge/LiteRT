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

import CLiteRT
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

  func testAutoRegisters() {
    let cpu = Int64(Accelerator.cpu.rawValue)
    let gpu = Int64(Accelerator.gpu.rawValue)

    // Every accelerator is registered unless `.autoRegisterAccelerators` is passed.
    XCTAssertTrue(Environment.autoRegisters(.gpu, options: []))
    XCTAssertTrue(Environment.autoRegisters(.npu, options: [.minLoggerSeverity(0)]))

    XCTAssertFalse(Environment.autoRegisters(.gpu, options: [.autoRegisterAccelerators(cpu)]))
    XCTAssertFalse(Environment.autoRegisters(.npu, options: [.autoRegisterAccelerators(cpu)]))
    XCTAssertTrue(Environment.autoRegisters(.gpu, options: [.autoRegisterAccelerators(cpu | gpu)]))
    XCTAssertFalse(Environment.autoRegisters(.npu, options: [.autoRegisterAccelerators(cpu | gpu)]))

    // An accelerator that any of several options includes counts as registered.
    XCTAssertTrue(
      Environment.autoRegisters(
        .gpu, options: [.autoRegisterAccelerators(cpu), .autoRegisterAccelerators(gpu)]))
  }

  func testMetalAcceleratorSearchRequiresGpuAutoRegistration() throws {
    // Puts a stand-in for the Metal accelerator where `Environment` searches for it: in a bundle
    // that has the identifier of the accelerator framework.
    let fileManager = FileManager.default
    let rootURL = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let frameworkURL = rootURL.appendingPathComponent("LiteRtMetalAccelerator.framework")
    try fileManager.createDirectory(at: frameworkURL, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: rootURL) }
    let identifier = "com.google.odml.litert.LiteRtMetalAccelerator"
    let infoPlist = try PropertyListSerialization.data(
      fromPropertyList: ["CFBundleIdentifier": identifier], format: .xml, options: 0)
    try infoPlist.write(to: frameworkURL.appendingPathComponent("Info.plist"))
    let acceleratorPath = frameworkURL.appendingPathComponent("LiteRtMetalAccelerator").path
    XCTAssertTrue(fileManager.createFile(atPath: acceleratorPath, contents: nil))
    // `Bundle(identifier:)` only finds a bundle while an instance of it exists.
    let framework = try XCTUnwrap(Bundle(url: frameworkURL))
    XCTAssertEqual(framework.bundleIdentifier, identifier)

    try withExtendedLifetime(framework) {
      // The GPU accelerator is registered by default, so the library is searched for.
      XCTAssertNotNil(try stringOption(kLiteRtEnvOptionTagRuntimeLibraryDir, of: Environment()))

      // A CPU-only environment never loads the library, so the search is skipped.
      let cpuOnlyEnvironment = try Environment(options: [
        .autoRegisterAccelerators(Int64(Accelerator.cpu.rawValue))
      ])
      XCTAssertNil(try stringOption(kLiteRtEnvOptionTagRuntimeLibraryDir, of: cpuOnlyEnvironment))
    }
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
