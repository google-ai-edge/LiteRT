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

final class OptionsTests: XCTestCase {
  func testExternalTensorBinding() throws {
    let options = try Options()
    var rawData: [Float] = [1.0, 2.0]
    try rawData.withUnsafeMutableBytes { rawBuffer in
      guard let baseAddr = rawBuffer.baseAddress else { return }
      try options.addExternalTensorBinding(
        signatureName: "main",
        tensorName: "input",
        data: baseAddr,
        sizeBytes: rawBuffer.count
      )
    }
  }

  func testAddCustomOpKernel() throws {
    let options = try Options()
    var customKernel = LiteRtCustomOpKernel()
    try options.addCustomOpKernel(
      name: "CustomOp",
      version: 1,
      kernel: &customKernel
    )
  }

  func testCpuOptionsDefaults() throws {
    let cpuOptions = try CpuOptions()
    XCTAssertNil(try cpuOptions.kernelMode())
    XCTAssertNil(try cpuOptions.enableYNNPack())
    XCTAssertNil(try cpuOptions.numThreads())
    XCTAssertNil(try cpuOptions.xnnPackFlags())
    XCTAssertNil(try cpuOptions.xnnPackWeightCachePath())
    XCTAssertNil(try cpuOptions.xnnPackWeightCacheFileDescriptor())
  }

  func testCpuOptionsSettersAndGetters() throws {
    let cpuOptions = try CpuOptions()

    try cpuOptions.setKernelMode(.builtin)
    XCTAssertEqual(try cpuOptions.kernelMode(), .builtin)

    try cpuOptions.setKernelMode(.reference)
    XCTAssertEqual(try cpuOptions.kernelMode(), .reference)

    try cpuOptions.setKernelMode(.delegate)
    XCTAssertEqual(try cpuOptions.kernelMode(), .delegate)

    try cpuOptions.setEnableYNNPack(true)
    XCTAssertEqual(try cpuOptions.enableYNNPack(), true)

    try cpuOptions.setEnableYNNPack(false)
    XCTAssertEqual(try cpuOptions.enableYNNPack(), false)

    try cpuOptions.setNumThreads(4)
    XCTAssertEqual(try cpuOptions.numThreads(), 4)

    try cpuOptions.setXNNPackFlags(0x1234)
    XCTAssertEqual(try cpuOptions.xnnPackFlags(), 0x1234)

    try cpuOptions.setHintFullyDelegatedToSingleDelegate(true)

    try cpuOptions.setXNNPackWeightCachePath("/tmp/weight_cache.bin")
    XCTAssertEqual(try cpuOptions.xnnPackWeightCachePath(), "/tmp/weight_cache.bin")
  }

  func testCpuOptionsWeightCacheFileDescriptor() throws {
    let cpuOptions = try CpuOptions()
    try cpuOptions.setXNNPackWeightCacheFileDescriptor(42)
    XCTAssertEqual(try cpuOptions.xnnPackWeightCacheFileDescriptor(), 42)
  }

  func testAddConcreteOptions() throws {
    let options = try Options()
    let cpuOptions = try CpuOptions()
    try cpuOptions.setNumThreads(2)
    try cpuOptions.setKernelMode(.delegate)

    try options.addConcreteOptions(cpuOptions)

    let opaqueOptions = try options.opaqueOptions()
    XCTAssertNotNil(opaqueOptions)
    XCTAssertEqual(opaqueOptions?.identifier, "xnnpack")
  }

  func testCompiledModelWithCpuOptions() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])

    let cpuOptions = try CpuOptions()
    try cpuOptions.setNumThreads(2)
    try cpuOptions.setKernelMode(.delegate)
    try options.addConcreteOptions(cpuOptions)

    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)
    XCTAssertNotNil(compiledModel)
  }
}
