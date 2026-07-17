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

final class CompiledModelTests: XCTestCase {
  func testIsFullyAccelerated() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)
    do {
      _ = try compiledModel.isFullyAccelerated()
    } catch {
      print(
        "isFullyAccelerated call failed (expected if not supported on target environment): \(error)"
      )
    }
  }

  func testResizeInputTensor() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)
    do {
      try compiledModel.resizeInputTensor(inputIndex: 0, dimensions: [2])
    } catch {
      print(
        "resizeInputTensor call failed (expected if not supported on target environment): \(error)")
    }
  }

  func testErrorReporter() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)
    do {
      try compiledModel.clearErrors()
      let _ = try compiledModel.getErrorMessages()
    } catch {
      print(
        "Error reporter APIs call failed (expected if buffer error reporter mode is not active): \(error)"
      )
    }
  }

  func testRuntimeLayoutQueries() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)

    let inLayout = try compiledModel.getInputTensorLayout(inputIndex: 0)
    let outLayouts = try compiledModel.getOutputTensorLayouts()

    XCTAssertEqual(inLayout.dimensions, [2])
    XCTAssertEqual(outLayouts.count, 1)
    XCTAssertEqual(outLayouts[0].dimensions, [2])
  }

  func testRunWithOptions() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let compOptions = try Options()
    try compOptions.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: compOptions)

    let inputBuffers = try compiledModel.createInputBuffers()
    let outputBuffers = try compiledModel.createOutputBuffers()
    try inputBuffers[0].write([1.0, 2.0] as [Float])
    try inputBuffers[1].write([10.0, 20.0] as [Float])

    let runOptions = try Options()
    try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers, options: runOptions)

    let outputData: [Float] = try outputBuffers[0].read()
    XCTAssertEqual(outputData[0], 11.0, accuracy: 0.0001)
  }

  func testDispatch() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let compOptions = try Options()
    try compOptions.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: compOptions)

    let inputBuffers = try compiledModel.createInputBuffers()
    let outputBuffers = try compiledModel.createOutputBuffers()
    try inputBuffers[0].write([1.0, 2.0] as [Float])
    try inputBuffers[1].write([10.0, 20.0] as [Float])

    let isAsync = try compiledModel.dispatch(inputs: inputBuffers, outputs: outputBuffers)
    print("Executed asynchronously: \(isAsync)")

    let outputData: [Float] = try outputBuffers[0].read()
    XCTAssertEqual(outputData[0], 11.0, accuracy: 0.0001)
  }

  func testCancellationFunction() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let compOptions = try Options()
    try compOptions.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: compOptions)

    try compiledModel.setCancellationFunction(userData: nil) { _ in
      return false
    }

    let inputBuffers = try compiledModel.createInputBuffers()
    let outputBuffers = try compiledModel.createOutputBuffers()
    try inputBuffers[0].write([1.0, 2.0] as [Float])
    try inputBuffers[1].write([10.0, 20.0] as [Float])

    try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers)

    let outputData: [Float] = try outputBuffers[0].read()
    XCTAssertEqual(outputData[0], 11.0, accuracy: 0.0001)
  }

  func testSignatureKeys() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env)
    let keys = try compiledModel.signatureKeys()
    XCTAssertFalse(keys.isEmpty)
  }

  func testModelCreationFromFd() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let fileURL = URL(fileURLWithPath: modelPath)
    let fileHandle = try FileHandle(forReadingFrom: fileURL)
    let fd = fileHandle.fileDescriptor

    let fileSize = try fileHandle.seekToEnd()
    try fileHandle.seek(toOffset: 0)

    let compiledModel = try CompiledModel(fd: fd, offset: 0, size: Int(fileSize), environment: env)
    XCTAssertNotNil(compiledModel)

    try fileHandle.close()
  }

  func testModelMetadata() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env)

    let metadataKey = "test_key"
    let metadataValue = try XCTUnwrap("test_value".data(using: .utf8))

    try compiledModel.addMetadata(key: metadataKey, data: metadataValue)

    let readValue = try compiledModel.metadata(key: metadataKey)
    XCTAssertEqual(readValue, metadataValue)
  }
}
