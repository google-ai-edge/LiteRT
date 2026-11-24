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
    let compiledModel = try CompiledModel(
      filePath: modelPath, environment: env, options: options)
    do {
      try compiledModel.clearErrors()
      let _ = try compiledModel.errorMessages()
    } catch {
      print(
        "Error reporter APIs call failed (expected if buffer error reporter mode is not "
          + "active): \(error)"
      )
    }
  }

  func testRuntimeLayoutQueries() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(
      filePath: modelPath, environment: env, options: options)

    let inLayout = try compiledModel.inputTensorLayout(inputIndex: 0)
    let outLayouts = try compiledModel.outputTensorLayouts()

    XCTAssertEqual(inLayout.dimensions, [2])
    XCTAssertEqual(outLayouts.count, 1)
    XCTAssertEqual(outLayouts[0].dimensions, [2])
  }

  func testRunWithOptions() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let compOptions = try Options()
    try compOptions.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(
      filePath: modelPath, environment: env, options: compOptions)

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
    let compiledModel = try CompiledModel(
      filePath: modelPath, environment: env, options: compOptions)

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
    let compiledModel = try CompiledModel(
      filePath: modelPath, environment: env, options: compOptions)

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

  func testCreateOutputBuffersWithDynamicDimensions() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/dynamic_add_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)

    // Resize input tensors from (?, 2, 3) to (4, 2, 3) -> 24 floats = 96 bytes each
    let newDims = [4, 2, 3]
    try compiledModel.resizeInputTensor(inputIndex: 0, dimensions: newDims)
    try compiledModel.resizeInputTensor(inputIndex: 1, dimensions: newDims)

    // Verify createOutputBuffers() correctly sizes output buffers from resolved layouts
    let outputBuffers = try compiledModel.createOutputBuffers()
    XCTAssertEqual(outputBuffers.count, 1)
    XCTAssertEqual(outputBuffers[0].size, 24 * MemoryLayout<Float>.stride)
    XCTAssertEqual(outputBuffers[0].tensorType?.layout.dimensions, [4, 2, 3])

    // Create input buffers matching resized input dimensions
    let inputBuffers = try [
      TensorBuffer(
        environment: env,
        bufferType: .hostMemory,
        tensorType: TensorType(elementType: .float32, layout: Layout(dimensions: newDims)),
        size: 24 * MemoryLayout<Float>.stride
      ),
      TensorBuffer(
        environment: env,
        bufferType: .hostMemory,
        tensorType: TensorType(elementType: .float32, layout: Layout(dimensions: newDims)),
        size: 24 * MemoryLayout<Float>.stride
      )
    ]
    let inData0 = [Float](repeating: 1.0, count: 24)
    let inData1 = [Float](repeating: 2.0, count: 24)
    try inputBuffers[0].write(inData0)
    try inputBuffers[1].write(inData1)

    // Run inference safely without buffer overflow
    try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers)

    let outputData: [Float] = try outputBuffers[0].read()
    XCTAssertEqual(outputData.count, 24)
    for val in outputData {
      XCTAssertEqual(val, 3.0, accuracy: 0.0001)
    }
  }

  func testCreateInputBuffersWithDynamicDimensions() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/dynamic_add_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)

    // The model declares its input tensors as (?, 2, 3), while the runtime resolves the dynamic
    // dimension to 1 until the input tensors are resized.
    XCTAssertEqual(try compiledModel.inputTensorType(inputIndex: 0).layout.dimensions, [-1, 2, 3])
    XCTAssertEqual(try compiledModel.inputTensorLayout(inputIndex: 0).dimensions, [1, 2, 3])

    // Verify createInputBuffers() sizes input buffers from the runtime layouts:
    // (1, 2, 3) -> 6 floats = 24 bytes each
    let inputBuffers = try compiledModel.createInputBuffers()
    XCTAssertEqual(inputBuffers.count, 2)
    XCTAssertEqual(inputBuffers[0].tensorType?.layout.dimensions, [1, 2, 3])
    XCTAssertEqual(inputBuffers[0].size, 6 * MemoryLayout<Float>.stride)
    XCTAssertEqual(inputBuffers[1].tensorType?.layout.dimensions, [1, 2, 3])
    XCTAssertEqual(inputBuffers[1].size, 6 * MemoryLayout<Float>.stride)

    // Run inference with the created input buffers
    let outputBuffers = try compiledModel.createOutputBuffers()
    try inputBuffers[0].write([Float](repeating: 1.0, count: 6))
    try inputBuffers[1].write([Float](repeating: 2.0, count: 6))
    try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers)

    let outputData: [Float] = try outputBuffers[0].read()
    XCTAssertEqual(outputData, [Float](repeating: 3.0, count: 6))
  }

  func testCreateInputBuffersWithResizedDynamicDimensions() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/dynamic_add_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)

    // Resize input tensors from (?, 2, 3) to (4, 2, 3) -> 24 floats = 96 bytes each
    let newDims = [4, 2, 3]
    try compiledModel.resizeInputTensor(inputIndex: 0, dimensions: newDims)
    try compiledModel.resizeInputTensor(inputIndex: 1, dimensions: newDims)

    // Verify createInputBuffers() sizes input buffers from the resized layouts
    let inputBuffers = try compiledModel.createInputBuffers()
    XCTAssertEqual(inputBuffers.count, 2)
    XCTAssertEqual(inputBuffers[0].tensorType?.layout.dimensions, newDims)
    XCTAssertEqual(inputBuffers[0].size, 24 * MemoryLayout<Float>.stride)
    XCTAssertEqual(inputBuffers[1].tensorType?.layout.dimensions, newDims)
    XCTAssertEqual(inputBuffers[1].size, 24 * MemoryLayout<Float>.stride)

    // Run inference with the created input buffers
    let outputBuffers = try compiledModel.createOutputBuffers()
    try inputBuffers[0].write([Float](repeating: 1.0, count: 24))
    try inputBuffers[1].write([Float](repeating: 2.0, count: 24))
    try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers)

    let outputData: [Float] = try outputBuffers[0].read()
    XCTAssertEqual(outputData, [Float](repeating: 3.0, count: 24))
  }

  func testCreateInputBuffersWithNonStrictResize() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)

    // Resize input tensors from the static shape (2) to (3) -> 3 floats = 12 bytes each. The
    // model keeps declaring its input tensors as (2).
    let newDims = [3]
    try compiledModel.resizeInputTensorNonStrict(inputIndex: 0, dimensions: newDims)
    try compiledModel.resizeInputTensorNonStrict(inputIndex: 1, dimensions: newDims)
    XCTAssertEqual(try compiledModel.inputTensorType(inputIndex: 0).layout.dimensions, [2])

    // Verify createInputBuffers() sizes input buffers from the resized layouts
    let inputBuffers = try compiledModel.createInputBuffers()
    XCTAssertEqual(inputBuffers.count, 2)
    XCTAssertEqual(inputBuffers[0].tensorType?.layout.dimensions, newDims)
    XCTAssertEqual(inputBuffers[0].size, 3 * MemoryLayout<Float>.stride)
    XCTAssertEqual(inputBuffers[1].tensorType?.layout.dimensions, newDims)
    XCTAssertEqual(inputBuffers[1].size, 3 * MemoryLayout<Float>.stride)

    // Run inference with the created input buffers
    let outputBuffers = try compiledModel.createOutputBuffers()
    try inputBuffers[0].write([1.0, 2.0, 3.0] as [Float])
    try inputBuffers[1].write([10.0, 20.0, 30.0] as [Float])
    try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers)

    let outputData: [Float] = try outputBuffers[0].read()
    XCTAssertEqual(outputData, [11.0, 22.0, 33.0])
  }

  func testCreateInputBuffersWithUnsupportedRank() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)

    // Resize an input tensor to rank 9, which exceeds the maximum rank of a tensor layout (8), so
    // the runtime cannot report its layout.
    try compiledModel.resizeInputTensorNonStrict(
      inputIndex: 0, dimensions: [1, 1, 1, 1, 1, 1, 1, 1, 2])
    XCTAssertThrowsError(try compiledModel.inputTensorLayout(inputIndex: 0))

    // Verify createInputBuffers() fails instead of falling back to the declared shape (2)
    XCTAssertThrowsError(try compiledModel.createInputBuffers()) { error in
      XCTAssertEqual(error as? LiteRtError, .invalidArgument)
    }
  }
}


