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

import Foundation
import LiteRt
import LiteRtC
import XCTest

class LiteRtTests: XCTestCase {

  private func getModelPath() -> String? {
    if let srcDir = ProcessInfo.processInfo.environment["TEST_SRCDIR"],
      let workspace = ProcessInfo.processInfo.environment["TEST_WORKSPACE"]
    {
      let path =
        "\(srcDir)/\(workspace)/third_party/odml/litert/litert/test/testdata/simple_model.tflite"
      if FileManager.default.fileExists(atPath: path) {
        return path
      }
    }

    // 2. Check Bundle Resources (for Apple platforms/Xcode)
    if let bundlePath = Bundle.allBundles.first(where: { $0.bundlePath.hasSuffix(".xctest") })?
      .bundlePath
    {
      let path = bundlePath + "/simple_model.tflite"
      if FileManager.default.fileExists(atPath: path) {
        return path
      }
      // Try nested path in bundle if flattened
      let nestedPath =
        bundlePath + "/third_party/odml/litert/litert/test/testdata/simple_model.tflite"
      if FileManager.default.fileExists(atPath: nestedPath) {
        return nestedPath
      }
    }

    // 3. Fallback for local execution
    let paths = [
      "third_party/odml/litert/litert/test/testdata/simple_model.tflite"
    ]

    let fileManager = FileManager.default
    for path in paths {
      if fileManager.fileExists(atPath: path) {
        return path
      }
    }
    return nil
  }

  func testE2E_AutomaticBuffers() throws {
    let env = try Environment()
    guard let modelPath = getModelPath() else {
      XCTFail("Could not find simple_model.tflite")
      return
    }
    let options = try CompiledModel.Options()
    try options.setAccelerators(.cpu)
    let model = try CompiledModel(file: modelPath, options: options, environment: env)

    // Create buffers (automatically)
    let inputs = try model.createInputBuffers()
    XCTAssertEqual(inputs.count, 2)

    try inputs[0].write([Float(1.0), Float(2.0)])
    try inputs[1].write([Float(10.0), Float(20.0)])

    let outputs = try model.createOutputBuffers()
    XCTAssertEqual(outputs.count, 1)

    // Run (assuming signature index 0 for default)
    try model.run(signatureIndex: 0, inputBuffers: inputs, outputBuffers: outputs)

    let results: [Float] = try outputs[0].read()
    XCTAssertEqual(results.count, 2)
    XCTAssertEqual(results[0], 11.0, accuracy: 0.0001)
    XCTAssertEqual(results[1], 22.0, accuracy: 0.0001)
  }

  func testE2E_ManualBuffers() throws {
    let env = try Environment()
    guard let modelPath = getModelPath() else {
      XCTFail("Could not find simple_model.tflite")
      return
    }
    let options = try CompiledModel.Options()
    // Test CPU options
    try options.setAccelerators(.cpu)

    let model = try CompiledModel(file: modelPath, options: options, environment: env)

    // Create input 0
    let input0 = try model.createInputBuffer(signature: nil, inputName: "arg0")
    try input0.write([Float(1.0), Float(2.0)])

    // Create input 1
    let input1 = try model.createInputBuffer(signature: nil, inputName: "arg1")
    try input1.write([Float(10.0), Float(20.0)])

    // Create output
    let output = try model.createOutputBuffer(signature: nil, outputName: "tfl.add")

    // Run
    try model.run(signatureIndex: 0, inputBuffers: [input0, input1], outputBuffers: [output])

    // Verify
    let results: [Float] = try output.read()
    XCTAssertEqual(results.count, 2)
    XCTAssertEqual(results[0], 11.0, accuracy: 0.0001)
    XCTAssertEqual(results[1], 22.0, accuracy: 0.0001)
  }

  func testTensorBufferOperations() throws {
    let env = try Environment()

    // Test Case 1: Float Buffer
    let floatLayout = TensorType.Layout(dimensions: [3])
    let floatType = TensorType(elementType: .float32, layout: floatLayout)

    let floatBuffer = try TensorBuffer(tensorType: floatType, environment: env)
    let floatData: [Float] = [1.5, 2.5, 3.5]
    try floatBuffer.write(floatData)
    let floatRead: [Float] = try floatBuffer.read()
    XCTAssertEqual(floatRead, floatData)

    // Test Case 2: Int32 Buffer
    let intLayout = TensorType.Layout(dimensions: [2])
    let intType = TensorType(elementType: .int32, layout: intLayout)

    let intBuffer = try TensorBuffer(tensorType: intType, environment: env)
    let intData: [Int32] = [42, -10]
    try intBuffer.write(intData)
    let intRead: [Int32] = try intBuffer.read()
    XCTAssertEqual(intRead, intData)

    // Test Case 3: Error handling (Capacity mismatch)
    let overflowData: [Float] = [1.0, 2.0, 3.0, 4.0]
    XCTAssertThrowsError(try floatBuffer.write(overflowData)) { error in
      // Verify it's the expected error type if possible, or just that it throws
    }
  }

  func testTensorBufferOperationsInt8_UInt8() throws {
    let env = try Environment()

    // Test Case: Int8 Buffer
    let int8Layout = TensorType.Layout(dimensions: [4])
    let int8Type = TensorType(elementType: .int8, layout: int8Layout)

    let int8Buffer = try TensorBuffer(tensorType: int8Type, environment: env)
    let int8Data: [Int8] = [12, -34, 56, -78]
    try int8Buffer.write(int8Data)
    let int8Read: [Int8] = try int8Buffer.read()
    XCTAssertEqual(int8Read, int8Data)

    // Test Case: UInt8 Buffer
    let uint8Layout = TensorType.Layout(dimensions: [3])
    let uint8Type = TensorType(elementType: .uint8, layout: uint8Layout)

    let uint8Buffer = try TensorBuffer(tensorType: uint8Type, environment: env)
    let uint8Data: [UInt8] = [200, 150, 50]
    try uint8Buffer.write(uint8Data)
    let uint8Read: [UInt8] = try uint8Buffer.read()
    XCTAssertEqual(uint8Read, uint8Data)
  }

  func testModelOptions() throws {
    let options = try CompiledModel.Options()

    // Test CPU options
    try options.setAccelerators(.cpu)
    XCTAssertEqual(try options.getAccelerators(), .cpu)

    let cpuOptions = try CompiledModel.Options.CpuOptions()
    try cpuOptions.setNumThreads(4)
    XCTAssertEqual(try cpuOptions.getNumThreads(), 4)

    try options.setCpuOptions(cpuOptions)

    // Verify no crash/error on valid configuration
  }

  func testInvalidModelPath() throws {
    let env = try Environment()
    let options = try CompiledModel.Options()

    XCTAssertThrowsError(
      try CompiledModel(file: "non_existent_model.tflite", options: options, environment: env))
  }

  func testCreateBufferWithInvalidName() throws {
    let env = try Environment()
    guard let modelPath = getModelPath() else { return }
    let options = try CompiledModel.Options()
    try options.setAccelerators(.cpu)
    let model = try CompiledModel(file: modelPath, options: options, environment: env)

    XCTAssertThrowsError(try model.createInputBuffer(inputName: "invalid_input_name")) { error in
      guard let liteRtError = error as? LiteRtError else {
        XCTFail("Expected LiteRtError but got \(type(of: error))")
        return
      }
      switch liteRtError {
      case .invalidArgument(let message):
        XCTAssertEqual(message, "Input not found: invalid_input_name")
      default:
        XCTFail("Expected invalidArgument error, got \(liteRtError)")
      }
    }
  }

  func testCreateBufferWithInvalidSignatureKey() throws {
    let env = try Environment()
    guard let modelPath = getModelPath() else { return }
    let options = try CompiledModel.Options()
    try options.setAccelerators(.cpu)
    let model = try CompiledModel(file: modelPath, options: options, environment: env)

    XCTAssertThrowsError(try model.createInputBuffers(signatureKey: "invalid_signature_key")) { error in
      guard let liteRtError = error as? LiteRtError else {
        XCTFail("Expected LiteRtError but got \(type(of: error))")
        return
      }
      switch liteRtError {
      case .invalidArgument(let message):
        XCTAssertEqual(message, "Signature not found: invalid_signature_key")
      default:
        XCTFail("Expected invalidArgument error, got \(liteRtError)")
      }
    }
  }

  func testTensorTypes() throws {
    let env = try Environment()
    guard let modelPath = getModelPath() else { return }
    let options = try CompiledModel.Options()
    try options.setAccelerators(.cpu)
    let model = try CompiledModel(file: modelPath, options: options, environment: env)

    let inputs = try model.createInputBuffers()
    XCTAssertEqual(inputs.count, 2)

    // Input 0: "arg0", float, [2]
    let input0Type = inputs[0].tensorType
    XCTAssertEqual(input0Type.elementType, TensorType.ElementType.float32)
    XCTAssertEqual(input0Type.layout.dimensions.count, 1)  // rank
    XCTAssertEqual(input0Type.layout.dimensions, [2])

    // Input 1: "arg1", float, [2]
    let input1Type = inputs[1].tensorType
    XCTAssertEqual(input1Type.elementType, TensorType.ElementType.float32)
    XCTAssertEqual(input1Type.layout.dimensions.count, 1)  // rank
    XCTAssertEqual(input1Type.layout.dimensions, [2])

    let outputs = try model.createOutputBuffers()
    XCTAssertEqual(outputs.count, 1)

    // Output 0: "tfl.add", float, [2]
    let outputType = outputs[0].tensorType
    XCTAssertEqual(outputType.elementType, TensorType.ElementType.float32)
    XCTAssertEqual(outputType.layout.dimensions.count, 1)  // rank
    XCTAssertEqual(outputType.layout.dimensions, [2])
  }

  func testTensorBufferRequirements() throws {
    let env = try Environment()
    guard let modelPath = getModelPath() else { return }
    let options = try CompiledModel.Options()
    try options.setAccelerators(.cpu)
    let model = try CompiledModel(file: modelPath, options: options, environment: env)

    // Input 0
    let input0Reqs = try model.getInputRequirements(signatureIndex: 0, inputIndex: 0)
    XCTAssertEqual(input0Reqs.bufferSize, 8)  // 2 floats * 4 bytes

    // Output 0
    let outputReqs = try model.getOutputRequirements(signatureIndex: 0, outputIndex: 0)
    XCTAssertEqual(outputReqs.bufferSize, 8)
  }


#if canImport(Metal)
  func testMetalZeroCopyInit() throws {
    let env = try Environment()
    guard let modelPath = getModelPath() else { return }
    let options = try CompiledModel.Options()
    let model = try CompiledModel(file: modelPath, options: options, environment: env)

    let inputType = try model.createInputBuffers()[0].tensorType

    // In a real test we would need an MTLDevice, but for compilation correctness
    // we just ensure the API signature is callable. Since MTLDevice.systemDefault
    // might be nil in CI, we skip the actual Buffer creation if it's nil.
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    let elementCount = 2
    let byteSize = elementCount * MemoryLayout<Float32>.stride
    guard let metalBuffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else { return }

    // Init the tensor buffer directly from the Metal buffer
    let tensorBuffer = try TensorBuffer(tensorType: inputType, environment: env, metalBuffer: metalBuffer)
    XCTAssertNotNil(tensorBuffer)
  }
#endif
}
