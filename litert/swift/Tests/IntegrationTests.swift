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

final class IntegrationTests: XCTestCase {
  func testEndToEnd() throws {
    // 1. Create Environment
    let env = try Environment()
    XCTAssertNotNil(env)

    // 2. Create CompiledModel with simple_model.tflite, environment, and options
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])

    let compiledModel = try CompiledModel(filePath: modelPath, environment: env, options: options)
    XCTAssertNotNil(compiledModel)

    // 3. Check input and output count
    let inputCount = try compiledModel.inputCount()
    let outputCount = try compiledModel.outputCount()
    XCTAssertEqual(inputCount, 2)
    XCTAssertEqual(outputCount, 1)

    // 4. Verify input and output names
    let name0 = try compiledModel.inputName(inputIndex: 0)
    let name1 = try compiledModel.inputName(inputIndex: 1)
    let outName = try compiledModel.outputName(outputIndex: 0)
    XCTAssertFalse(name0.isEmpty)
    XCTAssertFalse(name1.isEmpty)
    XCTAssertFalse(outName.isEmpty)

    // 5. Check tensor types
    let inputType0 = try compiledModel.getInputTensorType(inputIndex: 0)
    let inputType1 = try compiledModel.getInputTensorType(inputIndex: 1)
    let outputType = try compiledModel.getOutputTensorType(outputIndex: 0)

    XCTAssertEqual(inputType0.elementType, .float32)
    XCTAssertEqual(inputType1.elementType, .float32)
    XCTAssertEqual(outputType.elementType, .float32)

    XCTAssertEqual(inputType0.layout.dimensions, [2])
    XCTAssertEqual(inputType1.layout.dimensions, [2])
    XCTAssertEqual(outputType.layout.dimensions, [2])

    // 6. Create input and output buffers
    let inputBuffers = try compiledModel.createInputBuffers()
    let outputBuffers = try compiledModel.createOutputBuffers()

    XCTAssertEqual(inputBuffers.count, 2)
    XCTAssertEqual(outputBuffers.count, 1)

    // 7. Write input data
    let input0Data: [Float] = [1.0, 2.0]
    let input1Data: [Float] = [10.0, 20.0]
    try inputBuffers[0].write(input0Data)
    try inputBuffers[1].write(input1Data)

    // 8. Run inference
    try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers)

    // 9. Read and verify outputs
    let outputData: [Float] = try outputBuffers[0].read()
    XCTAssertEqual(outputData.count, 2)
    XCTAssertEqual(outputData[0], 11.0, accuracy: 0.0001)
    XCTAssertEqual(outputData[1], 22.0, accuracy: 0.0001)

    // 10. Hard-preserve ARC lifetimes of all C-bound wrapper instances until the test finishes.
    print("Test complete. Preserved instances: \(env), \(compiledModel), \(options), \(inputBuffers), \(outputBuffers)")
  }
}
