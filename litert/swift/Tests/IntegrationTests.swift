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
import LiteRtC
@testable import LiteRt

final class IntegrationTests: XCTestCase {
  func testEndToEnd() throws {
    // 1. Create Environment
    let env = try Environment()
    XCTAssertNotNil(env)

    // 2. Load simple_model.tflite
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let model = try Model(filePath: modelPath)
    XCTAssertNotNil(model)

    // 3. Compile Model options
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])

    // 4. Create Compiled Model
    let compiledModel = try CompiledModel(environment: env, model: model, options: options)
    XCTAssertNotNil(compiledModel)

    // 5. Check input and output count
    let inputCount = try compiledModel.inputCount()
    let outputCount = try compiledModel.outputCount()
    XCTAssertEqual(inputCount, 2)
    XCTAssertEqual(outputCount, 1)

    // 5b. Verify input and output names
    let name0 = try compiledModel.inputName(inputIndex: 0)
    let name1 = try compiledModel.inputName(inputIndex: 1)
    let outName = try compiledModel.outputName(outputIndex: 0)
    XCTAssertFalse(name0.isEmpty)
    XCTAssertFalse(name1.isEmpty)
    XCTAssertFalse(outName.isEmpty)

    // 6. Check tensor types
    let inputType0 = try compiledModel.getInputTensorType(inputIndex: 0)
    let inputType1 = try compiledModel.getInputTensorType(inputIndex: 1)
    let outputType = try compiledModel.getOutputTensorType(outputIndex: 0)

    XCTAssertEqual(inputType0.elementType, .float32)
    XCTAssertEqual(inputType1.elementType, .float32)
    XCTAssertEqual(outputType.elementType, .float32)

    XCTAssertEqual(inputType0.layout.dimensions, [2])
    XCTAssertEqual(inputType1.layout.dimensions, [2])
    XCTAssertEqual(outputType.layout.dimensions, [2])

    // 7. Create input and output buffers
    let inputBuffers = try compiledModel.createInputBuffers()
    let outputBuffers = try compiledModel.createOutputBuffers()

    XCTAssertEqual(inputBuffers.count, 2)
    XCTAssertEqual(outputBuffers.count, 1)

    // 8. Write input data
    let input0Data: [Float] = [1.0, 2.0]
    let input1Data: [Float] = [10.0, 20.0]
    try inputBuffers[0].write(input0Data)
    try inputBuffers[1].write(input1Data)

    // 9. Run inference
    try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers)

    // 10. Read and verify outputs
    let outputData: [Float] = try outputBuffers[0].read()
    XCTAssertEqual(outputData.count, 2)
    XCTAssertEqual(outputData[0], 11.0, accuracy: 0.0001)
    XCTAssertEqual(outputData[1], 22.0, accuracy: 0.0001)

    // 11. Hard-preserve ARC lifetimes of all C-bound wrapper instances until the test finishes.
    print("Test complete. Preserved instances: \(env), \(model), \(options), \(compiledModel), \(inputBuffers), \(outputBuffers)")
  }
}
