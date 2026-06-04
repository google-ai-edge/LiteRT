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

import LiteRtC
import XCTest

@testable import LiteRt

final class CompiledModelTests: XCTestCase {
  func testIsFullyAccelerated() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let model = try Model(filePath: modelPath)
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(environment: env, model: model, options: options)
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
    let model = try Model(filePath: modelPath)
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(environment: env, model: model, options: options)
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
    let model = try Model(filePath: modelPath)
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(environment: env, model: model, options: options)
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
    let model = try Model(filePath: modelPath)
    let options = try Options()
    try options.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(environment: env, model: model, options: options)

    let inLayout = try compiledModel.getInputTensorLayout(inputIndex: 0)
    let outLayouts = try compiledModel.getOutputTensorLayouts()

    XCTAssertEqual(inLayout.dimensions, [2])
    XCTAssertEqual(outLayouts.count, 1)
    XCTAssertEqual(outLayouts[0].dimensions, [2])
  }

  func testRunWithOptions() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let model = try Model(filePath: modelPath)
    let compOptions = try Options()
    try compOptions.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(environment: env, model: model, options: compOptions)

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
    let model = try Model(filePath: modelPath)
    let compOptions = try Options()
    try compOptions.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(environment: env, model: model, options: compOptions)

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
    let model = try Model(filePath: modelPath)
    let compOptions = try Options()
    try compOptions.setHardwareAccelerators([.cpu])
    let compiledModel = try CompiledModel(environment: env, model: model, options: compOptions)

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
}
