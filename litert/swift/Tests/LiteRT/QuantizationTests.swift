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

final class QuantizationTests: XCTestCase {
  func testQuantizationTypeID() {
    XCTAssertEqual(QuantizationTypeID.none.rawValue, 0)
    XCTAssertEqual(QuantizationTypeID.perTensor.rawValue, 1)
    XCTAssertEqual(QuantizationTypeID.perChannel.rawValue, 2)
    XCTAssertEqual(QuantizationTypeID.blockWise.rawValue, 3)

    XCTAssertEqual(
      QuantizationTypeID(cType: kLiteRtQuantizationNone), QuantizationTypeID.none)
    XCTAssertEqual(
      QuantizationTypeID(cType: kLiteRtQuantizationPerTensor), QuantizationTypeID.perTensor)
    XCTAssertEqual(
      QuantizationTypeID(cType: kLiteRtQuantizationPerChannel), QuantizationTypeID.perChannel)
    XCTAssertEqual(
      QuantizationTypeID(cType: kLiteRtQuantizationBlockWise), QuantizationTypeID.blockWise)

    XCTAssertEqual(QuantizationTypeID.none.cType, kLiteRtQuantizationNone)
    XCTAssertEqual(QuantizationTypeID.perTensor.cType, kLiteRtQuantizationPerTensor)
    XCTAssertEqual(QuantizationTypeID.perChannel.cType, kLiteRtQuantizationPerChannel)
    XCTAssertEqual(QuantizationTypeID.blockWise.cType, kLiteRtQuantizationBlockWise)

    XCTAssertEqual(QuantizationTypeID.none.description, "none")
    XCTAssertEqual(QuantizationTypeID.perTensor.description, "perTensor")
    XCTAssertEqual(QuantizationTypeID.perChannel.description, "perChannel")
    XCTAssertEqual(QuantizationTypeID.blockWise.description, "blockWise")
  }

  func testQuantizationPerTensor() {
    let q = QuantizationPerTensor(scale: 0.5, zeroPoint: 12)
    XCTAssertEqual(q.scale, 0.5)
    XCTAssertEqual(q.zeroPoint, 12)

    let cQ = q.cQuantization
    XCTAssertEqual(cQ.scale, 0.5)
    XCTAssertEqual(cQ.zero_point, 12)

    let roundtrip = QuantizationPerTensor(cQuantization: cQ)
    XCTAssertEqual(roundtrip, q)
    XCTAssertEqual(roundtrip.hashValue, q.hashValue)
    XCTAssertTrue(q.description.contains("0.5"))
    XCTAssertTrue(q.description.contains("12"))
  }

  func testQuantizationPerChannel() {
    let scales: [Float] = [0.1, 0.2, 0.3]
    let zeroPoints: [Int64] = [1, 2, 3]
    let q = QuantizationPerChannel(quantizedDimension: 1, scales: scales, zeroPoints: zeroPoints)

    XCTAssertEqual(q.quantizedDimension, 1)
    XCTAssertEqual(q.channelCount, 3)
    XCTAssertEqual(q.scales, scales)
    XCTAssertEqual(q.zeroPoints, zeroPoints)

    q.withCQuantization { cQ in
      XCTAssertEqual(cQ.quantized_dimension, 1)
      XCTAssertEqual(cQ.num_channels, 3)
      XCTAssertNotNil(cQ.scales)
      XCTAssertNotNil(cQ.zero_points)

      let roundtrip = QuantizationPerChannel(cQuantization: cQ)
      XCTAssertEqual(roundtrip, q)
      XCTAssertEqual(roundtrip.hashValue, q.hashValue)
    }

    XCTAssertTrue(q.description.contains("quantizedDimension: 1"))
    XCTAssertTrue(q.description.contains("channelCount: 3"))
  }

  func testQuantizationBlockWise() {
    let q = QuantizationBlockWise(scalesTensor: nil, zeroPointsTensor: nil, blockSize: 64)
    XCTAssertEqual(q.blockSize, 64)
    XCTAssertNil(q.scalesTensor)
    XCTAssertNil(q.zeroPointsTensor)

    let cQ = q.cQuantization
    XCTAssertEqual(cQ.block_size, 64)
    XCTAssertNil(cQ.scales)
    XCTAssertNil(cQ.zero_points)

    let roundtrip = QuantizationBlockWise(cQuantization: cQ)
    XCTAssertEqual(roundtrip, q)
    XCTAssertEqual(roundtrip.hashValue, q.hashValue)
    XCTAssertTrue(q.description.contains("blockSize: 64"))
  }

  func testQuantizationEnum() {
    let qNone = Quantization.none
    XCTAssertEqual(qNone.typeID, QuantizationTypeID.none)
    XCTAssertFalse(qNone.isQuantized)
    XCTAssertNil(qNone.perTensor)
    XCTAssertNil(qNone.perChannel)
    XCTAssertNil(qNone.blockWise)
    XCTAssertEqual(qNone.description, "Quantization.none")

    let pt = QuantizationPerTensor(scale: 0.25, zeroPoint: 5)
    let qPerTensor = Quantization.perTensor(pt)
    XCTAssertEqual(qPerTensor.typeID, QuantizationTypeID.perTensor)
    XCTAssertTrue(qPerTensor.isQuantized)
    XCTAssertEqual(qPerTensor.perTensor, pt)
    XCTAssertNil(qPerTensor.perChannel)
    XCTAssertNil(qPerTensor.blockWise)
    XCTAssertTrue(qPerTensor.description.contains("perTensor"))

    let pc = QuantizationPerChannel(quantizedDimension: 0, scales: [0.5], zeroPoints: [10])
    let qPerChannel = Quantization.perChannel(pc)
    XCTAssertEqual(qPerChannel.typeID, QuantizationTypeID.perChannel)
    XCTAssertTrue(qPerChannel.isQuantized)
    XCTAssertNil(qPerChannel.perTensor)
    XCTAssertEqual(qPerChannel.perChannel, pc)
    XCTAssertNil(qPerChannel.blockWise)
    XCTAssertTrue(qPerChannel.description.contains("perChannel"))

    let bw = QuantizationBlockWise(blockSize: 128)
    let qBlockWise = Quantization.blockWise(bw)
    XCTAssertEqual(qBlockWise.typeID, QuantizationTypeID.blockWise)
    XCTAssertTrue(qBlockWise.isQuantized)
    XCTAssertNil(qBlockWise.perTensor)
    XCTAssertNil(qBlockWise.perChannel)
    XCTAssertEqual(qBlockWise.blockWise, bw)
    XCTAssertTrue(qBlockWise.description.contains("blockWise"))
  }

  func testElementTypeNewValues() {
    XCTAssertEqual(ElementType.uint4.rawValue, 21)
    XCTAssertEqual(ElementType.float8E4M3FN.rawValue, 22)
    XCTAssertEqual(ElementType.float8E5M2.rawValue, 23)

    XCTAssertEqual(ElementType(cType: kLiteRtElementTypeUInt4), ElementType.uint4)
    XCTAssertEqual(ElementType(cType: kLiteRtElementTypeFloat8E4M3FN), ElementType.float8E4M3FN)
    XCTAssertEqual(ElementType(cType: kLiteRtElementTypeFloat8E5M2), ElementType.float8E5M2)

    XCTAssertEqual(ElementType.uint4.cType, kLiteRtElementTypeUInt4)
    XCTAssertEqual(ElementType.float8E4M3FN.cType, kLiteRtElementTypeFloat8E4M3FN)
    XCTAssertEqual(ElementType.float8E5M2.cType, kLiteRtElementTypeFloat8E5M2)
  }

  func testTensorTypeWithQuantization() {
    let layout = Layout(dimensions: [1, 10])
    let pt = QuantizationPerTensor(scale: 0.1, zeroPoint: 0)
    let tensorType = TensorType(elementType: .int8, layout: layout, quantization: .perTensor(pt))

    XCTAssertEqual(tensorType.elementType, ElementType.int8)
    XCTAssertEqual(tensorType.layout, layout)
    XCTAssertEqual(tensorType.quantization, Quantization.perTensor(pt))
  }

  func testQuantizationQueriesOnSimpleModel() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env)

    let inputCount = try compiledModel.inputCount()
    XCTAssertGreaterThan(inputCount, 0)

    for i in 0..<inputCount {
      let quant = try compiledModel.inputTensorQuantization(inputIndex: i)
      XCTAssertEqual(quant, Quantization.none)
      XCTAssertEqual(quant.typeID, QuantizationTypeID.none)
      XCTAssertFalse(quant.isQuantized)
      XCTAssertNil(quant.perTensor)
      XCTAssertNil(quant.perChannel)
      XCTAssertNil(quant.blockWise)

      let inputType = try compiledModel.inputTensorType(inputIndex: i)
      XCTAssertEqual(inputType.quantization, Quantization.none)

      let inputName = try compiledModel.inputName(inputIndex: i)
      let quantByName = try compiledModel.inputTensorQuantization(inputName: inputName)
      XCTAssertEqual(quantByName, Quantization.none)

      let inputTypeByName = try compiledModel.inputTensorType(inputName: inputName)
      XCTAssertEqual(inputTypeByName.quantization, Quantization.none)
    }

    let outputCount = try compiledModel.outputCount()
    XCTAssertGreaterThan(outputCount, 0)

    for i in 0..<outputCount {
      let quant = try compiledModel.outputTensorQuantization(outputIndex: i)
      XCTAssertEqual(quant, Quantization.none)
      XCTAssertEqual(quant.typeID, QuantizationTypeID.none)
      XCTAssertFalse(quant.isQuantized)
      XCTAssertNil(quant.perTensor)
      XCTAssertNil(quant.perChannel)
      XCTAssertNil(quant.blockWise)

      let outputType = try compiledModel.outputTensorType(outputIndex: i)
      XCTAssertEqual(outputType.quantization, Quantization.none)

      let outputName = try compiledModel.outputName(outputIndex: i)
      let quantByName = try compiledModel.outputTensorQuantization(outputName: outputName)
      XCTAssertEqual(quantByName, Quantization.none)

      let outputTypeByName = try compiledModel.outputTensorType(outputName: outputName)
      XCTAssertEqual(outputTypeByName.quantization, Quantization.none)
    }
  }

  func testQuantizationQueriesOnQuantizedModel() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_quantized_ops.tflite"
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env)

    let inputCount = try compiledModel.inputCount()
    XCTAssertGreaterThan(inputCount, 0)

    for i in 0..<inputCount {
      let quant = try compiledModel.inputTensorQuantization(inputIndex: i)
      let inputType = try compiledModel.inputTensorType(inputIndex: i)
      XCTAssertEqual(inputType.quantization, quant)

      let inputName = try compiledModel.inputName(inputIndex: i)
      let quantByName = try compiledModel.inputTensorQuantization(inputName: inputName)
      XCTAssertEqual(quantByName, quant)

      let inputTypeByName = try compiledModel.inputTensorType(inputName: inputName)
      XCTAssertEqual(inputTypeByName.quantization, quant)

      switch quant {
      case .none:
        XCTAssertFalse(quant.isQuantized)
        XCTAssertEqual(quant.typeID, QuantizationTypeID.none)
        XCTAssertNil(quant.perTensor)
        XCTAssertNil(quant.perChannel)
        XCTAssertNil(quant.blockWise)
      case .perTensor(let pt):
        XCTAssertTrue(quant.isQuantized)
        XCTAssertEqual(quant.typeID, QuantizationTypeID.perTensor)
        XCTAssertEqual(quant.perTensor, pt)
        XCTAssertNil(quant.perChannel)
        XCTAssertNil(quant.blockWise)
      case .perChannel(let pc):
        XCTAssertTrue(quant.isQuantized)
        XCTAssertEqual(quant.typeID, QuantizationTypeID.perChannel)
        XCTAssertEqual(quant.perChannel, pc)
        XCTAssertNil(quant.perTensor)
        XCTAssertNil(quant.blockWise)
      case .blockWise(let bw):
        XCTAssertTrue(quant.isQuantized)
        XCTAssertEqual(quant.typeID, QuantizationTypeID.blockWise)
        XCTAssertEqual(quant.blockWise, bw)
        XCTAssertNil(quant.perTensor)
        XCTAssertNil(quant.perChannel)
      }
    }

    let outputCount = try compiledModel.outputCount()
    XCTAssertGreaterThan(outputCount, 0)

    for i in 0..<outputCount {
      let quant = try compiledModel.outputTensorQuantization(outputIndex: i)
      let outputType = try compiledModel.outputTensorType(outputIndex: i)
      XCTAssertEqual(outputType.quantization, quant)

      let outputName = try compiledModel.outputName(outputIndex: i)
      let quantByName = try compiledModel.outputTensorQuantization(outputName: outputName)
      XCTAssertEqual(quantByName, quant)

      let outputTypeByName = try compiledModel.outputTensorType(outputName: outputName)
      XCTAssertEqual(outputTypeByName.quantization, quant)
    }
  }
}
