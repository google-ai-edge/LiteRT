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
@testable import LiteRT
import Testing

struct QuantizationTests: ~Copyable {
  @Test func `quantization type ID`() {
    #expect(QuantizationTypeID.none.rawValue == 0)
    #expect(QuantizationTypeID.perTensor.rawValue == 1)
    #expect(QuantizationTypeID.perChannel.rawValue == 2)
    #expect(QuantizationTypeID.blockWise.rawValue == 3)

    #expect(
      QuantizationTypeID(cType: kLiteRtQuantizationNone) == QuantizationTypeID.none)
    #expect(
      QuantizationTypeID(cType: kLiteRtQuantizationPerTensor) == QuantizationTypeID.perTensor)
    #expect(
      QuantizationTypeID(cType: kLiteRtQuantizationPerChannel) == QuantizationTypeID.perChannel)
    #expect(
      QuantizationTypeID(cType: kLiteRtQuantizationBlockWise) == QuantizationTypeID.blockWise)

    #expect(QuantizationTypeID.none.cType == kLiteRtQuantizationNone)
    #expect(QuantizationTypeID.perTensor.cType == kLiteRtQuantizationPerTensor)
    #expect(QuantizationTypeID.perChannel.cType == kLiteRtQuantizationPerChannel)
    #expect(QuantizationTypeID.blockWise.cType == kLiteRtQuantizationBlockWise)

    #expect(QuantizationTypeID.none.description == "none")
    #expect(QuantizationTypeID.perTensor.description == "perTensor")
    #expect(QuantizationTypeID.perChannel.description == "perChannel")
    #expect(QuantizationTypeID.blockWise.description == "blockWise")
  }

  @Test func `quantization per tensor`() {
    let q = QuantizationPerTensor(scale: 0.5, zeroPoint: 12)
    #expect(q.scale == 0.5)
    #expect(q.zeroPoint == 12)

    let cQ = q.cQuantization
    #expect(cQ.scale == 0.5)
    #expect(cQ.zero_point == 12)

    let roundtrip = QuantizationPerTensor(cQuantization: cQ)
    #expect(roundtrip == q)
    #expect(roundtrip.hashValue == q.hashValue)
    #expect(q.description.contains("0.5") == true)
    #expect(q.description.contains("12") == true)
  }

  @Test func `quantization per channel`() {
    let scales: [Float] = [0.1, 0.2, 0.3]
    let zeroPoints: [Int64] = [1, 2, 3]
    let q = QuantizationPerChannel(quantizedDimension: 1, scales: scales, zeroPoints: zeroPoints)

    #expect(q.quantizedDimension == 1)
    #expect(q.channelCount == 3)
    #expect(q.scales == scales)
    #expect(q.zeroPoints == zeroPoints)

    q.withCQuantization { cQ in
      #expect(cQ.quantized_dimension == 1)
      #expect(cQ.num_channels == 3)
      #expect(cQ.scales != nil)
      #expect(cQ.zero_points != nil)

      let roundtrip = QuantizationPerChannel(cQuantization: cQ)
      #expect(roundtrip == q)
      #expect(roundtrip.hashValue == q.hashValue)
    }

    #expect(q.description.contains("quantizedDimension: 1") == true)
    #expect(q.description.contains("channelCount: 3") == true)
  }

  @Test func `quantization block wise`() {
    let q = QuantizationBlockWise(scalesTensor: nil, zeroPointsTensor: nil, blockSize: 64)
    #expect(q.blockSize == 64)
    #expect(q.scalesTensor == nil)
    #expect(q.zeroPointsTensor == nil)

    let cQ = q.cQuantization
    #expect(cQ.block_size == 64)
    #expect(cQ.scales == nil)
    #expect(cQ.zero_points == nil)

    let roundtrip = QuantizationBlockWise(cQuantization: cQ)
    #expect(roundtrip == q)
    #expect(roundtrip.hashValue == q.hashValue)
    #expect(q.description.contains("blockSize: 64") == true)
  }

  @Test func `quantization enum`() {
    let qNone = Quantization.none
    #expect(qNone.typeID == .none)
    #expect(qNone.isQuantized == false)
    #expect(qNone.perTensor == nil)
    #expect(qNone.perChannel == nil)
    #expect(qNone.blockWise == nil)
    #expect(qNone.description == "Quantization.none")

    let pt = QuantizationPerTensor(scale: 0.25, zeroPoint: 5)
    let qPerTensor = Quantization.perTensor(pt)
    #expect(qPerTensor.typeID == .perTensor)
    #expect(qPerTensor.isQuantized == true)
    #expect(qPerTensor.perTensor == pt)
    #expect(qPerTensor.perChannel == nil)
    #expect(qPerTensor.blockWise == nil)
    #expect(qPerTensor.description.contains("perTensor") == true)

    let pc = QuantizationPerChannel(quantizedDimension: 0, scales: [0.5], zeroPoints: [10])
    let qPerChannel = Quantization.perChannel(pc)
    #expect(qPerChannel.typeID == .perChannel)
    #expect(qPerChannel.isQuantized == true)
    #expect(qPerChannel.perTensor == nil)
    #expect(qPerChannel.perChannel == pc)
    #expect(qPerChannel.blockWise == nil)
    #expect(qPerChannel.description.contains("perChannel") == true)

    let bw = QuantizationBlockWise(blockSize: 128)
    let qBlockWise = Quantization.blockWise(bw)
    #expect(qBlockWise.typeID == .blockWise)
    #expect(qBlockWise.isQuantized == true)
    #expect(qBlockWise.perTensor == nil)
    #expect(qBlockWise.perChannel == nil)
    #expect(qBlockWise.blockWise == bw)
    #expect(qBlockWise.description.contains("blockWise") == true)
  }

  @Test func `element type new values`() {
    #expect(ElementType.uint4.rawValue == 21)
    #expect(ElementType.float8E4M3FN.rawValue == 22)
    #expect(ElementType.float8E5M2.rawValue == 23)

    #expect(ElementType(cType: kLiteRtElementTypeUInt4) == .uint4)
    #expect(ElementType(cType: kLiteRtElementTypeFloat8E4M3FN) == .float8E4M3FN)
    #expect(ElementType(cType: kLiteRtElementTypeFloat8E5M2) == .float8E5M2)

    #expect(ElementType.uint4.cType == kLiteRtElementTypeUInt4)
    #expect(ElementType.float8E4M3FN.cType == kLiteRtElementTypeFloat8E4M3FN)
    #expect(ElementType.float8E5M2.cType == kLiteRtElementTypeFloat8E5M2)
  }

  @Test func `tensor type with quantization`() {
    let layout = Layout(dimensions: [1, 10])
    let pt = QuantizationPerTensor(scale: 0.1, zeroPoint: 0)
    let tensorType = TensorType(elementType: .int8, layout: layout, quantization: .perTensor(pt))

    #expect(tensorType.elementType == .int8)
    #expect(tensorType.layout == layout)
    #expect(tensorType.quantization == .perTensor(pt))
  }

  @Test func `quantization queries on simple model`() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env)

    let inputCount = try compiledModel.inputCount()
    #expect(inputCount > 0)

    for i in 0..<inputCount {
      let quant = try compiledModel.inputTensorQuantization(inputIndex: i)
      #expect(quant == .none)
      #expect(quant.typeID == .none)
      #expect(quant.isQuantized == false)
      #expect(quant.perTensor == nil)
      #expect(quant.perChannel == nil)
      #expect(quant.blockWise == nil)

      let inputType = try compiledModel.inputTensorType(inputIndex: i)
      #expect(inputType.quantization == Quantization.none)

      let inputName = try compiledModel.inputName(inputIndex: i)
      let quantByName = try compiledModel.inputTensorQuantization(inputName: inputName)
      #expect(quantByName == .none)

      let inputTypeByName = try compiledModel.inputTensorType(inputName: inputName)
      #expect(inputTypeByName.quantization == Quantization.none)
    }

    let outputCount = try compiledModel.outputCount()
    #expect(outputCount > 0)

    for i in 0..<outputCount {
      let quant = try compiledModel.outputTensorQuantization(outputIndex: i)
      #expect(quant == .none)
      #expect(quant.typeID == .none)
      #expect(quant.isQuantized == false)
      #expect(quant.perTensor == nil)
      #expect(quant.perChannel == nil)
      #expect(quant.blockWise == nil)

      let outputType = try compiledModel.outputTensorType(outputIndex: i)
      #expect(outputType.quantization == Quantization.none)

      let outputName = try compiledModel.outputName(outputIndex: i)
      let quantByName = try compiledModel.outputTensorQuantization(outputName: outputName)
      #expect(quantByName == .none)

      let outputTypeByName = try compiledModel.outputTensorType(outputName: outputName)
      #expect(outputTypeByName.quantization == Quantization.none)
    }
  }

  @Test func `quantization queries on quantized model`() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_quantized_ops.tflite"
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env)

    let inputCount = try compiledModel.inputCount()
    #expect(inputCount > 0)

    for i in 0..<inputCount {
      let quant = try compiledModel.inputTensorQuantization(inputIndex: i)
      let inputType = try compiledModel.inputTensorType(inputIndex: i)
      #expect(inputType.quantization == quant)

      let inputName = try compiledModel.inputName(inputIndex: i)
      let quantByName = try compiledModel.inputTensorQuantization(inputName: inputName)
      #expect(quantByName == quant)

      let inputTypeByName = try compiledModel.inputTensorType(inputName: inputName)
      #expect(inputTypeByName.quantization == quant)

      switch quant {
      case .none:
        #expect(quant.isQuantized == false)
        #expect(quant.typeID == .none)
        #expect(quant.perTensor == nil)
        #expect(quant.perChannel == nil)
        #expect(quant.blockWise == nil)
      case .perTensor(let pt):
        #expect(quant.isQuantized == true)
        #expect(quant.typeID == .perTensor)
        #expect(quant.perTensor == pt)
        #expect(quant.perChannel == nil)
        #expect(quant.blockWise == nil)
      case .perChannel(let pc):
        #expect(quant.isQuantized == true)
        #expect(quant.typeID == .perChannel)
        #expect(quant.perChannel == pc)
        #expect(quant.perTensor == nil)
        #expect(quant.blockWise == nil)
      case .blockWise(let bw):
        #expect(quant.isQuantized == true)
        #expect(quant.typeID == .blockWise)
        #expect(quant.blockWise == bw)
        #expect(quant.perTensor == nil)
        #expect(quant.perChannel == nil)
      }
    }

    let outputCount = try compiledModel.outputCount()
    #expect(outputCount > 0)

    for i in 0..<outputCount {
      let quant = try compiledModel.outputTensorQuantization(outputIndex: i)
      let outputType = try compiledModel.outputTensorType(outputIndex: i)
      #expect(outputType.quantization == quant)

      let outputName = try compiledModel.outputName(outputIndex: i)
      let quantByName = try compiledModel.outputTensorQuantization(outputName: outputName)
      #expect(quantByName == quant)

      let outputTypeByName = try compiledModel.outputTensorType(outputName: outputName)
      #expect(outputTypeByName.quantization == quant)
    }
  }
}
