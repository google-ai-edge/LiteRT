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
import CLiteRT

/// The identifier for quantization scheme type union.
public enum QuantizationTypeID: UInt32, Equatable, Hashable, CustomStringConvertible {
  /// Tag for tensors without quantization.
  case none = 0

  /// Basic quantization, one set of q-params per tensor.
  case perTensor = 1

  /// Q-params for each element across a single dimension.
  case perChannel = 2

  /// Q-params across blocks of fixed size.
  case blockWise = 3

  public init?(cType: LiteRtQuantizationTypeId) {
    self.init(rawValue: cType.rawValue)
  }

  public var cType: LiteRtQuantizationTypeId {
    LiteRtQuantizationTypeId(rawValue: self.rawValue)
  }

  public var description: String {
    switch self {
    case .none: return "none"
    case .perTensor: return "perTensor"
    case .perChannel: return "perChannel"
    case .blockWise: return "blockWise"
    }
  }
}

/// Schema for tensors quantized with one set of quantization parameters.
public struct QuantizationPerTensor: Equatable, Hashable, CustomStringConvertible {
  /// Scaling factor.
  public let scale: Float

  /// The value that float:0 maps to in quantized space.
  public let zeroPoint: Int64

  public init(scale: Float, zeroPoint: Int64) {
    self.scale = scale
    self.zeroPoint = zeroPoint
  }

  public init(cQuantization: LiteRtQuantizationPerTensor) {
    self.scale = cQuantization.scale
    self.zeroPoint = cQuantization.zero_point
  }

  public var cQuantization: LiteRtQuantizationPerTensor {
    LiteRtQuantizationPerTensor(scale: scale, zero_point: zeroPoint)
  }

  public var description: String {
    "QuantizationPerTensor(scale: \(scale), zeroPoint: \(zeroPoint))"
  }
}

/// Schema for tensors quantized with one set of quantization parameters per channel.
public struct QuantizationPerChannel: Equatable, Hashable, CustomStringConvertible {
  /// The dimension along which the tensor is quantized.
  public let quantizedDimension: Int

  /// Scaling factors per channel.
  public let scales: [Float]

  /// Zero points per channel.
  public let zeroPoints: [Int64]

  /// Number of channels.
  public var channelCount: Int {
    scales.count
  }

  public init(quantizedDimension: Int, scales: [Float], zeroPoints: [Int64]) {
    self.quantizedDimension = quantizedDimension
    self.scales = scales
    self.zeroPoints = zeroPoints
  }

  public init(cQuantization: LiteRtQuantizationPerChannel) {
    self.quantizedDimension = Int(cQuantization.quantized_dimension)
    let count = Int(cQuantization.num_channels)
    if count > 0, let cScales = cQuantization.scales {
      self.scales = Array(UnsafeBufferPointer(start: cScales, count: count))
    } else {
      self.scales = []
    }
    if count > 0, let cZeroPoints = cQuantization.zero_points {
      self.zeroPoints = Array(UnsafeBufferPointer(start: cZeroPoints, count: count))
    } else {
      self.zeroPoints = []
    }
  }

  /// Executes a closure with a temporary `LiteRtQuantizationPerChannel` representation.
  public func withCQuantization<R>(
    _ body: (LiteRtQuantizationPerChannel) throws -> R
  ) rethrows -> R {
    var mutableScales = self.scales
    var mutableZeroPoints = self.zeroPoints
    return try mutableScales.withUnsafeMutableBufferPointer { scalesBuf in
      try mutableZeroPoints.withUnsafeMutableBufferPointer { zpBuf in
        let cQuant = LiteRtQuantizationPerChannel(
          quantized_dimension: Int32(quantizedDimension),
          num_channels: UInt64(scales.count),
          scales: scalesBuf.baseAddress,
          zero_points: zpBuf.baseAddress
        )
        return try body(cQuant)
      }
    }
  }

  public var description: String {
    "QuantizationPerChannel("
      + "quantizedDimension: \(quantizedDimension), "
      + "channelCount: \(channelCount), "
      + "scales: \(scales), "
      + "zeroPoints: \(zeroPoints))"
  }
}

/// Schema for tensors quantized across blocks of fixed size.
public struct QuantizationBlockWise: Equatable, Hashable, CustomStringConvertible {
  /// The scales tensor handle.
  public let scalesTensor: LiteRtTensor?

  /// The zero points tensor handle.
  public let zeroPointsTensor: LiteRtTensor?

  /// Block size for quantization.
  public let blockSize: Int

  public init(
    scalesTensor: LiteRtTensor? = nil,
    zeroPointsTensor: LiteRtTensor? = nil,
    blockSize: Int = 0
  ) {
    self.scalesTensor = scalesTensor
    self.zeroPointsTensor = zeroPointsTensor
    self.blockSize = blockSize
  }

  public init(cQuantization: LiteRtQuantizationBlockWise) {
    self.scalesTensor = cQuantization.scales
    self.zeroPointsTensor = cQuantization.zero_points
    self.blockSize = Int(cQuantization.block_size)
  }

  public var cQuantization: LiteRtQuantizationBlockWise {
    LiteRtQuantizationBlockWise(
      scales: scalesTensor,
      zero_points: zeroPointsTensor,
      block_size: Int32(blockSize)
    )
  }

  public var description: String {
    "QuantizationBlockWise("
      + "blockSize: \(blockSize), "
      + "scalesTensor: \(String(describing: scalesTensor)), "
      + "zeroPointsTensor: \(String(describing: zeroPointsTensor)))"
  }
}

/// Represents the quantization scheme and parameters of a tensor.
public enum Quantization: Equatable, CustomStringConvertible {
  /// Tensor is not quantized.
  case none

  /// Per-tensor quantization parameters.
  case perTensor(QuantizationPerTensor)

  /// Per-channel quantization parameters.
  case perChannel(QuantizationPerChannel)

  /// Block-wise quantization parameters.
  case blockWise(QuantizationBlockWise)

  /// The quantization type identifier.
  public var typeID: QuantizationTypeID {
    switch self {
    case .none: return .none
    case .perTensor: return .perTensor
    case .perChannel: return .perChannel
    case .blockWise: return .blockWise
    }
  }

  /// Whether this quantization scheme represents a quantized tensor.
  public var isQuantized: Bool {
    typeID != .none
  }

  /// The per-tensor quantization parameters, if applicable.
  public var perTensor: QuantizationPerTensor? {
    if case .perTensor(let q) = self { return q }
    return nil
  }

  /// The per-channel quantization parameters, if applicable.
  public var perChannel: QuantizationPerChannel? {
    if case .perChannel(let q) = self { return q }
    return nil
  }

  /// The block-wise quantization parameters, if applicable.
  public var blockWise: QuantizationBlockWise? {
    if case .blockWise(let q) = self { return q }
    return nil
  }

  public var description: String {
    switch self {
    case .none:
      return "Quantization.none"
    case .perTensor(let q):
      return "Quantization.perTensor(\(q))"
    case .perChannel(let q):
      return "Quantization.perChannel(\(q))"
    case .blockWise(let q):
      return "Quantization.blockWise(\(q))"
    }
  }
}
