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
import LiteRtC

public enum ElementType: UInt32 {
  case none = 0
  case bool = 6
  case int2 = 20
  case int4 = 18
  case int8 = 9
  case int16 = 7
  case int32 = 2
  case int64 = 4
  case uint8 = 3
  case uint16 = 17
  case uint32 = 16
  case uint64 = 13
  case float16 = 10
  case bfloat16 = 19
  case float32 = 1
  case float64 = 11
  case complex64 = 8
  case complex128 = 12
  case tfResource = 14
  case tfString = 5
  case tfVariant = 15

  public init?(cType: LiteRtElementType) {
    self.init(rawValue: cType.rawValue)
  }

  public var cType: LiteRtElementType {
    LiteRtElementType(rawValue: self.rawValue)
  }
}

public struct Layout: Equatable {
  public let dimensions: [Int]
  public let strides: [Int]

  public init(dimensions: [Int], strides: [Int] = []) {
    self.dimensions = dimensions
    self.strides = strides
  }

  public var rank: Int {
    return dimensions.count
  }

  public var hasStrides: Bool {
    return !strides.isEmpty
  }

  public var elementCount: Int? {
    var count = 1
    for dim in dimensions {
      if dim < 0 {
        return nil // Dynamic dimension
      }
      count *= dim
    }
    return count
  }
}

public struct TensorType: Equatable {
  public let elementType: ElementType
  public let layout: Layout

  public init(elementType: ElementType, layout: Layout) {
    self.elementType = elementType
    self.layout = layout
  }
}

extension TensorType {
  internal func toCRankedTensorType() -> LiteRtRankedTensorType {
    var cType = LiteRtRankedTensorType()
    cType.element_type = elementType.cType

    var cLayout = LiteRtLayout()
    withUnsafeMutableBytes(of: &cLayout) { bytes in
      if let baseAddress = bytes.baseAddress {
        _ = memset(baseAddress, 0, bytes.count)
      }
    }
    let rank = layout.rank
    let hasStrides = layout.hasStrides

    // Encode first byte containing rank and hasStrides (C Bitfields)
    let firstByte = UInt8(rank & 0x7F) | (hasStrides ? 0x80 : 0x00)
    withUnsafeMutableBytes(of: &cLayout) { bytes in
      bytes[0] = firstByte
    }

    withUnsafeMutableBytes(of: &cLayout.dimensions) { rawDims in
      let dimsPointer = rawDims.bindMemory(to: Int32.self)
      for i in 0..<min(rank, 8) {
        dimsPointer[i] = Int32(layout.dimensions[i])
      }
    }

    if hasStrides {
      withUnsafeMutableBytes(of: &cLayout.strides) { rawStrides in
        let stridesPointer = rawStrides.bindMemory(to: UInt32.self)
        for i in 0..<min(layout.strides.count, 8) {
          stridesPointer[i] = UInt32(layout.strides[i])
        }
      }
    }

    cType.layout = cLayout
    return cType
  }

  internal init(cRankedType: LiteRtRankedTensorType) {
    let elementType = ElementType(cType: cRankedType.element_type) ?? .none
    let cLayout = cRankedType.layout

    // Decode first byte containing rank and hasStrides (C Bitfields)
    let firstByte = withUnsafeBytes(of: cLayout) { bytes in
      bytes[0]
    }
    let rank = Int(firstByte & 0x7F)
    let hasStrides = (firstByte & 0x80) != 0

    var dimensions: [Int] = []
    withUnsafeBytes(of: cLayout.dimensions) { rawDims in
      let dimsPointer = rawDims.bindMemory(to: Int32.self)
      for i in 0..<min(rank, 8) {
        dimensions.append(Int(dimsPointer[i]))
      }
    }

    var strides: [Int] = []
    if hasStrides {
      withUnsafeBytes(of: cLayout.strides) { rawStrides in
        let stridesPointer = rawStrides.bindMemory(to: UInt32.self)
        for i in 0..<min(rank, 8) {
          strides.append(Int(stridesPointer[i]))
        }
      }
    }

    self.init(elementType: elementType, layout: Layout(dimensions: dimensions, strides: strides))
  }
}
