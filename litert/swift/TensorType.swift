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

/// The type of a tensor, including its element type and layout.
public struct TensorType: Equatable {
  public let elementType: ElementType
  public let layout: Layout

  public init(elementType: ElementType, layout: Layout) {
    self.elementType = elementType
    self.layout = layout
  }

  public var byteSize: Int {
    let elements = layout.dimensions.reduce(1, *)
    return elements * elementType.byteSize
  }
}

extension TensorType {
  /// Data type of tensor elements.
  public enum ElementType: Int, Equatable {
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

    public var byteSize: Int {
      switch self {
      case .bool, .int8, .uint8: return 1
      case .int16, .uint16, .float16, .bfloat16: return 2
      case .int32, .uint32, .float32: return 4
      case .int64, .uint64, .float64, .complex64: return 8
      case .complex128: return 16
      default: return 0
      }
    }
  }

  /// Layout of a tensor.
  public struct Layout: Equatable {
    public let dimensions: [Int]
    public let strides: [Int]

    public var rank: Int {
      return dimensions.count
    }

    public init(dimensions: [Int], strides: [Int] = []) {
      self.dimensions = dimensions
      self.strides = strides
    }
  }
}

extension TensorType {
  init(cType: LiteRtRankedTensorType) {
    self.elementType = ElementType(cType: cType.element_type)
    self.layout = Layout(cLayout: cType.layout)
  }

  var cType: LiteRtRankedTensorType {
    var type = LiteRtRankedTensorType()
    type.element_type = elementType.cType
    type.layout = layout.cLayout
    return type
  }
}

extension TensorType.ElementType {
  init(cType: LiteRtElementType) {
    self = TensorType.ElementType(rawValue: Int(cType.rawValue)) ?? .none
  }

  var cType: LiteRtElementType {
    return LiteRtElementType(rawValue: UInt32(self.rawValue))
  }
}

extension TensorType.Layout {
  init(cLayout: LiteRtLayout) {
    var dims = [Int]()
    withUnsafeBytes(of: cLayout.dimensions) { ptr in
      let dimPtr = ptr.bindMemory(to: UInt32.self)
      for i in 0..<Int(cLayout.rank) {
        dims.append(Int(dimPtr[i]))
      }
    }
    self.dimensions = dims

    var strides = [Int]()
    if cLayout.has_strides {
      withUnsafeBytes(of: cLayout.strides) { ptr in
        let stridePtr = ptr.bindMemory(to: UInt32.self)
        for i in 0..<Int(cLayout.rank) {
          strides.append(Int(stridePtr[i]))
        }
      }
    }
    self.strides = strides
  }

  var cLayout: LiteRtLayout {
    var layout = LiteRtLayout()
    layout.rank = UInt32(dimensions.count)
    layout.has_strides = !strides.isEmpty
    withUnsafeMutableBytes(of: &layout.dimensions) { ptr in
      let dimPtr = ptr.bindMemory(to: Int32.self)
      for (i, dim) in dimensions.enumerated() {
        if i < dimPtr.count {
          dimPtr[i] = Int32(dim)
        }
      }
    }
    if layout.has_strides {
      withUnsafeMutableBytes(of: &layout.strides) { ptr in
        let stridePtr = ptr.bindMemory(to: UInt32.self)
        for (i, stride) in strides.enumerated() {
          if i < stridePtr.count {
            stridePtr[i] = UInt32(stride)
          }
        }
      }
    }
    return layout
  }
}
