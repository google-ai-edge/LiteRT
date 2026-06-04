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

public enum TensorBufferType: Int32 {
  case unknown = 0
  case hostMemory = 1
  case ahwb = 2
  case ion = 3
  case dmaBuf = 4
  case fastRpc = 5
  case glBuffer = 6
  case glTexture = 7
  case openClBuffer = 10
  case openClBufferFp16 = 11
  case openClTexture = 12
  case openClTextureFp16 = 13
  case openClBufferPacked = 14
  case openClImageBuffer = 15
  case openClImageBufferFp16 = 16
  case webGpuBuffer = 20
  case webGpuBufferFp16 = 21
  case webGpuTexture = 22
  case webGpuTextureFp16 = 23
  case webGpuImageBuffer = 24
  case webGpuImageBufferFp16 = 25
  case webGpuBufferPacked = 26
  case metalBuffer = 30
  case metalBufferFp16 = 31
  case metalTexture = 32
  case metalTextureFp16 = 33
  case metalBufferPacked = 34
  case vulkanBuffer = 40
  case vulkanBufferFp16 = 41
  case vulkanTexture = 42
  case vulkanTextureFp16 = 43
  case vulkanImageBuffer = 44
  case vulkanImageBufferFp16 = 45
  case vulkanBufferPacked = 46

  public init(cType: LiteRtTensorBufferType) {
    switch cType {
    case kLiteRtTensorBufferTypeHostMemory: self = .hostMemory
    case kLiteRtTensorBufferTypeAhwb: self = .ahwb
    case kLiteRtTensorBufferTypeIon: self = .ion
    case kLiteRtTensorBufferTypeDmaBuf: self = .dmaBuf
    case kLiteRtTensorBufferTypeFastRpc: self = .fastRpc
    case kLiteRtTensorBufferTypeGlBuffer: self = .glBuffer
    case kLiteRtTensorBufferTypeGlTexture: self = .glTexture
    case kLiteRtTensorBufferTypeOpenClBuffer: self = .openClBuffer
    case kLiteRtTensorBufferTypeOpenClBufferFp16: self = .openClBufferFp16
    case kLiteRtTensorBufferTypeOpenClTexture: self = .openClTexture
    case kLiteRtTensorBufferTypeOpenClTextureFp16: self = .openClTextureFp16
    case kLiteRtTensorBufferTypeOpenClBufferPacked: self = .openClBufferPacked
    case kLiteRtTensorBufferTypeOpenClImageBuffer: self = .openClImageBuffer
    case kLiteRtTensorBufferTypeOpenClImageBufferFp16: self = .openClImageBufferFp16
    case kLiteRtTensorBufferTypeWebGpuBuffer: self = .webGpuBuffer
    case kLiteRtTensorBufferTypeWebGpuBufferFp16: self = .webGpuBufferFp16
    case kLiteRtTensorBufferTypeWebGpuTexture: self = .webGpuTexture
    case kLiteRtTensorBufferTypeWebGpuTextureFp16: self = .webGpuTextureFp16
    case kLiteRtTensorBufferTypeWebGpuImageBuffer: self = .webGpuImageBuffer
    case kLiteRtTensorBufferTypeWebGpuImageBufferFp16: self = .webGpuImageBufferFp16
    case kLiteRtTensorBufferTypeWebGpuBufferPacked: self = .webGpuBufferPacked
    case kLiteRtTensorBufferTypeMetalBuffer: self = .metalBuffer
    case kLiteRtTensorBufferTypeMetalBufferFp16: self = .metalBufferFp16
    case kLiteRtTensorBufferTypeMetalTexture: self = .metalTexture
    case kLiteRtTensorBufferTypeMetalTextureFp16: self = .metalTextureFp16
    case kLiteRtTensorBufferTypeMetalBufferPacked: self = .metalBufferPacked
    case kLiteRtTensorBufferTypeVulkanBuffer: self = .vulkanBuffer
    case kLiteRtTensorBufferTypeVulkanBufferFp16: self = .vulkanBufferFp16
    case kLiteRtTensorBufferTypeVulkanTexture: self = .vulkanTexture
    case kLiteRtTensorBufferTypeVulkanTextureFp16: self = .vulkanTextureFp16
    case kLiteRtTensorBufferTypeVulkanImageBuffer: self = .vulkanImageBuffer
    case kLiteRtTensorBufferTypeVulkanImageBufferFp16: self = .vulkanImageBufferFp16
    case kLiteRtTensorBufferTypeVulkanBufferPacked: self = .vulkanBufferPacked
    default: self = .unknown
    }
  }

  public var cType: LiteRtTensorBufferType {
    return LiteRtTensorBufferType(rawValue: UInt32(self.rawValue))
  }
}

public enum TensorBufferLockMode {
  case read
  case write
  case readWrite

  public var cMode: LiteRtTensorBufferLockMode {
    switch self {
    case .read: return kLiteRtTensorBufferLockModeRead
    case .write: return kLiteRtTensorBufferLockModeWrite
    case .readWrite: return kLiteRtTensorBufferLockModeReadWrite
    }
  }
}

public final class TensorBufferRequirements {
  internal let cRequirements: LiteRtTensorBufferRequirements
  internal let owned: Bool

  public var bufferSize: Int {
    var size: size_t = 0
    let status = LiteRtGetTensorBufferRequirementsBufferSize(cRequirements, &size)
    guard status == kLiteRtStatusOk else { return 0 }
    return Int(size)
  }

  public var alignment: Int {
    var alignmentSize: size_t = 0
    let status = LiteRtGetTensorBufferRequirementsAlignment(cRequirements, &alignmentSize)
    guard status == kLiteRtStatusOk else { return 0 }
    return Int(alignmentSize)
  }

  public var supportedTypes: [TensorBufferType] {
    var numTypes: Int32 = 0
    var status = LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(cRequirements, &numTypes)
    guard status == kLiteRtStatusOk else { return [] }

    var types: [TensorBufferType] = []
    for i in 0..<numTypes {
      var cType = kLiteRtTensorBufferTypeUnknown
      status = LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(cRequirements, i, &cType)
      if status == kLiteRtStatusOk {
        types.append(TensorBufferType(cType: cType))
      }
    }
    return types
  }

  public var strides: [Int] {
    var numStrides: Int32 = 0
    var cStrides: UnsafePointer<UInt32>?
    let status = LiteRtGetTensorBufferRequirementsStrides(cRequirements, &numStrides, &cStrides)
    guard status == kLiteRtStatusOk, let cStrides else { return [] }
    var result: [Int] = []
    for i in 0..<Int(numStrides) {
      result.append(Int(cStrides[i]))
    }
    return result
  }

  internal init(cRequirements: LiteRtTensorBufferRequirements, owned: Bool = true) {
    self.cRequirements = cRequirements
    self.owned = owned
  }

  deinit {
    if owned {
      LiteRtDestroyTensorBufferRequirements(cRequirements)
    }
  }
}

public final class TensorBuffer {
  internal let cTensorBuffer: LiteRtTensorBuffer
  internal let owned: Bool

  public var size: Int {
    var bufferSize: size_t = 0
    let status = LiteRtGetTensorBufferPackedSize(cTensorBuffer, &bufferSize)
    guard status == kLiteRtStatusOk else { return 0 }
    return Int(bufferSize)
  }

  public var bufferSize: Int {
    var bufferSize: size_t = 0
    let status = LiteRtGetTensorBufferSize(cTensorBuffer, &bufferSize)
    guard status == kLiteRtStatusOk else { return 0 }
    return Int(bufferSize)
  }

  public var offset: Int {
    var bufferOffset: size_t = 0
    let status = LiteRtGetTensorBufferOffset(cTensorBuffer, &bufferOffset)
    guard status == kLiteRtStatusOk else { return 0 }
    return Int(bufferOffset)
  }

  public var type: TensorBufferType {
    var cBufferType = kLiteRtTensorBufferTypeUnknown
    let status = LiteRtGetTensorBufferType(cTensorBuffer, &cBufferType)
    guard status == kLiteRtStatusOk else { return .unknown }
    return TensorBufferType(cType: cBufferType)
  }

  public var tensorType: TensorType? {
    var cTensorType = LiteRtRankedTensorType()
    let status = LiteRtGetTensorBufferTensorType(cTensorBuffer, &cTensorType)
    guard status == kLiteRtStatusOk else { return nil }
    return TensorType(cRankedType: cTensorType)
  }

  internal init(cTensorBuffer: LiteRtTensorBuffer, owned: Bool = true) {
    self.cTensorBuffer = cTensorBuffer
    self.owned = owned
  }

  /// Allocates a managed tensor buffer.
  ///
  /// - Parameters:
  ///   - environment: The runtime environment.
  ///   - bufferType: The type of tensor buffer to allocate.
  ///   - tensorType: The tensor type specification.
  ///   - size: The size of the buffer in bytes.
  /// - Throws: `LiteRtError` if allocation fails.
  public convenience init(
    environment: Environment,
    bufferType: TensorBufferType,
    tensorType: TensorType,
    size: Int
  ) throws {
    var cTensorType = tensorType.toCRankedTensorType()
    var tensorBuffer: LiteRtTensorBuffer?
    let status = LiteRtCreateManagedTensorBuffer(
      environment.cEnvironment,
      bufferType.cType,
      &cTensorType,
      size,
      &tensorBuffer
    )
    try checkStatus(status)
    guard let tensorBuffer else {
      throw LiteRtError.runtimeFailure
    }
    self.init(cTensorBuffer: tensorBuffer, owned: true)
  }

  /// Allocates a tensor buffer wrapping an existing host memory address.
  ///
  /// - Parameters:
  ///   - hostBufferAddressAddress: A pointer to the host memory buffer address.
  ///   - size: The size of the buffer in bytes.
  ///   - tensorType: The tensor type specification.
  /// - Throws: `LiteRtError` if buffer creation fails.
  public convenience init(
    hostBufferAddressAddress: UnsafeMutableRawPointer,
    size: Int,
    tensorType: TensorType
  ) throws {
    var cTensorType = tensorType.toCRankedTensorType()
    var tensorBuffer: LiteRtTensorBuffer?
    let status = LiteRtCreateTensorBufferFromHostMemory(
      &cTensorType,
      hostBufferAddressAddress,
      size,
      nil,  // Null deallocator implies we do not manage the address lifecycle
      &tensorBuffer
    )
    try checkStatus(status)
    guard let tensorBuffer else {
      throw LiteRtError.runtimeFailure
    }
    self.init(cTensorBuffer: tensorBuffer, owned: true)
  }

  /// Allocates a tensor buffer wrapping an existing Metal memory (`MTLBuffer`).
  ///
  /// - Parameters:
  ///   - metalBuffer: A pointer to the Metal buffer (`MTLBuffer`).
  ///   - size: The size of the buffer in bytes.
  ///   - tensorType: The tensor type specification.
  ///   - bufferType: The type of tensor buffer.
  ///   - environment: The runtime environment.
  /// - Throws: `LiteRtError` if buffer creation fails.
  public convenience init(
    metalBuffer: UnsafeMutableRawPointer,
    size: Int,
    tensorType: TensorType,
    bufferType: TensorBufferType,
    environment: Environment
  ) throws {
    var cTensorType = tensorType.toCRankedTensorType()
    var tensorBuffer: LiteRtTensorBuffer?
    let status = LiteRtCreateTensorBufferFromMetalMemory(
      environment.cEnvironment,
      &cTensorType,
      bufferType.cType,
      metalBuffer,
      size,
      nil,  // Null deallocator
      &tensorBuffer
    )
    try checkStatus(status)
    guard let tensorBuffer else {
      throw LiteRtError.runtimeFailure
    }
    self.init(cTensorBuffer: tensorBuffer, owned: true)
  }

  deinit {
    if owned {
      LiteRtDestroyTensorBuffer(cTensorBuffer)
    }
  }

  /// Locks the tensor buffer, writes data into it, and unlocks it.
  ///
  /// - Parameter data: An array of data elements to write into the buffer.
  /// - Throws: `LiteRtError` if locking, writing, or unlocking fails.
  public func write<T>(_ data: [T]) throws {
    var hostAddr: UnsafeMutableRawPointer?
    let status = LiteRtLockTensorBuffer(cTensorBuffer, &hostAddr, kLiteRtTensorBufferLockModeWrite)
    try checkStatus(status)
    guard let hostAddr else {
      throw LiteRtError.runtimeFailure
    }

    defer {
      LiteRtUnlockTensorBuffer(cTensorBuffer)
    }

    let byteCount = data.count * MemoryLayout<T>.stride
    guard byteCount <= size else {
      throw LiteRtError.invalidArgument
    }

    data.withUnsafeBytes { source in
      if let sourceAddr = source.baseAddress {
        hostAddr.copyMemory(from: sourceAddr, byteCount: byteCount)
      }
    }
  }

  /// Locks the tensor buffer, reads its contents into an array, and unlocks it.
  ///
  /// - Returns: An array containing the data read from the buffer.
  /// - Throws: `LiteRtError` if locking, reading, or unlocking fails.
  public func read<T>() throws -> [T] {
    var hostAddr: UnsafeMutableRawPointer?
    let status = LiteRtLockTensorBuffer(cTensorBuffer, &hostAddr, kLiteRtTensorBufferLockModeRead)
    try checkStatus(status)
    guard let hostAddr else {
      throw LiteRtError.runtimeFailure
    }

    defer {
      LiteRtUnlockTensorBuffer(cTensorBuffer)
    }

    let elementCount = size / MemoryLayout<T>.stride
    let typedPointer = hostAddr.bindMemory(to: T.self, capacity: elementCount)
    return Array(UnsafeBufferPointer(start: typedPointer, count: elementCount))
  }

  /// Locks the tensor buffer and obtains the backing host address pointer.
  ///
  /// - Parameter mode: The lock mode (read, write, or read-write).
  /// - Returns: A pointer to the backing host address.
  /// - Throws: `LiteRtError` if locking fails.
  public func lock(mode: TensorBufferLockMode) throws -> UnsafeMutableRawPointer {
    var hostAddr: UnsafeMutableRawPointer?
    let status = LiteRtLockTensorBuffer(cTensorBuffer, &hostAddr, mode.cMode)
    try checkStatus(status)
    guard let hostAddr else {
      throw LiteRtError.runtimeFailure
    }
    return hostAddr
  }

  /// Unlocks the tensor buffer.
  ///
  /// - Throws: `LiteRtError` if unlocking fails.
  public func unlock() throws {
    let status = LiteRtUnlockTensorBuffer(cTensorBuffer)
    try checkStatus(status)
  }

  /// Gets the backing Metal memory handle.
  ///
  /// - Returns: A pointer to the backing Metal memory handle.
  /// - Throws: `LiteRtError` if retrieving the Metal memory fails.
  public func metalMemory() throws -> UnsafeMutableRawPointer {
    var hwHandle: HwMemoryHandle?
    let status = LiteRtGetTensorBufferMetalMemory(cTensorBuffer, &hwHandle)
    try checkStatus(status)
    guard let hwHandle else {
      throw LiteRtError.runtimeFailure
    }
    return hwHandle
  }
}
