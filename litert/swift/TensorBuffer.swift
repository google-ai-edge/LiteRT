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
#if canImport(Metal)
import Metal
#endif
import LiteRtC

/// TensorBuffer represents the raw memory where tensor data is stored.
public class TensorBuffer {
  let handle: LiteRtTensorBuffer

  /// Requirements for allocating a TensorBuffer.
  public class Requirements {
    let handle: LiteRtTensorBufferRequirements
    let owned: Bool

    init(handle: LiteRtTensorBufferRequirements, owned: Bool = true) {
      self.handle = handle
      self.owned = owned
    }

    deinit {
      if owned {
        LiteRtDestroyTensorBufferRequirements(handle)
      }
    }

    public var bufferSize: Int {
      var size: Int = 0
      LiteRtGetTensorBufferRequirementsBufferSize(handle, &size)
      return size
    }

    public var supportedTypes: [TensorBufferType] {
      var numTypes: Int32 = 0
      LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(handle, &numTypes)
      var types: [TensorBufferType] = []
      types.reserveCapacity(Int(numTypes))
      for i in 0..<numTypes {
        var type: LiteRtTensorBufferType = kLiteRtTensorBufferTypeUnknown
        LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(handle, i, &type)
        if let swiftType = TensorBufferType(rawValue: Int(type.rawValue)) {
          types.append(swiftType)
        }
      }
      return types
    }
  }

  public let tensorType: TensorType

  init(handle: LiteRtTensorBuffer, tensorType: TensorType) {
    self.handle = handle
    self.tensorType = tensorType
  }



#if canImport(Metal)
  /// Initializes a TensorBuffer from an existing Metal memory buffer, enabling zero-copy on Apple hardware.
  /// - Parameters:
  ///   - tensorType: The type of the tensor.
  ///   - environment: The LiteRT environment, required for hardware contexts.
  ///   - metalBuffer: The existing Metal buffer.
  ///   - deallocator: An optional closure to execute when the tensor buffer is destroyed.
  public init(
    tensorType: TensorType,
    environment: Environment,
    metalBuffer: MTLBuffer,
    deallocator: LiteRtMetalDeallocator? = nil
  ) throws {
    self.tensorType = tensorType
    var typeCopy = tensorType.cType
    var buffer: LiteRtTensorBuffer?

    let metalPtr = Unmanaged.passUnretained(metalBuffer).toOpaque()

    let status = LiteRtCreateTensorBufferFromMetalMemory(
      environment.handle,
      &typeCopy,
      kLiteRtTensorBufferTypeMetalBuffer,
      metalPtr,
      metalBuffer.length,
      deallocator,
      &buffer
    )

    guard status == kLiteRtStatusOk, let buffer = buffer else {
      throw LiteRtError.memoryAllocationFailure("Failed to wrap Metal buffer")
    }
    self.handle = buffer
  }
#endif


  public init(tensorType: TensorType, environment: Environment) throws {
    self.tensorType = tensorType
    var buffer: LiteRtTensorBuffer?
    var typeCopy = tensorType.cType
    // Use kLiteRtTensorBufferTypeHostMemory = 1
    let status = LiteRtCreateManagedTensorBuffer(
      environment.handle,
      kLiteRtTensorBufferTypeHostMemory,
      &typeCopy,
      Int(tensorType.byteSize),
      &buffer
    )

    guard status == kLiteRtStatusOk, let buffer = buffer else {
      throw LiteRtError.memoryAllocationFailure("Failed to create tensor buffer")
    }
    self.handle = buffer
  }

  deinit {
    LiteRtDestroyTensorBuffer(handle)
  }
}

extension TensorBuffer {
  public func read<T>() throws -> [T] {
    var ptr: UnsafeMutableRawPointer?
    // Lock Read = 0
    let status = LiteRtLockTensorBuffer(handle, &ptr, LiteRtTensorBufferLockMode(rawValue: 0))
    guard status == kLiteRtStatusOk, let ptr = ptr else {
      throw status.toError(message: "Failed to lock buffer for reading")
        ?? .unknown("Lock failed")
    }

    defer {
      LiteRtUnlockTensorBuffer(handle)
    }

    var packedSize: Int = 0
    let sizeStatus = LiteRtGetTensorBufferPackedSize(handle, &packedSize)
    guard sizeStatus == kLiteRtStatusOk else {
      throw sizeStatus.toError(message: "Failed to get packed size")
        ?? .unknown("Failed to get packed size")
    }

    let count = packedSize / MemoryLayout<T>.size
    let bufferPtr = ptr.bindMemory(to: T.self, capacity: count)
    return Array(UnsafeBufferPointer(start: bufferPtr, count: count))
  }

  public func write<T>(_ data: [T]) throws {
    var ptr: UnsafeMutableRawPointer?
    // Lock Write = 1
    let status = LiteRtLockTensorBuffer(handle, &ptr, LiteRtTensorBufferLockMode(rawValue: 1))
    guard status == kLiteRtStatusOk, let ptr = ptr else {
      throw status.toError(message: "Failed to lock buffer for writing")
        ?? .unknown("Lock failed")
    }

    defer {
      LiteRtUnlockTensorBuffer(handle)
    }

    var packedSize: Int = 0
    let sizeStatus = LiteRtGetTensorBufferPackedSize(handle, &packedSize)
    guard sizeStatus == kLiteRtStatusOk else {
      throw sizeStatus.toError(message: "Failed to get packed size")
        ?? .unknown("Failed to get packed size")
    }

    let count = packedSize / MemoryLayout<T>.size
    guard data.count <= count else {
      throw LiteRtError.invalidArgument(
        "Data size \(data.count) exceeds buffer capacity \(count) (packedSize: \(packedSize))")
    }

    let bufferPtr = ptr.bindMemory(to: T.self, capacity: count)
    if !data.isEmpty {
      data.withUnsafeBufferPointer { dataPtr in
        if let baseAddress = dataPtr.baseAddress {
          bufferPtr.update(from: baseAddress, count: data.count)
        }
      }
    }
  }
}

/// Types of tensor buffers.
public enum TensorBufferType: Int {
  // LINT.IfChange(tensor_buffer_types)
  case unknown = 0
  case hostMemory = 1
  // case ahwb = 2
  // case ion = 3
  // case dmaBuf = 4
  // case fastRpc = 5
  // case glBuffer = 6
  // case glTexture = 7

  // 10-19 are reserved for OpenCL memory objects.

  // 20-29 are reserved for WebGpu memory objects.
  case webGpuBuffer = 20
  case webGpuBufferFp16 = 21
  case webGpuTexture = 22
  case webGpuTextureFp16 = 23
  case webGpuImageBuffer = 24
  case webGpuImageBufferFp16 = 25
  case webGpuBufferPacked = 26

  // 30-39 are reserved for Metal memory objects.
  case metalBuffer = 30
  case metalBufferFp16 = 31
  case metalTexture = 32
  case metalTextureFp16 = 33
  case metalBufferPacked = 34

  // 40-49 are reserved for Vulkan memory objects.

  // LINT.ThenChange(../c/litert_tensor_buffer_types.h:tensor_buffer_types)
}
