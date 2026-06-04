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

#if canImport(Metal)
import Metal
#endif

final class TensorBufferTests: XCTestCase {
  func testCustomHostBuffer() throws {
    let alignment = 64
    let byteCount = 8 // 2 * 4 bytes for Float
    let pointer = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: alignment)
    defer {
      pointer.deallocate()
    }

    let floatPointer = pointer.bindMemory(to: Float.self, capacity: 2)
    floatPointer[0] = 1.2
    floatPointer[1] = 3.4

    let layout = Layout(dimensions: [2])
    let tensorType = TensorType(elementType: .float32, layout: layout)

    let customBuf = try TensorBuffer(
      hostBufferAddressAddress: pointer,
      size: byteCount,
      tensorType: tensorType
    )
    XCTAssertEqual(customBuf.size, byteCount)
    let readBack: [Float] = try customBuf.read()
    XCTAssertEqual(readBack, [1.2, 3.4])
  }

  func testManualLocking() throws {
    let env = try Environment()
    let tensorType = TensorType(elementType: .float32, layout: Layout(dimensions: [2]))
    let buf = try TensorBuffer(
      environment: env,
      bufferType: .hostMemory,
      tensorType: tensorType,
      size: 8
    )
    XCTAssertEqual(buf.offset, 0)
    let addr = try buf.lock(mode: .write)
    let typedPtr = addr.bindMemory(to: Float.self, capacity: 2)
    typedPtr[0] = 5.5
    typedPtr[1] = 6.6
    try buf.unlock()

    let readBack: [Float] = try buf.read()
    XCTAssertEqual(readBack, [5.5, 6.6])
  }

#if canImport(Metal)
  // TODO: Enable this test when GPU accelerator dynamic loading is fixed for Swift package.
  func disabled_testMetalBuffer() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      print("Metal is not supported on this device/environment, skipping test.")
      return
    }

    let env = try Environment()
    let size = 1024
    guard let mtlBuffer = device.makeBuffer(length: size, options: []) else {
      XCTFail("Failed to create MTLBuffer")
      return
    }

    let layout = Layout(dimensions: [256]) // 256 * 4 bytes = 1024 bytes for Float32
    let tensorType = TensorType(elementType: .float32, layout: layout)

    let rawPointer = Unmanaged.passUnretained(mtlBuffer).toOpaque()

    let customBuf = try TensorBuffer(
      metalBuffer: rawPointer,
      size: size,
      tensorType: tensorType,
      bufferType: .metalBuffer,
      environment: env
    )
    XCTAssertEqual(customBuf.size, size)
    XCTAssertEqual(customBuf.type, .metalBuffer)

    let retrievedPointer = try customBuf.metalMemory()
    XCTAssertEqual(retrievedPointer, rawPointer)
  }
#endif
}
