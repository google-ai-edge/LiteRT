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

@testable import LiteRT

final class ModelTests: XCTestCase {
  func testSignatureKeys() throws {
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let model = try Model(filePath: modelPath)
    let keys = try model.signatureKeys()
    XCTAssertFalse(keys.isEmpty)
  }

  func testModelCreationFromFd() throws {
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let fileURL = URL(fileURLWithPath: modelPath)
    let fileHandle = try FileHandle(forReadingFrom: fileURL)
    let fd = fileHandle.fileDescriptor

    let fileSize = try fileHandle.seekToEnd()
    try fileHandle.seek(toOffset: 0)

    let model = try Model(fd: fd, offset: 0, size: Int(fileSize))
    XCTAssertNotNil(model)

    try fileHandle.close()
  }

  func testModelMetadata() throws {
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let model = try Model(filePath: modelPath)

    let metadataKey = "test_key"
    let metadataValue = try XCTUnwrap("test_value".data(using: .utf8))

    try model.addMetadata(key: metadataKey, data: metadataValue)

    let readValue = try model.metadata(key: metadataKey)
    XCTAssertEqual(readValue, metadataValue)
  }
}
