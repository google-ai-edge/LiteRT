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

public final class Model {
  internal let cModel: LiteRtModel
  private let data: Data?

  public init(filePath: String, environment: Environment? = nil) throws {
    var model: LiteRtModel?
    let status = LiteRtCreateModelFromFile(environment?.cEnvironment, filePath, &model)
    try checkStatus(status)
    guard let model else {
      throw LiteRtError.runtimeFailure
    }
    self.cModel = model
    self.data = nil
  }

  public init(data: Data, environment: Environment? = nil) throws {
    var model: LiteRtModel?
    let status = data.withUnsafeBytes { rawBuffer -> LiteRtStatus in
      guard let baseAddr = rawBuffer.baseAddress else {
        return kLiteRtStatusErrorInvalidArgument
      }
      return LiteRtCreateModelFromBuffer(environment?.cEnvironment, baseAddr, rawBuffer.count, &model)
    }
    try checkStatus(status)
    guard let model else {
      throw LiteRtError.runtimeFailure
    }
    self.cModel = model
    // Retain the data to ensure the buffer remains valid for the lifetime of the model.
    self.data = data
  }

  public init(fd: Int32, offset: Int, size: Int, environment: Environment? = nil) throws {
    var model: LiteRtModel?
    let status = LiteRtCreateModelFromFd(environment?.cEnvironment, fd, size_t(offset), size_t(size), &model)
    try checkStatus(status)
    guard let model else {
      throw LiteRtError.runtimeFailure
    }
    self.cModel = model
    self.data = nil
  }

  public func metadata(key: String) throws -> Data {
    var metadataBuffer: UnsafeRawPointer?
    var metadataBufferSize: size_t = 0
    let status = key.withCString { cKey in
      LiteRtGetModelMetadata(cModel, cKey, &metadataBuffer, &metadataBufferSize)
    }
    try checkStatus(status)
    guard let metadataBuffer else {
      throw LiteRtError.notFound
    }
    return Data(bytes: metadataBuffer, count: Int(metadataBufferSize))
  }

  public func addMetadata(key: String, data: Data) throws {
    let status = key.withCString { cKey in
      data.withUnsafeBytes { rawBuffer in
        LiteRtAddModelMetadata(cModel, cKey, rawBuffer.baseAddress, rawBuffer.count)
      }
    }
    try checkStatus(status)
  }

  public func signatureKeys() throws -> [String] {
    var numSignatures: LiteRtParamIndex = 0
    var status = LiteRtGetNumModelSignatures(cModel, &numSignatures)
    try checkStatus(status)
    var keys: [String] = []
    for i in 0..<numSignatures {
      var signature: LiteRtSignature?
      status = LiteRtGetModelSignature(cModel, i, &signature)
      try checkStatus(status)
      if let signature {
        var cKey: UnsafePointer<CChar>?
        status = LiteRtGetSignatureKey(signature, &cKey)
        try checkStatus(status)
        if let cKey {
          keys.append(String(cString: cKey))
        }
      }
    }
    return keys
  }

  deinit {
    LiteRtDestroyModel(cModel)
  }
}
