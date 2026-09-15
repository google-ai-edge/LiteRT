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

#if os(Linux)
  import SwiftGlibc
#else
  import Darwin
#endif

public enum LiteRtError: Error, CustomStringConvertible, Equatable {
  case invalidArgument
  case memoryAllocationFailure
  case runtimeFailure
  case missingInputTensor
  case unsupported
  case notFound
  case timeoutExpired
  case wrongVersion
  case unknown
  case alreadyExists
  case cancelled
  case fileIO
  case invalidFlatbuffer
  case dynamicLoading
  case serialization
  case compilation
  case indexOutOfBounds
  case invalidIrType
  case invalidGraphInvariant
  case graphModification
  case invalidToolConfig
  case legalizeNoMatch
  case invalidLegalization
  case patternNoMatch
  case invalidTransformation
  case unsupportedRuntimeVersion
  case unsupportedCompilerVersion
  case incompatibleByteCodeVersion
  case unsupportedOpShapeInferer
  case shapeInferenceFailed

  public init(status: LiteRtStatus) {
    switch status {
    case kLiteRtStatusErrorInvalidArgument: self = .invalidArgument
    case kLiteRtStatusErrorMemoryAllocationFailure: self = .memoryAllocationFailure
    case kLiteRtStatusErrorRuntimeFailure: self = .runtimeFailure
    case kLiteRtStatusErrorMissingInputTensor: self = .missingInputTensor
    case kLiteRtStatusErrorUnsupported: self = .unsupported
    case kLiteRtStatusErrorNotFound: self = .notFound
    case kLiteRtStatusErrorTimeoutExpired: self = .timeoutExpired
    case kLiteRtStatusErrorWrongVersion: self = .wrongVersion
    case kLiteRtStatusErrorUnknown: self = .unknown
    case kLiteRtStatusErrorAlreadyExists: self = .alreadyExists
    case kLiteRtStatusCancelled: self = .cancelled
    case kLiteRtStatusErrorFileIO: self = .fileIO
    case kLiteRtStatusErrorInvalidFlatbuffer: self = .invalidFlatbuffer
    case kLiteRtStatusErrorDynamicLoading: self = .dynamicLoading
    case kLiteRtStatusErrorSerialization: self = .serialization
    case kLiteRtStatusErrorCompilation: self = .compilation
    case kLiteRtStatusErrorIndexOOB: self = .indexOutOfBounds
    case kLiteRtStatusErrorInvalidIrType: self = .invalidIrType
    case kLiteRtStatusErrorInvalidGraphInvariant: self = .invalidGraphInvariant
    case kLiteRtStatusErrorGraphModification: self = .graphModification
    case kLiteRtStatusErrorInvalidToolConfig: self = .invalidToolConfig
    case kLiteRtStatusLegalizeNoMatch: self = .legalizeNoMatch
    case kLiteRtStatusErrorInvalidLegalization: self = .invalidLegalization
    case kLiteRtStatusPatternNoMatch: self = .patternNoMatch
    case kLiteRtStatusInvalidTransformation: self = .invalidTransformation
    case kLiteRtStatusErrorUnsupportedRuntimeVersion: self = .unsupportedRuntimeVersion
    case kLiteRtStatusErrorUnsupportedCompilerVersion: self = .unsupportedCompilerVersion
    case kLiteRtStatusErrorIncompatibleByteCodeVersion: self = .incompatibleByteCodeVersion
    case kLiteRtStatusErrorUnsupportedOpShapeInferer: self = .unsupportedOpShapeInferer
    case kLiteRtStatusErrorShapeInferenceFailed: self = .shapeInferenceFailed
    default: self = .unknown
    }
  }

  public var description: String {
    switch self {
    case .invalidArgument: return "Invalid argument"
    case .memoryAllocationFailure: return "Memory allocation failure"
    case .runtimeFailure: return "Runtime failure"
    case .missingInputTensor: return "Missing input tensor"
    case .unsupported: return "Unsupported operation"
    case .notFound: return "Not found"
    case .timeoutExpired: return "Timeout expired"
    case .wrongVersion: return "Wrong version"
    case .unknown: return "Unknown error"
    case .alreadyExists: return "Already exists"
    case .cancelled: return "Cancelled"
    case .fileIO: return "File I/O error"
    case .invalidFlatbuffer: return "Invalid flatbuffer"
    case .dynamicLoading: return "Dynamic loading error"
    case .serialization: return "Serialization error"
    case .compilation: return "Compilation error"
    case .indexOutOfBounds: return "Index out of bounds"
    case .invalidIrType: return "Invalid IR type"
    case .invalidGraphInvariant: return "Invalid graph invariant"
    case .graphModification: return "Graph modification error"
    case .invalidToolConfig: return "Invalid tool config"
    case .legalizeNoMatch: return "Legalize no match"
    case .invalidLegalization: return "Invalid legalization"
    case .patternNoMatch: return "Pattern no match"
    case .invalidTransformation: return "Invalid transformation"
    case .unsupportedRuntimeVersion: return "Unsupported runtime version"
    case .unsupportedCompilerVersion: return "Unsupported compiler version"
    case .incompatibleByteCodeVersion: return "Incompatible byte code version"
    case .unsupportedOpShapeInferer: return "Unsupported op shape inferer"
    case .shapeInferenceFailed: return "Shape inference failed"
    }
  }
}

@usableFromInline
internal func checkStatus(_ status: LiteRtStatus) throws {
  guard status == kLiteRtStatusOk else {
    throw LiteRtError(status: status)
  }
}
