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

/// Errors that can occur during LiteRT operations.
public enum LiteRtError: Error {
  case invalidArgument(String)
  case memoryAllocationFailure(String)
  case runtimeFailure(String)
  case missingInputTensor(String)
  case unsupported(String)
  case notFound(String)
  case timeoutExpired(String)
  case wrongVersion(String)
  case unknown(String)
  case alreadyExists(String)
  case cancelled(String)
  case fileIO(String)
  case invalidFlatbuffer(String)
  case dynamicLoading(String)
  case serialization(String)
  case compilation(String)
  case internalError(String)
}

extension LiteRtStatus {
  func toError(message: String) -> LiteRtError? {
    switch self {
    case kLiteRtStatusOk: return nil
    case kLiteRtStatusErrorInvalidArgument: return .invalidArgument(message)
    case kLiteRtStatusErrorMemoryAllocationFailure: return .memoryAllocationFailure(message)
    case kLiteRtStatusErrorRuntimeFailure: return .runtimeFailure(message)
    case kLiteRtStatusErrorMissingInputTensor: return .missingInputTensor(message)
    case kLiteRtStatusErrorUnsupported: return .unsupported(message)
    case kLiteRtStatusErrorNotFound: return .notFound(message)
    case kLiteRtStatusErrorTimeoutExpired: return .timeoutExpired(message)
    case kLiteRtStatusErrorWrongVersion: return .wrongVersion(message)
    case kLiteRtStatusErrorAlreadyExists: return .alreadyExists(message)
    case kLiteRtStatusCancelled: return .cancelled(message)
    case kLiteRtStatusErrorFileIO: return .fileIO(message)
    case kLiteRtStatusErrorInvalidFlatbuffer: return .invalidFlatbuffer(message)
    case kLiteRtStatusErrorDynamicLoading: return .dynamicLoading(message)
    case kLiteRtStatusErrorSerialization: return .serialization(message)
    case kLiteRtStatusErrorCompilation: return .compilation(message)
    case kLiteRtStatusErrorUnknown: return .unknown(message)
    default: return .unknown("\(message) (Status: \(self))")
    }
  }
}

/// Hardware accelerators supported by LiteRT.
public struct HwAccelerators: OptionSet {
  public let rawValue: Int32
  public init(rawValue: Int32) { self.rawValue = rawValue }

  public static let none = HwAccelerators([])
  public static let cpu = HwAccelerators(rawValue: Int32(kLiteRtHwAcceleratorCpu.rawValue))
  public static let gpu = HwAccelerators(rawValue: Int32(kLiteRtHwAcceleratorGpu.rawValue))
  public static let npu = HwAccelerators(rawValue: Int32(kLiteRtHwAcceleratorNpu.rawValue))
}
