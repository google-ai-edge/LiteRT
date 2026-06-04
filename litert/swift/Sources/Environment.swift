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

public final class Environment {
  internal let cEnvironment: LiteRtEnvironment

  public enum Option {
    case compilerPluginLibraryDir(String)
    case dispatchLibraryDir(String)
    case compilerCacheDir(String)
    case runtimeLibraryDir(String)
    case autoRegisterAccelerators(Int64)
    case minLoggerSeverity(Int64)
    case compilerCacheMaxConfigsPerModel(Int64)
    case compilerCacheMaxTotalSize(Int64)
    case metalDevice(UnsafeRawPointer)
    case metalCommandQueue(UnsafeRawPointer)

    internal var tag: LiteRtEnvOptionTag {
      switch self {
      case .compilerPluginLibraryDir: return kLiteRtEnvOptionTagCompilerPluginLibraryDir
      case .dispatchLibraryDir: return kLiteRtEnvOptionTagDispatchLibraryDir
      case .compilerCacheDir: return kLiteRtEnvOptionTagCompilerCacheDir
      case .runtimeLibraryDir: return kLiteRtEnvOptionTagRuntimeLibraryDir
      case .autoRegisterAccelerators: return kLiteRtEnvOptionTagAutoRegisterAccelerators
      case .minLoggerSeverity: return kLiteRtEnvOptionTagMinLoggerSeverity
      case .compilerCacheMaxConfigsPerModel:
        return kLiteRtEnvOptionTagCompilerCacheMaxConfigsPerModel
      case .compilerCacheMaxTotalSize: return kLiteRtEnvOptionTagCompilerCacheMaxTotalSize
      case .metalDevice: return kLiteRtEnvOptionTagMetalDevice
      case .metalCommandQueue: return kLiteRtEnvOptionTagMetalCommandQueue
      }
    }
  }

  public init(options: [Option] = []) throws {
    var environment: LiteRtEnvironment?
    var cOptions: [LiteRtEnvOption] = []

    // Recursive wrapper helper to keep C-strings alive until LiteRtCreateEnvironment finishes executing.
    func buildAndRun(index: Int) throws {
      if index == options.count {
        let status: LiteRtStatus
        if cOptions.isEmpty {
          status = LiteRtCreateEnvironment(0, nil, &environment)
        } else {
          status = LiteRtCreateEnvironment(Int32(cOptions.count), cOptions, &environment)
        }
        try checkStatus(status)
        return
      }

      let option = options[index]
      switch option {
      case .compilerPluginLibraryDir(let path),
        .dispatchLibraryDir(let path),
        .compilerCacheDir(let path),
        .runtimeLibraryDir(let path):
        try path.withCString { cStr in
          var anyVal = LiteRtAny()
          anyVal.type = kLiteRtAnyTypeString
          anyVal.str_value = cStr
          cOptions.append(LiteRtEnvOption(tag: option.tag, value: anyVal))
          try buildAndRun(index: index + 1)
          cOptions.removeLast()
        }
      case .autoRegisterAccelerators(let mask):
        var anyVal = LiteRtAny()
        anyVal.type = kLiteRtAnyTypeInt
        anyVal.int_value = mask
        cOptions.append(LiteRtEnvOption(tag: option.tag, value: anyVal))
        try buildAndRun(index: index + 1)
        cOptions.removeLast()
      case .minLoggerSeverity(let level):
        var anyVal = LiteRtAny()
        anyVal.type = kLiteRtAnyTypeInt
        anyVal.int_value = level
        cOptions.append(LiteRtEnvOption(tag: option.tag, value: anyVal))
        try buildAndRun(index: index + 1)
        cOptions.removeLast()
      case .compilerCacheMaxConfigsPerModel(let limit):
        var anyVal = LiteRtAny()
        anyVal.type = kLiteRtAnyTypeInt
        anyVal.int_value = limit
        cOptions.append(LiteRtEnvOption(tag: option.tag, value: anyVal))
        try buildAndRun(index: index + 1)
        cOptions.removeLast()
      case .compilerCacheMaxTotalSize(let size):
        var anyVal = LiteRtAny()
        anyVal.type = kLiteRtAnyTypeInt
        anyVal.int_value = size
        cOptions.append(LiteRtEnvOption(tag: option.tag, value: anyVal))
        try buildAndRun(index: index + 1)
        cOptions.removeLast()
      case .metalDevice(let ptr),
        .metalCommandQueue(let ptr):
        var anyVal = LiteRtAny()
        anyVal.type = kLiteRtAnyTypeVoidPtr
        anyVal.ptr_value = ptr
        cOptions.append(LiteRtEnvOption(tag: option.tag, value: anyVal))
        try buildAndRun(index: index + 1)
        cOptions.removeLast()
      }
    }

    try buildAndRun(index: 0)

    guard let environment else {
      throw LiteRtError.runtimeFailure
    }
    self.cEnvironment = environment
  }

  public func supportsFP16() throws -> Bool {
    var supported = false
    let status = LiteRtEnvironmentSupportsFP16(cEnvironment, &supported)
    try checkStatus(status)
    return supported
  }

  public func hasGpuEnvironment() -> Bool {
    var hasGpu = false
    LiteRtEnvironmentHasGpuEnvironment(cEnvironment, &hasGpu)
    return hasGpu
  }

  public func supportsClGlInterop() throws -> Bool {
    var supported = false
    let status = LiteRtEnvironmentSupportsClGlInterop(cEnvironment, &supported)
    try checkStatus(status)
    return supported
  }

  public func supportsAhwbClInterop() throws -> Bool {
    var supported = false
    let status = LiteRtEnvironmentSupportsAhwbClInterop(cEnvironment, &supported)
    try checkStatus(status)
    return supported
  }

  public func supportsAhwbGlInterop() throws -> Bool {
    var supported = false
    let status = LiteRtEnvironmentSupportsAhwbGlInterop(cEnvironment, &supported)
    try checkStatus(status)
    return supported
  }

  deinit {
    LiteRtDestroyEnvironment(cEnvironment)
  }
}
