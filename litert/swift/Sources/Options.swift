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

public enum Accelerator: Int32 {
  case none = 0
  case cpu = 1  // 1 << 0
  case gpu = 2  // 1 << 1
  case npu = 4  // 1 << 2
}

public final class Options {
  internal let cOptions: LiteRtOptions

  public init() throws {
    var options: LiteRtOptions?
    let status = LiteRtCreateOptions(&options)
    try checkStatus(status)
    guard let options else {
      throw LiteRtError.runtimeFailure
    }
    self.cOptions = options
  }

  deinit {
    LiteRtDestroyOptions(cOptions)
  }

  public func setHardwareAccelerators(_ accelerators: Set<Accelerator>) throws {
    var bitmask: Int32 = 0
    for accel in accelerators {
      bitmask |= accel.rawValue
    }
    let status = LiteRtSetOptionsHardwareAccelerators(cOptions, bitmask)
    try checkStatus(status)
  }

  public func hardwareAccelerators() throws -> Set<Accelerator> {
    var bitmask: Int32 = 0
    let status = LiteRtGetOptionsHardwareAccelerators(cOptions, &bitmask)
    try checkStatus(status)
    var accelerators: Set<Accelerator> = []
    if (bitmask & Accelerator.cpu.rawValue) != 0 {
      accelerators.insert(.cpu)
    }
    if (bitmask & Accelerator.gpu.rawValue) != 0 {
      accelerators.insert(.gpu)
    }
    if (bitmask & Accelerator.npu.rawValue) != 0 {
      accelerators.insert(.npu)
    }
    return accelerators
  }

  public func addExternalTensorBinding(
    signatureName: String,
    tensorName: String,
    data: UnsafeMutableRawPointer,
    sizeBytes: Int
  ) throws {
    let status = signatureName.withCString { cSigName in
      tensorName.withCString { cTensorName in
        LiteRtAddExternalTensorBinding(
          cOptions,
          cSigName,
          cTensorName,
          data,
          Int32(sizeBytes)
        )
      }
    }
    try checkStatus(status)
  }

  /// Adds a custom op kernel to the compilation options.
  ///
  /// - Parameters:
  ///   - name: The name of the custom op.
  ///   - version: The version of the custom op.
  ///   - kernel: A pointer to the custom op kernel implementation.
  ///   - userData: Optional user data to pass to the kernel.
  /// - Throws: `LiteRtError` if adding the custom op kernel fails.
  public func addCustomOpKernel(
    name: String,
    version: Int,
    kernel: UnsafePointer<LiteRtCustomOpKernel>,
    userData: UnsafeMutableRawPointer? = nil
  ) throws {
    let status = name.withCString { cName in
      LiteRtAddCustomOpKernelOption(
        cOptions,
        cName,
        Int32(version),
        kernel,
        userData
      )
    }
    try checkStatus(status)
  }

  /// Adds opaque options to the compilation options.
  ///
  /// - Parameter opaqueOptions: The opaque options to add.
  /// - Throws: `LiteRtError` if adding the opaque options fails.
  public func addOpaqueOptions(_ opaqueOptions: OpaqueOptions) throws {
    let status = LiteRtAddOpaqueOptions(cOptions, opaqueOptions.cOpaqueOptions)
    try checkStatus(status)
    opaqueOptions.owned = false
  }

  /// Retrieves the opaque options from the compilation options.
  ///
  /// - Returns: The opaque options if present, `nil` otherwise.
  /// - Throws: `LiteRtError` if retrieving the opaque options fails.
  public func opaqueOptions() throws -> OpaqueOptions? {
    var opaqueOptions: LiteRtOpaqueOptions?
    let status = LiteRtGetOpaqueOptions(cOptions, &opaqueOptions)
    try checkStatus(status)
    guard let opaqueOptions else {
      return nil
    }
    return OpaqueOptions(cOpaqueOptions: opaqueOptions, owned: false)
  }
}
