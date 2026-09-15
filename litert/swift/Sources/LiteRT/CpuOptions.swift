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

/// Selects how CPU ops are executed in LiteRT.
public enum CpuKernelMode: Int32 {
  /// Use the CPU delegate pipeline. This is the default CPU mode.
  /// XNNPACK delegates CPU ops, and an enabled YNNPACK delegate runs first
  /// when it was compiled into the runtime.
  case delegate = 0

  /// Use LiteRT's built-in reference kernels instead of CPU delegates.
  case reference = 1

  /// Use LiteRT's built-in optimized kernels instead of CPU delegates.
  case builtin = 2

  internal init(cKernelMode: LiteRtCpuKernelMode) {
    switch cKernelMode {
    case kLiteRtCpuKernelModeDelegate:
      self = .delegate
    case kLiteRtCpuKernelModeReference:
      self = .reference
    case kLiteRtCpuKernelModeBuiltin:
      self = .builtin
    default:
      self = .delegate
    }
  }

  internal var cKernelMode: LiteRtCpuKernelMode {
    switch self {
    case .delegate:
      kLiteRtCpuKernelModeDelegate
    case .reference:
      kLiteRtCpuKernelModeReference
    case .builtin:
      kLiteRtCpuKernelModeBuiltin
    }
  }
}

/// Configuration options for CPU acceleration in LiteRT.
public final class CpuOptions: ConcreteOptions {
  internal let cCpuOptions: OpaquePointer

  /// Creates a new `CpuOptions` instance.
  ///
  /// - Throws: `LiteRtError` if creating the CPU options fails.
  public init() throws {
    var options: OpaquePointer?
    let status = LrtCreateCpuOptions(&options)
    try checkStatus(status)
    guard let options else {
      throw LiteRtError.runtimeFailure
    }
    self.cCpuOptions = options
  }

  deinit {
    LrtDestroyCpuOptions(cCpuOptions)
  }

  /// Sets how LiteRT should execute CPU ops.
  ///
  /// - Parameter mode: The kernel mode to use.
  /// - Throws: `LiteRtError` if setting the kernel mode fails.
  public func setKernelMode(_ mode: CpuKernelMode) throws {
    let status = LrtSetCpuOptionsKernelMode(cCpuOptions, mode.cKernelMode)
    try checkStatus(status)
  }

  /// Gets the CPU kernel mode that was set.
  ///
  /// - Returns: The configured `CpuKernelMode` if set, or `nil` if not explicitly configured.
  /// - Throws: `LiteRtError` if querying fails.
  public func kernelMode() throws -> CpuKernelMode? {
    var mode = kLiteRtCpuKernelModeDelegate
    let status = LrtGetCpuOptionsKernelMode(cCpuOptions, &mode)
    if status == kLiteRtStatusErrorNotFound {
      return nil
    }
    try checkStatus(status)
    return CpuKernelMode(cKernelMode: mode)
  }

  /// Enables or disables YNNPACK delegation for supported CPU ops before XNNPACK.
  ///
  /// - Parameter enable: Whether to enable YNNPACK.
  /// - Throws: `LiteRtError` if setting the option fails.
  public func setEnableYNNPack(_ enable: Bool) throws {
    let status = LrtSetCpuOptionsEnableYNNPack(cCpuOptions, enable)
    try checkStatus(status)
  }

  /// Gets whether YNNPACK delegation was enabled.
  ///
  /// - Returns: `true` if enabled, `false` if disabled, or `nil` if not explicitly configured.
  /// - Throws: `LiteRtError` if querying fails.
  public func enableYNNPack() throws -> Bool? {
    var enabled = false
    let status = LrtGetCpuOptionsEnableYNNPack(cCpuOptions, &enabled)
    if status == kLiteRtStatusErrorNotFound {
      return nil
    }
    try checkStatus(status)
    return enabled
  }

  /// Sets the number of CPU threads used by the CPU accelerator.
  ///
  /// - Parameter numThreads: Number of threads to use.
  /// - Throws: `LiteRtError` if setting the thread count fails.
  public func setNumThreads(_ numThreads: Int) throws {
    let status = LrtSetCpuOptionsNumThread(cCpuOptions, Int32(numThreads))
    try checkStatus(status)
  }

  /// Gets the number of CPU threads configured for the CPU accelerator.
  ///
  /// - Returns: The number of threads if set, or `nil` if not explicitly configured.
  /// - Throws: `LiteRtError` if querying fails.
  public func numThreads() throws -> Int? {
    var numThreads: Int32 = 0
    let status = LrtGetCpuOptionsNumThread(cCpuOptions, &numThreads)
    if status == kLiteRtStatusErrorNotFound {
      return nil
    }
    try checkStatus(status)
    return Int(numThreads)
  }

  /// Sets the XNNPack flags used by XNNPACK in delegate mode.
  ///
  /// - Parameter flags: XNNPack flags bitmask.
  /// - Throws: `LiteRtError` if setting the flags fails.
  public func setXNNPackFlags(_ flags: UInt32) throws {
    let status = LrtSetCpuOptionsXNNPackFlags(cCpuOptions, flags)
    try checkStatus(status)
  }

  /// Gets the XNNPack flags configured for delegate mode.
  ///
  /// - Returns: The flags bitmask if set, or `nil` if not explicitly configured.
  /// - Throws: `LiteRtError` if querying fails.
  public func xnnPackFlags() throws -> UInt32? {
    var flags: UInt32 = 0
    let status = LrtGetCpuOptionsXNNPackFlags(cCpuOptions, &flags)
    if status == kLiteRtStatusErrorNotFound {
      return nil
    }
    try checkStatus(status)
    return flags
  }

  /// Sets whether to hint at fully delegating to a single delegate so certain allocations can
  /// be skipped.
  ///
  /// - Parameter hint: Whether to enable single delegate hinting.
  /// - Throws: `LiteRtError` if setting the hint fails.
  public func setHintFullyDelegatedToSingleDelegate(_ hint: Bool) throws {
    let status = LrtSetCpuOptionsHintFullyDelegatedToSingleDelegate(cCpuOptions, hint)
    try checkStatus(status)
  }

  /// Sets the XNNPack weight cache file path used by XNNPACK in delegate mode.
  ///
  /// Note: Weight cache file path and file descriptor must not both be set.
  ///
  /// - Parameter path: File path for the weight cache.
  /// - Throws: `LiteRtError` if setting the path fails.
  public func setXNNPackWeightCachePath(_ path: String) throws {
    let status = path.withCString { cPath in
      LrtSetCpuOptionsXnnPackWeightCachePath(cCpuOptions, cPath)
    }
    try checkStatus(status)
  }

  /// Gets the XNNPack weight cache file path if configured.
  ///
  /// - Returns: The weight cache file path if set, or `nil` if not set.
  /// - Throws: `LiteRtError` if querying fails.
  public func xnnPackWeightCachePath() throws -> String? {
    var cPath: UnsafePointer<CChar>?
    let status = LrtGetCpuOptionsXnnPackWeightCachePath(cCpuOptions, &cPath)
    if status == kLiteRtStatusErrorNotFound {
      return nil
    }
    try checkStatus(status)
    guard let cPath else { return nil }
    return String(cString: cPath)
  }

  /// Sets the XNNPack weight cache file descriptor used by XNNPACK in delegate mode.
  ///
  /// Note: Weight cache file path and file descriptor must not both be set.
  ///
  /// - Parameter fd: Open file descriptor for the weight cache.
  /// - Throws: `LiteRtError` if setting the file descriptor fails.
  public func setXNNPackWeightCacheFileDescriptor(_ fd: Int32) throws {
    let status = LrtSetCpuOptionsXnnPackWeightCacheFileDescriptor(cCpuOptions, fd)
    try checkStatus(status)
  }

  /// Gets the XNNPack weight cache file descriptor if configured.
  ///
  /// - Returns: The weight cache file descriptor if set, or `nil` if not set.
  /// - Throws: `LiteRtError` if querying fails.
  public func xnnPackWeightCacheFileDescriptor() throws -> Int32? {
    var fd: Int32 = 0
    let status = LrtGetCpuOptionsXnnPackWeightCacheFileDescriptor(cCpuOptions, &fd)
    if status == kLiteRtStatusErrorNotFound {
      return nil
    }
    try checkStatus(status)
    return fd
  }

  /// Serializes these CPU options into an `OpaqueOptions` instance.
  ///
  /// - Returns: An `OpaqueOptions` instance containing the serialized CPU options.
  /// - Throws: `LiteRtError` if serializing fails.
  public func createOpaqueOptions() throws -> OpaqueOptions {
    var identifier: UnsafePointer<CChar>?
    var payload: UnsafeMutableRawPointer?
    var payloadDeleter: (@convention(c) (UnsafeMutableRawPointer?) -> Void)?
    let status = LrtGetOpaqueCpuOptionsData(
      cCpuOptions,
      &identifier,
      &payload,
      &payloadDeleter
    )
    try checkStatus(status)
    guard let identifier, let payload, let payloadDeleter else {
      throw LiteRtError.runtimeFailure
    }
    return try OpaqueOptions(
      identifier: String(cString: identifier),
      payload: payload,
      destructor: payloadDeleter
    )
  }
}
