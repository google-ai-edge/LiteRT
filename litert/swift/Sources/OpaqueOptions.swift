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

public final class OpaqueOptions {
  internal var cOpaqueOptions: LiteRtOpaqueOptions?
  internal var owned: Bool

  public init(
    identifier: String,
    payload: UnsafeMutableRawPointer,
    destructor: @escaping @convention(c) (UnsafeMutableRawPointer?) -> Void = { _ in }
  ) throws {
    var options: LiteRtOpaqueOptions?
    let status = identifier.withCString { cId in
      LiteRtCreateOpaqueOptions(cId, payload, destructor, &options)
    }
    try checkStatus(status)
    guard let options else {
      throw LiteRtError.runtimeFailure
    }
    self.cOpaqueOptions = options
    self.owned = true
  }

  internal init(cOpaqueOptions: LiteRtOpaqueOptions?, owned: Bool = false) {
    self.cOpaqueOptions = cOpaqueOptions
    self.owned = owned
  }

  deinit {
    if owned, let opts = cOpaqueOptions {
      LiteRtDestroyOpaqueOptions(opts)
    }
  }

  public var identifier: String {
    guard let opts = cOpaqueOptions else { return "" }
    var cStr: UnsafePointer<CChar>?
    let status = LiteRtGetOpaqueOptionsIdentifier(opts, &cStr)
    guard status == kLiteRtStatusOk, let cStr else {
      return ""
    }
    return String(cString: cStr)
  }

  public var data: UnsafeMutableRawPointer? {
    guard let opts = cOpaqueOptions else { return nil }
    var ptr: UnsafeMutableRawPointer?
    let status = LiteRtGetOpaqueOptionsData(opts, &ptr)
    guard status == kLiteRtStatusOk else {
      return nil
    }
    return ptr
  }

  public func findData(identifier: String) -> UnsafeMutableRawPointer? {
    guard let opts = cOpaqueOptions else { return nil }
    var ptr: UnsafeMutableRawPointer?
    let status = identifier.withCString { cId in
      LiteRtFindOpaqueOptionsData(opts, cId, &ptr)
    }
    guard status == kLiteRtStatusOk else {
      return nil
    }
    return ptr
  }

  public func next() -> OpaqueOptions? {
    guard let opts = cOpaqueOptions else { return nil }
    var nextPtr: LiteRtOpaqueOptions? = opts
    let status = LiteRtGetNextOpaqueOptions(&nextPtr)
    guard status == kLiteRtStatusOk, let nextPtr else {
      return nil
    }
    return OpaqueOptions(cOpaqueOptions: nextPtr, owned: false)
  }

  public func append(_ other: OpaqueOptions) throws {
    guard let opts = cOpaqueOptions else { throw LiteRtError.runtimeFailure }
    guard let otherOpts = other.cOpaqueOptions else { throw LiteRtError.invalidArgument }
    var mutableSelf: LiteRtOpaqueOptions? = opts
    let status = LiteRtAppendOpaqueOptions(&mutableSelf, otherOpts)
    try checkStatus(status)
    other.owned = false
  }

  public func pop() throws {
    guard let opts = cOpaqueOptions else { throw LiteRtError.runtimeFailure }
    var mutableSelf: LiteRtOpaqueOptions? = opts
    let status = LiteRtPopOpaqueOptions(&mutableSelf)
    try checkStatus(status)
    self.cOpaqueOptions = mutableSelf
    if mutableSelf == nil {
      self.owned = false
    }
  }
}
