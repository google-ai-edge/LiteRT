// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import SwiftUI

@main
public struct BazelApp: App {
  public init() { }

  public var body: some Scene {
    WindowGroup {
      Text("Hello World")
        .accessibility(identifier: "HELLO_WORLD")
      Text("Example", comment: "Example text")
    }
  }

  /// A public API to test DooC documentation generation.
  ///
  /// Example referencing ``BazelApp``:
  ///
  /// ```swift
  /// let app = BazelApp()
  /// ```
  public func foo() { }

  /// An internal API to test `minimum_access_level` DooC option.
  internal func internalFoo() { }
}

// MARK: - View Extension

extension View {

  /// A public API extension on ``View`` to test DooC documentation generation.
  public func viewFoo() { }

  /// An internal API extension on ``View`` to test `minimum_access_level` DooC option.
  internal func internalViewFoo() { }
}

// MARK: - Custom type

/// My Struct
///
/// Example referencing ``MyStruct``:
///
/// ```swift
///
/// let foo = MyStruct()
/// ```
public struct MyStruct {
  public init() { }

  /// My Struct's foo API to test DooC documentation generation.
  /// - Parameters:
  ///  - bar: A bar parameter.
  public func foo(bar: Int) { }

  /// An internal foo API to test DooC documentation generation with `minimum_access_level` set to `internal`.
  ///
  /// - Parameters:
  /// - bar: A bar parameter.
  internal func internalFoo(bar: Int) { }
}
