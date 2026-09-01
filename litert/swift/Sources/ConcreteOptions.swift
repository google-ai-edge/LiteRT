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

/// A protocol for concrete configuration options in LiteRT (e.g. CPU, GPU, NPU options).
///
/// Types conforming to `ConcreteOptions` can be serialized into `OpaqueOptions`
/// and attached to compilation `Options`.
public protocol ConcreteOptions {
  /// Serializes these options into an `OpaqueOptions` instance.
  ///
  /// - Returns: An `OpaqueOptions` instance containing the serialized options.
  /// - Throws: `LiteRtError` if serializing fails.
  func createOpaqueOptions() throws -> OpaqueOptions
}
