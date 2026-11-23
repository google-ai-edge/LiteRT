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

import CLiteRT
import Foundation
import XCTest
@testable import LiteRT

/// Returns the string value of the option `tag` that `environment` was created with, if any.
func stringOption(_ tag: LiteRtEnvOptionTag, of environment: Environment) throws -> String? {
  var options: LiteRtEnvironmentOptions?
  try checkStatus(LiteRtGetEnvironmentOptions(environment.cEnvironment, &options))
  var value = LiteRtAny()
  guard LiteRtGetEnvironmentOptionsValue(options, tag, &value) == kLiteRtStatusOk else {
    return nil
  }
  return String(cString: value.str_value)
}

func createTestEnvironment(options: [Environment.Option] = []) throws -> Environment {
  var resolvedOptions = options
  if !options.contains(where: { if case .runtimeLibraryDir = $0 { return true }; return false }) {
    if let srcDir = ProcessInfo.processInfo.environment["TEST_SRCDIR"] {
      let workspaceName = ProcessInfo.processInfo.environment["TEST_WORKSPACE"] ?? "litert"
      let candidates = [
        "\(srcDir)/\(workspaceName)/external/litert_prebuilts/macos_arm64",
        "\(srcDir)/litert/prebuilts/macos_arm64"
      ]
      for path in candidates {
        if FileManager.default.fileExists(atPath: path) {
          resolvedOptions.append(.runtimeLibraryDir(path))
          break
        }
      }
    }
  }
  return try Environment(options: resolvedOptions)
}
