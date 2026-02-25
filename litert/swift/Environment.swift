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

/// The LiteRT environment.
public class Environment {
  let handle: LiteRtEnvironment

  /// Creates a new environment.
  public init(options: [EnvironmentOption] = []) throws {
    var env: LiteRtEnvironment?

    // TODO: Map options.
    // For now we support empty options.
    let status = LiteRtCreateEnvironment(0, nil, &env)

    guard status == kLiteRtStatusOk, let env = env else {
      throw status.toError(message: "Failed to create environment")
        ?? .unknown("Unknown error creating environment")
    }
    self.handle = env
  }

  deinit {
    LiteRtDestroyEnvironment(handle)
  }
}

/// Placeholder for environment options.
public struct EnvironmentOption {
  // TODO: Implement based on LiteRtEnvOption
}
