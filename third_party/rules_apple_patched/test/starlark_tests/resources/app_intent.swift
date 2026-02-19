// Copyright 2023 The Bazel Authors. All rights reserved.
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

import AppIntents
import Foundation

struct HelloWorldIntent: AppIntent {
  static var title: LocalizedStringResource = "Hello world intent"

  static var description = IntentDescription("Says hello the world.")

  func perform() async throws -> some ProvidesDialog {
    return .result(dialog: "Hello world")
  }
}

#if (arch(x86_64))
  struct IntelIntent: AppIntent {
    static var title: LocalizedStringResource = "Intel"

    static var description = IntentDescription("Intel x86_64 intent")

    func perform() async throws -> some ProvidesDialog {
      return .result(dialog: "I'm running on x86_64")
    }
  }
#endif

#if (arch(arm64))
  struct ArmIntent: AppIntent {
    static var title: LocalizedStringResource = "ARM64"

    static var description = IntentDescription("ARM64 intent")

    func perform() async throws -> some ProvidesDialog {
      return .result(dialog: "I'm running on arm64")
    }
  }
#endif

#if os(iOS)
  struct iOSIntent: AppIntent {
    static var title: LocalizedStringResource = "iOS"

    static var description = IntentDescription("iOS intent")

    func perform() async throws -> some ProvidesDialog {
      return .result(dialog: "This is an iOS intent")
    }
  }
#endif

#if os(macOS)
  struct macOSIntent: AppIntent {
    static var title: LocalizedStringResource = "macOS"

    static var description = IntentDescription("macOS intent")

    func perform() async throws -> some ProvidesDialog {
      return .result(dialog: "This is a macOS intent")
    }
  }
#endif

#if os(tvOS)
  struct tvOSIntent: AppIntent {
    static var title: LocalizedStringResource = "tvOS"

    static var description = IntentDescription("tvOS intent")

    func perform() async throws -> some ProvidesDialog {
      return .result(dialog: "This is a tvOS intent")
    }
  }
#endif

#if os(watchOS)
  struct watchOSIntent: AppIntent {
    static var title: LocalizedStringResource = "watchOS"

    static var description = IntentDescription("watchOS intent")

    func perform() async throws -> some ProvidesDialog {
      return .result(dialog: "This is a watchOS intent")
    }
  }
#endif
