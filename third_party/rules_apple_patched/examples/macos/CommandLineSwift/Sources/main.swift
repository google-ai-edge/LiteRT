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

import Foundation

// Since the Info.plist file gets embedded in the binary, we can access values
// like the bundle identifier using the NSBundle APIs.
let bundle = Bundle.main
NSLog("Hello World from \(bundle.bundleIdentifier ?? "<none>")")
NSLog("\nHere is the entire Info.plist dictionary: \(bundle.infoDictionary ?? [:])")
