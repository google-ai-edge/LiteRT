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

import XCTest
import LiteRtC
@testable import LiteRt

final class OptionsTests: XCTestCase {
  func testExternalTensorBinding() throws {
    let options = try Options()
    var rawData: [Float] = [1.0, 2.0]
    try rawData.withUnsafeMutableBytes { rawBuffer in
      guard let baseAddr = rawBuffer.baseAddress else { return }
      try options.addExternalTensorBinding(
        signatureName: "main",
        tensorName: "input",
        data: baseAddr,
        sizeBytes: rawBuffer.count
      )
    }
  }

  func testAddCustomOpKernel() throws {
    let options = try Options()
    var customKernel = LiteRtCustomOpKernel()
    try options.addCustomOpKernel(
      name: "CustomOp",
      version: 1,
      kernel: &customKernel
    )
  }
}
