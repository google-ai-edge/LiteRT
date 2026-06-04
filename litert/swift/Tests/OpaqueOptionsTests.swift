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

final class OpaqueOptionsTests: XCTestCase {
  func testOpaqueOptions() throws {
    let identifier = "test_opaque_options"
    var payloadValue = 42
    try withUnsafeMutablePointer(to: &payloadValue) { payloadPtr in
      let opaqueOptions = try OpaqueOptions(identifier: identifier, payload: payloadPtr)
      XCTAssertEqual(opaqueOptions.identifier, identifier)
      XCTAssertEqual(opaqueOptions.data, payloadPtr)

      let options = try Options()
      try options.addOpaqueOptions(opaqueOptions)

      let retrievedOpaqueOptions = try options.opaqueOptions()
      XCTAssertNotNil(retrievedOpaqueOptions)
      XCTAssertEqual(retrievedOpaqueOptions?.identifier, identifier)
      XCTAssertEqual(retrievedOpaqueOptions?.data, payloadPtr)
    }
  }

  func testOpaqueOptionsPopLast() throws {
    let identifier = "test_opaque_options"
    var payloadValue = 42
    try withUnsafeMutablePointer(to: &payloadValue) { payloadPtr in
      let opaqueOptions = try OpaqueOptions(identifier: identifier, payload: payloadPtr)
      XCTAssertEqual(opaqueOptions.identifier, identifier)

      try opaqueOptions.pop()

      XCTAssertEqual(opaqueOptions.identifier, "")
      XCTAssertNil(opaqueOptions.data)
    }
  }
}
