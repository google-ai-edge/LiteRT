// Copyright 2018 The Bazel Authors. All rights reserved.
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

import XCTest

class ButtonsMacUITests: XCTestCase {

  override func setUp() {
    super.setUp()
    continueAfterFailure = false
    XCUIApplication().launch()
  }

  func testLabelIncrementsWithClick() {
    let app = XCUIApplication()
    let clickButton = app.buttons["clickButton"]
    let clickCountLabel = app.staticTexts["clickCount"]

    for tapCount in 1...3 {
      clickButton.click()
      XCTAssertEqual(clickCountLabel.value! as! String, String(tapCount))
    }
  }

}

