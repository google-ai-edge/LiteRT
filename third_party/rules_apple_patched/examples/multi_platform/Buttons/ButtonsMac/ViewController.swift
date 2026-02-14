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

import Cocoa

class ViewController: NSViewController {

  @IBOutlet var countLabel: NSTextField!

  override func viewDidLoad() {
    super.viewDidLoad()
  }

  @IBAction func didClick(sender: NSButton) {
    incrementLabel(label: countLabel)
  }

  func incrementLabel(label: NSTextField) {
    let number = Int(label.stringValue)
    label.stringValue = String(number! + 1)
  }

}

