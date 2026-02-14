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
import Dispatch

class ViewController: NSViewController {

  @IBOutlet weak var inputField: NSTextField?
  @IBOutlet weak var outputLabel: NSTextField?

  var serviceConnection: NSXPCConnection?

  override func viewDidLoad() {
    super.viewDidLoad()

    serviceConnection = NSXPCConnection(serviceName: "com.example.xpc-service-app.xpc-service")
    serviceConnection!.remoteObjectInterface = NSXPCInterface(with: ServiceProtocol.self)
    serviceConnection!.resume()
  }

  override func viewDidAppear() {
    super.viewDidAppear()
    self.view.window?.title = "XPCServiceApp"
  }

  @IBAction func buttonClicked(sender: NSButton) {
    guard let message = self.inputField?.stringValue, message.lengthOfBytes(using: .utf8) > 0 else {
      self.outputLabel?.stringValue = "No message given"
      return
    }
    send(message: message)
  }

  func send(message: String) {
    guard let connection = self.serviceConnection else {
      return
    }
    let service = connection.remoteObjectProxyWithErrorHandler { (error) in
      print("Remote proxy error: %@", error)
    } as! ServiceProtocol

    service.process(message: message) { (response) in
      guard let response = response else {
        print("Nothing returned from service")
        return
      }
      DispatchQueue.main.async {
        self.outputLabel?.stringValue = response
      }
    }
  }
}
