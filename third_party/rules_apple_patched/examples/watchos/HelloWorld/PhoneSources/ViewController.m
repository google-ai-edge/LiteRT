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

#import <WatchConnectivity/WatchConnectivity.h>

#import "examples/watchos/HelloWorld/PhoneSources/ViewController.h"

@interface ViewController () <WCSessionDelegate>
@end

@implementation ViewController

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];

  if (![WCSession isSupported]) {
    [self showAlert:@"This example requires watchOS support."];
    return;
  }
}

- (IBAction)initiateGreetingTapped:(UIButton *)sender {
  WCSession *session = [WCSession defaultSession];
  session.delegate = self;
  [session activateSession];

  if (session.paired) {
    NSDictionary *watchAppContext = @{ @"name": self.nameField.text };

    NSError *error = nil;
    [session updateApplicationContext:watchAppContext error:&error];

    if (error) {
      NSLog(@"Error updating watch app context: %@", error);
    }
  }
}

#pragma mark - WCSessionDelegate

- (void)session:(WCSession *)session
    activationDidCompleteWithState:(WCSessionActivationState)activationState
                             error:(NSError *)error {
  if (activationState != WCSessionActivationStateActivated) {
    NSString *msg =
      [NSString stringWithFormat:@"Failed to activate: %@", error];
    [self showAlert:msg];
    return;
  }
}

- (void)sessionDidBecomeInactive:(WCSession *)session {
  // No state to update.
}

- (void)sessionDidDeactivate:(WCSession *)session {
   // Begin the activation process for the new Apple Watch.
   [[WCSession defaultSession] activateSession];
}

#pragma mark - Private methods

- (void)showAlert:(NSString *)message {
  UIAlertController *alertController =
      [UIAlertController alertControllerWithTitle:nil
                                          message:message
                                   preferredStyle:UIAlertControllerStyleAlert];
  [self presentViewController:alertController animated:YES completion:nil];
}

@end
