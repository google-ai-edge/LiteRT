# XCTestRunner
A tool for running prebuilt iOS tests on iOS real device and simulator.

## Features
- It supports XCTest (Xcode Unit Test), XCUITest (Xcode UI Test).
- It supports iOS 11+ iOS real device, iOS simulator.
- It supports launch options configuration: test methods to run, additional
environment variables, additional arguments.
- It supports Xcode 10+.

## Prerequisites
- Install Xcode (Xcode 10+).
- [Install bazel](https://docs.bazel.build/install.html) (optional).

## Installation
You can download the ios_test_runner.par binary in [release](https://github.com/google/xctestrunner/releases)

or build the ios_test_runner.par binary by bazel:
```
$ git clone https://github.com/google/xctestrunner.git
$ cd xctestrunner
$ bazel build :ios_test_runner.par
$ ls bazel-bin/ios_test_runner.par
```

## Usage
- Build your app under test and test bundle. You can use Xcode.app,
`xcodebuild` command line tool or [bazel](https://github.com/bazelbuild/bazel).
- Run the ios_test_runner.par binary.

In overview, there are two sub-commands in the runner binary.
- `test`: Run test directly on connecting iOS real device or existing iOS
simulator.
- `simulator_test`: Run test on a new created simulator, which will be deleted
after test finishes.

See more details by running `ios_test_runner.par -h` in terminal.

## Notes

Disclaimer: This is not an official Google product.

XCTestRunner uses Apple native tool `xcodebuild`, `simctl` to control iOS
Simulator and launch tests on iOS devices.
