# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The class contains the constants of iOS test runner."""


def enum(**enums):
  return type('Enum', (), enums)


ARCH = enum(
    ARMV7='armv7',
    ARMV7S='armv7s',
    ARM64='arm64',
    ARM64E='arm64e',
    I386='i386',
    X86_64='x86_64')
SDK = enum(IPHONEOS='iphoneos', IPHONESIMULATOR='iphonesimulator')
# It is consistent with bazel's apple platform:
# https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/rules/apple/ApplePlatform.java
PLATFORM = enum(IOS_DEVICE='ios_device', IOS_SIMULATOR='ios_simulator')
OS = enum(IOS='iOS', WATCHOS='watchOS', TVOS='tvOS')
TestType = enum(XCUITEST='xcuitest', XCTEST='xctest', LOGIC_TEST='logic_test')
SimState = enum(CREATING='Creating', SHUTDOWN='Shutdown', BOOTED='Booted',
                UNKNOWN='Unknown')

SUPPORTED_SDKS = [SDK.IPHONESIMULATOR, SDK.IPHONEOS]
SUPPORTED_PLATFORMS = [PLATFORM.IOS_SIMULATOR, PLATFORM.IOS_DEVICE]
SUPPORTED_TEST_TYPES = [TestType.XCUITEST, TestType.XCTEST, TestType.LOGIC_TEST]
SUPPORTED_SIM_OSS = [OS.IOS]

TEST_STARTED_SIGNAL = 'Test Suite'
XCTRUNNER_STARTED_SIGNAL = 'Running tests...'

CORESIMULATOR_INTERRUPTED_ERROR = 'CoreSimulatorService connection interrupted'

LAUNCH_OPTIONS_JSON_HELP = (
    """The path of json file, which contains options of launching test.

Available keys for the json:
  env_vars : dict
    Additional environment variables passing to test's process. The key and
    value should be string.
  args : array
    Additional arguments passing to test's process.
  app_under_test_env_vars: dict
    Additional environment variables passing to app under test's process. The
    key and value should be string.
    In xctest, the functionality is the same as "env_vars".
    In xcuitest, the process of app under test is different with the
    process of test.
  app_under_test_args: array
    Additional arguments passing to app under test's process.
    In xctest, the functionality is the same as "args".
    In xcuitest, the process of app under test is different with the
    process of test.
  keep_xcresult_data: bool
    Whether or not to keep the xcresult bundle produced by the test run
    in the output_dir.
  tests_to_run : array
    The specific test classes or test methods to run. Each item should be
    string and its format is Test-Class-Name[/Test-Method-Name]. It is supported
    in Xcode 8+.
  skip_tests: array
    The specific test classes or test methods to skip. Each item should be
    string and its format is Test-Class-Name[/Test-Method-Name]. Logic test
    does not support that.
  uitest_auto_screenshots: bool
    Whether captures screenshots automatically in ui test. If yes, will save the
    screenshots when the test failed. By default, it is false. Prior Xcode 9,
    this option does not work and the auto screenshot is enable by default.
  startup_timeout_sec: int
    Seconds until the xcodebuild command is deemed stuck.
  destination_timeout_sec: int
    Wait for the given seconds while searching for the destination device.
  """)

SIGNING_OPTIONS_JSON_HELP = (
    """The path of json file, which contains options of signing app.

The signing options only works when running on sdk iphoneos.

Available keys for the json:
  xctrunner_app_provisioning_profile: string
    The path of the provisioning profile of the generated xctrunner app.
    If this field is not set, will use app under test's provisioning profile
    for the generated xctrunner app.
  xctrunner_app_enable_ui_file_sharing: bool
    Whether enable UIFileSharingEnabled field in the generated xctrunner app's
    Info.plist.
  keychain_path: string
    The specified keychain to be used.
  """)
