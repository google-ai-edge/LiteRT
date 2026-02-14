# Copyright 2019 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration options for the Apple rule integration tests."""

# General configuration options used with `apple_shell_test`.
# Note that changing this to fastbuild or dbg will break several tests.
COMPILATION_MODE_OPTIONS = ["--compilation_mode opt"]

# Configuration options used with `apple_shell_test` to run tests for
# iOS simulator and device builds.
#
IOS_DEVICE_OPTIONS = COMPILATION_MODE_OPTIONS + ["--ios_multi_cpus=arm64,arm64e"]
IOS_SIMULATOR_OPTIONS = COMPILATION_MODE_OPTIONS + [
    "--ios_multi_cpus=sim_arm64,x86_64",
]

IOS_CONFIGURATIONS = {
    "device": IOS_DEVICE_OPTIONS,
    "simulator": IOS_SIMULATOR_OPTIONS,
}

# Configuration for 64 bit simulators and fat devices. Useful for testing sanitizers since thread
# sanitizer is only supported in 64 bit simulator. It's easier to skip device tests than a specific
# simulator architecture test, so instead create a special configuration for the sanitizer tests.
IOS_64BIT_SIMULATOR_FAT_DEVICE_CONFIGURATIONS = {
    "device": IOS_DEVICE_OPTIONS,
    "simulator": COMPILATION_MODE_OPTIONS + ["--ios_multi_cpus=x86_64"],
}

IOS_TEST_CONFIGURATIONS = {
    "device": IOS_DEVICE_OPTIONS,
    "simulator": IOS_SIMULATOR_OPTIONS,
}

# Configuration options used with `apple_shell_test` to run tests for
# macOS builds.
MACOS_CONFIGURATIONS = {
    "default": COMPILATION_MODE_OPTIONS + ["--macos_cpus=x86_64"],
}

# Configuration options used with `apple_shell_test` to run tests for
# tvOS simulator and device builds.
TVOS_DEVICE_OPTIONS = COMPILATION_MODE_OPTIONS + ["--tvos_cpus=arm64"]
TVOS_SIMULATOR_OPTIONS = COMPILATION_MODE_OPTIONS + ["--tvos_cpus=x86_64"]

TVOS_CONFIGURATIONS = {
    "device": TVOS_DEVICE_OPTIONS,
    "simulator": TVOS_SIMULATOR_OPTIONS,
}

TVOS_TEST_CONFIGURATIONS = {
    "device": TVOS_DEVICE_OPTIONS,
    "simulator": TVOS_SIMULATOR_OPTIONS,
}

# Configuration options used with `apple_shell_test` to run tests for
# watchOS simulator and device builds. Since watchOS apps are always bundled
# with an iOS host app, we include that platform's configuration options as
# well.
WATCHOS_DEVICE_OPTIONS = COMPILATION_MODE_OPTIONS + [
    "--ios_multi_cpus=arm64,arm64e",
    "--watchos_cpus=armv7k",
]
WATCHOS_SIMULATOR_OPTIONS = COMPILATION_MODE_OPTIONS + [
    "--ios_multi_cpus=sim_arm64,x86_64",
    "--watchos_cpus=x86_64",
]

WATCHOS_CONFIGURATIONS = {
    "device": WATCHOS_DEVICE_OPTIONS,
    "simulator": WATCHOS_SIMULATOR_OPTIONS,
}
