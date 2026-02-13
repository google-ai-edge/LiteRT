# Copyright 2018 The Bazel Authors. All rights reserved.
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

"""Support for things that are common to all Apple platforms."""

# The different modes for the validation of requested entitlements vs. those
# enabled in the provisioning profile.
#
# Provisioning profiles are not bundled into "simulator" builds, so while Xcode's
# UI will show errors/issues if you navigate to the capabilities UI, it doesn't
# actually prevent these simluator builds. However, with "device" builds, Xcode
# does validation before starting the build steps for a target and raises errors
# (preventing the build) if there is any issues.
#
# These values can be used for the rules that support the
# `entitlements_validation` attribute to set the desired behavior.
#
# * `error`: Perform the checks and raise errors for any issues found.
# * `warn`: Perform the checks and only raise warnings for any issues found.
# * `loose`: Perform the checks and raise warnings for "simulator" builds, but
#   raise errors for "device" builds.
# * `skip`: Perform no checks at all. This should only be need in cases where
#   the build result is _not_ used directly, and some external process resigns
#   the build after the fact correcting the entitlements/provisioning.
#
# The `loose` value is closest to what a native Xcode project would do in that
# it doesn't prevent a simulator build, but does stop a device build. The main
# difference being it at least raises warnings to let you know of the issue with
# the target since they would prevent one from deploying the build to a device.
entitlements_validation_mode = struct(
    error = "error",
    warn = "warn",
    loose = "loose",
    skip = "skip",
)
