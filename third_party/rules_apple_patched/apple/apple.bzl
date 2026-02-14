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

"""# Rules that apply to all Apple platforms."""

load(
    "//apple:apple_static_library.bzl",
    _apple_static_library = "apple_static_library",
)
load(
    "//apple/internal:apple_framework_import.bzl",
    _apple_dynamic_framework_import = "apple_dynamic_framework_import",
    _apple_static_framework_import = "apple_static_framework_import",
)
load(
    "//apple/internal:apple_universal_binary.bzl",
    _apple_universal_binary = "apple_universal_binary",
)
load(
    "//apple/internal:apple_xcframework_import.bzl",
    _apple_dynamic_xcframework_import = "apple_dynamic_xcframework_import",
    _apple_static_xcframework_import = "apple_static_xcframework_import",
)
load(
    "//apple/internal:local_provisioning_profiles.bzl",
    _local_provisioning_profile = "local_provisioning_profile",
    _provisioning_profile_repository = "provisioning_profile_repository",
    _provisioning_profile_repository_extension = "provisioning_profile_repository_extension",
)
load(
    "//apple/internal:xcframework_rules.bzl",
    _apple_static_xcframework = "apple_static_xcframework",
    _apple_xcframework = "apple_xcframework",
)

apple_dynamic_framework_import = _apple_dynamic_framework_import
apple_dynamic_xcframework_import = _apple_dynamic_xcframework_import
apple_static_framework_import = _apple_static_framework_import
apple_static_library = _apple_static_library
apple_static_xcframework = _apple_static_xcframework
apple_static_xcframework_import = _apple_static_xcframework_import
apple_universal_binary = _apple_universal_binary
apple_xcframework = _apple_xcframework
local_provisioning_profile = _local_provisioning_profile
provisioning_profile_repository = _provisioning_profile_repository
provisioning_profile_repository_extension = _provisioning_profile_repository_extension
