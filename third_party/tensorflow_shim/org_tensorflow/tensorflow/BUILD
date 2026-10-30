# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Config settings and package groups that the exported tflite/ BUILD files
# reference as @org_tensorflow//tensorflow:<name>. They mirror the OSS
# definitions in TensorFlow's tensorflow/BUILD.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_license//rules:license.bzl", "license")

package(default_visibility = ["//visibility:public"])

exports_files(glob(["*.bzl"]))

license(
    name = "license",
    package_name = "tensorflow",
)

bzl_library(
    name = "tensorflow_bzl",
    srcs = glob(["*.bzl"]),
)

package_group(
    name = "internal",
    packages = ["public"],
)

# Operating systems.

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
)

config_setting(
    name = "chromiumos",
    values = {"crosstool_top": "//external:android/chromiumos"},
)

config_setting(
    name = "emscripten",
    values = {"crosstool_top": "//external:android/emscripten"},
)

config_setting(
    name = "fuchsia",
    constraint_values = ["@platforms//os:fuchsia"],
)

config_setting(
    name = "ios",
    values = {"apple_platform_type": "ios"},
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

config_setting(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:macos",
    ],
    values = {"apple_platform_type": "macos"},
)

config_setting(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:macos",
    ],
    values = {"apple_platform_type": "macos"},
)

selects.config_setting_group(
    name = "macos",
    match_any = [
        ":macos_x86_64",
        ":macos_arm64",
    ],
)

selects.config_setting_group(
    name = "mobile",
    match_any = [
        ":android",
        ":chromiumos",
        ":fuchsia",
        ":emscripten",
        ":ios",
    ],
)

# OS and CPU combinations.

config_setting(
    name = "android_arm",
    constraint_values = [
        "@platforms//cpu:armv7",
        "@platforms//os:android",
    ],
    values = {"crosstool_top": "//external:android/crosstool"},
)

config_setting(
    name = "android_armeabi",
    constraint_values = [
        "@platforms//cpu:armv6-m",
        "@platforms//os:android",
    ],
    values = {"crosstool_top": "//external:android/crosstool"},
)

config_setting(
    name = "chromiumos_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:chromiumos",
    ],
)

config_setting(
    name = "ios_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:ios",
    ],
    values = {"crosstool_top": "//tools/osx/crosstool:crosstool"},
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "linux_ppc64le",
    constraint_values = [
        "@platforms//cpu:ppc64le",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "linux_s390x",
    constraint_values = [
        "@platforms//cpu:s390x",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "linux_x86_64_no_sse",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    values = {"copt": "-mno-sse4.2"},
)

selects.config_setting_group(
    name = "arm_any",
    match_any = [
        "@platforms//cpu:aarch32",
        "@platforms//cpu:aarch64",
        "@platforms//cpu:armv6-m",
        "@platforms//cpu:armv7",
        "@platforms//cpu:armv7-m",
        "@platforms//cpu:armv7e-m",
        "@platforms//cpu:armv7e-mf",
    ],
)

# Build modes and defines.

config_setting(
    name = "debug",
    values = {"compilation_mode": "dbg"},
)

config_setting(
    name = "optimized",
    values = {"compilation_mode": "opt"},
)

config_setting(
    name = "disable_tf_lite_py",
    define_values = {"disable_tf_lite_py": "true"},
)

config_setting(
    name = "lite_protos",
    define_values = {"TENSORFLOW_PROTOS": "lite"},
)

config_setting(
    name = "full_protos",
    define_values = {"TENSORFLOW_PROTOS": "full"},
)

selects.config_setting_group(
    name = "mobile_lite_protos",
    match_all = [
        ":lite_protos",
        ":mobile",
    ],
)

selects.config_setting_group(
    name = "mobile_full_protos",
    match_all = [
        ":full_protos",
        ":mobile",
    ],
)
