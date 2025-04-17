# Copyright 2025 Google LLC.
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

load("//java/com/google/devtools/staticanalysis/java/checker:build_defs.bzl", "android_library_with_nullness_check")
load("//tools/build_defs/android:rules.bzl", "android_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/google/sample_app_android/aot:__subpackages__"],
)

android_library_with_nullness_check(
    name = "image_segmentation_lib",
    srcs = glob([
        "src/**/*.kt",
    ]),
    assets = glob(["assets/*"]),
    assets_dir = "assets",
    custom_package = "com.google.aiedge.examples.image_segmentation",
    manifest = "src/main/AndroidManifest.xml",
    resource_files = glob(["src/main/res/**"]),
    # This test can be run only on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/v4",
        "//third_party/java/androidx/activity/compose",
        "//third_party/java/androidx/appcompat",
        "//third_party/java/androidx/camera/camera2",  # buildcleaner: keep
        "//third_party/java/androidx/camera/core",
        "//third_party/java/androidx/camera/video",
        "//third_party/java/androidx/camera/view",
        "//third_party/java/androidx/compose/material",
        "//third_party/java/androidx/compose/material/icons_extended",
        "//third_party/java/androidx/compose/runtime",
        "//third_party/java/androidx/constraintlayout/compose",
        "//third_party/java/androidx/legacy/support_v13",
        "//third_party/java/coil:base",
        "//third_party/java/coil:compose_base",
        "//third_party/java/coil:singleton",
        "//litert/kotlin:litert_kotlin_api",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v79",  # buildcleaner: keep
    ],
)

android_binary(
    name = "image_segmentation",
    manifest = "src/main/AndroidManifest.xml",
    deps = [
        ":image_segmentation_lib",
    ],
)
