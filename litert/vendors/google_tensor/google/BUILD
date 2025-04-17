# Copyright 2024 Google LLC.
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

load(
    "//tools/build_defs/android:rules.bzl",
    "android_feature_module",
    "android_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    # TODO(niuchl): make it visible to litert clients.
    default_visibility = ["//litert:__subpackages__"],
)

cc_import(
    name = "dispatch_api_so",
    shared_library = "//litert/vendors/google_tensor/dispatch:dispatch_api_so",
    visibility = ["//visibility:private"],
)

android_library(
    name = "google_tensor_runtime",
    exports = [
        ":dispatch_api_so",
    ],
)

android_feature_module(
    name = "google_tensor_runtime_module",
    custom_package = "org.tensoflow.tflite.experimental.litert.vendors.google_tensor",
    feature_name = "google_tensor_runtime",
    library = ":google_tensor_runtime",
    manifest = "FeatureManifest.xml",
    title = "Google Tensor TPU Module",
)
