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
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    # TODO(niuchl): make it visible to litert clients.
    default_visibility = ["//litert:__subpackages__"],
)

cc_import(
    name = "dispatch_api_so",
    shared_library = "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
    visibility = ["//visibility:private"],
)

_SUPPORTED_HTP_VERSIONS = [
    "73",
    "75",
    "79",
]

[
    android_library(
        name = "qualcomm_npu_runtime_v{}".format(version),
        exports = [
            ":dispatch_api_so",
            "//third_party/qairt/latest:libQnnHtp.so",
            "//third_party/qairt/latest:libQnnHtpV{}Skel.so".format(version),
            "//third_party/qairt/latest:libQnnHtpV{}Stub.so".format(version),
            "//third_party/qairt/latest:libQnnSystem.so",
        ],
    )
    for version in _SUPPORTED_HTP_VERSIONS
]

[
    android_feature_module(
        name = "qualcomm_npu_runtime_module_v{}".format(version),
        custom_package = "org.tensoflow.tflite.experimental.litert.vendors.qualcomm",
        feature_name = "qualcomm_npu_runtime_v{}".format(version),
        library = ":qualcomm_npu_runtime_v{}".format(version),
        manifest = "FeatureManifest_v{}.xml".format(version),
        title = "Qualcomm NPU Module",
    )
    for version in _SUPPORTED_HTP_VERSIONS
]
