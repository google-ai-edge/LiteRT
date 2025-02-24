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

load("//java/com/google/devtools/staticanalysis/java/checker:build_defs.bzl", "android_library_with_nullness_check")
load("//tools/build_defs/android:rules.bzl", "android_application", "android_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//litert/google/sample_app_android/aot:__subpackages__"],
)

android_library_with_nullness_check(
    name = "testapp_lib_no_jni",
    srcs = glob([
        "src/java/org/tensorflow/tflite/experimental/litert/aot/*.java",
    ]),
    assets = [
        "assets/simple_model_npu.tflite",
    ],
    assets_dir = "assets",
    custom_package = "org.tensorflow.tflite.experimental.litert.aot",
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    # This test can be run only on Android and Linux.
    target_compatible_with = select({
        "//third_party/bazel_platforms/os:android": [],
        "//third_party/bazel_platforms/os:linux": [],
        "//conditions:default": ["//third_party/bazel_platforms:incompatible"],
    }),
    deps = [
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/v4",
        "//third_party/java/androidx/appcompat",
        "//third_party/java/androidx/legacy/support_v13",
    ],
)

android_library_with_nullness_check(
    name = "testapp_lib",
    exports = [
        ":testapp_lib_no_jni",
        "//litert/google/sample_app_android/aot/src/native:litert_native",
    ],
)

android_binary(
    name = "testapp_google_tensor",
    manifest = "AndroidManifest.xml",
    multidex = "legacy",
    nocompress_extensions = [
        ".tflite",
        ".bin",
    ],
    deps = [
        ":testapp_lib",
        "//litert/google/sample_app_android/aot/npu_models/google_tensor",
        "//litert/vendors/google_tensor/google:google_tensor_runtime",
    ],
)

android_application(
    name = "testapp_google_tensor_aab",
    # The bundle_config is needed to allow Android to unpack .so files in the feature module to a
    # local dir, so that they could be loaded at runtime with dlopen().
    bundle_config_file = "bundle_config.pb.json",
    device_group_config = "device_targeting_configuration.json",
    feature_modules = [
        "//litert/vendors/google_tensor/google:google_tensor_runtime_module",
        "//litert/google/sample_app_android/aot/npu_models/google_tensor:ai_pack",
    ],
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "org.tensorflow.tflite.experimental.litert.aot",
        "versionCode": "1",
        "versionName": "1.0.0",
        "targetSdkVersion": "34",
    },
    multidex = "native",
    deps = [
        ":testapp_lib",
    ],
)

android_binary(
    name = "testapp_qualcomm",
    manifest = "AndroidManifest.xml",
    multidex = "legacy",
    nocompress_extensions = [
        ".tflite",
        ".bin",
    ],
    deps = [
        ":testapp_lib",
        "//litert/google/sample_app_android/aot/npu_models/qualcomm",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v73",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v75",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v79",
    ],
)

android_application(
    name = "testapp_qualcomm_aab",
    # The bundle_config is needed to allow Android to unpack .so files in the feature module to a
    # local dir, so that they could be loaded at runtime with dlopen().
    bundle_config_file = "bundle_config.pb.json",
    device_group_config = "device_targeting_configuration.json",
    feature_modules = [
        "//litert/google/sample_app_android/aot/npu_models/qualcomm:ai_pack",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_module_v73",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_module_v75",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_module_v79",
    ],
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "org.tensorflow.tflite.experimental.litert.aot",
        "versionCode": "1",
        "versionName": "1.0.0",
        "targetSdkVersion": "34",
    },
    multidex = "native",
    deps = [
        ":testapp_lib",
    ],
)

# TODO(niuchl): Re-enable this target once the issues are fixed..
# android_application(
#     name = "testapp_aab",
#     # The bundle_config is needed to allow Android to unpack .so files in the feature module to a
#     # local dir, so that they could be loaded at runtime with dlopen().
#     bundle_config_file = "bundle_config.pb.json",
#     device_group_config = "device_targeting_configuration.json",
#     feature_modules = [
#         "//litert/google/sample_app_android/aot/npu_models/google_tensor:ai_pack",
#         "//litert/google/sample_app_android/aot/npu_models/qualcomm:ai_pack",
#         "//litert/vendors/google_tensor/google:google_tensor_runtime_module",
#         "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_module_v73",
#         "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_module_v75",
#         "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_module_v79",
#     ],
#     manifest = "AndroidManifest.xml",
#     manifest_values = {
#         "applicationId": "org.tensorflow.tflite.experimental.litert.aot",
#         "versionCode": "1",
#         "versionName": "1.0.0",
#         "targetSdkVersion": "34",
#     },
#     multidex = "native",
#     deps = [
#         ":testapp_lib",
#     ],
# )
