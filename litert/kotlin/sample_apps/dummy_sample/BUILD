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
    "android_application",
    "android_binary",
    "android_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//third_party/odml/litert:__subpackages__"],
)

android_library(
    name = "dummy_sample_lib",
    srcs = glob([
        "src/java/com/google/ai/edge/litert/sample/dummy/*.java",
    ]),
    assets = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    assets_dir = "testdata",
    custom_package = "com.google.ai.edge.litert.sample.dummy",
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    deps = [
        "//java/com/google/android/apps/play/store/sdk/aipacks:1p_client_ai_delivery",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/v4",
        "//third_party/java/android_libs/guava_jdk5:concurrent",
        "//third_party/java/androidx/appcompat",
        "//litert/kotlin:litert_kotlin_api",
    ],
)

android_binary(
    name = "dummy_sample",
    manifest = "AndroidManifest.xml",
    deps = [
        ":dummy_sample_lib",
    ],
)

# TODO(niuchl): Use AiPacks and feature modules in the sample app.
android_binary(
    name = "dummy_sample_qualcomm_v75",
    manifest = "AndroidManifest.xml",
    deps = [
        ":dummy_sample_lib",
        "//litert/kotlin/sample_apps/dummy_sample/npu_models:simple_model_qualcomm_v75",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v75",
    ],
)

android_application(
    name = "dummy_sample_aab",
    bundle_config_file = "bundle_config.pb.json",
    device_group_config = "device_targeting_configuration.json",
    feature_modules = [
        "//litert/kotlin/sample_apps/dummy_sample/npu_models:simple_model_ai_pack",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_module_v73",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_module_v75",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_module_v79",
    ],
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.google.ai.edge.litert.sample.dummy",
        "versionCode": "1",
        "versionName": "1.0.0",
        "targetSdkVersion": "34",
    },
    multidex = "native",
    deps = [
        ":dummy_sample_lib",
    ],
)
