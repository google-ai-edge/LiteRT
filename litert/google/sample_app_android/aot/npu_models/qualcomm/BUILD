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
    default_visibility = ["//litert/google/sample_app_android/aot:__subpackages__"],
)

android_library(
    name = "qualcomm",
    assets = [
        "model#group_qti_v73/simple_model_qualcomm.bin",
        "model#group_qti_v75/simple_model_qualcomm.bin",
        "model#group_qti_v79/simple_model_qualcomm.bin",
        "model#group_other/text.txt",
    ],
    assets_dir = "",
    custom_package = "org.odml.litert.litert.aot.npu_models.qualcomm",
    manifest = "AiPackManifest.xml",
)

android_feature_module(
    name = "ai_pack",
    custom_package = "org.tensoflow.tflite.experimental.litert.aot.npu_models.qualcomm",
    feature_name = "qualcomm_ai_model",
    is_asset_pack = True,
    library = ":qualcomm",
    manifest = "AiPackManifest.xml",
    title = "Qualcomm AI Model",
)
