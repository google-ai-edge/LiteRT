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
    name = "google_tensor",
    assets = [
        "model#group_pixel_8_plus/simple_model_google_tensor.bin",
        "model#group_other/text.txt",
    ],
    custom_package = "org.odml.litert.litert.aot.npu_models.google_tensor",
    manifest = "AiPackManifest.xml",
)

android_feature_module(
    name = "ai_pack",
    custom_package = "org.tensoflow.tflite.experimental.litert.aot.npu_models.google_tensor",
    feature_name = "google_tensor_ai_model",
    is_asset_pack = True,
    library = ":google_tensor",
    manifest = "AiPackManifest.xml",
    title = "Google Tensor AI Model",
)
