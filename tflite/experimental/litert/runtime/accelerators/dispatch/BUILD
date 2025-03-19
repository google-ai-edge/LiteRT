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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//tflite/experimental/litert:__subpackages__",
    ],
)

cc_library(
    name = "dispatch_accelerator",
    srcs = ["dispatch_accelerator.cc"],
    hdrs = ["dispatch_accelerator.h"],
    deps = [
        "//tflite/c:c_api_types",
        "//tflite/experimental/litert/c:litert_accelerator",
        "//tflite/experimental/litert/c:litert_accelerator_compilation_options",
        "//tflite/experimental/litert/c:litert_accelerator_registration",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_dispatch_delegate",
        "//tflite/experimental/litert/cc:litert_any",
        "//tflite/experimental/litert/cc:litert_dispatch_delegate",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "//tflite/experimental/litert/core:environment",
        "//tflite/experimental/litert/runtime:accelerator_model_compilation_data",
        "@com_google_absl//absl/strings:string_view",
    ],
)
