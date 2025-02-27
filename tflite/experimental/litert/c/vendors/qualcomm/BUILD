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
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "accelerator_options",
    srcs = ["accelerator_options.cc"],
    hdrs = ["accelerator_options.h"],
    deps = [
        "//tflite/experimental/litert/c:litert_accelerator_options",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_macros",
        "//tflite/experimental/litert/core:accelerator",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "set_options_example",
    srcs = ["set_options_example.cc"],
    deps = [
        ":accelerator_options",
        "//tflite/experimental/litert/c:litert_accelerator_options",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_compiled_model",
        "//tflite/experimental/litert/c:litert_compiled_model_options",
        "//tflite/experimental/litert/cc:litert_macros",
    ],
)
