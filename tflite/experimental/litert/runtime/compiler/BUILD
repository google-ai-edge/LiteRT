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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/experimental/litert:__subpackages__"],
)

cc_test(
    name = "jit_compilation_qualcomm_test",
    srcs = ["jit_compilation_qualcomm_test.cc"],
    data = [
        "//tflite/experimental/litert/test:simple_model",
        "//tflite/experimental/litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so",
        "//tflite/experimental/litert/vendors/qualcomm/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_compiled_model",
        "//tflite/experimental/litert/cc:litert_environment",
        "//tflite/experimental/litert/cc:litert_model",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:matchers",
        "//tflite/experimental/litert/test:simple_model_npu",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "jit_compilation_mediatek_test",
    srcs = ["jit_compilation_mediatek_test.cc"],
    data = [
        "//tflite/experimental/litert/test:simple_model",
        "//tflite/experimental/litert/vendors/mediatek/compiler:compiler_plugin_so",
        "//tflite/experimental/litert/vendors/mediatek/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "no_oss",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/cc:litert_compiled_model",
        "//tflite/experimental/litert/cc:litert_environment",
        "//tflite/experimental/litert/cc:litert_model",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:matchers",
        "//tflite/experimental/litert/test:simple_model_npu",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
