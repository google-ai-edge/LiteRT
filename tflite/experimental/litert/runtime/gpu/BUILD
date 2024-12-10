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

cc_library(
    name = "gl_utils",
    testonly = True,
    srcs = ["gl_utils.cc"],
    hdrs = ["gl_utils.h"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [
            "-landroid",
            "-lGLESv2",
            "-lEGL",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/runtime:tensor_buffer",
        "@org_tensorflow//third_party/GL:EGL_headers",
        "@org_tensorflow//third_party/GL:GLES3_headers",
    ],
)

cc_test(
    name = "gl_utils_test",
    srcs = ["gl_utils_test.cc"],
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        ":gl_utils",
        "//tflite/delegates/gpu/gl:egl_environment",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/cc:litert_layout",
        "//tflite/experimental/litert/cc:litert_model",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/runtime:tensor_buffer",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)
