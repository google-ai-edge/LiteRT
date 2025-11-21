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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # By default, visibility is restricted to litert_cc_users_dynamic_link. Individual targets can be
        # made public by setting their visibility to litert_public.
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_dynamic_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

LITERT_C_API_DEPS = [
    "//litert/c:litert_common",
    "//litert/c:litert_opaque_options",
    "//litert/c/options:litert_cpu_options",
    "//litert/c/options:litert_darwinn_runtime_options",
    "//litert/c/options:litert_google_tensor_options",
    "//litert/c/options:litert_gpu_options",
    "//litert/c/options:litert_intel_openvino_options",
    "//litert/c/options:litert_mediatek_options",
    "//litert/c/options:litert_qualcomm_options",
    "//litert/c/options:litert_runtime_options",
]

cc_library(
    name = "litert_cpu_options",
    srcs = ["//litert/cc/options:litert_cpu_options.cc"],
    hdrs = ["//litert/cc/options:litert_cpu_options.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_gpu_options",
    srcs = ["//litert/cc/options:litert_gpu_options.cc"],
    hdrs = ["//litert/cc/options:litert_gpu_options.h"],
    deps = [
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_gpu_options_test",
    srcs = ["//litert/cc/options:litert_gpu_options_test.cc"],
    deps = [
        ":litert_gpu_options",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/c/options:litert_gpu_options",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_cpu_options_test",
    srcs = ["//litert/cc/options:litert_cpu_options_test.cc"],
    deps = [
        ":litert_cpu_options",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_qualcomm_options",
    srcs = ["//litert/cc/options:litert_qualcomm_options.cc"],
    hdrs = ["//litert/cc/options:litert_qualcomm_options.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_google_tensor_options",
    srcs = ["//litert/cc/options:litert_google_tensor_options.cc"],
    hdrs = ["//litert/cc/options:litert_google_tensor_options.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_mediatek_options",
    srcs = ["//litert/cc/options:litert_mediatek_options.cc"],
    hdrs = ["//litert/cc/options:litert_mediatek_options.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_intel_openvino_options",
    srcs = ["//litert/cc/options:litert_intel_openvino_options.cc"],
    hdrs = ["//litert/cc/options:litert_intel_openvino_options.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_runtime_options",
    srcs = ["//litert/cc/options:litert_runtime_options.cc"],
    hdrs = ["//litert/cc/options:litert_runtime_options.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_runtime_options_test",
    srcs = ["//litert/cc/options:litert_runtime_options_test.cc"],
    deps = [
        ":litert_runtime_options",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/c/options:litert_runtime_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_darwinn_options",
    srcs = ["//litert/cc/options:litert_darwinn_options.cc"],
    hdrs = ["//litert/cc/options:litert_darwinn_options.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "//litert/runtime:litert_darwinn_options",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_darwinn_options_test",
    srcs = ["//litert/cc/options:litert_darwinn_options_test.cc"],
    deps = [
        ":litert_darwinn_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_compiler_options",
    srcs = ["//litert/cc/options:litert_compiler_options.cc"],
    hdrs = ["//litert/cc/options:litert_compiler_options.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)
