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

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "litert_bin", "litert_dynamic_lib", "litert_lib")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "device_context.cc",
        "dispatch_api.cc",
        "invocation_context.cc",
    ],
    hdrs = [
        "device_context.h",
        "invocation_context.h",
    ],
    copts = [
        "-Os",
        "-fexceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-ffunction-sections",
        "-fdata-sections",
    ] + select({
        "//litert/runtime/dispatch:windows_msvc": [
            "/DLITERT_COMPILE_LIBRARY",
            "/wd4018",  # -Wno-sign-compare
        ],
        "//litert/runtime/dispatch:windows_lexan": [
            "-DLITERT_COMPILE_LIBRARY",
            "-Wno-sign-compare",
        ],
        "//conditions:default": [],
    }),
    export_litert_only = True,
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    linkopts = litert_android_linkopts() + [
        "-Wl,-soname=libLiteRtDispatch_IntelOpenvino.so",
        "-Wl,-lc++abi",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_IntelOpenvino.so",
    visibility = ["//litert:__subpackages__"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/intel_openvino:ov_utils",
        "@com_google_absl//absl/cleanup",
        "@intel_openvino//:openvino",
    ],
)

litert_bin(
    name = "LiteRtDispatch",
    srcs = [
        "device_context.cc",
        "dispatch_api.cc",
        "invocation_context.cc",
        "openvino_shared_core.cc",
        "remote_tensor_buffer.cc",
    ],
    copts = ["-fexceptions"] + select({
        "//litert/runtime/dispatch:windows_msvc": [
            "/DLITERT_COMPILE_LIBRARY",
            "/wd4018",  # -Wno-sign-compare
        ],
        "//litert/runtime/dispatch:windows_lexan": [
            "-DLITERT_COMPILE_LIBRARY",
            "-Wno-sign-compare",
        ],
        "//conditions:default": [],
    }),
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    linkopts = litert_android_linkopts(),
    linkshared = 1,
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":litert_dispatch_api",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c:litert_tensor_buffer",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/intel_openvino:ov_utils",
        "@com_google_absl//absl/cleanup",
        "@intel_openvino//:openvino",
    ],
)

litert_lib(
    name = "litert_dispatch_api",
    hdrs = [
        "device_context.h",
        "invocation_context.h",
        "openvino_shared_core.h",
        "remote_tensor_buffer.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/intel_openvino:ov_utils",
        "@com_google_absl//absl/cleanup",
        "@intel_openvino//:openvino",
    ],
)

#Test targets for Openvino Dispatch
cc_test(
    name = "dispatch_api_openvino_test",
    srcs = [
        "dispatch_api_openvino_test.cc",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    #TODO:Uncomment this when NPU bin test data file is added in follow-up PR
    #data = [
    #    "//litert/test:testdata/ov_model_add1d_FP32.blob",
    #],
    linkopts = litert_android_linkopts(),
    linkstatic = 1,
    tags = [
        "no-remote-exec",
        "no_oss",
        "notap",
    ],
    deps = [
        ":dispatch_api",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_any",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/core:filesystem_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@intel_openvino//:openvino",
    ],
)
