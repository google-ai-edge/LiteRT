# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_lib", "litert_test", "litert_bin")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_dynamic_lib(
    name = "openvino_compiler_plugin",
    srcs = ["openvino_compiler_plugin.cc"],
    hdrs = [
        "//litert/vendors/c:litert_compiler_plugin.h",
    ],
    export_litert_only = True,
    shared_lib_name = "intel_openvino_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_IntelOpenvino.so",
    tags = [
        "nobuilder",
        "notap",
    ],
    ungrte = True,
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":graph_iterator",
        "//litert/c:litert_common",
        "//litert/c:litert_dispatch_headers",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/c:litert_op_options",
        "//litert/c:litert_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "@com_google_absl//absl/strings",
        "@intel_openvino//:openvino",
    ],
)

litert_bin(
    name = "LiteRtCompilerPlugin",
    srcs = ["openvino_compiler_plugin.cc"],
    visibility = ["//visibility:public"],
    linkshared = 1,
    copts = select({
        "@org_tensorflow//tensorflow:windows": [
            "/DLITERT_COMPILE_LIBRARY",
            "/wd4018",  # -Wno-sign-compare
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":graph_iterator",
        "//litert/c:litert_common",
        "//litert/c:litert_dispatch_headers",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        ":litert_compiler_plugin_c_api",
        "@intel_openvino//:openvino",
    ],
)

litert_lib(
    name = "litert_compiler_plugin_c_api",
    hdrs = [
        "//litert/vendors/c:litert_compiler_plugin.h",
    ],
)

litert_lib(
    name = "graph_iterator",
    srcs = ["graph_iterator.cc"],
    hdrs = [
        "graph_iterator.h",
    ],
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        ":decoder",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_model",
        "//litert/vendors/intel_openvino:ov_utils",
        "@intel_openvino//:openvino",
    ],
)

litert_lib(
    name = "decoder",
    srcs = ["decoder.cc"],
    hdrs = ["decoder.h"],
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_dispatch_headers",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/c:litert_op_options",
        "//litert/c:litert_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/vendors/intel_openvino:ov_utils",
        "//tflite/schema:schema_fbs",
        "@intel_openvino//:openvino",
    ],
)

litert_test(
    name = "decoder_test",
    srcs = [
        "decoder_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        ":decoder",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:test_models",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@intel_openvino//:openvino",
    ],
)

litert_test(
    name = "openvino_compiler_plugin_test",
    srcs = [
        "openvino_compiler_plugin_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        "no_oss",
        "nobuilder",
    ],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":openvino_compiler_plugin",  # buildcleaner: keep
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:test_models",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@intel_openvino//:openvino",
    ],
)
