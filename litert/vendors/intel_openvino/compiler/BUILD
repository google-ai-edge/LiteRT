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

load("//litert/build_common:litert_build_defs.bzl", "litert_bin", "litert_dynamic_lib", "litert_lib", "litert_test")

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
    export_litert_only = True,
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    shared_lib_name = "intel_openvino_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_IntelOpenvino.so",
    ungrte = True,
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":graph_iterator",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/vendors/cc:options_helper",
        "@com_google_absl//absl/strings:str_format",
        "@intel_openvino//:openvino",
    ],
)

litert_bin(
    name = "LiteRtCompilerPlugin",
    srcs = ["openvino_compiler_plugin.cc"],
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
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":graph_iterator",
        ":litert_compiler_plugin_c_api",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/vendors/cc:options_helper",
        "@com_google_absl//absl/strings:str_format",
        "@intel_openvino//:openvino",
    ],
)

litert_lib(
    name = "litert_compiler_plugin_c_api",
    hdrs = [
        "//litert/vendors/c:litert_compiler_plugin.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
    ],
)

litert_lib(
    name = "graph_iterator",
    srcs = ["graph_iterator.cc"],
    hdrs = [
        "graph_iterator.h",
    ],
    copts = [
        "-Wno-unused-variable",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    deps = [
        ":decoder",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/intel_openvino:ov_utils",
        "@intel_openvino//:openvino",
    ],
)

litert_lib(
    name = "decoder",
    srcs = ["decoder.cc"],
    hdrs = ["decoder.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
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
    copts = ["-fexceptions"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    tags = [
        "notap",
    ],
    deps = [
        ":decoder",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc/dynamic_runtime:litert_extended_model",
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
    copts = ["-fexceptions"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        "no_oss",
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
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model_predicates",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:test_models",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@intel_openvino//:openvino",
    ],
)
