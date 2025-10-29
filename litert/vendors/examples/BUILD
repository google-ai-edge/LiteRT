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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_test", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

litert_dynamic_lib(
    name = "example_plugin",
    srcs = [
        "example_plugin.cc",
    ],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    linkstatic = 1,
    shared_lib_name = "example_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Example.so",
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":example_common",
        ":example_transformations",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_op_options",
        "@com_google_absl//absl/strings:str_format",
    ],
)

litert_dynamic_lib(
    name = "example_dispatch_lib",
    srcs = [
        "example_dispatch.cc",
    ],
    hdrs = ["//litert/vendors/c:litert_dispatch.h"],
    export_litert_only = True,
    linkstatic = 1,
    shared_lib_name = "example_dispatch_so",
    so_name = "libLiteRtDispatch_Example.so",
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":example_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer_requirements",
        "//litert/cc/internal:litert_handle",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "example_plugin_test",
    srcs = [
        "example_plugin_test.cc",
    ],
    data = ["//litert/test:mlir_test_data"],
    # TODO: b/448411540 - Re-enable sans for this test once test helpers support tests w/o dlopen
    # that depend on dynamic runtime libraries.
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    deps = [
        ":example_plugin",  # buildcleaner: keep
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/core/model:model_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "example_common",
    srcs = ["example_common.cc"],
    hdrs = ["example_common.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_detail",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "example_common_test",
    srcs = ["example_common_test.cc"],
    deps = [
        ":example_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "example_dispatch",
    testonly = True,
    srcs = ["example_dispatch.cc"],
    deps = [
        ":example_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer_requirements",
        "//litert/cc/internal:litert_handle",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "example_dispatch_test",
    srcs = ["example_dispatch_test.cc"],
    deps = [
        ":example_dispatch",  # buildcleaner: keep
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer_requirements",
        "//litert/cc/internal:litert_handle",
        "//litert/test:matchers",
        "//litert/test:simple_buffer",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "end_to_end_test",
    srcs = ["end_to_end_test.cc"],
    data = [
        ":example_dispatch_so",
        ":libLiteRtCompilerPlugin_Example.so",
        "//litert/test:mlir_test_data",
    ],
    # This test dlopens, so sanitzers are prone to false positive.
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "example_transformations",
    srcs = ["example_transformations.cc"],
    hdrs = ["example_transformations.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_rewriter",
    ],
)

litert_test(
    name = "example_transformations_test",
    srcs = ["example_transformations_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        "nosan",
    ],
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":example_transformations",
        "//litert/c:litert_op_code",
        "//litert/core/model:model_testonly",
        "//litert/test:common",
        "//litert/test:matchers_oss",
    ],
)
