# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:litert_build_defs.bzl", "gtest_main_no_heapcheck_deps", "litert_test")
load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
)

cc_library(
    name = "op_code",
    hdrs = ["op_code.h"],
)

cc_library(
    name = "tensor_pool",
    srcs = ["tensor_pool.cc"],
    hdrs = ["tensor_pool.h"],
    deps = [
        "//litert/vendors/qualcomm/core/utils:log",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_test(
    name = "tensor_pool_test",
    srcs = [
        "tensor_pool_test.cc",
    ],
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    # This test can be run only on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":tensor_pool",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@qairt//:qnn_lib_headers",
    ] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            # copybara:uncomment_begin(google-only)
            # "@qairt//:linux_deps",
            # copybara:uncomment_end
        ],
        "//conditions:default": [],
    }) + gtest_main_no_heapcheck_deps(),
)

litert_device_exec(
    name = "tensor_pool_device_test",
    backend_id = "qualcomm",
    target = ":tensor_pool_test",
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    linkopts = select({
        "//litert:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_test(
    name = "common_test",
    srcs = [
        "common_test.cc",
    ],
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    # This test can be run only on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":common",
    ] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            # copybara:uncomment_begin(google-only)
            # "@qairt//:linux_deps",
            # copybara:uncomment_end
        ],
        "//conditions:default": [],
    }) + gtest_main_no_heapcheck_deps(),
)
