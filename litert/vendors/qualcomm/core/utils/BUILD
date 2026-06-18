# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:litert_build_defs.bzl", "litert_test")
load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert/cc:__subpackages__",
        "//litert/vendors/qualcomm:__subpackages__",
    ],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
)

cc_library(
    name = "log",
    srcs = select({
        "//litert:android": ["log_android.cc"],
        "//conditions:default": ["log_default.cc"],
    }),
    hdrs = ["log.h"],
    linkopts = select({
        "//litert:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/vendors/qualcomm/core:common",
    ],
)

cc_library(
    name = "miscs",
    srcs = ["miscs.cc"],
    hdrs = ["miscs.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    linkopts = ["-ldl"],
    deps = [
        ":log",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "test_main",
    testonly = True,
    srcs = ["test_main.cc"],
    deps = [
        ":test_utils",
        # copybara:uncomment_begin(google-only)
        # "//testing/base/public:gunit_for_library_testonly",
        # copybara:uncomment_end
        # copybara:comment_begin(oss-only)
        "@com_google_googletest//:gtest",
        # copybara:comment_end
        "@com_google_absl//absl/strings",
        "//litert/vendors/qualcomm/core:common",
    ],
    alwayslink = 1,
)

cc_library(
    name = "test_main_no_heapcheck",
    testonly = True,
    srcs = ["test_main.cc"],
    deps = [
        ":test_utils",
        # copybara:uncomment_begin(google-only)
        # "//testing/base/public:gunit_for_library_testonly",
        # copybara:uncomment_end
        # copybara:comment_begin(oss-only)
        "@com_google_googletest//:gtest",
        # copybara:comment_end
        "@com_google_absl//absl/strings",
        "//litert/vendors/qualcomm/core:common",
    ],
    alwayslink = 1,
)

litert_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
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
        ":log",
        ":miscs",
        ":test_main_no_heapcheck",
        ":test_utils",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/backends:htp_backend",
        "//litert/vendors/qualcomm/core/backends:ir_backend",
    ] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            # copybara:uncomment_begin(google-only)
            # "@qairt//:linux_deps",
            # copybara:uncomment_end
        ],
        "//conditions:default": [],
    }),
)

litert_device_exec(
    name = "utils_device_test",
    backend_id = "qualcomm",
    target = ":utils_test",
)

cc_library(
    name = "qnn_model",
    srcs = ["qnn_model.cc"],
    hdrs = ["qnn_model.h"],
    deps = [
        ":log",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "flexbuffer_helpers",
    srcs = ["flexbuffer_helpers.cc"],
    hdrs = ["flexbuffer_helpers.h"],
    deps = [
        "@flatbuffers//:runtime_cc",
    ],
)

litert_test(
    name = "flexbuffer_helpers_test",
    srcs = ["flexbuffer_helpers_test.cc"],
    deps = [
        ":flexbuffer_helpers",
        "@flatbuffers",
    ],
)
