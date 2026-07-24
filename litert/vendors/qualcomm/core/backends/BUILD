# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "gtest_main_no_heapcheck_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
)

cc_library(
    name = "backend_factory",
    srcs = ["backend_factory.cc"],
    hdrs = ["backend_factory.h"],
    deps = [
        ":dsp_backend",
        ":gpu_backend",
        ":htp_backend",
        ":ir_backend",
        ":lpai_backend",
        ":qnn_backend",
        "//litert/vendors/qualcomm:common",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
    ],
)

cc_library(
    name = "backend_utils",
    hdrs = ["backend_utils.h"],
    deps = [
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "qnn_backend",
    srcs = ["qnn_backend.cc"],
    hdrs = ["qnn_backend.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        ":graph_config_builder",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "graph_config_builder",
    hdrs = ["graph_config_builder.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "ir_backend",
    srcs = ["ir_backend.cc"],
    hdrs = ["ir_backend.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        ":graph_config_builder",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "gpu_backend",
    srcs = ["gpu_backend.cc"],
    hdrs = ["gpu_backend.h"],
    deps = [
        ":backend_utils",
        ":graph_config_builder",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "htp_backend",
    srcs = ["htp_backend.cc"],
    hdrs = ["htp_backend.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        ":backend_utils",
        ":graph_config_builder",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "dsp_backend",
    srcs = ["dsp_backend.cc"],
    hdrs = ["dsp_backend.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        ":backend_utils",
        ":graph_config_builder",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "lpai_backend",
    srcs = ["lpai_backend.cc"],
    hdrs = ["lpai_backend.h"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
    deps = [
        ":backend_utils",
        ":graph_config_builder",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "gpu_backend_test",
    srcs = [
        "gpu_backend_test.cc",
    ],
    deps = [
        ":gpu_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "ir_backend_test",
    srcs = [
        "ir_backend_test.cc",
    ],
    deps = [
        ":ir_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "backend_factory_test",
    srcs = [
        "backend_factory_test.cc",
    ],
    deps = [
        ":backend_factory",
        ":dsp_backend",
        ":gpu_backend",
        ":htp_backend",
        ":ir_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "backend_utils_test",
    srcs = [
        "backend_utils_test.cc",
    ],
    deps = [
        ":backend_utils",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "graph_config_builder_test",
    srcs = [
        "graph_config_builder_test.cc",
    ],
    deps = [
        ":graph_config_builder",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "dsp_backend_test",
    srcs = [
        "dsp_backend_test.cc",
    ],
    deps = [
        ":backend_utils",
        ":dsp_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "htp_backend_test",
    srcs = [
        "htp_backend_test.cc",
    ],
    # copybara:uncomment malloc = "//base:system_malloc",
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    deps = [
        ":backend_utils",
        ":htp_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_absl//absl/base:no_destructor",
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

cc_test(
    name = "lpai_backend_test",
    srcs = [
        "lpai_backend_test.cc",
    ],
    deps = [
        ":lpai_backend",
        ":qnn_backend",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core/schema:soc_table",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)
