# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("//litert/build_common:litert_build_defs.bzl", "litert_test", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["//:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_test(
    name = "relu_test",
    srcs = [
        "relu_test.cc",
    ],
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
        # Don't build/test in OS until qnn is available.
        "no-remote-exec",
        "nobuilder",
        "notap",
    ],
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:relu_op_builder",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_test(
    name = "topk_test",
    srcs = [
        "topk_test.cc",
    ],
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
        # Don't build/test in OS until qnn is available.
        "no-remote-exec",
        "nobuilder",
        "notap",
    ],
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:topk_op_builder",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_test(
    name = "elementwise_test",
    srcs = [
        "elementwise_test.cc",
    ],
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
        # Don't build/test in OS until qnn is available.
        "no-remote-exec",
        "nobuilder",
        "notap",
    ],
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:elementwise_op_builder",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ],
)

litert_test(
    name = "fully_connected_int2_test",
    srcs = [
        "fully_connected_int2_test.cc",
    ],
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
        # Don't build/test in OS until qnn is available.
        "no-remote-exec",
        "nobuilder",
        "notap",
    ],
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:fully_connected_op_builder",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ],
)
