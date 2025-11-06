# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("//litert/build_common:litert_build_defs.bzl", "litert_test", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_test(
    name = "qnn_backend_test",
    srcs = [
        "qnn_backend_test.cc",
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
        "//litert/test:common",
        "//litert/vendors/qualcomm:qnn_manager",
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:relu_op_builder",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/utils:qnn_model",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/builders:elementwise_op_builder",
        "@qairt//:qnn_lib_headers",
    ],
)
