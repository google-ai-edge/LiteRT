# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

load("//litert/build_common:litert_build_defs.bzl", "litert_test", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
    features = ["-layering_check"],  # buildifier: disable=no-layering-check
)

_COMPATIBLE_OS = select({
    "//litert:android": [],
    "//litert:linux": [],
    "//litert:windows": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

_QNN_DEPS = select({
    "@org_tensorflow//tensorflow:linux_x86_64": [
        # copybara:uncomment_begin(google-only)
        # "@qairt//:linux_deps",
        # copybara:uncomment_end
    ],
    "//conditions:default": [],
})

litert_test(
    name = "relu_test",
    srcs = [
        "relu_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = _COMPATIBLE_OS,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core/builders:relu_op_builder",
        "//litert/vendors/qualcomm/core/utils:test_main",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ] + _QNN_DEPS,
)

litert_test(
    name = "topk_test",
    srcs = [
        "topk_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = _COMPATIBLE_OS,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core/builders:topk_op_builder",
        "//litert/vendors/qualcomm/core/utils:test_main",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ] + _QNN_DEPS,
)

litert_test(
    name = "elementwise_test",
    srcs = [
        "elementwise_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = _COMPATIBLE_OS,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core/builders:elementwise_op_builder",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/utils:test_main",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ] + _QNN_DEPS,
)

litert_test(
    name = "embedding_lookup_test",
    srcs = [
        "embedding_lookup_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = _COMPATIBLE_OS,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:embedding_lookup_op_builder",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/utils:test_main",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ] + _QNN_DEPS,
)

litert_test(
    name = "onehot_test",
    srcs = [
        "onehot_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = _COMPATIBLE_OS,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core/builders:onehot_op_builder",
        "//litert/vendors/qualcomm/core/utils:test_main",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ] + _QNN_DEPS,
)

litert_test(
    name = "pack_test",
    srcs = [
        "pack_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = _COMPATIBLE_OS,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core/builders:pack_op_builder",
        "//litert/vendors/qualcomm/core/utils:test_main",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ] + _QNN_DEPS,
)

litert_test(
    name = "splitv_test",
    srcs = [
        "splitv_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = _COMPATIBLE_OS,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core:op_code",
        "//litert/vendors/qualcomm/core/builders:splitv_op_builder",
        "//litert/vendors/qualcomm/core/utils:test_main",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ] + _QNN_DEPS,
)

litert_test(
    name = "fully_connected_int2_test",
    srcs = [
        "fully_connected_int2_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    no_main = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = _COMPATIBLE_OS,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core/builders:fully_connected_op_builder",
        "//litert/vendors/qualcomm/core/utils:miscs",
        "//litert/vendors/qualcomm/core/utils:test_main",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@qairt//:qnn_lib_headers",
    ] + _QNN_DEPS,
)

litert_test(
    name = "hadamard_transform_test",
    srcs = [
        "hadamard_transform_test.cc",
    ],
    linkopts = select({
        "//litert:android": [],
        "//conditions:default": [
            make_rpaths([
                "//litert/c:litert_runtime_c_api_so",
            ]),
        ],
    }),
    linkstatic = True,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    target_compatible_with = _COMPATIBLE_OS,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        "//litert/vendors/qualcomm/core:common",
        "//litert/vendors/qualcomm/core:tensor_pool",
        "//litert/vendors/qualcomm/core/builders:hadamard_transform_op_builder",
        "//litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//litert/vendors/qualcomm/qnn_backend_test:test_utils",
        "@com_google_absl//absl/numeric:bits",
        "@qairt//:qnn_lib_headers",
    ] + _QNN_DEPS,
)
