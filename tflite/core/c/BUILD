load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load(
    "//tflite:build_def.bzl",
    "tflite_cc_library_with_c_headers_test",
    "tflite_copts",
)
load(
    "//tflite/core/c:special_rules.bzl",
    "c_api_experimental_visibility_allowlist",
    "c_api_visibility_allowlist",
    "common_header_visibility_allowlist",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//tflite:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "builtin_op_data.h",
        "c_api.h",
        "c_api_experimental.h",
        "c_api_opaque.h",
        "c_api_types.h",
        "common.h",
        "operator.h",
    ],
    visibility = [
        "//tflite:__subpackages__",
    ],
)

bzl_library(
    name = "special_rules_bzl",
    srcs = ["special_rules.bzl"],
    visibility = ["//tflite:__subpackages__"],
)

filegroup(
    name = "headers_filegroup",
    srcs = [
        "builtin_op_data.h",
        "c_api.h",
        "c_api_types.h",
        "common.h",
        "operator.h",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core/c:lite_headers_filegroup",
    ],
)

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        "common.cc",
        "common.h",
    ],
    visibility = [
        "//tflite:__pkg__",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "c_api",
    hdrs = [
        "c_api.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tflite:__subpackages__",
    ] + c_api_visibility_allowlist(),
    deps = [
        ":c_api_types",
        ":c_api_without_op_resolver",
        ":operator",
        "//tflite:builtin_ops",
        "//tflite:create_op_resolver_with_builtin_ops",
        "//tflite/c:common",
        "//tflite/core/async/c:types",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "c_api_without_op_resolver",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        ":c_api_types",
        ":operator",
        "//tflite:builtin_ops",
        "//tflite:mutable_op_resolver",
        "//tflite:signature_runner",
        "//tflite:stderr_reporter",
        "//tflite:version",
        "//tflite/c:c_api_internal",
        "//tflite/c:common_internal",
        "//tflite/core:create_op_resolver_header",
        "//tflite/core:framework_stable",
        "//tflite/core/api",
        "//tflite/core/async/c:types",
        "//tflite/delegates:interpreter_utils",
        "//tflite/delegates/nnapi:nnapi_delegate",
        "//tflite/kernels/internal:compatibility",
        "//tflite/profiling/telemetry:profiler",
        "//tflite/schema:schema_fbs",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

tflite_cc_library_with_c_headers_test(
    name = "private_c_api_without_op_resolver",
    hdrs = ["c_api.h"],
    copts = tflite_copts(),
    tags = [
        "allow_undefined_symbols",  # For tflite::CreateOpResolver().
        "avoid_dep",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":c_api_types",
        ":operator",
        "//tflite:builtin_ops",
        "//tflite/core/async/c:types",
    ],
)

# Same as ":c_api_without_op_resolver", but without alwayslink=1.
tflite_cc_library_with_c_headers_test(
    name = "c_api_without_op_resolver_without_alwayslink",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_types",
        ":operator_without_alwayslink",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite:mutable_op_resolver",
        "//tflite:signature_runner",
        "//tflite:stderr_reporter",
        "//tflite:version",
        "//tflite/c:c_api_internal",
        "//tflite/c:common_internal",
        "//tflite/core:create_op_resolver_header",
        "//tflite/core:framework_stable",
        "//tflite/core/api",
        "//tflite/core/async/c:types",
        "//tflite/delegates:interpreter_utils",
        "//tflite/delegates/nnapi:nnapi_delegate",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:compatibility",
        "//tflite/profiling/telemetry:profiler",
        "//tflite/schema:schema_fbs",
    ],
)

# This is a private target, its visibility is set to public only to be
# used by "tflite_custom_c_library".
# Do not use this target directly and don't consider it as a part of the public API.
alias(
    name = "private_c_api_without_op_resolver_without_alwayslink",
    actual = ":c_api_without_op_resolver_without_alwayslink",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

cc_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    copts = tflite_copts(),
    data = [
        "//tflite:testdata/2_subgraphs.bin",
        "//tflite:testdata/add.bin",
        "//tflite:testdata/add_quantized.bin",
        "//tflite:testdata/custom_sinh.bin",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_types",
        ":common",
        "//tflite:builtin_ops",
        "//tflite:string_util",
        "//tflite/c:c_api_internal",
        "//tflite/core:subgraph",
        "//tflite/delegates:delegate_test_util",
        "//tflite/schema:schema_fbs",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "c_api_test_with_opaque_delegate",
    size = "small",
    srcs = ["c_api_test.cc"],
    copts = tflite_copts(),
    data = [
        "//tflite:testdata/2_subgraphs.bin",
        "//tflite:testdata/add.bin",
        "//tflite:testdata/add_quantized.bin",
        "//tflite:testdata/custom_sinh.bin",
    ],
    local_defines = ["TFLITE_USE_OPAQUE_DELEGATE"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_types",
        ":common",
        "//tflite:builtin_ops",
        "//tflite:string_util",
        "//tflite/c:c_api_internal",
        "//tflite/core:subgraph",
        "//tflite/delegates:delegate_test_util",
        "//tflite/schema:schema_fbs",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "selectively_built_c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    copts = tflite_copts(),
    data = [
        "//tflite:testdata/2_subgraphs.bin",
        "//tflite:testdata/add.bin",
        "//tflite:testdata/add_quantized.bin",
        "//tflite:testdata/custom_sinh.bin",
    ],
    deps = [
        ":c_api_experimental_without_op_resolver_without_alwayslink",
        ":c_api_types",
        ":c_api_without_op_resolver_without_alwayslink",
        ":common",
        "//tflite:builtin_ops",
        "//tflite:string_util",
        "//tflite/c:c_api_internal",
        "//tflite/c:selectively_built_c_api_test_lib",
        "//tflite/core:subgraph",
        "//tflite/delegates:delegate_test_util",
        "//tflite/schema:schema_fbs",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "builtin_op_data_test",
    size = "small",
    srcs = ["builtin_op_data_test.cc"],
    copts = ["-Wno-unused-variable"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
    ],
)

# This is a private target, its visibility is set to public only to be
# used by "tflite_custom_c_library".
# Do not use this target directly and don't consider it as a part of the public API.
alias(
    name = "private_c_api_types",
    actual = ":c_api_types",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tflite:__subpackages__",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/experimental/lrt:__subpackages__",
    ] + c_api_visibility_allowlist(),
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core/c:tflite_common",
    ],
)

# Test the C extension API code.
cc_test(
    name = "common_test",
    size = "small",
    srcs = ["common_test.cc"],
    deps = [
        ":c_api_types",
        ":common",
        "//tflite:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "c_api_opaque_test",
    size = "small",
    srcs = [
        "c_api_opaque.cc",
        "c_api_opaque.h",
        "c_api_opaque_test.cc",
    ],
    data = [
        "//tflite:testdata/custom_sinh.bin",
        "//tflite:testdata/with_metadata.bin",
    ],
    deps = [
        ":c_api",
        ":c_api_types",
        ":common",
        ":operator",
        "//tflite:builtin_ops",
        "//tflite:string_util",
        "//tflite:util",
        "//tflite/c:c_api_opaque_internal",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:subgraph",
        "//tflite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "builtin_op_data.h",
        "common.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//tflite:__subpackages__"] + common_header_visibility_allowlist(),
    deps = [
        ":c_api_types",
        "//tflite:tflite_kernel_use_xnnpack_optional",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core/c:tflite_common",
    ] + select({
        "//tflite:tensorflow_profiler_config": [
            "//tflite:macros",
            "//tflite:tensorflow_profiler_logger_shim",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

# This is a private target, its visibility is set to public only to be
# used by "tflite_custom_c_library" and "tflite_flex_cc_library".
# Do not use this target directly and don't consider it as a part of the public API.
alias(
    name = "private_common",
    actual = ":common",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "c_api_experimental",
    hdrs = [
        "c_api_experimental.h",
        # DEPRECATED
        # ':c_api_opaque' was promoted to the 'mostly stable' API.
        # Please explicitly add a dependency on ':c_api_opaque' if your target
        # needs both ':c_api_experimental' and ':c_api_opaque' dependencies.
        # We plan to remove 'c_api_opaque.h' from the list of exposed headers and
        # will remove the dependency from this target on ':c_api_opaque' in the future.
        "c_api_opaque.h",
    ],
    copts = tflite_copts(),
    visibility = [
        "//tflite:__subpackages__",
    ] + c_api_experimental_visibility_allowlist(),
    deps = [
        ":c_api",
        ":c_api_experimental_without_op_resolver",
        ":c_api_opaque",
        ":c_api_types",
        ":common",
        ":operator",
        "//tflite:builtin_ops",
        "//tflite:signature_runner",
        "//tflite/c:c_api_types",
        "//tflite/core:framework",
    ],
)

# Same as ":c_api_experimental", but without linking in the default CreateOpResolver implementation.
tflite_cc_library_with_c_headers_test(
    name = "c_api_experimental_without_op_resolver",
    srcs = [
        "c_api_experimental.cc",
    ],
    hdrs = [
        "c_api_experimental.h",
        # DEPRECATED
        # ':c_api_opaque_without_op_resolver' was promoted to the 'mostly stable' API.
        # Please explicitly add a dependency on ':c_api_opaque_without_op_resolver' if your target
        # needs both ':c_api_experimental_without_op_resolver' and ':c_api_opaque_without_op_resolver' dependencies.
        # We plan to remove 'c_api_opaque.h' from the list of exposed headers and
        # will remove the dependency from this target on ':c_api_opaque_without_op_resolver' in the future.
        "c_api_opaque.h",
    ],
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_opaque_without_op_resolver",
        ":c_api_types",
        ":c_api_without_op_resolver",
        ":common",
        ":operator",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/c:c_api_internal",
        "//tflite/c:c_api_types",
        "//tflite/core:framework",
        "//tflite/profiling/telemetry:profiler",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

# This is a private target, its visibility is set to public only to be
# used by "custom_c_library_with_tflite".
# Do not use this target directly and don't consider it as a part of the public API.
alias(
    name = "private_c_api_experimental_without_op_resolver",
    actual = ":c_api_experimental_without_op_resolver",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

# Same as ":c_api_experimental", but without linking in the default CreateOpResolver implementation.
tflite_cc_library_with_c_headers_test(
    name = "c_api_experimental_without_op_resolver_without_alwayslink",
    srcs = [
        "c_api_experimental.cc",
    ],
    hdrs = [
        "c_api_experimental.h",
        # DEPRECATED
        # ':c_api_opaque_without_op_resolver_without_alwayslink' was promoted to the 'mostly stable' API.
        # Please explicitly add a dependency on ':c_api_opaque_without_op_resolver_without_alwayslink'
        # if your target needs both ':c_api_experimental_without_op_resolver_without_alwayslink' and
        # ':c_api_opaque_without_op_resolver_without_alwayslink' dependencies.
        # We plan to remove 'c_api_opaque.h' from the list of exposed headers and
        # will remove the dependency from this target on
        # ':c_api_opaque_without_op_resolver_without_alwayslink' in the future.
        "c_api_opaque.h",
    ],
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_opaque_without_op_resolver_without_alwayslink",
        ":c_api_types",
        ":c_api_without_op_resolver_without_alwayslink",
        ":common",
        ":operator_without_alwayslink",
        "//tflite:framework",
        "//tflite:kernel_api",
        "//tflite/c:c_api_internal",
        "//tflite/c:c_api_types",
        "//tflite/core:framework",
        "//tflite/profiling/telemetry:profiler",
    ],
)

# This is a private target, its visibility is set to public only to be
# used by "tflite_custom_c_library".
# Do not use this target directly and don't consider it as a part of the public API.
alias(
    name = "private_c_api_experimental_without_op_resolver_without_alwayslink",
    actual = ":c_api_experimental_without_op_resolver_without_alwayslink",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "c_api_opaque",
    hdrs = [
        "c_api_opaque.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tflite:__subpackages__",
    ] + c_api_experimental_visibility_allowlist(),
    deps = [
        ":c_api",
        ":c_api_opaque_without_op_resolver",
        ":c_api_types",
        ":common",
        ":operator",
        "//tflite/core:framework",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "c_api_opaque_without_op_resolver",
    srcs = [
        "c_api_opaque.cc",
    ],
    hdrs = [
        "c_api_opaque.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_types",
        ":c_api_without_op_resolver",
        ":common",
        ":operator",
        "//tflite:string_util",
        "//tflite:util",
        "//tflite/c:c_api_opaque_internal",
        "//tflite/core:framework",
        "//tflite/kernels:kernel_util",
    ],
)

# This is a private target, its visibility is set to public only to be
# used by "custom_c_library_with_tflite".
# Do not use this target directly and don't consider it as a part of the public API.
alias(
    name = "private_c_api_opaque_without_op_resolver",
    actual = ":c_api_opaque_without_op_resolver",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

# Same as ":c_api_opaque_without_op_resolver", but without alwayslink=1.
tflite_cc_library_with_c_headers_test(
    name = "c_api_opaque_without_op_resolver_without_alwayslink",
    srcs = [
        "c_api_opaque.cc",
    ],
    hdrs = [
        "c_api_opaque.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_types",
        ":c_api_without_op_resolver_without_alwayslink",
        ":common",
        ":operator_without_alwayslink",
        "//tflite:string_util",
        "//tflite:util",
        "//tflite/c:c_api_opaque_internal_without_alwayslink",
        "//tflite/core:framework",
        "//tflite/kernels:kernel_util",
    ],
)

# This is a private target, its visibility is set to public only to be
# used by "tflite_custom_c_library".
# Do not use this target directly and don't consider it as a part of the public API.
alias(
    name = "private_c_api_opaque_without_op_resolver_without_alwayslink",
    actual = ":c_api_opaque_without_op_resolver_without_alwayslink",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

cc_test(
    name = "c_api_experimental_test",
    size = "small",
    srcs = ["c_api_experimental_test.cc"],
    copts = tflite_copts(),
    data = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/custom_sinh.bin",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":common",
        "//tflite:kernel_api",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/delegates:delegate_test_util",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "operator",
    srcs = ["operator.cc"],
    hdrs = ["operator.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tflite:__subpackages__",
    ] + c_api_visibility_allowlist(),
    deps = [
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/c:common_internal",
        "//tflite/core/async/c:types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
    # When building shared libraries, we need alwayslink here, since the functions
    # in this header are part of the public API.
    alwayslink = 1,
)

tflite_cc_library_with_c_headers_test(
    name = "operator_without_alwayslink",
    srcs = ["operator.cc"],
    hdrs = ["operator.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/c:common_internal",
        "//tflite/core/async/c:types",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
)
