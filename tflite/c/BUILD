load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load(
    "//tflite:build_def.bzl",
    "tflite_cc_library_with_c_headers_test",
    "tflite_cc_shared_object",
    "tflite_copts",
    "tflite_copts_warnings",
    "tflite_custom_c_library",
    "tflite_linkopts_no_undefined",
    "tflite_self_contained_libs_test_suite",
)
load("//tflite:special_rules.bzl", "c_api_opaque_internal_visibility_allowlist", "tflite_portable_test_suite")
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite_with_c_headers_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Generates a platform-specific shared library containing the TensorFlow Lite C
# API implementation as define in `c_api.h`. The exact output library name
# is platform dependent:
#   - Linux/Android: `libtensorflowlite_c.so`
#   - Mac: `libtensorflowlite_c.dylib`
#   - Windows: `tensorflowlite_c.dll`
tflite_cc_shared_object(
    name = "tensorflowlite_c",
    linkopts = tflite_linkopts_no_undefined() + select({
        "@org_tensorflow//tensorflow:ios": [
            "-Wl,-exported_symbols_list,$(location //tflite/c:exported_symbols.lds)",
        ],
        "@org_tensorflow//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tflite/c:exported_symbols.lds)",
        ],
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,--version-script,$(location //tflite/c:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":exported_symbols.lds",
        ":version_script.lds",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "c_api",
    hdrs = [
        "c_api.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    generate_opaque_delegate_target = True,
    deps = [
        ":c_api_without_op_resolver",
        "//tflite/core/async/c:types",
        "//tflite/core/c:c_api",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "c_api_for_testing",
    testonly = True,
    srcs = ["c_api_for_testing.cc"],
    hdrs = ["c_api_for_testing.h"],
    deps = [
        ":c_api",
        ":c_api_internal",
        "//tflite/core/c:c_api",
    ],
)

# TODO(b/248514738): Deprecate this target in favour
# of //tflite/core/c:c_api_without_op_resolver after all known existing uses of
# those targets have been changed to instead refer to the new target.
# Same as ":c_api", but doesn't link in the default CreateOpResolver implementation.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api_without_op_resolver",
    hdrs = ["c_api.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_types",
        "//tflite/core/c:c_api_without_op_resolver",
    ],
)

# TODO(b/248514738): Deprecate this target in favour
# of //tflite/core/c:c_api_without_op_resolver_without_alwayslink after all
# known existing uses of those targets have been changed to instead refer to the new target.
# Same as ":c_api_without_op_resolver", but without alwayslink=1.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api_without_op_resolver_without_alwayslink",
    hdrs = ["c_api.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = ["//tflite/core/c:c_api_without_op_resolver_without_alwayslink"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "c_api_experimental",
    hdrs = [
        "c_api_experimental.h",
        # ':c_api_opaque' was promoted to the 'mostly stable' API.
        # Please explicitly add a dependency on ':c_api_opaque' if your target
        # needs both ':c_api_experimental' and ':c_api_opaque' dependencies.
        # We plan to remove 'c_api_opaque.h' from the list of exposed headers and
        # will remove the dependency from this target on ':c_api_opaque' in the future.
        "c_api_opaque.h",
    ],
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        ":c_api",
    ],
    deps = [
        "//tflite/core/c:c_api_experimental",
        "//tflite/core/c:c_api_opaque",
    ],
)

# Same as ":c_api_experimental", but without linking in the default CreateOpResolver implementation.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api_experimental_without_op_resolver",
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
    copts = tflite_copts() + tflite_copts_warnings(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    tflite_deps = [
        ":c_api_without_op_resolver",
    ],
    deps = [
        "//tflite/core/c:c_api_experimental_without_op_resolver",
        "//tflite/core/c:c_api_opaque_without_op_resolver",
    ],
)

# Same as ":c_api_experimental", but without linking in the default CreateOpResolver implementation,
# and without depending on targets that use alwayslink=1.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api_experimental_without_op_resolver_without_alwayslink",
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
    copts = tflite_copts() + tflite_copts_warnings(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    tflite_deps = [":c_api_without_op_resolver_without_alwayslink"],
    deps = [
        "//tflite/core/c:c_api_experimental_without_op_resolver_without_alwayslink",
        "//tflite/core/c:c_api_opaque_without_op_resolver_without_alwayslink",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "c_api_opaque",
    hdrs = [
        "c_api_opaque.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    tflite_deps = [":c_api"],
    deps = ["//tflite/core/c:c_api_opaque"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "c_api_opaque_without_op_resolver",
    hdrs = [
        "c_api_opaque.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    tflite_deps = [":c_api_without_op_resolver"],
    deps = ["//tflite/core/c:c_api_opaque_without_op_resolver"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "c_api_opaque_without_op_resolver_without_alwayslink",
    hdrs = [
        "c_api_opaque.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    tflite_deps = [
        ":c_api_without_op_resolver_without_alwayslink",
    ],
    deps = ["//tflite/core/c:c_api_opaque_without_op_resolver_without_alwayslink"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    deps = ["//tflite/core/c:c_api_types"],
)

cc_library(
    name = "c_api_internal",
    hdrs = ["c_api_internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tflite/core/async/c:__subpackages__",
        "//tflite/core/c:__subpackages__",
    ],
    deps = [
        ":common_internal",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/profiling/telemetry/c:profiler",
    ],
)

filegroup(
    name = "c_api_test_builtin_op_models",
    testonly = 1,
    srcs = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/add_quantized.bin",
    ],
)

# TODO(b/248514738): Deprecate this target in favour
# of //tflite/core/c:c_api_test after all known existing uses of those targets
# have been changed to instead refer to the new target.
test_suite(
    name = "c_api_test",
    tests = [
        "//tflite/core/c:c_api_test",
    ],
)

tflite_custom_c_library(
    name = "selectively_built_c_api_test_lib",
    testonly = 1,
    experimental = True,
    models = [":c_api_test_builtin_op_models"],
    visibility = [
        "//tflite/core/c:__subpackages__",
    ],
)

# TODO(b/248514738): Deprecate this target in favour
# of //tflite/core/c:selectively_built_c_api_test after all known existing uses
# of those targets have been changed to instead refer to the new target.
test_suite(
    name = "selectively_built_c_api_test",
    tests = [
        "//tflite/core/c:selectively_built_c_api_test",
    ],
)

test_suite(
    name = "c_api_experimental_test",
    tests = [
        "//tflite/core/c:c_api_experimental_test",
    ],
)

cc_test(
    name = "c_api_signature_runner_test",
    size = "small",
    srcs = ["c_api_signature_runner_test.cc"],
    copts = tflite_copts(),
    data = [
        "//tflite:testdata/multi_signatures.bin",
        "//tflite:testdata/no_signatures.bin",
    ],
    deps = [
        ":c_api",
        "//tflite/core/c:c_api",
        "//tflite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "common",
    hdrs = [
        "builtin_op_data.h",
        "common.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        ":c_api_types",
    ],
    deps = [
        "//tflite:tflite_kernel_use_xnnpack_optional",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ] + select({
        "//tflite:tensorflow_profiler_config": [
            "//tflite:macros",
            "//tflite:tensorflow_profiler_logger_shim",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "common_internal",
    srcs = ["common_internal.cc"],
    hdrs = ["common_internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
)

cc_library(
    name = "c_api_opaque_internal",
    srcs = ["c_api_opaque_internal.cc"],
    hdrs = ["c_api_opaque_internal.h"],
    compatible_with = get_compatible_with_portable(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    visibility = [
        "//tflite:__subpackages__",
    ] + c_api_opaque_internal_visibility_allowlist(),
    deps = [
        ":c_api_types",
        "//tflite:framework",
        "//tflite/c:c_api_internal",
        "//tflite/core/api:op_resolver",
        "//tflite/core/c:common",
        "//tflite/core/c:operator",
    ],
)

cc_test(
    name = "c_api_opaque_internal_test",
    srcs = ["c_api_opaque_internal_test.cc"],
    data = [
        "//tflite:testdata/add.bin",
    ],
    deps = [
        ":c_api_internal",
        ":c_api_opaque_internal",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

# Same as c_api_opaque_internal, but depends on the
# '_without_alwayslink' variant of ':c_api_without_op_resolver'.
cc_library(
    name = "c_api_opaque_internal_without_alwayslink",
    srcs = ["c_api_opaque_internal.cc"],
    hdrs = ["c_api_opaque_internal.h"],
    compatible_with = get_compatible_with_portable(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        ":c_api_types",
        "//tflite:framework",
        "//tflite/c:c_api_internal",
        "//tflite/core/api:op_resolver",
        "//tflite/core/c:common",
        "//tflite/core/c:operator_without_alwayslink",
    ],
)

# For use with library targets that can't use relative paths.
# LINT.IfChange(exported_headers)
exports_files([
    "builtin_op_data.h",
    "c_api.h",
    "c_api_experimental.h",
    "c_api_opaque.h",
    "c_api_types.h",
    "common.h",
])

filegroup(
    name = "tensorflowlite_c_api_hdrs_filegroup",
    srcs = [
        "builtin_op_data.h",
        "c_api.h",
        "c_api_types.h",
        "common.h",
    ],
)

alias(
    name = "tensorflowlite_c_impl_hdrs_filegroup",
    actual = "//tflite/core/c:headers_filegroup",
)
# LINT.ThenChange(../java/BUILD:TFLITE_HEADERS)

# Test the C extension API code.
test_suite(
    name = "common_test",
    tests = [
        "//tflite/core/c:common_test",
    ],
)

test_suite(
    name = "builtin_op_data_test",
    tests = [
        "//tflite/core/c:builtin_op_data_test",
    ],
)

cc_test(
    name = "c_test",
    size = "small",
    srcs = ["c_test.c"],
    copts = tflite_copts(),
    data = [
        "//tflite:testdata/add.bin",
        "//tflite:testdata/multi_signatures.bin",
    ],
    tags = [
        # Testing on Android uses "--gunit_output=xml:test.xml" runtime flag,
        # but this test is C code, not C++, and hence doesn't use GUnit,
        # so doesn't support that flag.
        "tflite_not_portable_android",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        "//tflite/core/c:c_api",
        "//tflite/core/c:c_api_experimental",
        "//tflite/core/c:c_api_opaque",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    generate_opaque_delegate_target = True,
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")

tflite_portable_test_suite()
