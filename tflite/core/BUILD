load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_self_contained_libs_test_suite")
load("//tflite:special_rules.bzl", "internal_visibility_allowlist", "tflite_portable_test_suite")
load("//tflite/core:special_rules.bzl", "core_cc_api_stable_visibility_allowlist", "macros_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "create_op_resolver.h",
        "macros.h",
        "subgraph.h",
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

# The public target for the C++ API excluding experimental APIs.
# TODO(ahentz): investigate dependency on gemm_support requiring usage of tf_copts.
cc_library(
    name = "framework_stable",
    srcs = [
        "subgraph.h",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "macros.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//tflite:__subpackages__"],
    deps = [
        ":cc_api_stable",
        ":signature_runner",
        "//tflite:allocation",
        "//tflite:array",
        "//tflite:external_cpu_backend_context",
        "//tflite:graph_info",
        "//tflite:interpreter_options_header",
        "//tflite:macros",
        "//tflite:memory_planner",
        "//tflite:mutable_op_resolver",
        "//tflite:stderr_reporter",
        "//tflite:string",
        "//tflite:type_to_tflitetype",
        "//tflite:util",
        "//tflite/c:common_internal",
        "//tflite/converter/core:model_builder_base",
        "//tflite/converter/experimental/remat:metadata_util",
        "//tflite/core/api",
        "//tflite/core/async:async_signature_runner",
        "//tflite/core/c:common",
        "//tflite/experimental/resource",
        "//tflite/internal:signature_def",
        "//tflite/profiling:root_profiler",
        "//tflite/profiling/telemetry:profiler",
        "//tflite/profiling/telemetry/c:telemetry_setting",
        "//tflite/profiling/telemetry/c:telemetry_setting_internal",
        "//tflite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

# The public target for the full C++ API, including experimental APIs.
#
# Experimental APIs are functional, tested and usable in production; however,
# the corresponding API surface has not been finalized, and is subject to
# change.
alias(
    name = "framework",
    actual = "framework_experimental",
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

# The full C++ API, including experimental APIs.
#
# Experimental APIs are functional, tested and usable in production; however,
# the corresponding API surface has not been finalized, and is subject to
# change.
#
# Note that if you have code which depends on both stable and experimental API
# features, it's fine to depend only on 'framework_experimental', since
# that includes 'framework_stable' as a subset.
cc_library(
    name = "framework_experimental",
    srcs = [],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "macros.h",
        "model.h",
        "model_builder.h",
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        ":cc_api_experimental",
        ":cc_api_stable",
        ":model_builder",
        ":signature_runner",
        "//tflite:allocation",
        "//tflite:array",
        "//tflite:external_cpu_backend_context",
        "//tflite:graph_info",
        "//tflite:interpreter_options_header",
        "//tflite:macros",
        "//tflite:memory_planner",
        "//tflite:mutable_op_resolver",
        "//tflite:stderr_reporter",
        "//tflite:string",
        "//tflite:type_to_tflitetype",
        "//tflite:util",
        "//tflite/c:common_internal",
        "//tflite/converter/core:model_builder_base",
        "//tflite/converter/experimental/remat:metadata_util",
        "//tflite/core/api",
        "//tflite/core/async:async_signature_runner",
        "//tflite/core/c:common",
        "//tflite/experimental/resource",
        "//tflite/internal:signature_def",
        "//tflite/profiling:root_profiler",
        "//tflite/profiling/telemetry:profiler",
        "//tflite/profiling/telemetry/c:telemetry_setting",
        "//tflite/profiling/telemetry/c:telemetry_setting_internal",
        "//tflite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

# This is a private target, its visibility is set to public only to be
# used by LiteRT dependencies.
# Do not use this target directly and don't consider it as a part of the public API.
# TODO(weiyiw): Refactor LiteRT deps from TFLite.
alias(
    name = "private_cc_api_stable",
    actual = ":cc_api_stable",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

# TODO(b/242310498): move logger.cc from tensorflow/lite/ to here.
cc_library(
    name = "cc_api_stable",
    srcs = [
        "interpreter.cc",
        "interpreter_builder.cc",
        "subgraph.h",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "@org_tensorflow//third_party/deepmind/lyria_live/internal/odml:__subpackages__",
        # copybara:uncomment "@org_tensorflow//third_party/odml/litert/litert:__subpackages__",
        # "@org_tensorflow//third_party/odml/litert:__subpackages__",  # copybara:uncomment
        "//tflite:__subpackages__",
    ] + core_cc_api_stable_visibility_allowlist(),
    deps = [
        ":model_builder",
        ":signature_runner",
        ":subgraph",
        "//tflite:allocation",
        "//tflite:array",
        "//tflite:external_cpu_backend_context",
        "//tflite:graph_info",
        "//tflite:interpreter_options_header",
        "//tflite:macros",
        "//tflite:memory_planner",
        "//tflite:minimal_logging",
        "//tflite:mutable_op_resolver",
        "//tflite:shared_library",
        "//tflite:simple_memory_arena",
        "//tflite:stderr_reporter",
        "//tflite:string",
        "//tflite:tensorflow_profiler_logger_shim",
        "//tflite:type_to_tflitetype",
        "//tflite:util",
        "//tflite:version",
        "//tflite/c:common_internal",
        "//tflite/converter/core:model_builder_base",
        "//tflite/converter/experimental/remat:metadata_util",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_utils",
        "//tflite/core/api",
        "//tflite/core/api:verifier",
        "//tflite/core/async:async_signature_runner",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/delegates:telemetry",
        "//tflite/delegates/xnnpack:tflite_with_xnnpack_qs8",
        "//tflite/delegates/xnnpack:tflite_with_xnnpack_qu8",
        "//tflite/experimental/resource",
        "//tflite/internal:signature_def",
        "//tflite/kernels/internal:compatibility",
        "//tflite/profiling:platform_profiler",
        "//tflite/profiling:root_profiler",
        "//tflite/profiling/telemetry",
        "//tflite/profiling/telemetry:profiler",
        "//tflite/profiling/telemetry/c:telemetry_setting",
        "//tflite/profiling/telemetry/c:telemetry_setting_internal",
        "//tflite/schema:conversion_metadata_fbs",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
        "@ruy//ruy:denormal",
    ],
    alwayslink = 1,
)

# The key parts of the C++ API.  This target defines the TF Lite classes for
# loading models and interpreting them.
# DEPRECATED: prefer to depend on :cc_api_stable or :cc_api_experimental.
alias(
    name = "cc_api",
    actual = "cc_api_experimental",
    visibility = ["//tflite:__subpackages__"],
)

# The key parts of the C++ API, including experimental APIs.
#
# This target has restricted visibility; for a public target that exposes
# these APIs, see 'framework_experimental' above.
cc_library(
    name = "cc_api_experimental",
    srcs = [
        "interpreter_experimental.cc",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        ":cc_api_stable",
        ":signature_runner",
        "//tflite:allocation",
        "//tflite:array",
        "//tflite:external_cpu_backend_context",
        "//tflite:graph_info",
        "//tflite:interpreter_options_header",
        "//tflite:macros",
        "//tflite:memory_planner",
        "//tflite:mutable_op_resolver",
        "//tflite:stderr_reporter",
        "//tflite:string",
        "//tflite:type_to_tflitetype",
        "//tflite:util",
        "//tflite/c:common_internal",
        "//tflite/converter/core:model_builder_base",
        "//tflite/converter/experimental/remat:metadata_util",
        "//tflite/core/api",
        "//tflite/core/async:async_signature_runner",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/experimental/resource",
        "//tflite/internal:signature_def",
        "//tflite/profiling:root_profiler",
        "//tflite/profiling/telemetry:profiler",
        "//tflite/profiling/telemetry/c:telemetry_setting",
        "//tflite/profiling/telemetry/c:telemetry_setting_internal",
        "//tflite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

cc_library(
    name = "model_builder",
    hdrs = ["model_builder.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = internal_visibility_allowlist(),
    deps = [
        "//tflite:stderr_reporter",
        "//tflite/converter/core:model_builder_base",
        "//tflite/core/api:error_reporter",
    ],
    alwayslink = 1,
)

cc_library(
    name = "signature_runner",
    srcs = ["signature_runner.cc"],
    hdrs = ["signature_runner.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tflite:__pkg__",
        "//tflite/core:__subpackages__",
        "@org_tensorflow//third_party/odml/infra/genai/inference/executor/google_tensor:__subpackages__",
    ],
    deps = [
        "//tflite/c:common",
        "//tflite/core:subgraph",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/internal:signature_def",
    ],
)

# Test signature runner.
cc_test(
    name = "signature_runner_test",
    size = "small",
    srcs = ["signature_runner_test.cc"],
    data = [
        "//tflite:testdata/multi_signatures.bin",
        "//tflite:testdata/reverse_signature_model.bin",
    ],
    deps = [
        ":framework",
        ":signature_runner",
        "//tflite:model_builder",
        "//tflite/converter/core:model_builder_base",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Test model framework.
cc_test(
    name = "model_test",
    size = "small",
    srcs = ["model_test.cc"],
    data = [
        "//tflite:testdata/0_subgraphs.bin",
        "//tflite:testdata/2_subgraphs.bin",
        "//tflite:testdata/2_subgraphs_dont_delegate_name.bin",
        "//tflite:testdata/add_shared_tensors.bin",
        "//tflite:testdata/empty_model.bin",
        "//tflite:testdata/multi_add_flex.bin",
        "//tflite:testdata/segment_sum_invalid_buffer.bin",
        "//tflite:testdata/sparse_tensor.bin",
        "//tflite:testdata/test_min_runtime.bin",
        "//tflite:testdata/test_model.bin",
        "//tflite:testdata/test_model_broken.bin",
        "//tflite:testdata/test_model_redux_precision.bin",
        "//tflite:testdata/while_op_with_forwarding_input.bin",
        "//tflite:testdata/zero_size_constant.bin",
    ],
    tags = [
        "no_windows",  # TODO(b/194459105): the test is flaky.
        "noasan",
        "tflite_not_portable",
        "tflite_smoke_test",
    ],
    deps = [
        ":framework",
        "//tflite:framework",
        "//tflite:string_util",
        "//tflite/converter:allocation",
        "//tflite/core/api",
        "//tflite/core/api:verifier",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "create_op_resolver_header",
    hdrs = [
        "create_op_resolver.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        "//tflite:mutable_op_resolver",
        "//tflite:op_resolver",
    ],
)

# Defines CreateOpResolver with all builtin ops.
cc_library(
    name = "create_op_resolver_with_builtin_ops",
    srcs = ["create_op_resolver_with_builtin_ops.cc"],
    hdrs = ["create_op_resolver.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        "//tflite:mutable_op_resolver",
        "//tflite:op_resolver",
        "//tflite/core/kernels:builtin_ops",
    ],
    # Some targets only have an implicit dependency on CreateOpResolver.
    # This avoids warnings about backwards references when linking.
    alwayslink = True,
)

# This target is only for use by the "tflite_custom_c_library" and "tflite_custom_cc_library" build
# macro and should not be used anywhere other than in the implementation of that build macro.
# "tflite_custom_c_library" requires target to be public, that's why we duplicated
# :create_op_resolver_header target to be used only by "tflite_custom_c_library".
# Making :create_op_resolver_header public could cause some problems because it is widely used
# inside the TF Lite code base, that might lead others outside the TF Lite code base to copy that
# dependency and use it and subsequently depend on it, which would be bad. Using a separate
# :private_create_op_resolver_header target ensures that the only use of the unwantedly-"public"
# target is inside the "tflite_custom_c_library" itself, where it is less likely to get copied into
# third party code.
alias(
    name = "private_create_op_resolver_header",
    actual = ":create_op_resolver_header",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tflite:__subpackages__",
    ] + macros_visibility_allowlist(),
)

cc_library(
    name = "subgraph",
    srcs = [
        "subgraph.cc",
    ],
    hdrs = [
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//platforms/darwinn/tflite:__subpackages__",
        "//tflite:__subpackages__",
        "//tflite/core:__subpackages__",
        "//tflite/kernels:__subpackages__",
    ],
    deps = [
        "//tflite:allocation",
        "//tflite:array",
        "//tflite:graph_info",
        "//tflite:interpreter_options_header",
        "//tflite:kernel_api",
        "//tflite:macros",
        "//tflite:memory_planner",
        "//tflite:minimal_logging",
        "//tflite:util",
        "//tflite/c:common_internal",
        "//tflite/converter/experimental/remat:metadata_util",
        "//tflite/core/api",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/experimental/resource",
        "//tflite/profiling:root_profiler",
        "//tflite/profiling/telemetry",
        "//tflite/schema:schema_fbs",
    ] + select({
        "//tflite:tflite_use_simple_memory_planner": [
            "//tflite:simple_planner",
        ],
        "//conditions:default": [
            "//tflite:arena_planner",
        ],
    }) + select({
        "//tflite:tensorflow_profiler_config": [
            "//tflite:tensorflow_profiler_logger_shim",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

# Test subgraph.
cc_test(
    name = "subgraph_test",
    size = "small",
    srcs = [
        "subgraph_test.cc",
    ],
    deps = [
        ":framework_stable",
        "//tflite:framework",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/kernels:builtin_ops",  # build_cleaner: keep
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_building",
    srcs = ["model_building.cc"],
    hdrs = ["model_building.h"],
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        ":subgraph",
        "//tflite:array",
        "//tflite:framework",
        "//tflite:type_to_tflitetype",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "model_building_test",
    srcs = ["model_building_test.cc"],
    deps = [
        ":model_building",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "subgraph_composite_inlining_test",
    srcs = ["subgraph_composite_inlining_test.cc"],
    deps = [
        ":model_building",
        ":subgraph",
        "//tflite:array",
        "//tflite:builtin_ops",
        "//tflite:framework",
        "//tflite:util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:subgraph_test_util",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:tensor_ctypes",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")

tflite_portable_test_suite()
