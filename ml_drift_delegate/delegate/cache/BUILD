load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "simple_cache",
    srcs = ["simple_cache.cc"],
    hdrs = ["simple_cache.h"],
    deps = [
        "//ml_drift_delegate/delegate/serialization_weight_cache:file_util",
        "//ml_drift_delegate/delegate/serialization_weight_cache:mmap_handle",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_test(
    name = "simple_cache_test",
    srcs = ["simple_cache_test.cc"],
    deps = [
        ":simple_cache",
        "//ml_drift_delegate/delegate/serialization_weight_cache:file_util",
        "//ml_drift_delegate/delegate/serialization_weight_cache:mmap_handle",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "webgpu_pipeline_cache",
    srcs = ["webgpu_pipeline_cache.cc"],
    hdrs = ["webgpu_pipeline_cache.h"],
    deps = [
        ":simple_cache",
        ":webgpu_pipeline_cache_cc_fbs",
        "//ml_drift_delegate/delegate/serialization_weight_cache:mmap_handle",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
    ],
)

flatbuffer_cc_library(
    name = "webgpu_pipeline_cache_cc_fbs",
    srcs = ["webgpu_pipeline_cache.fbs"],
    flatc_args = [
        "--scoped-enums",
        "-I .",
    ],
)

cc_test(
    name = "webgpu_pipeline_cache_test",
    srcs = ["webgpu_pipeline_cache_test.cc"],
    deps = [
        ":simple_cache",
        ":webgpu_pipeline_cache",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
