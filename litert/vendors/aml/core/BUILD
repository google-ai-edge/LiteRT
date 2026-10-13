# Copyright (C) 2026 Amlogic, Inc. All rights reserved.

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
)

# ---------------------------------------------------------------------------
# Compatibility aliases.
# Compiler core: external @aml_npu//compiler_core
# Runtime log: this package (dispatch / nnsdk).
# ---------------------------------------------------------------------------
cc_library(
    name = "aml_adla_log",
    srcs = ["adla_log.cc"],
    hdrs = ["adla_log.h"],
    includes = ["."],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

alias(
    name = "Aml_common_lib",
    actual = "@aml_npu//compiler_core:aml_compiler_core_lib",
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

alias(
    name = "aml_compiler_core_lib",
    actual = "@aml_npu//compiler_core:aml_compiler_core_lib",
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

alias(
    name = "aml_compiler_core",
    actual = "@aml_npu//compiler_core:aml_compiler_core",
    visibility = ["//visibility:public"],
)

alias(
    name = "Aml_common",
    actual = "@aml_npu//compiler_core:aml_compiler_core_lib",
    visibility = ["//visibility:public"],
)

# ---------------------------------------------------------------------------
# Runtime-only pieces used by LiteRtDispatch_Aml (inference).
# Must NOT depend on libcompiler_adla.so.
# ---------------------------------------------------------------------------
cc_library(
    name = "aml_dispatch_info",
    srcs = ["aml_dispatch_info.cc"],
    hdrs = ["aml_dispatch_info.h"],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)

cc_library(
    name = "Aml_runtime_lib",
    srcs = [
        "nnsdk_func.cc",
    ],
    hdrs = [
        "nnsdk_func.h",
    ],
    deps = [
        ":aml_adla_log",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/options:litert_aml_options",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_aml_options",
        ":aml_dispatch_info",
        "@com_google_absl//absl/strings:string_view",
        "@aml_npu//runtime_lib:nnsdk_headers",
    ],
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
)
