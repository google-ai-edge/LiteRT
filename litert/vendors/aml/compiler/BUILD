# Copyright (C) 2026 Amlogic, Inc. All rights reserved.

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "make_rpaths")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
)

# LiteRT-only frontend: map LiteRT graph -> Context API, AmlContextCompile, pack DispatchInfo.
# Links external @aml_npu//compiler_core; does NOT link libcompiler_adla.so.
litert_dynamic_lib(
    name = "aml_compiler_plugin",
    srcs = [
        "aml_compiler_plugin.cc",
        "node_binary_info.cc",
    ],
    hdrs = [
        "//litert/vendors/c:litert_compiler_plugin.h",
        "node_binary_info.h",
    ],
    # Keep RTTI on for NodeBuilder::SetParameter dynamic_cast across
    # libLiteRtCompilerPlugin_Aml.so <-> libcompiler_adla.so.
    copts = [
        "-frtti",
    ],
    # Do NOT use export_litert_only: that hides adla typeinfo and breaks
    # cross-DSO dynamic_cast. Pass the curated version script as linkopts
    # (cc_library accepts these; litert_dynamic_lib has no version_script attr).
    export_litert_only = False,
    additional_linker_inputs = [":export_aml_compiler_plugin_linux.lds"],
    linkopts = litert_android_linkopts() + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            make_rpaths([
                "external/aml_npu/compiler_core/libs/linux/x86_64",
                "external/aml_npu/runtime_lib/libs/linux/x86_64",
            ]),
        ],
        "//conditions:default": [],
    }) + [
        "-Wl,--version-script=$(location :export_aml_compiler_plugin_linux.lds)",
        "-Wl,-soname=libLiteRtCompilerPlugin_Aml.so",
        # Wheel package places the shared library in the root directory.
        "-Wl,-rpath,$$ORIGIN",
        "-ldl",
    ],
    shared_lib_name = "aml_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Aml.so",
    ungrte = True,
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
    deps = [
        "//litert/vendors/aml/core:aml_dispatch_info",
        "@aml_npu//compiler_core:aml_compiler_core_lib",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_logging_helper_with_compiler_context",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c/options:litert_aml_options",
        "//litert/c:litert_op_options",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_types",
        "//litert/compiler/cc:litert_model",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_op_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)
