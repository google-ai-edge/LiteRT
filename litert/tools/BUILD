# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# copybara:uncomment load("//devtools/deps/check:deps_check.bzl", "check_dependencies")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "gtest_main_no_heapcheck_deps", "litert_bin")
load("//litert/build_common:special_rule.bzl", "litert_gpu_accelerator_deps", "litert_gpu_accelerator_prebuilts")
load("//litert/integration_test:litert_device.bzl", "litert_device_exec", "litert_device_test")
load("//litert/integration_test:litert_device_common.bzl", "dispatch_device_rlocation", "plugin_device_rlocation")
load("//litert/tools:tool_test.bzl", "runfile_path", "tool_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

# This user library should depend on libLiteRt.so to allow
# CompilerPlugin to use the same libLiteRt.so
cc_library(
    name = "apply_plugin_runtimecapi",
    srcs = ["apply_plugin.cc"],
    hdrs = ["apply_plugin.h"],
    deps = [
        ":dump_runtimecapi",
        ":outstream",
        ":tool_display",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core/model:model_serialize",
        "//litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "apply_plugin_test",
    srcs = ["apply_plugin_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/vendors/examples:example_plugin_so",
    ],
    # Workaround to disable mismatched-size-delete.
    # This test uses`libLiteRt.so` with some system components.
    # So mixing of the two will cause mismatched-size-delete error.
    # copybara:uncomment malloc = "//base:system_malloc",
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    deps = [
        ":apply_plugin_runtimecapi",
        ":outstream",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/core:build_stamp",
        "//litert/core:dispatch_op_schema",
        "//litert/core/model",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ] + gtest_main_no_heapcheck_deps(),
)

litert_bin(
    name = "apply_plugin_main",
    srcs = ["apply_plugin_main.cc"],
    data = select({
        "@platforms//os:windows": [],
        "//conditions:default": [
            # copybara:uncomment_begin(google-only)
            # "//platforms/darwinn/compiler/external:litert_plugin_compiler",
            # copybara:uncomment_end
            "//litert/vendors/google_tensor/compiler:google_tensor_compiler_plugin_so",
            "//litert/vendors/mediatek/compiler:compiler_plugin_so",
            "//litert/vendors/examples:example_plugin_so",
            "//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so",
            "@qairt//:lib/x86_64-linux-clang/libQnnHtp.so",
            "@qairt//:lib/x86_64-linux-clang/libQnnSystem.so",
            "@neuro_pilot//:v9_latest/host/lib/libneuron_adapter.so",
            "@neuro_pilot//:v8_latest/host/lib/libneuron_adapter.so",
            "@neuro_pilot//:v7_latest/host/lib/libneuron_adapter.so",
        ],
    }),
    export_litert_only = 1,
    linkstatic = 1,
    # copybara:uncomment malloc = "//base:system_malloc",
    tags = [
        "noasan",
        "nomsan",
        "nosan",
    ],
    target_compatible_with = select({
        "@platforms//os:linux": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    visibility = ["//litert:litert_public"],
    deps = [
        ":apply_plugin_runtimecapi",
        ":outstream",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/core:build_stamp",
        "//litert/tools/flags:apply_plugin_flags",
        "//litert/tools/flags:common_flags",
        "//litert/tools/flags:flag_types",
        "//litert/tools/flags/vendors:intel_openvino_flags_with_dynamic_runtime",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@platforms//os:windows": [],
        "//conditions:default": [
            "//litert/tools/flags/vendors:google_tensor_flags_with_dynamic_runtime",
            "//litert/tools/flags/vendors:mediatek_flags_with_dynamic_runtime",
            "//litert/tools/flags/vendors:qualcomm_flags_with_dynamic_runtime",
        ],
    }),
)

litert_bin(
    name = "apply_plugin_main_for_google_tensor",
    srcs = ["apply_plugin_main.cc"],
    data = [
        # copybara:uncomment_begin(google-only)
        # "//platforms/darwinn/compiler/external:litert_plugin_compiler",
        # copybara:uncomment_end
        "//litert/vendors/google_tensor/compiler:google_tensor_compiler_plugin_so",
    ],
    # copybara:uncomment malloc = "//base:system_malloc",
    tags = [
        "noasan",
        "nomsan",
        "nosan",
    ],
    target_compatible_with = select({
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = [
        "//litert:litert_public",
        "//platforms/darwinn/tests/sdk_3p/litert_tools:__subpackages__",
    ],
    deps = [
        ":apply_plugin_runtimecapi",
        ":outstream",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/core:build_stamp",
        "//litert/tools/flags:apply_plugin_flags",
        "//litert/tools/flags:common_flags",
        "//litert/tools/flags:flag_types",
        "//litert/tools/flags/vendors:google_tensor_flags_with_dynamic_runtime",
        "//litert/tools/flags/vendors:intel_openvino_flags_with_dynamic_runtime",
        "//litert/tools/flags/vendors:mediatek_flags_with_dynamic_runtime",
        "//litert/tools/flags/vendors:qualcomm_flags_with_dynamic_runtime",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# Apply plugin test tests the core logic of the tool (with a dummy plugin impl), while the
# following tests the actual plugins will function correctly.
[
    tool_test(
        name = "apply_vendor_plugin_test_{}_{}".format(
            manufacturer.lower(),
            soc_model,
        ),
        data = ["//litert/test:testdata/simple_quantized_ops.tflite"],
        tool = ":apply_plugin_main",
        tool_args = [
            "--soc_model={}".format(soc_model),
            "--soc_manufacturer={}".format(manufacturer),
            "--model={}".format(
                runfile_path("//litert/test:testdata/simple_quantized_ops.tflite"),
            ),
            "--libs={}".format(lib_dir),
            "--cmd=apply",
            "--o=$$(mktemp)",
        ],
    )
    for soc_model, manufacturer, lib_dir in [
        [
            "SA8295",
            "Qualcomm",
            # copybara:uncomment_begin(google-only)
            # "third_party/odml/litert/litert/vendors/qualcomm/compiler",
            # copybara:uncomment_end_and_comment_begin
            "litert/vendors/qualcomm/compiler",
            # copybara:comment_end
        ],
        [
            "mt6991",
            "MediaTek",
            # copybara:uncomment_begin(google-only)
            # "third_party/odml/litert/litert/vendors/mediatek/compiler",
            # copybara:uncomment_end_and_comment_begin
            "litert/vendors/mediatek/compiler",
            # copybara:comment_end
        ],
    ]
]

test_suite(
    name = "apply_vendor_plugin_tests",
    tests = [
        ":apply_vendor_plugin_test_mediatek_mt6991",
        ":apply_vendor_plugin_test_qualcomm_SA8295",
    ],
)

# Fork of "apply_plugin_main" without the "ungrte" so this tool can be used as part of larger
# integration test pipelines with example_plugin.
cc_binary(
    name = "apply_plugin_main_for_test",
    testonly = 1,
    srcs = ["apply_plugin_main.cc"],
    data = ["//litert/vendors/examples:example_plugin_so"],
    linkstatic = 1,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
    ],
    deps = [
        ":apply_plugin_runtimecapi",
        ":outstream",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/tools/flags:apply_plugin_flags",
        "//litert/tools/flags:common_flags",
        "//litert/tools/flags:flag_types",
        "//litert/tools/flags/vendors:google_tensor_flags_with_dynamic_runtime",
        "//litert/tools/flags/vendors:intel_openvino_flags_with_dynamic_runtime",
        "//litert/tools/flags/vendors:mediatek_flags_with_dynamic_runtime",
        "//litert/tools/flags/vendors:qualcomm_flags_with_dynamic_runtime",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "tool_display",
    srcs = ["tool_display.cc"],
    hdrs = ["tool_display.h"],
    deps = [
        ":outstream",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "tool_display_test",
    srcs = ["tool_display_test.cc"],
    data = ["//litert/test:mlir_test_data"],
    deps = [
        ":tool_display",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    # TODO: Either refactor this code to be in cc/ or setup to only build in debug mode.
    # Then, we can remove this visibility.
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core/model",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "dump_runtimecapi",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    # TODO: Either refactor this code to be in cc/ or setup to only build in debug mode.
    # Then, we can remove this visibility.
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core/model",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "dump_test",
    srcs = ["dump_test.cc"],
    data = ["//litert/test:mlir_test_data"],
    deps = [
        ":dump",
        "//litert/c:litert_tensor_buffer",
        "//litert/core/model",
        "//litert/test:common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "outstream",
    hdrs = ["outstream.h"],
    deps = [
        "//litert/c/internal:litert_logging",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "benchmark_litert_model",
    srcs = ["benchmark_litert_model.cc"],
    hdrs = ["benchmark_litert_model.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/c/options:litert_mediatek_options",
        "//litert/c/options:litert_qualcomm_options",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc:litert_profiler",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc/internal:litert_tflite_error_status_builder",
        "//litert/cc/options:litert_cpu_options",
        "//litert/cc/options:litert_gpu_options",
        "//litert/cc/options:litert_mediatek_options",
        "//litert/cc/options:litert_qualcomm_options",
        "//litert/cc/options:litert_runtime_options",
        "//litert/core/util:perfetto_profiling",
        "//litert/runtime:compiled_model",
        "//tflite:framework_stable",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/profiling:model_runtime_info",
        "//tflite/profiling:profile_buffer",
        "//tflite/profiling:profile_summarizer",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:utils",
        "//tflite/tools/benchmark:benchmark_model_lib",
        "//tflite/tools/benchmark:benchmark_params",
        "//tflite/tools/benchmark/proto:benchmark_result_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

# copybara:uncomment_begin(google-only)
# # Ensure that benchmark tool is statically linked.
# check_dependencies(
#     of = [":benchmark_litert_model"],
#     dont_depend_on = [
#         "//litert/c:litert_runtime_c_api_shared_lib",
#     ],
# )
# copybara:uncomment_end

cc_test(
    name = "benchmark_litert_model_test",
    srcs = ["benchmark_litert_model_test.cc"],
    data = ["//litert/test:testdata/mobilenet_v2_1.0_224.tflite"],
    deps = [
        ":benchmark_litert_model",
        "//tflite/core/c:private_c_api_types",
        "//tflite/tools/benchmark:benchmark_model_lib",
        "//tflite/tools/benchmark:benchmark_params",
        "@com_google_googletest//:gtest_main",
    ],
)

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "benchmark_litert_model_gpu_test",
#     srcs = ["benchmark_litert_model_gpu_test.cc"],
#     data = ["//litert/test:testdata/mobilenet_v2_1.0_224.tflite"],
#     tags = ["requires-gpu-nvidia"],
#     deps = [
#         ":benchmark_litert_model",
#         "@com_google_googletest//:gtest_main",
#         "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",  # buildcleaner: keep
#         "//tflite/core/c:private_c_api_types",
#         "//tflite/tools/benchmark:benchmark_model_lib",
#         "//tflite/tools/benchmark:benchmark_params",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.cc"],
    hdrs = ["tensor_utils.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "run_model",
    srcs = ["run_model.cc"],
    copts = [
        "-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
    ],
    data = litert_gpu_accelerator_prebuilts(),
    deps = [
        ":tensor_utils",
        "//litert/c:litert_common",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/tools/flags/vendors:google_tensor_flags",
        "//litert/tools/flags/vendors:intel_openvino_flags",
        "//litert/tools/flags/vendors:mediatek_flags",
        "//litert/tools/flags/vendors:qualcomm_flags",
        "//tflite/profiling:time",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ] + litert_gpu_accelerator_deps(),
)

[
    litert_device_exec(
        name = "run_model_device_{}".format(backend),
        backend_id = backend,
        exec_args = [
            "--dispatch_library_dir={}".format(dispatch_device_rlocation(backend)),
            "--compiler_plugin_library_dir={}".format(plugin_device_rlocation(backend)),
            "--accelerator=npu",
        ],
        local_suffix = "",
        remote_suffix = None,
        target = ":run_model",
    )
    for backend in [
        "qualcomm",
        "qualcomm_v79",
        "mediatek",
        "cpu",
        "example",
    ]
]

# Simpler (single source file) version of run_model that only requires the C++ API.
cc_binary(
    name = "run_model_simple",
    srcs = ["run_model_simple.cc"],
    copts = [
        "-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

NUMERICS_CHECK_DEPS = [
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/flags:parse",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/strings:str_format",
    "@com_google_absl//absl/strings:string_view",
    "@com_google_absl//absl/types:span",
    "//litert/c:litert_common",
    "//litert/cc:litert_compiled_model",
    "//litert/cc:litert_element_type",
    "//litert/cc:litert_environment",
    "//litert/cc:litert_expected",
    "//litert/cc:litert_macros",
    "//litert/cc:litert_model",
    "//litert/cc:litert_options",
    "//litert/cc:litert_tensor_buffer",
    "//tflite/profiling:time",
]

cc_binary(
    name = "npu_numerics_check",
    srcs = ["npu_numerics_check.cc"],
    deps = NUMERICS_CHECK_DEPS + [
        "//litert/cc:litert_common",
        "//litert/tools:tensor_utils",
        "//litert/tools/flags/vendors:google_tensor_flags",
        "//litert/tools/flags/vendors:mediatek_flags",
        "//litert/tools/flags/vendors:qualcomm_flags",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            # copybara:uncomment_begin(google-only)
            # "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",  # buildcleaner: keep
            # copybara:uncomment_end
        ],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "gpu_numerics_check",
    srcs = ["gpu_numerics_check.cc"],
    data = litert_gpu_accelerator_prebuilts(),
    deps = NUMERICS_CHECK_DEPS + [
        "//litert/cc:litert_common",
        "//litert/cc/options:litert_gpu_options",
        "//litert/core:filesystem",
        "//tflite/c:c_api_types",
        "//tflite/tools:utils",
    ] + litert_gpu_accelerator_deps(),
)

cc_binary(
    name = "gpu_numerics_check_cl_gl",
    srcs = ["gpu_numerics_check.cc"],
    deps = NUMERICS_CHECK_DEPS + [
        "//litert/cc/options:litert_gpu_options",
        "//litert/cc:litert_common",
        "//litert/core:filesystem",
        # copybara:uncomment_begin(google-only)
        # "//litert/runtime/accelerators/gpu:ml_drift_cl_gl_accelerator",  # buildcleaner: keep
        # "//tflite/c:c_api_types",
        # "//tflite/tools:utils",
        # copybara:uncomment_end
    ],
)

cc_binary(
    name = "gpu_numerics_check_vulkan",
    srcs = ["gpu_numerics_check.cc"],
    tags = ["requires-gpu-nvidia"],
    deps = NUMERICS_CHECK_DEPS + [
        "//litert/cc/options:litert_gpu_options",
        "//litert/cc:litert_common",
        "//litert/core:filesystem",
        # copybara:uncomment_begin(google-only)
        # "//litert/runtime/accelerators/gpu:ml_drift_vulkan_accelerator",  # buildcleaner: keep
        # "//tflite/c:c_api_types",
        # "//tflite/tools:utils",
        # "//third_party/vulkan_loader",
        # copybara:uncomment_end
    ],
)

cc_binary(
    name = "gpu_numerics_check_jet",
    srcs = ["gpu_numerics_check.cc"],
    tags = [
        "manual",
        "nofastbuild",
        "noios",
        "nolinux",
        "nomacos",
        "nowindows",
        "requires-gpu-nvidia",
    ],
    deps = NUMERICS_CHECK_DEPS + [
        "//litert/cc/options:litert_gpu_options",
        "//litert/cc:litert_common",
        "//litert/core:filesystem",
        # copybara:uncomment_begin(google-only)
        # "//litert/runtime/accelerators/gpu/google:jet_gpu_accelerator",  # buildcleaner: keep
        # "//tflite/c:c_api_types",
        # "//tflite/tools:utils",
        # copybara:uncomment_end
    ],
)

cc_binary(
    name = "analyze_model_main",
    srcs = ["analyze_model_main.cc"],
    visibility = [
        "//litert:litert_internal_users",
        "//platforms/darwinn/tests/sdk_3p/litert_tools:__pkg__",
    ],
    deps = [
        ":dump",
        ":tool_display",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_logging",
        "//litert/core/model",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# We create a library for benchmark_main.cc to faciliate the creation of a
# customized benchmark model binary that only needs linking with extra
# dependency, e.g., enabling creating of benchmark binaries with a custom
# delegate provider.
cc_library(
    name = "benchmark_model_main",
    srcs = ["benchmark_litert_model_main.cc"],
    deps = [
        ":benchmark_litert_model",
        "//tflite/c:c_api_types",
        "//tflite/tools:logging",
    ],
)

litert_bin(
    name = "benchmark_model",
    data = litert_gpu_accelerator_prebuilts(),
    export_litert_only = True,
    ungrte = True,
    deps = [":benchmark_model_main"] + litert_gpu_accelerator_deps(),
)

cc_binary(
    name = "benchmark_model_allgpu",
    deps = [
        ":benchmark_model_main",
        # copybara:uncomment_begin(google-only)
        # "//litert/runtime/accelerators/gpu:ml_drift_allgpu_accelerator",  # buildcleaner: keep
        # copybara:uncomment_end
    ],
)

cc_binary(
    name = "benchmark_model_vulkan",
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":benchmark_model_main",
        # copybara:uncomment_begin(google-only)
        # "//litert/runtime/accelerators/gpu:ml_drift_vulkan_accelerator",  # buildcleaner: keep
        # "//third_party/vulkan_loader",
        # copybara:uncomment_end
    ],
)

cc_binary(
    name = "benchmark_model_jet",
    tags = [
        "manual",
        "nofastbuild",
        "noios",
        "nolinux",
        "nomacos",
        "nowindows",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":benchmark_model_main",
        # copybara:uncomment_begin(google-only)
        # "//litert/runtime/accelerators/gpu/google:jet_gpu_accelerator",  # buildcleaner: keep
        # copybara:uncomment_end
    ],
)

cc_library(
    name = "benchmark_stripped_litert_model",
    srcs = ["benchmark_stripped_litert_model.cc"],
    hdrs = ["benchmark_stripped_litert_model.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":benchmark_litert_model",
        "@flatbuffers//:runtime_cc",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//tflite:model_builder",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/tools:logging",
        "//tflite/tools/benchmark:benchmark_params",
        "//tflite/tools/strip_buffers:stripping_lib",
        # copybara:uncomment_begin(google-only)
        # "//litert/runtime/accelerators/gpu:ml_drift_allgpu_accelerator",  # buildcleaner: keep
        # copybara:uncomment_end
    ],
)

litert_device_test(
    name = "benchmark_litert_model_qualcomm_test",
    srcs = ["benchmark_litert_model_qualcomm_test.cc"],
    backend_id = "qualcomm",
    data = [
        "//litert/test:testdata/simple_add_op.tflite",
        "//litert/test:testdata/simple_add_op_qc_v75_precompiled.tflite",
    ],
    deps = [
        ":benchmark_litert_model",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/test:common",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "extract_bytecode",
    srcs = ["extract_bytecode.cc"],
    deps = [
        ":tool_display",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_extended_model",
        "//litert/core:dispatch_op_schema",
        "//litert/core/model",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "metadata_util_main",
    srcs = ["metadata_util_main.cc"],
    deps = [
        ":tool_display",
        "//litert/c:litert_model_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_extended_model",
        "//litert/core/model",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
