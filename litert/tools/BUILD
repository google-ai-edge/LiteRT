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

load("//litert/build_common:litert_build_defs.bzl", "litert_bin")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")
load("//litert/tools:tool_test.bzl", "runfile_path", "tool_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "apply_plugin",
    srcs = ["apply_plugin.cc"],
    hdrs = ["apply_plugin.h"],
    deps = [
        ":dump",
        ":outstream",
        ":tool_display",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_environment",
        "//litert/cc/internal:litert_options",
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
    tags = [
        "noasan",
        "nomsan",
        "nosan",
        "notsan",
    ],
    deps = [
        ":apply_plugin",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_model",
        "//litert/core:build_stamp",
        "//litert/core:dispatch_op_schema",
        "//litert/core/model",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_bin(
    name = "apply_plugin_main",
    srcs = ["apply_plugin_main.cc"],
    data = [
        # copybara:uncomment_begin(google-only)
        # "//platforms/darwinn/compiler:compiler_api_wrapper",
        # "//litert/vendors/google_tensor/compiler:google_tensor_compiler_plugin_so",
        # "//litert/vendors/mediatek/compiler:compiler_plugin_so",
        # copybara:uncomment_end
        "//litert/vendors/examples:example_plugin_so",
        "//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so",
        "@qairt//:lib/x86_64-linux-clang/libQnnHtp.so",
        "@qairt//:lib/x86_64-linux-clang/libQnnSystem.so",
        # copybara:uncomment_begin(no-oss-yet)
        # "@neuro_pilot//:v8_latest/host/lib/libneuron_adapter.so",
        # "@neuro_pilot//:v7_latest/host/lib/libneuron_adapter.so",
        # copybara:uncomment_end
    ],
    export_litert_only = 1,
    linkstatic = 1,
    # copybara:uncomment malloc = "//base:system_malloc",
    tags = [
        "noasan",
        "nomsan",
        "nosan",
    ],
    target_compatible_with = select({
        "@org_tensorflow//tensorflow:android": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    ungrte = True,
    deps = [
        ":apply_plugin",
        ":outstream",
        "//litert/cc/internal:litert_options",
        "//litert/core:build_stamp",
        "//litert/tools/flags:apply_plugin_flags",
        "//litert/tools/flags:common_flags",
        "//litert/tools/flags:flag_types",
        "//litert/tools/flags/vendors:google_tensor_flags",
        "//litert/tools/flags/vendors:mediatek_flags",
        "//litert/tools/flags/vendors:qualcomm_flags",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
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
        data = [
            "//litert/test:testdata/simple_quantized_ops.tflite",
        ],
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
        # copybara:uncomment ":apply_vendor_plugin_test_mediatek_mt6991",
        ":apply_vendor_plugin_test_qualcomm_SA8295",
    ],
)

# Fork of "apply_plugin_main" without the "ungrte" so this tool can be used as part of larger
# integration test pipelines with example_plugin.
cc_binary(
    name = "apply_plugin_main_for_test",
    testonly = 1,
    srcs = ["apply_plugin_main.cc"],
    data = [
        "//litert/vendors/examples:example_plugin_so",
    ],
    linkstatic = 1,
    tags = [
        "noasan",
        "nomsan",
        "nosan",
    ],
    deps = [
        ":apply_plugin",
        ":outstream",
        "//litert/cc/internal:litert_options",
        "//litert/tools/flags:apply_plugin_flags",
        "//litert/tools/flags:common_flags",
        "//litert/tools/flags:flag_types",
        "//litert/tools/flags/vendors:google_tensor_flags",
        "//litert/tools/flags/vendors:mediatek_flags",
        "//litert/tools/flags/vendors:qualcomm_flags",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format",
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

cc_test(
    name = "dump_test",
    srcs = ["dump_test.cc"],
    data = ["//litert/test:mlir_test_data"],
    deps = [
        ":dump",
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
        "//litert/c:litert_logging",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "benchmark_litert_model",
    srcs = ["benchmark_litert_model.cc"],
    hdrs = ["benchmark_litert_model.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_compiled_model",
        "//litert/cc/internal:litert_environment",
        "//litert/cc/internal:litert_options",
        "//litert/cc/internal:litert_tensor_buffer",
        "//litert/cc/options:accelerator_options",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:utils",
        "//tflite/tools/benchmark:benchmark_model_lib",
        "//tflite/tools/benchmark:benchmark_params",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

#copybara:comment_begin(google-only)
cc_test(
    name = "benchmark_litert_model_test",
    srcs = ["benchmark_litert_model_test.cc"],
    data = [
        "//litert/test:testdata/mobilenet_v2_1.0_224.tflite",
    ],
    tags = [
        "requires-gpu-nvidia",
    ],
    deps =
        [
            ":benchmark_litert_model",
            "@com_google_googletest//:gtest_main",
            # copybara:uncomment_begin(google-only)
            # "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",  # buildcleaner: keep
            # copybara:uncomment_end
            "//tflite/core/c:private_c_api_types",
            "//tflite/tools/benchmark:benchmark_model_lib",
            "//tflite/tools/benchmark:benchmark_params",
        ],
)
#copybara:comment_end

cc_library(
    name = "tensor_utils",
    hdrs = ["tensor_utils.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_compiled_model",
        "//litert/cc/internal:litert_environment",
        "//litert/cc/internal:litert_tensor_buffer",
        "//tflite/profiling:time",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "run_model",
    srcs = ["run_model.cc"],
    deps = [
        ":tensor_utils",
        "//litert/c:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_compiled_model",
        "//litert/cc/internal:litert_environment",
        "//litert/cc/internal:litert_options",
        "//litert/cc/internal:litert_tensor_buffer",
        "//litert/tools/flags/vendors:google_tensor_flags",
        "//litert/tools/flags/vendors:mediatek_flags",
        "//litert/tools/flags/vendors:qualcomm_flags",
        "//tflite/profiling:time",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
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
    name = "analyze_model_main",
    srcs = ["analyze_model_main.cc"],
    deps = [
        ":dump",
        ":tool_display",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_logging",
        "//litert/cc:litert_model",
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
    srcs = [
        "benchmark_litert_model_main.cc",
    ],
    deps = [
        ":benchmark_litert_model",
        "//tflite/c:c_api_types",
        "//tflite/tools:logging",
    ],
)

cc_binary(
    name = "benchmark_model",
    deps = [
        ":benchmark_model_main",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            # copybara:uncomment_begin(google-only)
            # "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",  # buildcleaner: keep
            # copybara:uncomment_end
        ],
        "//conditions:default": [],
    }),
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
        # "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",  # buildcleaner: keep
        # copybara:uncomment_end
    ],
)

litert_device_test(
    name = "benchmark_litert_model_qualcomm_test",
    srcs = ["benchmark_litert_model_qualcomm_test.cc"],
    backend_id = "qualcomm",
    data = [
        "//litert/test:testdata/simple_add_op_qc_v75_precompiled.tflite",
    ],
    deps = [
        ":benchmark_litert_model",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_compiled_model",
        "//litert/cc/internal:litert_environment",
        "//litert/cc/internal:litert_options",
        "//litert/cc/internal:litert_tensor_buffer",
        "//litert/test:common",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
