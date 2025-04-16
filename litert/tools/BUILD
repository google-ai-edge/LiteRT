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

load("//litert/integration_test:litert_device.bzl", "litert_device_test")
load("//litert/vendors/qualcomm:qualcomm_build_defs.bzl", "litert_cc_bin_with_qnn")

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
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/compiler/plugin:compiler_flags",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core/model:model_serialize",
        "//litert/core/util:flatbuffer_tools",
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
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
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
    ],
)

# TODO - @lukeboyer: Figure out some selective inclusiion of the data deps, some are very large.
litert_cc_bin_with_qnn(
    name = "apply_plugin_main",
    srcs = ["apply_plugin_main.cc"],
    data = [
        # copybara:uncomment_begin(google-only)
        # "//platforms/darwinn/compiler:compiler_api_wrapper",
        # copybara:uncomment_end
        "//litert/vendors/examples:example_plugin_so",
        # TODO: b/407046837 - Remove the comment once we are able to add the deps in OSS.
        # copybara:uncomment_begin(google-only)
        # "//litert/vendors/google_tensor/compiler:google_tensor_compiler_plugin_so",
        # "//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so",
        # "//litert/vendors/mediatek/compiler:compiler_plugin_so",
        # "//third_party/neuro_pilot:latest/host/lib/libneuron_adapter.so",
        # copybara:uncomment_end
    ],
    export_litert_only = 1,
    include_system = 1,
    linkstatic = 1,
    # copybara:uncomment malloc = "//base:system_malloc",
    tags = [
        "noasan",
        "nobuilder",
        "nomsan",
        "nosan",
    ],
    ungrte = True,
    deps = [
        ":apply_plugin",
        ":outstream",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "//litert/compiler/plugin:compiler_flags",
        "//litert/core:build_stamp",
        "//litert/tools/flags:apply_plugin_flags",
        "//litert/tools/flags:common_flags",
        "//litert/tools/flags:flag_types",
        "//litert/tools/flags/vendors:qualcomm_flags",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format",
        "//litert/compiler/plugin:compiler_flags",
        "//litert/tools/flags:apply_plugin_flags",
        "//litert/tools/flags:common_flags",
        "//litert/tools/flags:flag_types",
        "//litert/tools/flags/vendors:qualcomm_flags",
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
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    # TODO: Either refactor this code to be in cc/ or setup to only build in debug mode.
    # Then, we can remove this visibility.
    visibility = ["//litert:litert_stable_abi_users"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core/model",
    ],
)

cc_test(
    name = "dump_test",
    srcs = ["dump_test.cc"],
    data = ["//litert/test:mlir_test_data"],
    deps = [
        ":dump",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//litert/core/model",
        "//litert/test:common",
    ],
)

cc_library(
    name = "outstream",
    hdrs = ["outstream.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_logging",
    ],
)

cc_library(
    name = "benchmark_litert_model",
    srcs = ["benchmark_litert_model.cc"],
    hdrs = ["benchmark_litert_model.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:utils",
        "//tflite/tools/benchmark:benchmark_model_lib",
        "//tflite/tools/benchmark:benchmark_params",
    ],
)

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
            # "//third_party/odml/infra/ml_drift_delegate/litert:ml_drift_cl_accelerator",  # buildcleaner: keep
            # copybara:uncomment_end
            "//tflite/core/c:private_c_api_types",
            "//tflite/tools/benchmark:benchmark_model_lib",
            "//tflite/tools/benchmark:benchmark_params",
        ],
)

cc_binary(
    name = "run_model",
    srcs = ["run_model.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment_begin(google-only)
        # "//third_party/odml/infra/ml_drift_delegate/litert:ml_drift_cl_accelerator",  # buildcleaner: keep
        # copybara:uncomment_end
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//tflite/profiling:time",
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
        # "//third_party/odml/infra/ml_drift_delegate/litert:ml_drift_cl_accelerator",  # buildcleaner: keep
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
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:common",
    ],
)
