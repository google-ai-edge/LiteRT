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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:special_rule.bzl", "gles_linkopts", "litert_metal_opts")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

# copybara:uncomment_begin(google-only)
# config_setting(
#     name = "windows_msvc",
#     constraint_values = [
#         "@platforms//os:windows",
#     ],
#     flag_values = {"//tools/cpp:compiler": "msvc"},
# )
# copybara:uncomment_end

# copybara:comment_begin(oss-only)
config_setting(
    name = "windows_msvc",
    values = {
        "compiler": "msvc",
        "cpu": "x64_windows",
    },
)
# copybara:comment_end

# copybara:uncomment_begin(google-only)
# config_setting(
#     name = "windows_lexan",
#     constraint_values = [
#         "@platforms//os:windows",
#     ],
#     flag_values = {"//tools/cpp:compiler": "lexan"},
# )
# copybara:uncomment_end

# copybara:comment_begin(oss-only)
config_setting(
    name = "windows_lexan",
    values = {
        "compiler": "lexan",
        "cpu": "x64_windows",
    },
)
# copybara:comment_end

# Dispatch API implementation, it is used by the dispatch delegate to call the vendor's dispatch
# API.
cc_library(
    name = "dispatch",
    srcs = select({
        "//conditions:default": ["litert_dispatch.cc"],
    }),
    hdrs = [
        "//litert/vendors/c:litert_dispatch.h",
        "//litert/vendors/c:litert_dispatch_api.h",
    ],
    copts = select({
        ":windows_msvc": [
            "/DLITERT_COMPILE_LIBRARY",
            "/wd4018",  # -Wno-sign-compare
        ],
        ":windows_lexan": [
            "-DLITERT_COMPILE_LIBRARY",
            "-Wno-sign-compare",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_header",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_metrics",
        "//litert/c:litert_model_types",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_shared_library",
        "//litert/core:dynamic_loading",
        "//litert/core:version",
        "//litert/core/util:perfetto_profiling",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dispatch_opaque_options",
    srcs = [
        "dispatch_opaque_options.cc",
    ],
    hdrs = [
        "dispatch_opaque_options.h",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
    ],
)

cc_test(
    name = "dispatch_opaque_options_test",
    srcs = [
        "dispatch_opaque_options_test.cc",
    ],
    deps = [
        ":dispatch_opaque_options",
        "//litert/cc:litert_opaque_options",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dispatch_delegate_test",
    srcs = ["dispatch_delegate_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/vendors/examples:example_dispatch_so",
        "//litert/vendors/examples:example_plugin_so",
    ],
    deps = [
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dispatch_delegate",
    srcs = [
        "dispatch_delegate.cc",
        "dispatch_delegate_kernel.cc",
    ],
    hdrs = [
        "dispatch_delegate_kernel.h",
        "//litert/c/internal:litert_dispatch_delegate.h",
        "//litert/cc/internal:litert_dispatch_delegate.h",
    ],
    copts = litert_metal_opts(),
    deps = [
        ":dispatch",
        ":dispatch_opaque_options",
        "//litert/c:litert_common",
        "//litert/c:litert_metrics",
        "//litert/c/internal:litert_dispatch_headers",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_tflite_error_status_builder",
        "//litert/core:build_stamp",
        "//litert/core:dispatch_op_schema",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/runtime:metrics",
        "//litert/runtime:tensor_buffer",
        "//litert/runtime:tfl_utils",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:private_c_api_opaque_without_op_resolver",
        "//tflite/delegates/utils:simple_opaque_delegate",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

litert_device_test(
    name = "dispatch_delegate_google_tensor_test",
    srcs = ["dispatch_delegate_google_tensor_test.cc"],
    backend_id = "google_tensor",
    data = [
        "//litert/test:testdata/shared_input_cpu_npu_google_tensor_precompiled.tflite",
        "//litert/test:testdata/simple_model_npu_google_tensor_precompiled.tflite",
    ],
    features = select({
        "//tools/cc_target_os:android": ["-layering_check"],
        "//conditions:default": [],
    }),
    linkopts = gles_linkopts(),
    # TODO: gt team update visibility.
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        ":dispatch_delegate",
        ":dispatch_opaque_options",
        # TODO: gt team update visibility.
        # "//platforms/darwinn/tachyon/core/fence",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/darwinn/driver_shared/fence:fence_test_util",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc/internal:litert_handle",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc/internal:litert_compiled_model_next",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_event",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/runtime:tensor_buffer",
        "//litert/runtime:tensor_identifier",
        "//litert/runtime:tfl_utils",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
    ],
)

litert_device_test(
    name = "dispatch_delegate_cascade_google_tensor_test",
    srcs = ["dispatch_delegate_cascade_google_tensor_test.cc"],
    backend_id = "google_tensor",
    data = [
        "//litert/test:testdata/simple_cascade_model_npu.tflite",
        "//litert/test:testdata/simple_model_google_tensor.bin",
    ],
    features = select({
        "//tools/cc_target_os:android": ["-layering_check"],
        "//conditions:default": [],
    }),
    linkopts = gles_linkopts(),
    deps = [
        ":dispatch_delegate",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/darwinn/driver_shared/fence:fence_test_util",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_event",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_event",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/core/model:model_buffer",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_cascade_model_npu",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
    ],
)

litert_device_test(
    name = "dispatch_delegate_mixed_cascade_google_tensor_test",
    srcs = ["dispatch_delegate_mixed_cascade_google_tensor_test.cc"],
    backend_id = "google_tensor",
    data = [
        "//litert/test:testdata/simple_mixed_cascade_model_npu.tflite",
        "//litert/test:testdata/simple_model_google_tensor.bin",
    ],
    features = select({
        "//tools/cc_target_os:android": ["-layering_check"],
        "//conditions:default": [],
    }),
    linkopts = gles_linkopts(),
    deps = [
        ":dispatch_delegate",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/darwinn/driver_shared/fence:fence_test_util",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_event",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_event",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/core/model:model_buffer",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_mixed_cascade_model_npu",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
    ],
)

litert_device_test(
    name = "dispatch_delegate_qualcomm_test",
    srcs = ["dispatch_delegate_qualcomm_test.cc"],
    backend_id = "qualcomm",
    data = [
        "//litert/test:testdata/shared_input_cpu_npu.tflite",
        "//litert/test:testdata/simple_model_npu.tflite",
        "//litert/test:testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        ":dispatch_opaque_options",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc/internal:litert_dispatch_delegate",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/runtime:tensor_buffer",
        "//litert/runtime:tensor_identifier",
        "//litert/runtime:tfl_utils",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_device_test(
    name = "dispatch_delegate_cascade_qualcomm_test",
    srcs = ["dispatch_delegate_cascade_qualcomm_test.cc"],
    backend_id = "qualcomm",
    data = [
        "//litert/test:testdata/simple_cascade_model_npu.tflite",
        "//litert/test:testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        ":dispatch_delegate",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc/internal:litert_dispatch_delegate",
        "//litert/core/model:model_buffer",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_cascade_model_npu",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_device_test(
    name = "dispatch_delegate_mixed_cascade_qualcomm_test",
    srcs = ["dispatch_delegate_mixed_cascade_qualcomm_test.cc"],
    backend_id = "qualcomm",
    data = [
        "//litert/test:testdata/simple_mixed_cascade_model_npu.tflite",
        "//litert/test:testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        ":dispatch_delegate",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc/internal:litert_dispatch_delegate",
        "//litert/core/model:model_buffer",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_mixed_cascade_model_npu",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_device_test(
    name = "dispatch_delegate_mediatek_test",
    srcs = ["dispatch_delegate_mediatek_test.cc"],
    backend_id = "mediatek",
    data = [
        "//litert/test:testdata/shared_input_cpu_npu.tflite",
        "//litert/test:testdata/simple_model_mtk.bin",
        "//litert/test:testdata/simple_model_npu.tflite",
    ],
    deps = [
        ":dispatch_delegate",
        ":dispatch_opaque_options",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/runtime:tensor_buffer",
        "//litert/runtime:tensor_identifier",
        "//litert/runtime:tfl_utils",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_device_test(
    name = "dispatch_delegate_cascade_mediatek_test",
    srcs = ["dispatch_delegate_cascade_mediatek_test.cc"],
    backend_id = "mediatek",
    data = [
        "//litert/test:testdata/simple_cascade_model_npu.tflite",
        "//litert/test:testdata/simple_model_mtk.bin",
    ],
    deps = [
        ":dispatch_delegate",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/core/model:model_buffer",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_cascade_model_npu",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_device_test(
    name = "dispatch_delegate_mixed_cascade_mediatek_test",
    srcs = ["dispatch_delegate_mixed_cascade_mediatek_test.cc"],
    backend_id = "mediatek",
    data = [
        "//litert/test:testdata/simple_mixed_cascade_model_npu.tflite",
        "//litert/test:testdata/simple_model_mtk.bin",
    ],
    deps = [
        ":dispatch_delegate",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/core/model:model_buffer",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_mixed_cascade_model_npu",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
