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

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//litert/build_common:litert_build_defs.bzl", "litert_dispatch_api")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

litert_dispatch_api(
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_invocation_context.cc",
    ],
    hdrs = [
        "litert_dispatch_device_context.h",
        "litert_dispatch_invocation_context.h",
    ],
    copts = [
        "-Os",
        "-fno-exceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    export_litert_only = True,
    linkopts = litert_android_linkopts() + [
        "-Wl,-soname=libLiteRtDispatch_MediaTek.so",
        "-Wl,-lc++abi",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_MediaTek.so",
    tags = [
        # Remove when sdk is available to bazel.
        "nobuilder",
        "notap",
    ],
    visibility = [
        "//litert:__subpackages__",
        "//litert:litert_friends",
    ],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_header",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_event_header",
        "//litert/c:litert_event_type",
        "//litert/c:litert_model_header",
        "//litert/c:litert_model_types",
        "//litert/c:litert_opaque_options_header",
        "//litert/c:litert_options_header",
        "//litert/c:litert_tensor_buffer_header",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_logging_helper_with_runtime_context",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_context_wrapper",
        "//litert/cc/internal:litert_opaque_options_wrapper",
        "//litert/cc/internal:litert_options_wrapper",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/mediatek:neuron_adapter_api",
        "//litert/vendors/mediatek/schema:mediatek_litert_schema",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@neuro_pilot//:v8_latest_host_headers",
    ],
)

sh_test(
    name = "mediatek_dispatch_so_symbol_test",
    srcs = ["//litert/test:litert_c_api_dependency_test.sh"],
    args = [
        "LiteRtDispatchGetApi",
        "$(location :dispatch_api_so)",
    ],
    data = [
        ":dispatch_api_so",
    ],
)

litert_device_test(
    name = "dispatch_api_mediatek_test",
    srcs = ["dispatch_api_mediatek_test.cc"],
    backend_id = "mediatek",
    data = [
        "//litert/test:testdata/simple_model_mtk.bin",
        "//litert/test:testdata/simple_model_npu.tflite",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_runtime_context_testonly",
        "//litert/cc:litert_any",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_environment_testonly",
        "//litert/cc:litert_options_testonly",
        "//litert/core:filesystem_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
