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

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "viplite_adapter_api.cc",
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_invocation_context.cc",
    ],
    hdrs = [
        "vip_lite.h",
        "viplite_adapter_api.h",
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
        "-Wl,-soname=libLiteRtDispatch_Verisilicon.so",
        "-Wl,-lc++abi",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_Verisilicon.so",
    tags = [
        # Remove when sdk is available to bazel.
        "nobuilder",
        "notap",
    ],
    visibility = ["//litert:__subpackages__"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/cc/internal:litert_shared_library",
        "//litert/cc/options:litert_verisilicon_options",
        "//litert/cc/internal:litert_handle",
        "//litert/cc:litert_tensor_buffer_requirements",
        "//litert/cc:litert_tensor_buffer_types",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# litert_device_test(
#     name = "dispatch_api_verisilicon_test",
#     srcs = ["dispatch_api_verisilicon_test.cc"],
#     backend_id = "verisilicon",
#     data = [
#         "//litert/test:testdata/simple_model_vsi.nbg",
#         "//litert/test:testdata/simple_model_vsi_acuity.tflite",
#     ],
#     deps = [
#         ":dispatch_api",
#         "//litert/c:litert_runtime_c_api_shared_lib",
#         "//litert/cc:litert_any",
#         "//litert/cc:litert_environment_testonly",
#         "//litert/cc:litert_options_testonly",
#         "//litert/core:filesystem_testonly",
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model_npu",
#         "//litert/vendors/c:litert_dispatch_c_api",
#         "@com_google_absl//absl/log",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/types:span",
#         "@com_google_googletest//:gtest_main",
#     ],
# )
