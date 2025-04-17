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

load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

exports_files(
    srcs = ["run_test_on_android.sh"],
)

filegroup(
    name = "litert_scripts",
    srcs = glob(["*.sh"]),
)

cc_test(
    name = "invoke_qualcomm_test",
    srcs = ["invoke_qualcomm_test.cc"],
    copts = ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    data = [
        "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    deps = [
        ":invoke_qualcomm_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/core:build_stamp",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/runtime/dispatch:dispatch_delegate",
        "//litert/test:common",
        "//litert/tools:outstream",
        "//litert/tools:tool_display",
        "//tflite:framework",
        "//tflite:model_builder",
        "//tflite/core:private_cc_api_stable",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:builtin_ops_gms",
        "//tflite/profiling:time",
    ],
)

cc_test(
    name = "qualcomm_dispatcher_numeric_test",
    srcs = ["qualcomm_dispatcher_numeric_test.cc"],
    copts = ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    data = [
        "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    deps = [
        ":invoke_qualcomm_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "//litert/cc:litert_expected",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/test:common",
        "//tflite:framework",
        "//tflite:model_builder",
        "//tflite/core:private_cc_api_stable",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:builtin_ops_gms",
        "//tflite/kernels:kernel_util",
    ],
)

cc_library(
    name = "invoke_qualcomm_util",
    testonly = 1,
    hdrs = ["invoke_qualcomm_util.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_dispatch_delegate",
        "//litert/c:litert_environment",
        "//litert/c:litert_environment_options",
        "//litert/cc:litert_dispatch_delegate",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/core:build_stamp",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:common",
        "//litert/tools:outstream",
        "//litert/tools:tool_display",
        "//tflite:framework_stable",
        "//tflite:model_builder",
    ],
)

pytype_strict_binary(
    name = "build_torch_sources",
    srcs = ["build_torch_sources.py"],
    data = [
        ":litert_scripts",
    ],
    deps = [
        "//file/colossus/public:cns",
        "//pyglib/contrib/gpathlib:gpath",
        "//third_party/odml/infra/testing/odml_flow/commands:torch_exports",
        "//third_party/py/ai_edge_torch/aot:prepare_for_npu",
        "//third_party/py/etils/epath",
        "//third_party/py/fire",
    ],
)
