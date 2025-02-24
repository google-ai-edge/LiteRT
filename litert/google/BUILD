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

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

exports_files(
    srcs = ["run_test_on_android.sh"],
)

cc_test(
    name = "invoke_qualcomm_test",
    srcs = ["invoke_qualcomm_test.cc"],
    copts = ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    data = [
        "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "//third_party/tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    deps = [
        ":invoke_qualcomm_util",
        "//testing/base/public:gunit_main",
        "//third_party/absl/flags:flag",
        "//third_party/absl/log:absl_check",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/core:build_stamp",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/runtime/dispatch:dispatch_delegate",
        "//litert/test:common",
        "//litert/tools:outstream",
        "//litert/tools:tool_display",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite:model_builder",
        "//third_party/tensorflow/lite/core:cc_api_stable",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
        "//third_party/tensorflow/lite/kernels:builtin_ops_gms",
        "//third_party/tensorflow/lite/profiling:time",
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
        "//third_party/tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    deps = [
        ":invoke_qualcomm_util",
        "//testing/base/public:gunit_main",
        "//third_party/absl/flags:flag",
        "//third_party/absl/log:absl_check",
        "//third_party/absl/strings:string_view",
        "//litert/cc:litert_expected",
        "//litert/core/util:flatbuffer_tools",
        "//litert/runtime:external_litert_buffer_context",
        "//litert/test:common",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite:model_builder",
        "//third_party/tensorflow/lite/core:cc_api_stable",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
        "//third_party/tensorflow/lite/kernels:builtin_ops_gms",
        "//third_party/tensorflow/lite/kernels:kernel_util",
    ],
)

cc_library(
    name = "invoke_qualcomm_util",
    testonly = 1,
    hdrs = ["invoke_qualcomm_util.h"],
    deps = [
        "//third_party/absl/log:absl_check",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_dispatch_delegate",
        "//litert/c:litert_environment",
        "//litert/cc:litert_dispatch_delegate",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/core:build_stamp",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:common",
        "//litert/tools:outstream",
        "//litert/tools:tool_display",
        "//third_party/tensorflow/lite:framework_stable",
        "//third_party/tensorflow/lite:model_builder",
    ],
)
