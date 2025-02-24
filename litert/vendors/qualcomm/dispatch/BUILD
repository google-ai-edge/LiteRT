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

load("//litert/build_common:litert_build_defs.bzl", "copy_file", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_invocation_context.cc",
    ],
    hdrs = [
        "litert_dispatch_device_context.h",
        "litert_dispatch_invocation_context.h",
        "registry.h",
    ],
    export_litert_only = True,
    linkopts = select({
        "//third_party/tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_Qualcomm.so",
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//litert:__subpackages__"],
    deps = [
        "//third_party/absl/log:absl_check",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/qualcomm:common",
        "//litert/vendors/qualcomm:context_binary_info",
        "//litert/vendors/qualcomm:qnn_manager",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)

# This is cc_library target for `libLiteRtDispatch_Qualcomm.so`.
cc_library(
    name = "dispatch_api_shared_lib",
    srcs = [":dispatch_api_so"],
    linkstatic = 1,
)

# Copies the shared library so that it is available for use in test data as libLiteRtDispatch_Qualcomm.so.
copy_file(
    name = "copy_dispatch_api_so",
    src = "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
    target = "libLiteRtDispatch_Qualcomm.so",
)

cc_test(
    name = "dispatch_api_qualcomm_test",
    srcs = [
        "dispatch_api_qualcomm_test.cc",
    ],
    data = [
        ":dispatch_api_so",
    ],
    linkopts = select({
        "//third_party/tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "no-remote-exec",
        "no_oss",
        "notap",
    ],
    deps = [
        "//testing/base/public:gunit_main",
        "//third_party/absl/log",
        "//third_party/absl/log:absl_log",
        "//third_party/absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/core:filesystem",
        "//litert/test:common",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)
