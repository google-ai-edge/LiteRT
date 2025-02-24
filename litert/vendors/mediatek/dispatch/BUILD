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
    ],
    export_litert_only = True,
    linkopts = select({
        "//third_party/tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_Mediatek.so",
    tags = [
        # Remove when sdk is available to bazel.
        "nobuilder",
        "notap",
    ],
    visibility = ["//litert:__subpackages__"],
    deps = [
        "//third_party/absl/base:core_headers",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/container:node_hash_set",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/synchronization",
        "//third_party/absl/types:span",
        # copybara:uncomment "//third_party/neuro_pilot:latest_host_headers",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_tensor_buffer",
        "//litert/core:dynamic_loading",
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

# This is cc_library target for `libLiteRtDispatch_Mediatek.so`.
cc_library(
    name = "dispatch_api_shared_lib",
    srcs = [":dispatch_api_so"],
    linkstatic = 1,
)

cc_test(
    name = "dispatch_api_mediatek_test",
    srcs = [
        "dispatch_api_mediatek_test.cc",
    ],
    data = [
        ":dispatch_api_so",
    ],
    linkopts = select({
        "//third_party/tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "no-remote-exec",
        "no_oss",
        "nobuilder",
        "nosan",
        "notap",
    ],
    deps = [
        "//testing/base/public:gunit_main",
        "//third_party/absl/log",
        "//third_party/absl/log:absl_log",
        "//third_party/absl/types:span",
        # copybara:uncomment "//third_party/neuro_pilot:latest_host_headers",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/core:filesystem",
        "//litert/test:common",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)
