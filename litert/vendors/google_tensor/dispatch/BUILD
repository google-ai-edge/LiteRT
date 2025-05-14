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
load("//litert/integration_test:litert_device.bzl", "litert_device_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_graph.cc",
        "litert_dispatch_invocation_context.cc",
        "southbound.cc",
    ],
    hdrs = [
        "dispatch_api.h",
        "litert_dispatch_device_context.h",
        "litert_dispatch_graph.h",
        "litert_dispatch_invocation_context.h",
        "litert_dispatch_metrics.h",
        "southbound.h",
        # copybara:uncomment "//third_party/odml/infra/southbound:sb_api.h",
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
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }) + [
        "-Wl,-soname=libLiteRtDispatch_GoogleTensor.so",
        "-Wl,-lc++abi",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_GoogleTensor.so",
    tags = [
        # Don't build/test in OSS until Southbound is available.
        "nobuilder",
    ],
    visibility = [
        "//litert:__subpackages__",
        "//litert:litert_internal_users",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_any",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_environment",
        "//litert/cc/internal:litert_options",
        # TODO: Remove this dependency.
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

litert_device_test(
    name = "dispatch_api_google_tensor_test",
    srcs = ["dispatch_api_google_tensor_test.cc"],
    backend_id = "google_tensor",
    data = [
        "//litert/test:testdata/simple_model_google_tensor.bin",
        "//litert/test:testdata/simple_model_npu.tflite",
    ],
    deps = [
        ":dispatch_api",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_any",
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

litert_device_test(
    name = "dispatch_api_async_google_tensor_test",
    srcs = ["dispatch_api_async_google_tensor_test.cc"],
    backend_id = "google_tensor",
    data = [
        "//litert/test:testdata/simple_model_google_tensor.bin",
        "//litert/test:testdata/simple_model_npu.tflite",
    ],
    deps = [
        ":dispatch_api",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/darwinn/driver_shared/fence:fence_test_util",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_event",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_any",
        "//litert/core:filesystem_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)
