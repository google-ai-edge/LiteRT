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

load("//tflite/experimental/litert/build_common:special_rule.bzl", "gles_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//tflite/experimental/litert:__subpackages__",
        "@org_tensorflow//third_party/odml/infra/perf/mobile_tests/litert:__subpackages__",
    ],
)

# Dispatch API implementation, it is used by the dispatch delegate to call the vendor's dispatch
# API.
cc_library(
    name = "dispatch",
    srcs = [
        "litert_dispatch.cc",
    ],
    hdrs = [
        "//tflite/experimental/litert/vendors/c:litert_dispatch.h",
        "//tflite/experimental/litert/vendors/c:litert_dispatch_api.h",
    ],
    deps = [
        "//tflite/experimental/litert/c:litert_any",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_event",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/c:litert_tensor_buffer",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/core:dynamic_loading",
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
        "dispatch_delegate_options.h",
        "//tflite/experimental/litert/c:litert_dispatch_delegate.h",
        "//tflite/experimental/litert/cc:litert_dispatch_delegate.h",
    ],
    deps = [
        ":dispatch",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:c_api_opaque_without_op_resolver",
        "//tflite/delegates/utils:simple_opaque_delegate",
        "//tflite/experimental/litert/c:litert_any",
        "//tflite/experimental/litert/c:litert_common",
        "//tflite/experimental/litert/c:litert_event",
        "//tflite/experimental/litert/c:litert_logging",
        "//tflite/experimental/litert/c:litert_model",
        "//tflite/experimental/litert/c:litert_tensor_buffer",
        "//tflite/experimental/litert/cc:litert_any",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_model",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/core:build_stamp",
        "//tflite/experimental/litert/core:dispatch_op_schema",
        "//tflite/experimental/litert/core:dynamic_loading",
        "//tflite/experimental/litert/core:environment",
        "//tflite/experimental/litert/runtime:external_litert_buffer_context",
        "//tflite/experimental/litert/runtime:tfl_utils",
        "//tflite/experimental/litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "dispatch_delegate_google_tensor_test",
    srcs = ["dispatch_delegate_google_tensor_test.cc"],
    data = [
        "//tflite/experimental/litert/test:testdata/shared_input_cpu_npu.tflite",
        "//tflite/experimental/litert/vendors/google_tensor/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }) + gles_linkopts(),
    deps = [
        ":dispatch_delegate",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "@org_tensorflow//third_party/darwinn/driver_shared/fence:fence_test_util",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/experimental/litert/c:litert_runtime_c_api_shared_lib",
        "//tflite/experimental/litert/cc:litert_buffer_ref",
        "//tflite/experimental/litert/cc:litert_compilation_options",
        "//tflite/experimental/litert/cc:litert_compiled_model",
        "//tflite/experimental/litert/cc:litert_environment",
        "//tflite/experimental/litert/cc:litert_event",
        "//tflite/experimental/litert/cc:litert_expected",
        "//tflite/experimental/litert/cc:litert_model",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/core/model:model_buffer",
        "//tflite/experimental/litert/core/util:flatbuffer_tools",
        "//tflite/experimental/litert/runtime:external_litert_buffer_context",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:matchers",
        "//tflite/experimental/litert/test:simple_model_npu",
    ],
)

cc_test(
    name = "dispatch_delegate_qualcomm_test",
    srcs = ["dispatch_delegate_qualcomm_test.cc"],
    data = [
        "//tflite/experimental/litert/test:testdata/shared_input_cpu_npu.tflite",
        "//tflite/experimental/litert/vendors/qualcomm/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":dispatch_delegate",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/experimental/litert/c:litert_runtime_c_api_shared_lib",
        "//tflite/experimental/litert/cc:litert_compilation_options",
        "//tflite/experimental/litert/cc:litert_compiled_model",
        "//tflite/experimental/litert/cc:litert_dispatch_delegate",
        "//tflite/experimental/litert/cc:litert_environment",
        "//tflite/experimental/litert/cc:litert_model",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/runtime:external_litert_buffer_context",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:simple_model_npu",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dispatch_delegate_mediatek_test",
    srcs = ["dispatch_delegate_mediatek_test.cc"],
    data = [
        "//tflite/experimental/litert/test:testdata/shared_input_cpu_npu.tflite",
        "//tflite/experimental/litert/vendors/mediatek/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "no_oss",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":dispatch_delegate",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/experimental/litert/c:litert_runtime_c_api_shared_lib",
        "//tflite/experimental/litert/cc:litert_compilation_options",
        "//tflite/experimental/litert/cc:litert_compiled_model",
        "//tflite/experimental/litert/cc:litert_environment",
        "//tflite/experimental/litert/cc:litert_model",
        "//tflite/experimental/litert/cc:litert_tensor_buffer",
        "//tflite/experimental/litert/core/model:model_buffer",
        "//tflite/experimental/litert/core/util:flatbuffer_tools",
        "//tflite/experimental/litert/runtime:external_litert_buffer_context",
        "//tflite/experimental/litert/test:common",
        "//tflite/experimental/litert/test:simple_model_npu",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
