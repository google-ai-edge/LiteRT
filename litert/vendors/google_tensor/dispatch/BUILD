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

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

# copybara:uncomment_begin(google-only)
# genrule(
#     name = "sb_api_h",
#     srcs = ["//third_party/odml/infra/southbound:sb_api.h"],
#     outs = ["sb_api.h"],
#     cmd = "cp $< $@",
# )
#
# genrule(
#     name = "sb_api_late_binding_cc",
#     srcs = ["INTERNAL_SB_API_LATE_BINDING"],
#     outs = ["sb_api_late_binding.cc"],
#     # Replace g3 header path with that of the copy made above.
#     cmd = "sed 's|third_party/odml/infra/southbound/sb_api.h|third_party/odml/litert/litert/vendors/google_tensor/dispatch/sb_api.h|g' $< > $@",
# )
# copybara:uncomment_end

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_graph.cc",
        "litert_dispatch_invocation_context.cc",
        "sb_api_late_binding.cc",
    ],
    hdrs = [
        "dispatch_api.h",
        "litert_dispatch_device_context.h",
        "litert_dispatch_graph.h",
        "litert_dispatch_invocation_context.h",
        "litert_dispatch_metrics.h",
        "sb_api.h",
        "sb_dispatch_annotations.h",
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
        "-Wl,-soname=libLiteRtDispatch_GoogleTensor.so",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_GoogleTensor.so",
    tags = [
        # Don't build/test in OSS until Southbound is available.
        "nobuilder",
    ],
    visibility = ["//litert:litert_public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c/internal:litert_logging",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_macros",
        # TODO: Remove this dependency.
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime/options:litert_darwinn_options",
        "//litert/vendors/cc:options_helper",
    ],
)

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "dispatch_api_google_tensor_test_x86",
#     srcs = ["dispatch_api_google_tensor_test.cc"],
#     data = [
#         ":dispatch_api_so",
#         "//third_party/darwinn/litert/dispatch:libedgetpu_litert.so",
#         "//litert/test:testdata/simple_model_reference_google_tensor.bin",
#     ],
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/base:no_destructor",
#         "@com_google_absl//absl/log",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/types:span",
#         "//litert/c:litert_runtime_c_api_shared_lib",
#         "//litert/cc:litert_expected",
#         "//litert/cc/dynamic_runtime:litert_environment",
#         "//litert/cc/dynamic_runtime:litert_options",
#         "//litert/core:filesystem_testonly",
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model_npu",
#         "//litert/vendors/c:litert_dispatch_c_api",
#     ],
# )
# copybara:uncomment_end

litert_device_test(
    name = "dispatch_api_google_tensor_test",
    srcs = ["dispatch_api_google_tensor_test.cc"],
    backend_id = "google_tensor",
    data = [
        "//litert/test:testdata/simple_model_google_tensor.bin",
        "//litert/test:testdata/simple_model_npu.tflite",
    ],
    # TODO: google_tensor team fix visibility for fence deps.
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/core:filesystem_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/base:no_destructor",
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
    features = select({
        "//tools/cc_target_os:android": ["-layering_check"],
        "//conditions:default": [],
    }),
    # TODO: google_tensor team fix visibility for fence deps.
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/darwinn/driver_shared/fence:fence_test_util",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_any",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/core:filesystem_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

cc_test(
    name = "darwinn_options_test",
    srcs = ["darwinn_options_test.cc"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime/options:litert_darwinn_options",
        "//litert/cc/internal:litert_handle",
        "@com_google_googletest//:gtest_main",
    ],
)
