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
# copybara:uncomment load("INTERNAL_ACCEL_RUNTIME/common:build_defs.bzl", "compile_model")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
#     srcs = ["INTERNAL_ACCEL_RUNTIME/star_ship/south_bound:sb_api_late_binding.cc"],
#     outs = ["sb_api_late_binding.cc"],
#     # Replace g3 header path with that of the copy made above.
#     cmd = "sed 's|third_party/odml/infra/southbound/sb_api.h|third_party/odml/litert/litert/vendors/google_tensor/dispatch/sb_api.h|g' $< > $@",
# )
#
# genrule(
#     name = "sb_api_features_h",
#     srcs = ["INTERNAL_ACCEL_RUNTIME/star_ship/south_bound:sb_api_features.h"],
#     outs = ["sb_api_features.h"],
#     # Replace g3 header path with that of the copy made above.
#     cmd = "sed 's|third_party/odml/infra/southbound/sb_api.h|third_party/odml/litert/litert/vendors/google_tensor/dispatch/sb_api.h|g' $< > $@",
# )
# copybara:uncomment_end

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
        "dispatch_api_config.cc",
        "dispatch_api_utils.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_graph.cc",
        "litert_dispatch_invocation_context.cc",
        "sb_api_late_binding.cc",
    ],
    hdrs = [
        "dispatch_api.h",
        "dispatch_api_config.h",
        "dispatch_api_macros.h",
        "dispatch_api_utils.h",
        "litert_dispatch_device_context.h",
        "litert_dispatch_graph.h",
        "litert_dispatch_invocation_context.h",
        "litert_dispatch_metrics.h",
        "sb_api.h",
        "sb_api_features.h",
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
    visibility = [
        "//litert:litert_public",
        "//platforms/darwinn/tests/sdk_3p/litert_tools:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
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

cc_library(
    name = "dispatch_api_test_fixtures",
    testonly = True,
    srcs = ["dispatch_api_test_fixtures.cc"],
    hdrs = ["dispatch_api_test_fixtures.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/core:filesystem_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//testing/base/public:gunit_for_library_testonly",
        "@com_google_absl//absl/base:no_destructor",
    ],
)

# copybara:uncomment_begin(google-only)
# compile_model(
#     name = "simple_model_reference_bin",
#     chip_family = INTERNAL_ACCEL_CHIP_FAMILY_24,
#     chip_variant = INTERNAL_ACCEL_CHIP_VARIANT_0,
#     compiled_binary_file = "simple_model_reference_google_tensor.bin",
#     model_source = "//litert/test:testdata/simple_model.mlir",
# )
#
# cc_test(
#     name = "dispatch_api_simple_model_test_x86",
#     srcs = ["dispatch_api_simple_model_test.cc"],
#     data = [
#         "simple_model_reference_google_tensor.bin",
#         ":dispatch_api_so",
#         "//third_party/darwinn/litert/dispatch:libedgetpu_litert.so",
#     ],
#     deps = [
#         ":dispatch_api_test_fixtures",
#         "//platforms/darwinn/tachyon/core/fence",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/strings:str_format",
#         "@com_google_absl//absl/time",
#         "@com_google_absl//absl/types:span",
#         "//third_party/darwinn/driver_shared/fence:fence_test_util",
#         "//litert/c:litert_runtime_c_api_shared_lib",
#         "//litert/c/internal:litert_logging",
#         "//litert/test:matchers",
#         "//litert/test:simple_model_npu",
#         "//litert/vendors/c:litert_dispatch_c_api",
#         "INTERNAL_THREAD_PKG",
#     ],
# )
# copybara:uncomment_end

litert_device_test(
    name = "dispatch_api_simple_model_test",
    srcs = ["dispatch_api_simple_model_test.cc"],
    backend_id = "google_tensor",
    data = [
        "//litert/test:testdata/simple_model_google_tensor.bin",
    ],
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        "INTERNAL_THREAD_PKG",
        ":dispatch_api_test_fixtures",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/test:matchers",
        "//litert/test:simple_model_npu",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//platforms/darwinn/tachyon/core/fence",
        "//third_party/darwinn/driver_shared/fence:fence_test_util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
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
