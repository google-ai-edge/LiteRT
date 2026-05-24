# Copyright 2026 Google LLC.
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

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_test")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "device_context.cc",
        "dispatch_api.cc",
        "invocation_context.cc",
    ],
    hdrs = [
        "device_context.h",
        "invocation_context.h",
    ],
    copts = [
        "-Os",
        "-fexceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    export_litert_only = True,
    features = ["-use_header_modules"],
    linkopts = litert_android_linkopts() + [
        "-Wl,-soname=libLiteRtDispatch_Hailo.so",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_Hailo.so",
    visibility = [
        "//litert:__subpackages__",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model_header",
        "//litert/c:litert_tensor_buffer_header",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_custom_tensor_buffer_handlers_def",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_logging_helper_with_runtime_context",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core/util:tensor_type_util",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@hailo_sdk//:hailort",
    ],
)

litert_test(
    name = "dispatch_api_hailo_test",
    srcs = [
        "dispatch_api_hailo_test.cc",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    linkstatic = True,
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":dispatch_api",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_runtime_context",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:simple_buffer",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
