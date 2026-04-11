# Copyright 2025 Google LLC.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

py_binary(
    name = "generate_registry",
    srcs = ["generate_registry.py"],
)

cc_library(
    name = "auto_registration",
    srcs = ["auto_registration.cc"],
    hdrs = ["auto_registration.h"],
    deps = [
        ":cpu_registry",
        ":gpu_registry",
        ":registration_helper",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_accelerator_registration",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_tensor_buffer_registry_header",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//litert/runtime/accelerators/dispatch:dispatch_accelerator",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "accelerator_implementation_helper",
    hdrs = ["accelerator_implementation_helper.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_accelerator_registration",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "registration_helper",
    srcs = ["registration_helper.cc"],
    hdrs = ["registration_helper.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_accelerator_registration",
        "//litert/c/internal:litert_tensor_buffer_registry_header",
        "//litert/cc:litert_macros",
    ],
)

genrule(
    name = "generate_builtin_accelerators_cc",
    outs = [
        "builtin_accelerators.cc",
        "builtin_accelerators.h",
    ],
    cmd = "$(location :generate_registry) $(location builtin_accelerators.cc) $(location builtin_accelerators.h) LiteRtCpuAccelerator --gpu " + select({
        "@platforms//os:android": "LiteRtOpenClAccelerator LiteRtOpenClGlAccelerator LiteRtWebGpuAccelerator LiteRtVulkanAccelerator LiteRtGpuAccelerator",
        "@platforms//os:ios": "",
        "@platforms//os:macos": "",
        "@platforms//os:emscripten": "LiteRtWebNnAccelerator LiteRtWebGpuAccelerator",
        "@platforms//os:windows": "",
        "//conditions:default": "LiteRtOpenClAccelerator LiteRtOpenClGlAccelerator LiteRtWebGpuAccelerator LiteRtVulkanAccelerator LiteRtGpuAccelerator",
    }),
    tools = [":generate_registry"],
)

cc_library(
    name = "builtin_accelerators",
    srcs = ["builtin_accelerators.cc"],
    hdrs = ["builtin_accelerators.h"],
    deps = [
        "//litert/c/internal:litert_accelerator_def",
    ],
)

cc_library(
    name = "cpu_registry",
    srcs = ["cpu_registry.cc"],
    hdrs = ["cpu_registry.h"],
    deps = [
        ":builtin_accelerators",
        ":registration_helper",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_logging",
    ],
)

cc_library(
    name = "gpu_registry",
    srcs = ["gpu_registry.cc"],
    hdrs = ["gpu_registry.h"],
    deps = [
        ":builtin_accelerators",
        ":registration_helper",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_shared_library",
        "//litert/core:environment",
        "@com_google_absl//absl/strings",
    ],
)
