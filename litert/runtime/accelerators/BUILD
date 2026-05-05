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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "auto_registration",
    srcs = ["auto_registration.cc"],
    hdrs = ["auto_registration.h"],
    deps = [
        ":cpu_registry",
        ":gpu_registry",
        ":npu_registry",
        ":webnn_registry",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c/internal:litert_accelerator_registration",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/core:environment",
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
        "//litert/c/internal:litert_custom_tensor_buffer_handlers_def",
        "//litert/c/internal:litert_tensor_buffer_registry_header",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_shared_library",
        "//litert/core:environment",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "cpu_registry",
    srcs = ["cpu_registry.cc"],
    hdrs = ["cpu_registry.h"],
    copts = select({
        "//litert/build_common:litert_disable_cpu": [],
        "//conditions:default": [
            "-DLITERT_USE_XNNPACK",
        ],
    }),
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
    ] + select({
        "//litert/build_common:litert_disable_cpu": [],
        "//conditions:default": [
            ":registration_helper",
            "//litert/c/internal:litert_accelerator_def",
            "//litert/runtime/accelerators/xnnpack:xnnpack_accelerator",
        ],
    }),
)

cc_library(
    name = "gpu_static_registry",
    hdrs = ["gpu_static_registry.h"],
    deps = [
        "//litert/c/internal:litert_accelerator_def",
    ],
)

cc_library(
    name = "gpu_registry",
    srcs = ["gpu_registry.cc"],
    hdrs = ["gpu_registry.h"],
    deps = [
        ":gpu_static_registry",
        ":registration_helper",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/core:environment",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "npu_registry",
    srcs = ["npu_registry.cc"],
    hdrs = ["npu_registry.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/runtime/accelerators/dispatch:dispatch_accelerator",
    ],
)

cc_library(
    name = "webnn_registry",
    srcs = ["webnn_registry.cc"],
    hdrs = ["webnn_registry.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/core:environment",
    ],
)
