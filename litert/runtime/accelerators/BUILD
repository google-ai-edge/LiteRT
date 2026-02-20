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
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_accelerator_registration",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_tensor_buffer_registry_header",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_shared_library",
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
