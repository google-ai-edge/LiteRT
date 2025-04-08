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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "auto_registration",
    srcs = ["auto_registration.cc"],
    hdrs = ["auto_registration.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_shared_library",
        "//litert/core:environment",
        "//litert/runtime/accelerators/dispatch:dispatch_accelerator",
        "//litert/runtime/accelerators/xnnpack:xnnpack_accelerator",
    ],
)

cc_library(
    name = "accelerator_implementation_helper",
    hdrs = ["accelerator_implementation_helper.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_accelerator",
        "//litert/c:litert_accelerator_compilation_options",
        "//litert/c:litert_accelerator_registration",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/runtime:accelerator_model_compilation_data",
    ],
)
