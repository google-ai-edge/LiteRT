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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors:__subpackages__"],
)

cc_library(
    name = "vendor_dispatch_base",
    srcs = ["vendor_traits.cc"],
    hdrs = [
        "vendor_dispatch_base.h",
        "vendor_traits.h",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_logging",
        "//litert/c:litert_metrics",
        "//litert/c:litert_model",
        "//litert/c:litert_options",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "vendor_dispatch_migration_example",
    srcs = ["vendor_dispatch_migration_example.cc"],
    deps = [
        ":vendor_dispatch_base",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
    ],
)

cc_test(
    name = "vendor_dispatch_test",
    srcs = ["vendor_dispatch_test.cc"],
    deps = [
        ":vendor_dispatch_base",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_googletest//:gtest_main",
    ],
)
