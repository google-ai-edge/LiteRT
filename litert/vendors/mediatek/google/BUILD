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

load("//litert/build_common:litert_build_defs.bzl", "litert_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

litert_test(
    name = "neuron_adapter_api_test",
    srcs = ["neuron_adapter_api_test.cc"],
    linkstatic = True,
    tags = ["local"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "//third_party/neuro_pilot:latest_host_headers",
        "//litert/core:filesystem_testonly",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)
