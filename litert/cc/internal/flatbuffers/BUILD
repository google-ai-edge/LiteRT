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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_internal_users",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    features = ["allow_starlark_transitions"],
)

flatbuffer_cc_library(
    name = "tensor_buffer_requirements_fbs",
    srcs = ["tensor_buffer_requirements.fbs"],
)

cc_test(
    name = "tensor_buffer_requirements_fbs_test",
    srcs = ["tensor_buffer_requirements_fbs_test.cc"],
    deps = [
        ":tensor_buffer_requirements_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)
