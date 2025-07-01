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
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Core LiteRT Tensor library
cc_library(
    name = "litert_tensor",
    srcs = ["litert_tensor.cc"],
    hdrs = ["litert_tensor.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/types:span",
    ],
)

# Example usage binary
cc_binary(
    name = "tensor_examples",
    srcs = ["example_usage.cc"],
    deps = [
        ":litert_tensor",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_model",
        "//litert/cc:litert_macros",
    ],
)

# Simple test binary
cc_binary(
    name = "simple_test",
    srcs = ["simple_test.cc"],
    deps = [
        ":litert_tensor",
    ],
)

# Working demo binary
cc_binary(
    name = "working_demo",
    srcs = ["working_demo.cc"],
    deps = [
        ":litert_tensor",
    ],
)

# Minimal demo binary
cc_binary(
    name = "minimal_demo",
    srcs = ["minimal_demo.cc"],
    deps = [
        ":litert_tensor",
    ],
)

# Safe demo binary
cc_binary(
    name = "safe_demo",
    srcs = ["safe_demo.cc"],
    deps = [
        ":litert_tensor",
    ],
)

# Unit tests
cc_test(
    name = "litert_tensor_test",
    srcs = ["litert_tensor_test.cc"],
    deps = [
        ":litert_tensor",
        "@com_google_googletest//:gtest_main",
    ],
)