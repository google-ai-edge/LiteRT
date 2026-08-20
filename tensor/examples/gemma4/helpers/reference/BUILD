# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

py_library(
    name = "rmsnorm_lib",
    srcs = ["rmsnorm.py"],
    deps = [
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

py_binary(
    name = "rmsnorm",
    srcs = ["rmsnorm.py"],
    deps = [
        ":rmsnorm_lib",
    ],
)

py_library(
    name = "rope_lib",
    srcs = ["rope.py"],
    deps = [
        ":utils",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

py_binary(
    name = "rope",
    srcs = ["rope.py"],
    deps = [
        ":rope_lib",
    ],
)

py_library(
    name = "feed_forward_network_lib",
    srcs = ["feed_forward_network.py"],
    deps = [
        ":utils",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

py_binary(
    name = "feed_forward_network",
    srcs = ["feed_forward_network.py"],
    deps = [
        ":feed_forward_network_lib",
    ],
)

py_library(
    name = "attention_lib",
    srcs = ["attention.py"],
    deps = [
        ":rmsnorm_lib",
        ":rope_lib",
        ":utils",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

py_binary(
    name = "attention",
    srcs = ["attention.py"],
    deps = [
        ":attention_lib",
    ],
)

py_library(
    name = "transformer_lib",
    srcs = ["transformer.py"],
    deps = [
        ":attention_lib",
        ":feed_forward_network_lib",
        ":rmsnorm_lib",
        ":rope_lib",
        ":utils",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

py_binary(
    name = "transformer",
    srcs = ["transformer.py"],
    deps = [
        ":transformer_lib",
    ],
)

py_library(
    name = "gemma4_graph_lib",
    srcs = ["gemma4_graph.py"],
    deps = [
        ":rmsnorm_lib",
        ":transformer_lib",
        ":utils",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

py_binary(
    name = "gemma4_graph",
    srcs = ["gemma4_graph.py"],
    deps = [
        ":gemma4_graph_lib",
    ],
)
