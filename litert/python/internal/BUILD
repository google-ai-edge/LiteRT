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

# TODO(b/445163709): Remove this module once litert_lm publishes a pypi package.

load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")
load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

flatbuffer_py_library(
    name = "litertlm_header_schema_py",
    srcs = ["litertlm_header_schema.fbs"],
)

proto_library(
    name = "sampler_params_proto",
    srcs = ["sampler_params.proto"],
)

py_proto_library(
    name = "sampler_params_py_pb2",
    deps = [":sampler_params_proto"],
)

proto_library(
    name = "llm_metadata_proto",
    srcs = ["llm_metadata.proto"],
    deps = [":sampler_params_proto"],
)

py_proto_library(
    name = "llm_metadata_py_pb2",
    deps = [":llm_metadata_proto"],
)

pytype_strict_library(
    name = "litertlm_core",
    srcs = ["litertlm_core.py"],
    deps = [
        ":litertlm_header_schema_py",
    ],
)

pytype_strict_library(
    name = "litertlm_builder",
    srcs = ["litertlm_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":litertlm_core",
        ":litertlm_header_schema_py",
        ":llm_metadata_py_pb2",
        "@com_google_protobuf//:protobuf_python",
        "@flatbuffers//:runtime_py",
    ],
)
