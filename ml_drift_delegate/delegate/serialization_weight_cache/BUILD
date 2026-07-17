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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# copybara:uncomment_begin
# # TODO: b/378108578 - Add support for open sourcing. The serialization_schema.fbs currently causes
# # issues for copybara.
# load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
# load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
#
# # Experimental libraries & runtimes -- DO NOT use or depend on these.
# package(default_applicable_licenses = ["//devtools/compliance/licenses:no_external_contributions"])
#
# copybara:uncomment_end

cc_library(
    name = "file_util",
    srcs = ["file_util.cc"],
    hdrs = ["file_util.h"],
    visibility = [
        "//ml_drift_delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = ["//tflite:minimal_logging"],
)

cc_test(
    name = "file_util_test",
    srcs = ["file_util_test.cc"],
    deps = [
        ":file_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mmap_handle",
    srcs = ["mmap_handle.cc"],
    hdrs = ["mmap_handle.h"],
    visibility = [
        "//ml_drift_delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        ":file_util",
        ":windows_util",
        "//ml_drift_delegate/delegate:unowned_tensor_desc",
        "//tflite/c:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "mmap_handle_test",
    srcs = ["mmap_handle_test.cc"],
    deps = [
        ":mmap_handle",
        ":testing_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "windows_util",
    srcs = ["windows_util.cc"],
    hdrs = ["windows_util.h"],
    tags = [
        "keep_dep",  # Build cleaner will want to remove this because it doesn't check for Windows builds.
    ],
    deps = [],
)

flatbuffer_cc_library(
    name = "serialization_schema",
    srcs = ["serialization_schema.fbs"],
    flatc_args = [
        "--gen-mutable",
    ],
    includes = [
        "@ml_drift//ml_drift/common/task:serialization_base_cc_fbs_includes",
    ],
)

# This is used to generate a file called "srcs/build_identifier.cc". This generated file contains a
# "build identifier" that is the hash of all the files provided in "srcs".
#
# To just test that the file is generated correctly, run
# bazel build -s //third_party/odml/litert/ml_drift/delegate/serialization_weight_cache:generate_build_identifier
# This will generate the file and print out its location under bazel-bin/.
#
# This generated source file is then used as the .cc file in the build_identifier library.
#
# The best way to test that everything is working end-to-end is to run the build_identifier_test. It
# will automatically cause the "srcs/build_identifier.cc" to be generated, the build_identifier
# library to be compiled and then it will perform some basic tests.
genrule(
    name = "generate_build_identifier",
    # The following source files were chosen because they are used to rearrange the shared memory
    # weights that can be serialized to disk. Currently this is only fully connected operations,
    # but as new ops are supported, their relevant source files that impact their weight
    # rearrangement should be added here.
    srcs = [
        "@ml_drift//ml_drift/common/task:weights_conversion.h",
        "@ml_drift//ml_drift/common/task:weights_conversion.cc",
        "@ml_drift//ml_drift/common/task:weights_layout.h",
        "@ml_drift//ml_drift/common/task:weights_layout.cc",
        "@ml_drift//ml_drift/common/kernels:fully_connected.h",
        "@ml_drift//ml_drift/common/kernels:fully_connected.cc",
    ],
    outs = ["srcs/build_identifier.cc"],
    cmd = "$(location //third_party/odml/litert/ml_drift/delegate/serialization_weight_cache:generate_build_identifier_py) --output $@ $(SRCS)",
    tools = [":generate_build_identifier_py"],
)

py_binary(
    name = "generate_build_identifier_py",
    srcs = ["generate_build_identifier.py"],
    main = "generate_build_identifier.py",
    strict_deps = False,
    tags = ["notap"],  # This should only be built as part of the related genrule.
)

cc_library(
    name = "build_identifier",
    srcs = ["srcs/build_identifier.cc"],
    hdrs = ["build_identifier.h"],
    hdrs_check = "strict",
    deps = ["@com_google_absl//absl/types:span"],
)

cc_test(
    name = "build_identifier_test",
    srcs = ["build_identifier_test.cc"],
    deps = [
        ":build_identifier",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cache_builder",
    srcs = ["cache_builder.cc"],
    hdrs = ["cache_builder.h"],
    deps = [
        ":build_identifier",
        ":file_util",
        ":mmap_handle",
        ":serialization_schema",
        "//tflite/c:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/task:serialization_base",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_test(
    name = "cache_builder_test",
    srcs = ["cache_builder_test.cc"],
    tags = ["nomsan"],  # TODO: b/401276576 - fix and enable msan.
    deps = [
        ":cache_builder",
        ":file_util",
        ":mmap_handle",
        ":serialization_schema",
        ":testing_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:serialization_base_cc_fbs",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "serialization_weight_cache",
    srcs = ["serialization_weight_cache.cc"],
    hdrs = ["serialization_weight_cache.h"],
    visibility = [
        "//ml_drift_delegate:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
    ],
    deps = [
        ":build_identifier",
        ":cache_builder",
        ":file_util",
        ":mmap_handle",
        ":serialization_schema",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/task:serialization_base",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//ml_drift_delegate/delegate:precision",
        "//ml_drift_delegate/delegate:unowned_tensor_desc",
        "//tflite/c:common",
        "//tflite/delegates:serialization",
        # placeholder for farmhash_fingerprint
    ],
)

cc_test(
    name = "serialization_weight_cache_test",
    srcs = ["serialization_weight_cache_test.cc"],
    tags = ["nomsan"],  # TODO: b/401276576 - fix and enable msan.
    deps = [
        ":build_identifier",
        ":cache_builder",
        ":serialization_schema",
        ":serialization_weight_cache",
        ":testing_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@ml_drift//ml_drift/common:access_type",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:serialization_base_cc_fbs",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "testing_util",
    hdrs = ["testing_util.h"],
    deps = ["@com_google_absl//absl/strings:string_view"],
)
