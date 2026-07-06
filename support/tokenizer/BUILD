# Copyright 2025 The ODML Authors.
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

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_hdrs_check = "strict",
    default_visibility = [
        # copybara:uncomment_begin(oss litert)
        # "//third_party/odml/litert:__subpackages__",
        # "//third_party/odml/litert_lm:__subpackages__",
        # "//third_party/odml/litert_lm:clients",
        # "//third_party/odml/litert_lm:internal",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

licenses(["notice"])

cc_library(
    name = "tokenizer",
    hdrs = ["tokenizer.h"],
    deps = [
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//support/util:convert_tensor_buffer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "tokenizer_test",
    srcs = ["tokenizer_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":tokenizer",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:matchers",
        "//support/util:convert_tensor_buffer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer",
    srcs = ["sentencepiece_tokenizer.cc"],
    hdrs = ["sentencepiece_tokenizer.h"],
    defines = ["ENABLE_SENTENCEPIECE_TOKENIZER"],
    deps = [
        ":tokenizer",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@sentencepiece//:sentencepiece_model_cc_proto",
        "@sentencepiece//:sentencepiece_processor",
    ],
)

cc_test(
    name = "sentencepiece_tokenizer_test",
    srcs = ["sentencepiece_tokenizer_test.cc"],
    data = ["//support/tokenizer/testdata"],
    deps = [
        ":sentencepiece_tokenizer",
        ":tokenizer",
        "//support/util:test_utils",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "huggingface_tokenizer",
    srcs = ["huggingface_tokenizer.cc"],
    hdrs = ["huggingface_tokenizer.h"],
    defines = ["ENABLE_HUGGINGFACE_TOKENIZER"],
    deps = [
        ":tokenizer",
        # copybara:uncomment "//devtools/rust:rust_okay_here",  # build_cleaner: keep
        "@com_google_absl//absl/debugging:leak_check",  # See b/402708346
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/util:litert_status_util",
        "//support/util:memory_mapped_file",
        "@tokenizers_cpp//:huggingface_tokenizer",
    ] + select({
        "@platforms//os:windows": [
            "//support/util:alloc_defs",
            "//support/util:global_allocator",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "huggingface_tokenizer_test",
    srcs = ["huggingface_tokenizer_test.cc"],
    data = ["//support/tokenizer/testdata"],
    deps = [
        ":huggingface_tokenizer",
        ":tokenizer",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
