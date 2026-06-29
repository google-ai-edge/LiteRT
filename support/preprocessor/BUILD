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
    name = "audio_preprocessor",
    hdrs = ["audio_preprocessor.h"],
    deps = [
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:litert_status_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "audio_preprocessor_test",
    srcs = ["audio_preprocessor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":audio_preprocessor",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:matchers",
        "//support/util:convert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:test_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "audio_preprocessor_miniaudio",
    srcs = ["audio_preprocessor_miniaudio.cc"],
    hdrs = ["audio_preprocessor_miniaudio.h"],
    deps = [
        ":audio_preprocessor",
        ":mel_filterbank",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "@kissfft",
        "@kissfft//:kissfftr",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer_types",
        "//support/util:io_types",
        "//support/util:litert_status_util",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_tensor_buffer",
        # copybara:uncomment_begin(oss litert)
        # ] + select({
        # "@platforms//os:ios": ["//third_party/rive_miniaudio:rive_miniaudio_objc"],
        # "@platforms//os:macos": ["//third_party/rive_miniaudio:rive_miniaudio_objc"],
        # "//conditions:default": ["//third_party/rive_miniaudio"],
        # }),
        # copybara:uncomment_end_and_comment_begin
    ] + select({
        "@platforms//os:ios": ["@miniaudio//:miniaudio_objc"],
        "@platforms//os:macos": ["@miniaudio//:miniaudio_objc"],
        "//conditions:default": ["@miniaudio"],
    }),
    # copybara:comment_end
)

cc_test(
    name = "audio_preprocessor_miniaudio_test",
    srcs = ["audio_preprocessor_miniaudio_test.cc"],
    data = [
        "//support/preprocessor/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":audio_preprocessor",
        ":audio_preprocessor_miniaudio",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mel_filterbank",
    srcs = [
        "mel_filterbank.cc",
    ],
    hdrs = [
        "mel_filterbank.h",
    ],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "mel_filterbank_test",
    srcs = ["mel_filterbank_test.cc"],
    deps = [
        ":mel_filterbank",
        ":signal_vector_util",
        "//support/util:test_utils",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "signal_vector_util",
    hdrs = ["signal_vector_util.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
    ],
)

cc_test(
    name = "signal_vector_util_test",
    srcs = ["signal_vector_util_test.cc"],
    deps = [
        ":signal_vector_util",
        "//support/util:test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "image_preprocessor",
    hdrs = ["image_preprocessor.h"],
    deps = [
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:litert_status_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

alias(
    name = "image_preprocessor_impl",
    actual = select({
        ":enable_skia_image_preprocessor": ":skia_image_preprocessor",
        "//conditions:default": ":stb_image_preprocessor",
    }),
    visibility = ["//visibility:public"],
)

cc_test(
    name = "image_preprocessor_test",
    srcs = ["image_preprocessor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":image_preprocessor",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:matchers",
        "//support/util:convert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "by_pass_audio_preprocessor",
    hdrs = ["by_pass_audio_preprocessor.h"],
    deps = [
        ":audio_preprocessor",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:litert_status_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "by_pass_audio_preprocessor_test",
    srcs = ["by_pass_audio_preprocessor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":by_pass_audio_preprocessor",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:matchers",
        "//support/util:convert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "by_pass_image_preprocessor",
    hdrs = ["by_pass_image_preprocessor.h"],
    deps = [
        ":image_preprocessor",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:litert_status_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "by_pass_image_preprocessor_test",
    srcs = ["by_pass_image_preprocessor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":by_pass_image_preprocessor",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:matchers",
        "//support/util:convert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "stb_image_preprocessor",
    srcs = ["stb_image_preprocessor.cc"],
    hdrs = ["stb_image_preprocessor.h"],
    deps = [
        ":image_preprocessor",
        ":image_preprocessor_utils",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//support/util:convert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:litert_status_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@stblib",
    ],
)

cc_test(
    name = "stb_image_preprocessor_test",
    srcs = ["stb_image_preprocessor_test.cc"],
    data = ["//support/preprocessor/testdata"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":image_preprocessor",
        ":stb_image_preprocessor",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//support/util:io_types",
        "//support/util:test_utils",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

config_setting(
    name = "enable_skia_image_preprocessor",
    values = {
        "define": "enable_skia_image_preprocessor=true",
    },
)

cc_library(
    name = "skia_image_preprocessor",
    srcs = ["skia_image_preprocessor.cc"],
    hdrs = ["skia_image_preprocessor.h"],
    deps = [
        ":image_preprocessor",
        ":image_preprocessor_utils",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//support/util:convert_tensor_buffer",
        "//support/util:io_types",
        "//support/util:litert_status_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@skia//:bmp_decode_codec",
        "@skia//:core",
    ],
)

# copybara:uncomment_begin(oss litert)
# cc_test(
#     name = "skia_image_preprocessor_test",
#     srcs = ["skia_image_preprocessor_test.cc"],
#     data = ["//support/preprocessor/testdata"],
#     tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
#     deps = [
#         ":image_preprocessor",
#         ":skia_image_preprocessor",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/status",
#         "//litert/cc:litert_layout",
#         "//litert/cc:litert_tensor_buffer",
#         "//litert/cc:litert_tensor_buffer_types",
#         "//support/util:io_types",
#         "//support/util:test_utils",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "image_preprocessor_utils",
    srcs = ["image_preprocessor_utils.cc"],
    hdrs = ["image_preprocessor_utils.h"],
    deps = [
        ":image_preprocessor",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//support/util:io_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "image_preprocessor_utils_test",
    srcs = ["image_preprocessor_utils_test.cc"],
    deps = [
        ":image_preprocessor",
        ":image_preprocessor_utils",
        "//litert/cc:litert_tensor_buffer",
        "//support/util:test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
