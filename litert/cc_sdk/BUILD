load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@bazel_skylib//rules:diff_test.bzl", "diff_test")

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
load("//litert/build_common:special_rule.bzl", "litert_sdk_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

# Every file of the LiteRT C++ SDK, collected from the packages that own them.
filegroup(
    name = "litert_cc_sdk_files",
    srcs = [
        "//litert/build_common:litert_cc_sdk_files",
        "//litert/c:litert_cc_sdk_files",
        "//litert/c/internal:litert_cc_sdk_files",
        "//litert/c/options:litert_cc_sdk_files",
        "//litert/cc:litert_cc_sdk_files",
        "//litert/cc/internal:litert_cc_sdk_files",
        "//litert/cc/options:litert_cc_sdk_files",
        "//litert/core:litert_cc_sdk_files",
    ],
    visibility = litert_sdk_friends(),
)

# ------------------------------------------------------------------------------
# Tests for the LiteRT C++ SDK file lists.

# The prefix where the LiteRT source tree is located.
_LITERT_ROOT_PREFIX = package_name().removesuffix("litert/cc_sdk")

# Makes a sorted list of the SDK files that end with `suffix`, relative to the LiteRT root. This is
# the same format as litert_cc_sdk_hdrs.txt and litert_cc_sdk_srcs.txt.
[
    genrule(
        name = "generated_litert_cc_sdk_{}".format(kind),
        testonly = True,
        srcs = [":litert_cc_sdk_files"],
        outs = ["generated_litert_cc_sdk_{}.txt".format(kind)],
        cmd = """
        for f in $(SRCS); do
          case "$$f" in
            *{suffix}) echo "$${{f#{prefix}}}" ;;
          esac
        done | LC_ALL=C sort -u > $@
        """.format(
            prefix = _LITERT_ROOT_PREFIX,
            suffix = suffix,
        ),
    )
    for kind, suffix in [
        ("hdrs", ".h"),
        ("srcs", ".cc"),
    ]
]

# Makes sure that the headers in the SDK filegroups and in litert_cc_sdk_hdrs.txt are the same.
diff_test(
    name = "litert_cc_sdk_hdrs_test",
    failure_message = "Update litert_cc_sdk_hdrs.txt with the output of " +
                      ":generated_litert_cc_sdk_hdrs.",
    file1 = ":generated_litert_cc_sdk_hdrs",
    file2 = "litert_cc_sdk_hdrs.txt",
)

# Makes sure that the sources in the SDK filegroups and in litert_cc_sdk_srcs.txt are the same.
diff_test(
    name = "litert_cc_sdk_srcs_test",
    failure_message = "Update litert_cc_sdk_srcs.txt with the output of " +
                      ":generated_litert_cc_sdk_srcs.",
    file1 = ":generated_litert_cc_sdk_srcs",
    file2 = "litert_cc_sdk_srcs.txt",
)

# A source file that includes all the SDK headers, so that each header is compiled at least once.
genrule(
    name = "litert_cc_sdk_include_all_hdrs",
    testonly = True,
    srcs = [":litert_cc_sdk_files"],
    outs = ["litert_cc_sdk_include_all_hdrs.cc"],
    cmd = """
    for f in $(SRCS); do
      case "$$f" in
        *.h) echo "#include \\"$$f\\"" ;;
      esac
    done > $@
    """,
)

# Compiles the SDK files alone. If an SDK file includes a LiteRT file that is not in
# :litert_cc_sdk_files, the build fails. Do not add LiteRT targets to deps. The deps must be only
# the external dependencies of the SDK: absl, flatbuffers and some TFLite headers.
# This library is only compiled, nothing links it.
# buildifier: disable=no-hdrs-no-alwayslink
cc_library(
    name = "litert_cc_sdk_compile_test_lib",
    testonly = True,
    srcs = [
        ":litert_cc_sdk_files",
        ":litert_cc_sdk_include_all_hdrs",
    ],
    tags = ["nofixdeps"],  # The SDK files are srcs, not deps.
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        # The generated build_config.h. CMake makes it from build_config.h.in.
        "//litert/build_common:build_config",
        # Headers from the TFLite repo that the SDK uses. They are not part of :litert_cc_sdk_files.
        "//tflite/converter:allocation",
        "//tflite/c:c_api_types",
    ],
)

build_test(
    name = "litert_cc_sdk_compile_test",
    targets = [":litert_cc_sdk_compile_test_lib"],
)
