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
load("//litert/build_common:litert_build_defs.bzl", "litert_lib", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

litert_lib(
    name = "adapter",
    srcs = ["adapter.cc"],
    hdrs = ["adapter.h"],
    data = [
        # copybara:uncomment_begin(google-only)
        # "//platforms/darwinn/compiler:compiler_api_wrapper",
        # copybara:uncomment_end
    ],
    linkopts = [
        # TODO(abhirs): Make this work for OS.
        #copybara:comment_begin(google-only)
        make_rpaths(["platforms/darwinn/compiler"]),
        # copybara:uncomment_end
    ],
    tags = [
        # Don't build/test in OS until libcompiler_api_wrapper.so is available.
        "nobuilder",
        "no_oss",
    ],
    ungrte = False,
    deps = [
        "//third_party/absl/log",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/core:dynamic_loading",
    ],
)

cc_test(
    name = "adapter_test",
    srcs = ["adapter_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    linkstatic = 1,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        #Don't build/test in OS until libcompiler_api_wrapper.so is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that uses dlopen.
        "nosan",
    ],
    # This test can only be run on Android and Linux.
    target_compatible_with = select({
        "//third_party/bazel_platforms/os:android": [],
        "//third_party/bazel_platforms/os:linux": [],
        "//conditions:default": ["//third_party/bazel_platforms:incompatible"],
    }),
    deps = [
        ":adapter",
        "//testing/base/public:gunit_main",
        "//litert/c:litert_logging",
    ],
)
