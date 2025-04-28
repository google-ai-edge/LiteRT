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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm/compiler:__subpackages__"],
)

cc_library(
    name = "qnn_tensor",
    srcs = ["qnn_tensor.cc"],
    hdrs = ["qnn_tensor.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/vendors/qualcomm:common",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "qnn_tensor_test",
    srcs = ["qnn_tensor_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
    ],
    deps = [
        ":qnn_tensor",
        "//litert/c:litert_model",
        "//litert/cc:litert_model",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:test_models",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_library(
    name = "qnn_op",
    srcs = ["qnn_op.cc"],
    hdrs = ["qnn_op.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_model",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "qnn_op_test",
    srcs = ["qnn_op_test.cc"],
    data = ["//litert/test:mlir_test_data"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
    ],
    deps = [
        ":qnn_op",
        "//litert/c:litert_common",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)

cc_test(
    name = "op_compatibility_test",
    srcs = ["op_compatibility_test.cc"],
    data = ["//litert/test:mlir_test_data"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
    ],
    deps = [
        ":qnn_op",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@qairt//:qnn_lib_headers",
    ],
)
