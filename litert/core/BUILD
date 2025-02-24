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
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [
        "//third_party/odml/infra/ml_drift_delegate/litert:__subpackages__",
        "//litert:__subpackages__",
    ],
)

cc_library(
    name = "build_stamp",
    srcs = ["build_stamp.cc"],
    hdrs = ["build_stamp.h"],
    deps = [
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
    ],
)

cc_test(
    name = "build_stamp_test",
    srcs = ["build_stamp_test.cc"],
    data = [
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":build_stamp",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/test:matchers",
    ],
)

cc_library(
    name = "dynamic_loading",
    srcs = ["dynamic_loading.cc"],
    hdrs = ["dynamic_loading.h"],
    linkopts = ["-ldl"],
    deps = [
        ":filesystem",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",  # buildcleaner: keep
    ],
)

cc_library(
    name = "insert_order_map",
    hdrs = ["insert_order_map.h"],
    deps = [
        "//third_party/absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "insert_order_map_test",
    srcs = ["insert_order_map_test.cc"],
    deps = [
        ":insert_order_map",
        "//testing/base/public:gunit_main",
    ],
)

cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = [
        "environment.h",
        "//litert/c:litert_environment.h",
    ],
    deps = [
        ":accelerator",
        "//third_party/absl/types:span",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_expected",
        "//third_party/tensorflow/lite:shared_library",
    ],
)

cc_test(
    name = "environment_test",
    srcs = ["environment_test.cc"],
    deps = [
        ":environment",
        "//testing/base/public:gunit_main",
        "//third_party/absl/types:span",
        "//litert/c:litert_any",
        "//litert/cc:litert_any",
    ],
)

cc_library(
    name = "accelerator",
    srcs = ["accelerator.cc"],
    hdrs = ["accelerator.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
    ],
)

cc_library(
    name = "filesystem",
    srcs = ["filesystem.cc"],
    hdrs = ["filesystem.h"],
    deps = [
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
    ],
)

cc_library(
    name = "dispatch_op_schema",
    srcs = ["dispatch_op_schema.cc"],
    hdrs = ["dispatch_op_schema.h"],
    copts = ["-DFLATBUFFERS_LOCALE_INDEPENDENT=0"],
    deps = [
        "//third_party/flatbuffers:runtime_cc",
        "//litert/cc:litert_buffer_ref",
    ],
)

cc_test(
    name = "filesystem_test",
    srcs = ["filesystem_test.cc"],
    deps = [
        ":filesystem",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
    ],
)

# copybara:uncomment_begin(no OSS for unique-test-directory)
# cc_test(
#     name = "dynamic_loading_test",
#     srcs = ["dynamic_loading_test.cc"],
#     tags = [
#         # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
#         "noasan",
#         "nomsan",
#         "nosan",
#     ],
#     deps = [
#         ":dynamic_loading",
#         ":filesystem",
#         "//testing/base/public:gunit_main",
#         "//third_party/absl/strings:string_view",
#         "//litert/c:litert_logging",  # buildcleaner: keep
#         "//litert/test:common",
#         "//litert/test:matchers",
#     ],
# )
# copybara:uncomment_end

cc_test(
    name = "dispatch_op_schema_test",
    srcs = ["dispatch_op_schema_test.cc"],
    deps = [
        ":dispatch_op_schema",
        "//testing/base/public:gunit_main",
    ],
)
