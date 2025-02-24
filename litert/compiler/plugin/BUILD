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
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["compiler_plugin.h"],
    deps = [
        ":algo",
        "//third_party/absl/log:absl_check",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/core:build_stamp",
        "//litert/core:dynamic_loading",
        "//litert/core:environment",
        "//litert/core:filesystem",
        "//litert/core/model",
        "//litert/core/model:buffer_manager",
        "//litert/core/model:ir_allocator",
        "//litert/core/util:flatbuffer_tools",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)

# copybara:uncomment_begin(no OSS for unique-test-directory)
# cc_test(
#     name = "compiler_plugin_test",
#     srcs = ["compiler_plugin_test.cc"],
#     data = [
#         "//litert/test:mlir_test_data",
#         "//litert/vendors/examples:example_plugin_so",
#     ],
#     tags = [
#         # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
#         "noasan",
#         "nomsan",
#         "nosan",
#         "notsan",
#     ],
#     deps = [
#         ":compiler_plugin",
#         "//testing/base/public:gunit_main",
#         "//third_party/absl/strings:string_view",
#         "//litert/c:litert_common",
#         "//litert/c:litert_model",
#         "//litert/c:litert_op_code",
#         "//litert/cc:litert_environment",
#         "//litert/core:build_stamp",
#         "//litert/core:filesystem",
#         "//litert/core/model",
#         "//litert/test:common",
#         "//litert/tools:dump",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "algo",
    srcs = ["algo.cc"],
    hdrs = ["algo.h"],
    deps = [
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/log:absl_check",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/core:insert_order_map",
        "//litert/core/model",
        "//litert/core/model:model_graph",
    ],
)

cc_test(
    name = "algo_test",
    srcs = ["algo_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
    ],
    deps = [
        ":algo",
        "//testing/base/public:gunit_main",
        "//third_party/absl/container:flat_hash_set",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/core/model",
        "//litert/core/model:graph_validation",
        "//litert/test:common",
    ],
)
