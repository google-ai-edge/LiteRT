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
load(
    "//litert/build_common:litert_build_defs.bzl",
    "litert_cc_library",
    "litert_cc_test",
)
load("//tflite:build_def.bzl", "gen_selected_ops")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//litert:litert_public",
    ],
    features = ["allow_starlark_transitions"],
)

cc_library(
    name = "selected_ops_registry",
    srcs = ["selected_ops_registry.cc"],
    hdrs = ["selected_ops_registry.h"],
)

cc_library(
    name = "op_resolver",
    srcs = ["op_resolver.cc"],
    hdrs = ["op_resolver.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_op_resolver",
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_options",
        "//litert/cc/internal:litert_runtime_proxy_api",
        "//tflite:mutable_op_resolver",
        "//tflite/c:common",
        "//tflite/schema:schema_fbs",
    ],
)

gen_selected_ops(
    name = "simple_model_selected_ops",
    model = [
        "//litert/test:testdata/simple_model.tflite",
    ],
)

cc_test(
    name = "op_resolver_test",
    srcs = [
        "op_resolver_test.cc",
        ":simple_model_selected_ops",
    ],
    data = [
        "//litert/test:testdata/simple_model.tflite",
        "//litert/test:testdata/simple_model_custom_op.tflite",
    ],
    deps = [
        ":op_resolver",
        "//litert/c:litert_common",
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "//tflite:framework",
        "//tflite:mutable_op_resolver",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_cc_test(
    name = "op_resolver_selective_test",
    srcs = [
        "op_resolver_test.cc",
        ":simple_model_selected_ops",
    ],
    build_include = "cpu_only",
    cpu_backend = "selective",
    data = [
        "//litert/test:testdata/simple_model.tflite",
        "//litert/test:testdata/simple_model_custom_op.tflite",
    ],
    tags = [
        "noios",
        "nomacos",
        "nowindows",
    ],
    deps = [
        ":op_resolver",
        "//litert/c:litert_common",
        "//litert/c/options:litert_cpu_options",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "//tflite:framework",
        "//tflite:mutable_op_resolver",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_cc_library(
    name = "simple_model_auto_selected_ops_lib",
    testonly = True,
    models = [
        "//litert/test:testdata/simple_model.tflite",
    ],
)

litert_cc_test(
    name = "selected_ops_auto_register_test",
    srcs = ["selected_ops_auto_register_test.cc"],
    build_include = "cpu_only",
    cpu_backend = "selective",
    data = [
        "//litert/test:testdata/simple_model.tflite",
        "//litert/test:testdata/transformer.tflite",
    ],
    models = [
        "//litert/test:testdata/transformer.tflite",
    ],
    tags = [
        "noios",
        "nomacos",
        "nowindows",
    ],
    deps = [
        ":op_resolver",
        ":simple_model_auto_selected_ops_lib",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "//tflite:mutable_op_resolver",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
