# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# copybara:uncomment load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

proto_library(
    name = "evaluation_proto",
    srcs = ["evaluation.proto"],
)

cc_proto_library(
    name = "evaluation_cc_proto",
    deps = [":evaluation_proto"],
)

py_proto_library(
    name = "evaluation_py_proto",
    deps = [":evaluation_proto"],
)

cc_library(
    name = "data",
    srcs = [
        "data.cc",
        "image.cc",
    ],
    hdrs = [
        "data.h",
        "image.h",
    ],
    copts = ["-fno-exceptions"],
    deps = [
        ":evaluation_cc_proto",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@stblib//:stb_image_hdrs",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
    ],
)

cc_library(
    name = "execution",
    srcs = ["execution.cc"],
    hdrs = ["execution.h"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c/options:litert_cpu_options",
        "//litert/c/options:litert_intel_openvino_options",
        "//litert/cc:litert_buffer_ref",  # buildcleaner: keep
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model_types",
        "//litert/cc:litert_options",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc/options:litert_arm_vulkan_ml_options",
        "//litert/cc/options:litert_cpu_options",
        "//litert/cc/options:litert_gpu_options",
        "//litert/cc/options:litert_intel_openvino_options",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "run_lock",
    srcs = ["run_lock.cc"],
    hdrs = ["run_lock.h"],
    copts = ["-fno-exceptions"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
    ],
)

cc_library(
    name = "runner",
    srcs = ["runner.cc"],
    hdrs = ["runner.h"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        ":evaluation_cc_proto",
        ":execution",
        ":metrics",
        ":run_lock",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "image_test",
    size = "small",
    srcs = ["image_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        "//litert/c:litert_model_types",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:types",
        "@XNNPACK",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "data_test",
    size = "small",
    srcs = ["data_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        "//litert/cc:litert_expected",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        ":metrics",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "execution_test",
    size = "small",
    srcs = ["execution_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        ":execution",
        "//litert/c:litert_common",
        "//litert/c/options:litert_cpu_options",
        "//litert/c/options:litert_gpu_options",
        "//litert/c/options:litert_intel_openvino_options",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc/options:litert_arm_vulkan_ml_options",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "run_lock_test",
    size = "small",
    srcs = ["run_lock_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        ":run_lock",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "runner_test_util",
    testonly = True,
    srcs = ["runner_test_util.cc"],
    hdrs = ["runner_test_util.h"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        ":evaluation_cc_proto",
        ":runner",
        "//litert/cc:litert_expected",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "runner_test",
    size = "small",
    srcs = ["runner_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [
        ":data",
        ":evaluation_cc_proto",
        ":run_lock",
        ":runner",
        ":runner_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "vision_eval_for_test",
    testonly = True,
    srcs = ["//litert/tools:vision_eval_main.cc"],
    copts = ["-fno-exceptions"],
    deps = [":runner"],
)

# The NumPy implementation is a test oracle for the original dense-task
# protocols. It is not linked or invoked by the evaluator.
py_library(
    name = "metrics_reference",
    testonly = True,
    srcs = ["metrics_reference.py"],
    deps = ["//third_party/py/numpy"],
)

py_test(
    name = "metrics_parity_test",
    size = "small",
    srcs = ["metrics_parity_test.py"],
    data = [":vision_eval_for_test"],
    env = {"LITERT_VISION_EVAL_BIN": "$(rootpath :vision_eval_for_test)"},
    deps = [
        ":evaluation_py_proto",
        ":metrics_reference",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

# copybara:uncomment_begin(google-only)
# py_library(
#     name = "export_imagenet_val_jpegs",
#     srcs = ["export_imagenet_val_jpegs.py"],
#     deps = [
#         "@absl_py//absl:app",
#         "@absl_py//absl/flags",
#         "//third_party/py/pyarrow",
#     ],
# )
#
# py_test(
#     name = "export_imagenet_val_jpegs_test",
#     size = "small",
#     srcs = ["export_imagenet_val_jpegs_test.py"],
#     deps = [
#         ":export_imagenet_val_jpegs",
#         "@absl_py//absl/testing:absltest",
#         "//third_party/py/pyarrow",
#     ],
# )
#
# py_library(
#     name = "prepare_coco2017",
#     srcs = ["prepare_coco2017.py"],
#     deps = [
#         ":evaluation_py_proto",
#         "//third_party/py/PIL:pil",
#         "@absl_py//absl:app",
#         "@absl_py//absl/flags",
#         "@com_google_protobuf//:protobuf_python",
#         "//third_party/py/numpy",
#         "//third_party/py/pycocotools",
#     ],
# )
#
# py_test(
#     name = "prepare_coco2017_test",
#     size = "small",
#     srcs = ["prepare_coco2017_test.py"],
#     deps = [
#         ":evaluation_py_proto",
#         ":prepare_coco2017",
#         "@absl_py//absl/testing:absltest",
#         "//third_party/py/PIL:pil",
#         "@com_google_protobuf//:protobuf_python",
#         "//third_party/py/numpy",
#     ],
# )
#
# py_library(
#     name = "prepare_voc2012",
#     srcs = ["prepare_voc2012.py"],
#     deps = [
#         ":evaluation_py_proto",
#         "//third_party/py/PIL:pil",
#         "@absl_py//absl:app",
#         "@absl_py//absl/flags",
#         "@com_google_protobuf//:protobuf_python",
#         "//third_party/py/numpy",
#         "//third_party/py/pyarrow",
#     ],
# )
#
# py_test(
#     name = "prepare_voc2012_test",
#     size = "small",
#     srcs = ["prepare_voc2012_test.py"],
#     deps = [
#         ":evaluation_py_proto",
#         ":prepare_voc2012",
#         "@absl_py//absl/testing:absltest",
#         "//third_party/py/PIL:pil",
#         "@com_google_protobuf//:protobuf_python",
#         "//third_party/py/numpy",
#         "//third_party/py/pyarrow",
#     ],
# )
#
# py_library(
#     name = "prepare_tipsv2",
#     srcs = ["prepare_tipsv2.py"],
#     deps = [
#         ":evaluation_py_proto",
#         "//third_party/py/PIL:pil",
#         "@absl_py//absl:app",
#         "@absl_py//absl/flags",
#         "@com_google_protobuf//:protobuf_python",
#         "//third_party/py/numpy",
#         "//third_party/py/pyarrow",
#     ],
# )
#
# py_test(
#     name = "prepare_tipsv2_test",
#     size = "small",
#     srcs = ["prepare_tipsv2_test.py"],
#     deps = [
#         ":evaluation_py_proto",
#         ":prepare_tipsv2",
#         "@absl_py//absl/testing:absltest",
#         "//third_party/py/PIL:pil",
#         "@com_google_protobuf//:protobuf_python",
#         "//third_party/py/numpy",
#         "//third_party/py/pyarrow",
#     ],
# )
# copybara:uncomment_end
