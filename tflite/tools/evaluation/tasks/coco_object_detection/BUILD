# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

py_binary(
    name = "preprocess_coco_minival",
    srcs = ["preprocess_coco_minival.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/tools/evaluation/proto:evaluation_stages_py",
        "@absl_py//absl/logging",
    ],
)

cc_library(
    name = "run_eval_lib",
    srcs = ["run_eval.cc"],
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tflite/core/c:common",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "//tflite/tools/evaluation:evaluation_delegate_provider",
        "//tflite/tools/evaluation:evaluation_stage",
        "//tflite/tools/evaluation:utils",
        "//tflite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tflite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//tflite/tools/evaluation/stages:object_detection_stage",
        "//tflite/tools/evaluation/tasks:task_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "run_eval",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":run_eval_lib",
        "//tflite/tools/evaluation/tasks:task_executor_main",
    ],
)
