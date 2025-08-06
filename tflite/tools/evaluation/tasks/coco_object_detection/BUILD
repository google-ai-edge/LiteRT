load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_binary")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

py_strict_binary(
    name = "preprocess_coco_minival",
    srcs = ["preprocess_coco_minival.py"],
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
