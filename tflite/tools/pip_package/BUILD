load("@org_tensorflow//tensorflow/tools/pip_package/utils:data_deps.bzl", "collect_data_files")
load("@org_tensorflow//tensorflow/tools/pip_package/utils:py_deps.bzl", "transitive_py_deps")
load("@org_tensorflow//tensorflow/tools/pip_package/utils:tf_wheel.bzl", "tf_wheel")

package(default_visibility = ["//visibility:private"])

collect_data_files(
    name = "cc_deps",
    deps = [
        "//tflite/experimental/genai:pywrap_genai_ops",
        "//tflite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper",
    ],
)

genrule(
    name = "setup_py",
    srcs = ["//tflite/tools/pip_package:setup_with_binary.py"],
    outs = ["setup.py"],
    cmd = "cat $< > $@",
)

transitive_py_deps(
    name = "py_deps",
    deps = [
        "//tflite/profiling/proto:model_runtime_info_py",
        "//tflite/profiling/proto:profiling_info_py",
        "//tflite/python:schema_py",
    ],
)

tf_wheel(
    name = "wheel",
    platform_name = select({
        "@platforms//os:osx": "macosx",
        "@platforms//os:macos": "macosx",
        "@platforms//os:windows": "win",
        "@platforms//os:linux": "linux",
    }),
    platform_tag = select({
        "@platforms//cpu:aarch64": "arm64",
        "@platforms//cpu:arm64": "arm64",
        "@platforms//cpu:x86_64": "x86_64",
        "@platforms//cpu:ppc": "ppc64le",
    }),
    source_files = [
        "//tflite/tools/pip_package:MANIFEST.in",
        ":setup_py",
        ":cc_deps",
        ":py_deps",
    ],
)
