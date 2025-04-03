load("//ci/tools/python/wheel/utils:py_manylinux_compliance_test.bzl", "verify_manylinux_compliance_test")
load("//ci/tools/python/wheel/utils:py_wheel.bzl", "py_wheel")

package(
    default_visibility = ["//visibility:private"],
)

MANYLINUX_X86_64_TAG = "manylinux_2_17_x86_64"

genrule(
    name = "setup_py",
    srcs = ["//ci/tools/python/wheel:setup_with_binary.py"],
    outs = ["setup.py"],
    cmd = "cat $< > $@",
)

py_wheel(
    name = "litert_wheel",
    srcs = [
        "//tflite/experimental/genai:pywrap_genai_ops.so",
        "//tflite/profiling/proto:model_runtime_info_py",
        "//tflite/profiling/proto:profiling_info_py",
        "//tflite/python:interpreter",
        "//tflite/python:schema_py",
        "//tflite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper.so",
        "//tflite/python/metrics:metrics_interface",
        "//tflite/python/metrics:metrics_portable.py",
    ],
    platform_name = MANYLINUX_X86_64_TAG,
    setup_py = ":setup_py",
    version = "1.1.2",
)

verify_manylinux_compliance_test(
    name = "manylinux_compliance_test",
    aarch64_compliance_tag = "manylinux_2_17_aarch64",
    ppc64le_compliance_tag = "manylinux_2_17_ppc64le",
    wheel = ":litert_wheel",
    x86_64_compliance_tag = MANYLINUX_X86_64_TAG,
)
