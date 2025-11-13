load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//ci/tools/python/wheel/utils:platforms.bzl", "get_wheel_platform_name")
load("//ci/tools/python/wheel/utils:py_manylinux_compliance_test.bzl", "verify_manylinux_compliance_test")
load("//ci/tools/python/wheel/utils:py_wheel.bzl", "py_wheel")
load("//litert:litert.bzl", "MANYLINUX_LEVEL")
load("//litert:version.bzl", "LITERT_EXPERIMENTAL_VERSION")

package(
    default_visibility = ["//ci/tools/python:__subpackages__"],
)

string_flag(
    name = "nightly_iso_date",
    build_setting_default = "",
)

genrule(
    name = "setup_py",
    srcs = ["//ci/tools/python/wheel:setup_with_binary.py"],
    outs = ["setup.py"],
    cmd = "cat $< > $@",
)

genrule(
    name = "fixed_pydeps",
    srcs = [
        "//tflite/python/metrics:metrics_portable.py",
        "//tflite/python:interpreter",
    ],
    outs = [
        "metrics_portable.py",
        "interpreter.py",
    ],
    cmd = "for f in $(SRCS); do sed -e 's/tflite_runtime/ai_edge_litert/g' $$f > $(RULEDIR)/$$(basename $$f); done",
)

# Rename the namespace in the generated proto files to ai_edge_litert.
# This is required to maintain dependency between the two protos.
genrule(
    name = "fixed_protos",
    srcs = ["//tflite/profiling/proto:model_runtime_info_py"],
    outs = ["model_runtime_info_pb2.py"],
    cmd = "for f in $(SRCS); do sed -e 's/tflite\\.profiling\\.proto/ai_edge_litert/g' $$f > $(RULEDIR)/$$(basename $$f); done",
)

# This includes the dependencies that lives in litert/python/
# TODO(weiyiw): Walk the dependencies and pull in related files.
# TODO(weiyiw): Support data files.
ALL_PY_SRC_MODULES = select({
    "@org_tensorflow//tensorflow:linux_x86_64": [
        "//litert/python/aot:prepare_for_npu",
        "//litert/python/aot:aot_compile",
        "//litert/python/aot/ai_pack:export_lib",
        "//litert/python/aot/core:common",
        "//litert/python/aot/core:types",
        "//litert/python/aot/core:apply_plugin",
        "//litert/python/aot/core:mlir_transforms",
        "//litert/python/aot/core:components",
        "//litert/python/aot/core:tflxx_util",
        "//litert/python/aot/vendors:import_vendor",
        "//litert/python/aot/vendors",
        "//litert/python/aot/vendors:fallback_backend",
        "//litert/python/aot/vendors/example:example_backend",
        "//litert/python/aot/vendors/google_tensor:google_tensor_backend",
        "//litert/python/aot/vendors/google_tensor:target",
        "//litert/python/aot/vendors/mediatek:mediatek_backend",
        "//litert/python/aot/vendors/mediatek:target",
        "//litert/python/aot/vendors/qualcomm:qualcomm_backend",
        "//litert/python/aot/vendors/qualcomm:target",
        "//litert/python/mlir",
        "//litert/python/mlir/_mlir_libs",
        # TODO(b/445163709): Remove the following module once litert_lm publishes a pypi package.
        "//litert/python/internal:litertlm_core",
        "//litert/python/internal:litertlm_header_schema_py",
        "//litert/python/internal:llm_metadata_py_pb2",
        "//litert/python/internal:sampler_params_py_pb2",
        "//litert/python/internal:litertlm_builder",
    ],
    "//conditions:default": [],
})

# This includes the dependencies that lives outside of litert/python/. e.g. litert/tools/
PACKAGE_DATA = select({
    "@org_tensorflow//tensorflow:linux_x86_64": [
        "//litert/tools:apply_plugin_main",
        "//litert/vendors/google_tensor/compiler:google_tensor_compiler_plugin_so",
        "//litert/vendors/mediatek/compiler:compiler_plugin_so",
        "//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so",
    ],
    "//conditions:default": [],
})

py_wheel(
    name = "litert_wheel",
    srcs = [
        ":fixed_protos",
        ":fixed_pydeps",
        "//litert/c:libLiteRtRuntimeCApi.so",
        "//litert/python:pywrap_litert_binaries",
        "//litert/python/litert_wrapper/compiled_model_wrapper:compiled_model",
        "//litert/python/litert_wrapper/tensor_buffer_wrapper:tensor_buffer",
        "//litert/python/mlir/_mlir_libs:libLiteRTCompilerMLIR.so",
        "//tflite/profiling/proto:profiling_info_py",
        "//tflite/python:schema_py",
        "//tflite/python/metrics:metrics_interface",
    ],
    nightly_suffix = ":nightly_iso_date",
    package_data = PACKAGE_DATA,
    platform_name = get_wheel_platform_name(),
    project_name = "ai_edge_litert",
    py_srcs = ALL_PY_SRC_MODULES,
    setup_py = ":setup_py",
    # NOTE: There is no plan to release more 1.x versions. But if that's needed in the future,
    # LITERT_FULL_VERSION should be used here instead.
    version = LITERT_EXPERIMENTAL_VERSION,
)

verify_manylinux_compliance_test(
    name = "manylinux_compliance_test",
    aarch64_compliance_tag = MANYLINUX_LEVEL + "_aarch64",
    ppc64le_compliance_tag = MANYLINUX_LEVEL + "_ppc64le",
    wheel = ":litert_wheel",
    x86_64_compliance_tag = MANYLINUX_LEVEL + "_x86_64",
)
