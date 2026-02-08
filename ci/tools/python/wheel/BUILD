load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//ci/tools/python/wheel/utils:platforms.bzl", "get_wheel_platform_name")
load("//ci/tools/python/wheel/utils:py_manylinux_compliance_test.bzl", "verify_manylinux_compliance_test")
load("//ci/tools/python/wheel/utils:py_wheel.bzl", "py_wheel")
load("//litert:litert.bzl", "MANYLINUX_LEVEL")
load("//litert:version.bzl", "LITERT_EXPERIMENTAL_VERSION")
load("//litert/build_common:special_rule.bzl", "litert_gpu_accelerator_prebuilts")

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

genrule(
    name = "relocated_tflite_protos",
    srcs = [
        "//tflite/converter:converter_flags_proto_py",
        "//tflite/converter:model_flags_proto_py",
        "//tflite/converter:types_proto_py",
        "//tflite/converter/debug:debug_options_proto_py",
    ],
    outs = [
        "ai_edge_litert/tflite/converter/converter_flags_pb2.py",
        "ai_edge_litert/tflite/converter/model_flags_pb2.py",
        "ai_edge_litert/tflite/converter/types_pb2.py",
        "ai_edge_litert/tflite/converter/debug/debug_options_pb2.py",
    ],
    cmd = select({
        "@bazel_tools//src/conditions:windows": """powershell -Command "
            \\$$srcs = '$(SRCS)'.Split(' ');
            \\$$base_out = '$(RULEDIR)/ai_edge_litert/tflite/converter';
            New-Item -Path \\$$base_out/debug -ItemType Directory -Force | Out-Null;
            foreach (\\$$f in \\$$srcs) {
                \\$$fname = Split-Path \\$$f -Leaf;
                if (\\$$fname -eq 'debug_options_pb2.py') {
                    \\$$out_dir = Join-Path \\$$base_out 'debug'
                } else {
                    \\$$out_dir = \\$$base_out
                };
                \\$$out_path = Join-Path \\$$out_dir \\$$fname;
                (Get-Content \\$$f -Raw).Replace('tflite.converter', 'ai_edge_litert.tflite.converter') | Set-Content \\$$out_path -Encoding UTF8
            }
        " """,
        "//conditions:default": """
            mkdir -p $(RULEDIR)/ai_edge_litert/tflite/converter/debug
            for f in $(SRCS); do
                fname=$$(basename $$f)
                if [ "$$fname" == "debug_options_pb2.py" ]; then
                    out_dir=$(RULEDIR)/ai_edge_litert/tflite/converter/debug
                else
                    out_dir=$(RULEDIR)/ai_edge_litert/tflite/converter
                fi
                sed 's/tflite\\.converter/ai_edge_litert.tflite.converter/g' $$f > $$out_dir/$$fname
            done
        """,
    }),
)

py_library(
    name = "tflite_converter_protos",
    srcs = [":relocated_tflite_protos"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = ["@com_google_protobuf//:protobuf_python"],
)

# This includes the dependencies that lives in litert/python/
# TODO(weiyiw): Walk the dependencies and pull in related files.
# TODO(weiyiw): Support data files.
ALL_PY_SRC_MODULES = [
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
    # TODO(b/460805332): stop building mlir until protobuf symbol conflict is resolved.
    # "//litert/python/mlir",
    # "//litert/python/tools/model_utils",
    # TODO(b/445163709): Remove the following module once litert_lm publishes a pypi package.
    "//litert/python/internal:litertlm_core",
    "//litert/python/internal:litertlm_header_schema_py",
    "//litert/python/internal:llm_metadata_py_pb2",
    "//litert/python/internal:sampler_params_py_pb2",
    "//litert/python/internal:litertlm_builder",
    "//litert/python/tools:flatbuffer_utils",
]

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

LITERT_RUNTIME_SHARED_LIB = select({
    "@bazel_tools//src/conditions:windows": [
        "//litert/c:litert_runtime_c_api_dll",
    ],
    "//conditions:default": ["//litert/c:libLiteRt.so"],
}) + select({
    "@platforms//os:macos": [
        "@litert_prebuilts//:macos_arm64/libLiteRt.dylib",
    ],
    "//conditions:default": [],
})

py_wheel(
    name = "litert_wheel",
    srcs = [
        ":fixed_protos",
        ":fixed_pydeps",
    ] + LITERT_RUNTIME_SHARED_LIB + [
        "//litert/python:pywrap_litert_binaries",
        "//litert/python/litert_wrapper/compiled_model_wrapper:compiled_model",
        "//litert/python/litert_wrapper/tensor_buffer_wrapper:tensor_buffer",
        # TODO(b/460805332): stop building converter until protobuf symbol conflict is resolved.
        # "//litert/python/mlir/_mlir_libs:libLiteRTCompilerMLIR.so",
        "//tflite/profiling/proto:profiling_info_py",
        "//tflite/python:schema_py",
        "//tflite/python/metrics:metrics_interface",
    ] + litert_gpu_accelerator_prebuilts(),
    nightly_suffix = ":nightly_iso_date",
    package_data = PACKAGE_DATA,
    platform_name = get_wheel_platform_name(),
    project_name = "ai_edge_litert",
    py_srcs = ALL_PY_SRC_MODULES + [
        ":relocated_tflite_protos",
    ],
    setup_py = ":setup_py",
    structured_deps = [],
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
