def _openvino_native_impl(repository_ctx):
    openvino_native_dir = repository_ctx.os.environ["OPENVINO_NATIVE_DIR"]
    repository_ctx.symlink(openvino_native_dir, "openvino")
    repository_ctx.file("BUILD", """
cc_library(
    name = "openvino",
    hdrs = glob(["openvino/runtime/include", "openvino/runtime/include/ie/cpp", "openvino/runtime/include/ie"]),
    srcs = ["openvino/runtime/lib/intel64/libopenvino.so"],
    includes = ["openvino/runtime/include/ie/cpp",
                "openvino/runtime/include/ie",
                "openvino/runtime/include"],
    visibility = ["//visibility:public"],
)
    """)

openvino_configure = repository_rule(
    implementation = _openvino_native_impl,
    local = True,
    environ = ["OPENVINO_NATIVE_DIR"],
)
