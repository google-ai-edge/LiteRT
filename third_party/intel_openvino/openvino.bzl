def _openvino_native_impl(repository_ctx):
    openvino_native_dir = repository_ctx.os.environ["OPENVINO_NATIVE_DIR"]
    repository_ctx.symlink(openvino_native_dir, "openvino")
    build_file_content = repository_ctx.read(repository_ctx.attr.build_file)
    repository_ctx.file("BUILD", build_file_content)

openvino_configure = repository_rule(
    implementation = _openvino_native_impl,
    local = True,
    environ = ["OPENVINO_NATIVE_DIR"],
    attrs = {
        # Define an attribute to hold the label of the external BUILD file content
        "build_file": attr.label(
            doc = "The label of the BUILD file content to be written.",
            allow_single_file = True, # This attribute expects a single file
            mandatory = True,
        ),
    },
)
