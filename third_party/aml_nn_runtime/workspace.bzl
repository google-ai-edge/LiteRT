# Copyright 2026 Amlogic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Workspace: one fetch of aml_compiler_core + nnsdk from amlnn-toolkit nn_runtime."""

# Pin Amlogic-NN/amlnn-toolkit main. Bump COMMIT, strip_prefix, and sha256 together.
# Source tree: amlnn_runtime/nn_runtime/{aml_compiler_core,nnsdk}
_AML_TOOLKIT_COMMIT = "11410d28ff079656e13a702b9ed8e8ed7d7f381b"
_AML_TOOLKIT_SHA256 = "49dd665d1d1e51f7a701b8881f3c9b77356136a3ea15c125a3329e4bfca82b09"

def _aml_nn_runtime_impl(ctx):
    local = ctx.getenv("LITERT_AML_NN_RUNTIME")
    if local:
        host_os = ctx.os.name.lower()
        if "windows" not in host_os and not local.startswith("/"):
            fail("LITERT_AML_NN_RUNTIME must be an absolute path.")
        sdk = ctx.path(local)
        if not sdk.is_dir:
            fail("LITERT_AML_NN_RUNTIME is not a directory: %s" % local)
        for child in sdk.readdir():
            ctx.symlink(child, child.basename)
    elif ctx.getenv("LITERT_FETCH_AML_SDK") == "1":
        ctx.download_and_extract(
            url = ctx.attr.url,
            sha256 = ctx.attr.sha256,
            stripPrefix = ctx.attr.strip_prefix,
        )
    else:
        fail(
            "AML nn_runtime is not configured, so this LiteRT build does not " +
            "download amlnn-toolkit. Set LITERT_AML_NN_RUNTIME to a directory " +
            "with nnsdk/ and aml_compiler_core/, or set LITERT_FETCH_AML_SDK=1 " +
            "to download it.",
        )
    ctx.delete("BUILD.bazel")
    ctx.symlink(ctx.attr.build_file, "BUILD.bazel")

_aml_nn_runtime_repo = repository_rule(
    implementation = _aml_nn_runtime_impl,
    environ = [
        "LITERT_AML_NN_RUNTIME",
        "LITERT_FETCH_AML_SDK",
    ],
    attrs = {
        "build_file": attr.label(allow_single_file = True),
        "sha256": attr.string(),
        "strip_prefix": attr.string(),
        "url": attr.string(),
    },
)

def aml_npu_nn_runtime():
    """Register @aml_npu_nn_runtime. Download only when a target needs it.

    The archive is not fetched for an ordinary LiteRT build.
    LITERT_AML_NN_RUNTIME = absolute path containing nnsdk/ and
    aml_compiler_core/. LITERT_FETCH_AML_SDK=1 downloads amlnn-toolkit and
    keeps amlnn_runtime/nn_runtime.
    """
    _aml_nn_runtime_repo(
        name = "aml_npu_nn_runtime",
        build_file = "@//third_party/aml_nn_runtime:aml_nn_runtime.BUILD",
        sha256 = _AML_TOOLKIT_SHA256,
        strip_prefix = "amlnn-toolkit-%s/amlnn_runtime/nn_runtime" % _AML_TOOLKIT_COMMIT,
        url = "https://github.com/Amlogic-NN/amlnn-toolkit/archive/%s.tar.gz" % _AML_TOOLKIT_COMMIT,
    )
