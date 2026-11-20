# Copyright 2026 Amlogic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Workspace: one fetch of amlnn-toolkit nn_runtime (nnsdk + aml_compiler_core)."""

load("//litert/sdk_util:repo.bzl", "configurable_repo")

# Pin Amlogic-NN/amlnn-toolkit. Bump COMMIT + strip_prefix together.
_AMLNN_TOOLKIT_COMMIT = "5efa2cd45971766bb7cb1f2ddab727e3ce301463"

def aml_npu_nn_runtime():
    """Fetch nn_runtime into @aml_npu_nn_runtime (nnsdk + aml_compiler_core).

    Same pattern as qairt: one URL, optional local override.

    Local override: LITERT_AML_NN_RUNTIME = absolute path to nn_runtime root
    that contains both nnsdk/ and aml_compiler_core/ (as on GitHub).
    """
    configurable_repo(
        name = "aml_npu_nn_runtime",
        build_file = "@//third_party/aml_nn_runtime:aml_nn_runtime.BUILD",
        local_path_env = "LITERT_AML_NN_RUNTIME",
        strip_prefix = "amlnn-toolkit-%s/amlnn_runtime/nn_runtime" % _AMLNN_TOOLKIT_COMMIT,
        url = "https://github.com/Amlogic-NN/amlnn-toolkit/archive/%s.tar.gz" % _AMLNN_TOOLKIT_COMMIT,
        file_extension = "tar.gz",
    )
