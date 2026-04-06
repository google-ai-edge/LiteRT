# Copyright (C) 2026 Samsung Electronics Co. LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build definitions for Samsung backend."""

load("//litert/build_common:litert_build_defs.bzl", "append_rule_kwargs", "litert_bin", "litert_lib", "make_rpaths")

# Samsung AI LiteCore libraries for x86_64-linux
_LITECORE_LIBS_X86_64 = [
    "@exynos_ai_litecore//:lib/x86_64-linux/libgraph_wrapper.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libgraphgen.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libgraphgen_api.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libcompiler_api_lib.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libnpu_compiler.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libcommon.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libir.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libisa.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libsnc_api.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libncp_wrapper_lib.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libsait_npu_compiler_lib.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libveriben_lib.so",
    "@exynos_ai_litecore//:lib/x86_64-linux/libcdi_wrapper.so",
]

def _litert_with_litecore_base(
        litert_rule,
        **litert_rule_kwargs):
    """Base function for creating LiteRT rules with Samsung AI LiteCore dependencies."""

    data_x86_64 = _LITECORE_LIBS_X86_64
    data = select({
        "@org_tensorflow//tensorflow:linux_x86_64": data_x86_64,
        "//conditions:default": [],
    })

    append_rule_kwargs(
        litert_rule_kwargs,
        data = data,
        linkopts = select({
            "@org_tensorflow//tensorflow:linux_x86_64": [make_rpaths(_LITECORE_LIBS_X86_64)],
            "//conditions:default": [],
        }),
    )

    litert_rule(**litert_rule_kwargs)

def litert_cc_lib_with_litecore(**litert_lib_kwargs):
    """Creates a litert_lib target with Samsung AI LiteCore backend dependencies.

    Args:
        **litert_lib_kwargs: Keyword arguments passed to litert_lib.
    """
    _litert_with_litecore_base(
        litert_lib,
        **litert_lib_kwargs
    )

def litert_cc_bin_with_litecore(**litert_bin_kwargs):
    """Creates a litert_bin target with Samsung AI LiteCore backend dependencies.

    Args:
        **litert_bin_kwargs: Keyword arguments passed to litert_bin.
    """
    _litert_with_litecore_base(
        litert_bin,
        **litert_bin_kwargs
    )
