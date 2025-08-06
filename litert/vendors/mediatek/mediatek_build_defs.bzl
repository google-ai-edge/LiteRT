# Copyright 2024 Google LLC.
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

"""Build definitions for Mediatek backend."""

load("//litert/build_common:litert_build_defs.bzl", "append_rule_kwargs", "litert_bin", "litert_lib", "make_rpaths")

_MTK_STD_LIBS_HOST = [
    # copybara:uncomment_begin(google-only)
    # "@neuro_pilot//:latest/host/lib/libc++.so.1",
    # "@neuro_pilot//:latest/host/lib/libstdc++.so.6",
    # copybara:uncomment_end
]  # @unused

def _litert_with_mtk_base(
        litert_rule,
        use_custom_libcc,
        **litert_rule_kwargs):
    if use_custom_libcc:
        # TODO: Figure out strategy for custom libcc.
        fail("Custom libcc not yet supported")

    # v7 vs v8 sdks can be toggled in linux builds via the ":mtk_sdk_version" flag
    # defined in litert/BUILD.
    append_rule_kwargs(
        litert_rule_kwargs,
        data = select({
            "@org_tensorflow//tensorflow:linux_x86_64": [
                "@neuro_pilot//:v9_latest/host/lib/libneuron_adapter.so",
                "@neuro_pilot//:v8_latest/host/lib/libneuron_adapter.so",
                "@neuro_pilot//:v7_latest/host/lib/libneuron_adapter.so",
            ],
            "//conditions:default": [],
        }),
        linkopts = select({
            "@org_tensorflow//tensorflow:linux_x86_64": [
                make_rpaths(["@neuro_pilot//:v9_latest/host/lib/libneuron_adapter.so"]),
                make_rpaths(["@neuro_pilot//:v8_latest/host/lib/libneuron_adapter.so"]),
                make_rpaths(["@neuro_pilot//:v7_latest/host/lib/libneuron_adapter.so"]),
            ],
            "//conditions:default": [],
        }),
    )

    litert_rule(**litert_rule_kwargs)

def litert_cc_lib_with_mtk(
        use_custom_libcc = False,
        **litert_lib_kwargs):
    """Creates a litert_lib target with Mediatek backend dependencies.

    Args:
        use_custom_libcc: Whether to use a custom libcc. Not yet supported.
        **litert_lib_kwargs: Keyword arguments passed to litert_lib.
    """
    _litert_with_mtk_base(
        litert_lib,
        use_custom_libcc,
        **litert_lib_kwargs
    )

def litert_cc_bin_with_mtk(
        use_custom_libcc = False,
        **litert_bin_kwargs):
    """Creates a litert_bin target with Mediatek backend dependencies.

    Args:
        use_custom_libcc: Whether to use a custom libcc. Not yet supported.
        **litert_bin_kwargs: Keyword arguments passed to litert_bin.
    """
    _litert_with_mtk_base(
        litert_bin,
        use_custom_libcc,
        **litert_bin_kwargs
    )
