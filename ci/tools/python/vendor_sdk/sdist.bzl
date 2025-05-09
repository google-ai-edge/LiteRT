# Copyright 2025 The AI Edge LiteRT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This module contains a rule to generate a sdist archive for a Python package."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def _get_full_sdist_name(sdist_name, version):
    sdist_version = version.replace("-", "")
    return "{sdist_name}-{sdist_version}.tar.gz".format(
        sdist_name = sdist_name,
        sdist_version = sdist_version,
    )

def _sdist_impl(ctx):
    version = ctx.attr.version
    package_name = ctx.attr.package_name
    if ctx.attr.nightly_suffix and ctx.attr.nightly_suffix[BuildSettingInfo].value != "":
        version = version + ".dev" + ctx.attr.nightly_suffix[BuildSettingInfo].value
        sdist_name = _get_full_sdist_name(package_name + "_nightly", version)
    else:
        sdist_name = _get_full_sdist_name(package_name, version)

    output_sdist = ctx.actions.declare_file(sdist_name)

    sdist_wrapper_executable = ctx.executable._sdist_wrapper

    setup_py_dir = ctx.file.setup_py.path.rpartition("/")[0]
    setup_py_name = ctx.file.setup_py.path.rpartition("/")[-1]
    if not setup_py_dir:
        setup_py_dir = "."

    args = ctx.actions.args()
    args.add("--project_name", package_name)
    args.add("--version", version)
    args.add("--dir", setup_py_dir)
    args.add("--setup_py", setup_py_name)
    args.add("--output_sdist_path", output_sdist.path)

    if ctx.attr.nightly_suffix and ctx.attr.nightly_suffix[BuildSettingInfo].value != "":
        args.add("--nightly_suffix", "_nightly")

    all_input_files = depset(
        direct = [ctx.file.setup_py, ctx.file.manifest_in],
        transitive = [depset(ctx.files.package_srcs)],
    )

    ctx.actions.run(
        executable = sdist_wrapper_executable,
        arguments = [args],
        inputs = all_input_files,
        outputs = [output_sdist],
        progress_message = "Creating sdist for {} via Python wrapper".format(ctx.attr.package_name),
        mnemonic = "PySdistWrapper",
    )
    return [DefaultInfo(files = depset(direct = [output_sdist]))]

sdist_rule = rule(
    implementation = _sdist_impl,
    attrs = {
        "setup_py": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "Label to the setup.py file.",
        ),
        "manifest_in": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "Label to the MANIFEST.in file.",
        ),
        "package_srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
            doc = "List of labels to all files/dirs within the Python package.",
        ),
        "package_name": attr.string(
            mandatory = True,
            doc = "Name of the Python package.",
        ),
        "version": attr.string(
            mandatory = True,
            doc = "Version of the Python package.",
        ),
        "_sdist_wrapper": attr.label(
            default = Label("//ci/tools/python/vendor_sdk:sdist_wrapper"),
            executable = True,
            cfg = "exec",
            doc = "The py_binary wrapper script for sdist creation.",
        ),
        "nightly_suffix": attr.label(),
    },
)
