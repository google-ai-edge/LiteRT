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

def _sdist_impl(ctx):
    output_sdist = ctx.outputs.sdist_archive

    sdist_wrapper_executable = ctx.executable._sdist_wrapper

    setup_py_dir = ctx.file.setup_py.path.rpartition("/")[0]
    setup_py_name = ctx.file.setup_py.path.rpartition("/")[-1]
    if not setup_py_dir:
        setup_py_dir = "."

    args = ctx.actions.args()
    args.add("--dir", setup_py_dir)
    args.add("--setup_py", setup_py_name)
    args.add("--output_sdist_path", output_sdist.path)

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
    return [DefaultInfo(files = depset([output_sdist]))]

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
        "sdist_archive_name": attr.string(
            mandatory = True,
            doc = "The desired name for the output sdist .tar.gz file (e.g., 'my_package-1.0.tar.gz').",
        ),
        "_sdist_wrapper": attr.label(
            default = Label("//ci/tools/python/vendor_sdk:sdist_wrapper"),
            executable = True,
            cfg = "exec",
            doc = "The py_binary wrapper script for sdist creation.",
        ),
    },
    outputs = {"sdist_archive": "%{sdist_archive_name}"},
)
