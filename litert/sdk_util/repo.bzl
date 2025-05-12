# Copyright 2025 Google LLC.
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

"""Repository rules that can toggle between local and remote repos."""

load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "get_auth",
)

def _get_path(ctx, s):
    """Turns string into path, ensuring its valid."""
    path = ctx.workspace_root.get_child(s)
    if not path.exists:
        fail(
            ("local_path path is \"%s\" (absolute: \"%s\") but it does not exist.") % (s, path),
        )
    return path

def _prepare_repo_files(ctx):
    if ctx.attr.local_path_env and ctx.getenv(ctx.attr.local_path_env, None):
        sdk_path_from_env = ctx.getenv(ctx.attr.local_path_env, None)

        if sdk_path_from_env:
            if not sdk_path_from_env.startswith("/"):
                fail("Local path must be absolute.")

            if ctx.path(sdk_path_from_env).is_dir:
                for child in ctx.path(sdk_path_from_env + ctx.attr.strip_prefix).readdir():
                    ctx.symlink(child, child.basename)

            elif not sdk_path_from_env.endswith(".gz"):  # MTK gives us .gz instead of .tar.gz
                fail("Local path is not a dir or a tarball")

            else:
                ctx.extract(sdk_path_from_env, stripPrefix = ctx.attr.strip_prefix)

    elif not ctx.attr.url:
        fail("A URL must be specified if local repo is not enabled.")

    else:
        ctx.download_and_extract(
            url = ctx.attr.url,
            auth = get_auth(ctx, [ctx.attr.url]),
            stripPrefix = ctx.attr.strip_prefix,
            type = "tar.gz",  # MTK gives us .gz instead of .tar.gz
        )

    if ctx.attr.symlink_mapping:
        for dst, src in ctx.attr.symlink_mapping.items():
            ctx.symlink(src, dst)

def _prepare_build_file(ctx):
    build_file = ctx.attr.build_file
    build_file_contents = ctx.attr.build_file_contents

    if build_file and build_file_contents:
        fail("Only one of build_file or build_file_contents may be specified")

    if build_file:
        ctx.delete("BUILD.bazel")
        ctx.symlink(ctx.attr.build_file, "BUILD.bazel")

    elif build_file_contents:
        ctx.file("BUILD.bazel", build_file_contents)

    else:
        has_build = False
        for outfile in ctx.path(".").readdir():
            if outfile.basename in ["BUILD.bazel"]:
                has_build = True
                break
        if not has_build:
            fail("No BUILD.bazel file in out dir, please configure one or add it to the target repo.")

def _configurable_repo_impl(ctx):
    _prepare_repo_files(ctx)
    _prepare_build_file(ctx)

# PUBLIC ###########################################################################################

configurable_repo = repository_rule(
    implementation = _configurable_repo_impl,
    attrs = {
        "local_path_env": attr.string(
            doc = """
            An environment variable to look for an absolute path to local repository files. 
            NOTE: If this is specified, and the environment variable is set, it will take 
            precedence over the URL.
            """,
            mandatory = False,
        ),
        "url": attr.string(
            doc = """
            The url to the hosted litert repo archive to download. This must be a tarball.
            This may be unspecified if local path approach is used.
            """,
            mandatory = False,
        ),
        "build_file": attr.label(
            allow_single_file = True,
            doc = """
            A build file to associate with the root of the repo.
            """,
        ),
        "build_file_contents": attr.string(
            doc = """
            String contents to associate with the root of the repo.
            Only one of `build_file` or `build_file_contents` may be specified.
            """,
        ),
        "strip_prefix": attr.string(
            doc = """
            Prefix to strip when extracting archives.
            """,
        ),
        "symlink_mapping": attr.string_dict(
            doc = """
            A mapping of files to symlink to in the repo.
            """,
            mandatory = False,
        ),
    },
)
