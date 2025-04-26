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

def _get_path_str_if_do_local(ctx):
    if not ctx.attr.allow_local:
        return None

    if not ctx.getenv(ctx.attr.allow_local_env, None):
        return None

    local_repo_path = ctx.attr.local_path
    if local_repo_path:
        return local_repo_path

    local_path_env_val = ctx.getenv(ctx.attr.local_path_env, None)
    return local_path_env_val

def _get_path(ctx, s):
    """Turns string into path, ensuring its valid."""
    path = ctx.workspace_root.get_child(s)
    if not path.exists:
        fail(
            ("local_path path is \"%s\" (absolute: \"%s\") but it does not exist.") % (s, path),
        )
    return path

def _configurable_repo_impl(ctx):
    # Resolve remote/vs local
    local_repo_path = _get_path_str_if_do_local(ctx)

    # Setup files
    if not local_repo_path:
        if not ctx.attr.url:
            fail("A URL must be specified if local repo is not enabled.")
        fail("http download not implemented yet.")

    if not local_repo_path.startswith("/"):
        fail("Only absolute paths to local files are supported")

    pth = _get_path(ctx, local_repo_path)

    if pth.is_dir:
        for child in pth.readdir():
            ctx.symlink(child, child.basename)

    elif pth.basename.endswith("tar.gz"):
        ctx.extract(pth, stripPrefix = ctx.attr.strip_prefix)

    else:
        fail("Local path is not a dir or a tarball")

    # Resolve BUILD
    build_file = ctx.attr.build_file
    build_file_contents = ctx.attr.build_file_contents

    if build_file and build_file_contents:
        fail("Only one of build_file or build_file_contents may be specified")

    if build_file:
        # TODO forward file
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

configurable_repo = repository_rule(
    implementation = _configurable_repo_impl,
    attrs = {
        "local_path": attr.string(
            doc = "A path to the repository files on the local machine, either absolute or relative to workspace root. This can be either a directory or a tarball. This always takes precedence over `local_path_env`",
        ),
        "allow_local": attr.bool(
            doc = "Specifies whether or not pulling the repository from local machine is permitted. Url must be specified if this is false. This takes precedence over `allow_local_env`",
            mandatory = True,
        ),
        "local_path_env": attr.string(
            doc = "An environment variable to look for a path to local repository files, either absolute or relative to workspace root. This is used when `local_path` is unspecified and `allow_local` is true.",
        ),
        "allow_local_env": attr.string(
            doc = "An environment variable to serve as a toggle for whether or not local repository functionality is enabled. It can be set by the user to swap between local/remote repos. This is used when `allow_local` is true. The non-existance of this env var is interpreted as disabling local repo.",
            default = "LITERT_ALLOW_LOCAL_REPO",
        ),
        "url": attr.string(
            doc = "The url to the hosted litert repo archive to download. This must be a tarball. The remote files are used when local repository functionality is not enabled.",
        ),
        "build_file": attr.label(
            allow_single_file = True,
            doc = "A build file to associate with the root of the repo.",
        ),
        "build_file_contents": attr.string(
            doc = "String contents to associate with the root of the repo. Only one of `build_file` or `build_file_contents` may be specified.",
        ),
        "strip_prefix": attr.string(
            doc = "Prefix to strip when extracting archives.",
        ),
    },
)
