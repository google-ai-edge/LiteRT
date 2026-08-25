# Copyright 2026 Google LLC.
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

"""Repository rules for locally installed NVIDIA SDKs."""

def _local_sdk_repository_impl(repository_ctx):
    root = repository_ctx.os.environ.get(repository_ctx.attr.env_var)
    if not root:
        root = repository_ctx.attr.default_root
    if not root:
        fail("Set %s to the SDK root" % repository_ctx.attr.env_var)
    root_path = repository_ctx.path(root)
    if not root_path.exists:
        fail("SDK root does not exist: %s" % root)
    for directory in repository_ctx.attr.directories:
        source = root_path.get_child(directory)
        if not source.exists:
            fail("SDK directory does not exist: %s" % source)
        repository_ctx.symlink(source, directory)
    repository_ctx.symlink(repository_ctx.path(repository_ctx.attr.build_file), "BUILD.bazel")

_local_sdk_repository = repository_rule(
    implementation = _local_sdk_repository_impl,
    attrs = {
        "build_file": attr.label(allow_single_file = True, mandatory = True),
        "default_root": attr.string(),
        "directories": attr.string_list(mandatory = True),
        "env_var": attr.string(mandatory = True),
    },
    environ = ["CUDA_HOME", "TENSORRT_RTX_ROOT"],
    local = True,
)

def local_cuda_repository(name):
    _local_sdk_repository(
        name = name,
        build_file = "//third_party/nvidia_cuda:cuda.BUILD",
        default_root = "/usr/local/cuda",
        directories = ["bin", "include", "lib64", "nvvm"],
        env_var = "CUDA_HOME",
    )

def local_tensorrt_rtx_repository(name):
    _local_sdk_repository(
        name = name,
        build_file = "//third_party/nvidia_tensorrt_rtx:tensorrt_rtx.BUILD",
        directories = ["include", "lib"],
        env_var = "TENSORRT_RTX_ROOT",
    )
