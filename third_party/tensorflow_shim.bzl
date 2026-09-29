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

"""Stand-in `@org_tensorflow` repository.

The exported `tflite/` BUILD files still load a few Starlark macros from
TensorFlow and depend on a handful of its header-only targets. Rather than
fetching the source tree, `tensorflow_shim_repositories()` creates a small
repository that contains:

  * the build files under `third_party/tensorflow_shim/<name>/`, and
  * the few source files those build files need, downloaded individually from
    the pinned TensorFlow release and verified by SHA256.

Keep the file list in sync with `tflite/tools/cmake/modules/
Findtensorflow_headers.cmake`.
"""

_TENSORFLOW_BASE_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.21.0/"

# <path in the shim repository>: [<path in the TensorFlow repository>, <sha256>]
_TENSORFLOW_FILES = {
    "LICENSE": [
        "LICENSE",
        "71c6915d04265772a0339bed47276942c678b45cc01534210ebe6984fd1aec65",
    ],
    "tensorflow/core/example/example.proto": [
        "tensorflow/core/example/example.proto",
        "24e188200634bd8ac122f3abbb1f17e17dddf995179c066b20c5dbf4d0631ac8",
    ],
    "tensorflow/core/example/feature.proto": [
        "tensorflow/core/example/feature.proto",
        "17faedb8aabddc32936c003ba58b4e16ffbcd102f22a5cf6bbdfe3365a800a99",
    ],
    "tensorflow/core/public/release_version.h": [
        "tensorflow/core/public/release_version.h",
        "2c39c80166dc4f6d44675eab72919bc21ceb3b270ddcfb95fd9e802777f24268",
    ],
    "tensorflow/core/util/stat_summarizer_options.h": [
        "tensorflow/core/util/stat_summarizer_options.h",
        "e91f94b2fbb7e3a8d8b99664a1ff2443088ca1433b4372b1fcd2477b78492b3b",
    ],
    "tensorflow/core/util/stats_calculator.h": [
        "tensorflow/core/util/stats_calculator.h",
        "8d80ca00444162a55a288b3b7369f314e5cf65a3258387aef4af1df39db08e0a",
    ],
    "tensorflow/python/lib/core/pybind11_lib.h": [
        "tensorflow/python/lib/core/pybind11_lib.h",
        "94762f9cbddd0ee5d2e4a985c0759d1d5db5cae5f03f054cf89fc3084022b433",
    ],
    "third_party/fft2d/fft.h": [
        "third_party/fft2d/fft.h",
        "2db045d17dfd4b4fa5201e86a1653f0c0b7741e14c927f0c426007127109e825",
    ],
    "third_party/fft2d/fft2d.h": [
        "third_party/fft2d/fft2d.h",
        "b24c63e77d5daf3affd7386085c41c046f3ad0c8dcab56a621fd8e84a5af5c1a",
    ],
}

# Upper bound on the number of directories in the shim tree. Starlark has no
# recursion, so the tree is walked with a bounded loop.
_MAX_DIRS = 1000

def _tensorflow_shim_repository_impl(ctx):
    # `anchor` is a file in the main repository's `third_party` package; the
    # shim tree lives next to it and is ignored by Bazel via `.bazelignore`.
    shim_root = ctx.path(ctx.attr.anchor).dirname.get_child("tensorflow_shim").get_child(ctx.attr.shim_dir)
    if not shim_root.exists:
        fail("Shim directory not found: %s" % shim_root)
    if hasattr(ctx, "watch_tree"):
        ctx.watch_tree(shim_root)

    pending = [(shim_root, "")]
    for _ in range(_MAX_DIRS):
        if not pending:
            break
        directory, prefix = pending.pop()
        for entry in directory.readdir():
            relative = prefix + entry.basename
            if entry.is_dir:
                pending.append((entry, relative + "/"))
            else:
                ctx.symlink(entry, relative)
    if pending:
        fail("Shim directory %s is too deep." % shim_root)

    for output, (source, sha256) in ctx.attr.files.items():
        ctx.download(
            url = [ctx.attr.base_url + source],
            output = output,
            sha256 = sha256,
        )

_tensorflow_shim_repository = repository_rule(
    implementation = _tensorflow_shim_repository_impl,
    attrs = {
        "anchor": attr.label(default = Label("//third_party:BUILD"), allow_single_file = True),
        "base_url": attr.string(),
        "files": attr.string_list_dict(),
        "shim_dir": attr.string(mandatory = True),
    },
)

def tensorflow_shim_repositories():
    """Defines the `@org_tensorflow` shim repository."""
    _tensorflow_shim_repository(
        name = "org_tensorflow",
        base_url = _TENSORFLOW_BASE_URL,
        files = _TENSORFLOW_FILES,
        shim_dir = "org_tensorflow",
    )
