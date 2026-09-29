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

"""Stand-in `@org_tensorflow`, `@xla` and `@llvm-project` repositories.

The exported `tflite/` BUILD files still load a few Starlark macros from
TensorFlow, XLA and MLIR and depend on a handful of their header-only targets.
Rather than fetching those source trees, `tensorflow_shim_repositories()`
creates small repositories that contain:

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

_XLA_FILES = {
    "LICENSE": [
        "third_party/xla/LICENSE",
        "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
    ],
    "xla/tsl/framework/convolution/eigen_convolution_helpers.h": [
        "third_party/xla/xla/tsl/framework/convolution/eigen_convolution_helpers.h",
        "3fd52ebb0f14b9c4f3b3e225d1e7b29c712c3046d34e07936bfec0e0c1f41152",
    ],
    "xla/tsl/framework/convolution/eigen_spatial_convolutions-inl.h": [
        "third_party/xla/xla/tsl/framework/convolution/eigen_spatial_convolutions-inl.h",
        "c0bec189723d52b4495dec62e197e2a4ffe7725bc313676984384e8b8fbb7a13",
    ],
    "xla/tsl/lib/random/philox_random.h": [
        "third_party/xla/xla/tsl/lib/random/philox_random.h",
        "7a7659f95c59419373261af305736311bfd839b281259703b23f4df9b391a43e",
    ],
    "xla/tsl/lib/random/random_distributions_utils.h": [
        "third_party/xla/xla/tsl/lib/random/random_distributions_utils.h",
        "1d65158a878510a1ec5835c5f26962b0ebe45bd56ecaec7e38314a8ac5e7a517",
    ],
    "xla/tsl/util/stat_summarizer_options.h": [
        "third_party/xla/xla/tsl/util/stat_summarizer_options.h",
        "9f7d7cc5de38ae6e97a8982dc386a012ed70144c768ef2525fd25054de41c2b2",
    ],
    "xla/tsl/util/stats_calculator.cc": [
        "third_party/xla/xla/tsl/util/stats_calculator.cc",
        "704c6d22240a521a7cbee591cd932121f0f6dcbaf179037758a7f3bd4d851e82",
    ],
    "xla/tsl/util/stats_calculator.h": [
        "third_party/xla/xla/tsl/util/stats_calculator.h",
        "f5628ba1fbf39e7c4d2b280ff78c8daba40d9d462a97f8aa2abc0c5371b9f241",
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
    """Defines the `@org_tensorflow`, `@xla` and `@llvm-project` shim repositories."""
    _tensorflow_shim_repository(
        name = "org_tensorflow",
        base_url = _TENSORFLOW_BASE_URL,
        files = _TENSORFLOW_FILES,
        shim_dir = "org_tensorflow",
    )
    _tensorflow_shim_repository(
        name = "xla",
        base_url = _TENSORFLOW_BASE_URL,
        files = _XLA_FILES,
        shim_dir = "xla",
    )
    _tensorflow_shim_repository(
        name = "llvm-project",
        shim_dir = "llvm_project",
    )
