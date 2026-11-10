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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:jvm.bzl", "jvm_import_external")
load("//:tensorflow_source_rules.bzl", "tensorflow_source_repo")

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

def _litert_tf_config_impl(ctx):
    value = ctx.os.environ.get("LITERT_WITH_TENSORFLOW", "").strip().lower()
    ctx.file("BUILD", "")
    ctx.file(
        "config.bzl",
        "# Generated from LITERT_WITH_TENSORFLOW.\nWITH_TENSORFLOW = %s\n" %
        (value in ["1", "true", "yes"]),
    )

# Exposes `WITH_TENSORFLOW` from the LITERT_WITH_TENSORFLOW environment variable
# so that the WORKSPACE can choose between the shim and the real TensorFlow.
litert_tf_config = repository_rule(
    implementation = _litert_tf_config_impl,
    environ = ["LITERT_WITH_TENSORFLOW"],
    local = True,
)

def tensorflow_shim_repositories(with_tensorflow = False):
    """Defines the `@org_tensorflow`, `@xla` and `@llvm-project` repositories.

    Args:
      with_tensorflow: If True, `@org_tensorflow` is the real TensorFlow source
        tree, which the converter needs. TensorFlow's workspace macros then
        define `@xla`, `@llvm-project` and the other dependencies. Otherwise
        the three repositories are the shims.
    """
    if with_tensorflow:
        tensorflow_source_repo(
            name = "org_tensorflow",
            sha256 = "7bf06cfd5ff9b462b1b25ca4dc3613fa5e3847fd8e291ff0a8de2ca5a812590a",
            strip_prefix = "tensorflow-5c0b7a5946f0f485e3a532b2a00e03f42a6e14c1",
            urls = ["https://github.com/tensorflow/tensorflow/archive/5c0b7a5946f0f485e3a532b2a00e03f42a6e14c1.tar.gz"],
        )
        return
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

def _archive_with_build_files_impl(ctx):
    ctx.download_and_extract(
        url = ctx.attr.urls,
        sha256 = ctx.attr.sha256,
        stripPrefix = ctx.attr.strip_prefix,
    )
    for label, path in ctx.attr.build_files.items():
        ctx.delete(path)
        ctx.symlink(label, path)

# Like `http_archive`, but can add build files to subdirectories.
_archive_with_build_files = repository_rule(
    implementation = _archive_with_build_files_impl,
    attrs = {
        "build_files": attr.label_keyed_string_dict(allow_files = True),
        "sha256": attr.string(mandatory = True),
        "strip_prefix": attr.string(),
        "urls": attr.string_list(mandatory = True),
    },
)

def tensorflow_shim_dependencies(with_tensorflow = False):
    """Defines the external repositories that TensorFlow's workspace used to add.

    LiteRT build files use these directly. The versions match the ones that
    TensorFlow's `tf_workspace*()` macros declared, except for zlib, which uses
    the version and build file from protobuf.

    Args:
      with_tensorflow: If True, does nothing. TensorFlow's workspace macros
        define these repositories.
    """
    if with_tensorflow:
        return

    if not native.existing_rule("zlib"):
        http_archive(
            name = "zlib",
            build_file = "@com_google_protobuf//third_party:zlib.BUILD",
            sha256 = "38ef96b8dfe510d42707d9c781877914792541133e1870841463bfa73f883e32",
            strip_prefix = "zlib-1.3.1",
            urls = [
                "https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.xz",
                "https://zlib.net/zlib-1.3.1.tar.xz",
            ],
        )

    if not native.existing_rule("absl_py"):
        http_archive(
            name = "absl_py",
            sha256 = "8a3d0830e4eb4f66c4fa907c06edf6ce1c719ced811a12e26d9d3162f8471758",
            strip_prefix = "abseil-py-2.1.0",
            urls = ["https://github.com/abseil/abseil-py/archive/refs/tags/v2.1.0.tar.gz"],
        )

    if not native.existing_rule("ml_dtypes_py"):
        _archive_with_build_files(
            name = "ml_dtypes_py",
            build_files = {
                Label("//third_party/py/ml_dtypes:ml_dtypes_py.BUILD"): "BUILD.bazel",
                Label("//third_party/py/ml_dtypes:ml_dtypes.BUILD"): "ml_dtypes/BUILD.bazel",
            },
            sha256 = "f6e5880666661351e6cd084ac4178ddc4dabcde7e9a73722981c0d1500cf5937",
            strip_prefix = "ml_dtypes-00d98cd92ade342fef589c0470379abb27baebe9",
            urls = ["https://github.com/jax-ml/ml_dtypes/archive/00d98cd92ade342fef589c0470379abb27baebe9/ml_dtypes-00d98cd92ade342fef589c0470379abb27baebe9.tar.gz"],
        )

    if not native.existing_rule("com_google_fuzztest"):
        http_archive(
            name = "com_google_fuzztest",
            sha256 = "c75f224b34c3c62ee901381fb743f6326f7b91caae0ceb8fe62f3fd36f187627",
            strip_prefix = "fuzztest-58b4e7065924f1a284952b84ea827ce35a87e4dc",
            urls = ["https://github.com/google/fuzztest/archive/58b4e7065924f1a284952b84ea827ce35a87e4dc.zip"],
        )

    if not native.existing_rule("jsoncpp_git"):
        http_archive(
            name = "jsoncpp_git",
            sha256 = "f409856e5920c18d0c2fb85276e24ee607d2a09b5e7d5f0a371368903c275da2",
            strip_prefix = "jsoncpp-1.9.5",
            urls = ["https://github.com/open-source-parsers/jsoncpp/archive/1.9.5.tar.gz"],
        )

    if not native.existing_rule("tflite_mobilenet_float"):
        http_archive(
            name = "tflite_mobilenet_float",
            build_file = Label("//third_party/tflite_mobilenet:tflite_mobilenet_float.BUILD"),
            sha256 = "2fadeabb9968ec6833bee903900dda6e61b3947200535874ce2fe42a8493abc0",
            urls = ["https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"],
        )

    if not native.existing_rule("com_google_benchmark"):
        http_archive(
            name = "com_google_benchmark",
            sha256 = "552ca3d4d1af4beeb1907980f7096315aa24150d6baf5ac1e5ad90f04846c670",
            strip_prefix = "benchmark-f7547e29ccaed7b64ef4f7495ecfff1c9f6f3d03",
            urls = ["https://github.com/google/benchmark/archive/f7547e29ccaed7b64ef4f7495ecfff1c9f6f3d03.tar.gz"],
        )

    if not native.existing_rule("hexagon_nn"):
        http_archive(
            name = "hexagon_nn",
            build_file = Label("//third_party/hexagon:hexagon.BUILD"),
            sha256 = "f577b4c150b72e11e9dfb3f9d14f9772ba8fe460f7d65c84a7327ea9bef44d8e",
            urls = ["https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.9.tgz"],
        )

    if not native.existing_rule("kissfft"):
        http_archive(
            name = "kissfft",
            build_file = Label("//third_party/kissfft:kissfft.BUILD"),
            sha256 = "76c1aac87ddb7258f34b08a13f0eebf9e53afa299857568346aa5c82bcafaf1a",
            strip_prefix = "kissfft-131.1.0",
            urls = ["https://github.com/mborgerding/kissfft/archive/refs/tags/131.1.0.tar.gz"],
        )

    if not native.existing_rule("vulkan_headers"):
        _archive_with_build_files(
            name = "vulkan_headers",
            build_files = {
                Label("//third_party/vulkan_headers:vulkan_headers.BUILD"): "BUILD.bazel",
                Label("//third_party/vulkan_headers:tensorflow/vulkan_hpp_dispatch_loader_dynamic.cc"): "tensorflow/vulkan_hpp_dispatch_loader_dynamic.cc",
            },
            sha256 = "602aedcc4c6057473d0f7fee1bcc3aa01bf191371b2b5bbca949cebc03cf393a",
            strip_prefix = "Vulkan-Headers-32c07c0c5334aea069e518206d75e002ccd85389",
            urls = ["https://github.com/KhronosGroup/Vulkan-Headers/archive/32c07c0c5334aea069e518206d75e002ccd85389.tar.gz"],
        )

    if not native.existing_rule("tflite_mobilenet_ssd_quant_protobuf"):
        http_archive(
            name = "tflite_mobilenet_ssd_quant_protobuf",
            build_file = Label("//third_party/tflite_mobilenet:tflite_mobilenet.BUILD"),
            sha256 = "09280972c5777f1aa775ef67cb4ac5d5ed21970acd8535aeca62450ef14f0d79",
            strip_prefix = "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18",
            urls = ["https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz"],
        )

    if not native.existing_rule("org_checkerframework_qual"):
        jvm_import_external(
            name = "org_checkerframework_qual",
            artifact_sha256 = "d261fde25d590f6b69db7721d469ac1b0a19a17ccaaaa751c31f0d8b8260b894",
            artifact_urls = ["https://repo1.maven.org/maven2/org/checkerframework/checker-qual/2.10.0/checker-qual-2.10.0.jar"],
            licenses = ["notice"],
            rule_name = "java_import",
        )

    if not native.existing_rule("com_google_auto_value_annotations"):
        jvm_import_external(
            name = "com_google_auto_value_annotations",
            artifact_sha256 = "d095936c432f2afc671beaab67433e7cef50bba4a861b77b9c46561b801fae69",
            artifact_urls = ["https://repo1.maven.org/maven2/com/google/auto/value/auto-value-annotations/1.6/auto-value-annotations-1.6.jar"],
            default_visibility = ["@com_google_auto_value//:__pkg__"],
            licenses = ["notice"],
            neverlink = True,
            rule_name = "java_import",
        )

    if not native.existing_rule("com_google_auto_value"):
        jvm_import_external(
            name = "com_google_auto_value",
            artifact_sha256 = "fd811b92bb59ae8a4cf7eb9dedd208300f4ea2b6275d726e4df52d8334aaae9d",
            artifact_urls = ["https://repo1.maven.org/maven2/com/google/auto/value/auto-value/1.6/auto-value-1.6.jar"],
            exports = ["@com_google_auto_value_annotations"],
            extra_build_file_content = _AUTO_VALUE_PLUGINS,
            generated_rule_name = "processor",
            licenses = ["notice"],
            rule_name = "java_import",
        )

    # Test-only Java dependencies of `tflite/java`.
    if not native.existing_rule("junit"):
        jvm_import_external(
            name = "junit",
            artifact_sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a",
            artifact_urls = ["https://repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar"],
            licenses = ["reciprocal"],
            rule_name = "java_import",
            testonly_ = True,
            deps = ["@org_hamcrest_core"],
        )

    if not native.existing_rule("org_hamcrest_core"):
        jvm_import_external(
            name = "org_hamcrest_core",
            artifact_sha256 = "66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9",
            artifact_urls = ["https://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar"],
            licenses = ["notice"],
            rule_name = "java_import",
            testonly_ = True,
        )

    if not native.existing_rule("com_google_truth"):
        jvm_import_external(
            name = "com_google_truth",
            artifact_sha256 = "032eddc69652b0a1f8d458f999b4a9534965c646b8b5de0eba48ee69407051df",
            artifact_urls = ["https://repo1.maven.org/maven2/com/google/truth/truth/0.32/truth-0.32.jar"],
            licenses = ["notice"],
            rule_name = "java_import",
            testonly_ = True,
            deps = ["@com_google_guava"],
        )

    if not native.existing_rule("com_google_guava"):
        jvm_import_external(
            name = "com_google_guava",
            artifact_sha256 = "6db0c3a244c397429c2e362ea2837c3622d5b68bb95105d37c21c36e5bc70abf",
            artifact_urls = ["https://repo1.maven.org/maven2/com/google/guava/guava/25.1-jre/guava-25.1-jre.jar"],
            exports = [
                "@com_google_code_findbugs_jsr305",
                "@com_google_errorprone_error_prone_annotations",
            ],
            licenses = ["notice"],
            rule_name = "java_import",
        )

    if not native.existing_rule("com_google_code_findbugs_jsr305"):
        jvm_import_external(
            name = "com_google_code_findbugs_jsr305",
            artifact_sha256 = "bec0b24dcb23f9670172724826584802b80ae6cbdaba03bdebdef9327b962f6a",
            artifact_urls = ["https://repo1.maven.org/maven2/com/google/code/findbugs/jsr305/2.0.3/jsr305-2.0.3.jar"],
            licenses = ["notice"],
            rule_name = "java_import",
        )

    if not native.existing_rule("com_google_errorprone_error_prone_annotations"):
        jvm_import_external(
            name = "com_google_errorprone_error_prone_annotations",
            artifact_sha256 = "03d0329547c13da9e17c634d1049ea2ead093925e290567e1a364fd6b1fc7ff8",
            artifact_urls = ["https://repo1.maven.org/maven2/com/google/errorprone/error_prone_annotations/2.1.3/error_prone_annotations-2.1.3.jar"],
            licenses = ["notice"],
            rule_name = "java_import",
        )

_AUTO_VALUE_PLUGINS = """
java_plugin(
    name = "AutoAnnotationProcessor",
    output_licenses = ["unencumbered"],
    processor_class = "com.google.auto.value.processor.AutoAnnotationProcessor",
    deps = [":processor"],
)

java_plugin(
    name = "AutoOneOfProcessor",
    output_licenses = ["unencumbered"],
    processor_class = "com.google.auto.value.processor.AutoOneOfProcessor",
    deps = [":processor"],
)

java_plugin(
    name = "AutoValueProcessor",
    output_licenses = ["unencumbered"],
    processor_class = "com.google.auto.value.processor.AutoValueProcessor",
    deps = [":processor"],
)

java_library(
    name = "com_google_auto_value",
    exported_plugins = [
        ":AutoAnnotationProcessor",
        ":AutoOneOfProcessor",
        ":AutoValueProcessor",
    ],
    exports = ["@com_google_auto_value_annotations"],
)
"""
