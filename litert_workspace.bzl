# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

"""LiteRT external dependencies initialization."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def litert_workspace():
    """Declares all external repositories directly required by LiteRT and TFLite."""

    # Abseil C++
    tf_http_archive(
        name = "com_google_absl",
        sha256 = "6e1aee535473414164bf83e4ebc40240dec71a4701f8a642d906e95bea1aea0c",
        strip_prefix = "abseil-cpp-20260526.0",
        urls = tf_mirror_urls("https://github.com/abseil/abseil-cpp/archive/20260526.0.tar.gz"),
        patch_file = [
            "//third_party/absl:btree.patch",
            "//third_party/absl:build_dll.patch",
            "//third_party/absl:endian.patch",
            "//third_party/absl:raw_hash_set.patch",
        ],
        repo_mapping = {
            "@google_benchmark": "@com_google_benchmark",
            "@googletest": "@com_google_googletest",
        },
    )

    # FlatBuffers
    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-25.9.23",
        sha256 = "9102253214dea6ae10c2ac966ea1ed2155d22202390b532d1dea64935c518ada",
        urls = tf_mirror_urls("https://github.com/google/flatbuffers/archive/v25.9.23.tar.gz"),
        build_file = "//third_party/flatbuffers:flatbuffers.BUILD",
        system_build_file = "//third_party/flatbuffers:system.BUILD",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )

    # Googletest
    tf_http_archive(
        name = "com_google_googletest",
        sha256 = "a4cb11930215b071168811982dfbebc82a2bb0f90db0e8713245931eb742ea46",
        strip_prefix = "googletest-d72f9c8aea6817cdf1ca0ac10887f328de7f3da2",
        patch_file = ["//third_party/googletest:googletest.patch"],
        urls = tf_mirror_urls("https://github.com/google/googletest/archive/d72f9c8aea6817cdf1ca0ac10887f328de7f3da29.zip"),
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
            "@re2": "@com_googlesource_code_re2",
        },
    )

    # Protobuf
    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = [
            "//third_party/protobuf:protobuf.patch",
            "//third_party/protobuf:protobuf_arena.patch",
        ],
        sha256 = "61e5e5b7f29c4a719d9691b97c2b8937b8bd5ab1b6b7586f3f55934011806280",
        strip_prefix = "protobuf-34.1",
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/releases/download/v34.1/protobuf-34.1.zip"),
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
            "@protobuf_pip_deps": "@pypi",
        },
    )

    # XNNPACK
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "aab0ed2b3972125ca057b8e2b16127fbf68f8128ea07dd7efe56cdb92f2c1917",
        strip_prefix = "XNNPACK-d89ef6669a14db203b3b7935b1b3862cb63fb6df",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/d89ef6669a14db203b3b7935b1b3862cb63fb6df.zip"),
    )

    # KleidiAI (XNNPACK dependency)
    tf_http_archive(
        name = "KleidiAI",
        sha256 = "652ee046ebf99bee5ed78aaf8961ea1df6ac09b6ebbd3a33106d9ce255efba69",
        strip_prefix = "kleidiai-4bc7bd457930de119f8b826ecfdebfbd9c08cf8c",
        urls = tf_mirror_urls("https://gitlab.arm.com/kleidi/kleidiai/-/archive/4bc7bd457930de119f8b826ecfdebfbd9c08cf8c/kleidiai-4bc7bd457930de119f8b826ecfdebfbd9c08cf8c.zip"),
    )

    # FXdiv
    tf_http_archive(
        name = "FXdiv",
        sha256 = "3d7b0e9c4c658a84376a1086126be02f9b7f753caa95e009d9ac38d11da444db",
        strip_prefix = "FXdiv-63058eff77e11aa15bf531df5dd34395ec3017c8",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip"),
    )

    # pthreadpool
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "9b9fb1179b71021c0c048504eab636424c58e9c8374404754ece6d7cd90f26d4",
        strip_prefix = "pthreadpool-15a6644ba1c45f1acc16ac1e883efc3e56c6bed2",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/15a6644ba1c45f1acc16ac1e883efc3e56c6bed2.zip"),
    )

    # Slinky (XNNPACK dependency)
    tf_http_archive(
        name = "slinky",
        sha256 = "6ba811c39fd400149d58a42a1a5ad33e7fe2554c9a45990788779e0657039405",
        strip_prefix = "slinky-2c9e129fc82215530af851cfd8d3094de0519241",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/2c9e129fc82215530af851cfd8d3094de0519241.zip"),
    )

    # cpuinfo
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "fe2aa43254838a2eb5658d1742696473a1d834a57f2a0b38d533346bcd212482",
        strip_prefix = "cpuinfo-8ce83db858065145192c97af90cb668ad72a12e9",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/8ce83db858065145192c97af90cb668ad72a12e9.zip"),
    )

    # Eigen
    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = "35c6126e246585d9cf6600b65471582c2701aae64b784a6fd19168a90cfc841e",
        strip_prefix = "eigen-ea13a98decd497a8c5588fb5de71b57bcf10d864",
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/ea13a98decd497a8c5588fb5de71b57bcf10d864/eigen-ea13a98decd497a8c5588fb5de71b57bcf10d864.tar.gz"),
    )

    # Ruy
    tf_http_archive(
        name = "ruy",
        sha256 = "a22c42e80c7bb450db8492728e4742ee66f46d5458c45fe67ce2c9b61240630c",
        strip_prefix = "ruy-3286a34cc8de6149ac6844107dfdffac91531e72",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/3286a34cc8de6149ac6844107dfdffac91531e72.zip"),
        build_file = "//third_party/ruy:ruy.BUILD",
    )

    # Gemmlowp
    tf_http_archive(
        name = "gemmlowp",
        sha256 = "7dc418717c8456473fac4ff2288b71057e3dcb72894524c734a4362cdb51fa8b",
        strip_prefix = "gemmlowp-16e8662c34917be0065110bfcd9cc27d30f52fdf",
        urls = tf_mirror_urls("https://github.com/google/gemmlowp/archive/16e8662c34917be0065110bfcd9cc27d30f52fdf.zip"),
    )

    # Farmhash
    tf_http_archive(
        name = "farmhash_archive",
        build_file = "//third_party/farmhash:farmhash.BUILD",
        sha256 = "18392cf0736e1d62ecbb8d695c31496b6507859e8c75541d7ad0ba092dc52115",
        strip_prefix = "farmhash-0d859a811870d10f53a594927d0d0b97573ad06d",
        urls = tf_mirror_urls("https://github.com/google/farmhash/archive/0d859a811870d10f53a594927d0d0b97573ad06d.tar.gz"),
    )

    # FFT2D
    tf_http_archive(
        name = "fft2d",
        build_file = "//third_party/fft2d:fft2d.BUILD",
        sha256 = "5f4dabc2ae21e1f537425d58a49cdca1c49ea11db0d6271e2a4b27e9697548eb",
        strip_prefix = "OouraFFT-1.0",
        urls = tf_mirror_urls("https://github.com/petewarden/OouraFFT/archive/v1.0.tar.gz"),
    )

    # ARM NEON 2 x86 SSE
    tf_http_archive(
        name = "arm_neon_2_x86_sse",
        build_file = "//third_party/arm_neon_2_x86_sse:arm_neon_2_x86_sse.BUILD",
        sha256 = "019fbc7ec25860070a1d90e12686fc160cfb33e22aa063c80f52b363f1361e9d",
        strip_prefix = "ARM_NEON_2_x86_SSE-a15b489e1222b2087007546b4912e21293ea86ff",
        urls = tf_mirror_urls("https://github.com/intel/ARM_NEON_2_x86_SSE/archive/a15b489e1222b2087007546b4912e21293ea86ff.tar.gz"),
    )

    # OpenCL Headers
    tf_http_archive(
        name = "opencl_headers",
        strip_prefix = "OpenCL-Headers-dcd5bede6859d26833cd85f0d6bbcee7382dc9b3",
        sha256 = "ca8090359654e94f2c41e946b7e9d826253d795ae809ce7c83a7d3c859624693",
        urls = tf_mirror_urls("https://github.com/KhronosGroup/OpenCL-Headers/archive/dcd5bede6859d26833cd85f0d6bbcee7382dc9b3.tar.gz"),
        build_file = "//third_party/opencl_headers:opencl_headers.BUILD",
    )
