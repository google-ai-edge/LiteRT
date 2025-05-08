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

# Description:
#   STBLIB - A collection of public-domain single-file C/C++ libraries,
#            primarily aimed at game developers.

package(
    default_visibility = ["//visibility:public"],
)

# dual public domain and perpetual, irrevocable license to copy, modify,
# publish, and distribute
licenses(["notice"])

# exports_files(["stb_image.h", "stb_image_write.h"])

cc_library(
    name = "stb_image_hdrs",
    srcs = [
        "stb_image.h",
        "stb_image_write.h",
    ],
    includes = ["."],
)

cc_library(
    name = "stb_image",
    srcs = ["stb_image.h"],
    copts = [
        "-Wno-unused-function",
        "$(STACK_FRAME_UNLIMITED)",
    ],
    includes = ["."],
)

cc_library(
    name = "stb_image_write",
    srcs = ["stb_image_write.h"],
    includes = ["."],
)
