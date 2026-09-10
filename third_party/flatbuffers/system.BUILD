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
# ==============================================================================

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "LICENSE.txt",
    visibility = ["//visibility:public"],
)

# Public flatc library to compile flatbuffer files at runtime.
cc_library(
    name = "flatbuffers",
    linkopts = ["-lflatbuffers"],
    visibility = ["//visibility:public"],
)

# Public flatc compiler library.
cc_library(
    name = "flatc_library",
    linkopts = ["-lflatbuffers"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "lnflatc",
    outs = ["flatc.bin"],
    cmd = "ln -s $$(which flatc) $@",
)

# Public flatc compiler.
sh_binary(
    name = "flatc",
    srcs = ["flatc.bin"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "runtime_cc",
    visibility = ["//visibility:public"],
)

py_library(
    name = "runtime_py",
    visibility = ["//visibility:public"],
)
