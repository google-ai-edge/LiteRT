#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>.
# SPDX-License-Identifier: Apache-2.0
#

package(default_visibility = ["//visibility:public"])

# Core decoder/runtime sources from the upstream VGF archive.
cc_library(
    name = "vgf",
    srcs = [
        "src/decoder.cpp",
        "src/logging.cpp",
    ],
    hdrs = glob([
        "include/**/*.hpp",
        "schema/vgf_generated.h",
        "src/*.hpp",
    ]),
    includes = [
        "include",
        "schema",
        "src",
    ],
    deps = ["@flatbuffers//:runtime_cc"],
)

# Utility parser target kept separate to match the upstream source layout.
cc_library(
    name = "vgf_utils",
    srcs = ["utils/src/parse_vgf.cpp"],
    hdrs = ["utils/src/parse_vgf.hpp"],
    includes = [
        "include",
        "schema",
        "src",
        "utils/src",
    ],
    deps = [
        ":vgf",
        "@flatbuffers//:runtime_cc",
    ],
)
