# Copyright 2026 Samsung Electronics Co. LTD.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
)
# licenses(["reciprocal"])

# ARM64-v8a library group
filegroup(
    name = "lib_arm64_v8a",
    srcs = glob(["lib/arm64-v8a/*.so"]),
)

# x86_64-linux library group
filegroup(
    name = "lib_x86_64_linux",
    srcs = glob(["lib/x86_64-linux/*.so"]),
)
