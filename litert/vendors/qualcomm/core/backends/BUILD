# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "htp_device_config",
    srcs = ["htp_device_config.cc"],
    hdrs = ["htp_device_config.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//litert/vendors/qualcomm/core/schema:soc_table",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
    ],
)
