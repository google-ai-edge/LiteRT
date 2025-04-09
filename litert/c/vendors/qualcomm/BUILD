# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "accelerator_options",
    srcs = ["accelerator_options.cc"],
    hdrs = ["accelerator_options.h"],
    deps = [
        "//litert/c:litert_compilation_options",
        "//litert/c:litert_common",
    ],
)

# cc_library(
#     name = "set_options_example",
#     srcs = ["set_options_example.cc"],
#     deps = [
#         ":accelerator_options",
#         "//litert/c:litert_accelerator_options",
#         "//litert/c:litert_common",
#         "//litert/c:litert_compiled_model",
#         "//litert/c:litert_compiled_model_options",
#         "//litert/cc:litert_macros",
#     ],
# )