load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "@org_tensorflow//tensorflow_lite_support:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "zip_writable_mem_file",
    srcs = ["zip_writable_mem_file.cc"],
    hdrs = ["zip_writable_mem_file.h"],
    deps = [
        "//third_party/minizip:zlib_minizip",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "zip_readonly_mem_file",
    srcs = ["zip_readonly_mem_file.cc"],
    hdrs = ["zip_readonly_mem_file.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "//third_party/minizip:zlib_minizip",
        "@com_google_absl//absl/strings",
    ],
)
