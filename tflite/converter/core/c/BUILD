load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite/converter:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        ":tflite_types.h",
    ],
    visibility = ["//tflite:__pkg__"],
)

filegroup(
    name = "lite_headers_filegroup",
    srcs = [
        "builtin_op_data.h",
        "tflite_types.h",
    ],
    visibility = ["//tflite:__subpackages__"],
)

exports_files(
    srcs = [
        "builtin_op_data.h",
        "tflite_types.h",
    ],
    visibility = [
        # copybara:uncomment_begin(google-only)
        # "//tflite:__subpackages__",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "tflite_types",
    hdrs = [
        "tflite_types.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//tflite/ios:__subpackages__"],
)

# LINT.IfChange(common)
cc_library(
    name = "tflite_common",
    srcs = [],
    hdrs = [
        "builtin_op_data.h",
        "tflite_types.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tflite/converter:__subpackages__",
        "//tflite/core/c:__subpackages__",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/utils:__pkg__",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)
# LINT.ThenChange(//tflite/core/c:common)
