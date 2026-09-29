# The sources are downloaded from the pinned TensorFlow release. See
# //third_party:tensorflow_shim.bzl.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Matches the pinned TensorFlow release in //third_party:tensorflow_shim.bzl.
cc_library(
    name = "release_version",
    hdrs = ["release_version.h"],
    defines = [
        "TF_MAJOR_VERSION=2",
        "TF_MINOR_VERSION=21",
        "TF_PATCH_VERSION=0",
        'TF_VERSION_SUFFIX=\\"\\"',
    ],
)
