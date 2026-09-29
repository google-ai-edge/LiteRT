# The sources are downloaded from the pinned TensorFlow release. See
# //third_party:tensorflow_shim.bzl.

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "example_protos",
    srcs = [
        "example.proto",
        "feature.proto",
    ],
)

# Keeps the name TensorFlow uses, which tflite/ BUILD files depend on.
# buildifier: disable=cc_proto_library-naming
cc_proto_library(
    name = "example_protos_cc",
    deps = [":example_protos"],
)

alias(
    name = "example_protos_cc_impl",
    actual = ":example_protos_cc",
)
