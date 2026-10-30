# The headers are downloaded from the pinned TensorFlow release. See
# //third_party:tensorflow_shim.bzl.

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "philox_random",
    hdrs = ["philox_random.h"],
)

cc_library(
    name = "random_distributions_utils",
    hdrs = ["random_distributions_utils.h"],
    deps = [":philox_random"],
)
