load("//litert/build_common:litert_build_defs.bzl", "litert_lib", "litert_test", "litert_dynamic_lib")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "common",
#    hdrs = ["common.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/cc:litert_model",
    ],
)
