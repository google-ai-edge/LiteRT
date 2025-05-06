load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_binary")

package(
    default_visibility = ["//visibility:public"],
)

pytype_strict_binary(
    name = "sdist_wrapper",
    srcs = [
        "sdist_wrapper.py",
    ],
    main = "sdist_wrapper.py",
    deps = [
        "@pypi_setuptools//:pkg",
        "@pypi_wheel//:pkg",
    ],
)
