# Description: Nice progress bars for Python.

package(default_visibility = ["//visibility:public"])

licenses(["reciprocal"])

exports_files(["LICENSE"])  # LICENCE from the original repository

py_library(
    name = "tqdm",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "tests/**/*",
        ],
    ),
    visibility = ["//visibility:public"],
)
