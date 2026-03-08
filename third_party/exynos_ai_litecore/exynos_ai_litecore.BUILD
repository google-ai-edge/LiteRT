package(default_visibility = ["//visibility:public"])

licenses(["reciprocal"])

exports_files(["LICENSE"])  # LICENCE from the original repository

cc_library(
    name = "ai_litecore_headers",
    hdrs = glob(
        [
            "include/*.h",
        ],
    ),
    includes = [
        "include/",
    ],
    visibility = ["//visibility:public"],
)

exports_files(glob(["**/*.so"]))

