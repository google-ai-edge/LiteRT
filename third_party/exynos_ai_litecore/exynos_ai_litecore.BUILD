package(default_visibility = ["//visibility:public"])

licenses(["reciprocal"])

exports_files(["LICENSE"])  # LICENCE from the original repository

# ARM64-v8a library group
filegroup(
    name = "lib_arm64_v8a",
    srcs = glob(["lib/arm64-v8a/*.so"]),
)

# x86_64-linux library group
filegroup(
    name = "lib_x86_64_linux",
    srcs = glob(["lib/x86_64-linux/*.so"]),
)

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
