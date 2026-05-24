cc_binary(
    name = "libhailort.so",
    srcs = ["src/mock_hailort.cc"],
    linkshared = 1,
)

cc_library(
    name = "hailort",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    srcs = [":libhailort.so"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
