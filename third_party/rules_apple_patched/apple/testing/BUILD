licenses(["notice"])

filegroup(
    name = "DefaultTestBundlePlist",
    srcs = [
        "DefaultTestBundle.plist",
    ],
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

# Consumed by bazel tests.
filegroup(
    name = "for_bazel_tests",
    testonly = True,
    srcs = glob(["**"]) + [
        "//apple/testing/default_runner:for_bazel_tests",
    ],
    visibility = ["//:__subpackages__"],
)
