load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")

package(default_visibility = ["//apple/internal:__subpackages__"])

swift_library(
    name = "BinaryHeaderMapTool",
    srcs = [
        "BinaryHeaderMap.swift",
        "BinaryHeaderMapEncoder.swift",
        "BinaryHeaderMapTool.swift",
    ],
)

swift_binary(
    name = "hmaptool",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
    deps = [":BinaryHeaderMapTool"],
)
