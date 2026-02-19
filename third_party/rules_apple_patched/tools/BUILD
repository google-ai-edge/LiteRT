licenses(["notice"])

# Consumed by bazel tests.
filegroup(
    name = "for_bazel_tests",
    testonly = True,
    srcs = glob(["**"]) + [
        "//tools/alticonstool:for_bazel_tests",
        "//tools/bitcode_strip:for_bazel_tests",
        "//tools/bundletool:for_bazel_tests",
        "//tools/clangrttool:for_bazel_tests",
        "//tools/codesigningtool:for_bazel_tests",
        "//tools/dossier_codesigningtool:for_bazel_tests",
        "//tools/environment_plist:for_bazel_tests",
        "//tools/imported_dynamic_framework_processor:for_bazel_tests",
        "//tools/main_thread_checker_tool:for_bazel_tests",
        "//tools/plisttool:for_bazel_tests",
        "//tools/provisioning_profile_tool:for_bazel_tests",
        "//tools/swift_stdlib_tool:for_bazel_tests",
        "//tools/versiontool:for_bazel_tests",
        "//tools/wrapper_common:for_bazel_tests",
        "//tools/xcframework_processor_tool:for_bazel_tests",
        "//tools/xctoolrunner:for_bazel_tests",
    ],
    visibility = [
        "//:__pkg__",
    ],
)
