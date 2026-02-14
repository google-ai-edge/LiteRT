load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("//test/starlark_tests:common.bzl", "common")

package(
    default_visibility = ["//test/starlark_tests:__subpackages__"],
)

licenses(["notice"])

# The following targets reside under this package due to conflicting actions
# generating the compiled Swift module interface file (`.swiftmodule`) from
# this target and the associated framework/XCFramework import rules which
# compile the Swift module interface file.
# TODO: Use `library_evolution = True` and remove `features` once rules_swift 3+ is required.
swift_library(
    name = "swift_library",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.swift"],
    features = [
        "swift.emit_swiftinterface",
        "swift.enable_library_evolution",
    ],
    generates_header = True,
    module_name = "iOSSwiftStaticFramework",
    tags = common.fixture_tags,
)

# TODO: Use `library_evolution = True` and remove `features` once rules_swift 3+ is required.
swift_library(
    name = "swift_lib_for_static_xcframework",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.swift"],
    features = [
        "swift.emit_swiftinterface",
        "swift.enable_library_evolution",
    ],
    generates_header = True,
    module_name = "generated_swift_static_xcframework",
    tags = common.fixture_tags,
)

# Exports all files in this package as targets to make it easier to depend on them.
# Because these are for Starlark tests, we can scope the visibility to just this
# package.
exports_files(
    glob(["**"]),
    visibility = ["//test:__subpackages__"],
)
