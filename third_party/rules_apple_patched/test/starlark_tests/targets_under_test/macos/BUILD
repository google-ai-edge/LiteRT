load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//apple:apple.bzl",
    "apple_dynamic_framework_import",
    "apple_dynamic_xcframework_import",
    "apple_static_framework_import",
    "apple_static_xcframework",
    "apple_xcframework",
)
load(
    "//apple:macos.bzl",
    "macos_application",
    "macos_bundle",
    "macos_command_line_application",
    "macos_dylib",
    "macos_dynamic_framework",
    "macos_extension",
    "macos_framework",
    "macos_kernel_extension",
    "macos_quick_look_plugin",
    "macos_static_framework",
    "macos_ui_test",
    "macos_unit_test",
)
load(
    "//apple:resources.bzl",
    "apple_resource_group",
)
load(
    "//test/starlark_tests:common.bzl",
    "common",
)
load(
    "//test/starlark_tests/rules:generate_framework.bzl",
    "generate_import_framework",
)
load("//test/starlark_tests/rules:generate_framework_dsym.bzl", "generate_import_framework_dsym")
load(
    "//test/starlark_tests/rules:generate_xcframework.bzl",
    "generate_dynamic_xcframework",
)
load("//test/testdata/rules:substitution.bzl", "substitution")

licenses(["notice"])

package(
    default_testonly = True,
    default_visibility = ["//test/starlark_tests:__subpackages__"],
)

# ---------------------------------------------------------------------------------------
# Targets for the app/test resource deduping test.

objc_library(
    name = "shared_lib",
    data = [
        "//test/starlark_tests/resources:basic_bundle",
    ],
    tags = common.fixture_tags,
)

objc_library(
    name = "app_lib",
    data = [
        "//test/starlark_tests/resources:empty.strings",
    ],
    tags = common.fixture_tags,
    deps = [":shared_lib"],
)

objc_library(
    name = "test_lib",
    data = [
        "//test/starlark_tests/resources:nonlocalized.strings",
    ],
    tags = common.fixture_tags,
    deps = [":app_lib"],
)

objc_library(
    name = "main_lib",
    tags = common.fixture_tags,
    deps = [
        ":app_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_framework(
    name = "dedupe_test_fmwk",
    bundle_id = "com.fmwk",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":shared_lib",
    ],
)

macos_application(
    name = "dedupe_test_app",
    bundle_id = "com.app",
    frameworks = [
        ":dedupe_test_fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_macos.provisionprofile",
    tags = common.fixture_tags,
    deps = [
        ":main_lib",
    ],
)

macos_unit_test(
    name = "dedupe_test_test",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    test_host = ":dedupe_test_app",
    deps = [
        ":test_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for macOS dynamic framework tests.

swift_library(
    name = "basic_framework_lib",
    srcs = [
        "//test/starlark_tests/resources:BasicFramework.swift",
    ],
    features = ["swift.no_generated_module_map"],
    generates_header = True,
    module_name = "BasicFramework",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
)

macos_dynamic_framework(
    name = "basic_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "BasicFramework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_lib",
    ],
)

swift_library(
    name = "basic_framework_with_direct_dependency_lib",
    srcs = [
        "//test/starlark_tests/resources:DirectDependencyTest.swift",
    ],
    features = ["swift.no_generated_module_map"],
    generates_header = True,
    module_name = "DirectDependencyTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":basic_framework_lib",
    ],
)

macos_dynamic_framework(
    name = "basic_framework_with_direct_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "DirectDependencyTest",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_direct_dependency_lib",
    ],
)

swift_library(
    name = "basic_framework_with_transitive_dependency_lib",
    srcs = [
        "//test/starlark_tests/resources:TransitiveDependencyTest.swift",
    ],
    features = ["swift.no_generated_module_map"],
    generates_header = True,
    module_name = "TransitiveDependencyTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":basic_framework_lib",
        ":basic_framework_with_direct_dependency_lib",
    ],
)

macos_dynamic_framework(
    name = "basic_framework_with_transitive_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "TransitiveDependencyTest",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_transitive_dependency_lib",
    ],
)

swift_library(
    name = "basic_framework_with_dynamic_framework_dependency_lib",
    srcs = [
        "//test/starlark_tests/resources:DirectDependencyTest.swift",
    ],
    features = ["swift.no_generated_module_map"],
    module_name = "DirectDependencyWithDynamicFrameworkTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":basic_framework",
    ],
)

macos_dynamic_framework(
    name = "basic_framework_with_dynamic_framework_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "DirectDependencyWithDynamicFrameworkTest",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_dynamic_framework_dependency_lib",
    ],
)

swift_library(
    name = "basic_framework_with_transitive_dependency_with_dynamic_frameworks_lib",
    srcs = [
        "//test/starlark_tests/resources:TransitiveDependencyTest.swift",
    ],
    features = ["swift.no_generated_module_map"],
    module_name = "TransitiveDependencyWithDynamicFrameworksTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":basic_framework",
        ":basic_framework_with_direct_dependency",
    ],
)

macos_dynamic_framework(
    name = "basic_framework_with_transitive_dependency_with_dynamic_frameworks",
    bundle_id = "com.google.example.framework",
    bundle_name = "TransitiveDependencyWithDynamicFrameworksTest",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_transitive_dependency_with_dynamic_frameworks_lib",
    ],
)

macos_dynamic_framework(
    name = "dynamic_framework_with_resources",
    bundle_id = "com.google.example.framework",
    bundle_name = "swift_lib_with_macos_resources",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-application_extension"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_structured_macos_resources",
    ],
)

macos_application(
    name = "app_with_dynamic_framework_and_resources",
    bundle_id = "com.google.example",
    frameworks = [
        ":dynamic_framework_with_resources",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:macos_swift_lib_with_resources",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_dynamic_framework(
    name = "swift_common_lib_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "swift_common_lib",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/resources:swift_common_lib"],
)

macos_dynamic_framework(
    name = "swift_shared_lib_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "swift_shared_lib",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_shared_lib",
    ],
)

macos_dynamic_framework(
    name = "swift_transitive_lib_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "swift_transitive_lib",
    frameworks = [":swift_shared_lib_framework"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/resources:swift_transitive_lib"],
)

macos_application(
    name = "app_with_dynamic_framework_with_dynamic_framework",
    bundle_id = "com.google.example",
    frameworks = [":swift_transitive_lib_framework"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:macos_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_dynamic_framework(
    name = "dynamic_fmwk_with_multiple_dependencies",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_lib",
        ":basic_framework_with_direct_dependency_lib",
    ],
)

macos_dynamic_framework(
    name = "dynamic_fmwk_with_objc_library",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

macos_static_framework(
    name = "objc_static_framework",
    hdrs = [
        "//test/starlark_tests/resources:common.h",
    ],
    bundle_name = "objc_static_framework",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_common_lib",
    ],
)

macos_static_framework(
    name = "objc_static_framework_without_modulemap",
    bundle_name = "objc_static_framework_without_modulemap",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_common_lib",
    ],
)

macos_static_framework(
    name = "objc_static_framework_with_sdk_dylibs_dep",
    bundle_name = "objc_static_framework_with_sdk_dylibs_dep",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_with_sdk_dylibs",
    ],
)

macos_static_framework(
    name = "objc_static_framework_with_sdk_fmwks_dep",
    bundle_name = "objc_static_framework_with_sdk_fmwks_dep",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_with_sdk_fmwks",
    ],
)

swift_library(
    name = "basic_framework_with_dynamic_framework_import_lib",
    srcs = [
        "//test/starlark_tests/resources:BasicFramework.swift",
    ],
    features = ["swift.no_generated_module_map"],
    module_name = "DynamicFrameworkImportTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":macOSImportedDynamicFramework",
    ],
)

macos_dynamic_framework(
    name = "basic_framework_with_dynamic_framework_import",
    bundle_id = "com.google.example.framework",
    bundle_name = "DynamicFrameworkImportTest",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_dynamic_framework_import_lib",
    ],
)

swift_library(
    name = "basic_framework_with_static_framework_import_lib",
    srcs = [
        "//test/starlark_tests/resources:BasicFramework.swift",
    ],
    features = ["swift.no_generated_module_map"],
    module_name = "StaticFrameworkImportTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":macOSImportedStaticFramework",
    ],
)

macos_dynamic_framework(
    name = "basic_framework_with_static_framework_import",
    bundle_id = "com.google.example.framework",
    bundle_name = "StaticFrameworkImportTest",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_static_framework_import_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for macOS static framework tests.

macos_static_framework(
    name = "swift_macos_static_framework",
    bundle_name = "SwiftFmwk",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [":SwiftFmwk"],
)

macos_static_framework(
    name = "static_fmwk_with_swift_and_avoid_deps",
    avoid_deps = [":SwiftFmwkLowerLib"],
    bundle_name = "SwiftFmwkUpperLib",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [":SwiftFmwkUpperLib"],
)

macos_static_framework(
    name = "static_framework_with_generated_header",
    bundle_name = "SwiftFmwkWithGenHeader",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [":SwiftFmwkWithGenHeader"],
)

macos_static_framework(
    name = "static_framework_from_objc_library",
    minimum_os_version = common.min_os_macos.baseline,
    deps = [":objc_library_with_sdk_dylib_and_framework"],
)

macos_static_framework(
    name = "static_framework_with_header_and_exclude_resources",
    hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    avoid_deps = [":framework_dependent_lib_with_resource"],
    exclude_resources = True,
    minimum_os_version = common.min_os_macos.baseline,
    deps = [":static_framework_lib_with_resources"],
)

macos_static_framework(
    name = "static_framework_with_no_header_and_exclude_resources",
    avoid_deps = [":framework_dependent_lib_with_resource"],
    exclude_resources = True,
    minimum_os_version = common.min_os_macos.baseline,
    deps = [":static_framework_lib_with_resources"],
)

macos_static_framework(
    name = "static_framework_with_transitive_resources",
    hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    avoid_deps = [":framework_dependent_lib_with_resource"],
    exclude_resources = False,
    minimum_os_version = common.min_os_macos.baseline,
    deps = [":static_framework_lib_with_resources"],
)

apple_static_framework_import(
    name = "macOSImportedStaticFramework",
    features = ["-swift.layering_check"],
    framework_imports = [":macOSStaticFramework"],
)

generate_import_framework(
    name = "macOSStaticFramework",
    src = "//test/starlark_tests/resources/frameworks:SharedClass.m",
    archs = ["x86_64"],
    include_resource_bundle = True,
    include_versioned_frameworks = False,
    libtype = "static",
    minimum_os_version = "11.11",
    sdk = "macosx",
)

genrule(
    name = "dummy_swift_src",
    outs = ["Dummy.swift"],
    cmd = "echo 'public struct Dummy {}' > $@",
)

swift_library(
    name = "SwiftFmwk",
    srcs = ["Dummy.swift"],
    module_name = "SwiftFmwk",
    tags = common.fixture_tags,
)

swift_library(
    name = "SwiftFmwkUpperLib",
    srcs = ["Dummy.swift"],
    module_name = "SwiftFmwkUpperLib",
    tags = common.fixture_tags,
    deps = [":SwiftFmwkLowerLib"],
)

swift_library(
    name = "SwiftFmwkLowerLib",
    srcs = ["Dummy.swift"],
    module_name = "SwiftFmwkLowerLib",
    tags = common.fixture_tags,
    deps = [":SwiftFmwkLowestLib"],
)

swift_library(
    name = "SwiftFmwkLowestLib",
    srcs = ["Dummy.swift"],
    module_name = "SwiftFmwkLowestLib",
    tags = common.fixture_tags,
)

swift_library(
    name = "SwiftFmwkWithGenHeader",
    srcs = ["Dummy.swift"],
    generates_header = True,
    module_name = "SwiftFmwkWithGenHeader",
    tags = common.fixture_tags,
)

apple_dynamic_framework_import(
    name = "bundle_only_imported_fmwk",
    bundle_only = True,
    framework_imports = [":generated_macos_dynamic_fmwk"],
    tags = common.fixture_tags,
)

objc_library(
    name = "objc_library_with_sdk_dylib_and_framework",
    linkopts = ["-lc++"],
    sdk_dylibs = ["sqlite3"],
    sdk_frameworks = ["AppKit"],
    tags = common.fixture_tags,
    deps = [
        ":fmwk_lib",
    ],
)

objc_library(
    name = "static_framework_lib_with_resources",
    srcs = [
        "//test/starlark_tests/resources:shared.h",
        "//test/starlark_tests/resources:shared.m",
    ],
    data = [
        "//test/testdata/resources:assets",
        "//test/testdata/resources:nonlocalized.strings",
        "//test/testdata/resources:storyboard_macos.storyboard",
        "//test/testdata/resources:unversioned_datamodel",
        "//test/testdata/resources:versioned_datamodel",
    ],
    tags = common.fixture_tags,
    deps = [":framework_dependent_lib_with_resource"],
    alwayslink = True,
)

objc_library(
    name = "framework_dependent_lib_with_resource",
    data = [
        "//test/testdata/resources:storyboard_macos.storyboard",
    ],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for Apple Frameworks propagated from objc_library/swift_library data attribute tests.

macos_application(
    name = "app_with_fmwks_from_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_macos_framework",
        ":swift_lib_with_macos_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_fmwks_from_transitive_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_inner_lib_with_macos_framework",
        ":swift_lib_with_macos_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_fmwk_and_ext_with_objc_lib_with_nested_macos_framework",
    bundle_id = "com.google.example",
    extensions = [":ext_with_objc_lib_with_nested_macos_framework"],
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_nested_macos_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_extension(
    name = "ext_with_objc_lib_with_nested_macos_framework",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_nested_macos_framework",
    ],
)

objc_library(
    name = "objc_lib_with_nested_macos_framework",
    srcs = [
        "//test/starlark_tests/resources:shared.h",
        "//test/starlark_tests/resources:shared.m",
    ],
    data = [":fmwk_with_fmwk"],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for importing XCFramework tests.

macos_application(
    name = "app_with_imported_dynamic_xcfmwk",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:swift_lib_importing_imported_dynamic_xcfw",
    ],
)

macos_application(
    name = "app_with_imported_dynamic_xcfmwk_with_lib_ids",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:swift_lib_importing_imported_dynamic_xcfw",
    ],
)

macos_application(
    name = "app_with_imported_static_xcfmwk",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_importing_imported_static_xcfw",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_imported_static_xcfmwk_with_lib_ids",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_importing_imported_static_xcfw",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_imported_xcfmwk_bundling_static_fmwks",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:swift_lib_importing_imported_xcfw_bundling_static_fmwks",
    ],
)

macos_application(
    name = "app_with_imported_xcfmwk_bundling_static_fmwks_with_resources",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/targets_under_test/apple:macos_static_xcframework_with_resources_import",
    ],
)

macos_application(
    name = "app_with_imported_xcfmwk_bundling_resources_and_data",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/targets_under_test/apple:macos_static_xcframework_with_resources_and_data",
    ],
)

macos_application(
    name = "app_with_imported_static_xcfmwk_with_module_map",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:swift_lib_importing_imported_static_xcfw",
    ],
)

macos_application(
    name = "app_min_os_baseline",
    bundle_id = "com.google.example",
    extensions = [":ext_min_os_baseline"],
    frameworks = [":fmwk_min_os_baseline"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":fmwk_lib",
        ":main_lib_with_fmwk_lib_dep",
    ],
)

macos_extension(
    name = "ext_min_os_baseline",
    bundle_id = "com.google.example.ext",
    frameworks = [":fmwk_min_os_baseline"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":fmwk_lib",
        ":main_lib_with_fmwk_lib_dep",
    ],
)

macos_framework(
    name = "fmwk_min_os_baseline",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":fmwk_lib",
    ],
)

objc_library(
    name = "fmwk_lib",
    srcs = [
        "//test/starlark_tests/resources:shared.h",
        "//test/starlark_tests/resources:shared.m",
    ],
    tags = common.fixture_tags,
)

objc_library(
    name = "main_lib_with_fmwk_lib_dep",
    srcs = [
        "//test/starlark_tests/resources:main.m",
    ],
    tags = common.fixture_tags,
    deps = [":fmwk_lib"],
)

macos_application(
    name = "app",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    app_icons = ["//test/starlark_tests/resources:MacAppIcon.xcassets"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:apple_localized_assets_lib",
        "//test/starlark_tests/resources:apple_non_localized_assets_lib",
        "//test/starlark_tests/resources:basic_bundle_lib",
        "//test/starlark_tests/resources:bundle_library_apple_lib",
        "//test/starlark_tests/resources:empty_strings_file_lib",
        "//test/starlark_tests/resources:nested_bundle_lib",
        "//test/starlark_tests/resources:objc_main_lib_with_common_lib",
    ],
)

macos_application(
    name = "app_basic_swift",
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

macos_application(
    name = "app_with_icon_bundle",
    app_icons = ["//test/starlark_tests/resources:icon_bundle_assets"],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.icon_bundle_required,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_icon_bundle_only_for_low_minimum_os_version",
    app_icons = ["//test/starlark_tests/resources:icon_bundle_assets"],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_icon_bundle_and_xcassets_app_icons",
    app_icons = [
        "//test/starlark_tests/resources:MacAppIcon.xcassets",
        "//test/starlark_tests/resources:icon_bundle_assets",
    ],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_multiple_infoplists",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Another.plist",
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_special_linkopts",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = [
        "-alias",
        "_main",
        "_linkopts_test_main",
        "-exported_symbol",
        "_linkopts_test_main",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_space",
    bundle_id = "com.google.example",
    bundle_name = "app with space",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_minimum_deployment_os_version",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_deployment_os_version = "11.0",
    minimum_os_version = "10.10",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

macos_application(
    name = "app_with_ext_with_fmwk_provisioned",
    bundle_id = "com.google.example",
    extensions = [":ext_with_fmwk_provisioned"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_framework_and_resources",
    bundle_id = "com.google.example",
    frameworks = [
        ":fmwk_with_resources",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

macos_framework(
    name = "fmwk_with_resources",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-application_extension"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

macos_application(
    name = "app_with_framework_and_shared_resources",
    bundle_id = "com.google.example",
    frameworks = [
        ":fmwk_with_resources",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":lib_with_structured_resources",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_multiple_refs_to_same_structured_resources",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":app_structured_resources_lib",
        ":fmwk_structured_resources_lib",
        ":lib_with_structured_resources",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_multiple_resource_bundles_with_shared_resources",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:bundle_library_apple_lib",
        "//test/starlark_tests/resources:bundle_library_macos_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_resource_bundle_with_structured_resource_group",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    resources = ["//test/starlark_tests/resources:resource_bundle_with_structured_resource_group"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_resource_bundle_with_bundle_id",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    resources = ["//test/starlark_tests/resources:resource_bundle_with_bundle_id"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_resource_group_with_resource_bundle",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    resources = ["//test/starlark_tests/resources:resources_with_resource_bundle"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "lib_with_structured_resources",
    data = [
        "//test/starlark_tests/resources:structured_resources",
    ],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

macos_application(
    name = "app_with_ext",
    bundle_id = "com.google.example",
    extensions = [":ext"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_extensionkit_ext",
    bundle_id = "com.google.example",
    extensions = [":extensionkit_ext"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_extension(
    name = "ext",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    bundle_id = "com.google.example.ext",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_extension(
    name = "extensionkit_ext",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    bundle_id = "com.google.example.ext",
    extensionkit_extension = True,
    infoplists = [
        "//test/starlark_tests/resources:ExtensionKitInfo.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_ext_and_symbols_in_bundle",
    bundle_id = "com.google.example",
    extensions = [":ext"],
    include_symbols_in_bundle = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for Apple dynamic framework import tests.

macos_application(
    name = "app_with_imported_fmwk",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_swift_ext",
    bundle_id = "com.google.example",
    extensions = [":swift_ext"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_ext_and_fmwk_provisioned",
    bundle_id = "com.google.example",
    extensions = [":ext_with_fmwk_provisioned"],
    frameworks = [
        ":fmwk_with_provisioning",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_ext_and_fmwk_and_symbols_in_bundle",
    bundle_id = "com.google.example",
    extensions = [":ext_with_fmwk_provisioned"],
    frameworks = [
        ":fmwk_with_provisioning",
    ],
    include_symbols_in_bundle = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":macOSImportedDynamicFrameworkWithDebugInfo",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_extension(
    name = "ext_multiple_infoplists",
    bundle_id = "com.google.example.ext",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
        "//test/starlark_tests/resources:Another.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_extension(
    name = "ext_with_fmwk_provisioned",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    frameworks = [
        ":fmwk_with_provisioning",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_extension(
    name = "swift_ext",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

apple_dynamic_framework_import(
    name = "macOSImportedDynamicFramework",
    features = ["-swift.layering_check"],
    framework_imports = [":macOSDynamicFramework"],
)

apple_dynamic_framework_import(
    name = "macOSImportedDynamicFrameworkWithDsym",
    dsym_imports = [":macOSDynamicFrameworkDsym"],
    framework_imports = [":macOSDynamicFramework"],
)

generate_import_framework(
    name = "macOSDynamicFramework",
    archs = ["x86_64"],
    include_resource_bundle = True,
    libtype = "dynamic",
    minimum_os_version = "11.11",
    sdk = "macosx",
)

generate_import_framework_dsym(
    name = "macOSDynamicFrameworkDsym",
    framework_imports = [":macOSDynamicFramework"],
)

apple_dynamic_framework_import(
    name = "macOSImportedDynamicFrameworkWithBitcode",
    framework_imports = [":macOSDynamicFrameworkWithBitcode"],
)

generate_import_framework(
    name = "macOSDynamicFrameworkWithBitcode",
    src = "//test/starlark_tests/resources/frameworks:SharedClass.m",
    archs = ["x86_64"],
    embed_bitcode = True,
    libtype = "dynamic",
    minimum_os_version = "11.11",
    sdk = "macosx",
)

apple_dynamic_framework_import(
    name = "macOSImportedDynamicFrameworkWithDebugInfo",
    framework_imports = [":macOSDynamicFrameworkWithDebugInfo"],
)

generate_import_framework(
    name = "macOSDynamicFrameworkWithDebugInfo",
    archs = ["x86_64"],
    embed_debug_info = True,
    libtype = "dynamic",
    minimum_os_version = "11.11",
    sdk = "macosx",
)

# ---------------------------------------------------------------------------------------

substitution(
    name = "resource_rules",
    src = "//test/starlark_tests/resources:ResourceRules.plist",
    tags = common.fixture_tags,
    var_name = "RULES",
)

macos_application(
    name = "app_with_codesignopts",
    bundle_id = "com.google.example",
    codesign_inputs = [":resource_rules"],
    codesignopts = [
        "--digest-algorithm=sha1",
        "--digest-algorithm=sha384",
        "--resource-rules=$(RULES)",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    toolchains = [":resource_rules"],
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_fmwk",
    bundle_id = "com.google.example",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:macos_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_bundle_only_fmwks",
    bundle_id = "com.google.example",
    frameworks = [":bundle_only_fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":bundle_only_imported_fmwk",
        "//test/starlark_tests/resources:macos_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_two_fmwk_provisioned",
    bundle_id = "com.google.example",
    frameworks = [
        ":fmwk_with_provisioning",
        ":second_fmwk_with_provisioning",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:macos_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_fmwk_with_fmwk_provisioned",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_fmwk_with_provisioning"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:macos_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_fmwk_importing_fmwk",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_provisioning_importing_fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:macos_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_fmwk_with_fmwk",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:macos_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_fmwk_with_multiple_objc_library_and_swift_library_deps",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_multiple_objc_library_and_swift_library_deps"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_framework(
    name = "fmwk_no_version",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info-noversion.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "fmwk_with_multiple_objc_library_and_swift_library_deps",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
        "//test/starlark_tests/resources:swift_shared_lib",
    ],
)

macos_framework(
    name = "fmwk",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "bundle_only_fmwk",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    bundle_only = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "fmwk_dead_stripped",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "fmwk_dead_stripped_two_exported_symbol_lists",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
        "//test/starlark_tests/resources:ExportDontCallMeShared.exp",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "fmwk_stripped",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-x"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "fmwk_with_provisioning",
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "fmwk_with_provisioning_importing_fmwk",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "second_fmwk_with_provisioning",
    bundle_id = "com.google.example.frameworktoo",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "fmwk_with_fmwk",
    bundle_id = "com.google.example.frameworkception",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
)

macos_framework(
    name = "fmwk_with_fmwk_with_provisioning",
    bundle_id = "com.google.example.frameworkception",
    frameworks = [":fmwk_with_provisioning"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "fmwk_multiple_infoplists",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
        "//test/starlark_tests/resources:Another.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_framework(
    name = "fmwk_with_dot.dynamic_framework",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

# ---------------------------------------------------------------------------------------

macos_framework(
    name = "fmwk_min_os_baseline_with_bundle",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:basic_bundle_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_application(
    name = "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_macos_framework",
        ":swift_lib_with_macos_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "objc_lib_with_inner_lib_with_macos_framework",
    srcs = ["//test/starlark_tests/resources:main.m"],
    tags = common.fixture_tags,
    deps = [":objc_lib_with_macos_framework"],
)

objc_library(
    name = "objc_lib_with_macos_framework",
    srcs = [
        "//test/starlark_tests/resources:shared.h",
        "//test/starlark_tests/resources:shared.m",
    ],
    data = [
        ":fmwk_with_resources",
        ":resource_group_with_framework",
    ],
    tags = common.fixture_tags,
)

apple_resource_group(
    name = "resource_group_with_framework",
    resources = [":fmwk_no_version"],
    tags = common.fixture_tags,
)

swift_library(
    name = "swift_lib_with_macos_framework",
    srcs = ["//test/starlark_tests/resources:DummySwiftFile"],
    data = [":fmwk_min_os_baseline_with_bundle"],
    tags = common.fixture_tags,
)

macos_application(
    name = "app_with_resources_and_fmwks_with_resources_from_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_macos_framework",
        ":swift_lib_with_macos_framework",
        "//test/starlark_tests/resources:another_basic_bundle_lib",
        "//test/starlark_tests/resources:objc_lib_with_resources",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_framework(
    name = "fmwk_stripped_two_exported_symbol_lists",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
        "//test/starlark_tests/resources:ExportDontCallMeShared.exp",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-x"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

# ---------------------------------------------------------------------------------------

macos_application(
    name = "app_with_same_resource_names_as_framework",
    bundle_id = "com.google.example",
    frameworks = [
        ":fmwk_with_duplicate_resource_names",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":app_structured_resources_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_framework(
    name = "fmwk_with_duplicate_resource_names",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":fmwk_structured_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

objc_library(
    name = "fmwk_structured_resources_lib",
    data = ["//test/starlark_tests/resources:structured_resources"],
    tags = common.fixture_tags,
)

objc_library(
    name = "app_structured_resources_lib",
    data = ["//test/starlark_tests/resources:structured_resources_duplicate"],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

macos_application(
    name = "app_with_transitive_swift_libraries_with_storyboards",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_macos_storyboard_a",
    ],
)

macos_application(
    name = "app_with_transitive_swift_libraries_with_asset_catalogs",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:macos_swift_lib_with_asset_catalog_a",
    ],
)

macos_application(
    name = "app_with_multiple_storyboards_in_bundle_root_from_multiple_swift_libraries",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_macos_storyboard_copy",
    ],
)

macos_application(
    name = "app_with_swift_library_scoped_resources",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_macos_resources",
    ],
)

macos_application(
    name = "app_with_transitive_swift_library_scoped_resources",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_transitive_macos_resources",
    ],
)

macos_application(
    name = "app_with_inner_and_outer_fmwk",
    bundle_id = "com.google.example",
    frameworks = [
        ":fmwk_with_imported_fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_framework(
    name = "fmwk_with_imported_fmwk",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":macOSImportedDynamicFramework",
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

macos_application(
    name = "app_with_imported_dynamic_fmwk_with_bitcode",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":macOSImportedDynamicFrameworkWithBitcode",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_imported_dynamic_fmwk_with_dsym",
    bundle_id = "com.google.example",
    include_symbols_in_bundle = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":macOSImportedDynamicFrameworkWithDebugInfo",
        ":macOSImportedDynamicFrameworkWithDsym",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_application(
    name = "app_with_swift_dep",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

macos_static_framework(
    name = "swift_static_framework",
    bundle_name = "swift_framework_lib",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_framework_lib_with_swiftinterface",
    ],
)

macos_application(
    name = "app_with_inner_and_outer_static_fmwk",
    bundle_id = "com.google.example",
    frameworks = [
        ":fmwk_with_imported_static_fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_framework(
    name = "fmwk_with_imported_static_fmwk",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":macOSImportedStaticFramework",
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

# ---------------------------------------------------------------------------------------

objc_library(
    name = "dynamic_fmwk_depending_lib",
    tags = common.fixture_tags,
    deps = [":imported_macos_dynamic_fmwk"],
)

apple_dynamic_framework_import(
    name = "imported_macos_dynamic_fmwk",
    framework_imports = [":generated_macos_dynamic_fmwk"],
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "generated_macos_dynamic_fmwk",
    archs = ["x86_64"],
    include_versioned_frameworks = False,
    libtype = "dynamic",
    minimum_os_version = common.min_os_macos.baseline,
    sdk = "macosx",
    tags = common.fixture_tags,
)

macos_application(
    name = "app_with_imported_versioned_fmwk",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_versioned_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "dynamic_versioned_fmwk_depending_lib",
    tags = common.fixture_tags,
    deps = [":imported_dynamic_versioned_fmwk"],
)

apple_dynamic_framework_import(
    name = "imported_dynamic_versioned_fmwk",
    framework_imports = [":generated_macos_dynamic_versioned_fmwk"],
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "generated_macos_dynamic_versioned_fmwk",
    archs = ["x86_64"],
    libtype = "dynamic",
    minimum_os_version = common.min_os_macos.baseline,
    sdk = "macosx",
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for Apple dynamic XCFramework import tests.

macos_application(
    name = "app_with_imported_dynamic_xcframework",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_xcframework_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "dynamic_xcframework_depending_lib",
    tags = common.fixture_tags,
    deps = [
        ":macos_imported_dynamic_xcframework",
    ],
)

apple_dynamic_xcframework_import(
    name = "macos_imported_dynamic_xcframework",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_dynamic_macos_xcframework"],
)

generate_dynamic_xcframework(
    name = "generated_dynamic_macos_xcframework",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    include_versioned_frameworks = False,
    minimum_os_versions = {
        "macos": common.min_os_macos.arm64_support,
    },
    platforms = {
        "macos": [
            "arm64",
            "arm64e",
            "x86_64",
        ],
    },
    tags = common.fixture_tags,
)

macos_application(
    name = "app_with_imported_dynamic_versioned_xcframework",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_versioned_xcframework_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "dynamic_versioned_xcframework_depending_lib",
    tags = common.fixture_tags,
    deps = [
        ":imported_dynamic_versioned_xcframework",
    ],
)

apple_dynamic_xcframework_import(
    name = "imported_dynamic_versioned_xcframework",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_dynamic_macos_versioned_xcframework"],
)

generate_dynamic_xcframework(
    name = "generated_dynamic_macos_versioned_xcframework",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    minimum_os_versions = {
        "macos": common.min_os_macos.arm64_support,
    },
    platforms = {
        "macos": [
            "arm64",
            "arm64e",
            "x86_64",
        ],
    },
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

macos_bundle(
    name = "bundle",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_bundle(
    name = "bundle_for_app",
    bundle_id = "com.google.example",
    bundle_loader = ":app",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_bundle(
    name = "bundle_for_cmd_app",
    bundle_id = "com.google.example",
    bundle_loader = ":cmd_app_basic",
    infoplists = [
        "//test/starlark_tests/resources:Info-nopackage.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_bundle(
    name = "bundle_dead_stripped",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    bundle_id = "com.google.example",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_bundle(
    name = "bundle_special_linkopts",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = [
        "-alias",
        "_anotherFunctionShared",
        "_linkopts_test_anotherFunctionShared",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_bundle(
    name = "bundle_with_different_extension",
    bundle_extension = "prefPane",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

# ---------------------------------------------------------------------------------------

macos_command_line_application(
    name = "cmd_app_basic",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_common_lib",
    ],
)

macos_command_line_application(
    name = "cmd_app_basic_swift",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

macos_command_line_application(
    name = "cmd_app_info_plists",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Another.plist",
        "//test/starlark_tests/resources:Info-nopackage.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_command_line_application(
    name = "cmd_app_info_and_launchd_plists",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Another.plist",
        "//test/starlark_tests/resources:Info-nopackage.plist",
    ],
    launchdplists = [
        "//test/starlark_tests/resources:Another.plist",
        "//test/starlark_tests/resources:launchd.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_command_line_application(
    name = "cmd_app_special_linkopts",
    linkopts = [
        "-alias",
        "_main",
        "_linkopts_test_main",
        "-u",
        "_linkopts_test_main",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_command_line_application(
    name = "cmd_app_dead_stripped",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionMain.exp",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_command_line_application(
    name = "cmd_app_basic_version",
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    version = "//test/starlark_tests/targets_under_test/apple:manual_1_2_build_1_2_3_version",
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

macos_dylib(
    name = "dylib",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info-nopackage.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_dylib(
    name = "dylib_dead_stripped",
    bundle_id = "com.google.example",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info-nopackage.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

# ---------------------------------------------------------------------------------------

macos_kernel_extension(
    name = "kext",
    bundle_id = "com.google.kext",
    infoplists = [
        "//test/starlark_tests/resources/kext_resources:Info-IOKit.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources/kext_resources:kext_lib",
    ],
)

macos_kernel_extension(
    name = "kext_dead_stripped",
    bundle_id = "com.google.kext",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
    ],
    infoplists = [
        "//test/starlark_tests/resources/kext_resources:Info-IOKit.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
        "//test/starlark_tests/resources/kext_resources:kext_lib",
    ],
)

# ---------------------------------------------------------------------------------------

macos_quick_look_plugin(
    name = "ql_plugin",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_quick_look_plugin(
    name = "ql_plugin_dead_stripped",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    bundle_id = "com.google.example",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionMain.exp",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_quick_look_plugin(
    name = "ql_plugin_special_linkopts",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = [
        "-alias",
        "_main",
        "_linkopts_test_main",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

macos_ui_test(
    name = "ui_test",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

macos_ui_test(
    name = "ui_test_custom_bundle_id",
    bundle_id = "my.test.bundle.id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

macos_ui_test(
    name = "ui_test_invalid_bundle_id",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

macos_ui_test(
    name = "ui_test_dummy_runner",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    runner = "//test/starlark_tests/targets_under_test/apple:dummy_test_runner",
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

# ---------------------------------------------------------------------------------------

macos_unit_test(
    name = "unit_test",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

macos_unit_test(
    name = "unit_test_no_host",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.arm64_support,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

macos_unit_test(
    name = "unit_test_custom_bundle_id",
    bundle_id = "my.test.bundle.id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

macos_unit_test(
    name = "unit_test_invalid_bundle_id",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

macos_unit_test(
    name = "unit_test_dummy_runner",
    additional_contents = {
        "//test/starlark_tests/resources:additional.txt": "Additional",
        "//test/starlark_tests/resources:all_nested": "Nested",
    },
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    runner = "//test/starlark_tests/targets_under_test/apple:dummy_test_runner",
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

macos_unit_test(
    name = "unit_test_data_deps",
    data = [":runfiles1"],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
)

filegroup(
    name = "runfiles1",
    # NOTE: we specify `data` instead of `srcs` because `data` puts the files
    # into runfiles, while `srcs` puts them into the default outputs.
    data = [
        "runfile1.txt",
        # Add a second-order dependency to check that the transitive runfiles
        # are being added.
        ":runfiles2",
    ],
    tags = common.fixture_tags,
)

filegroup(
    name = "runfiles2",
    # NOTE: we specify `data` instead of `srcs` because `data` puts the files
    # into runfiles, while `srcs` puts them into the default outputs.
    data = ["runfile2.txt"],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for AppIntents

macos_application(
    name = "app_with_app_intents",
    app_intents = [
        "//test/starlark_tests/resources:app_intent",
    ],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.app_intents_support,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_common_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for base_bundle_id and bundle_id flows with and without shared capabilities.

macos_application(
    name = "app_with_capability_set_derived_bundle_id",
    bundle_name = "app-with-capability-set-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_bundle(
    name = "bundle_with_capability_set_derived_bundle_id",
    bundle_name = "bundle-with-capability-set-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info-nopackage.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_command_line_application(
    name = "cmd_app_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "cmd-app-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info-nopackage.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_common_lib",
    ],
)

macos_dylib(
    name = "dylib_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "dylib-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info-nopackage.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

macos_extension(
    name = "ext_with_capability_set_derived_bundle_id",
    bundle_name = "ext-with-capability-set-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_kernel_extension(
    name = "kext_with_capability_set_derived_bundle_id",
    bundle_name = "kext-with-capability-set-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info-nopackage.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources/kext_resources:kext_lib",
    ],
)

macos_quick_look_plugin(
    name = "ql_plugin_with_capability_set_derived_bundle_id",
    bundle_name = "ql-plugin-with-capability-set-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

macos_ui_test(
    name = "ui_test_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "ui-test-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

macos_unit_test(
    name = "unit_test_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "unit-test-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets to test resource processing with C/C++ rules.

macos_application(
    name = "app_with_cc_library_with_runfiles",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:cc_library_with_runfile_a",
    ],
)

macos_application(
    name = "app_with_cc_library_with_resources",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:cc_library_with_resource_a",
    ],
)

macos_application(
    name = "app_with_cc_library_suppress_resources",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:cc_library_with_suppressed_resources",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for CcInfo dylib bundling tests

macos_application(
    name = "app_with_ccinfo_dylib_deps",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_macos.baseline,
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/resources:ccinfo_dylibs_libapp"],
)

# ---------------------------------------------------------------------------------------
# Targets for macOS xcframework tests.

apple_xcframework(
    name = "macos_dynamic_xcframework",
    bundle_id = "com.google.example",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    macos = [
        "arm64",
        "x86_64",
    ],
    minimum_os_versions = {
        "macos": common.min_os_macos.baseline,
    },
    public_hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    tags = common.fixture_tags,
    deps = [":fmwk_lib"],
)

apple_static_xcframework(
    name = "macos_static_xcframework",
    macos = [
        "arm64",
        "x86_64",
    ],
    minimum_os_versions = {
        "macos": common.min_os_macos.baseline,
    },
    public_hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    tags = common.fixture_tags,
    deps = [":fmwk_lib"],
)
