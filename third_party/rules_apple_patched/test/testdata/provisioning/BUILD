# Public only because these are used by the integration tests from generated
# workspaces. Please no not depend on them as they can change at any time.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files([
    "integration_testing_ios.mobileprovision",
    "integration_testing_macos.provisionprofile",
    "integration_testing_tvos.mobileprovision",
    "integration_testing_visionos.mobileprovision",
])

filegroup(
    name = "integration_testing_profiles",
    srcs = [
        "BUILD",
        "integration_testing_ios.mobileprovision",
        "integration_testing_macos.provisionprofile",
        "integration_testing_tvos.mobileprovision",
        "integration_testing_visionos.mobileprovision",
    ],
)
