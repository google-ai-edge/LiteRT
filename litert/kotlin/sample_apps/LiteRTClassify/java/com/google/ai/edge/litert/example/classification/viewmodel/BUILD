load("//tools/build_defs/android:rules.bzl", "android_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert/kotlin/sample_apps/LiteRTClassify:__subpackages__",
    ],
)

android_library(
    name = "viewmodel",
    srcs = glob(["*.kt"]),
    deps = [
        "//third_party/java/androidx/lifecycle",
        "//third_party/java/androidx/lifecycle/livedata_core",
        "//third_party/java/androidx/lifecycle/viewmodel",
    ],
)
