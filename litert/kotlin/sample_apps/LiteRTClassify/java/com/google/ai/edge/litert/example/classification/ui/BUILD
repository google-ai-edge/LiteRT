load("//tools/build_defs/android:rules.bzl", "android_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert/kotlin/sample_apps/LiteRTClassify:__subpackages__",
    ],
)

android_library(
    name = "ui",
    srcs = glob(["*.kt"]),
    deps = [
        "//third_party/java/androidx/recyclerview",
        "//litert/kotlin/sample_apps/LiteRTClassify:res",
        "//litert/kotlin/sample_apps/LiteRTClassify/java/com/google/ai/edge/litert/example/classification/viewmodel",
    ],
)
