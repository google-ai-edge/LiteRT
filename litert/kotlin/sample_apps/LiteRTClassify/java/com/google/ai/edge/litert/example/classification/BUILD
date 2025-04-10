load("//tools/build_defs/android:rules.bzl", "android_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert/kotlin/sample_apps/LiteRTClassify:__subpackages__",
    ],
)

android_library(
    name = "classification",
    srcs = glob(["*.kt"]),
    deps = [
        "//third_party/java/androidx/activity",
        "//third_party/java/androidx/annotation/experimental",
        "//third_party/java/androidx/appcompat",
        "//third_party/java/androidx/camera/camera2",
        "//third_party/java/androidx/camera/core",
        "//third_party/java/androidx/camera/lifecycle",
        "//third_party/java/androidx/camera/view",
        "//third_party/java/androidx/core",
        "//third_party/java/androidx/lifecycle/livedata_core",
        "//third_party/java/androidx/recyclerview",
        "//litert/kotlin/sample_apps/LiteRTClassify:res",
        "//litert/kotlin/sample_apps/LiteRTClassify/java/com/google/ai/edge/litert/example/classification/litert",
        "//litert/kotlin/sample_apps/LiteRTClassify/java/com/google/ai/edge/litert/example/classification/ui",
        "//litert/kotlin/sample_apps/LiteRTClassify/java/com/google/ai/edge/litert/example/classification/util",
        "//litert/kotlin/sample_apps/LiteRTClassify/java/com/google/ai/edge/litert/example/classification/viewmodel",
        "//third_party/tensorflow_lite_support/java:tensorflowlite_support_java",
    ],
)
