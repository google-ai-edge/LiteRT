load("//tools/build_defs/android:rules.bzl", "android_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert/kotlin/sample_apps/LiteRTClassify:__subpackages__",
    ],
)

android_library(
    name = "litert",
    srcs = glob(["*.kt"]),
    deps = [
        "//litert/kotlin:litert_kotlin_api",
        "//tflite/java:tensorflowlite_api",
        "//third_party/tensorflow_lite_support/java:tensorflowlite_support_java",
    ],
)
