load("//tools/build_defs/android:rules.bzl", "android_application", "android_library")

package(default_applicable_licenses = ["//third_party/odml:license"])

kt_android_library(
    name = "activity",
    srcs = ["BenchmarkActivity.kt"],
    deps = [
        "//java/com/google/android/libraries/material/gm3/theme",
        "//third_party/java/androidx/activity",
        "//third_party/java/androidx/activity/compose",
        "//third_party/java/androidx/compose/foundation",
        "//third_party/java/androidx/compose/foundation/layout",
        "//third_party/java/androidx/compose/material3",
        "//third_party/java/androidx/compose/runtime",
        "//litert/kotlin:litert_kotlin_api",
    ],
)

android_application(
    name = "benchmark",
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.google.ai.edge.litert.benchmark",
        "targetSdkVersion": "35",
    },
    deps = [":activity"],
)
