load(
    "//tools/build_defs/android:rules.bzl",
    "android_application",
    "android_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert/kotlin/sample_apps/LiteRTClassify:__subpackages__",
    ],
)

android_library(
    name = "res",
    custom_package = "com.google.ai.edge.litert.example.classification",
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    deps = [
        "//third_party/java/android_libs/material_components:sidesheet",
        "//third_party/java/androidx/appcompat",
        "//third_party/java/androidx/camera/view",
        "//third_party/java/androidx/recyclerview",
    ],
)

android_application(
    name = "LiteRTClassify",
    assets = glob(["assets/**"]),
    assets_dir = "assets",
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.google.ai.edge.litert.example.classification",
        "versionCode": "1",
        "versionName": "1.0.0",
        "targetSdkVersion": "34",
    },
    multidex = "native",
    deps = [
        ":res",
        "//litert/kotlin/sample_apps/LiteRTClassify/java/com/google/ai/edge/litert/example/classification",
    ],
)
