load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

android_binary(
    name = "TfLiteCameraDemo",
    srcs = glob(["java/**/*.java"]),
    assets = [
        "//tflite/java/demo/app/src/main/assets:labels_mobilenet_quant_v1_224.txt",
        "@tflite_mobilenet_quant//:mobilenet_v1_1.0_224_quant.tflite",
        "@tflite_mobilenet_float//:mobilenet_v1_1.0_224.tflite",
    ],
    assets_dir = "",
    custom_package = "com.example.android.tflitecamerademo",
    manifest = "AndroidManifest.xml",
    multidex = "native",
    nocompress_extensions = [
        ".tflite",
    ],
    resource_files = glob(["res/**"]),
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    deps = [
        "//tflite/java:tensorflowlite",
        "//tflite/java:tensorflowlite_gpu",
        "//tflite/java/src/testhelper/java/org/tensorflow/lite:testhelper",
    ],
)
