# Description:
#   BenchmarkModel Android harness for TensorFlow Lite benchmarks.

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_jni_binary")
load("//tflite:special_rules.bzl", "tflite_hexagon_nn_skel_libraries")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

# See README.md for details about building and executing the benchmark in APK
# format.

APK_VARIANTS = [
    # (suffix, extra deps)
    ("", []),
    (
        "_plus_flex",
        ["//tflite/delegates/flex:delegate"],
    ),
]

[android_binary(
    name = "benchmark_model%s" % suffix,
    srcs = glob([
        "src/**/*.java",
    ]),
    custom_package = "org.tensorflow.lite.benchmark",
    manifest = "AndroidManifest.xml",
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    deps = [
        ":hexagon_libs",
        ":tensorflowlite_benchmark_native%s" % suffix,
    ],
) for suffix, _ in APK_VARIANTS]

[tflite_jni_binary(
    name = "libtensorflowlite_benchmark%s.so" % suffix,
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
    deps = [
        "//tflite/java/jni",
        "//tflite/tools/benchmark:benchmark_tflite_model_lib",
    ] + extra_deps,
) for suffix, extra_deps in APK_VARIANTS]

[cc_library(
    name = "tensorflowlite_benchmark_native%s" % suffix,
    srcs = ["libtensorflowlite_benchmark%s.so" % suffix],
    visibility = ["//visibility:private"],
) for suffix, _ in APK_VARIANTS]

cc_library(
    name = "hexagon_libs",
    srcs = select({
        "@org_tensorflow//tensorflow:android_arm64": [
            "//tflite/delegates/hexagon/hexagon_nn:libhexagon_interface.so",
        ] + tflite_hexagon_nn_skel_libraries(),
        "@org_tensorflow//tensorflow:android_arm": [
            "//tflite/delegates/hexagon/hexagon_nn:libhexagon_interface.so",
        ] + tflite_hexagon_nn_skel_libraries(),
        "//conditions:default": [],
    }),
    visibility = ["//visibility:private"],
)
