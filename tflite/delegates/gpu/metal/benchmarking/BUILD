load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

objc_library(
    name = "benchmark_lib",
    srcs = ["main.mm"],
    data = glob([
        "models/*.tflite",
    ]),
    sdk_frameworks = [
        "Metal",
        "QuartzCore",
    ],
    deps = [
        "//tflite:framework",
        "//tflite:kernel_api",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_builder",
        "//tflite/delegates/gpu/common:model_transformer",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common/transformations:model_transformations",
        "//tflite/delegates/gpu/metal:inference_context",
        "//tflite/kernels:kernel_util",
    ],
)

# Application for local benchmarking in XCode
ios_application(
    name = "iOSBenchmark",
    testonly = 1,
    bundle_id = "com.tensorflow.lite.delegates.gpu.metal.benchmarking",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = "12.0",
    provisioning_profile = "//tflite/delegates/gpu/metal/benchmarking:provisioning_profile.mobileprovision",
    tags = [
        "local",
        "notap",
    ],
    deps = [":benchmark_lib"],
)
