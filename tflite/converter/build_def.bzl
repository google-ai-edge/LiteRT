"""Build macros for TF Lite."""

load("//tflite:build_def.bzl", "clean_dep")
load("//tflite/converter:special_rules.bzl", "tflite_copts_extra")

# LINT.IfChange(tflite_copts)
def tflite_copts():
    """Defines common compile time flags for TFLite libraries."""
    copts = [
        "-DFARMHASH_NO_CXX_STRING",
        "-DEIGEN_ALLOW_UNALIGNED_SCALARS",  # TODO(b/296071640): Remove when underlying bugs are fixed.
    ] + select({
        clean_dep("@org_tensorflow//tensorflow:android_arm"): [
            "-mfpu=neon",
        ],
        # copybara:uncomment_begin(google-only)
        # clean_dep("@org_tensorflow//tensorflow:chromiumos_x86_64"): [],
        # copybara:uncomment_end
        clean_dep("@org_tensorflow//tensorflow:ios_x86_64"): [
            "-msse4.1",
        ],
        clean_dep("@org_tensorflow//tensorflow:linux_x86_64"): [
            "-msse4.2",
        ],
        clean_dep("@org_tensorflow//tensorflow:linux_x86_64_no_sse"): [],
        clean_dep("@org_tensorflow//tensorflow:windows"): [
            # copybara:uncomment_begin(no MSVC flags in google)
            # "-DTFL_COMPILE_LIBRARY",
            # "-Wno-sign-compare",
            # copybara:uncomment_end_and_comment_begin
            "/DTFL_COMPILE_LIBRARY",
            "/wd4018",  # -Wno-sign-compare
            # copybara:comment_end
        ],
        "//conditions:default": [
            "-Wno-sign-compare",
        ],
    }) + select({
        clean_dep("@org_tensorflow//tensorflow:optimized"): ["-O3"],
        "//conditions:default": [],
    }) + select({
        clean_dep("@org_tensorflow//tensorflow:android"): [
            "-ffunction-sections",  # Helps trim binary size.
            "-fdata-sections",  # Helps trim binary size.
        ],
        "//conditions:default": [],
    }) + select({
        clean_dep("@org_tensorflow//tensorflow:windows"): [],
        "//conditions:default": [
            "-fno-exceptions",  # Exceptions are unused in TFLite.
        ],
    }) + select({
        clean_dep("//tflite/converter:tflite_with_xnnpack_explicit_false"): ["-DTFLITE_WITHOUT_XNNPACK"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tflite/converter:tensorflow_profiler_config"): ["-DTF_LITE_TENSORFLOW_PROFILER"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tflite/converter/delegates:tflite_debug_delegate"): ["-DTFLITE_DEBUG_DELEGATE"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tflite/converter:tflite_mmap_disabled"): ["-DTFLITE_MMAP_DISABLED"],
        "//conditions:default": [],
    })

    return copts + tflite_copts_extra()

# LINT.ThenChange(//tflite/build_def.bzl:tflite_copts)

# LINT.IfChange(tflite_copts_warnings)
def tflite_copts_warnings():
    """Defines common warning flags used primarily by internal TFLite libraries."""

    # TODO(b/155906820): Include with `tflite_copts()` after validating clients.

    return select({
        clean_dep("@org_tensorflow//tensorflow:windows"): [
            # We run into trouble on Windows toolchains with warning flags,
            # as mentioned in the comments below on each flag.
            # We could be more aggressive in enabling supported warnings on each
            # Windows toolchain, but we compromise with keeping BUILD files simple
            # by limiting the number of config_setting's.
        ],
        "//conditions:default": [
            "-Wall",
        ],
    })

# LINT.ThenChange(//tflite/build_def.bzl:tflite_copts_warnings)
