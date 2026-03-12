"""TensorFlow Lite Build Configurations for iOS"""

load("@build_bazel_rules_apple//apple:apple.bzl", "apple_static_xcframework")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")

load("//tflite:build_def.bzl", "clean_dep")

# Minimum supported iOS version
TFL_MINIMUM_OS_VERSION = "12.0"

# Default tags for Apple targets
TFL_DEFAULT_TAGS = ["apple"]

# Sanitizers not supported on iOS
TFL_DISABLED_SANITIZER_TAGS = [
    "noasan",
    "nomsan",
    "notsan",
]


def _symbol_hiding_genrule(name, bundle_name, framework_target,
                           allowlist_symbols_file, hide_script):

    extract_script = clean_dep("//tflite/ios:extract_object_files_main")

    native.genrule(
        name = name,
        srcs = [
            framework_target,
            allowlist_symbols_file,
        ],
        outs = [name + ".zip"],
        tools = [
            extract_script,
            hide_script,
        ],
        cmd = """
INPUT_FRAMEWORK="$(location {framework})" \
BUNDLE_NAME="{bundle}" \
ALLOWLIST_FILE_PATH="$(location {allowlist})" \
EXTRACT_SCRIPT_PATH="$(location {extract})" \
OUTPUT="$@" \
"$(location {hide})"
""".format(
            framework = framework_target,
            bundle = bundle_name,
            allowlist = allowlist_symbols_file,
            extract = extract_script,
            hide = hide_script,
        ),
    )


def tflite_ios_framework(
        name,
        bundle_name,
        allowlist_symbols_file,
        exclude_resources = True,
        **kwargs):

    preprocessed_name = "Preprocessed_" + name

    ios_static_framework(
        name = preprocessed_name,
        bundle_name = bundle_name,
        exclude_resources = exclude_resources,
        **kwargs
    )

    framework_target = ":{}.zip".format(preprocessed_name)

    _symbol_hiding_genrule(
        name = name,
        bundle_name = bundle_name,
        framework_target = framework_target,
        allowlist_symbols_file = allowlist_symbols_file,
        hide_script = clean_dep("//tflite/ios:hide_symbols_with_allowlist"),
    )


def tflite_ios_xcframework(
        name,
        bundle_name,
        allowlist_symbols_file,
        **kwargs):

    preprocessed_name = "Preprocessed_" + name

    apple_static_xcframework(
        name = preprocessed_name,
        bundle_name = bundle_name,
        **kwargs
    )

    xcframework_target = ":{}.xcframework.zip".format(preprocessed_name)

    _symbol_hiding_genrule(
        name = name,
        bundle_name = bundle_name,
        framework_target = xcframework_target,
        allowlist_symbols_file = allowlist_symbols_file,
        hide_script = clean_dep("//tflite/ios:hide_xcframework_symbols_with_allowlist"),
    )


def strip_common_include_path_prefix(name, hdr_labels, prefix = ""):

    for hdr_label in hdr_labels:
        hdr_filename = hdr_label.split(":")[-1]
        hdr_basename = hdr_filename.split(".")[0]

        native.genrule(
            name = "{}_{}".format(name, hdr_basename),
            srcs = [hdr_label],
            outs = [hdr_filename],
            cmd = """
sed -E 's|#include ".*/([^/]+\\.h)"|#include "{prefix}\\1"|g' \
"$(location {src})" > "$@"
""".format(
                prefix = prefix,
                src = hdr_label,
            ),
        )
