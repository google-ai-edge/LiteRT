"""# Bazel rules for creating watchOS applications and bundles."""

load(
    "//apple:watchos.bzl",
    _watchos_build_test = "watchos_build_test",
)
load(
    "//apple/internal:watchos_rules.bzl",
    _watchos_application = "watchos_application",
    _watchos_dynamic_framework = "watchos_dynamic_framework",
    _watchos_extension = "watchos_extension",
    _watchos_framework = "watchos_framework",
    _watchos_static_framework = "watchos_static_framework",
)

# Re-export original rules rather than their wrapper macros
# so that stardoc documents the rule attributes, not an opaque
# **kwargs argument.
load(
    "//apple/internal/testing:watchos_rules.bzl",
    _watchos_ui_test = "watchos_ui_test",
    _watchos_unit_test = "watchos_unit_test",
)

watchos_application = _watchos_application
watchos_dynamic_framework = _watchos_dynamic_framework
watchos_extension = _watchos_extension
watchos_framework = _watchos_framework
watchos_static_framework = _watchos_static_framework
watchos_ui_test = _watchos_ui_test
watchos_unit_test = _watchos_unit_test
watchos_build_test = _watchos_build_test
