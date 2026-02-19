"""# Bazel rules for creating tvOS applications and bundles."""

# Re-export original rules rather than their wrapper macros
# so that stardoc documents the rule attributes, not an opaque
# **kwargs argument.
load(
    "//apple/internal:tvos_rules.bzl",
    _tvos_application = "tvos_application",
    _tvos_dynamic_framework = "tvos_dynamic_framework",
    _tvos_extension = "tvos_extension",
    _tvos_framework = "tvos_framework",
    _tvos_static_framework = "tvos_static_framework",
)
load(
    "//apple/internal/testing:tvos_rules.bzl",
    _tvos_ui_test = "tvos_ui_test",
    _tvos_unit_test = "tvos_unit_test",
)
load(":tvos.bzl", _tvos_build_test = "tvos_build_test")

tvos_application = _tvos_application
tvos_dynamic_framework = _tvos_dynamic_framework
tvos_extension = _tvos_extension
tvos_framework = _tvos_framework
tvos_static_framework = _tvos_static_framework
tvos_ui_test = _tvos_ui_test
tvos_unit_test = _tvos_unit_test
tvos_build_test = _tvos_build_test
