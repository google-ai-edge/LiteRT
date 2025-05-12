"""Build definitions for TFLite Runtime."""

def tflxx_deps_if_enabled():
    """
    Returns a list of TFLite Runtime dependencies if enabled, otherwise an empty list.
    """
    return select(
        {
            "//litert/python/aot/core:tflxx_enabled": ["//litert/python/tools/model_utils/litert:passes"],
            "//conditions:default": [],
        },
    )
