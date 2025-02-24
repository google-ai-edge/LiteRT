"""Build definitions for TFLite Runtime."""

def tflxx_deps_if_enabled():
    """
    Returns a list of TFLite Runtime dependencies if enabled, otherwise an empty list.
    """
    return select(
        {
            "//third_party/odml/litert/litert/python/google/core:tflxx_enabled": ["//%s/tflxx" % "experimental"],
            "//conditions:default": [],
        },
    )
