"""Common BUILD-related definitions across different tasks"""

load("//tflite:build_def.bzl", "tflite_linkopts")

def task_linkopts():
    return tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            # Hexagon delegate libraries should be in /data/local/tmp
            "-Wl,--rpath=/data/local/tmp/",
        ],
        "//conditions:default": [],
    })
