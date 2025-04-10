load("//devtools/python/blaze:strict.bzl", "py_strict_library", "py_strict_test")
load("//research/colab:build_defs.bzl", "colab_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:private"],
)

py_strict_library(
    name = "model_utils",
    srcs = ["__init__.py"],
    visibility = ["//litert/python/google/tools/model_utils:model_utils_users"],
    deps = ["//litert/python/google/tools/model_utils"],
)

py_strict_library(
    name = "visualization",
    visibility = ["//litert/python/google/tools/model_utils:model_utils_users"],
    deps = [
        "//litert/python/google/tools/model_utils:visualization",
    ],
)

py_strict_test(
    name = "model_utils_availability_test",
    srcs = ["model_utils_availability_test.py"],
    deps = [
        ":model_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

colab_binary(
    name = "colab",
    exec_properties = {
        "py_par.mem": "28g",
    },
    g3_multiprocessing = True,
    paropts = ["--compress"],
    tags = ["optonly"],
    deps = [
        ":model_utils",
        ":visualization",
        "//file/colossus/public:cns",
        "//file/google_src",
        "//file/liball",
        "//file/memfile:inlinefile",
        "//file/memfile/python:inlinefile",
        "//file/xfile",
        "//pyglib:build_data",
        "//pyglib:file_util",
        "//pyglib:gfile",
        "//pyglib:timer",
        "//pyglib/contrib/g3_multiprocessing:additional_reductions",
        "//pyglib/contrib/gpathlib:gpath",
        "//pyglib/contrib/gpathlib:gpath_flag",
        "//pyglib/function_utils:memoize",
        "//tech/env:envelope_loader",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "//third_party/py/ai_edge_jax",
        "//third_party/py/ai_edge_torch",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental/jax2tf",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/torch:pytorch",
    ],
)
