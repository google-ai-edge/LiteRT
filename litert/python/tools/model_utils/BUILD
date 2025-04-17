load("//research/colab:build_defs.bzl", "colab_binary", "colab_notebook_test")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

package_group(
    name = "model_utils_users",
    packages = [
        "//litert/python/...",
        "//litert/python/tools/model_utils/...",
    ],
)

py_library(
    name = "model_utils",
    srcs = glob(
        include = ["**/*.py"],
        exclude = [
            "tool/**/*",
            "test/**/*",
            "model_explorer_integration/**/*",
        ],
    ),
    visibility = [":model_utils_users"],
    deps = [
        "//pyglib:gfile",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "//litert/python/tools/model_utils/core/tblgen",
        "//third_party/py/jax",
        "//third_party/py/mlir",
        "//third_party/py/mlir:arithmetic_dialect",
        "//third_party/py/mlir:builtin_dialect",
        "//third_party/py/mlir:chlo_dialect",
        "//third_party/py/mlir:func_dialect",
        "//third_party/py/mlir:ir",
        "//third_party/py/mlir:pass_manager",
        "//third_party/py/mlir:stablehlo_dialect",
        "//third_party/py/mlir/_mlir_libs:_mlirRegisterEverything",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/xdsl",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/integrations:model_utils_core_pybind",
        "//tflite/tools:flatbuffer_utils",
    ],
)

py_library(
    name = "visualization",
    srcs = glob(include = ["model_explorer_integration/**/*"]),
    visibility = [":model_utils_users"],
    deps = [
        ":model_utils",
        "//third_party/py/model_explorer:model_explorer_lib",
        "//third_party/py/numpy",
        "//third_party/py/xdsl",
    ],
)

colab_notebook_test(
    name = "tutorial_notebook_test",
    colab_binary = ":colab",
    cpu = 4,
    default_cell_diff = "ignore",
    ipynb = "tutorial.ipynb",
    tags = [
        # colab binary is too large to build
        "nopresubmit",
        "nozapfhahn",
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
        "//pyglib:functional",
        "//pyglib:gfile",
        "//pyglib:timer",
        "//pyglib/contrib/g3_multiprocessing",
        "//pyglib/contrib/g3_multiprocessing:additional_reductions",
        "//pyglib/contrib/gpathlib:gpath",
        "//pyglib/contrib/gpathlib:gpath_flag",
        "//tech/env:envelope_loader",
        "//third_party/py/ai_edge_jax",
        "//third_party/py/ai_edge_torch",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental/jax2tf",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/torch:pytorch",
    ],
)
