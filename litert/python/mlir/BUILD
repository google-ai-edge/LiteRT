load("@rules_python//python:py_library.bzl", "py_library")
load("//litert/build_common:symlink_files.bzl", "symlink_inputs")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "mlir",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":arithmetic_dialect",
        ":builtin_dialect",
        ":core",
        ":func_dialect",
        ":ir",
    ],
)

symlink_inputs(
    name = "core",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@llvm-project//mlir/python:DialectCorePyFiles"],
    }},
)

symlink_inputs(
    name = "extras",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "extras": ["@llvm-project//mlir/python:ExtrasPyFiles"],
    }},
    deps = [
        ":ir",
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "ir",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        ".": [
            "@llvm-project//mlir/python:IRPyFiles",
        ],
    }},
    deps = [
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "func_dialect",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@llvm-project//mlir/python:FuncPyFiles"],
    }},
    deps = [
        ":core",
        ":ir",
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "arithmetic_dialect",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@llvm-project//mlir/python:ArithOpsPyFiles"],
    }},
    deps = [
        ":core",
        ":ir",
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "builtin_dialect",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@llvm-project//mlir/python:BuiltinOpsPyFiles"],
    }},
    deps = [
        ":core",
        ":extras",
        ":ir",
        "//litert/python/mlir/_mlir_libs",
    ],
)

symlink_inputs(
    name = "pass_manager",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        ".": [
            "@llvm-project//mlir/python:PassManagerPyFiles",
        ],
    }},
    deps = [
        "//litert/python/mlir/_mlir_libs",
    ],
)
