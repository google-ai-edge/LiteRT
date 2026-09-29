# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal replacement for TensorFlow's tensorflow/tensorflow.bzl.

The exported tflite/ BUILD files load build macros from
@org_tensorflow//tensorflow:tensorflow.bzl. LiteRT does not depend on the
TensorFlow repository, so this file provides the subset of those macros with
the same OSS behavior, implemented with plain Bazel rules.

Macros that only make sense with real TensorFlow sources (custom ops, op
wrappers) create targets that fail when built.
"""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "@rules_ml_toolchain//py/rules_pywrap:pywrap.default.bzl",
    _pybind_extension = "pybind_extension",
    _use_pywrap_rules = "use_pywrap_rules",
)
load("@rules_python//python:py_test.bzl", _py_test = "py_test")

def clean_dep(target):
    """Returns a label string that resolves inside this repository."""
    return str(Label(target))

workspace_root = Label("//:WORKSPACE").workspace_root or "."

def if_google(
        google_value,  # @unused
        oss_value = []):
    return oss_value

def if_oss(
        oss_value,
        google_value = []):  # @unused
    return oss_value

def if_mobile(a):
    return select({
        clean_dep("//tensorflow:mobile"): a,
        "//conditions:default": [],
    })

def if_not_mobile(a):
    return select({
        clean_dep("//tensorflow:mobile"): [],
        "//conditions:default": a,
    })

def if_android(a):
    return select({
        clean_dep("//tensorflow:android"): a,
        "//conditions:default": [],
    })

def if_ios(a, otherwise = []):
    return select({
        clean_dep("//tensorflow:ios"): a,
        "//conditions:default": otherwise,
    })

def if_not_windows(a):
    return select({
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": a,
    })

def get_compatible_with_portable():
    return []

def if_portable(
        if_true,
        if_false = []):  # @unused
    return if_true

def _get_win_copts(is_external = False):
    copts = [
        "/DPLATFORM_WINDOWS",
        "/DEIGEN_HAS_C99_MATH",
        "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
        "/DEIGEN_AVOID_STL_ARRAY",
        "/Iexternal/gemmlowp",
        "/wd4018",
        "/wd4577",
        "/DNOGDI",
    ]
    if is_external:
        return copts + ["/UTF_COMPILE_LIBRARY"]
    return copts + ["/DTF_COMPILE_LIBRARY"]

def tf_copts(
        android_optimization_level_override = "-O2",
        is_external = False,
        allow_exceptions = False):
    """Returns the default TensorFlow copts for an OSS CPU-only build.

    Args:
      android_optimization_level_override: Optimization flag for Android, or
        None to keep the toolchain default.
      is_external: Whether the target is outside TensorFlow. Selects the
        Windows TF_COMPILE_LIBRARY define.
      allow_exceptions: Whether to keep C++ exceptions enabled.

    Returns:
      A list of compiler options.
    """
    android_copts = [
        "-DTF_LEAN_BINARY",
        "-Wno-narrowing",
    ]
    if android_optimization_level_override:
        android_copts.append(android_optimization_level_override)
    return (
        if_not_windows([
            "-DEIGEN_AVOID_STL_ARRAY",
            "-Iexternal/gemmlowp",
            "-Wno-sign-compare",
            "-ftemplate-depth=900",
        ]) +
        (if_not_windows(["-fno-exceptions"]) if not allow_exceptions else []) +
        select({
            clean_dep("//tensorflow:android_arm"): ["-mfpu=neon", "-fomit-frame-pointer"],
            "//conditions:default": [],
        }) +
        select({
            clean_dep("//tensorflow:linux_x86_64"): ["-msse3"],
            "//conditions:default": [],
        }) +
        select({
            clean_dep("//tensorflow:ios_x86_64"): ["-msse4.1"],
            "//conditions:default": [],
        }) +
        ["-DTENSORFLOW_MONOLITHIC_BUILD"] +
        select({
            clean_dep("//tensorflow:android"): android_copts,
            clean_dep("//tensorflow:emscripten"): [],
            clean_dep("//tensorflow:macos"): [],
            clean_dep("//tensorflow:windows"): _get_win_copts(is_external),
            clean_dep("//tensorflow:ios"): [],
            "//conditions:default": ["-pthread"],
        })
    )

def _tf_opts_nortti():
    return [
        "-fno-rtti",
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-DGOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
    ]

def _tf_defines_nortti():
    return [
        "GOOGLE_PROTOBUF_NO_RTTI",
        "GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
    ]

def tf_portable_full_lite_protos(full, lite):
    return select({
        clean_dep("//tensorflow:mobile_lite_protos"): lite,
        clean_dep("//tensorflow:mobile_full_protos"): full,
        "//conditions:default": full,
    })

def tf_opts_nortti_if_lite_protos():
    return tf_portable_full_lite_protos(full = [], lite = _tf_opts_nortti())

def tf_defines_nortti_if_lite_protos():
    return tf_portable_full_lite_protos(full = [], lite = _tf_defines_nortti())

def tf_opts_nortti_if_android():
    return if_android(_tf_opts_nortti())

def tf_opts_nortti_if_mobile():
    return if_mobile(_tf_opts_nortti())

def tf_features_nomodules_if_mobile():
    return if_mobile(["-use_header_modules"])

def _lrt_if_needed():
    lrt = ["-lrt"]
    return select({
        clean_dep("//tensorflow:linux_aarch64"): lrt,
        clean_dep("//tensorflow:linux_x86_64"): lrt,
        clean_dep("//tensorflow:linux_ppc64le"): lrt,
        "//conditions:default": [],
    })

def _make_search_paths(prefix, levels_to_root):
    return ",".join([
        "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
        for search_level in range(levels_to_root + 1)
    ])

def _rpath_linkopts(name):
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        clean_dep("//tensorflow:macos"): [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
            "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
        ],
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
        ],
    })

def _default_linkopts():
    return select({
        clean_dep("//tensorflow:android"): ["-pie"],
        clean_dep("//tensorflow:windows"): [],
        clean_dep("//tensorflow:macos"): ["-lm"],
        "//conditions:default": [
            "-lpthread",
            "-lm",
        ],
    })

def tf_cc_test(
        name,
        srcs,
        deps,
        data = [],
        extra_copts = [],
        suffix = "",
        linkopts = None,
        kernels = [],  # @unused
        **kwargs):
    if linkopts == None:
        linkopts = _lrt_if_needed()
    cc_test(
        name = "%s%s" % (name, suffix),
        srcs = srcs,
        copts = tf_copts() + extra_copts,
        linkopts = _default_linkopts() + linkopts + _rpath_linkopts(name),
        deps = deps,
        data = data,
        **kwargs
    )

def tf_cc_binary(
        name,
        srcs = [],
        deps = [],
        data = [],
        linkopts = None,
        copts = None,
        kernels = [],  # @unused
        per_os_targets = False,  # @unused
        visibility = None,
        default_copts = [],
        **kwargs):
    """Builds a cc_binary with TensorFlow's default copts and linkopts.

    Args:
      name: Target name.
      srcs: Source files.
      deps: Dependencies.
      data: Runtime data files.
      linkopts: Extra link options. Defaults to librt where needed.
      copts: Compiler options. Defaults to tf_copts().
      kernels: Unused. Kept for compatibility with TensorFlow's macro.
      per_os_targets: Unused. Kept for compatibility with TensorFlow's macro.
      visibility: Target visibility.
      default_copts: Compiler options placed before `copts`.
      **kwargs: Passed to cc_binary.
    """
    if linkopts == None:
        linkopts = _lrt_if_needed()
    if copts == None:
        copts = tf_copts()
    cc_binary(
        name = name,
        copts = default_copts + copts,
        srcs = srcs,
        deps = deps,
        data = data,
        linkopts = linkopts + _rpath_linkopts(name),
        visibility = visibility,
        **kwargs
    )

def tf_native_cc_binary(name, copts = None, linkopts = [], **kwargs):
    if copts == None:
        copts = tf_copts()
    cc_binary(
        name = name,
        copts = copts,
        linkopts = select({
            clean_dep("//tensorflow:windows"): [],
            clean_dep("//tensorflow:macos"): ["-lm"],
            "//conditions:default": [
                "-lpthread",
                "-lm",
            ],
        }) + linkopts + _rpath_linkopts(name) + _lrt_if_needed(),
        **kwargs
    )

# Shared library names for `per_os_targets`: Linux, macOS and Windows.
SHARED_LIBRARY_NAME_PATTERNS = [
    "lib%s.so%s",
    "lib%s%s.dylib",
    "%s%s.dll",
]

def tf_cc_shared_object(
        name,
        srcs = [],
        deps = [],
        data = [],
        linkopts = None,
        framework_so = [],  # @unused
        soversion = None,
        kernels = [],  # @unused
        per_os_targets = False,
        visibility = None,
        **kwargs):
    """Builds a shared object, like TensorFlow's macro without `framework_so`.

    Args:
      name: Target name.
      srcs: Source files.
      deps: Dependencies.
      data: Runtime data files.
      linkopts: Extra link options. Defaults to librt where needed.
      framework_so: Unused. LiteRT does not link libtensorflow_framework.
      soversion: Optional version appended to the library file name.
      kernels: Unused. Kept for compatibility with TensorFlow's macro.
      per_os_targets: Whether to name the outputs per OS (lib*.so, lib*.dylib,
        *.dll).
      visibility: Target visibility.
      **kwargs: Passed to cc_binary.
    """
    if linkopts == None:
        linkopts = _lrt_if_needed()
    if soversion != None:
        suffix = "." + str(soversion).split(".")[0]
        longsuffix = "." + str(soversion)
    else:
        suffix = ""
        longsuffix = ""

    if per_os_targets:
        names = [
            (
                pattern % (name, ""),
                pattern % (name, suffix),
                pattern % (name, longsuffix),
            )
            for pattern in SHARED_LIBRARY_NAME_PATTERNS
        ]
    else:
        names = [(
            name,
            name + suffix,
            name + longsuffix,
        )]

    testonly = kwargs.pop("testonly", False)

    for name_os, name_os_major, name_os_full in names:
        # Windows DLLs can't be versioned
        if name_os.endswith(".dll"):
            name_os_major = name_os
            name_os_full = name_os

        if name_os != name_os_major:
            native.genrule(
                name = name_os + "_sym",
                outs = [name_os],
                srcs = [name_os_major],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename $<) $@",
            )
            native.genrule(
                name = name_os_major + "_sym",
                outs = [name_os_major],
                srcs = [name_os_full],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename $<) $@",
            )

        soname = name_os_major.split("/")[-1]

        cc_binary(
            name = name_os_full,
            srcs = srcs,
            deps = deps,
            linkshared = 1,
            data = data,
            linkopts = linkopts + _rpath_linkopts(name_os_full) + select({
                clean_dep("//tensorflow:ios"): [
                    "-Wl,-install_name,@rpath/" + soname,
                ],
                clean_dep("//tensorflow:macos"): [
                    "-Wl,-install_name,@rpath/" + soname,
                ],
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-Wl,-soname," + soname,
                ],
            }),
            testonly = testonly,
            visibility = visibility,
            **kwargs
        )

    flat_names = [item for sublist in names for item in sublist]
    if name not in flat_names:
        native.filegroup(
            name = name,
            srcs = select({
                clean_dep("//tensorflow:windows"): [":%s.dll" % (name)],
                clean_dep("//tensorflow:macos"): [":lib%s%s.dylib" % (name, longsuffix)],
                "//conditions:default": [":lib%s.so%s" % (name, longsuffix)],
            }),
            visibility = visibility,
            testonly = testonly,
        )

# Attributes accepted internally but not by OSS rules_python.
_UNSUPPORTED_PY_ARGS = [
    "flaky_test_attempts",
    "lazy_imports",
    "linking_mode",
    "strict_deps",
]

def filter_py_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if k not in _UNSUPPORTED_PY_ARGS}

def py_test(
        deps = [],
        data = [],
        kernels = [],  # @unused
        exec_properties = None,  # @unused
        env = {},
        extra_pywrap_deps = [],  # @unused
        **kwargs):
    _py_test(
        deps = deps,
        data = data,
        env = env,
        **filter_py_kwargs(kwargs)
    )

def pybind_extension(name, common_lib_packages = [], pywrap_only = False, **kwargs):
    if _use_pywrap_rules():
        _pybind_extension(
            name = name,
            common_lib_packages = common_lib_packages + ["tensorflow", "tensorflow/python"],
            **kwargs
        )
    elif not pywrap_only:
        fail("LiteRT OSS builds Python extensions with USE_PYWRAP_RULES=True.")

def _transitive_hdrs_impl(ctx):
    outputs = depset(transitive = [
        dep[CcInfo].compilation_context.headers
        for dep in ctx.attr.deps
    ])
    return [DefaultInfo(files = outputs)]

_transitive_hdrs = rule(
    attrs = {"deps": attr.label_list(allow_files = True, providers = [CcInfo])},
    implementation = _transitive_hdrs_impl,
)

def transitive_hdrs(name, deps = [], **kwargs):
    _transitive_hdrs(name = name + "_gather", deps = deps)
    native.filegroup(name = name, srcs = [":" + name + "_gather"], **kwargs)

def _requires_tensorflow(name, **kwargs):
    native.genrule(
        name = name,
        outs = [name + ".requires_tensorflow"],
        cmd = "echo '%s needs the TensorFlow repository, which LiteRT OSS does not use.' >&2; exit 1" % name,
        tags = kwargs.get("tags", []) + ["manual"],
        testonly = kwargs.get("testonly", False),
        visibility = kwargs.get("visibility", None),
    )

def tf_custom_op_library(name, **kwargs):
    _requires_tensorflow(name, **kwargs)

def tf_gen_op_libs(op_lib_names, **kwargs):
    for n in op_lib_names:
        _requires_tensorflow(n + "_op_lib", **kwargs)

def tf_gen_op_wrapper_py(name, **kwargs):
    _requires_tensorflow(name, **kwargs)
