# Copyright 2024 Google LLC.
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

"""Common LiteRT Build Utilities."""

# copybara:uncomment_begin(google-only)
# load("//devtools/build_cleaner/skylark:build_defs.bzl", "register_extension_info")
# copybara:uncomment_end

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

####################################################################################################
# Util

_LRT_SO_PREFIX = "libLiteRt"
_SO_EXT = ".so"
_SHARED_LIB_SUFFIX = "_so"

# Public

def if_oss(oss_value, google_value = []):  # buildifier: disable=unused-variable
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return oss_value  # copybara:comment_replace return google_value

def if_google(google_value, oss_value = []):  # buildifier: disable=unused-variable
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return oss_value  # copybara:comment_replace return google_value

def make_linkopt(opt):
    return "-Wl,{}".format(opt)

def to_path(label, get_parent = False):
    path = label.removeprefix("//").replace(":", "/")
    if not get_parent:
        return path
    return path[:path.rfind("/")]

def make_rpaths(rpaths):
    paths = []
    for rp in rpaths:
        if rp.startswith("@"):
            # Handle repository paths (E.g. @repo_name//:target/file.txt), relavant in OSS.
            repo_name, repo_targ = rp.removeprefix("@").split("//")
            pref = "//external/{}".format(repo_name)
            if repo_targ.startswith(":"):
                repo_path = pref + repo_targ
            else:
                repo_path = pref + "/" + repo_targ
            paths.append(to_path(repo_path, get_parent = True))
        elif ":" in rp:
            paths.append(to_path(rp, get_parent = True))
        else:
            paths.append(rp)

    return make_linkopt("-rpath={}".format(":".join(paths)))

def append_rule_kwargs(rule_kwargs, **append):
    for k, v in append.items():
        append_to = rule_kwargs.pop(k, [])
        append_to += v
        rule_kwargs[k] = append_to

def absolute_label(label, package_name = None):
    """Get the absolute label for a given label.

    Args:
      label: The label to convert to absolute.
      package_name: The package name to use if the label is relative.

    Returns:
      The absolute label.
    """
    if label.startswith("//") or label.startswith("@"):
        if ":" in label:
            return label
        return "%s:%s" % (label, label.rsplit("/", 1)[-1])
    if not package_name:
        package_name = native.package_name()
    if label.startswith(":"):
        return "//%s%s" % (package_name, label)
    if ":" in label:
        return "//%s/%s" % (package_name, label)
    return "//%s:%s" % (package_name, label)

# Private

def _valid_shared_lib_name(name):
    return name.endswith(_SHARED_LIB_SUFFIX)

def _valid_so_name(name):
    return name.startswith(_LRT_SO_PREFIX) and name.endswith(_SO_EXT)

def _make_target_ref(name):
    return ":{}".format(name)

####################################################################################################
# Explicitly Link System Libraries ("ungrte")

_SYS_RPATHS_X86_64 = [
    "/usr/lib/x86_64-linux-gnu",
    "/lib/x86_64-linux-gnu",
]
_SYS_RPATHS_LINKOPT_X86_64 = make_rpaths(_SYS_RPATHS_X86_64)

_SYS_ELF_INTERPRETER_X86_64 = "/lib64/ld-linux-x86-64.so.2"
_SYS_ELF_INTERPRETER_LINKOPT_X86_64 = make_linkopt("--dynamic-linker={}".format(_SYS_ELF_INTERPRETER_X86_64))

####################################################################################################
# Symbol Hiding

_EXPORT_LRT_ONLY_SCRIPT_LINUX = "//litert/build_common:export_litert_only_linux.lds"
_EXPORT_LRT_ONLY_SCRIPT_DARWIN = "//litert/build_common:export_litert_only_darwin.lds"
_EXPORT_LRT_ONLY_LINKOPT_LINUX = make_linkopt("--version-script=$(location {})".format(_EXPORT_LRT_ONLY_SCRIPT_LINUX))
_EXPORT_LRT_ONLY_LINKOPT_DARWIN = make_linkopt("-exported_symbols_list,$(location {})".format(_EXPORT_LRT_ONLY_SCRIPT_DARWIN))

def symbol_opts():
    """Defines linker flags whether to include symbols or not."""
    return select({
        "@org_tensorflow//tensorflow:debug": [],
        "@org_tensorflow//tensorflow:macos": [],
        "@org_tensorflow//tensorflow:ios": [],
        "//conditions:default": [
            # Omit symbol table, for all non debug builds
            "-Wl,-s",
        ],
    })

def export_lrt_only_script():
    return select({
        "@org_tensorflow//tensorflow:linux_x86_64": [_EXPORT_LRT_ONLY_SCRIPT_LINUX],
        "@org_tensorflow//tensorflow:android": [_EXPORT_LRT_ONLY_SCRIPT_LINUX],
        "@org_tensorflow//tensorflow:chromiumos": [_EXPORT_LRT_ONLY_SCRIPT_LINUX],
        "@org_tensorflow//tensorflow:macos": [_EXPORT_LRT_ONLY_SCRIPT_DARWIN],
        "@org_tensorflow//tensorflow:ios": [_EXPORT_LRT_ONLY_SCRIPT_DARWIN],
        "//conditions:default": [],
    })

def export_lrt_only_linkopt():
    return select({
        "@org_tensorflow//tensorflow:linux_x86_64": [_EXPORT_LRT_ONLY_LINKOPT_LINUX],
        "@org_tensorflow//tensorflow:android": [
            "-Wl,-z,max-page-size=16384",
            _EXPORT_LRT_ONLY_LINKOPT_LINUX,
        ],
        "@org_tensorflow//tensorflow:chromiumos": [_EXPORT_LRT_ONLY_LINKOPT_LINUX],
        "@org_tensorflow//tensorflow:macos": [_EXPORT_LRT_ONLY_LINKOPT_DARWIN],
        "@org_tensorflow//tensorflow:ios": [_EXPORT_LRT_ONLY_LINKOPT_DARWIN],
        "//conditions:default": [],
    }) + symbol_opts()

_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_LINUX = "//litert/build_common:export_litert_runtime_only_linux.lds"
_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_DARWIN = "//litert/build_common:export_litert_runtime_only_darwin.lds"
_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_LINUX = make_linkopt("--version-script=$(location {})".format(_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_LINUX))
_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_DARWIN = make_linkopt("-exported_symbols_list,$(location {})".format(_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_DARWIN))

# TODO b/391390553: Add "-Wl,--no-undefined" to make sure all symbols are defined.
_EXPORT_LRT_COMMON_LINKOPTS_LINUX = [
    "-Wl,--no-export-dynamic",  # Only inc syms referenced by dynamic obj.
    "-Wl,--gc-sections",  # Eliminate unused code and data.
    "-Wl,--as-needed",  # Don't link unused libs.a
]

def export_lrt_runtime_only_script():
    return select({
        "@org_tensorflow//tensorflow:linux_x86_64": [_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_LINUX],
        "@org_tensorflow//tensorflow:android": [_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_LINUX],
        "@org_tensorflow//tensorflow:chromiumos": [_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_LINUX],
        "@org_tensorflow//tensorflow:macos": [_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_DARWIN],
        "@org_tensorflow//tensorflow:ios": [_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_DARWIN],
        "//conditions:default": [],
    })

def export_lrt_runtime_only_linkopt():
    return select({
        "@org_tensorflow//tensorflow:linux_x86_64": _EXPORT_LRT_COMMON_LINKOPTS_LINUX + [_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_LINUX],
        "@org_tensorflow//tensorflow:android": _EXPORT_LRT_COMMON_LINKOPTS_LINUX + [
            "-Wl,-z,max-page-size=16384",
            _EXPORT_LRT_RUNTIME_ONLY_LINKOPT_LINUX,
        ],
        "@org_tensorflow//tensorflow:chromiumos": _EXPORT_LRT_COMMON_LINKOPTS_LINUX + [_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_LINUX],
        "@org_tensorflow//tensorflow:macos": [_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_DARWIN],
        "@org_tensorflow//tensorflow:ios": [_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_DARWIN],
        "//conditions:default": [],
    }) + symbol_opts()

_EXPORT_LRT_TFLITE_RUNTIME_SCRIPT_LINUX = "//litert/build_common:export_litert_tflite_runtime_linux.lds"
_EXPORT_LRT_TFLITE_RUNTIME_SCRIPT_DARWIN = "//litert/build_common:export_litert_tflite_runtime_darwin.lds"
_EXPORT_LRT_TFLITE_RUNTIME_LINKOPT_LINUX = make_linkopt("--version-script=$(location {})".format(_EXPORT_LRT_TFLITE_RUNTIME_SCRIPT_LINUX))
_EXPORT_LRT_TFLITE_RUNTIME_LINKOPT_DARWIN = make_linkopt("-exported_symbols_list,$(location {})".format(_EXPORT_LRT_TFLITE_RUNTIME_SCRIPT_DARWIN))

def export_lrt_tflite_runtime_script():
    return select({
        "@org_tensorflow//tensorflow:linux_x86_64": [_EXPORT_LRT_TFLITE_RUNTIME_SCRIPT_LINUX],
        "@org_tensorflow//tensorflow:android": [_EXPORT_LRT_TFLITE_RUNTIME_SCRIPT_LINUX],
        "@org_tensorflow//tensorflow:chromiumos": [_EXPORT_LRT_TFLITE_RUNTIME_SCRIPT_LINUX],
        "@org_tensorflow//tensorflow:macos": [_EXPORT_LRT_TFLITE_RUNTIME_SCRIPT_DARWIN],
        "@org_tensorflow//tensorflow:ios": [_EXPORT_LRT_TFLITE_RUNTIME_SCRIPT_DARWIN],
        "//conditions:default": [],
    })

def export_lrt_tflite_runtime_linkopt():
    return select({
        "@org_tensorflow//tensorflow:linux_x86_64": _EXPORT_LRT_COMMON_LINKOPTS_LINUX + [_EXPORT_LRT_TFLITE_RUNTIME_LINKOPT_LINUX],
        "@org_tensorflow//tensorflow:android": _EXPORT_LRT_COMMON_LINKOPTS_LINUX + [
            "-Wl,-z,max-page-size=16384",
            _EXPORT_LRT_TFLITE_RUNTIME_LINKOPT_LINUX,
        ],
        "@org_tensorflow//tensorflow:chromiumos": _EXPORT_LRT_COMMON_LINKOPTS_LINUX + [_EXPORT_LRT_TFLITE_RUNTIME_LINKOPT_LINUX],
        "@org_tensorflow//tensorflow:macos": [_EXPORT_LRT_TFLITE_RUNTIME_LINKOPT_DARWIN],
        "@org_tensorflow//tensorflow:ios": [_EXPORT_LRT_TFLITE_RUNTIME_LINKOPT_DARWIN],
        "//conditions:default": [],
    }) + symbol_opts()

####################################################################################################
# Macros

# Private

def _litert_base(
        rule,
        ungrte = False,
        **cc_rule_kwargs):
    """
    Base rule for LiteRT targets.

    Args:
      rule: The underlying rule to use (e.g., cc_test, cc_library).
      ungrte: Whether to link against system libraries ("ungrte"). Even if
      ungrte is set to true by the library, ungrte might not happen. An
      additional build flag "ungrte" will also be inspected to determine whether
      the behavior should occur. The default behavior without specifying the
      build flag is to always ungrte.
      **cc_rule_kwargs: Keyword arguments to pass to the underlying rule.
    """

    _DEFAULT_LINK_OPTS = ["-Wl,--disable-new-dtags"]

    _UNGRTE_LINK_OPTS = [_SYS_ELF_INTERPRETER_LINKOPT_X86_64, _SYS_RPATHS_LINKOPT_X86_64]

    if ungrte:
        append_rule_kwargs(
            cc_rule_kwargs,
            linkopts = selects.with_or({
                ("//conditions:default", "//litert/build_common:linux_x86_64_grte"): _DEFAULT_LINK_OPTS,
                "@org_tensorflow//tensorflow:macos": [],
                "//litert/build_common:linux_x86_64_ungrte": _UNGRTE_LINK_OPTS + _DEFAULT_LINK_OPTS,
            }),
        )

    else:
        append_rule_kwargs(
            cc_rule_kwargs,
            linkopts = _DEFAULT_LINK_OPTS,
        )

    rule(**cc_rule_kwargs)

# Public

def litert_test(
        ungrte = False,
        use_sys_malloc = False,
        no_main = False,
        **cc_test_kwargs):
    """
    LiteRT test rule.

    Args:
      ungrte: Whether to link against system libraries ("ungrte").
      use_sys_malloc: Whether to use the system malloc.
      no_main: Whether to use the default main function.
      **cc_test_kwargs: Keyword arguments to pass to the underlying rule.
    """
    if use_sys_malloc:
        # copybara:uncomment cc_test_kwargs["malloc"] = "//base:system_malloc"
        pass

    if not no_main:
        append_rule_kwargs(
            cc_test_kwargs,
            deps = ["@com_google_googletest//:gtest_main"],
        )

    _litert_base(
        cc_test,
        ungrte,
        **cc_test_kwargs
    )

def litert_lib(
        ungrte = False,
        **cc_lib_kwargs):
    """
    LiteRT library rule.

    Args:
      ungrte: Whether to link against system libraries ("ungrte").
      **cc_lib_kwargs: Keyword arguments to pass to the underlying rule.
    """
    _litert_base(
        cc_library,
        ungrte,
        **cc_lib_kwargs
    )

def litert_bin(
        ungrte = False,
        export_litert_only = False,
        **cc_bin_kwargs):
    """
    LiteRT binary rule.

    Args:
      ungrte: Whether to link against system libraries ("ungrte").
      export_litert_only: Whether to export only LiteRT symbols.
      **cc_bin_kwargs: Keyword arguments to pass to the underlying rule.
    """
    if export_litert_only:
        append_rule_kwargs(
            cc_bin_kwargs,
            linkopts = export_lrt_only_linkopt(),
            deps = export_lrt_only_script(),
        )

    _litert_base(
        cc_binary,
        ungrte,
        **cc_bin_kwargs
    )

def litert_dynamic_lib(
        name,
        shared_lib_name,
        so_name,
        export_litert_only = False,
        ungrte = False,
        **cc_lib_kwargs):
    """
    LiteRT dynamic library rule.

    Args:
      name: The name of the library.
      shared_lib_name: The name of the shared library.
      so_name: The name of the shared object file.
      export_litert_only: Whether to export only LiteRT symbols.
      ungrte: Whether to link against system libraries ("ungrte").
      **cc_lib_kwargs: Keyword arguments to pass to the underlying rule.
    """
    if not _valid_shared_lib_name(shared_lib_name):
        fail("\"shared_lib_name\" must end with \"_so\"")
    if not _valid_so_name(so_name):
        fail("\"so_name\" must be \"libLiteRt*.so\"")

    lib_name = name
    cc_lib_kwargs["name"] = lib_name

    lib_target_ref = _make_target_ref(lib_name)

    vis = cc_lib_kwargs.get("visibility", None)

    # Share tags for all targets.
    tags = cc_lib_kwargs.get("tags", [])

    litert_lib(
        ungrte = ungrte,
        **cc_lib_kwargs
    )

    user_link_flags = []
    additional_linker_inputs = []
    if export_litert_only:
        user_link_flags = export_lrt_only_linkopt()
        additional_linker_inputs = export_lrt_only_script()
    cc_shared_library(
        name = shared_lib_name,
        shared_lib_name = so_name,
        user_link_flags = user_link_flags,
        additional_linker_inputs = additional_linker_inputs,
        tags = tags,
        visibility = vis,
        deps = [lib_target_ref],
        features = select({
            # Allow unresolved symbols which will be defined in the executable. There are no
            # linker flags to allow unresolved symbols only of a given pattern like LiteRt*.
            "//litert/c:resolve_symbols_in_exec": ["-no_undefined"],
            "//conditions:default": [],
        }),
    )

    # Workaround needed because `xeno_mobile_test` conflates target names with target output
    # files. Can be removed when we hand roll our own mobile_test wrapper.
    native.filegroup(
        name = so_name,
        srcs = [":" + shared_lib_name],
        visibility = vis,
    )

# copybara:uncomment_begin(google-only)
# register_extension_info(
#     extension = litert_dynamic_lib,
#     label_regex_map = {
#         "deps": "deps:{extension_name}",
#     },
# )
# copybara:uncomment_end

def copy_file(name, src, target, visibility = None):
    input_path = "$(location %s)" % src
    output_path = "$(@D)/" + target

    native.genrule(
        name = name,
        srcs = [src],
        outs = [target],
        visibility = visibility,
        cmd = "cp %s %s" % (input_path, output_path),
    )

def gtest_main_no_heapcheck_deps():
    # copybara:uncomment_begin(google-only)
    # return ["@com_google_googletest//:gtest_main_no_heapcheck"]
    # copybara:uncomment_end
    # copybara:comment_begin(oss-only)
    return ["@com_google_googletest//:gtest_main"]
    # copybara:comment_end

def cc_library_with_testonly_vis(
        name,
        vis = "//litert:litert_internal_users",
        testonly_vis = "//litert:litert_public",
        rule = cc_library,
        **rule_kwargs):
    """
    Defines a cc_library with different visibilities for normal and testonly targets.

    This macro defines two cc_library targets:
    - `{name}`: A normal cc_library with the given `vis` visibility.
    - `{name}_testonly`: A testonly cc_library with the given `testonly_vis` visibility.

    Args:
      name: The name of the library.
      vis: The visibility for the normal cc_library target.
      testonly_vis: The visibility for the testonly cc_library target.
      rule: The underlying rule to use (e.g., cc_library).
      **rule_kwargs: Keyword arguments to pass to the underlying rule.
    """
    if "append" not in dir(vis):
        vis = [vis]
    if "append" not in dir(testonly_vis):
        testonly_vis = [testonly_vis]
    rule(
        name = name,
        visibility = vis,
        **rule_kwargs
    )
    rule(
        name = name + "_testonly",
        testonly = True,
        visibility = testonly_vis,
        **rule_kwargs
    )
