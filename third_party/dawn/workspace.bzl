"""Repository rule providing Dawn (WebGPU), optionally built from source.

By default the pinned Dawn release is downloaded and only its headers are used: the Dawn library is
loaded at run time. Use `--config=dawn_from_source` to also build it with CMake (needs cmake, git,
python3, a C++20 compiler and network access), to be deployed next to the accelerator library. The
path of the built library is printed. Works on Linux and Windows, optionally with:

  --repo_env=LITERT_DAWN_SRC_DIR=<path>   Build a local Dawn checkout instead of the pinned release
                                          (its dependencies are fetched into it if missing).
  --repo_env=LITERT_DAWN_CMAKE_ARGS=...   Extra Dawn CMake options, space separated,
                                          e.g. "-DDAWN_USE_X11=ON -DDAWN_ENABLE_OPENGLES=ON".
"""

_DAWN_VERSION = "20260720.160313"

_ENV_BUILD = "LITERT_BUILD_DAWN"
_ENV_SRC_DIR = "LITERT_DAWN_SRC_DIR"
_ENV_CMAKE_ARGS = "LITERT_DAWN_CMAKE_ARGS"
_ENV_SH = "BAZEL_SH"

# C++17 compatibility patches for the generated Dawn C++ header and dawn_proc templates, applied to the
# downloaded release.
_PATCH_CMDS = [
    # 1. C++17 fallback for std::span -> absl::Span
    "$(if python3 -V >/dev/null 2>/dev/null; then echo python3; else echo python; fi) -c 'p=\"generator/templates/api_cpp.h\"; c=open(p).read().replace(\"#include <span>\", \"#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)\\n#include <span>\\n#else\\n#include \\\"absl/types/span.h\\\"\\nnamespace std { template <typename T> using span = absl::Span<T>; }\\n#endif\"); open(p,\"w\").write(c)'",
    # 2. C++17 fallback for C++20 unevaluated lambdas in BaseArgsTuple -> take_tuple_t
    "$(if python3 -V >/dev/null 2>/dev/null; then echo python3; else echo python; fi) -c 'p=\"generator/templates/api_cpp.h\"; c=open(p).read().replace(\"template <typename CppFT, typename CppFPtr, typename T>\\nstruct CppFTraitsImpl;\", \"template <size_t N, typename Tuple, typename Seq = std::make_index_sequence<N>>\\nstruct take_tuple;\\ntemplate <size_t N, typename... Args, size_t... Is>\\nstruct take_tuple<N, std::tuple<Args...>, std::index_sequence<Is...>> {\\n    using type = std::tuple<std::tuple_element_t<Is, std::tuple<Args...>>...>;\\n};\\ntemplate <size_t N, typename... Args>\\nusing take_tuple_t = typename take_tuple<N, std::tuple<Args...>>::type;\\n\\ntemplate <typename CppFT, typename CppFPtr, typename T>\\nstruct CppFTraitsImpl;\").replace(\"    using BaseArgsTuple = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {\\n        return std::type_identity<std::tuple<std::tuple_element_t<Is, std::tuple<CppArgs...>>...>>{};\\n    }(std::make_index_sequence<std::is_same_v<T, Untyped> ? NumCppArgs : NumCppArgs - 1>{})\\n    )::type;\", \"    using BaseArgsTuple = take_tuple_t<std::is_same_v<T, Untyped> ? NumCppArgs : NumCppArgs - 1, CppArgs...>;\"); open(p,\"w\").write(c)'",
    # 3. C++17 SFINAE fix for CallbackInfoHelper::Create capturing lambda constraints
    "$(if python3 -V >/dev/null 2>/dev/null; then echo python3; else echo python; fi) -c 'p=\"generator/templates/api_cpp.h\"; c=open(p).read().replace(\"if constexpr (requires(CInfoT x) { x.mode; }) {\", \"if constexpr (has_mode<CInfoT>::value) {\").replace(\"template <typename CInfoT, typename F>\\nstruct CallbackInfoHelper {\", \"template <typename T, typename = void>\\nstruct has_mode : std::false_type {};\\ntemplate <typename T>\\nstruct has_mode<T, std::void_t<decltype(std::declval<T>().mode)>> : std::true_type {};\\ntemplate <typename CInfoT, typename F>\\nstruct CallbackInfoHelper {\").replace(\"    template <typename T>\\n    requires (!CppFTraits<F>::capturing)\\n    static CInfoT Create(F lambda, T userdata) {\", \"    template <typename T, typename F_ = F, typename = std::enable_if_t<!CppFTraits<F_>::capturing>>\\n    static CInfoT Create(F lambda, T userdata) {\"); open(p,\"w\").write(c)'",
    # 4. C++17 fallback for std::ranges::copy / std::ranges::equal in dawn_proc templates, and flatten src/utils subpackage for dawn.BUILD
    "rm -f src/utils/BUILD.bazel src/utils/chromium_test_compat/BUILD.bazel",
    "$(if python3 -V >/dev/null 2>/dev/null; then echo python3; else echo python; fi) -c 'for p in (\"generator/templates/dawn_proc.cpp\", \"generator/templates/dawn_thread_dispatch_proc.cpp\"): c=open(p).read().replace(\"std::ranges::copy(dawn::kDawnVersion, procs.version);\", \"for (size_t i = 0; i < 20; ++i) { procs.version[i] = dawn::kDawnVersion[i]; }\").replace(\"std::ranges::equal(procs_->version, dawn::kDawnVersion)\", \"std::equal(std::begin(procs_->version), std::end(procs_->version), std::begin(dawn::kDawnVersion))\"); open(p,\"w\").write(c)'",
]

# third_party dependencies that Dawn's CMake build needs: directory -> file that exists once fetched.
_DEPENDENCIES = {
    "abseil-cpp": "CMakeLists.txt",
    "spirv-headers/src": "CMakeLists.txt",
    "spirv-tools/src": "CMakeLists.txt",
    "glslang/src": "CMakeLists.txt",
    "vulkan-headers/src": "CMakeLists.txt",
    "vulkan-loader/src": "CMakeLists.txt",
    "vulkan-utility-libraries/src": "CMakeLists.txt",
    "webgpu-headers/src": "schema.json",
}

# Needed on Windows only: the D3D12 backend.
_WINDOWS_DEPENDENCIES = {
    "directx-headers/src": "CMakeLists.txt",
}

# Files of a local Dawn checkout that dawn.BUILD replaces.
_REPLACED_FILES = ["BUILD.bazel", "WORKSPACE.bazel", "MODULE.bazel", "MODULE.bazel.lock", ".git"]

# Directories of Dawn whose BUILD.bazel is dropped, so that dawn.BUILD can use their files: the last patch
# does the same for the release. _FLATTENED_PARENTS are the directories on the way to them.
_FLATTENED = ["src/utils", "src/utils/chromium_test_compat"]
_FLATTENED_PARENTS = ["src"] + _FLATTENED

def _find_python(rctx):
    """Finds a working Python executable in PATH, skipping broken stubs like the Windows Store alias."""
    for name in ["python3", "python"]:
        candidate = rctx.which(name)
        if candidate and rctx.execute([str(candidate), "-V"]).return_code == 0:
            return candidate
    fail("Could not find a working `python3` or `python` executable in PATH to fetch Dawn dependencies.")

def _fetch_dependencies(rctx, src):
    dependencies = dict(_DEPENDENCIES)
    if rctx.os.name.startswith("windows"):
        dependencies.update(_WINDOWS_DEPENDENCIES)

    python = None
    for _ in range(3):
        missing = [d for d, marker in dependencies.items() if not rctx.path("%s/third_party/%s/%s" % (src, d, marker)).exists]
        if not missing:
            return
        if not python:
            python = _find_python(rctx)
        for d in missing:
            # Dawn's script ignores git failures and then skips the empty clone left behind.
            rctx.delete("%s/third_party/%s/.git" % (src, d))
        rctx.execute([str(python), "tools/fetch_dawn_dependencies.py"], working_directory = str(src), timeout = 3600)
    fail("Could not fetch Dawn's dependencies (needs git and network access). Still missing: " + ", ".join(missing))

def _link_local_dawn(rctx, src):
    """Makes the files of a local Dawn checkout available in the repository, as the release would be."""
    pending = [""]
    for _ in range(len(_FLATTENED_PARENTS) + 1):
        if not pending:
            break
        directory = pending.pop()
        for entry in rctx.path("%s/%s" % (src, directory)).readdir():
            path = "%s/%s" % (directory, entry.basename) if directory else entry.basename
            if (not directory and entry.basename in _REPLACED_FILES) or (directory in _FLATTENED and entry.basename == "BUILD.bazel"):
                continue
            if path in _FLATTENED_PARENTS:
                pending.append(path)
            else:
                rctx.symlink(entry, path)

def _build_dawn(rctx, src):
    """Builds the monolithic shared Dawn library with CMake, returns its path in the repository."""
    cmake = rctx.which("cmake")
    if not cmake:
        fail("%s needs `cmake` in PATH." % _ENV_BUILD)
    _fetch_dependencies(rctx, src)

    build_dir = rctx.path("_litert_dawn_build")
    configure = [
        str(cmake),
        "-S",
        str(src),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=20",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED",
        "-DDAWN_FETCH_DEPENDENCIES=OFF",
        "-DDAWN_BUILD_PROTOBUF=OFF",
        "-DDAWN_BUILD_SAMPLES=OFF",
        "-DDAWN_BUILD_TESTS=OFF",
        "-DTINT_BUILD_TESTS=OFF",
        "-DTINT_BUILD_CMD_TOOLS=OFF",
    ]
    if rctx.which("ninja") and not rctx.os.name.startswith("windows"):  # Needs the MSVC environment.
        configure.append("-GNinja")
    configure += [a for a in (rctx.getenv(_ENV_CMAKE_ARGS) or "").split(" ") if a]

    for step, cmd in [
        ("configure", configure),
        ("build", [str(cmake), "--build", str(build_dir), "--target", "webgpu_dawn", "--config", "Release"]),
    ]:
        rctx.report_progress("Dawn: CMake " + step)
        result = rctx.execute(cmd, timeout = 7200)
        if result.return_code != 0:
            output = (result.stdout + "\n" + result.stderr).splitlines()
            fail("Building Dawn failed in the CMake %s step, last lines:\n%s" % (step, "\n".join(output[-60:])))

    name = "webgpu_dawn.dll" if rctx.os.name.startswith("windows") else "libwebgpu_dawn.so"

    # Visual Studio generators put the library in a Release/ subdirectory.
    for directory in ["Release", "src/dawn/native", "src/dawn/native/Release", "."]:
        lib = "_litert_dawn_build/%s/%s" % (directory, name)
        if rctx.path(lib).exists:
            return lib
    fail("Dawn was built but %s was not found in _litert_dawn_build/." % name)

def _dawn_repo_impl(rctx):
    local_dir = rctx.getenv(_ENV_SRC_DIR)

    if local_dir:
        src = rctx.workspace_root.get_child(local_dir)
        if not src.is_dir:
            fail("%s path '%s' (resolved to '%s') is not a directory." % (_ENV_SRC_DIR, local_dir, src))
        _link_local_dawn(rctx, src)
    else:
        src = rctx.path(".")
        rctx.download_and_extract(
            url = "https://github.com/google/dawn/archive/v%s.tar.gz" % _DAWN_VERSION,
            stripPrefix = "dawn-" + _DAWN_VERSION,
        )
        bash = rctx.getenv(_ENV_SH) or "bash"
        for cmd in _PATCH_CMDS:
            result = rctx.execute([bash, "-c", cmd])
            if result.return_code != 0:
                fail("Patching Dawn failed:\n%s\n%s\n%s" % (cmd, result.stdout, result.stderr))
        rctx.delete("BUILD.bazel")

    if local_dir or rctx.getenv(_ENV_BUILD, "0") not in ["", "0"]:
        # buildifier: disable=print
        print("Dawn built from source: %s" % rctx.path(_build_dawn(rctx, src)))
    rctx.symlink(rctx.attr.build_file, "BUILD.bazel")

_dawn_repo = repository_rule(
    implementation = _dawn_repo_impl,
    attrs = {"build_file": attr.label(allow_single_file = True, mandatory = True)},
    environ = [_ENV_BUILD, _ENV_SRC_DIR, _ENV_CMAKE_ARGS, _ENV_SH],
)

def repo():
    _dawn_repo(
        name = "dawn",
        build_file = Label("//third_party/dawn:dawn.BUILD"),
    )
