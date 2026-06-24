# This file is used to define a custom repository rule for TensorFlow submodule used by LiteRT.
#
# The function below allows us to select between a local TensorFlow source from local_repository
# or a remote http_archive based on the 'USE_LOCAL_TF' environment variable.
#
# To use this rule, you must set the 'USE_LOCAL_TF' environment variable to
# 'true' and the 'TF_LOCAL_SOURCE_PATH' environment variable to the absolute path of
# the tensorflow source directory. We need to pass the absolute path because we cannot use
# ctx.path() on a relative path for lower bazel versions.

"""
Implementation function for custom TensorFlow source repository rule.
This rule is used to select between a local TensorFlow source from local_repository
or a remote http_archive based on the 'USE_LOCAL_TF' environment variable.
"""

def _tensorflow_source_repo_impl(ctx):
    use_local_tf = ctx.os.environ.get("USE_LOCAL_TF", "false") == "true"

    if use_local_tf:
        # TF_LOCAL_SOURCE_PATH must be set to the absolute path to the tensorflow source directory.
        TF_LOCAL_SOURCE_PATH_ENV = ctx.os.environ.get("TF_LOCAL_SOURCE_PATH", "")
        if not TF_LOCAL_SOURCE_PATH_ENV:
            fail("""ERROR: USE_LOCAL_TF is true, but TF_LOCAL_SOURCE_PATH environment variable
                 is not set with the absolute path to TensorFlow source.""")

        local_path_str = TF_LOCAL_SOURCE_PATH_ENV  # Get the path from the environment variable
        resolved_local_path = ctx.path(local_path_str)

        for f in resolved_local_path.readdir():
            ctx.symlink(f, f.basename)
    else:
        ctx.download_and_extract(
            url = ctx.attr.urls[0],
            sha256 = ctx.attr.sha256,
            stripPrefix = ctx.attr.strip_prefix,
        )

    _patch_xla_windows_copts(ctx)
    _patch_xnnpack_windows_arm64(ctx)
    _patch_farmhash_windows_arm64(ctx)
    _patch_cpuinfo_windows_arm64(ctx)
    _apply_repo_patches(ctx)

def _patch_xla_windows_copts(ctx):
    """Switch XLA Windows copts to MSVC-style flags to avoid -Wno-sign-compare."""
    tsl_bzl = "third_party/xla/xla/tsl/tsl.bzl"
    content = ctx.read(tsl_bzl)
    old = 'clean_dep("//xla/tsl:windows"): get_win_copts(is_external, is_msvc = False),'
    new = 'clean_dep("//xla/tsl:windows"): get_win_copts(is_external, is_msvc = True),'
    if old in content:
        ctx.file(tsl_bzl, content.replace(old, new))

def _patch_xnnpack_windows_arm64(ctx):
    """Patch XNNPACK's Bazel Windows ARM64 config to use MSVC C flags."""
    patch_path = "third_party/xla/third_party/xnnpack/windows_arm64_msvc.patch"
    ctx.file(patch_path, content = """diff --git a/BUILD.bazel b/BUILD.bazel
--- a/BUILD.bazel
+++ b/BUILD.bazel
@@ -2057,6 +2057,7 @@ alias(
     actual = select({
         ":xnn_enable_assembly_explicit_true": ":xnn_enable_assembly_explicit_true",
         ":xnn_enable_assembly_explicit_false": ":xnn_enable_assembly_explicit_true",
+        "//build_config:windows_aarch64": ":xnn_enable_assembly_explicit_true",
         "//conditions:default": ":assembly_enabled_by_default",
     }),
 )
diff --git a/build_config/BUILD.bazel b/build_config/BUILD.bazel
--- a/build_config/BUILD.bazel
+++ b/build_config/BUILD.bazel
@@ -144,6 +144,6 @@ config_setting(
 config_setting(
     name = "windows_aarch64",
     values = {
-         "cpu": "aarch64_windows",
+         "cpu": "arm64_windows",
     },
 )
@@ -332,6 +332,14 @@ selects.config_setting_group(
     ],
 )
@SP@
+selects.config_setting_group(
+    name = "aarch64_or_windows_aarch64",
+    match_any = [
+        ":aarch64",
+        ":windows_aarch64",
+    ],
+)
+
 selects.config_setting_group(
     name = "arm",
     match_any = [
@@ -342,6 +350,14 @@ selects.config_setting_group(
     ],
 )
@SP@
+selects.config_setting_group(
+    name = "arm_or_windows_aarch64",
+    match_any = [
+        ":arm",
+        ":windows_aarch64",
+    ],
+)
+
 selects.config_setting_group(
     name = "x86",
     match_any = [
diff --git a/build_defs.bzl b/build_defs.bzl
--- a/build_defs.bzl
+++ b/build_defs.bzl
@@ -378,7 +378,8 @@ def xnnpack_cc_library(
             "//build_config:windows_x86_64_clangcl": ["/clang:" + opt for opt in gcc_copts],
             "//build_config:windows_x86_64_clang": gcc_copts,
             "//build_config:windows_x86_64_mingw": gcc_copts,
             "//build_config:windows_x86_64_msys": gcc_copts,
             "//build_config:windows_x86_64": msvc_copts,
+            "//build_config:windows_aarch64": msvc_copts,
             "//conditions:default": gcc_copts,
         }) + select({
diff --git a/build_params.bzl b/build_params.bzl
--- a/build_params.bzl
+++ b/build_params.bzl
@@ -281,7 +281,7 @@ XNNPACK_PARAMS_FOR_ARCH = {
     ),
     "neon": _create_params(
-        cond = "//build_config:arm",
+        cond = "//build_config:arm_or_windows_aarch64",
         copts = xnnpack_select_if(
             "//build_config:aarch32",
             [
@@ -295,7 +295,7 @@ XNNPACK_PARAMS_FOR_ARCH = {
         ]),
     ),
     "neon_aarch64": _create_params(
-        cond = "//build_config:aarch64",
+        cond = "//build_config:aarch64_or_windows_aarch64",
         extra_deps = xnnpack_if_kleidiai_enabled([
             "@KleidiAI//kai/ukernels/matmul",
         ]),
@@ -301,7 +301,7 @@ XNNPACK_PARAMS_FOR_ARCH = {
         ]),
     ),
     "neonfp16": _create_params(
-        cond = "//build_config:arm",
+        cond = "//build_config:arm_or_windows_aarch64",
         copts = xnnpack_select_if(
             "//build_config:aarch32",
             [
@@ -312,7 +312,7 @@ XNNPACK_PARAMS_FOR_ARCH = {
         ),
     ),
     "neonfma": _create_params(
-        cond = "//build_config:arm",
+        cond = "//build_config:arm_or_windows_aarch64",
         copts = xnnpack_select_if(
             "//build_config:aarch32",
             [
@@ -323,10 +323,10 @@ XNNPACK_PARAMS_FOR_ARCH = {
         ),
     ),
     "neonfma_aarch64": _create_params(
-        cond = "//build_config:aarch64",
+        cond = "//build_config:aarch64_or_windows_aarch64",
     ),
     "neonv8": _create_params(
-        cond = "//build_config:arm",
+        cond = "//build_config:arm_or_windows_aarch64",
         copts = xnnpack_select_if(
             "//build_config:aarch32",
             [
""".replace("@SP@", " "))

    workspace_bzl = "third_party/xla/third_party/xnnpack/workspace.bzl"
    content = ctx.read(workspace_bzl)
    if "windows_arm64_msvc.patch" in content:
        return

    old = '        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/d0004f80c78fed80c230045ee83ff34dc55be81a.zip"),'
    new = old + '\n        patch_file = ["//third_party/xnnpack:windows_arm64_msvc.patch"],'
    if old in content:
        ctx.file(workspace_bzl, content.replace(old, new))

def _patch_farmhash_windows_arm64(ctx):
    """Disable FarmHash builtin-expect usage for Windows ARM64 MSVC."""
    farmhash_build = "third_party/xla/third_party/farmhash/farmhash.BUILD"
    content = ctx.read(farmhash_build)

    if 'name = "windows_arm64"' not in content:
        old = """config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)
"""
        new = old + """
config_setting(
    name = "windows_arm64",
    values = {
        "cpu": "arm64_windows",
    },
)
"""
        content = content.replace(old, new)

    old = """        ":windows_x86_64_clang": ["-DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        ":windows": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        "//conditions:default": [],
"""
    new = """        ":windows_x86_64_clang": ["-DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        ":windows": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        ":windows_arm64": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        "//conditions:default": [],
"""
    if '":windows_arm64": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],' not in content:
        content = content.replace(old, new)

    ctx.file(farmhash_build, content)

def _patch_cpuinfo_windows_arm64(ctx):
    """Patch cpuinfo ARM sources to use their MSVC-safe restrict/static macro."""
    patch_path = "third_party/xla/third_party/cpuinfo/windows_arm64_msvc.patch"
    ctx.file(patch_path, content = """diff --git a/src/arm/cache.c b/src/arm/cache.c
--- a/src/arm/cache.c
+++ b/src/arm/cache.c
@@ -9,12 +9,12 @@
@SP@void cpuinfo_arm_decode_cache(
@SP@@TAB@enum cpuinfo_uarch uarch,
@SP@@TAB@uint32_t cluster_cores,
@SP@@TAB@uint32_t midr,
-	const struct cpuinfo_arm_chipset chipset[restrict static 1],
+	const struct cpuinfo_arm_chipset chipset[RESTRICT_STATIC 1],
@SP@@TAB@uint32_t cluster_id,
@SP@@TAB@uint32_t arch_version,
-	struct cpuinfo_cache l1i[restrict static 1],
-	struct cpuinfo_cache l1d[restrict static 1],
-	struct cpuinfo_cache l2[restrict static 1],
-	struct cpuinfo_cache l3[restrict static 1]) {
+	struct cpuinfo_cache l1i[RESTRICT_STATIC 1],
+	struct cpuinfo_cache l1d[RESTRICT_STATIC 1],
+	struct cpuinfo_cache l2[RESTRICT_STATIC 1],
+	struct cpuinfo_cache l3[RESTRICT_STATIC 1]) {
@SP@@TAB@switch (uarch) {
""".replace("@SP@", " ").replace("@TAB@", "\t"))

    workspace_bzl = "third_party/xla/third_party/cpuinfo/workspace.bzl"
    content = ctx.read(workspace_bzl)
    if "windows_arm64_msvc.patch" in content:
        return

    old = '        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/8a9210069b5a37dd89ed118a783945502a30a4ae.zip"),'
    new = old + '\n        patch_file = ["//third_party/cpuinfo:windows_arm64_msvc.patch"],'
    if old in content:
        ctx.file(workspace_bzl, content.replace(old, new))

def _apply_repo_patches(ctx):
    # Apply patches after extraction/symlinking.
    for patch in ctx.attr.patches:
        ctx.patch(ctx.path(patch), strip = 1)

    # Append protobuf-specific patches to XLA's protobuf.patch so that
    # tf_workspace2()'s protobuf fetch applies them automatically.
    if ctx.attr.protobuf_patches:
        pb_patch_path = ctx.path("third_party/xla/third_party/protobuf/protobuf.patch")
        existing = ctx.read(pb_patch_path)
        extras = []
        for pb_patch in ctx.attr.protobuf_patches:
            extras.append(ctx.read(ctx.path(pb_patch)))
        ctx.file(
            "third_party/xla/third_party/protobuf/protobuf.patch",
            content = existing + "\n" + "\n".join(extras),
        )

tensorflow_source_repo = repository_rule(
    implementation = _tensorflow_source_repo_impl,
    local = False,
    attrs = {
        "sha256": attr.string(mandatory = False),
        "strip_prefix": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
        "patches": attr.label_list(default = []),
        "protobuf_patches": attr.label_list(default = []),
    },
    doc = """
    A custom repository rule to select between a local TensorFlow source or a remote http_archive
    based on the'USE_LOCAL_TF' environment variable and TF_LOCAL_SOURCE_PATH flag.
    """,
)
