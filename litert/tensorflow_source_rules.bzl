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

tensorflow_source_repo = repository_rule(
    implementation = _tensorflow_source_repo_impl,
    local = False,
    attrs = {
        "sha256": attr.string(mandatory = False),
        "strip_prefix": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
    },
    doc = """
    A custom repository rule to select between a local TensorFlow source or a remote http_archive
    based on the'USE_LOCAL_TF' environment variable and TF_LOCAL_SOURCE_PATH flag.
    """,
)
