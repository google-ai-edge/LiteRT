"""Repository rules and macros for Android SDK environment checks."""

def _check_android_sdk_env_impl(repository_ctx):
    is_github = repository_ctx.os.environ.get("GITHUB_ACTIONS", "")
    sdk_home = repository_ctx.os.environ.get("ANDROID_HOME", "")
    if not sdk_home:
        sdk_home = repository_ctx.os.environ.get("ANDROID_SDK_HOME", "")
    is_set = "True" if is_github and sdk_home else "False"
    repository_ctx.file("current_android_sdk_env.bzl", "ANDROID_SDK_HOME_IS_SET = " + is_set + "\n")
    repository_ctx.file("BUILD", "")

check_android_sdk_env = repository_rule(
    implementation = _check_android_sdk_env_impl,
    environ = ["ANDROID_HOME", "ANDROID_SDK_HOME"],
)

def declare_android_sdk(is_set, name = "androidsdk"):
    if is_set:
        # buildifier: disable=native-android
        native.android_sdk_repository(name = name)
