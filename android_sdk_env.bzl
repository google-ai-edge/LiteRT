"""Repository rules and macros for Android SDK environment checks."""

def _check_android_sdk_env_impl(repository_ctx):
    sdk_home = repository_ctx.os.environ.get("ANDROID_HOME", "")
    if not sdk_home:
        sdk_home = repository_ctx.os.environ.get("ANDROID_SDK_HOME", "")
    is_set = "True" if sdk_home else "False"

    internal_build = repository_ctx.os.environ.get("LITERT_INTERNAL_BUILD", "False")

    repository_ctx.file(
        "current_android_sdk_env.bzl",
        "ANDROID_SDK_HOME_IS_SET = " + is_set + "\n" +
        "IS_INTERNAL_BUILD = " + internal_build + "\n",
    )
    repository_ctx.file("BUILD", "")

check_android_sdk_env = repository_rule(
    implementation = _check_android_sdk_env_impl,
    environ = ["ANDROID_HOME", "ANDROID_SDK_HOME", "LITERT_INTERNAL_BUILD"],
)

def declare_android_sdk(is_set, name = "androidsdk", should_declare = True):
    if should_declare and is_set:
        # buildifier: disable=native-android
        native.android_sdk_repository(name = name)

def register_android_sdk_toolchains(should_register = True, sdk_is_set = True, name = "register_android_sdk_toolchains"):
    if should_register and sdk_is_set:
        native.register_toolchains("@androidsdk//:sdk-toolchain", "@androidsdk//:all")
