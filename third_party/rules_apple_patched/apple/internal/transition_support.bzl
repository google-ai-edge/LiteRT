# Copyright 2019 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Starlark transition support for Apple rules.

This module makes the following distinctions around Apple CPU-adjacent values for clarity, based in
part on the language used for XCFramework library identifiers:

- `architecture`s or "arch"s represent the type of binary slice ("arm64", "x86_64").

- `environment`s represent a platform variant ("device", "sim"). These sometimes appear in the "cpu"
    keys out of necessity to distinguish new "cpu"s from an existing Apple "cpu" when a new
    Crosstool-provided toolchain is established.

- `platform_type`s represent the Apple OS being built for ("ios", "macos", "tvos", "visionos",
    "watchos").

- `cpu`s are keys to match a Crosstool-provided toolchain ("ios_sim_arm64", "ios_x86_64").
    Essentially it is a raw key that implicitly references the other three values for the purpose of
    getting the right Apple toolchain to build outputs with from the Apple Crosstool.
"""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load(
    "@build_bazel_apple_support//configs:platforms.bzl",
    "CPU_TO_DEFAULT_PLATFORM_NAME",
)
load(
    "//apple/build_settings:build_settings.bzl",
    "build_settings_labels",
)

_supports_visionos = hasattr(apple_common.platform_type, "visionos")
_is_bazel_7 = not hasattr(apple_common, "apple_crosstool_transition")

_PLATFORM_TYPE_TO_CPUS_FLAG = {
    "ios": "//command_line_option:ios_multi_cpus",
    "macos": "//command_line_option:macos_cpus",
    "tvos": "//command_line_option:tvos_cpus",
    "visionos": "//command_line_option:visionos_cpus",
    "watchos": "//command_line_option:watchos_cpus",
}

_CPU_TO_DEFAULT_PLATFORM_FLAG = {
    cpu: "@build_bazel_apple_support//platforms:{}_platform".format(
        platform_name,
    )
    for cpu, platform_name in CPU_TO_DEFAULT_PLATFORM_NAME.items()
}

_IOS_ARCH_TO_EARLIEST_WATCHOS = {
    "x86_64": "x86_64",
    "sim_arm64": "arm64",
    "arm64": "armv7k",
}

_IOS_ARCH_TO_64_BIT_WATCHOS = {
    "x86_64": "x86_64",
    "sim_arm64": "arm64",
    "arm64": "arm64_32",
}

def _platform_specific_cpu_setting_name(platform_type):
    """Returns the name of a platform-specific CPU setting.

    Args:
        platform_type: A string denoting the platform type; `"ios"`, `"macos"`, `"tvos"`,
            "visionos", or `"watchos"`.

    Returns:
        The `"//command_line_option:..."` string that is used as the key for the CPUs flag of the
            given platform in settings dictionaries. This function never returns `None`; if the
            platform type is invalid, the build fails.
    """
    flag = _PLATFORM_TYPE_TO_CPUS_FLAG.get(platform_type, None)
    if not flag:
        fail("ERROR: Unknown platform type: {}".format(platform_type))
    return flag

def _environment_arch_from_cpu(*, cpu_value, platform_prefix):
    """Returns a specific platform's environment arch if found from the `--cpu` command line option.

    Args:
        cpu_value: String found from an incoming `--cpu` value.
        platform_prefix: The platform prefix to search for within the incoming `--cpu` string.

    Returns:
        The value following the platform_prefix if it was found in the incoming `--cpu` value, which
            is expected to be a valid environment arch, or `None`.
    """
    if cpu_value.startswith(platform_prefix):
        return cpu_value[len(platform_prefix):]
    return None

def _watchos_environment_archs_from_ios(*, cpu_value, minimum_os_version, settings):
    """Returns a set of watchOS environment archs based on incoming iOS archs.

    Args:
        cpu_value: String found from an incoming `--cpu` value.
        minimum_os_version: A string coming directly from a rule's `minimum_os_version` attribute.
        settings: A dictionary whose set of keys is defined by the inputs parameter, typically from
            the settings argument found on the implementation function of the current Starlark
            transition.

    Returns:
        A list of watchOS environment archs if any were found from the iOS environment archs, or an
            empty list if none were found.
    """
    environment_archs = []
    ios_archs = settings[_platform_specific_cpu_setting_name("ios")]
    if not ios_archs:
        ios_arch = _environment_arch_from_cpu(
            cpu_value = cpu_value,
            platform_prefix = "ios_",
        )
        if ios_arch:
            ios_archs = [ios_arch]
    if ios_archs:
        # Make sure to return a fallback compatible with the rule's assigned minimum OS.
        ios_to_watchos_arch_dict = _IOS_ARCH_TO_64_BIT_WATCHOS
        if apple_common.dotted_version(minimum_os_version) < apple_common.dotted_version("9.0"):
            ios_to_watchos_arch_dict = _IOS_ARCH_TO_EARLIEST_WATCHOS
        environment_archs = [
            ios_to_watchos_arch_dict[arch]
            for arch in ios_archs
            if ios_to_watchos_arch_dict.get(arch)
        ]
    return environment_archs

def _environment_archs(platform_type, minimum_os_version, settings):
    """Returns a full set of environment archs from the incoming command line options.

    Args:
        platform_type: A string denoting the platform type; `"ios"`, `"macos"`,
            `"tvos"`, `"visionos"`, or `"watchos"`.
        minimum_os_version: A string coming directly from a rule's `minimum_os_version` attribute.
        settings: A dictionary whose set of keys is defined by the inputs parameter, typically from
            the settings argument found on the implementation function of the current Starlark
            transition.

    Returns:
        A list of valid Apple environments with its architecture as a string (for example
        `sim_arm64` from `ios_sim_arm64`, or `arm64` from `ios_arm64`).
    """
    environment_archs = settings[_platform_specific_cpu_setting_name(platform_type)]
    if not environment_archs:
        cpu_value = settings["//command_line_option:cpu"]
        if platform_type == "ios":
            # Legacy handling to interpret the --cpu as an iOS environment arch.
            ios_arch = _environment_arch_from_cpu(
                cpu_value = cpu_value,
                platform_prefix = "ios_",
            )
            if ios_arch:
                environment_archs = [ios_arch]
        if platform_type == "watchos":
            # Interpret the --cpu as a watchOS environment arch; often will be set by a transition.
            watchos_arch = _environment_arch_from_cpu(
                cpu_value = cpu_value,
                platform_prefix = "watchos_",
            )
            if watchos_arch:
                environment_archs = [watchos_arch]
            else:
                # If not found, generate watchOS archs via incoming iOS environment arch(s).
                environment_archs = _watchos_environment_archs_from_ios(
                    cpu_value = cpu_value,
                    minimum_os_version = minimum_os_version,
                    settings = settings,
                )
        if not environment_archs:
            environment_archs = [
                _cpu_string(
                    environment_arch = None,
                    platform_type = platform_type,
                    settings = settings,
                ).split("_", 1)[1],
            ]
    return environment_archs

def _cpu_string(*, environment_arch, platform_type, settings = {}):
    """Generates a <platform>_<environment?>_<arch> string for the current target based on args.

    Args:
        environment_arch: A valid Apple environment when applicable with its architecture as a
            string (for example `sim_arm64` from `ios_sim_arm64`, or `arm64` from `ios_arm64`), or
            None to infer a value from command line options passed through settings.
        platform_type: The Apple platform for which the rule should build its targets (`"ios"`,
            `"macos"`, `"tvos"`, `"visionos"`, or `"watchos"`).
        settings: A dictionary whose set of keys is defined by the inputs parameter, typically from
            the settings argument found on the implementation function of the current Starlark
            transition. If not defined, defaults to an empty dictionary. Used as a fallback if the
            `environment_arch` argument is None.

    Returns:
        A <platform>_<arch> string defined for the current target.
    """
    if platform_type == "ios":
        if environment_arch:
            return "ios_{}".format(environment_arch)
        ios_cpus = settings["//command_line_option:ios_multi_cpus"]
        if ios_cpus:
            return "ios_{}".format(ios_cpus[0])
        cpu_value = settings["//command_line_option:cpu"]
        if cpu_value.startswith("ios_"):
            return cpu_value
        if cpu_value == "darwin_arm64":
            return "ios_sim_arm64"
        return "ios_x86_64"
    if platform_type == "visionos":
        if environment_arch:
            return "visionos_{}".format(environment_arch)
        visionos_cpus = settings["//command_line_option:visionos_cpus"]
        if visionos_cpus:
            return "visionos_{}".format(visionos_cpus[0])
        cpu_value = settings["//command_line_option:cpu"]
        if cpu_value.startswith("visionos_"):
            return cpu_value
        return "visionos_sim_arm64"
    if platform_type == "macos":
        if environment_arch:
            return "darwin_{}".format(environment_arch)
        macos_cpus = settings["//command_line_option:macos_cpus"]
        if macos_cpus:
            return "darwin_{}".format(macos_cpus[0])
        cpu_value = settings["//command_line_option:cpu"]
        if cpu_value.startswith("darwin_"):
            return cpu_value
        return "darwin_x86_64"
    if platform_type == "tvos":
        if environment_arch:
            return "tvos_{}".format(environment_arch)
        tvos_cpus = settings["//command_line_option:tvos_cpus"]
        if tvos_cpus:
            return "tvos_{}".format(tvos_cpus[0])
        return "tvos_x86_64"
    if platform_type == "watchos":
        if environment_arch:
            return "watchos_{}".format(environment_arch)
        watchos_cpus = settings["//command_line_option:watchos_cpus"]
        if watchos_cpus:
            return "watchos_{}".format(watchos_cpus[0])
        return "watchos_x86_64"

    fail("ERROR: Unknown platform type: {}".format(platform_type))

def _min_os_version_or_none(*, minimum_os_version, platform, platform_type):
    if platform_type == platform:
        return minimum_os_version
    return None

def _is_arch_supported_for_target_tuple(*, environment_arch, minimum_os_version, platform_type):
    """Indicates if the environment_arch selected is supported for the given platform and min os.

    Args:
        environment_arch: A valid Apple environment when applicable with its architecture as a
            string (for example `sim_arm64` from `ios_sim_arm64`, or `arm64` from `ios_arm64`), or
            None to infer a value from command line options passed through settings.
        minimum_os_version: A string representing the minimum OS version specified for this
            platform, represented as a dotted version number (for example, `"9.0"`).
        platform_type: The Apple platform for which the rule should build its targets (`"ios"`,
            `"macos"`, `"tvos"`, `"visionos"`, or `"watchos"`).

    Returns:
        True if the architecture is supported for the given config, False otherwise.
    """

    dotted_minimum_os_version = apple_common.dotted_version(minimum_os_version)

    if (environment_arch == "armv7k" and platform_type == "watchos" and
        dotted_minimum_os_version >= apple_common.dotted_version("9.0")):
        return False

    return True

def _command_line_options(
        *,
        apple_platforms = [],
        environment_arch = None,
        force_bundle_outputs = False,
        minimum_os_version,
        platform_type,
        settings):
    """Generates a dictionary of command line options suitable for the current target.

    Args:
        apple_platforms: A list of labels referencing platforms if any should be set by the current
            rule. This will be applied directly to `apple_platforms` to allow for forwarding
            multiple platforms to rules evaluated after the transition is applied, and only the
            first element will be applied to `platforms` as that will be what is resolved by the
            underlying rule. Defaults to an empty list, which will signal to Bazel that platform
            mapping can take place as a fallback measure.
        environment_arch: A valid Apple environment when applicable with its architecture as a
            string (for example `sim_arm64` from `ios_sim_arm64`, or `arm64` from `ios_arm64`), or
            None to infer a value from command line options passed through settings.
        force_bundle_outputs: Indicates if the rule should always emit tree artifact outputs, which
            are effectively bundles that aren't enclosed within a zip file (ipa). If not `True`,
            this will be set to the incoming value instead. Defaults to `False`.
        minimum_os_version: A string representing the minimum OS version specified for this
            platform, represented as a dotted version number (for example, `"9.0"`).
        platform_type: The Apple platform for which the rule should build its targets (`"ios"`,
            `"macos"`, `"tvos"`, `"visionos"`, or `"watchos"`).
        settings: A dictionary whose set of keys is defined by the inputs parameter, typically from
            the settings argument found on the implementation function of the current Starlark
            transition.

    Returns:
        A dictionary of `"//command_line_option"`s defined for the current target.
    """
    cpu = _cpu_string(
        environment_arch = environment_arch,
        platform_type = platform_type,
        settings = settings,
    )

    default_platforms = [settings[_CPU_TO_DEFAULT_PLATFORM_FLAG[cpu]]] if _is_bazel_7 else []
    return {
        build_settings_labels.use_tree_artifacts_outputs: force_bundle_outputs if force_bundle_outputs else settings[build_settings_labels.use_tree_artifacts_outputs],
        "//command_line_option:apple_platform_type": platform_type,
        "//command_line_option:apple_platforms": apple_platforms,
        # apple_split_cpu is still needed for Bazel built-in objc_library transition logic and Apple
        # fragment APIs, and it's also required to keep Bazel from optimizing away splits when deps
        # are identical between platforms.
        "//command_line_option:apple_split_cpu": environment_arch if environment_arch else "",
        "//command_line_option:compiler": None,
        "//command_line_option:cpu": cpu,
        "//command_line_option:fission": [],
        "//command_line_option:grte_top": None,
        "//command_line_option:platforms": [apple_platforms[0]] if apple_platforms else default_platforms,
        "//command_line_option:ios_minimum_os": _min_os_version_or_none(
            minimum_os_version = minimum_os_version,
            platform = "ios",
            platform_type = platform_type,
        ),
        "//command_line_option:macos_minimum_os": _min_os_version_or_none(
            minimum_os_version = minimum_os_version,
            platform = "macos",
            platform_type = platform_type,
        ),
        "//command_line_option:minimum_os_version": minimum_os_version,
        "//command_line_option:tvos_minimum_os": _min_os_version_or_none(
            minimum_os_version = minimum_os_version,
            platform = "tvos",
            platform_type = platform_type,
        ),
        "//command_line_option:watchos_minimum_os": _min_os_version_or_none(
            minimum_os_version = minimum_os_version,
            platform = "watchos",
            platform_type = platform_type,
        ),
    }

def _xcframework_split_attr_key(*, arch, environment, platform_type):
    """Return the split attribute key for this target within the XCFramework given linker options.

     Args:
        arch: The architecture of the target that was built. For example, `x86_64` or `arm64`.
        environment: The environment of the target that was built, which corresponds to the
            toolchain's target triple values as reported by `apple_support.link_multi_arch_binary`
            for environment. Typically `device` or `simulator`.
        platform_type: The platform of the target that was built, which corresponds to the
            toolchain's target triple values as reported by `apple_support.link_multi_arch_binary`
            for platform. For example, `ios`, `macos`, `tvos`, `visionos` or `watchos`.

    Returns:
        A string representing the key for this target build found within the XCFramework with a
            format of <platform>_<arch>_<target_environment>, for example `darwin_arm64_device`.
    """

    # NOTE: This only passes "arch" to _cpu_string. To get the keys in the format XCFramework rules
    # expect, the environment is applied to the end of the key, much like it is for an XCFramework's
    # library identifiers as they are generated by Xcode.
    return _cpu_string(
        environment_arch = arch,
        platform_type = platform_type,
    ) + "_" + environment

def _resolved_environment_arch_for_arch(*, arch, environment, platform_type):
    # TODO: Remove `watchos` after https://github.com/bazelbuild/bazel/pull/16181
    if arch == "arm64" and environment == "simulator" and platform_type != "watchos":
        return "sim_arm64"
    return arch

def _command_line_options_for_xcframework_platform(
        *,
        minimum_os_version,
        platform_attr,
        platform_type,
        settings,
        target_environments):
    """Generates a dictionary of command line options keyed by 1:2+ transition for this platform.

    Args:
        minimum_os_version: A string representing the minimum OS version specified for this
            platform, represented as a dotted version number (for example, `"9.0"`).
        platform_attr: The attribute for the apple platform specifying in dictionary form which
            architectures to build for given a target environment as the key for this platform.
        platform_type: The Apple platform for which the rule should build its targets (`"ios"`,
            `"macos"`, `"tvos"`, `"visionos"`, or `"watchos"`).
        settings: A dictionary whose set of keys is defined by the inputs parameter, typically from
            the settings argument found on the implementation function of the current Starlark
            transition.
        target_environments: A list of strings representing target environments supported by the
            platform. Possible strings include "device" and "simulator".

    Returns:
        A dictionary of keys for each <platform>_<arch>_<target_environment> found with a
            corresponding dictionary of `"//command_line_option"`s as each key's value.
    """
    output_dictionary = {}
    for target_environment in target_environments:
        if not platform_attr.get(target_environment):
            continue
        for arch in platform_attr[target_environment]:
            resolved_environment_arch = _resolved_environment_arch_for_arch(
                arch = arch,
                environment = target_environment,
                platform_type = platform_type,
            )

            found_cpu = {
                _xcframework_split_attr_key(
                    arch = arch,
                    environment = target_environment,
                    platform_type = platform_type,
                ): _command_line_options(
                    environment_arch = resolved_environment_arch,
                    minimum_os_version = minimum_os_version,
                    platform_type = platform_type,
                    settings = settings,
                ),
            }
            output_dictionary = dicts.add(found_cpu, output_dictionary)

    return output_dictionary

def _apple_rule_base_transition_impl(settings, attr):
    """Rule transition for Apple rules using Bazel CPUs and a valid Apple split transition."""
    minimum_os_version = attr.minimum_os_version
    platform_type = attr.platform_type
    return _command_line_options(
        environment_arch = _environment_archs(platform_type, minimum_os_version, settings)[0],
        minimum_os_version = minimum_os_version,
        platform_type = platform_type,
        settings = settings,
    )

# These flags are a mix of options defined in native Bazel from the following fragments:
# - https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/analysis/config/CoreOptions.java
# - https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/rules/apple/AppleCommandLineOptions.java
# - https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/rules/cpp/CppOptions.java
_apple_rule_common_transition_inputs = [
    build_settings_labels.use_tree_artifacts_outputs,
] + _CPU_TO_DEFAULT_PLATFORM_FLAG.values()
_apple_rule_base_transition_inputs = _apple_rule_common_transition_inputs + [
    "//command_line_option:cpu",
    "//command_line_option:ios_multi_cpus",
    "//command_line_option:macos_cpus",
    "//command_line_option:tvos_cpus",
    "//command_line_option:watchos_cpus",
] + (["//command_line_option:visionos_cpus"] if _supports_visionos else [])
_apple_platforms_rule_base_transition_inputs = _apple_rule_base_transition_inputs + [
    "//command_line_option:apple_platforms",
    "//command_line_option:incompatible_enable_apple_toolchain_resolution",
]
_apple_platform_transition_inputs = _apple_platforms_rule_base_transition_inputs + [
    "//command_line_option:platforms",
]
_apple_rule_base_transition_outputs = [
    build_settings_labels.use_tree_artifacts_outputs,
    "//command_line_option:apple_platform_type",
    "//command_line_option:apple_platforms",
    "//command_line_option:apple_split_cpu",
    "//command_line_option:compiler",
    "//command_line_option:cpu",
    "//command_line_option:fission",
    "//command_line_option:grte_top",
    "//command_line_option:ios_minimum_os",
    "//command_line_option:macos_minimum_os",
    "//command_line_option:minimum_os_version",
    "//command_line_option:platforms",
    "//command_line_option:tvos_minimum_os",
    "//command_line_option:watchos_minimum_os",
]
_apple_universal_binary_rule_transition_outputs = _apple_rule_base_transition_outputs + [
    "//command_line_option:ios_multi_cpus",
    "//command_line_option:macos_cpus",
    "//command_line_option:tvos_cpus",
    "//command_line_option:watchos_cpus",
] + (["//command_line_option:visionos_cpus"] if _supports_visionos else [])

_apple_rule_base_transition = transition(
    implementation = _apple_rule_base_transition_impl,
    inputs = _apple_rule_base_transition_inputs,
    outputs = _apple_rule_base_transition_outputs,
)

def _apple_platforms_rule_base_transition_impl(settings, attr):
    """Rule transition for Apple rules using Bazel platforms and the Starlark split transition."""
    minimum_os_version = attr.minimum_os_version
    platform_type = attr.platform_type
    environment_arch = None
    if not settings["//command_line_option:incompatible_enable_apple_toolchain_resolution"]:
        # Add fallback to match an anticipated split of Apple cpu-based resolution
        environment_arch = _environment_archs(platform_type, minimum_os_version, settings)[0]
    return _command_line_options(
        apple_platforms = settings["//command_line_option:apple_platforms"],
        environment_arch = environment_arch,
        minimum_os_version = minimum_os_version,
        platform_type = platform_type,
        settings = settings,
    )

_apple_platforms_rule_base_transition = transition(
    implementation = _apple_platforms_rule_base_transition_impl,
    inputs = _apple_platforms_rule_base_transition_inputs,
    outputs = _apple_rule_base_transition_outputs,
)

def _apple_platforms_rule_bundle_output_base_transition_impl(settings, attr):
    """Rule transition for Apple rules using Bazel platforms which force bundle outputs."""
    minimum_os_version = attr.minimum_os_version
    platform_type = attr.platform_type
    environment_arch = None
    if not settings["//command_line_option:incompatible_enable_apple_toolchain_resolution"]:
        # Add fallback to match an anticipated split of Apple cpu-based resolution
        environment_arch = _environment_archs(
            platform_type = platform_type,
            settings = settings,
            minimum_os_version = minimum_os_version,
        )
    return _command_line_options(
        apple_platforms = settings["//command_line_option:apple_platforms"],
        environment_arch = environment_arch[0],
        force_bundle_outputs = True,
        minimum_os_version = minimum_os_version,
        platform_type = platform_type,
        settings = settings,
    )

_apple_platforms_rule_bundle_output_base_transition = transition(
    implementation = _apple_platforms_rule_bundle_output_base_transition_impl,
    inputs = _apple_platforms_rule_base_transition_inputs,
    outputs = _apple_rule_base_transition_outputs,
)

def _apple_rule_arm64_as_arm64e_transition_impl(settings, attr):
    """Rule transition for Apple rules that map arm64 to arm64e."""
    key = "//command_line_option:macos_cpus"

    # These additional settings are sent to both the base implementation and the final transition.
    additional_settings = {key: [arch if arch != "arm64" else "arm64e" for arch in settings[key]]}
    return dicts.add(
        _apple_rule_base_transition_impl(dicts.add(settings, additional_settings), attr),
        additional_settings,
    )

_apple_rule_arm64_as_arm64e_transition = transition(
    implementation = _apple_rule_arm64_as_arm64e_transition_impl,
    inputs = _apple_rule_base_transition_inputs,
    outputs = _apple_rule_base_transition_outputs + ["//command_line_option:macos_cpus"],
)

def _apple_universal_binary_rule_transition_impl(settings, attr):
    """Rule transition for `apple_universal_binary` supporting forced CPUs."""
    forced_cpus = attr.forced_cpus
    platform_type = attr.platform_type
    new_settings = dict(settings)

    # If forced CPUs were given, first we overwrite the existing CPU settings
    # for the target's platform type with those CPUs. We do this before applying
    # the base rule transition in case it wants to read that setting.
    if forced_cpus:
        new_settings[_platform_specific_cpu_setting_name(platform_type)] = forced_cpus

    # Next, apply the base transition and get its output settings.
    new_settings = _apple_rule_base_transition_impl(new_settings, attr)

    # The output settings from applying the base transition won't have the
    # platform-specific CPU flags, so we need to re-apply those before returning
    # our result. For the target's platform type, use the forced CPUs if they
    # were given or use the original value otherwise. For every other platform
    # type, re-propagate the original input.
    #
    # Note that even if we don't have `forced_cpus`, we must provide values for
    # all of the platform-specific CPU flags because they are declared outputs
    # of the transition; the build will fail at analysis time if any are
    # missing.
    for other_type, flag in _PLATFORM_TYPE_TO_CPUS_FLAG.items():
        if not _supports_visionos and other_type == "visionos":
            continue
        if forced_cpus and platform_type == other_type:
            new_settings[flag] = forced_cpus
        else:
            new_settings[flag] = settings[flag]

    return new_settings

_apple_universal_binary_rule_transition = transition(
    implementation = _apple_universal_binary_rule_transition_impl,
    inputs = _apple_rule_base_transition_inputs,
    outputs = _apple_universal_binary_rule_transition_outputs,
)

def _apple_platform_split_transition_impl(settings, attr):
    """Starlark 1:2+ transition for Apple platform-aware rules"""
    output_dictionary = {}
    invalid_requested_archs = []

    if settings["//command_line_option:incompatible_enable_apple_toolchain_resolution"]:
        platforms = (
            settings["//command_line_option:apple_platforms"] or
            settings["//command_line_option:platforms"]
        )
        # Currently there is no "default" platform for Apple-based platforms. If necessary, a
        # default platform could be generated for the rule's underlying platform_type, but for now
        # we work with the assumption that all users of the rules should set an appropriate set of
        # platforms when building Apple targets with `apple_platforms`.

        for index, platform in enumerate(platforms):
            # Create a new, reordered list so that the platform we need to resolve is always first,
            # and the other platforms will follow.
            apple_platforms = list(platforms)
            platform_to_resolve = apple_platforms.pop(index)
            apple_platforms.insert(0, platform_to_resolve)

            if str(platform) not in output_dictionary:
                output_dictionary[str(platform)] = _command_line_options(
                    apple_platforms = apple_platforms,
                    minimum_os_version = attr.minimum_os_version,
                    platform_type = attr.platform_type,
                    settings = settings,
                )

    else:
        minimum_os_version = attr.minimum_os_version
        platform_type = attr.platform_type
        for environment_arch in _environment_archs(platform_type, minimum_os_version, settings):
            found_cpu = _cpu_string(
                environment_arch = environment_arch,
                platform_type = platform_type,
                settings = settings,
            )
            if found_cpu in output_dictionary:
                continue

            environment_arch_is_supported = _is_arch_supported_for_target_tuple(
                environment_arch = environment_arch,
                minimum_os_version = minimum_os_version,
                platform_type = platform_type,
            )
            if not environment_arch_is_supported:
                invalid_requested_arch = {
                    "environment_arch": environment_arch,
                    "minimum_os_version": minimum_os_version,
                    "platform_type": platform_type,
                }

                # NOTE: This logic to filter unsupported Apple CPUs would be good to implement on
                # the platforms side, but it is presently not possible as constraint resolution
                # cannot be performed within a transition.
                #
                # Propagate a warning to the user so that the dropped arch becomes actionable.
                # buildifier: disable=print
                print(
                    ("Warning: The architecture {environment_arch} is not valid for " +
                     "{platform_type} with a minimum OS of {minimum_os_version}. This " +
                     "architecture will be ignored in this build. This will be an error in a " +
                     "future version of the Apple rules. Please address this in your build " +
                     "invocation.").format(
                        **invalid_requested_arch
                    ),
                )
                invalid_requested_archs.append(invalid_requested_arch)
                continue

            output_dictionary[found_cpu] = _command_line_options(
                environment_arch = environment_arch,
                minimum_os_version = minimum_os_version,
                platform_type = platform_type,
                settings = settings,
            )

    if not bool(output_dictionary):
        error_msg = "Could not find any valid architectures to build for the current target.\n\n"
        if invalid_requested_archs:
            error_msg += "Requested the following invalid architectures:\n"
            for invalid_requested_arch in invalid_requested_archs:
                error_msg += (
                    " - {environment_arch} for {platform_type} {minimum_os_version}\n".format(
                        **invalid_requested_arch
                    )
                )
            error_msg += (
                "\nPlease check that the specified architectures are valid for the target's " +
                "specified minimum_os_version.\n"
            )
        fail(error_msg)

    return output_dictionary

_apple_platform_split_transition = transition(
    implementation = _apple_platform_split_transition_impl,
    inputs = _apple_platform_transition_inputs,
    outputs = _apple_rule_base_transition_outputs,
)

def _xcframework_transition_impl(settings, attr):
    """Starlark 1:2+ transition for generation of multiple frameworks for the current target."""
    output_dictionary = {}

    # TODO(b/288582842): Update for visionOS when we're ready to support it in XCFramework rules.
    for platform_type in ["ios", "tvos", "visionos", "watchos", "macos"]:
        platform_attr = getattr(attr, platform_type, None)
        if not platform_attr:
            continue

        # On the macOS platform the platform attr is a list and not a dict as device is the only option.
        # To make the transition logic consistent with the other platforms,
        # we convert the attr to a dict here so that the rest of the logic can be the same.
        platform_attr = {"device": platform_attr} if platform_type == "macos" else platform_attr

        target_environments = ["device"]
        if platform_type != "macos":
            target_environments.append("simulator")

        command_line_options = _command_line_options_for_xcframework_platform(
            minimum_os_version = attr.minimum_os_versions.get(platform_type),
            platform_attr = platform_attr,
            platform_type = platform_type,
            settings = settings,
            target_environments = target_environments,
        )
        output_dictionary = dicts.add(command_line_options, output_dictionary)

    if not output_dictionary:
        fail("Missing a platform type attribute. At least one of 'ios', " +
             "'tvos', 'visionos', 'watchos', or 'macos' attribute is mandatory.")

    return output_dictionary

_xcframework_transition = transition(
    implementation = _xcframework_transition_impl,
    inputs = _apple_rule_common_transition_inputs,
    outputs = _apple_rule_base_transition_outputs,
)

transition_support = struct(
    apple_platform_split_transition = _apple_platform_split_transition,
    apple_platforms_rule_base_transition = _apple_platforms_rule_base_transition,
    apple_platforms_rule_bundle_output_base_transition = _apple_platforms_rule_bundle_output_base_transition,
    apple_rule_arm64_as_arm64e_transition = _apple_rule_arm64_as_arm64e_transition,
    apple_rule_transition = _apple_rule_base_transition,
    apple_universal_binary_rule_transition = _apple_universal_binary_rule_transition,
    xcframework_split_attr_key = _xcframework_split_attr_key,
    xcframework_transition = _xcframework_transition,
)
