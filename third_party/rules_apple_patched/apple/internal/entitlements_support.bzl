# Copyright 2017 The Bazel Authors. All rights reserved.
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

"""Actions that manipulate entitlements and provisioning profiles."""

load(
    "@build_bazel_apple_support//lib:apple_support.bzl",
    "apple_support",
)
load(
    "//apple:common.bzl",
    "entitlements_validation_mode",
)
load(
    "//apple/internal:apple_product_type.bzl",
    "apple_product_type",
)
load(
    "//apple/internal:bundling_support.bzl",
    "bundling_support",
)
load(
    "//apple/internal:resource_actions.bzl",
    "resource_actions",
)
load(
    "//apple/internal/utils:defines.bzl",
    "defines",
)

def _tool_validation_mode(*, is_device, rules_mode):
    """Returns the tools validation_mode to use.

    Args:
      is_device: True if this is a device build, False otherwise.
      rules_mode: The validation_mode attribute of the rule.

    Returns:
      The value to use the for the validation_mode in the entitlement
      options of the tool.
    """
    value = {
        entitlements_validation_mode.error: "error",
        entitlements_validation_mode.warn: "warn",
        entitlements_validation_mode.loose: "error" if is_device else "warn",
        entitlements_validation_mode.skip: "skip",
    }[rules_mode]

    return value

def _new_entitlements_artifact(*, actions, extension, label_name):
    """Returns a new file artifact for entitlements.

    This function creates a new file in an "entitlements" directory in the
    target's location whose name is the target's name with the given extension.

    Args:
      actions: The actions provider from `ctx.actions`.
      extension: The file extension (including the leading dot).
      label_name: The name of the target.

    Returns:
      The requested file object.
    """
    return actions.declare_file(
        "entitlements/%s%s" % (label_name, extension),
    )

def _include_debug_entitlements(*, platform_prerequisites):
    """Returns a value indicating whether debug entitlements should be used.

    Debug entitlements are used if the --device_debug_entitlements command-line
    option indicates that they should be included.

    Debug entitlements are also not used on macOS.

    Args:
      platform_prerequisites: Struct containing information on the platform being targeted.

    Returns:
      True if the debug entitlements should be included, otherwise False.
    """
    if platform_prerequisites.platform_type == apple_common.platform_type.macos:
        return False
    add_debugger_entitlement = defines.bool_value(
        config_vars = platform_prerequisites.config_vars,
        default = None,
        define_name = "apple.add_debugger_entitlement",
    )
    if add_debugger_entitlement != None:
        return add_debugger_entitlement
    if not platform_prerequisites.objc_fragment.uses_device_debug_entitlements:
        return False
    return True

def _include_app_clip_entitlements(*, product_type):
    """Returns a value indicating whether app clip entitlements should be used.

    Args:
      product_type: The product type identifier used to describe the current bundle type.

    Returns:
      True if the app clip entitlements should be included, otherwise False.
    """
    return product_type == apple_product_type.app_clip

def _extract_signing_info(
        *,
        actions,
        entitlements,
        platform_prerequisites,
        provisioning_profile,
        provisioning_profile_tool,
        rule_label):
    """Inspects the current context and extracts the signing information.

    Args:
      actions: The actions provider from `ctx.actions`.
      entitlements: The entitlements file to sign with. Can be `None` if one was not provided.
      platform_prerequisites: Struct containing information on the platform being targeted.
      provisioning_profile: File for the provisioning profile.
      provisioning_profile_tool: A files_to_run for a tool used to extract info from a provisioning
        profile.
      rule_label: The label of the target being analyzed.

    Returns:
      A `struct` with two items: the entitlements file to use, a
      profile_metadata file.
    """
    profile_metadata = None

    if provisioning_profile:
        profile_metadata = _new_entitlements_artifact(
            actions = actions,
            extension = ".profile_metadata",
            label_name = rule_label.name,
        )
        outputs = [profile_metadata]
        control = {
            "profile_metadata": profile_metadata.path,
            "provisioning_profile": provisioning_profile.path,
            "target": str(rule_label),
        }
        if not entitlements:
            # No entitlements, extract the default one from the profile.
            entitlements = _new_entitlements_artifact(
                actions = actions,
                extension = ".extracted_entitlements",
                label_name = rule_label.name,
            )
            control["entitlements"] = entitlements.path
            outputs.append(entitlements)

        control_file = _new_entitlements_artifact(
            actions = actions,
            extension = "provisioning_profile_tool-control",
            label_name = rule_label.name,
        )
        actions.write(
            output = control_file,
            content = json.encode(struct(**control)),
        )

        apple_support.run(
            actions = actions,
            apple_fragment = platform_prerequisites.apple_fragment,
            arguments = [control_file.path],
            executable = provisioning_profile_tool,
            # Since the tools spawns openssl and/or security tool, it doesn't
            # support being sandboxed.
            execution_requirements = {"no-sandbox": "1"},
            inputs = [control_file, provisioning_profile],
            mnemonic = "ExtractFromProvisioningProfile",
            outputs = outputs,
            xcode_config = platform_prerequisites.xcode_version_config,
        )

    return struct(
        entitlements = entitlements,
        profile_metadata = profile_metadata,
    )

def _process_entitlements(
        actions,
        apple_mac_toolchain_info,
        bundle_id,
        entitlements_file,
        platform_prerequisites,
        product_type,
        provisioning_profile,
        rule_label,
        validation_mode):
    """Processes the entitlements for a binary or bundle.

    Entitlements are generated based on a plist-format entitlements file passed
    into the target's entitlements attribute, or extracted from the provisioning
    profile if that attribute is not present. The team prefix is extracted from
    the provisioning profile and the following substitutions are performed on the
    entitlements:

    - "PREFIX.*" -> "PREFIX.BUNDLE_ID" (where BUNDLE_ID is the target's bundle
      ID)
    - "$(AppIdentifierPrefix)" -> "PREFIX."
    - "$(CFBundleIdentifier)" -> "BUNDLE_ID"

    For a device build, the entitlements will be passed as part of the code
    signing process. For a simulator build, the entitlements will be written
    into a Mach-O section `__TEXT,__entitlements` during linking. Each of these
    are handled during those respective processes later in the build, not here.

    Args:
        actions: The object used to register actions.
        apple_mac_toolchain_info: The `struct` of tools from the shared Apple
            toolchain.
        bundle_id: The bundle identifier.
        entitlements_file: The `File` containing the unprocessed entitlements
            (or `None` if none were provided).
        platform_prerequisites: The platform prerequisites.
        product_type: The product type being built.
        provisioning_profile: The `File` representing the provisioning profile,
            from which entitlements will be extracted if `entitlements_file` is
            `None`. This argument may also be `None`.
        rule_label: The `Label` of the target being built.
        validation_mode: A value from `entitlements_validation_mode` describing
            how the entitlements should be validated.

    Returns:
        A struct containing entitlements that should be used for code signing,
        entitlements that should be used for linking, and another (which is one
        of the previous two) that should be returned in `AppleBundleInfo`. Each
        is a `File` containing the processed entitlements, or `None` if there
        are no entitlements being used in the build or no entitlements should be
        embedded via linking.
    """
    bundling_support.validate_bundle_id(bundle_id)

    signing_info = _extract_signing_info(
        actions = actions,
        entitlements = entitlements_file,
        platform_prerequisites = platform_prerequisites,
        provisioning_profile = provisioning_profile,
        provisioning_profile_tool = apple_mac_toolchain_info.provisioning_profile_tool,
        rule_label = rule_label,
    )
    plists = []
    forced_plists = []
    if signing_info.entitlements:
        plists.append(signing_info.entitlements)
    if _include_debug_entitlements(platform_prerequisites = platform_prerequisites):
        get_task_allow = {"get-task-allow": True}
        forced_plists.append(struct(**get_task_allow))
    if _include_app_clip_entitlements(product_type = product_type):
        app_clip = {"com.apple.developer.on-demand-install-capable": True}
        forced_plists.append(struct(**app_clip))

    inputs = list(plists)

    # Return early if there is no entitlements to use.
    if not inputs and not forced_plists:
        return struct(bundle = None, codesigning = None, linking = None)

    final_entitlements = actions.declare_file(
        "%s_entitlements.entitlements" % rule_label.name,
    )

    entitlements_options = {
        "bundle_id": bundle_id,
    }
    if signing_info.profile_metadata:
        inputs.append(signing_info.profile_metadata)
        entitlements_options["profile_metadata_file"] = signing_info.profile_metadata.path
        entitlements_options["validation_mode"] = _tool_validation_mode(
            is_device = platform_prerequisites.platform.is_device,
            rules_mode = validation_mode,
        )

    control = struct(
        plists = [f.path for f in plists],
        forced_plists = forced_plists,
        entitlements_options = struct(**entitlements_options),
        output = final_entitlements.path,
        target = str(rule_label),
        variable_substitutions = struct(CFBundleIdentifier = bundle_id),
    )
    control_file = _new_entitlements_artifact(
        actions = actions,
        extension = "plisttool-control",
        label_name = rule_label.name,
    )
    actions.write(
        output = control_file,
        content = json.encode(control),
    )

    resource_actions.plisttool_action(
        actions = actions,
        control_file = control_file,
        inputs = inputs,
        mnemonic = "ProcessEntitlementsFiles",
        outputs = [final_entitlements],
        platform_prerequisites = platform_prerequisites,
        plisttool = apple_mac_toolchain_info.plisttool,
    )

    if platform_prerequisites.platform.is_device:
        return struct(
            bundle = final_entitlements,
            codesigning = final_entitlements,
            linking = None,
        )

    simulator_entitlements = None
    if _include_debug_entitlements(platform_prerequisites = platform_prerequisites):
        simulator_entitlements = actions.declare_file(
            "%s_entitlements.simulator.entitlements" % rule_label.name,
        )

        simulator_control = struct(
            plists = [],
            forced_plists = [struct(**{"com.apple.security.get-task-allow": True})],
            output = simulator_entitlements.path,
            target = str(rule_label),
        )
        simulator_control_file = _new_entitlements_artifact(
            actions = actions,
            extension = "simulator-plisttool-control",
            label_name = rule_label.name,
        )
        actions.write(
            output = simulator_control_file,
            content = json.encode(simulator_control),
        )

        resource_actions.plisttool_action(
            actions = actions,
            control_file = simulator_control_file,
            inputs = inputs,
            mnemonic = "ProcessSimulatorEntitlementsFile",
            outputs = [simulator_entitlements],
            platform_prerequisites = platform_prerequisites,
            plisttool = apple_mac_toolchain_info.plisttool,
        )

    return struct(
        bundle = final_entitlements,
        codesigning = simulator_entitlements,
        linking = final_entitlements,
    )

def _generate_der_entitlements(
        *,
        actions,
        apple_fragment,
        entitlements,
        label_name,
        xcode_version_config):
    """Creates a DER formatted entitlements file given an existing entitlements plist.

    This converts an entitlements plist into a DER encoded representation identical to that of a
    provisioning profile's "Entitlements" section under the "DER-Encoded-Profile" plist property.

    See Apple's TN3125 for more details on this representation of DER.

    Args:
      actions: The actions provider from `ctx.actions`.
      apple_fragment: An Apple fragment (ctx.fragments.apple).
      entitlements: The entitlements file to sign with.
      label_name: The name of the target being built.
      xcode_version_config: The `apple_common.XcodeVersionConfig` provider from the current context.

    Returns:
      A `File` referencing the generated DER formatted entitlements.
    """

    der_entitlements = actions.declare_file(
        "entitlements/%s.der" % label_name,
    )
    apple_support.run(
        actions = actions,
        apple_fragment = apple_fragment,
        arguments = [
            "query",
            "-f",
            "xml",
            "-i",
            entitlements.path,
            "-o",
            der_entitlements.path,
            "--raw",
        ],
        executable = "/usr/bin/derq",
        inputs = [entitlements],
        mnemonic = "ProcessDEREntitlements",
        outputs = [der_entitlements],
        xcode_config = xcode_version_config,
    )
    return der_entitlements

entitlements_support = struct(
    generate_der_entitlements = _generate_der_entitlements,
    process_entitlements = _process_entitlements,
)
