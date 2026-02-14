# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""AppIntents intents related actions."""

load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")
load("//apple/internal:intermediates.bzl", "intermediates")

def generate_app_intents_metadata_bundle(
        *,
        actions,
        apple_fragment,
        bundle_binary,
        constvalues_files,
        intents_module_names,
        label,
        source_files,
        target_triples,
        xcode_version_config):
    """Process and generate AppIntents metadata bundle (Metadata.appintents).

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_fragment: An Apple fragment (ctx.fragments.apple).
        bundle_binary: File referencing an application/extension/framework binary.
        constvalues_files: List of swiftconstvalues files generated from Swift source files
            implementing the AppIntents protocol.
        intents_module_names: List of Strings with the module names corresponding to the modules
            found which have intents compiled.
        label: Label for the current target (`ctx.label`).
        source_files: List of Swift source files implementing the AppIntents protocol.
        target_triples: List of Apple target triples from `CcToolchainInfo` providers.
        xcode_version_config: The `apple_common.XcodeVersionConfig` provider from the current ctx.
    Returns:
        File referencing the Metadata.appintents bundle.
    """

    output = intermediates.directory(
        actions = actions,
        target_name = label.name,
        output_discriminator = None,
        dir_name = "Metadata.appintents",
    )

    args = actions.args()
    args.add("/usr/bin/xcrun")
    args.add("appintentsmetadataprocessor")

    args.add("--binary-file", bundle_binary)

    if len(intents_module_names) > 1:
        fail("""
Found the following module names in the top level target {label} for app_intents: {intents_module_names}

App Intents must have only one module name for metadata generation to work correctly.
""".format(
            intents_module_names = ", ".join(intents_module_names),
            label = str(label),
        ))
    elif len(intents_module_names) == 0:
        fail("""
Could not find a module name for app_intents. One is required for App Intents metadata generation.
""")

    args.add("--module-name", intents_module_names[0])
    args.add("--output", output.dirname)
    args.add_all(
        source_files,
        before_each = "--source-files",
    )
    transitive_inputs = [depset(source_files)]
    args.add("--sdk-root", apple_support.path_placeholders.sdkroot())
    args.add_all(target_triples, before_each = "--target-triple")
    if xcode_version_config.xcode_version() >= apple_common.dotted_version("15.0"):
        args.add_all(
            constvalues_files,
            before_each = "--swift-const-vals",
        )
        transitive_inputs.append(depset(constvalues_files))
        args.add("--compile-time-extraction")
    if xcode_version_config.xcode_version() >= apple_common.dotted_version("15.3"):
        # Read the build version from the fourth component of the Xcode version.
        xcode_version_split = str(xcode_version_config.xcode_version()).split(".")
        if len(xcode_version_split) < 4:
            fail("""\
Internal Error: Expected xcode_config to report the Xcode version with the build version as the \
fourth component of the full version string, but instead found {xcode_version_string}. Please file \
an issue with the Apple BUILD rules with repro steps.
""".format(
                xcode_version_string = str(xcode_version_config.xcode_version()),
            ))
        args.add("--xcode-version", xcode_version_split[3])

    apple_support.run_shell(
        actions = actions,
        apple_fragment = apple_fragment,
        arguments = [args],
        command = '''\
set -euo pipefail

exit_status=0
output=$($@ --sdk-root "$SDKROOT" --toolchain-dir "$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain" 2>&1) || exit_status=$?

if [[ "$exit_status" -ne 0 ]]; then
  echo "$output" >&2
  exit $exit_status
elif [[ "$output" == *error:* ]]; then
  echo "$output" >&2
  exit 1
elif [[ "$output" == *"skipping writing output"* ]]; then
  echo "$output" >&2
  exit 1
fi
''',
        inputs = depset([bundle_binary], transitive = transitive_inputs),
        outputs = [output],
        mnemonic = "AppIntentsMetadataProcessor",
        xcode_config = xcode_version_config,
    )

    return output
