<!-- Generated with Stardoc: http://skydoc.bazel.build -->

# Bazel rules for creating macOS applications and bundles.

<a id="macos_application"></a>

## macos_application

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_application")

macos_application(<a href="#macos_application-name">name</a>, <a href="#macos_application-deps">deps</a>, <a href="#macos_application-resources">resources</a>, <a href="#macos_application-additional_contents">additional_contents</a>, <a href="#macos_application-additional_linker_inputs">additional_linker_inputs</a>, <a href="#macos_application-app_icons">app_icons</a>,
                  <a href="#macos_application-app_intents">app_intents</a>, <a href="#macos_application-bundle_id">bundle_id</a>, <a href="#macos_application-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_application-bundle_name">bundle_name</a>, <a href="#macos_application-codesign_inputs">codesign_inputs</a>,
                  <a href="#macos_application-codesignopts">codesignopts</a>, <a href="#macos_application-entitlements">entitlements</a>, <a href="#macos_application-entitlements_validation">entitlements_validation</a>, <a href="#macos_application-executable_name">executable_name</a>,
                  <a href="#macos_application-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_application-extensions">extensions</a>, <a href="#macos_application-families">families</a>, <a href="#macos_application-frameworks">frameworks</a>, <a href="#macos_application-include_symbols_in_bundle">include_symbols_in_bundle</a>,
                  <a href="#macos_application-infoplists">infoplists</a>, <a href="#macos_application-ipa_post_processor">ipa_post_processor</a>, <a href="#macos_application-linkopts">linkopts</a>, <a href="#macos_application-locales_to_include">locales_to_include</a>,
                  <a href="#macos_application-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_application-minimum_os_version">minimum_os_version</a>, <a href="#macos_application-platform_type">platform_type</a>,
                  <a href="#macos_application-provisioning_profile">provisioning_profile</a>, <a href="#macos_application-shared_capabilities">shared_capabilities</a>, <a href="#macos_application-stamp">stamp</a>, <a href="#macos_application-strings">strings</a>, <a href="#macos_application-version">version</a>, <a href="#macos_application-xpc_services">xpc_services</a>)
</pre>

Builds and bundles a macOS Application.

This rule creates an application that is a `.app` bundle. If you want to build a
simple command line tool as a standalone binary, use
[`macos_command_line_application`](#macos_command_line_application) instead.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_application-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_application-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-additional_contents"></a>additional_contents |  Files that should be copied into specific subdirectories of the Contents folder in the bundle. The keys of this dictionary are labels pointing to single files, filegroups, or targets; the corresponding value is the name of the subdirectory of Contents where they should be placed.<br><br>The relative directory structure of filegroup contents is preserved when they are copied into the desired Contents subdirectory.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: Label -> String</a> | optional |  `{}`  |
| <a id="macos_application-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcassets/*..appiconset` and there may be only one such `..appiconset` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcassets` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-app_intents"></a>app_intents |  List of dependencies implementing the AppIntents protocol.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="macos_application-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"_"`  |
| <a id="macos_application-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_application-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_application-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_application-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="macos_application-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_application-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-extensions"></a>extensions |  A list of macOS extensions to include in the final application bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_application-frameworks"></a>frameworks |  A list of framework targets (see [`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-include_symbols_in_bundle"></a>include_symbols_in_bundle |  If true and --output_groups=+dsyms is specified, generates `$UUID.symbols` files from all `{binary: .dSYM, ...}` pairs for the application and its dependencies, then packages them under the `Symbols/` directory in the final application bundle.   | Boolean | optional |  `False`  |
| <a id="macos_application-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_application-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_application-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_application-locales_to_include"></a>locales_to_include |  A list of locales to include in the bundle. Only *.lproj directories that are matched will be copied as a part of the build. This value takes precedence (and is preferred) over locales defined using `--define "apple.locales_to_include=..."`.   | List of strings | optional |  `[]`  |
| <a id="macos_application-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_application-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_application-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_application-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.provisionprofile` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_application-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_application-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_application-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_application-xpc_services"></a>xpc_services |  A list of macOS XPC Services to include in the final application bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |


<a id="macos_build_test"></a>

## macos_build_test

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_build_test")

macos_build_test(<a href="#macos_build_test-name">name</a>, <a href="#macos_build_test-minimum_os_version">minimum_os_version</a>, <a href="#macos_build_test-platform_type">platform_type</a>, <a href="#macos_build_test-targets">targets</a>)
</pre>

Test rule to check that the given library targets (Swift, Objective-C, C++)
build for macOS.

Typical usage:

```starlark
macos_build_test(
    name = "my_build_test",
    minimum_os_version = "10.14",
    targets = [
        "//some/package:my_library",
    ],
)
```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_build_test-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_build_test-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version that will be used as the deployment target when building the targets, represented as a dotted version number (for example, `"9.0"`).   | String | required |  |
| <a id="macos_build_test-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_build_test-targets"></a>targets |  The targets to check for successful build.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |


<a id="macos_bundle"></a>

## macos_bundle

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_bundle")

macos_bundle(<a href="#macos_bundle-name">name</a>, <a href="#macos_bundle-deps">deps</a>, <a href="#macos_bundle-resources">resources</a>, <a href="#macos_bundle-additional_contents">additional_contents</a>, <a href="#macos_bundle-additional_linker_inputs">additional_linker_inputs</a>, <a href="#macos_bundle-app_icons">app_icons</a>,
             <a href="#macos_bundle-bundle_extension">bundle_extension</a>, <a href="#macos_bundle-bundle_id">bundle_id</a>, <a href="#macos_bundle-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_bundle-bundle_loader">bundle_loader</a>, <a href="#macos_bundle-bundle_name">bundle_name</a>,
             <a href="#macos_bundle-codesign_inputs">codesign_inputs</a>, <a href="#macos_bundle-codesignopts">codesignopts</a>, <a href="#macos_bundle-entitlements">entitlements</a>, <a href="#macos_bundle-entitlements_validation">entitlements_validation</a>, <a href="#macos_bundle-executable_name">executable_name</a>,
             <a href="#macos_bundle-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_bundle-families">families</a>, <a href="#macos_bundle-infoplists">infoplists</a>, <a href="#macos_bundle-ipa_post_processor">ipa_post_processor</a>, <a href="#macos_bundle-linkopts">linkopts</a>,
             <a href="#macos_bundle-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_bundle-minimum_os_version">minimum_os_version</a>, <a href="#macos_bundle-platform_type">platform_type</a>, <a href="#macos_bundle-provisioning_profile">provisioning_profile</a>,
             <a href="#macos_bundle-shared_capabilities">shared_capabilities</a>, <a href="#macos_bundle-stamp">stamp</a>, <a href="#macos_bundle-strings">strings</a>, <a href="#macos_bundle-version">version</a>)
</pre>

Builds and bundles a macOS Loadable Bundle.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_bundle-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_bundle-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_bundle-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_bundle-additional_contents"></a>additional_contents |  Files that should be copied into specific subdirectories of the Contents folder in the bundle. The keys of this dictionary are labels pointing to single files, filegroups, or targets; the corresponding value is the name of the subdirectory of Contents where they should be placed.<br><br>The relative directory structure of filegroup contents is preserved when they are copied into the desired Contents subdirectory.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: Label -> String</a> | optional |  `{}`  |
| <a id="macos_bundle-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_bundle-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcassets/*..appiconset` and there may be only one such `..appiconset` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcassets` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_bundle-bundle_extension"></a>bundle_extension |  The extension, without a leading dot, that will be used to name the bundle. If this attribute is not set, then the extension will be `.bundle`.   | String | optional |  `""`  |
| <a id="macos_bundle-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="macos_bundle-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="macos_bundle-bundle_loader"></a>bundle_loader |  The target representing the executable that will be loading this bundle. Undefined symbols from the bundle are checked against this execuable during linking as if it were one of the dynamic libraries the bundle was linked with.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_bundle-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_bundle-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_bundle-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_bundle-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_bundle-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="macos_bundle-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_bundle-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_bundle-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_bundle-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_bundle-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_bundle-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_bundle-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_bundle-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_bundle-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_bundle-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.provisionprofile` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_bundle-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_bundle-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_bundle-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_bundle-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_command_line_application"></a>

## macos_command_line_application

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_command_line_application")

macos_command_line_application(<a href="#macos_command_line_application-name">name</a>, <a href="#macos_command_line_application-deps">deps</a>, <a href="#macos_command_line_application-additional_linker_inputs">additional_linker_inputs</a>, <a href="#macos_command_line_application-base_bundle_id">base_bundle_id</a>, <a href="#macos_command_line_application-bundle_id">bundle_id</a>,
                               <a href="#macos_command_line_application-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_command_line_application-codesign_inputs">codesign_inputs</a>, <a href="#macos_command_line_application-codesignopts">codesignopts</a>,
                               <a href="#macos_command_line_application-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_command_line_application-infoplists">infoplists</a>, <a href="#macos_command_line_application-launchdplists">launchdplists</a>, <a href="#macos_command_line_application-linkopts">linkopts</a>,
                               <a href="#macos_command_line_application-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_command_line_application-minimum_os_version">minimum_os_version</a>, <a href="#macos_command_line_application-platform_type">platform_type</a>,
                               <a href="#macos_command_line_application-provisioning_profile">provisioning_profile</a>, <a href="#macos_command_line_application-stamp">stamp</a>, <a href="#macos_command_line_application-version">version</a>)
</pre>

Builds a macOS Command Line Application binary.

A command line application is a standalone binary file, rather than a `.app`
bundle like those produced by [`macos_application`](#macos_application). Unlike
a plain `apple_binary` target, however, this rule supports versioning and
embedding an `Info.plist` into the binary and allows the binary to be
code-signed.

Targets created with `macos_command_line_application` can be executed using
`bazel run`.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_command_line_application-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_command_line_application-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_command_line_application-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_command_line_application-base_bundle_id"></a>base_bundle_id |  The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_command_line_application-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID referenced by `base_bundle_id`.   | String | optional |  `""`  |
| <a id="macos_command_line_application-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from the base bundle ID rule referenced by `base_bundle_id`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"_"`  |
| <a id="macos_command_line_application-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_command_line_application-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_command_line_application-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_command_line_application-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist that represents the application and is embedded into the binary. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_command_line_application-launchdplists"></a>launchdplists |  A list of system wide and per-user daemon/agent configuration files, as specified by the launch plist manual that can be found via `man launchd.plist`. These are XML files that can be loaded into launchd with launchctl, and are required of command line applications that are intended to be used as launch daemons and agents on macOS. All `launchd.plist`s referenced by this attribute will be merged into a single plist and written directly into the `__TEXT`,`__launchd_plist` section of the linked binary.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_command_line_application-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_command_line_application-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_command_line_application-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_command_line_application-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_command_line_application-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.provisionprofile` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_command_line_application-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_command_line_application-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_dylib"></a>

## macos_dylib

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_dylib")

macos_dylib(<a href="#macos_dylib-name">name</a>, <a href="#macos_dylib-deps">deps</a>, <a href="#macos_dylib-additional_linker_inputs">additional_linker_inputs</a>, <a href="#macos_dylib-base_bundle_id">base_bundle_id</a>, <a href="#macos_dylib-bundle_id">bundle_id</a>, <a href="#macos_dylib-bundle_id_suffix">bundle_id_suffix</a>,
            <a href="#macos_dylib-codesign_inputs">codesign_inputs</a>, <a href="#macos_dylib-codesignopts">codesignopts</a>, <a href="#macos_dylib-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_dylib-infoplists">infoplists</a>, <a href="#macos_dylib-linkopts">linkopts</a>,
            <a href="#macos_dylib-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_dylib-minimum_os_version">minimum_os_version</a>, <a href="#macos_dylib-platform_type">platform_type</a>, <a href="#macos_dylib-provisioning_profile">provisioning_profile</a>,
            <a href="#macos_dylib-stamp">stamp</a>, <a href="#macos_dylib-version">version</a>)
</pre>

Builds a macOS Dylib binary.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_dylib-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_dylib-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dylib-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dylib-base_bundle_id"></a>base_bundle_id |  The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_dylib-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID referenced by `base_bundle_id`.   | String | optional |  `""`  |
| <a id="macos_dylib-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from the base bundle ID rule referenced by `base_bundle_id`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"_"`  |
| <a id="macos_dylib-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dylib-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_dylib-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dylib-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist that represents the application and is embedded into the binary. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dylib-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_dylib-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_dylib-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_dylib-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_dylib-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.provisionprofile` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_dylib-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_dylib-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_dynamic_framework"></a>

## macos_dynamic_framework

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_dynamic_framework")

macos_dynamic_framework(<a href="#macos_dynamic_framework-name">name</a>, <a href="#macos_dynamic_framework-deps">deps</a>, <a href="#macos_dynamic_framework-resources">resources</a>, <a href="#macos_dynamic_framework-hdrs">hdrs</a>, <a href="#macos_dynamic_framework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#macos_dynamic_framework-base_bundle_id">base_bundle_id</a>,
                        <a href="#macos_dynamic_framework-bundle_id">bundle_id</a>, <a href="#macos_dynamic_framework-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_dynamic_framework-bundle_name">bundle_name</a>, <a href="#macos_dynamic_framework-bundle_only">bundle_only</a>, <a href="#macos_dynamic_framework-codesign_inputs">codesign_inputs</a>,
                        <a href="#macos_dynamic_framework-codesignopts">codesignopts</a>, <a href="#macos_dynamic_framework-executable_name">executable_name</a>, <a href="#macos_dynamic_framework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_dynamic_framework-extension_safe">extension_safe</a>,
                        <a href="#macos_dynamic_framework-families">families</a>, <a href="#macos_dynamic_framework-frameworks">frameworks</a>, <a href="#macos_dynamic_framework-infoplists">infoplists</a>, <a href="#macos_dynamic_framework-ipa_post_processor">ipa_post_processor</a>, <a href="#macos_dynamic_framework-linkopts">linkopts</a>,
                        <a href="#macos_dynamic_framework-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_dynamic_framework-minimum_os_version">minimum_os_version</a>, <a href="#macos_dynamic_framework-platform_type">platform_type</a>,
                        <a href="#macos_dynamic_framework-provisioning_profile">provisioning_profile</a>, <a href="#macos_dynamic_framework-stamp">stamp</a>, <a href="#macos_dynamic_framework-strings">strings</a>, <a href="#macos_dynamic_framework-version">version</a>)
</pre>

Builds and bundles a macOS dynamic framework that is consumable by Xcode.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_dynamic_framework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_dynamic_framework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dynamic_framework-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dynamic_framework-hdrs"></a>hdrs |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dynamic_framework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dynamic_framework-base_bundle_id"></a>base_bundle_id |  The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_dynamic_framework-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID referenced by `base_bundle_id`.   | String | optional |  `""`  |
| <a id="macos_dynamic_framework-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from the base bundle ID rule referenced by `base_bundle_id`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="macos_dynamic_framework-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_dynamic_framework-bundle_only"></a>bundle_only |  Avoid linking the dynamic framework, but still include it in the app. This is useful when you want to manually dlopen the framework at runtime.   | Boolean | optional |  `False`  |
| <a id="macos_dynamic_framework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dynamic_framework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_dynamic_framework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_dynamic_framework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dynamic_framework-extension_safe"></a>extension_safe |  If true, compiles and links this framework with `-application-extension`, restricting the binary to use only extension-safe APIs.   | Boolean | optional |  `False`  |
| <a id="macos_dynamic_framework-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_dynamic_framework-frameworks"></a>frameworks |  A list of framework targets (see [`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dynamic_framework-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_dynamic_framework-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_dynamic_framework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_dynamic_framework-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_dynamic_framework-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_dynamic_framework-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_dynamic_framework-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_dynamic_framework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_dynamic_framework-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_dynamic_framework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_extension"></a>

## macos_extension

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_extension")

macos_extension(<a href="#macos_extension-name">name</a>, <a href="#macos_extension-deps">deps</a>, <a href="#macos_extension-resources">resources</a>, <a href="#macos_extension-additional_contents">additional_contents</a>, <a href="#macos_extension-additional_linker_inputs">additional_linker_inputs</a>, <a href="#macos_extension-app_icons">app_icons</a>,
                <a href="#macos_extension-bundle_id">bundle_id</a>, <a href="#macos_extension-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_extension-bundle_name">bundle_name</a>, <a href="#macos_extension-codesign_inputs">codesign_inputs</a>, <a href="#macos_extension-codesignopts">codesignopts</a>, <a href="#macos_extension-entitlements">entitlements</a>,
                <a href="#macos_extension-entitlements_validation">entitlements_validation</a>, <a href="#macos_extension-executable_name">executable_name</a>, <a href="#macos_extension-exported_symbols_lists">exported_symbols_lists</a>,
                <a href="#macos_extension-extensionkit_extension">extensionkit_extension</a>, <a href="#macos_extension-families">families</a>, <a href="#macos_extension-frameworks">frameworks</a>, <a href="#macos_extension-infoplists">infoplists</a>, <a href="#macos_extension-ipa_post_processor">ipa_post_processor</a>,
                <a href="#macos_extension-linkopts">linkopts</a>, <a href="#macos_extension-locales_to_include">locales_to_include</a>, <a href="#macos_extension-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_extension-minimum_os_version">minimum_os_version</a>,
                <a href="#macos_extension-platform_type">platform_type</a>, <a href="#macos_extension-provisioning_profile">provisioning_profile</a>, <a href="#macos_extension-shared_capabilities">shared_capabilities</a>, <a href="#macos_extension-stamp">stamp</a>, <a href="#macos_extension-strings">strings</a>, <a href="#macos_extension-version">version</a>)
</pre>

Builds and bundles a macOS Application Extension.

Most macOS app extensions use a plug-in-based architecture where the
executable's entry point is provided by a system framework. However, macOS 11
introduced Widget Extensions that use a traditional `main` entry
point (typically expressed through Swift's `@main` attribute).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_extension-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_extension-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_extension-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_extension-additional_contents"></a>additional_contents |  Files that should be copied into specific subdirectories of the Contents folder in the bundle. The keys of this dictionary are labels pointing to single files, filegroups, or targets; the corresponding value is the name of the subdirectory of Contents where they should be placed.<br><br>The relative directory structure of filegroup contents is preserved when they are copied into the desired Contents subdirectory.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: Label -> String</a> | optional |  `{}`  |
| <a id="macos_extension-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_extension-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcassets/*..appiconset` and there may be only one such `..appiconset` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcassets` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_extension-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="macos_extension-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="macos_extension-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_extension-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_extension-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_extension-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_extension-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="macos_extension-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_extension-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_extension-extensionkit_extension"></a>extensionkit_extension |  Indicates if this target should be treated as an ExtensionKit extension.   | Boolean | optional |  `False`  |
| <a id="macos_extension-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_extension-frameworks"></a>frameworks |  A list of framework targets (see [`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_extension-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_extension-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_extension-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_extension-locales_to_include"></a>locales_to_include |  A list of locales to include in the bundle. Only *.lproj directories that are matched will be copied as a part of the build. This value takes precedence (and is preferred) over locales defined using `--define "apple.locales_to_include=..."`.   | List of strings | optional |  `[]`  |
| <a id="macos_extension-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_extension-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_extension-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_extension-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.provisionprofile` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_extension-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_extension-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_extension-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_extension-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_framework"></a>

## macos_framework

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_framework")

macos_framework(<a href="#macos_framework-name">name</a>, <a href="#macos_framework-deps">deps</a>, <a href="#macos_framework-resources">resources</a>, <a href="#macos_framework-hdrs">hdrs</a>, <a href="#macos_framework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#macos_framework-base_bundle_id">base_bundle_id</a>, <a href="#macos_framework-bundle_id">bundle_id</a>,
                <a href="#macos_framework-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_framework-bundle_name">bundle_name</a>, <a href="#macos_framework-bundle_only">bundle_only</a>, <a href="#macos_framework-codesign_inputs">codesign_inputs</a>, <a href="#macos_framework-codesignopts">codesignopts</a>,
                <a href="#macos_framework-executable_name">executable_name</a>, <a href="#macos_framework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_framework-extension_safe">extension_safe</a>, <a href="#macos_framework-families">families</a>, <a href="#macos_framework-frameworks">frameworks</a>,
                <a href="#macos_framework-infoplists">infoplists</a>, <a href="#macos_framework-ipa_post_processor">ipa_post_processor</a>, <a href="#macos_framework-linkopts">linkopts</a>, <a href="#macos_framework-minimum_deployment_os_version">minimum_deployment_os_version</a>,
                <a href="#macos_framework-minimum_os_version">minimum_os_version</a>, <a href="#macos_framework-platform_type">platform_type</a>, <a href="#macos_framework-provisioning_profile">provisioning_profile</a>, <a href="#macos_framework-stamp">stamp</a>, <a href="#macos_framework-strings">strings</a>, <a href="#macos_framework-version">version</a>)
</pre>

Builds and bundles an macOS Dynamic Framework.

To use this framework for your app and extensions, list it in the `frameworks` attributes
of those `macos_application` and/or `macos_extension` rules.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_framework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_framework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_framework-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_framework-hdrs"></a>hdrs |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_framework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_framework-base_bundle_id"></a>base_bundle_id |  The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_framework-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID referenced by `base_bundle_id`.   | String | optional |  `""`  |
| <a id="macos_framework-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from the base bundle ID rule referenced by `base_bundle_id`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="macos_framework-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_framework-bundle_only"></a>bundle_only |  Avoid linking the dynamic framework, but still include it in the app. This is useful when you want to manually dlopen the framework at runtime.   | Boolean | optional |  `False`  |
| <a id="macos_framework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_framework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_framework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_framework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_framework-extension_safe"></a>extension_safe |  If true, compiles and links this framework with `-application-extension`, restricting the binary to use only extension-safe APIs.   | Boolean | optional |  `False`  |
| <a id="macos_framework-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_framework-frameworks"></a>frameworks |  A list of framework targets (see [`macos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-macos.md#macos_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_framework-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_framework-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_framework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_framework-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_framework-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_framework-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_framework-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_framework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_framework-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_framework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_kernel_extension"></a>

## macos_kernel_extension

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_kernel_extension")

macos_kernel_extension(<a href="#macos_kernel_extension-name">name</a>, <a href="#macos_kernel_extension-deps">deps</a>, <a href="#macos_kernel_extension-resources">resources</a>, <a href="#macos_kernel_extension-additional_contents">additional_contents</a>, <a href="#macos_kernel_extension-additional_linker_inputs">additional_linker_inputs</a>,
                       <a href="#macos_kernel_extension-bundle_id">bundle_id</a>, <a href="#macos_kernel_extension-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_kernel_extension-bundle_name">bundle_name</a>, <a href="#macos_kernel_extension-codesign_inputs">codesign_inputs</a>, <a href="#macos_kernel_extension-codesignopts">codesignopts</a>,
                       <a href="#macos_kernel_extension-entitlements">entitlements</a>, <a href="#macos_kernel_extension-entitlements_validation">entitlements_validation</a>, <a href="#macos_kernel_extension-executable_name">executable_name</a>, <a href="#macos_kernel_extension-exported_symbols_lists">exported_symbols_lists</a>,
                       <a href="#macos_kernel_extension-families">families</a>, <a href="#macos_kernel_extension-infoplists">infoplists</a>, <a href="#macos_kernel_extension-ipa_post_processor">ipa_post_processor</a>, <a href="#macos_kernel_extension-linkopts">linkopts</a>,
                       <a href="#macos_kernel_extension-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_kernel_extension-minimum_os_version">minimum_os_version</a>, <a href="#macos_kernel_extension-platform_type">platform_type</a>,
                       <a href="#macos_kernel_extension-provisioning_profile">provisioning_profile</a>, <a href="#macos_kernel_extension-shared_capabilities">shared_capabilities</a>, <a href="#macos_kernel_extension-stamp">stamp</a>, <a href="#macos_kernel_extension-strings">strings</a>, <a href="#macos_kernel_extension-version">version</a>)
</pre>

Builds and bundles a macOS Kernel Extension.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_kernel_extension-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_kernel_extension-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_kernel_extension-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_kernel_extension-additional_contents"></a>additional_contents |  Files that should be copied into specific subdirectories of the Contents folder in the bundle. The keys of this dictionary are labels pointing to single files, filegroups, or targets; the corresponding value is the name of the subdirectory of Contents where they should be placed.<br><br>The relative directory structure of filegroup contents is preserved when they are copied into the desired Contents subdirectory.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: Label -> String</a> | optional |  `{}`  |
| <a id="macos_kernel_extension-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_kernel_extension-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="macos_kernel_extension-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"_"`  |
| <a id="macos_kernel_extension-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_kernel_extension-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_kernel_extension-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_kernel_extension-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_kernel_extension-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="macos_kernel_extension-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_kernel_extension-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_kernel_extension-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_kernel_extension-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_kernel_extension-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_kernel_extension-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_kernel_extension-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_kernel_extension-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_kernel_extension-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_kernel_extension-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.provisionprofile` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_kernel_extension-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_kernel_extension-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_kernel_extension-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_kernel_extension-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_quick_look_plugin"></a>

## macos_quick_look_plugin

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_quick_look_plugin")

macos_quick_look_plugin(<a href="#macos_quick_look_plugin-name">name</a>, <a href="#macos_quick_look_plugin-deps">deps</a>, <a href="#macos_quick_look_plugin-resources">resources</a>, <a href="#macos_quick_look_plugin-additional_contents">additional_contents</a>, <a href="#macos_quick_look_plugin-additional_linker_inputs">additional_linker_inputs</a>,
                        <a href="#macos_quick_look_plugin-bundle_id">bundle_id</a>, <a href="#macos_quick_look_plugin-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_quick_look_plugin-bundle_name">bundle_name</a>, <a href="#macos_quick_look_plugin-codesign_inputs">codesign_inputs</a>, <a href="#macos_quick_look_plugin-codesignopts">codesignopts</a>,
                        <a href="#macos_quick_look_plugin-entitlements">entitlements</a>, <a href="#macos_quick_look_plugin-entitlements_validation">entitlements_validation</a>, <a href="#macos_quick_look_plugin-executable_name">executable_name</a>,
                        <a href="#macos_quick_look_plugin-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_quick_look_plugin-families">families</a>, <a href="#macos_quick_look_plugin-infoplists">infoplists</a>, <a href="#macos_quick_look_plugin-ipa_post_processor">ipa_post_processor</a>, <a href="#macos_quick_look_plugin-linkopts">linkopts</a>,
                        <a href="#macos_quick_look_plugin-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_quick_look_plugin-minimum_os_version">minimum_os_version</a>, <a href="#macos_quick_look_plugin-platform_type">platform_type</a>,
                        <a href="#macos_quick_look_plugin-provisioning_profile">provisioning_profile</a>, <a href="#macos_quick_look_plugin-shared_capabilities">shared_capabilities</a>, <a href="#macos_quick_look_plugin-stamp">stamp</a>, <a href="#macos_quick_look_plugin-strings">strings</a>, <a href="#macos_quick_look_plugin-version">version</a>)
</pre>

Builds and bundles a macOS Quick Look Plugin.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_quick_look_plugin-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_quick_look_plugin-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_quick_look_plugin-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_quick_look_plugin-additional_contents"></a>additional_contents |  Files that should be copied into specific subdirectories of the Contents folder in the bundle. The keys of this dictionary are labels pointing to single files, filegroups, or targets; the corresponding value is the name of the subdirectory of Contents where they should be placed.<br><br>The relative directory structure of filegroup contents is preserved when they are copied into the desired Contents subdirectory.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: Label -> String</a> | optional |  `{}`  |
| <a id="macos_quick_look_plugin-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_quick_look_plugin-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="macos_quick_look_plugin-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"_"`  |
| <a id="macos_quick_look_plugin-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_quick_look_plugin-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_quick_look_plugin-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_quick_look_plugin-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_quick_look_plugin-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="macos_quick_look_plugin-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_quick_look_plugin-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_quick_look_plugin-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_quick_look_plugin-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_quick_look_plugin-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_quick_look_plugin-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_quick_look_plugin-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_quick_look_plugin-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_quick_look_plugin-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_quick_look_plugin-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.provisionprofile` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_quick_look_plugin-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_quick_look_plugin-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_quick_look_plugin-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_quick_look_plugin-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_spotlight_importer"></a>

## macos_spotlight_importer

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_spotlight_importer")

macos_spotlight_importer(<a href="#macos_spotlight_importer-name">name</a>, <a href="#macos_spotlight_importer-deps">deps</a>, <a href="#macos_spotlight_importer-resources">resources</a>, <a href="#macos_spotlight_importer-additional_contents">additional_contents</a>, <a href="#macos_spotlight_importer-additional_linker_inputs">additional_linker_inputs</a>,
                         <a href="#macos_spotlight_importer-bundle_id">bundle_id</a>, <a href="#macos_spotlight_importer-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_spotlight_importer-bundle_name">bundle_name</a>, <a href="#macos_spotlight_importer-codesign_inputs">codesign_inputs</a>, <a href="#macos_spotlight_importer-codesignopts">codesignopts</a>,
                         <a href="#macos_spotlight_importer-entitlements">entitlements</a>, <a href="#macos_spotlight_importer-entitlements_validation">entitlements_validation</a>, <a href="#macos_spotlight_importer-executable_name">executable_name</a>,
                         <a href="#macos_spotlight_importer-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_spotlight_importer-families">families</a>, <a href="#macos_spotlight_importer-infoplists">infoplists</a>, <a href="#macos_spotlight_importer-ipa_post_processor">ipa_post_processor</a>, <a href="#macos_spotlight_importer-linkopts">linkopts</a>,
                         <a href="#macos_spotlight_importer-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_spotlight_importer-minimum_os_version">minimum_os_version</a>, <a href="#macos_spotlight_importer-platform_type">platform_type</a>,
                         <a href="#macos_spotlight_importer-provisioning_profile">provisioning_profile</a>, <a href="#macos_spotlight_importer-shared_capabilities">shared_capabilities</a>, <a href="#macos_spotlight_importer-stamp">stamp</a>, <a href="#macos_spotlight_importer-strings">strings</a>, <a href="#macos_spotlight_importer-version">version</a>)
</pre>

Builds and bundles a macOS Spotlight Importer.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_spotlight_importer-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_spotlight_importer-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_spotlight_importer-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_spotlight_importer-additional_contents"></a>additional_contents |  Files that should be copied into specific subdirectories of the Contents folder in the bundle. The keys of this dictionary are labels pointing to single files, filegroups, or targets; the corresponding value is the name of the subdirectory of Contents where they should be placed.<br><br>The relative directory structure of filegroup contents is preserved when they are copied into the desired Contents subdirectory.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: Label -> String</a> | optional |  `{}`  |
| <a id="macos_spotlight_importer-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_spotlight_importer-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="macos_spotlight_importer-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"_"`  |
| <a id="macos_spotlight_importer-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_spotlight_importer-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_spotlight_importer-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_spotlight_importer-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_spotlight_importer-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="macos_spotlight_importer-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_spotlight_importer-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_spotlight_importer-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_spotlight_importer-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_spotlight_importer-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_spotlight_importer-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_spotlight_importer-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_spotlight_importer-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_spotlight_importer-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_spotlight_importer-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.provisionprofile` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_spotlight_importer-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_spotlight_importer-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_spotlight_importer-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_spotlight_importer-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_static_framework"></a>

## macos_static_framework

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_static_framework")

macos_static_framework(<a href="#macos_static_framework-name">name</a>, <a href="#macos_static_framework-deps">deps</a>, <a href="#macos_static_framework-resources">resources</a>, <a href="#macos_static_framework-hdrs">hdrs</a>, <a href="#macos_static_framework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#macos_static_framework-avoid_deps">avoid_deps</a>,
                       <a href="#macos_static_framework-bundle_name">bundle_name</a>, <a href="#macos_static_framework-codesign_inputs">codesign_inputs</a>, <a href="#macos_static_framework-codesignopts">codesignopts</a>, <a href="#macos_static_framework-exclude_resources">exclude_resources</a>, <a href="#macos_static_framework-executable_name">executable_name</a>,
                       <a href="#macos_static_framework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_static_framework-families">families</a>, <a href="#macos_static_framework-ipa_post_processor">ipa_post_processor</a>, <a href="#macos_static_framework-linkopts">linkopts</a>,
                       <a href="#macos_static_framework-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_static_framework-minimum_os_version">minimum_os_version</a>, <a href="#macos_static_framework-platform_type">platform_type</a>, <a href="#macos_static_framework-stamp">stamp</a>,
                       <a href="#macos_static_framework-strings">strings</a>, <a href="#macos_static_framework-umbrella_header">umbrella_header</a>, <a href="#macos_static_framework-version">version</a>)
</pre>

Builds and bundles a macOS static framework for third-party distribution.

A static framework is bundled like a dynamic framework except that the embedded
binary is a static library rather than a dynamic library. It is intended to
create distributable static SDKs or artifacts that can be easily imported into
other Xcode projects; it is specifically **not** intended to be used as a
dependency of other Bazel targets. For that use case, use the corresponding
`objc_library` targets directly.

Unlike other macOS bundles, the fat binary in an `macos_static_framework` may
simultaneously contain simulator and device architectures (that is, you can
build a single framework artifact that works for all architectures by specifying
`--macos_cpus=x86_64,arm64` when you build).

`macos_static_framework` supports Swift, but there are some constraints:

* `macos_static_framework` with Swift only works with Xcode 12 and above, since
  the required Swift functionality for module compatibility is available in
  Swift 5.1.
* `macos_static_framework` only supports a single direct `swift_library` target
  that does not depend transitively on any other `swift_library` targets. The
  Swift compiler expects a framework to contain a single Swift module, and each
  `swift_library` target is its own module by definition.
* `macos_static_framework` does not support mixed Objective-C and Swift public
  interfaces. This means that the `umbrella_header` and `hdrs` attributes are
  unavailable when using `swift_library` dependencies. You are allowed to depend
  on `objc_library` from the main `swift_library` dependency, but note that only
  the `swift_library`'s public interface will be available to users of the
  static framework.

When using Swift, the `macos_static_framework` bundles `swiftinterface` and
`swiftdocs` file for each of the required architectures. It also bundles an
umbrella header which is the header generated by the single `swift_library`
target. Finally, it also bundles a `module.modulemap` file pointing to the
umbrella header for Objetive-C module compatibility. This umbrella header and
modulemap can be skipped by disabling the `swift.no_generated_header` feature (
i.e. `--features=-swift.no_generated_header`).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_static_framework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_static_framework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_static_framework-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_static_framework-hdrs"></a>hdrs |  A list of `.h` files that will be publicly exposed by this framework. These headers should have framework-relative imports, and if non-empty, an umbrella header named `%{bundle_name}.h` will also be generated that imports all of the headers listed here.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_static_framework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_static_framework-avoid_deps"></a>avoid_deps |  A list of library targets on which this framework depends in order to compile, but the transitive closure of which will not be linked into the framework's binary.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_static_framework-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_static_framework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_static_framework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_static_framework-exclude_resources"></a>exclude_resources |  Indicates whether resources should be excluded from the bundle. This can be used to avoid unnecessarily bundling resources if the static framework is being distributed in a different fashion, such as a Cocoapod.   | Boolean | optional |  `False`  |
| <a id="macos_static_framework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_static_framework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_static_framework-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_static_framework-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_static_framework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_static_framework-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_static_framework-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_static_framework-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_static_framework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_static_framework-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_static_framework-umbrella_header"></a>umbrella_header |  An optional single .h file to use as the umbrella header for this framework. Usually, this header will have the same name as this target, so that clients can load the header using the #import <MyFramework/MyFramework.h> format. If this attribute is not specified (the common use case), an umbrella header will be generated under the same name as this target.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_static_framework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="macos_ui_test"></a>

## macos_ui_test

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_ui_test")

macos_ui_test(<a href="#macos_ui_test-name">name</a>, <a href="#macos_ui_test-deps">deps</a>, <a href="#macos_ui_test-data">data</a>, <a href="#macos_ui_test-bundle_name">bundle_name</a>, <a href="#macos_ui_test-collect_code_coverage">collect_code_coverage</a>, <a href="#macos_ui_test-env">env</a>, <a href="#macos_ui_test-env_inherit">env_inherit</a>,
              <a href="#macos_ui_test-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_ui_test-minimum_os_version">minimum_os_version</a>, <a href="#macos_ui_test-platform_type">platform_type</a>, <a href="#macos_ui_test-runner">runner</a>,
              <a href="#macos_ui_test-test_coverage_manifest">test_coverage_manifest</a>, <a href="#macos_ui_test-test_filter">test_filter</a>, <a href="#macos_ui_test-test_host">test_host</a>, <a href="#macos_ui_test-test_host_is_bundle_loader">test_host_is_bundle_loader</a>)
</pre>

Builds and bundles an iOS UI `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

Note: macOS UI tests are not currently supported in the default test runner.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_ui_test-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_ui_test-deps"></a>deps |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_ui_test-data"></a>data |  Files to be made available to the test during its execution.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_ui_test-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_ui_test-collect_code_coverage"></a>collect_code_coverage |  Whether to collect code coverage for this test if `--collect_code_coverage=yes`.   | Boolean | optional |  `True`  |
| <a id="macos_ui_test-env"></a>env |  Dictionary of environment variables that should be set during the test execution. The values of the dictionary are subject to "Make" variable expansion.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="macos_ui_test-env_inherit"></a>env_inherit |  List of environment variables to inherit from the external environment.   | List of strings | optional |  `[]`  |
| <a id="macos_ui_test-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_ui_test-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_ui_test-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_ui_test-runner"></a>runner |  The runner target that will provide the logic on how to run the tests. Needs to provide the AppleTestRunnerInfo provider.   | <a href="https://bazel.build/concepts/labels">Label</a> | required |  |
| <a id="macos_ui_test-test_coverage_manifest"></a>test_coverage_manifest |  A file that will be used in lcov export calls to limit the scope of files instrumented with coverage.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_ui_test-test_filter"></a>test_filter |  Test filter string that will be passed into the test runner to select which tests will run.   | String | optional |  `""`  |
| <a id="macos_ui_test-test_host"></a>test_host |  -   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_ui_test-test_host_is_bundle_loader"></a>test_host_is_bundle_loader |  Whether the 'test_host' should be used as the -bundle_loader to allow testing the symbols from the test host app   | Boolean | optional |  `True`  |


<a id="macos_unit_test"></a>

## macos_unit_test

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_unit_test")

macos_unit_test(<a href="#macos_unit_test-name">name</a>, <a href="#macos_unit_test-deps">deps</a>, <a href="#macos_unit_test-data">data</a>, <a href="#macos_unit_test-bundle_name">bundle_name</a>, <a href="#macos_unit_test-collect_code_coverage">collect_code_coverage</a>, <a href="#macos_unit_test-env">env</a>, <a href="#macos_unit_test-env_inherit">env_inherit</a>,
                <a href="#macos_unit_test-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#macos_unit_test-minimum_os_version">minimum_os_version</a>, <a href="#macos_unit_test-platform_type">platform_type</a>, <a href="#macos_unit_test-runner">runner</a>,
                <a href="#macos_unit_test-test_coverage_manifest">test_coverage_manifest</a>, <a href="#macos_unit_test-test_filter">test_filter</a>, <a href="#macos_unit_test-test_host">test_host</a>, <a href="#macos_unit_test-test_host_is_bundle_loader">test_host_is_bundle_loader</a>)
</pre>

Builds and bundles a macOS unit `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

`macos_unit_test` targets can work in two modes: as app or library tests. If the
`test_host` attribute is set to an `macos_application` target, the tests will
run within that application's context. If no `test_host` is provided, the tests
will run outside the context of an macOS application. Because of this, certain
functionalities might not be present (e.g. UI layout, NSUserDefaults). You can
find more information about testing for Apple platforms
[here](https://developer.apple.com/library/content/documentation/DeveloperTools/Conceptual/testing_with_xcode/chapters/03-testing_basics.html).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_unit_test-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_unit_test-deps"></a>deps |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_unit_test-data"></a>data |  Files to be made available to the test during its execution.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_unit_test-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_unit_test-collect_code_coverage"></a>collect_code_coverage |  Whether to collect code coverage for this test if `--collect_code_coverage=yes`.   | Boolean | optional |  `True`  |
| <a id="macos_unit_test-env"></a>env |  Dictionary of environment variables that should be set during the test execution. The values of the dictionary are subject to "Make" variable expansion.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="macos_unit_test-env_inherit"></a>env_inherit |  List of environment variables to inherit from the external environment.   | List of strings | optional |  `[]`  |
| <a id="macos_unit_test-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_unit_test-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_unit_test-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_unit_test-runner"></a>runner |  The runner target that will provide the logic on how to run the tests. Needs to provide the AppleTestRunnerInfo provider.   | <a href="https://bazel.build/concepts/labels">Label</a> | required |  |
| <a id="macos_unit_test-test_coverage_manifest"></a>test_coverage_manifest |  A file that will be used in lcov export calls to limit the scope of files instrumented with coverage.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_unit_test-test_filter"></a>test_filter |  Test filter string that will be passed into the test runner to select which tests will run.   | String | optional |  `""`  |
| <a id="macos_unit_test-test_host"></a>test_host |  -   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_unit_test-test_host_is_bundle_loader"></a>test_host_is_bundle_loader |  Whether the 'test_host' should be used as the -bundle_loader to allow testing the symbols from the test host app   | Boolean | optional |  `True`  |


<a id="macos_xpc_service"></a>

## macos_xpc_service

<pre>
load("@rules_apple//apple:macos.doc.bzl", "macos_xpc_service")

macos_xpc_service(<a href="#macos_xpc_service-name">name</a>, <a href="#macos_xpc_service-deps">deps</a>, <a href="#macos_xpc_service-resources">resources</a>, <a href="#macos_xpc_service-additional_contents">additional_contents</a>, <a href="#macos_xpc_service-additional_linker_inputs">additional_linker_inputs</a>, <a href="#macos_xpc_service-bundle_id">bundle_id</a>,
                  <a href="#macos_xpc_service-bundle_id_suffix">bundle_id_suffix</a>, <a href="#macos_xpc_service-bundle_name">bundle_name</a>, <a href="#macos_xpc_service-codesign_inputs">codesign_inputs</a>, <a href="#macos_xpc_service-codesignopts">codesignopts</a>, <a href="#macos_xpc_service-entitlements">entitlements</a>,
                  <a href="#macos_xpc_service-entitlements_validation">entitlements_validation</a>, <a href="#macos_xpc_service-executable_name">executable_name</a>, <a href="#macos_xpc_service-exported_symbols_lists">exported_symbols_lists</a>, <a href="#macos_xpc_service-families">families</a>,
                  <a href="#macos_xpc_service-infoplists">infoplists</a>, <a href="#macos_xpc_service-ipa_post_processor">ipa_post_processor</a>, <a href="#macos_xpc_service-linkopts">linkopts</a>, <a href="#macos_xpc_service-minimum_deployment_os_version">minimum_deployment_os_version</a>,
                  <a href="#macos_xpc_service-minimum_os_version">minimum_os_version</a>, <a href="#macos_xpc_service-platform_type">platform_type</a>, <a href="#macos_xpc_service-provisioning_profile">provisioning_profile</a>, <a href="#macos_xpc_service-shared_capabilities">shared_capabilities</a>, <a href="#macos_xpc_service-stamp">stamp</a>,
                  <a href="#macos_xpc_service-strings">strings</a>, <a href="#macos_xpc_service-version">version</a>)
</pre>

Builds and bundles a macOS XPC Service.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="macos_xpc_service-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="macos_xpc_service-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_xpc_service-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_xpc_service-additional_contents"></a>additional_contents |  Files that should be copied into specific subdirectories of the Contents folder in the bundle. The keys of this dictionary are labels pointing to single files, filegroups, or targets; the corresponding value is the name of the subdirectory of Contents where they should be placed.<br><br>The relative directory structure of filegroup contents is preserved when they are copied into the desired Contents subdirectory.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: Label -> String</a> | optional |  `{}`  |
| <a id="macos_xpc_service-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_xpc_service-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="macos_xpc_service-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"_"`  |
| <a id="macos_xpc_service-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_xpc_service-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_xpc_service-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="macos_xpc_service-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_xpc_service-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="macos_xpc_service-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="macos_xpc_service-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_xpc_service-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["mac"]`  |
| <a id="macos_xpc_service-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="macos_xpc_service-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_xpc_service-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="macos_xpc_service-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="macos_xpc_service-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="macos_xpc_service-platform_type"></a>platform_type |  -   | String | optional |  `"macos"`  |
| <a id="macos_xpc_service-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.provisionprofile` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="macos_xpc_service-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_xpc_service-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="macos_xpc_service-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="macos_xpc_service-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


