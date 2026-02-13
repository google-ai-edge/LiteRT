<!-- Generated with Stardoc: http://skydoc.bazel.build -->

# Bazel rules for creating visionOS applications and bundles.

<a id="visionos_application"></a>

## visionos_application

<pre>
load("@rules_apple//apple:visionos.doc.bzl", "visionos_application")

visionos_application(<a href="#visionos_application-name">name</a>, <a href="#visionos_application-deps">deps</a>, <a href="#visionos_application-resources">resources</a>, <a href="#visionos_application-additional_linker_inputs">additional_linker_inputs</a>, <a href="#visionos_application-app_icons">app_icons</a>, <a href="#visionos_application-app_intents">app_intents</a>,
                     <a href="#visionos_application-bundle_id">bundle_id</a>, <a href="#visionos_application-bundle_id_suffix">bundle_id_suffix</a>, <a href="#visionos_application-bundle_name">bundle_name</a>, <a href="#visionos_application-codesign_inputs">codesign_inputs</a>, <a href="#visionos_application-codesignopts">codesignopts</a>,
                     <a href="#visionos_application-entitlements">entitlements</a>, <a href="#visionos_application-entitlements_validation">entitlements_validation</a>, <a href="#visionos_application-executable_name">executable_name</a>, <a href="#visionos_application-exported_symbols_lists">exported_symbols_lists</a>,
                     <a href="#visionos_application-extensions">extensions</a>, <a href="#visionos_application-families">families</a>, <a href="#visionos_application-frameworks">frameworks</a>, <a href="#visionos_application-infoplists">infoplists</a>, <a href="#visionos_application-ipa_post_processor">ipa_post_processor</a>,
                     <a href="#visionos_application-launch_storyboard">launch_storyboard</a>, <a href="#visionos_application-linkopts">linkopts</a>, <a href="#visionos_application-locales_to_include">locales_to_include</a>, <a href="#visionos_application-minimum_deployment_os_version">minimum_deployment_os_version</a>,
                     <a href="#visionos_application-minimum_os_version">minimum_os_version</a>, <a href="#visionos_application-platform_type">platform_type</a>, <a href="#visionos_application-primary_app_icon">primary_app_icon</a>, <a href="#visionos_application-provisioning_profile">provisioning_profile</a>,
                     <a href="#visionos_application-shared_capabilities">shared_capabilities</a>, <a href="#visionos_application-stamp">stamp</a>, <a href="#visionos_application-strings">strings</a>, <a href="#visionos_application-version">version</a>)
</pre>

Builds and bundles a visionOS Application.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="visionos_application-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="visionos_application-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcassets/*..solidimagestack` and there may be only one such `..solidimagestack` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcassets` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-app_intents"></a>app_intents |  List of dependencies implementing the AppIntents protocol.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="visionos_application-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="visionos_application-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_application-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="visionos_application-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_application-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="visionos_application-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_application-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-extensions"></a>extensions |  A list of visionOS extensions to include in the final application bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["vision"]`  |
| <a id="visionos_application-frameworks"></a>frameworks |  A list of framework targets (see [`visionos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-visionos.md#visionos_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="visionos_application-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_application-launch_storyboard"></a>launch_storyboard |  The `.storyboard` file that should be used as the launch screen for the application. The provided file will be compiled into the appropriate format (`.storyboardc`) and placed in the root of the final bundle. The generated file will also be registered in the bundle's Info.plist under the key `UILaunchStoryboardName`.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_application-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="visionos_application-locales_to_include"></a>locales_to_include |  A list of locales to include in the bundle. Only *.lproj directories that are matched will be copied as a part of the build. This value takes precedence (and is preferred) over locales defined using `--define "apple.locales_to_include=..."`.   | List of strings | optional |  `[]`  |
| <a id="visionos_application-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="visionos_application-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="visionos_application-platform_type"></a>platform_type |  -   | String | optional |  `"visionos"`  |
| <a id="visionos_application-primary_app_icon"></a>primary_app_icon |  An optional String to identify the name of the primary app icon when alternate app icons have been provided for the app. This should match both the `*.icon` bundle in iOS/macOS/watchOS 26+ and the `*..xcassets/.appiconset` bundle's AppIcon resource in previous versions of iOS/macOS/watchOS.   | String | optional |  `""`  |
| <a id="visionos_application-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_application-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="visionos_application-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_application-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="visionos_build_test"></a>

## visionos_build_test

<pre>
load("@rules_apple//apple:visionos.doc.bzl", "visionos_build_test")

visionos_build_test(<a href="#visionos_build_test-name">name</a>, <a href="#visionos_build_test-minimum_os_version">minimum_os_version</a>, <a href="#visionos_build_test-platform_type">platform_type</a>, <a href="#visionos_build_test-targets">targets</a>)
</pre>

Test rule to check that the given library targets (Swift, Objective-C, C++)
build for visionOS.

Typical usage:

```starlark
visionos_build_test(
    name = "my_build_test",
    minimum_os_version = "1.0",
    targets = [
        "//some/package:my_library",
    ],
)
```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="visionos_build_test-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="visionos_build_test-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version that will be used as the deployment target when building the targets, represented as a dotted version number (for example, `"9.0"`).   | String | required |  |
| <a id="visionos_build_test-platform_type"></a>platform_type |  -   | String | optional |  `"visionos"`  |
| <a id="visionos_build_test-targets"></a>targets |  The targets to check for successful build.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |


<a id="visionos_dynamic_framework"></a>

## visionos_dynamic_framework

<pre>
load("@rules_apple//apple:visionos.doc.bzl", "visionos_dynamic_framework")

visionos_dynamic_framework(<a href="#visionos_dynamic_framework-name">name</a>, <a href="#visionos_dynamic_framework-deps">deps</a>, <a href="#visionos_dynamic_framework-resources">resources</a>, <a href="#visionos_dynamic_framework-hdrs">hdrs</a>, <a href="#visionos_dynamic_framework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#visionos_dynamic_framework-base_bundle_id">base_bundle_id</a>,
                           <a href="#visionos_dynamic_framework-bundle_id">bundle_id</a>, <a href="#visionos_dynamic_framework-bundle_id_suffix">bundle_id_suffix</a>, <a href="#visionos_dynamic_framework-bundle_name">bundle_name</a>, <a href="#visionos_dynamic_framework-bundle_only">bundle_only</a>, <a href="#visionos_dynamic_framework-codesign_inputs">codesign_inputs</a>,
                           <a href="#visionos_dynamic_framework-codesignopts">codesignopts</a>, <a href="#visionos_dynamic_framework-executable_name">executable_name</a>, <a href="#visionos_dynamic_framework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#visionos_dynamic_framework-extension_safe">extension_safe</a>,
                           <a href="#visionos_dynamic_framework-families">families</a>, <a href="#visionos_dynamic_framework-frameworks">frameworks</a>, <a href="#visionos_dynamic_framework-infoplists">infoplists</a>, <a href="#visionos_dynamic_framework-ipa_post_processor">ipa_post_processor</a>, <a href="#visionos_dynamic_framework-linkopts">linkopts</a>,
                           <a href="#visionos_dynamic_framework-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#visionos_dynamic_framework-minimum_os_version">minimum_os_version</a>, <a href="#visionos_dynamic_framework-platform_type">platform_type</a>,
                           <a href="#visionos_dynamic_framework-provisioning_profile">provisioning_profile</a>, <a href="#visionos_dynamic_framework-stamp">stamp</a>, <a href="#visionos_dynamic_framework-strings">strings</a>, <a href="#visionos_dynamic_framework-version">version</a>)
</pre>

Builds and bundles a visionos dynamic framework that is consumable by Xcode.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="visionos_dynamic_framework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="visionos_dynamic_framework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_dynamic_framework-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_dynamic_framework-hdrs"></a>hdrs |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_dynamic_framework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_dynamic_framework-base_bundle_id"></a>base_bundle_id |  The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_dynamic_framework-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID referenced by `base_bundle_id`.   | String | optional |  `""`  |
| <a id="visionos_dynamic_framework-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from the base bundle ID rule referenced by `base_bundle_id`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="visionos_dynamic_framework-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_dynamic_framework-bundle_only"></a>bundle_only |  Avoid linking the dynamic framework, but still include it in the app. This is useful when you want to manually dlopen the framework at runtime.   | Boolean | optional |  `False`  |
| <a id="visionos_dynamic_framework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_dynamic_framework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="visionos_dynamic_framework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_dynamic_framework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_dynamic_framework-extension_safe"></a>extension_safe |  If true, compiles and links this framework with `-application-extension`, restricting the binary to use only extension-safe APIs.   | Boolean | optional |  `False`  |
| <a id="visionos_dynamic_framework-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["vision"]`  |
| <a id="visionos_dynamic_framework-frameworks"></a>frameworks |  A list of framework targets (see [`visionos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-visionos.md#visionos_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_dynamic_framework-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="visionos_dynamic_framework-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_dynamic_framework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="visionos_dynamic_framework-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="visionos_dynamic_framework-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="visionos_dynamic_framework-platform_type"></a>platform_type |  -   | String | optional |  `"visionos"`  |
| <a id="visionos_dynamic_framework-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_dynamic_framework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="visionos_dynamic_framework-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_dynamic_framework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="visionos_framework"></a>

## visionos_framework

<pre>
load("@rules_apple//apple:visionos.doc.bzl", "visionos_framework")

visionos_framework(<a href="#visionos_framework-name">name</a>, <a href="#visionos_framework-deps">deps</a>, <a href="#visionos_framework-resources">resources</a>, <a href="#visionos_framework-hdrs">hdrs</a>, <a href="#visionos_framework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#visionos_framework-base_bundle_id">base_bundle_id</a>, <a href="#visionos_framework-bundle_id">bundle_id</a>,
                   <a href="#visionos_framework-bundle_id_suffix">bundle_id_suffix</a>, <a href="#visionos_framework-bundle_name">bundle_name</a>, <a href="#visionos_framework-bundle_only">bundle_only</a>, <a href="#visionos_framework-codesign_inputs">codesign_inputs</a>, <a href="#visionos_framework-codesignopts">codesignopts</a>,
                   <a href="#visionos_framework-executable_name">executable_name</a>, <a href="#visionos_framework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#visionos_framework-extension_safe">extension_safe</a>, <a href="#visionos_framework-families">families</a>, <a href="#visionos_framework-frameworks">frameworks</a>,
                   <a href="#visionos_framework-infoplists">infoplists</a>, <a href="#visionos_framework-ipa_post_processor">ipa_post_processor</a>, <a href="#visionos_framework-linkopts">linkopts</a>, <a href="#visionos_framework-minimum_deployment_os_version">minimum_deployment_os_version</a>,
                   <a href="#visionos_framework-minimum_os_version">minimum_os_version</a>, <a href="#visionos_framework-platform_type">platform_type</a>, <a href="#visionos_framework-provisioning_profile">provisioning_profile</a>, <a href="#visionos_framework-stamp">stamp</a>, <a href="#visionos_framework-strings">strings</a>, <a href="#visionos_framework-version">version</a>)
</pre>

Builds and bundles a visionos Dynamic Framework.

To use this framework for your app and extensions, list it in the frameworks attributes of those visionos_application and/or visionos_extension rules.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="visionos_framework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="visionos_framework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_framework-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_framework-hdrs"></a>hdrs |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_framework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_framework-base_bundle_id"></a>base_bundle_id |  The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_framework-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID referenced by `base_bundle_id`.   | String | optional |  `""`  |
| <a id="visionos_framework-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from the base bundle ID rule referenced by `base_bundle_id`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="visionos_framework-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_framework-bundle_only"></a>bundle_only |  Avoid linking the dynamic framework, but still include it in the app. This is useful when you want to manually dlopen the framework at runtime.   | Boolean | optional |  `False`  |
| <a id="visionos_framework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_framework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="visionos_framework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_framework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_framework-extension_safe"></a>extension_safe |  If true, compiles and links this framework with `-application-extension`, restricting the binary to use only extension-safe APIs.   | Boolean | optional |  `False`  |
| <a id="visionos_framework-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["vision"]`  |
| <a id="visionos_framework-frameworks"></a>frameworks |  A list of framework targets (see [`visionos_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-visionos.md#visionos_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_framework-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="visionos_framework-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_framework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="visionos_framework-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="visionos_framework-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="visionos_framework-platform_type"></a>platform_type |  -   | String | optional |  `"visionos"`  |
| <a id="visionos_framework-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_framework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="visionos_framework-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_framework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="visionos_static_framework"></a>

## visionos_static_framework

<pre>
load("@rules_apple//apple:visionos.doc.bzl", "visionos_static_framework")

visionos_static_framework(<a href="#visionos_static_framework-name">name</a>, <a href="#visionos_static_framework-deps">deps</a>, <a href="#visionos_static_framework-resources">resources</a>, <a href="#visionos_static_framework-hdrs">hdrs</a>, <a href="#visionos_static_framework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#visionos_static_framework-avoid_deps">avoid_deps</a>,
                          <a href="#visionos_static_framework-bundle_name">bundle_name</a>, <a href="#visionos_static_framework-codesign_inputs">codesign_inputs</a>, <a href="#visionos_static_framework-codesignopts">codesignopts</a>, <a href="#visionos_static_framework-exclude_resources">exclude_resources</a>,
                          <a href="#visionos_static_framework-executable_name">executable_name</a>, <a href="#visionos_static_framework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#visionos_static_framework-families">families</a>, <a href="#visionos_static_framework-ipa_post_processor">ipa_post_processor</a>,
                          <a href="#visionos_static_framework-linkopts">linkopts</a>, <a href="#visionos_static_framework-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#visionos_static_framework-minimum_os_version">minimum_os_version</a>, <a href="#visionos_static_framework-platform_type">platform_type</a>,
                          <a href="#visionos_static_framework-stamp">stamp</a>, <a href="#visionos_static_framework-strings">strings</a>, <a href="#visionos_static_framework-umbrella_header">umbrella_header</a>, <a href="#visionos_static_framework-version">version</a>)
</pre>

Builds and bundles an visionos static framework for third-party distribution.

A static framework is bundled like a dynamic framework except that the embedded
binary is a static library rather than a dynamic library. It is intended to
create distributable static SDKs or artifacts that can be easily imported into
other Xcode projects; it is specifically **not** intended to be used as a
dependency of other Bazel targets. For that use case, use the corresponding
`objc_library` targets directly.

Unlike other visionos bundles, the fat binary in an `visionos_static_framework` may
simultaneously contain simulator and device architectures (that is, you can
build a single framework artifact that works for all architectures by specifying
`--visionos_cpus=sim_arm64,arm64` when you build).

`visionos_static_framework` supports Swift, but there are some constraints:

* `visionos_static_framework` with Swift only works with Xcode 11 and above, since
  the required Swift functionality for module compatibility is available in
  Swift 5.1.
* `visionos_static_framework` only supports a single direct `swift_library` target
  that does not depend transitively on any other `swift_library` targets. The
  Swift compiler expects a framework to contain a single Swift module, and each
  `swift_library` target is its own module by definition.
* `visionos_static_framework` does not support mixed Objective-C and Swift public
  interfaces. This means that the `umbrella_header` and `hdrs` attributes are
  unavailable when using `swift_library` dependencies. You are allowed to depend
  on `objc_library` from the main `swift_library` dependency, but note that only
  the `swift_library`'s public interface will be available to users of the
  static framework.

When using Swift, the `visionos_static_framework` bundles `swiftinterface` and
`swiftdocs` file for each of the required architectures. It also bundles an
umbrella header which is the header generated by the single `swift_library`
target. Finally, it also bundles a `module.modulemap` file pointing to the
umbrella header for Objetive-C module compatibility. This umbrella header and
modulemap can be skipped by disabling the `swift.no_generated_header` feature (
i.e. `--features=-swift.no_generated_header`).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="visionos_static_framework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="visionos_static_framework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_static_framework-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_static_framework-hdrs"></a>hdrs |  A list of `.h` files that will be publicly exposed by this framework. These headers should have framework-relative imports, and if non-empty, an umbrella header named `%{bundle_name}.h` will also be generated that imports all of the headers listed here.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_static_framework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_static_framework-avoid_deps"></a>avoid_deps |  A list of library targets on which this framework depends in order to compile, but the transitive closure of which will not be linked into the framework's binary.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_static_framework-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_static_framework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_static_framework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="visionos_static_framework-exclude_resources"></a>exclude_resources |  Indicates whether resources should be excluded from the bundle. This can be used to avoid unnecessarily bundling resources if the static framework is being distributed in a different fashion, such as a Cocoapod.   | Boolean | optional |  `False`  |
| <a id="visionos_static_framework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_static_framework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_static_framework-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["vision"]`  |
| <a id="visionos_static_framework-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_static_framework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="visionos_static_framework-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="visionos_static_framework-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="visionos_static_framework-platform_type"></a>platform_type |  -   | String | optional |  `"visionos"`  |
| <a id="visionos_static_framework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="visionos_static_framework-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_static_framework-umbrella_header"></a>umbrella_header |  An optional single .h file to use as the umbrella header for this framework. Usually, this header will have the same name as this target, so that clients can load the header using the #import <MyFramework/MyFramework.h> format. If this attribute is not specified (the common use case), an umbrella header will be generated under the same name as this target.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_static_framework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="visionos_ui_test"></a>

## visionos_ui_test

<pre>
load("@rules_apple//apple:visionos.doc.bzl", "visionos_ui_test")

visionos_ui_test(<a href="#visionos_ui_test-name">name</a>, <a href="#visionos_ui_test-deps">deps</a>, <a href="#visionos_ui_test-data">data</a>, <a href="#visionos_ui_test-bundle_name">bundle_name</a>, <a href="#visionos_ui_test-collect_code_coverage">collect_code_coverage</a>, <a href="#visionos_ui_test-env">env</a>, <a href="#visionos_ui_test-env_inherit">env_inherit</a>,
                 <a href="#visionos_ui_test-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#visionos_ui_test-minimum_os_version">minimum_os_version</a>, <a href="#visionos_ui_test-platform_type">platform_type</a>, <a href="#visionos_ui_test-runner">runner</a>,
                 <a href="#visionos_ui_test-test_coverage_manifest">test_coverage_manifest</a>, <a href="#visionos_ui_test-test_filter">test_filter</a>, <a href="#visionos_ui_test-test_host">test_host</a>, <a href="#visionos_ui_test-test_host_is_bundle_loader">test_host_is_bundle_loader</a>)
</pre>

Builds and bundles a visionOS UI `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

Note: visionOS UI tests are not currently supported in the default test runner.

The following is a list of the `visionos_ui_test` specific attributes; for a list of
the attributes inherited by all test rules, please check the
[Bazel documentation](https://bazel.build/reference/be/common-definitions#common-attributes-tests).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="visionos_ui_test-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="visionos_ui_test-deps"></a>deps |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="visionos_ui_test-data"></a>data |  Files to be made available to the test during its execution.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_ui_test-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_ui_test-collect_code_coverage"></a>collect_code_coverage |  Whether to collect code coverage for this test if `--collect_code_coverage=yes`.   | Boolean | optional |  `True`  |
| <a id="visionos_ui_test-env"></a>env |  Dictionary of environment variables that should be set during the test execution. The values of the dictionary are subject to "Make" variable expansion.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="visionos_ui_test-env_inherit"></a>env_inherit |  List of environment variables to inherit from the external environment.   | List of strings | optional |  `[]`  |
| <a id="visionos_ui_test-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="visionos_ui_test-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="visionos_ui_test-platform_type"></a>platform_type |  -   | String | optional |  `"visionos"`  |
| <a id="visionos_ui_test-runner"></a>runner |  The runner target that will provide the logic on how to run the tests. Needs to provide the AppleTestRunnerInfo provider.   | <a href="https://bazel.build/concepts/labels">Label</a> | required |  |
| <a id="visionos_ui_test-test_coverage_manifest"></a>test_coverage_manifest |  A file that will be used in lcov export calls to limit the scope of files instrumented with coverage.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_ui_test-test_filter"></a>test_filter |  Test filter string that will be passed into the test runner to select which tests will run.   | String | optional |  `""`  |
| <a id="visionos_ui_test-test_host"></a>test_host |  -   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_ui_test-test_host_is_bundle_loader"></a>test_host_is_bundle_loader |  Whether the 'test_host' should be used as the -bundle_loader to allow testing the symbols from the test host app   | Boolean | optional |  `True`  |


<a id="visionos_unit_test"></a>

## visionos_unit_test

<pre>
load("@rules_apple//apple:visionos.doc.bzl", "visionos_unit_test")

visionos_unit_test(<a href="#visionos_unit_test-name">name</a>, <a href="#visionos_unit_test-deps">deps</a>, <a href="#visionos_unit_test-data">data</a>, <a href="#visionos_unit_test-bundle_name">bundle_name</a>, <a href="#visionos_unit_test-collect_code_coverage">collect_code_coverage</a>, <a href="#visionos_unit_test-env">env</a>, <a href="#visionos_unit_test-env_inherit">env_inherit</a>,
                   <a href="#visionos_unit_test-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#visionos_unit_test-minimum_os_version">minimum_os_version</a>, <a href="#visionos_unit_test-platform_type">platform_type</a>, <a href="#visionos_unit_test-runner">runner</a>,
                   <a href="#visionos_unit_test-test_coverage_manifest">test_coverage_manifest</a>, <a href="#visionos_unit_test-test_filter">test_filter</a>, <a href="#visionos_unit_test-test_host">test_host</a>, <a href="#visionos_unit_test-test_host_is_bundle_loader">test_host_is_bundle_loader</a>)
</pre>

Builds and bundles a visionOS Unit `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

Note: visionOS unit tests are not currently supported in the default test runner.

`visionos_unit_test` targets can work in two modes: as app or library tests. If the
`test_host` attribute is set to an `visionos_application` target, the tests will run
within that application's context. If no `test_host` is provided, the tests will
run outside the context of a visionOS application. Because of this, certain
functionalities might not be present (e.g. UI layout, NSUserDefaults). You can
find more information about app and library testing for Apple platforms
[here](https://developer.apple.com/library/content/documentation/DeveloperTools/Conceptual/testing_with_xcode/chapters/03-testing_basics.html).

The following is a list of the `visionos_unit_test` specific attributes; for a list
of the attributes inherited by all test rules, please check the
[Bazel documentation](https://bazel.build/reference/be/common-definitions#common-attributes-tests).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="visionos_unit_test-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="visionos_unit_test-deps"></a>deps |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="visionos_unit_test-data"></a>data |  Files to be made available to the test during its execution.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="visionos_unit_test-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="visionos_unit_test-collect_code_coverage"></a>collect_code_coverage |  Whether to collect code coverage for this test if `--collect_code_coverage=yes`.   | Boolean | optional |  `True`  |
| <a id="visionos_unit_test-env"></a>env |  Dictionary of environment variables that should be set during the test execution. The values of the dictionary are subject to "Make" variable expansion.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="visionos_unit_test-env_inherit"></a>env_inherit |  List of environment variables to inherit from the external environment.   | List of strings | optional |  `[]`  |
| <a id="visionos_unit_test-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="visionos_unit_test-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="visionos_unit_test-platform_type"></a>platform_type |  -   | String | optional |  `"visionos"`  |
| <a id="visionos_unit_test-runner"></a>runner |  The runner target that will provide the logic on how to run the tests. Needs to provide the AppleTestRunnerInfo provider.   | <a href="https://bazel.build/concepts/labels">Label</a> | required |  |
| <a id="visionos_unit_test-test_coverage_manifest"></a>test_coverage_manifest |  A file that will be used in lcov export calls to limit the scope of files instrumented with coverage.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_unit_test-test_filter"></a>test_filter |  Test filter string that will be passed into the test runner to select which tests will run.   | String | optional |  `""`  |
| <a id="visionos_unit_test-test_host"></a>test_host |  -   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="visionos_unit_test-test_host_is_bundle_loader"></a>test_host_is_bundle_loader |  Whether the 'test_host' should be used as the -bundle_loader to allow testing the symbols from the test host app   | Boolean | optional |  `True`  |


