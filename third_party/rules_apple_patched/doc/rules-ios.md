<!-- Generated with Stardoc: http://skydoc.bazel.build -->

# Build rules for iOS

<a id="ios_app_clip"></a>

## ios_app_clip

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_app_clip")

ios_app_clip(<a href="#ios_app_clip-name">name</a>, <a href="#ios_app_clip-deps">deps</a>, <a href="#ios_app_clip-resources">resources</a>, <a href="#ios_app_clip-additional_linker_inputs">additional_linker_inputs</a>, <a href="#ios_app_clip-app_icons">app_icons</a>, <a href="#ios_app_clip-bundle_id">bundle_id</a>,
             <a href="#ios_app_clip-bundle_id_suffix">bundle_id_suffix</a>, <a href="#ios_app_clip-bundle_name">bundle_name</a>, <a href="#ios_app_clip-codesign_inputs">codesign_inputs</a>, <a href="#ios_app_clip-codesignopts">codesignopts</a>, <a href="#ios_app_clip-entitlements">entitlements</a>,
             <a href="#ios_app_clip-entitlements_validation">entitlements_validation</a>, <a href="#ios_app_clip-executable_name">executable_name</a>, <a href="#ios_app_clip-exported_symbols_lists">exported_symbols_lists</a>, <a href="#ios_app_clip-extensions">extensions</a>, <a href="#ios_app_clip-families">families</a>,
             <a href="#ios_app_clip-frameworks">frameworks</a>, <a href="#ios_app_clip-infoplists">infoplists</a>, <a href="#ios_app_clip-ipa_post_processor">ipa_post_processor</a>, <a href="#ios_app_clip-launch_storyboard">launch_storyboard</a>, <a href="#ios_app_clip-linkopts">linkopts</a>,
             <a href="#ios_app_clip-locales_to_include">locales_to_include</a>, <a href="#ios_app_clip-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#ios_app_clip-minimum_os_version">minimum_os_version</a>, <a href="#ios_app_clip-platform_type">platform_type</a>,
             <a href="#ios_app_clip-provisioning_profile">provisioning_profile</a>, <a href="#ios_app_clip-shared_capabilities">shared_capabilities</a>, <a href="#ios_app_clip-stamp">stamp</a>, <a href="#ios_app_clip-strings">strings</a>, <a href="#ios_app_clip-version">version</a>)
</pre>

Builds and bundles an iOS App Clip.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_app_clip-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_app_clip-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcassets/*..appiconset` and there may be only one such `..appiconset` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcassets` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="ios_app_clip-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="ios_app_clip-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_app_clip-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="ios_app_clip-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_app_clip-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="ios_app_clip-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_app_clip-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-extensions"></a>extensions |  A list of ios_extension live activity extensions to include in the final app clip bundle. It is only possible to embed live activity WidgetKit extensions. Visit Apple developer documentation page for more info https://developer.apple.com/documentation/appclip/offering-live-activities-with-your-app-clip.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | required |  |
| <a id="ios_app_clip-frameworks"></a>frameworks |  A list of framework targets (see [`ios_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-ios.md#ios_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_app_clip-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_app_clip-launch_storyboard"></a>launch_storyboard |  The `.storyboard` or `.xib` file that should be used as the launch screen for the app clip. The provided file will be compiled into the appropriate format (`.storyboardc` or `.nib`) and placed in the root of the final bundle. The generated file will also be registered in the bundle's Info.plist under the key `UILaunchStoryboardName`.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_app_clip-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="ios_app_clip-locales_to_include"></a>locales_to_include |  A list of locales to include in the bundle. Only *.lproj directories that are matched will be copied as a part of the build. This value takes precedence (and is preferred) over locales defined using `--define "apple.locales_to_include=..."`.   | List of strings | optional |  `[]`  |
| <a id="ios_app_clip-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_app_clip-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_app_clip-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_app_clip-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_app_clip-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="ios_app_clip-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_app_clip-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="ios_application"></a>

## ios_application

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_application")

ios_application(<a href="#ios_application-name">name</a>, <a href="#ios_application-deps">deps</a>, <a href="#ios_application-resources">resources</a>, <a href="#ios_application-additional_linker_inputs">additional_linker_inputs</a>, <a href="#ios_application-alternate_icons">alternate_icons</a>, <a href="#ios_application-app_clips">app_clips</a>,
                <a href="#ios_application-app_icons">app_icons</a>, <a href="#ios_application-app_intents">app_intents</a>, <a href="#ios_application-bundle_id">bundle_id</a>, <a href="#ios_application-bundle_id_suffix">bundle_id_suffix</a>, <a href="#ios_application-bundle_name">bundle_name</a>, <a href="#ios_application-codesign_inputs">codesign_inputs</a>,
                <a href="#ios_application-codesignopts">codesignopts</a>, <a href="#ios_application-entitlements">entitlements</a>, <a href="#ios_application-entitlements_validation">entitlements_validation</a>, <a href="#ios_application-executable_name">executable_name</a>,
                <a href="#ios_application-exported_symbols_lists">exported_symbols_lists</a>, <a href="#ios_application-extensions">extensions</a>, <a href="#ios_application-families">families</a>, <a href="#ios_application-frameworks">frameworks</a>, <a href="#ios_application-include_symbols_in_bundle">include_symbols_in_bundle</a>,
                <a href="#ios_application-infoplists">infoplists</a>, <a href="#ios_application-ipa_post_processor">ipa_post_processor</a>, <a href="#ios_application-launch_images">launch_images</a>, <a href="#ios_application-launch_storyboard">launch_storyboard</a>, <a href="#ios_application-linkopts">linkopts</a>,
                <a href="#ios_application-locales_to_include">locales_to_include</a>, <a href="#ios_application-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#ios_application-minimum_os_version">minimum_os_version</a>, <a href="#ios_application-platform_type">platform_type</a>,
                <a href="#ios_application-primary_app_icon">primary_app_icon</a>, <a href="#ios_application-provisioning_profile">provisioning_profile</a>, <a href="#ios_application-sdk_frameworks">sdk_frameworks</a>, <a href="#ios_application-settings_bundle">settings_bundle</a>,
                <a href="#ios_application-shared_capabilities">shared_capabilities</a>, <a href="#ios_application-stamp">stamp</a>, <a href="#ios_application-strings">strings</a>, <a href="#ios_application-version">version</a>, <a href="#ios_application-watch_application">watch_application</a>)
</pre>

Builds and bundles an iOS Application.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_application-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_application-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-alternate_icons"></a>alternate_icons |  Files that comprise the alternate app icons for the application. Each file must have a containing directory named after the alternate icon identifier.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-app_clips"></a>app_clips |  A list of iOS app clips to include in the final application bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcassets/*..appiconset` and there may be only one such `..appiconset` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcassets` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-app_intents"></a>app_intents |  List of dependencies implementing the AppIntents protocol.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="ios_application-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"_"`  |
| <a id="ios_application-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_application-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="ios_application-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_application-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="ios_application-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_application-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-extensions"></a>extensions |  A list of iOS application extensions to include in the final application bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | required |  |
| <a id="ios_application-frameworks"></a>frameworks |  A list of framework targets (see [`ios_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-ios.md#ios_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-include_symbols_in_bundle"></a>include_symbols_in_bundle |  If true and --output_groups=+dsyms is specified, generates `$UUID.symbols` files from all `{binary: .dSYM, ...}` pairs for the application and its dependencies, then packages them under the `Symbols/` directory in the final application bundle.   | Boolean | optional |  `False`  |
| <a id="ios_application-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_application-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_application-launch_images"></a>launch_images |  Files that comprise the launch images for the application. Each file must have a containing directory named `*.xcassets/*.launchimage` and there may be only one such `.launchimage` directory in the list.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-launch_storyboard"></a>launch_storyboard |  The `.storyboard` or `.xib` file that should be used as the launch screen for the application. The provided file will be compiled into the appropriate format (`.storyboardc` or `.nib`) and placed in the root of the final bundle. The generated file will also be registered in the bundle's Info.plist under the key `UILaunchStoryboardName`.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_application-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="ios_application-locales_to_include"></a>locales_to_include |  A list of locales to include in the bundle. Only *.lproj directories that are matched will be copied as a part of the build. This value takes precedence (and is preferred) over locales defined using `--define "apple.locales_to_include=..."`.   | List of strings | optional |  `[]`  |
| <a id="ios_application-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_application-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_application-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_application-primary_app_icon"></a>primary_app_icon |  An optional String to identify the name of the primary app icon when alternate app icons have been provided for the app. This should match both the `*.icon` bundle in iOS/macOS/watchOS 26+ and the `*..xcassets/.appiconset` bundle's AppIcon resource in previous versions of iOS/macOS/watchOS.   | String | optional |  `""`  |
| <a id="ios_application-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_application-sdk_frameworks"></a>sdk_frameworks |  Names of SDK frameworks to link with (e.g., `AddressBook`, `QuartzCore`). `UIKit` and `Foundation` are always included, even if this attribute is provided and does not list them.<br><br>This attribute is discouraged; in general, targets should list system framework dependencies in the library targets where that framework is used, not in the top-level bundle.   | List of strings | optional |  `[]`  |
| <a id="ios_application-settings_bundle"></a>settings_bundle |  A resource bundle (e.g. `apple_bundle_import`) target that contains the files that make up the application's settings bundle. These files will be copied into the root of the final application bundle in a directory named `Settings.bundle`.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_application-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="ios_application-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_application-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_application-watch_application"></a>watch_application |  A `watchos_application` target that represents an Apple Watch application or a `watchos_single_target_application` target that represents a single-target Apple Watch application that should be embedded in the application bundle.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="ios_build_test"></a>

## ios_build_test

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_build_test")

ios_build_test(<a href="#ios_build_test-name">name</a>, <a href="#ios_build_test-minimum_os_version">minimum_os_version</a>, <a href="#ios_build_test-platform_type">platform_type</a>, <a href="#ios_build_test-targets">targets</a>)
</pre>

Test rule to check that the given library targets (Swift, Objective-C, C++)
build for iOS.

Typical usage:

```starlark
ios_build_test(
    name = "my_build_test",
    minimum_os_version = "12.0",
    targets = [
        "//some/package:my_library",
    ],
)
```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_build_test-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_build_test-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version that will be used as the deployment target when building the targets, represented as a dotted version number (for example, `"9.0"`).   | String | required |  |
| <a id="ios_build_test-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_build_test-targets"></a>targets |  The targets to check for successful build.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |


<a id="ios_dynamic_framework"></a>

## ios_dynamic_framework

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_dynamic_framework")

ios_dynamic_framework(<a href="#ios_dynamic_framework-name">name</a>, <a href="#ios_dynamic_framework-deps">deps</a>, <a href="#ios_dynamic_framework-resources">resources</a>, <a href="#ios_dynamic_framework-hdrs">hdrs</a>, <a href="#ios_dynamic_framework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#ios_dynamic_framework-base_bundle_id">base_bundle_id</a>,
                      <a href="#ios_dynamic_framework-bundle_id">bundle_id</a>, <a href="#ios_dynamic_framework-bundle_id_suffix">bundle_id_suffix</a>, <a href="#ios_dynamic_framework-bundle_name">bundle_name</a>, <a href="#ios_dynamic_framework-bundle_only">bundle_only</a>, <a href="#ios_dynamic_framework-codesign_inputs">codesign_inputs</a>,
                      <a href="#ios_dynamic_framework-codesignopts">codesignopts</a>, <a href="#ios_dynamic_framework-executable_name">executable_name</a>, <a href="#ios_dynamic_framework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#ios_dynamic_framework-extension_safe">extension_safe</a>, <a href="#ios_dynamic_framework-families">families</a>,
                      <a href="#ios_dynamic_framework-frameworks">frameworks</a>, <a href="#ios_dynamic_framework-infoplists">infoplists</a>, <a href="#ios_dynamic_framework-ipa_post_processor">ipa_post_processor</a>, <a href="#ios_dynamic_framework-linkopts">linkopts</a>,
                      <a href="#ios_dynamic_framework-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#ios_dynamic_framework-minimum_os_version">minimum_os_version</a>, <a href="#ios_dynamic_framework-platform_type">platform_type</a>,
                      <a href="#ios_dynamic_framework-provisioning_profile">provisioning_profile</a>, <a href="#ios_dynamic_framework-stamp">stamp</a>, <a href="#ios_dynamic_framework-strings">strings</a>, <a href="#ios_dynamic_framework-version">version</a>)
</pre>

Builds and bundles an iOS dynamic framework that is consumable by Xcode.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_dynamic_framework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_dynamic_framework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_dynamic_framework-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_dynamic_framework-hdrs"></a>hdrs |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_dynamic_framework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_dynamic_framework-base_bundle_id"></a>base_bundle_id |  The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_dynamic_framework-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID referenced by `base_bundle_id`.   | String | optional |  `""`  |
| <a id="ios_dynamic_framework-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from the base bundle ID rule referenced by `base_bundle_id`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="ios_dynamic_framework-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_dynamic_framework-bundle_only"></a>bundle_only |  Avoid linking the dynamic framework, but still include it in the app. This is useful when you want to manually dlopen the framework at runtime.   | Boolean | optional |  `False`  |
| <a id="ios_dynamic_framework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_dynamic_framework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="ios_dynamic_framework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_dynamic_framework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_dynamic_framework-extension_safe"></a>extension_safe |  If true, compiles and links this framework with `-application-extension`, restricting the binary to use only extension-safe APIs.   | Boolean | optional |  `False`  |
| <a id="ios_dynamic_framework-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["iphone", "ipad"]`  |
| <a id="ios_dynamic_framework-frameworks"></a>frameworks |  A list of framework targets (see [`ios_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-ios.md#ios_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_dynamic_framework-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_dynamic_framework-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_dynamic_framework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="ios_dynamic_framework-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_dynamic_framework-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_dynamic_framework-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_dynamic_framework-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_dynamic_framework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="ios_dynamic_framework-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_dynamic_framework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="ios_extension"></a>

## ios_extension

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_extension")

ios_extension(<a href="#ios_extension-name">name</a>, <a href="#ios_extension-deps">deps</a>, <a href="#ios_extension-resources">resources</a>, <a href="#ios_extension-additional_linker_inputs">additional_linker_inputs</a>, <a href="#ios_extension-app_icons">app_icons</a>, <a href="#ios_extension-app_intents">app_intents</a>, <a href="#ios_extension-bundle_id">bundle_id</a>,
              <a href="#ios_extension-bundle_id_suffix">bundle_id_suffix</a>, <a href="#ios_extension-bundle_name">bundle_name</a>, <a href="#ios_extension-codesign_inputs">codesign_inputs</a>, <a href="#ios_extension-codesignopts">codesignopts</a>, <a href="#ios_extension-entitlements">entitlements</a>,
              <a href="#ios_extension-entitlements_validation">entitlements_validation</a>, <a href="#ios_extension-executable_name">executable_name</a>, <a href="#ios_extension-exported_symbols_lists">exported_symbols_lists</a>,
              <a href="#ios_extension-extensionkit_extension">extensionkit_extension</a>, <a href="#ios_extension-families">families</a>, <a href="#ios_extension-frameworks">frameworks</a>, <a href="#ios_extension-infoplists">infoplists</a>, <a href="#ios_extension-ipa_post_processor">ipa_post_processor</a>, <a href="#ios_extension-linkopts">linkopts</a>,
              <a href="#ios_extension-locales_to_include">locales_to_include</a>, <a href="#ios_extension-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#ios_extension-minimum_os_version">minimum_os_version</a>, <a href="#ios_extension-platform_type">platform_type</a>,
              <a href="#ios_extension-provisioning_profile">provisioning_profile</a>, <a href="#ios_extension-sdk_frameworks">sdk_frameworks</a>, <a href="#ios_extension-shared_capabilities">shared_capabilities</a>, <a href="#ios_extension-stamp">stamp</a>, <a href="#ios_extension-strings">strings</a>, <a href="#ios_extension-version">version</a>)
</pre>

Builds and bundles an iOS Application Extension.

Most iOS app extensions use a plug-in-based architecture where the executable's entry point
is provided by a system framework.
However, iOS 14 introduced Widget Extensions that use a traditional `main` entry point
(typically expressed through Swift's `@main` attribute).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_extension-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_extension-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcassets/*..appiconset` and there may be only one such `..appiconset` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcassets` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-app_intents"></a>app_intents |  List of dependencies implementing the AppIntents protocol.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="ios_extension-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="ios_extension-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_extension-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="ios_extension-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_extension-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="ios_extension-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_extension-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-extensionkit_extension"></a>extensionkit_extension |  Indicates if this target should be treated as an ExtensionKit extension.   | Boolean | optional |  `False`  |
| <a id="ios_extension-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | required |  |
| <a id="ios_extension-frameworks"></a>frameworks |  A list of framework targets (see [`ios_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-ios.md#ios_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_extension-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_extension-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="ios_extension-locales_to_include"></a>locales_to_include |  A list of locales to include in the bundle. Only *.lproj directories that are matched will be copied as a part of the build. This value takes precedence (and is preferred) over locales defined using `--define "apple.locales_to_include=..."`.   | List of strings | optional |  `[]`  |
| <a id="ios_extension-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_extension-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_extension-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_extension-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_extension-sdk_frameworks"></a>sdk_frameworks |  Names of SDK frameworks to link with (e.g., `AddressBook`, `QuartzCore`). `UIKit` and `Foundation` are always included, even if this attribute is provided and does not list them.<br><br>This attribute is discouraged; in general, targets should list system framework dependencies in the library targets where that framework is used, not in the top-level bundle.   | List of strings | optional |  `[]`  |
| <a id="ios_extension-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="ios_extension-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_extension-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="ios_framework"></a>

## ios_framework

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_framework")

ios_framework(<a href="#ios_framework-name">name</a>, <a href="#ios_framework-deps">deps</a>, <a href="#ios_framework-resources">resources</a>, <a href="#ios_framework-hdrs">hdrs</a>, <a href="#ios_framework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#ios_framework-base_bundle_id">base_bundle_id</a>, <a href="#ios_framework-bundle_id">bundle_id</a>,
              <a href="#ios_framework-bundle_id_suffix">bundle_id_suffix</a>, <a href="#ios_framework-bundle_name">bundle_name</a>, <a href="#ios_framework-bundle_only">bundle_only</a>, <a href="#ios_framework-codesign_inputs">codesign_inputs</a>, <a href="#ios_framework-codesignopts">codesignopts</a>,
              <a href="#ios_framework-executable_name">executable_name</a>, <a href="#ios_framework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#ios_framework-extension_safe">extension_safe</a>, <a href="#ios_framework-families">families</a>, <a href="#ios_framework-frameworks">frameworks</a>,
              <a href="#ios_framework-infoplists">infoplists</a>, <a href="#ios_framework-ipa_post_processor">ipa_post_processor</a>, <a href="#ios_framework-linkopts">linkopts</a>, <a href="#ios_framework-minimum_deployment_os_version">minimum_deployment_os_version</a>,
              <a href="#ios_framework-minimum_os_version">minimum_os_version</a>, <a href="#ios_framework-platform_type">platform_type</a>, <a href="#ios_framework-provisioning_profile">provisioning_profile</a>, <a href="#ios_framework-stamp">stamp</a>, <a href="#ios_framework-strings">strings</a>, <a href="#ios_framework-version">version</a>)
</pre>

Builds and bundles an iOS Dynamic Framework.

To use this framework for your app and extensions, list it in the `frameworks` attributes
of those `ios_application` and/or `ios_extension` rules.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_framework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_framework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_framework-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_framework-hdrs"></a>hdrs |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_framework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_framework-base_bundle_id"></a>base_bundle_id |  The base bundle ID rule to dictate the form that a given bundle rule's bundle ID prefix should take.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_framework-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID referenced by `base_bundle_id`.   | String | optional |  `""`  |
| <a id="ios_framework-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from the base bundle ID rule referenced by `base_bundle_id`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="ios_framework-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_framework-bundle_only"></a>bundle_only |  Avoid linking the dynamic framework, but still include it in the app. This is useful when you want to manually dlopen the framework at runtime.   | Boolean | optional |  `False`  |
| <a id="ios_framework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_framework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="ios_framework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_framework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_framework-extension_safe"></a>extension_safe |  If true, compiles and links this framework with `-application-extension`, restricting the binary to use only extension-safe APIs.   | Boolean | optional |  `False`  |
| <a id="ios_framework-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | required |  |
| <a id="ios_framework-frameworks"></a>frameworks |  A list of framework targets (see [`ios_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-ios.md#ios_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_framework-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_framework-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_framework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="ios_framework-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_framework-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_framework-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_framework-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_framework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="ios_framework-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_framework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="ios_imessage_application"></a>

## ios_imessage_application

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_imessage_application")

ios_imessage_application(<a href="#ios_imessage_application-name">name</a>, <a href="#ios_imessage_application-resources">resources</a>, <a href="#ios_imessage_application-app_icons">app_icons</a>, <a href="#ios_imessage_application-bundle_id">bundle_id</a>, <a href="#ios_imessage_application-bundle_id_suffix">bundle_id_suffix</a>, <a href="#ios_imessage_application-bundle_name">bundle_name</a>,
                         <a href="#ios_imessage_application-entitlements">entitlements</a>, <a href="#ios_imessage_application-entitlements_validation">entitlements_validation</a>, <a href="#ios_imessage_application-executable_name">executable_name</a>, <a href="#ios_imessage_application-extension">extension</a>, <a href="#ios_imessage_application-families">families</a>,
                         <a href="#ios_imessage_application-infoplists">infoplists</a>, <a href="#ios_imessage_application-ipa_post_processor">ipa_post_processor</a>, <a href="#ios_imessage_application-locales_to_include">locales_to_include</a>,
                         <a href="#ios_imessage_application-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#ios_imessage_application-minimum_os_version">minimum_os_version</a>, <a href="#ios_imessage_application-platform_type">platform_type</a>,
                         <a href="#ios_imessage_application-provisioning_profile">provisioning_profile</a>, <a href="#ios_imessage_application-shared_capabilities">shared_capabilities</a>, <a href="#ios_imessage_application-strings">strings</a>, <a href="#ios_imessage_application-version">version</a>)
</pre>

Builds and bundles an iOS iMessage Application.

iOS iMessage applications do not have any dependencies, as it works mostly as a wrapper
for either an iOS iMessage extension or a Sticker Pack extension.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_imessage_application-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_imessage_application-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_application-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcassets/*..appiconset` and there may be only one such `..appiconset` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcassets` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_application-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="ios_imessage_application-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"_"`  |
| <a id="ios_imessage_application-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_imessage_application-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_imessage_application-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="ios_imessage_application-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_imessage_application-extension"></a>extension |  Single label referencing either an ios_imessage_extension or ios_sticker_pack_extension target. Required.   | <a href="https://bazel.build/concepts/labels">Label</a> | required |  |
| <a id="ios_imessage_application-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | required |  |
| <a id="ios_imessage_application-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_imessage_application-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_imessage_application-locales_to_include"></a>locales_to_include |  A list of locales to include in the bundle. Only *.lproj directories that are matched will be copied as a part of the build. This value takes precedence (and is preferred) over locales defined using `--define "apple.locales_to_include=..."`.   | List of strings | optional |  `[]`  |
| <a id="ios_imessage_application-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_imessage_application-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_imessage_application-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_imessage_application-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_imessage_application-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_application-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_application-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="ios_imessage_extension"></a>

## ios_imessage_extension

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_imessage_extension")

ios_imessage_extension(<a href="#ios_imessage_extension-name">name</a>, <a href="#ios_imessage_extension-deps">deps</a>, <a href="#ios_imessage_extension-resources">resources</a>, <a href="#ios_imessage_extension-additional_linker_inputs">additional_linker_inputs</a>, <a href="#ios_imessage_extension-app_icons">app_icons</a>, <a href="#ios_imessage_extension-bundle_id">bundle_id</a>,
                       <a href="#ios_imessage_extension-bundle_id_suffix">bundle_id_suffix</a>, <a href="#ios_imessage_extension-bundle_name">bundle_name</a>, <a href="#ios_imessage_extension-codesign_inputs">codesign_inputs</a>, <a href="#ios_imessage_extension-codesignopts">codesignopts</a>, <a href="#ios_imessage_extension-entitlements">entitlements</a>,
                       <a href="#ios_imessage_extension-entitlements_validation">entitlements_validation</a>, <a href="#ios_imessage_extension-executable_name">executable_name</a>, <a href="#ios_imessage_extension-exported_symbols_lists">exported_symbols_lists</a>, <a href="#ios_imessage_extension-families">families</a>,
                       <a href="#ios_imessage_extension-frameworks">frameworks</a>, <a href="#ios_imessage_extension-infoplists">infoplists</a>, <a href="#ios_imessage_extension-ipa_post_processor">ipa_post_processor</a>, <a href="#ios_imessage_extension-linkopts">linkopts</a>, <a href="#ios_imessage_extension-locales_to_include">locales_to_include</a>,
                       <a href="#ios_imessage_extension-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#ios_imessage_extension-minimum_os_version">minimum_os_version</a>, <a href="#ios_imessage_extension-platform_type">platform_type</a>,
                       <a href="#ios_imessage_extension-provisioning_profile">provisioning_profile</a>, <a href="#ios_imessage_extension-shared_capabilities">shared_capabilities</a>, <a href="#ios_imessage_extension-stamp">stamp</a>, <a href="#ios_imessage_extension-strings">strings</a>, <a href="#ios_imessage_extension-version">version</a>)
</pre>

Builds and bundles an iOS iMessage Extension.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_imessage_extension-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_imessage_extension-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_extension-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_extension-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_extension-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcassets/*..appiconset` and there may be only one such `..appiconset` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcassets` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_extension-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="ios_imessage_extension-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="ios_imessage_extension-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_imessage_extension-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_extension-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="ios_imessage_extension-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_imessage_extension-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="ios_imessage_extension-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_imessage_extension-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_extension-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | required |  |
| <a id="ios_imessage_extension-frameworks"></a>frameworks |  A list of framework targets (see [`ios_framework`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-ios.md#ios_framework)) that this target depends on.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_extension-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_imessage_extension-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_imessage_extension-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="ios_imessage_extension-locales_to_include"></a>locales_to_include |  A list of locales to include in the bundle. Only *.lproj directories that are matched will be copied as a part of the build. This value takes precedence (and is preferred) over locales defined using `--define "apple.locales_to_include=..."`.   | List of strings | optional |  `[]`  |
| <a id="ios_imessage_extension-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_imessage_extension-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_imessage_extension-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_imessage_extension-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_imessage_extension-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_extension-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="ios_imessage_extension-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_imessage_extension-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="ios_static_framework"></a>

## ios_static_framework

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_static_framework")

ios_static_framework(<a href="#ios_static_framework-name">name</a>, <a href="#ios_static_framework-deps">deps</a>, <a href="#ios_static_framework-resources">resources</a>, <a href="#ios_static_framework-hdrs">hdrs</a>, <a href="#ios_static_framework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#ios_static_framework-avoid_deps">avoid_deps</a>, <a href="#ios_static_framework-bundle_name">bundle_name</a>,
                     <a href="#ios_static_framework-codesign_inputs">codesign_inputs</a>, <a href="#ios_static_framework-codesignopts">codesignopts</a>, <a href="#ios_static_framework-exclude_resources">exclude_resources</a>, <a href="#ios_static_framework-executable_name">executable_name</a>,
                     <a href="#ios_static_framework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#ios_static_framework-families">families</a>, <a href="#ios_static_framework-ipa_post_processor">ipa_post_processor</a>, <a href="#ios_static_framework-linkopts">linkopts</a>,
                     <a href="#ios_static_framework-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#ios_static_framework-minimum_os_version">minimum_os_version</a>, <a href="#ios_static_framework-platform_type">platform_type</a>, <a href="#ios_static_framework-stamp">stamp</a>, <a href="#ios_static_framework-strings">strings</a>,
                     <a href="#ios_static_framework-umbrella_header">umbrella_header</a>, <a href="#ios_static_framework-version">version</a>)
</pre>

Builds and bundles an iOS static framework for third-party distribution.

A static framework is bundled like a dynamic framework except that the embedded
binary is a static library rather than a dynamic library. It is intended to
create distributable static SDKs or artifacts that can be easily imported into
other Xcode projects; it is specifically **not** intended to be used as a
dependency of other Bazel targets. For that use case, use the corresponding
`objc_library` targets directly.

Unlike other iOS bundles, the fat binary in an `ios_static_framework` may
simultaneously contain simulator and device architectures (that is, you can
build a single framework artifact that works for all architectures by specifying
`--ios_multi_cpus=i386,x86_64,armv7,arm64` when you build).

`ios_static_framework` supports Swift, but there are some constraints:

* `ios_static_framework` with Swift only works with Xcode 11 and above, since
  the required Swift functionality for module compatibility is available in
  Swift 5.1.
* `ios_static_framework` only supports a single direct `swift_library` target
  that does not depend transitively on any other `swift_library` targets. The
  Swift compiler expects a framework to contain a single Swift module, and each
  `swift_library` target is its own module by definition.
* `ios_static_framework` does not support mixed Objective-C and Swift public
  interfaces. This means that the `umbrella_header` and `hdrs` attributes are
  unavailable when using `swift_library` dependencies. You are allowed to depend
  on `objc_library` from the main `swift_library` dependency, but note that only
  the `swift_library`'s public interface will be available to users of the
  static framework.

When using Swift, the `ios_static_framework` bundles `swiftinterface` and
`swiftdocs` file for each of the required architectures. It also bundles an
umbrella header which is the header generated by the single `swift_library`
target. Finally, it also bundles a `module.modulemap` file pointing to the
umbrella header for Objetive-C module compatibility. This umbrella header and
modulemap can be skipped by disabling the `swift.no_generated_header` feature (
i.e. `--features=-swift.no_generated_header`).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_static_framework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_static_framework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_static_framework-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_static_framework-hdrs"></a>hdrs |  A list of `.h` files that will be publicly exposed by this framework. These headers should have framework-relative imports, and if non-empty, an umbrella header named `%{bundle_name}.h` will also be generated that imports all of the headers listed here.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_static_framework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_static_framework-avoid_deps"></a>avoid_deps |  A list of library targets on which this framework depends in order to compile, but the transitive closure of which will not be linked into the framework's binary.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_static_framework-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_static_framework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_static_framework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="ios_static_framework-exclude_resources"></a>exclude_resources |  Indicates whether resources should be excluded from the bundle. This can be used to avoid unnecessarily bundling resources if the static framework is being distributed in a different fashion, such as a Cocoapod.   | Boolean | optional |  `False`  |
| <a id="ios_static_framework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_static_framework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_static_framework-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | optional |  `["iphone", "ipad"]`  |
| <a id="ios_static_framework-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_static_framework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="ios_static_framework-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_static_framework-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_static_framework-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_static_framework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="ios_static_framework-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_static_framework-umbrella_header"></a>umbrella_header |  An optional single .h file to use as the umbrella header for this framework. Usually, this header will have the same name as this target, so that clients can load the header using the #import <MyFramework/MyFramework.h> format. If this attribute is not specified (the common use case), an umbrella header will be generated under the same name as this target.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_static_framework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="ios_sticker_pack_extension"></a>

## ios_sticker_pack_extension

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_sticker_pack_extension")

ios_sticker_pack_extension(<a href="#ios_sticker_pack_extension-name">name</a>, <a href="#ios_sticker_pack_extension-resources">resources</a>, <a href="#ios_sticker_pack_extension-app_icons">app_icons</a>, <a href="#ios_sticker_pack_extension-bundle_id">bundle_id</a>, <a href="#ios_sticker_pack_extension-bundle_id_suffix">bundle_id_suffix</a>, <a href="#ios_sticker_pack_extension-bundle_name">bundle_name</a>,
                           <a href="#ios_sticker_pack_extension-entitlements">entitlements</a>, <a href="#ios_sticker_pack_extension-entitlements_validation">entitlements_validation</a>, <a href="#ios_sticker_pack_extension-executable_name">executable_name</a>, <a href="#ios_sticker_pack_extension-families">families</a>,
                           <a href="#ios_sticker_pack_extension-infoplists">infoplists</a>, <a href="#ios_sticker_pack_extension-ipa_post_processor">ipa_post_processor</a>, <a href="#ios_sticker_pack_extension-minimum_deployment_os_version">minimum_deployment_os_version</a>,
                           <a href="#ios_sticker_pack_extension-minimum_os_version">minimum_os_version</a>, <a href="#ios_sticker_pack_extension-platform_type">platform_type</a>, <a href="#ios_sticker_pack_extension-provisioning_profile">provisioning_profile</a>,
                           <a href="#ios_sticker_pack_extension-shared_capabilities">shared_capabilities</a>, <a href="#ios_sticker_pack_extension-sticker_assets">sticker_assets</a>, <a href="#ios_sticker_pack_extension-strings">strings</a>, <a href="#ios_sticker_pack_extension-version">version</a>)
</pre>

Builds and bundles an iOS Sticker Pack Extension.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_sticker_pack_extension-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_sticker_pack_extension-resources"></a>resources |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_sticker_pack_extension-app_icons"></a>app_icons |  Files that comprise the app icons for the application. Each file must have a containing directory named `*..xcstickers/*..stickersiconset` and there may be only one such `..stickersiconset` directory in the list. In Xcode 26+ for iOS/macOS/watchOS, an `*.icon` bundle can be provided along with the `*..xcstickers` bundle to support 26 and pre-26 Apple OS rendering.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_sticker_pack_extension-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for this target. Only use this attribute if the bundle ID is not intended to be composed through an assigned base bundle ID rule found within `signed_capabilities`.   | String | optional |  `""`  |
| <a id="ios_sticker_pack_extension-bundle_id_suffix"></a>bundle_id_suffix |  A string to act as the suffix of the composed bundle ID. If this target's bundle ID is composed from a base bundle ID rule found within `signed_capabilities`, then this string will be appended to the end of the bundle ID following a "." separator.   | String | optional |  `"bundle_name"`  |
| <a id="ios_sticker_pack_extension-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_sticker_pack_extension-entitlements"></a>entitlements |  The entitlements file required for device builds of this target. If absent, the default entitlements from the provisioning profile will be used.<br><br>The following variables are substituted in the entitlements file: `$(CFBundleIdentifier)` with the bundle ID of the application and `$(AppIdentifierPrefix)` with the value of the `ApplicationIdentifierPrefix` key from the target's provisioning profile.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_sticker_pack_extension-entitlements_validation"></a>entitlements_validation |  An `entitlements_validation_mode` to control the validation of the requested entitlements against the provisioning profile to ensure they are supported.   | String | optional |  `"loose"`  |
| <a id="ios_sticker_pack_extension-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_sticker_pack_extension-families"></a>families |  A list of device families supported by this rule. At least one must be specified.   | List of strings | required |  |
| <a id="ios_sticker_pack_extension-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for this target. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_sticker_pack_extension-ipa_post_processor"></a>ipa_post_processor |  A tool that edits this target's archive after it is assembled but before it is signed. The tool is invoked with a single command-line argument that denotes the path to a directory containing the unzipped contents of the archive; this target's bundle will be the directory's only contents.<br><br>Any changes made by the tool must be made in this directory, and the tool's execution must be hermetic given these inputs to ensure that the result can be safely cached.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_sticker_pack_extension-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_sticker_pack_extension-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_sticker_pack_extension-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_sticker_pack_extension-provisioning_profile"></a>provisioning_profile |  The provisioning profile (`.mobileprovision` file) to use when creating the bundle. This value is optional for simulator builds as the simulator doesn't fully enforce entitlements, but is required for device builds.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_sticker_pack_extension-shared_capabilities"></a>shared_capabilities |  A list of shared `apple_capability_set` rules to represent the capabilities that a code sign aware Apple bundle rule output should have. These can define the formal prefix for the target's `bundle_id` and can further be merged with information provided by `entitlements`, if defined by any capabilities found within the `apple_capability_set`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_sticker_pack_extension-sticker_assets"></a>sticker_assets |  List of sticker files to bundle. The collection of assets should be under a folder named `*.*.xcstickers`. The icons go in a `*.stickersiconset` (instead of `*.appiconset`); and the files for the stickers should all be in Sticker Pack directories, so `*.stickerpack/*.sticker` or `*.stickerpack/*.stickersequence`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_sticker_pack_extension-strings"></a>strings |  A list of `.strings` files, often localizable. These files are converted to binary plists (if they are not already) and placed in the root of the final bundle, unless a file's immediate containing directory is named `*.lproj`, in which case it will be placed under a directory with the same name in the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_sticker_pack_extension-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-general.md?cl=head#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="ios_test_runner"></a>

## ios_test_runner

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_test_runner")

ios_test_runner(<a href="#ios_test_runner-name">name</a>, <a href="#ios_test_runner-device_type">device_type</a>, <a href="#ios_test_runner-execution_requirements">execution_requirements</a>, <a href="#ios_test_runner-os_version">os_version</a>, <a href="#ios_test_runner-post_action">post_action</a>,
                <a href="#ios_test_runner-post_action_determines_exit_code">post_action_determines_exit_code</a>, <a href="#ios_test_runner-pre_action">pre_action</a>, <a href="#ios_test_runner-test_environment">test_environment</a>)
</pre>

Rule to identify an iOS runner that runs tests for iOS.

The runner will create a new simulator according to the given arguments to run
tests.

Outputs:
  AppleTestRunnerInfo:
    test_runner_template: Template file that contains the specific mechanism
        with which the tests will be performed.
    execution_requirements: Dictionary that represents the specific hardware
        requirements for this test.
  Runfiles:
    files: The files needed during runtime for the test to be performed.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_test_runner-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_test_runner-device_type"></a>device_type |  The device type of the iOS simulator to run test. The supported types correspond to the output of `xcrun simctl list devicetypes`. E.g., iPhone 6, iPad Air. By default, it is the latest supported iPhone type.'   | String | optional |  `""`  |
| <a id="ios_test_runner-execution_requirements"></a>execution_requirements |  Dictionary of strings to strings which specifies the execution requirements for the runner. In most common cases, this should not be used.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{"requires-darwin": ""}`  |
| <a id="ios_test_runner-os_version"></a>os_version |  The os version of the iOS simulator to run test. The supported os versions correspond to the output of `xcrun simctl list runtimes`. ' 'E.g., 11.2, 9.3. By default, it is the latest supported version of the device type.'   | String | optional |  `""`  |
| <a id="ios_test_runner-post_action"></a>post_action |  A binary to run following test execution. Runs after testing but before test result handling and coverage processing. Sets the `$TEST_EXIT_CODE` environment variable, in addition to any other variables available to the test runner.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_test_runner-post_action_determines_exit_code"></a>post_action_determines_exit_code |  When true, the exit code of the test run will be set to the exit code of the post action. This is useful for tests that need to fail the test run based on their own criteria.   | Boolean | optional |  `False`  |
| <a id="ios_test_runner-pre_action"></a>pre_action |  A binary to run prior to test execution. Runs after simulator creation. Sets any environment variables available to the test runner.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_test_runner-test_environment"></a>test_environment |  Optional dictionary with the environment variables that are to be propagated into the XCTest invocation.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |


<a id="ios_ui_test"></a>

## ios_ui_test

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_ui_test")

ios_ui_test(<a href="#ios_ui_test-name">name</a>, <a href="#ios_ui_test-deps">deps</a>, <a href="#ios_ui_test-data">data</a>, <a href="#ios_ui_test-bundle_name">bundle_name</a>, <a href="#ios_ui_test-collect_code_coverage">collect_code_coverage</a>, <a href="#ios_ui_test-env">env</a>, <a href="#ios_ui_test-env_inherit">env_inherit</a>,
            <a href="#ios_ui_test-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#ios_ui_test-minimum_os_version">minimum_os_version</a>, <a href="#ios_ui_test-platform_type">platform_type</a>, <a href="#ios_ui_test-runner">runner</a>,
            <a href="#ios_ui_test-test_coverage_manifest">test_coverage_manifest</a>, <a href="#ios_ui_test-test_filter">test_filter</a>, <a href="#ios_ui_test-test_host">test_host</a>, <a href="#ios_ui_test-test_host_is_bundle_loader">test_host_is_bundle_loader</a>)
</pre>

iOS UI Test rule.

Builds and bundles an iOS UI `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

The `provisioning_profile` attribute needs to be set to run the test on a real device.

To run the same test on multiple simulators/devices see
[ios_ui_test_suite](#ios_ui_test_suite).

The following is a list of the `ios_ui_test` specific attributes; for a list
of the attributes inherited by all test rules, please check the
[Bazel documentation](https://bazel.build/reference/be/common-definitions#common-attributes-tests).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_ui_test-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_ui_test-deps"></a>deps |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_ui_test-data"></a>data |  Files to be made available to the test during its execution.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_ui_test-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_ui_test-collect_code_coverage"></a>collect_code_coverage |  Whether to collect code coverage for this test if `--collect_code_coverage=yes`.   | Boolean | optional |  `True`  |
| <a id="ios_ui_test-env"></a>env |  Dictionary of environment variables that should be set during the test execution. The values of the dictionary are subject to "Make" variable expansion.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="ios_ui_test-env_inherit"></a>env_inherit |  List of environment variables to inherit from the external environment.   | List of strings | optional |  `[]`  |
| <a id="ios_ui_test-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_ui_test-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_ui_test-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_ui_test-runner"></a>runner |  The runner target that will provide the logic on how to run the tests. Needs to provide the AppleTestRunnerInfo provider.   | <a href="https://bazel.build/concepts/labels">Label</a> | required |  |
| <a id="ios_ui_test-test_coverage_manifest"></a>test_coverage_manifest |  A file that will be used in lcov export calls to limit the scope of files instrumented with coverage.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_ui_test-test_filter"></a>test_filter |  Test filter string that will be passed into the test runner to select which tests will run.   | String | optional |  `""`  |
| <a id="ios_ui_test-test_host"></a>test_host |  -   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_ui_test-test_host_is_bundle_loader"></a>test_host_is_bundle_loader |  Whether the 'test_host' should be used as the -bundle_loader to allow testing the symbols from the test host app   | Boolean | optional |  `True`  |


<a id="ios_unit_test"></a>

## ios_unit_test

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_unit_test")

ios_unit_test(<a href="#ios_unit_test-name">name</a>, <a href="#ios_unit_test-deps">deps</a>, <a href="#ios_unit_test-data">data</a>, <a href="#ios_unit_test-bundle_name">bundle_name</a>, <a href="#ios_unit_test-collect_code_coverage">collect_code_coverage</a>, <a href="#ios_unit_test-env">env</a>, <a href="#ios_unit_test-env_inherit">env_inherit</a>,
              <a href="#ios_unit_test-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#ios_unit_test-minimum_os_version">minimum_os_version</a>, <a href="#ios_unit_test-platform_type">platform_type</a>, <a href="#ios_unit_test-runner">runner</a>,
              <a href="#ios_unit_test-test_coverage_manifest">test_coverage_manifest</a>, <a href="#ios_unit_test-test_filter">test_filter</a>, <a href="#ios_unit_test-test_host">test_host</a>, <a href="#ios_unit_test-test_host_is_bundle_loader">test_host_is_bundle_loader</a>)
</pre>

Builds and bundles an iOS Unit `.xctest` test bundle. Runs the tests using the
provided test runner when invoked with `bazel test`.

`ios_unit_test` targets can work in two modes: as app or library
tests. If the `test_host` attribute is set to an `ios_application` target, the
tests will run within that application's context. If no `test_host` is provided,
the tests will run outside the context of an iOS application. Because of this,
certain functionalities might not be present (e.g. UI layout, NSUserDefaults).
You can find more information about app and library testing for Apple platforms
[here](https://developer.apple.com/library/content/documentation/DeveloperTools/Conceptual/testing_with_xcode/chapters/03-testing_basics.html).

The `provisioning_profile` attribute needs to be set to run the test on a real device.

To run the same test on multiple simulators/devices see
[ios_unit_test_suite](#ios_unit_test_suite).

The following is a list of the `ios_unit_test` specific attributes; for a list
of the attributes inherited by all test rules, please check the
[Bazel documentation](https://bazel.build/reference/be/common-definitions#common-attributes-tests).

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_unit_test-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_unit_test-deps"></a>deps |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="ios_unit_test-data"></a>data |  Files to be made available to the test during its execution.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="ios_unit_test-bundle_name"></a>bundle_name |  The desired name of the bundle (without the extension). If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="ios_unit_test-collect_code_coverage"></a>collect_code_coverage |  Whether to collect code coverage for this test if `--collect_code_coverage=yes`.   | Boolean | optional |  `True`  |
| <a id="ios_unit_test-env"></a>env |  Dictionary of environment variables that should be set during the test execution. The values of the dictionary are subject to "Make" variable expansion.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="ios_unit_test-env_inherit"></a>env_inherit |  List of environment variables to inherit from the external environment.   | List of strings | optional |  `[]`  |
| <a id="ios_unit_test-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="ios_unit_test-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="ios_unit_test-platform_type"></a>platform_type |  -   | String | optional |  `"ios"`  |
| <a id="ios_unit_test-runner"></a>runner |  The runner target that will provide the logic on how to run the tests. Needs to provide the AppleTestRunnerInfo provider.   | <a href="https://bazel.build/concepts/labels">Label</a> | required |  |
| <a id="ios_unit_test-test_coverage_manifest"></a>test_coverage_manifest |  A file that will be used in lcov export calls to limit the scope of files instrumented with coverage.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_unit_test-test_filter"></a>test_filter |  Test filter string that will be passed into the test runner to select which tests will run.   | String | optional |  `""`  |
| <a id="ios_unit_test-test_host"></a>test_host |  -   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_unit_test-test_host_is_bundle_loader"></a>test_host_is_bundle_loader |  Whether the 'test_host' should be used as the -bundle_loader to allow testing the symbols from the test host app   | Boolean | optional |  `True`  |


<a id="ios_xctestrun_runner"></a>

## ios_xctestrun_runner

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_xctestrun_runner")

ios_xctestrun_runner(<a href="#ios_xctestrun_runner-name">name</a>, <a href="#ios_xctestrun_runner-attachment_lifetime">attachment_lifetime</a>, <a href="#ios_xctestrun_runner-command_line_args">command_line_args</a>, <a href="#ios_xctestrun_runner-create_xcresult_bundle">create_xcresult_bundle</a>,
                     <a href="#ios_xctestrun_runner-destination_timeout">destination_timeout</a>, <a href="#ios_xctestrun_runner-device_type">device_type</a>, <a href="#ios_xctestrun_runner-os_version">os_version</a>, <a href="#ios_xctestrun_runner-post_action">post_action</a>,
                     <a href="#ios_xctestrun_runner-post_action_determines_exit_code">post_action_determines_exit_code</a>, <a href="#ios_xctestrun_runner-pre_action">pre_action</a>, <a href="#ios_xctestrun_runner-random">random</a>, <a href="#ios_xctestrun_runner-reuse_simulator">reuse_simulator</a>,
                     <a href="#ios_xctestrun_runner-xcodebuild_args">xcodebuild_args</a>)
</pre>

This rule creates a test runner for iOS tests that uses xctestrun files to run
hosted tests, and uses xctest directly to run logic tests.

You can use this rule directly if you need to override 'device_type' or
'os_version', otherwise you can use the predefined runners:

```
"@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_ordered_runner"
```

or:

```
"@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_random_runner"
```

Depending on if you want random test ordering or not. Set these as the `runner`
attribute on your `ios_unit_test` target:

```bzl
ios_unit_test(
    name = "Tests",
    minimum_os_version = "15.5",
    runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_random_runner",
    deps = [":TestsLib"],
)
```

If you would like this test runner to generate xcresult bundles for your tests,
pass `--test_env=CREATE_XCRESULT_BUNDLE=1`. It is preferable to use the
`create_xcresult_bundle` on the test runner itself instead of this parameter.

This rule automatically handles running x86_64 tests on arm64 hosts. The only
exception is that if you want to generate xcresult bundles or run tests in
random order, the test must have a test host. This is because of a limitation
in Xcode.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="ios_xctestrun_runner-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="ios_xctestrun_runner-attachment_lifetime"></a>attachment_lifetime |  Attachment lifetime to set in the xctestrun file when running the test bundle - `"keepNever"` (default), `"keepAlways"` or `"deleteOnSuccess"`. This affects presence of attachments in the XCResult output. This does not force using `xcodebuild` or an XCTestRun file but the value will be used in that case.   | String | optional |  `"keepNever"`  |
| <a id="ios_xctestrun_runner-command_line_args"></a>command_line_args |  CommandLineArguments to pass to xctestrun file when running the test bundle. This means it will always use `xcodebuild test-without-building` to run the test bundle.   | List of strings | optional |  `[]`  |
| <a id="ios_xctestrun_runner-create_xcresult_bundle"></a>create_xcresult_bundle |  Force the test runner to always create an XCResult bundle. This means it will always use `xcodebuild test-without-building` to run the test bundle.   | Boolean | optional |  `False`  |
| <a id="ios_xctestrun_runner-destination_timeout"></a>destination_timeout |  Use the specified timeout when searching for a destination device. The default is 30 seconds.   | Integer | optional |  `0`  |
| <a id="ios_xctestrun_runner-device_type"></a>device_type |  The device type of the iOS simulator to run test. The supported types correspond to the output of `xcrun simctl list devicetypes`. E.g., iPhone X, iPad Air. By default, it reads from --ios_simulator_device or falls back to some device.   | String | optional |  `""`  |
| <a id="ios_xctestrun_runner-os_version"></a>os_version |  The os version of the iOS simulator to run test. The supported os versions correspond to the output of `xcrun simctl list runtimes`. E.g., 15.5. By default, it reads --ios_simulator_version and then falls back to the latest supported version.   | String | optional |  `""`  |
| <a id="ios_xctestrun_runner-post_action"></a>post_action |  A binary to run following test execution. Runs after testing but before test result handling and coverage processing. Sets the `$TEST_EXIT_CODE`, `$TEST_LOG_FILE`, and `$SIMULATOR_UDID` environment variables, the `$TEST_XCRESULT_BUNDLE_PATH` environment variable if the test run produces an XCResult bundle, and any other variables available to the test runner.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_xctestrun_runner-post_action_determines_exit_code"></a>post_action_determines_exit_code |  When true, the exit code of the test run will be set to the exit code of the post action. This is useful for tests that need to fail the test run based on their own criteria.   | Boolean | optional |  `False`  |
| <a id="ios_xctestrun_runner-pre_action"></a>pre_action |  A binary to run prior to test execution. Runs after simulator creation. Sets the `$SIMULATOR_UDID` environment variable, in addition to any other variables available to the test runner.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="ios_xctestrun_runner-random"></a>random |  Whether to run the tests in random order to identify unintended state dependencies.   | Boolean | optional |  `False`  |
| <a id="ios_xctestrun_runner-reuse_simulator"></a>reuse_simulator |  Toggle simulator reuse. The default behavior is to reuse an existing device of the same type and OS version. When disabled, a new simulator is created before testing starts and shutdown when the runner completes.   | Boolean | optional |  `True`  |
| <a id="ios_xctestrun_runner-xcodebuild_args"></a>xcodebuild_args |  Arguments to pass to `xcodebuild` when running the test bundle. This means it will always use `xcodebuild test-without-building` to run the test bundle.   | List of strings | optional |  `[]`  |


<a id="ios_ui_test_suite"></a>

## ios_ui_test_suite

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_ui_test_suite")

ios_ui_test_suite(<a href="#ios_ui_test_suite-name">name</a>, <a href="#ios_ui_test_suite-runners">runners</a>, <a href="#ios_ui_test_suite-kwargs">**kwargs</a>)
</pre>

Generates a [test_suite] containing an [ios_ui_test] for each of the given `runners`.

`ios_ui_test_suite` takes the same parameters as [ios_ui_test], except `runner` is replaced by `runners`.

[test_suite]: https://docs.bazel.build/versions/master/be/general.html#test_suite
[ios_ui_test]: #ios_ui_test


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="ios_ui_test_suite-name"></a>name |  <p align="center"> - </p>   |  none |
| <a id="ios_ui_test_suite-runners"></a>runners |  a list of runner targets   |  `None` |
| <a id="ios_ui_test_suite-kwargs"></a>kwargs |  passed to the [ios_ui_test]   |  none |


<a id="ios_unit_test_suite"></a>

## ios_unit_test_suite

<pre>
load("@rules_apple//apple:ios.doc.bzl", "ios_unit_test_suite")

ios_unit_test_suite(<a href="#ios_unit_test_suite-name">name</a>, <a href="#ios_unit_test_suite-runners">runners</a>, <a href="#ios_unit_test_suite-kwargs">**kwargs</a>)
</pre>

Generates a [test_suite] containing an [ios_unit_test] for each of the given `runners`.

`ios_unit_test_suite` takes the same parameters as [ios_unit_test], except `runner` is replaced by `runners`.

[test_suite]: https://docs.bazel.build/versions/master/be/general.html#test_suite
[ios_unit_test]: #ios_unit_test


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="ios_unit_test_suite-name"></a>name |  <p align="center"> - </p>   |  none |
| <a id="ios_unit_test_suite-runners"></a>runners |  a list of runner targets   |  `None` |
| <a id="ios_unit_test_suite-kwargs"></a>kwargs |  passed to the [ios_unit_test]   |  none |


