<!-- Generated with Stardoc: http://skydoc.bazel.build -->

# Providers

Defines providers and related types used throughout the rules in this repository.

Most users will not need to use these providers to simply create and build Apple
targets, but if you want to write your own custom rules that interact with these
rules, then you will use these providers to communicate between them.

These providers are part of the public API of the bundling rules. Other rules that want to propagate
information to the bundling rules or that want to consume the bundling rules as their own inputs
should use these to handle the relevant information that they need.

Public initializers must be defined in apple:providers.bzl instead of apple/internal:providers.bzl.
These should build from the "raw initializer" where possible, but not export it, to allow for a safe
boundary with well-defined public APIs for broader usage.

<a id="AppleBaseBundleIdInfo"></a>

## AppleBaseBundleIdInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleBaseBundleIdInfo")

AppleBaseBundleIdInfo(<a href="#AppleBaseBundleIdInfo-_init-kwargs">*kwargs</a>)
</pre>

Provides the base bundle ID prefix for an Apple rule.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleBaseBundleIdInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleBaseBundleIdInfo-base_bundle_id"></a>base_bundle_id |  `String`. The bundle ID prefix, composed from an organization ID and an optional variant name.    |


<a id="AppleBinaryInfo"></a>

## AppleBinaryInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleBinaryInfo")

AppleBinaryInfo(<a href="#AppleBinaryInfo-_init-kwargs">*kwargs</a>)
</pre>

Provides information about an Apple binary target.

This provider propagates general information about an Apple binary that is not
specific to any particular binary type.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleBinaryInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleBinaryInfo-binary"></a>binary |  `File`. The binary (executable, dynamic library, etc.) file that the target represents.    |
| <a id="AppleBinaryInfo-infoplist"></a>infoplist |  `File`. The complete (binary-formatted) `Info.plist` embedded in the binary.    |
| <a id="AppleBinaryInfo-product_type"></a>product_type |  `String`. The dot-separated product type identifier associated with the binary (for example, `com.apple.product-type.tool`).    |


<a id="AppleBinaryInfoplistInfo"></a>

## AppleBinaryInfoplistInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleBinaryInfoplistInfo")

AppleBinaryInfoplistInfo(<a href="#AppleBinaryInfoplistInfo-infoplist">infoplist</a>)
</pre>

Provides information about the Info.plist that was linked into an Apple binary
target.

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleBinaryInfoplistInfo-infoplist"></a>infoplist |  `File`. The complete (binary-formatted) `Info.plist` embedded in the binary.    |


<a id="AppleBundleInfo"></a>

## AppleBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleBundleInfo")

AppleBundleInfo(<a href="#AppleBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

This provider propagates general information about an Apple bundle that is not
specific to any particular bundle type. It is propagated by most bundling
rules (applications, extensions, frameworks, test bundles, and so forth).

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleBundleInfo-archive"></a>archive |  `File`. The archive that contains the built bundle.    |
| <a id="AppleBundleInfo-archive_root"></a>archive_root |  `String`. The file system path (relative to the workspace root) where the signed bundle was constructed (before archiving). Other rules **should not** depend on this field; it is intended to support IDEs that want to read that path from the provider to avoid performance issues from unzipping the output archive.    |
| <a id="AppleBundleInfo-binary"></a>binary |  `File`. The binary (executable, dynamic library, etc.) that was bundled. The physical file is identical to the one inside the bundle except that it is always unsigned, so note that it is _not_ a path to the binary inside your output bundle. The primary purpose of this field is to provide a way to access the binary directly at analysis time; for example, for code coverage.    |
| <a id="AppleBundleInfo-bundle_extension"></a>bundle_extension |  `String`. The bundle extension.    |
| <a id="AppleBundleInfo-bundle_id"></a>bundle_id |  `String`. The bundle identifier (i.e., `CFBundleIdentifier` in `Info.plist`) of the bundle.    |
| <a id="AppleBundleInfo-bundle_name"></a>bundle_name |  `String`. The name of the bundle, without the extension.    |
| <a id="AppleBundleInfo-entitlements"></a>entitlements |  `File`. Entitlements file used, if any.    |
| <a id="AppleBundleInfo-executable_name"></a>executable_name |  `string`. The name of the executable that was bundled.    |
| <a id="AppleBundleInfo-extension_safe"></a>extension_safe |  `Boolean`. True if the target propagating this provider was compiled and linked with -application-extension, restricting it to extension-safe APIs only.    |
| <a id="AppleBundleInfo-infoplist"></a>infoplist |  `File`. The complete (binary-formatted) `Info.plist` file for the bundle.    |
| <a id="AppleBundleInfo-minimum_deployment_os_version"></a>minimum_deployment_os_version |  `string`. The minimum deployment OS version (as a dotted version number like "9.0") that this bundle was built to support. This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.    |
| <a id="AppleBundleInfo-minimum_os_version"></a>minimum_os_version |  `String`. The minimum OS version (as a dotted version number like "9.0") that this bundle was built to support.    |
| <a id="AppleBundleInfo-platform_type"></a>platform_type |  `String`. The platform type for the bundle (i.e. `ios` for iOS bundles).    |
| <a id="AppleBundleInfo-product_type"></a>product_type |  `String`. The dot-separated product type identifier associated with the bundle (for example, `com.apple.product-type.application`).    |
| <a id="AppleBundleInfo-uses_swift"></a>uses_swift |  Boolean. True if Swift is used by the target propagating this provider. This does not consider embedded bundles; for example, an Objective-C application containing a Swift extension would have this field set to true for the extension but false for the application.    |


<a id="AppleBundleVersionInfo"></a>

## AppleBundleVersionInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleBundleVersionInfo")

AppleBundleVersionInfo(<a href="#AppleBundleVersionInfo-_init-kwargs">*kwargs</a>)
</pre>

Provides versioning information for an Apple bundle.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleBundleVersionInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleBundleVersionInfo-version_file"></a>version_file |  Required. A `File` containing JSON-formatted text describing the version number information propagated by the target.<br><br>It contains two keys:<br><br>*   `build_version`, which corresponds to `CFBundleVersion`.<br><br>*   `short_version_string`, which corresponds to `CFBundleShortVersionString`.    |


<a id="AppleCodesigningDossierInfo"></a>

## AppleCodesigningDossierInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleCodesigningDossierInfo")

AppleCodesigningDossierInfo(<a href="#AppleCodesigningDossierInfo-_init-kwargs">*kwargs</a>)
</pre>

Provides information around the use of a code signing dossier.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleCodesigningDossierInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleCodesigningDossierInfo-dossier"></a>dossier |  A `File` reference to the code signing dossier zip that acts as a direct dependency of the given target if one was generated.    |


<a id="AppleDebugOutputsInfo"></a>

## AppleDebugOutputsInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleDebugOutputsInfo")

AppleDebugOutputsInfo(<a href="#AppleDebugOutputsInfo-_init-kwargs">*kwargs</a>)
</pre>

Holds debug outputs of an Apple binary rule.

This provider is DEPRECATED. Preferably use `AppleDsymBundleInfo` instead.

The only field is `output_map`, which is a dictionary of:
  `{ arch: { "dsym_binary": File, "linkmap": File }`

Where `arch` is any Apple architecture such as "arm64" or "armv7".

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleDebugOutputsInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleDebugOutputsInfo-outputs_map"></a>outputs_map |  -    |


<a id="AppleDeviceTestRunnerInfo"></a>

## AppleDeviceTestRunnerInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleDeviceTestRunnerInfo")

AppleDeviceTestRunnerInfo(<a href="#AppleDeviceTestRunnerInfo-device_type">device_type</a>, <a href="#AppleDeviceTestRunnerInfo-os_version">os_version</a>)
</pre>

Provider that device-based runner targets must propagate.

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleDeviceTestRunnerInfo-device_type"></a>device_type |  The device type of the iOS simulator to run test. The supported types correspond to the output of `xcrun simctl list devicetypes`. E.g., iPhone X, iPad Air.    |
| <a id="AppleDeviceTestRunnerInfo-os_version"></a>os_version |  The os version of the iOS simulator to run test. The supported os versions correspond to the output of `xcrun simctl list runtimes`. E.g., 15.5.    |


<a id="AppleDsymBundleInfo"></a>

## AppleDsymBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleDsymBundleInfo")

AppleDsymBundleInfo(<a href="#AppleDsymBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Provides information for an Apple dSYM bundle.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleDsymBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleDsymBundleInfo-direct_dsyms"></a>direct_dsyms |  `List` containing `File` references to each of the dSYM bundles that act as direct dependencies of the given target if any were generated.    |
| <a id="AppleDsymBundleInfo-transitive_dsyms"></a>transitive_dsyms |  `depset` containing `File` references to each of the dSYM bundles that act as transitive dependencies of the given target if any were generated.    |


<a id="AppleDynamicFrameworkInfo"></a>

## AppleDynamicFrameworkInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleDynamicFrameworkInfo")

AppleDynamicFrameworkInfo(<a href="#AppleDynamicFrameworkInfo-framework_dirs">framework_dirs</a>, <a href="#AppleDynamicFrameworkInfo-framework_files">framework_files</a>, <a href="#AppleDynamicFrameworkInfo-binary">binary</a>, <a href="#AppleDynamicFrameworkInfo-cc_info">cc_info</a>)
</pre>

Contains information about an Apple dynamic framework.

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleDynamicFrameworkInfo-framework_dirs"></a>framework_dirs |  The framework path names used as link inputs in order to link against the dynamic framework.    |
| <a id="AppleDynamicFrameworkInfo-framework_files"></a>framework_files |  The full set of artifacts that should be included as inputs to link against the dynamic framework.    |
| <a id="AppleDynamicFrameworkInfo-binary"></a>binary |  The dylib binary artifact of the dynamic framework.    |
| <a id="AppleDynamicFrameworkInfo-cc_info"></a>cc_info |  A `CcInfo` which contains information about the transitive dependencies linked into the binary.    |


<a id="AppleExecutableBinaryInfo"></a>

## AppleExecutableBinaryInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleExecutableBinaryInfo")

AppleExecutableBinaryInfo(<a href="#AppleExecutableBinaryInfo-objc">objc</a>, <a href="#AppleExecutableBinaryInfo-binary">binary</a>, <a href="#AppleExecutableBinaryInfo-cc_info">cc_info</a>)
</pre>

Contains the executable binary output that was built using
`link_multi_arch_binary` with the `executable` binary type.

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleExecutableBinaryInfo-objc"></a>objc |  apple_common.Objc provider used for legacy linking behavior.    |
| <a id="AppleExecutableBinaryInfo-binary"></a>binary |  The executable binary artifact output by `link_multi_arch_binary`.    |
| <a id="AppleExecutableBinaryInfo-cc_info"></a>cc_info |  A `CcInfo` which contains information about the transitive dependencies linked into the binary.    |


<a id="AppleExtraOutputsInfo"></a>

## AppleExtraOutputsInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleExtraOutputsInfo")

AppleExtraOutputsInfo(<a href="#AppleExtraOutputsInfo-_init-kwargs">*kwargs</a>)
</pre>

Provides information about extra outputs that should be produced from the build.

This provider propagates supplemental files that should be produced as outputs
even if the bundle they are associated with is not a direct output of the rule.
For example, an application that contains an extension will build both targets
but only the application will be a rule output. However, if dSYM bundles are
also being generated, we do want to produce the dSYMs for *both* application and
extension as outputs of the build, not just the dSYMs of the explicit target
being built (the application).

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleExtraOutputsInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleExtraOutputsInfo-files"></a>files |  `depset` of `File`s. These files will be propagated from embedded bundles (such as frameworks and extensions) to the top-level bundle (such as an application) to ensure that they are explicitly produced as outputs of the build.    |


<a id="AppleFrameworkBundleInfo"></a>

## AppleFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleFrameworkBundleInfo")

AppleFrameworkBundleInfo(<a href="#AppleFrameworkBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes a target is an Apple framework bundle.

This provider does not reference 3rd party or precompiled frameworks.
Propagated by Apple framework rules: `ios_framework`, and `tvos_framework`.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleFrameworkBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="AppleFrameworkImportInfo"></a>

## AppleFrameworkImportInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleFrameworkImportInfo")

AppleFrameworkImportInfo(<a href="#AppleFrameworkImportInfo-_init-kwargs">*kwargs</a>)
</pre>

Provider that propagates information about 3rd party imported framework targets.

Propagated by framework and XCFramework import rules: `apple_dynamic_framework_import`,
`apple_dynamic_xcframework_import`, `apple_static_framework_import`, and
`apple_static_xcframework_import`

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleFrameworkImportInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleFrameworkImportInfo-framework_imports"></a>framework_imports |  `depset` of `File`s that represent framework imports that need to be bundled in the top level application bundle under the Frameworks directory.    |
| <a id="AppleFrameworkImportInfo-dsym_imports"></a>dsym_imports |  Depset of Files that represent dSYM imports that need to be processed to provide .symbols files for packaging into the .ipa file if requested in the build with --define=apple.package_symbols=(yes\|true\|1).    |
| <a id="AppleFrameworkImportInfo-build_archs"></a>build_archs |  `depset` of `String`s that represent binary architectures reported from the current build.    |
| <a id="AppleFrameworkImportInfo-debug_info_binaries"></a>debug_info_binaries |  Depset of Files that represent framework binaries and dSYM binaries that provide debug info.    |


<a id="ApplePlatformInfo"></a>

## ApplePlatformInfo

<pre>
load("@rules_apple//apple:providers.bzl", "ApplePlatformInfo")

ApplePlatformInfo(<a href="#ApplePlatformInfo-_init-kwargs">*kwargs</a>)
</pre>

Provides information for the currently selected Apple platforms.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="ApplePlatformInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="ApplePlatformInfo-target_os"></a>target_os |  `String` representing the selected Apple OS.    |
| <a id="ApplePlatformInfo-target_arch"></a>target_arch |  `String` representing the selected target architecture or cpu type.    |
| <a id="ApplePlatformInfo-target_environment"></a>target_environment |  `String` representing the selected target environment (e.g. "device", "simulator").    |


<a id="AppleProvisioningProfileInfo"></a>

## AppleProvisioningProfileInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleProvisioningProfileInfo")

AppleProvisioningProfileInfo(<a href="#AppleProvisioningProfileInfo-provisioning_profile">provisioning_profile</a>, <a href="#AppleProvisioningProfileInfo-profile_name">profile_name</a>, <a href="#AppleProvisioningProfileInfo-team_id">team_id</a>)
</pre>

Provides information about a provisioning profile.

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleProvisioningProfileInfo-provisioning_profile"></a>provisioning_profile |  `File`. The provisioning profile.    |
| <a id="AppleProvisioningProfileInfo-profile_name"></a>profile_name |  string. The profile name (e.g. "iOS Team Provisioning Profile: com.example.app").    |
| <a id="AppleProvisioningProfileInfo-team_id"></a>team_id |  `string`. The Team ID the profile is associated with (e.g. "A12B3CDEFG"), or `None` if it's not known at analysis time.    |


<a id="AppleResourceBundleInfo"></a>

## AppleResourceBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleResourceBundleInfo")

AppleResourceBundleInfo(<a href="#AppleResourceBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is an Apple resource bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an Apple resource bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is an Apple resource bundle should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleResourceBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="AppleResourceInfo"></a>

## AppleResourceInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleResourceInfo")

AppleResourceInfo(<a href="#AppleResourceInfo-_init-kwargs">*kwargs</a>)
</pre>

Provider that propagates buckets of resources that are differentiated by type.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleResourceInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleResourceInfo-alternate_icons"></a>alternate_icons |  Alternate icons to be included in the App bundle.    |
| <a id="AppleResourceInfo-asset_catalogs"></a>asset_catalogs |  Resources that need to be embedded into Assets.car.    |
| <a id="AppleResourceInfo-datamodels"></a>datamodels |  Datamodel files.    |
| <a id="AppleResourceInfo-framework"></a>framework |  Apple framework bundle from `ios_framework` and `tvos_framework` targets.    |
| <a id="AppleResourceInfo-infoplists"></a>infoplists |  Plist files to be merged and processed. Plist files that should not be merged into the root Info.plist should be propagated in `plists`. Because of this, infoplists should only be bucketed with the `bucketize_typed` method.    |
| <a id="AppleResourceInfo-metals"></a>metals |  Metal Shading Language source files to be compiled into a single .metallib file and bundled at the top level.    |
| <a id="AppleResourceInfo-mlmodels"></a>mlmodels |  Core ML model files that should be processed and bundled at the top level.    |
| <a id="AppleResourceInfo-plists"></a>plists |  Resource Plist files that should not be merged into Info.plist    |
| <a id="AppleResourceInfo-pngs"></a>pngs |  PNG images which are not bundled in an .xcassets folder or an .icon folder in Xcode 26+.    |
| <a id="AppleResourceInfo-processed"></a>processed |  Typed resources that have already been processed.    |
| <a id="AppleResourceInfo-storyboards"></a>storyboards |  Storyboard files.    |
| <a id="AppleResourceInfo-strings"></a>strings |  Localization strings files.    |
| <a id="AppleResourceInfo-texture_atlases"></a>texture_atlases |  Texture atlas files.    |
| <a id="AppleResourceInfo-unprocessed"></a>unprocessed |  Generic resources not mapped to the other types.    |
| <a id="AppleResourceInfo-xcstrings"></a>xcstrings |  String catalog files.    |
| <a id="AppleResourceInfo-xibs"></a>xibs |  XIB Interface files.    |
| <a id="AppleResourceInfo-owners"></a>owners |  `depset` of (resource, owner) pairs.    |
| <a id="AppleResourceInfo-processed_origins"></a>processed_origins |  `depset` of (processed resource, resource list) pairs.    |
| <a id="AppleResourceInfo-unowned_resources"></a>unowned_resources |  `depset` of unowned resources.    |


<a id="AppleSharedCapabilityInfo"></a>

## AppleSharedCapabilityInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleSharedCapabilityInfo")

AppleSharedCapabilityInfo(<a href="#AppleSharedCapabilityInfo-_init-kwargs">*kwargs</a>)
</pre>

Provides information on a mergeable set of shared capabilities.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleSharedCapabilityInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleSharedCapabilityInfo-base_bundle_id"></a>base_bundle_id |  `String`. The bundle ID prefix, composed from an organization ID and an optional variant name.    |


<a id="AppleStaticXcframeworkBundleInfo"></a>

## AppleStaticXcframeworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleStaticXcframeworkBundleInfo")

AppleStaticXcframeworkBundleInfo(<a href="#AppleStaticXcframeworkBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a static library XCFramework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an XCFramework bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is an XCFramework should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleStaticXcframeworkBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="AppleTestInfo"></a>

## AppleTestInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleTestInfo")

AppleTestInfo(<a href="#AppleTestInfo-_init-kwargs">*kwargs</a>)
</pre>

Provider that test targets propagate to be used for IDE integration.

This includes information regarding test source files, transitive include paths,
transitive module maps, and transitive Swift modules. Test source files are
considered to be all of which belong to the first-level dependencies on the test
target.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleTestInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleTestInfo-includes"></a>includes |  `depset` of `String`s representing transitive include paths which are needed by IDEs to be used for indexing the test sources.    |
| <a id="AppleTestInfo-module_maps"></a>module_maps |  `depset` of `File`s representing module maps which are needed by IDEs to be used for indexing the test sources.    |
| <a id="AppleTestInfo-module_name"></a>module_name |  `String` representing the module name used by the test's sources. This is only set if the test only contains a single top-level Swift dependency. This may be used by an IDE to identify the Swift module (if any) used by the test's sources.    |
| <a id="AppleTestInfo-non_arc_sources"></a>non_arc_sources |  `depset` of `File`s containing non-ARC sources from the test's immediate deps.    |
| <a id="AppleTestInfo-sources"></a>sources |  `depset` of `File`s containing sources and headers from the test's immediate deps.    |
| <a id="AppleTestInfo-swift_modules"></a>swift_modules |  `depset` of `File`s representing transitive swift modules which are needed by IDEs to be used for indexing the test sources.    |
| <a id="AppleTestInfo-test_bundle"></a>test_bundle |  The artifact representing the XCTest bundle for the test target.    |
| <a id="AppleTestInfo-test_host"></a>test_host |  The artifact representing the test host for the test target, if the test requires a test host.    |
| <a id="AppleTestInfo-deps"></a>deps |  `depset` of `String`s representing the labels of all immediate deps of the test. Only source files from these deps will be present in `sources`. This may be used by IDEs to differentiate a test target's transitive module maps from its direct module maps, as including the direct module maps may break indexing for the source files of the immediate deps.    |


<a id="AppleTestRunnerInfo"></a>

## AppleTestRunnerInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleTestRunnerInfo")

AppleTestRunnerInfo(<a href="#AppleTestRunnerInfo-_init-kwargs">*kwargs</a>)
</pre>

Provider that runner targets must propagate.

In addition to the fields, all the runfiles that the runner target declares will be added to the
test rules runfiles.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleTestRunnerInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="AppleTestRunnerInfo-execution_requirements"></a>execution_requirements |  Optional dictionary that represents the specific hardware requirements for this test.    |
| <a id="AppleTestRunnerInfo-execution_environment"></a>execution_environment |  Optional dictionary with the environment variables that are to be set in the test action, and are not propagated into the XCTest invocation. These values will _not_ be added into the %(test_env)s substitution, but will be set in the test action.    |
| <a id="AppleTestRunnerInfo-test_environment"></a>test_environment |  Optional dictionary with the environment variables that are to be propagated into the XCTest invocation. These values will be included in the %(test_env)s substitution and will _not_ be set in the test action.    |
| <a id="AppleTestRunnerInfo-test_runner_template"></a>test_runner_template |  Required template file that contains the specific mechanism with which the tests will be run. The *_ui_test and *_unit_test rules will substitute the following values:     * %(test_host_path)s:   Path to the app being tested.     * %(test_bundle_path)s: Path to the test bundle that contains the tests.     * %(test_env)s:         Environment variables for the XCTest invocation (e.g FOO=BAR,BAZ=QUX).     * %(test_type)s:        The test type, whether it is unit or UI.    |


<a id="AppleXcframeworkBundleInfo"></a>

## AppleXcframeworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "AppleXcframeworkBundleInfo")

AppleXcframeworkBundleInfo(<a href="#AppleXcframeworkBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is an XCFramework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an XCFramework bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is an XCFramework should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="AppleXcframeworkBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="DocCBundleInfo"></a>

## DocCBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "DocCBundleInfo")

DocCBundleInfo(<a href="#DocCBundleInfo-bundle">bundle</a>, <a href="#DocCBundleInfo-bundle_files">bundle_files</a>)
</pre>

Provides general information about a .docc bundle.

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="DocCBundleInfo-bundle"></a>bundle |  the path to the .docc bundle    |
| <a id="DocCBundleInfo-bundle_files"></a>bundle_files |  the file targets contained within the .docc bundle    |


<a id="DocCSymbolGraphsInfo"></a>

## DocCSymbolGraphsInfo

<pre>
load("@rules_apple//apple:providers.bzl", "DocCSymbolGraphsInfo")

DocCSymbolGraphsInfo(<a href="#DocCSymbolGraphsInfo-symbol_graphs">symbol_graphs</a>)
</pre>

Provides the symbol graphs required to archive a .docc bundle.

**FIELDS**

| Name  | Description |
| :------------- | :------------- |
| <a id="DocCSymbolGraphsInfo-symbol_graphs"></a>symbol_graphs |  the depset of paths to the symbol graphs    |


<a id="IosAppClipBundleInfo"></a>

## IosAppClipBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "IosAppClipBundleInfo")

IosAppClipBundleInfo(<a href="#IosAppClipBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is an iOS app clip.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS app clip bundle (and
not some other Apple bundle). Rule authors who wish to require that a dependency
is an iOS app clip should use this provider to describe that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="IosAppClipBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="IosApplicationBundleInfo"></a>

## IosApplicationBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "IosApplicationBundleInfo")

IosApplicationBundleInfo(<a href="#IosApplicationBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is an iOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS application bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is an iOS application should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="IosApplicationBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="IosExtensionBundleInfo"></a>

## IosExtensionBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "IosExtensionBundleInfo")

IosExtensionBundleInfo(<a href="#IosExtensionBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is an iOS application extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS application
extension bundle (and not some other Apple bundle). Rule authors who wish to
require that a dependency is an iOS application extension should use this
provider to describe that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="IosExtensionBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="IosFrameworkBundleInfo"></a>

## IosFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "IosFrameworkBundleInfo")

IosFrameworkBundleInfo(<a href="#IosFrameworkBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is an iOS dynamic framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS dynamic framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an iOS dynamic framework should use this provider to describe
that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="IosFrameworkBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="IosImessageApplicationBundleInfo"></a>

## IosImessageApplicationBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "IosImessageApplicationBundleInfo")

IosImessageApplicationBundleInfo(<a href="#IosImessageApplicationBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is an iOS iMessage application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS iMessage application
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an iOS iMessage application should use this provider to describe
that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="IosImessageApplicationBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="IosImessageExtensionBundleInfo"></a>

## IosImessageExtensionBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "IosImessageExtensionBundleInfo")

IosImessageExtensionBundleInfo(<a href="#IosImessageExtensionBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is an iOS iMessage extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS iMessage extension
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an iOS iMessage extension should use this provider to describe
that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="IosImessageExtensionBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="IosStaticFrameworkBundleInfo"></a>

## IosStaticFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "IosStaticFrameworkBundleInfo")

IosStaticFrameworkBundleInfo(<a href="#IosStaticFrameworkBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is an iOS static framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS static framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an iOS static framework should use this provider to describe
that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="IosStaticFrameworkBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="IosStickerPackExtensionBundleInfo"></a>

## IosStickerPackExtensionBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "IosStickerPackExtensionBundleInfo")

IosStickerPackExtensionBundleInfo()
</pre>

Denotes that a target is an iOS Sticker Pack extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS Sticker Pack extension
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an iOS Sticker Pack extension should use this provider to describe
that requirement.


<a id="IosXcTestBundleInfo"></a>

## IosXcTestBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "IosXcTestBundleInfo")

IosXcTestBundleInfo(<a href="#IosXcTestBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes a target that is an iOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an iOS .xctest bundle (and
not some other Apple bundle). Rule authors who wish to require that a dependency
is an iOS .xctest bundle should use this provider to describe that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="IosXcTestBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="MacosApplicationBundleInfo"></a>

## MacosApplicationBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosApplicationBundleInfo")

MacosApplicationBundleInfo(<a href="#MacosApplicationBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a macOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS application bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS application should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="MacosApplicationBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="MacosBundleBundleInfo"></a>

## MacosBundleBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosBundleBundleInfo")

MacosBundleBundleInfo(<a href="#MacosBundleBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a macOS loadable bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS loadable bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS loadable bundle should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="MacosBundleBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="MacosExtensionBundleInfo"></a>

## MacosExtensionBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosExtensionBundleInfo")

MacosExtensionBundleInfo(<a href="#MacosExtensionBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a macOS application extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS application
extension bundle (and not some other Apple bundle). Rule authors who wish to
require that a dependency is a macOS application extension should use this
provider to describe that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="MacosExtensionBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="MacosFrameworkBundleInfo"></a>

## MacosFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosFrameworkBundleInfo")

MacosFrameworkBundleInfo()
</pre>

Denotes that a target is an macOS dynamic framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an macOS dynamic framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an macOS dynamic framework should use this provider to describe
that requirement.


<a id="MacosKernelExtensionBundleInfo"></a>

## MacosKernelExtensionBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosKernelExtensionBundleInfo")

MacosKernelExtensionBundleInfo(<a href="#MacosKernelExtensionBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a macOS kernel extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS kernel extension
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS kernel extension should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="MacosKernelExtensionBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="MacosQuickLookPluginBundleInfo"></a>

## MacosQuickLookPluginBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosQuickLookPluginBundleInfo")

MacosQuickLookPluginBundleInfo(<a href="#MacosQuickLookPluginBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a macOS Quick Look Generator bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS Quick Look generator
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a macOS Quick Look generator should use this provider to describe
that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="MacosQuickLookPluginBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="MacosSpotlightImporterBundleInfo"></a>

## MacosSpotlightImporterBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosSpotlightImporterBundleInfo")

MacosSpotlightImporterBundleInfo(<a href="#MacosSpotlightImporterBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a macOS Spotlight Importer bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS Spotlight importer
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS Spotlight importer should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="MacosSpotlightImporterBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="MacosStaticFrameworkBundleInfo"></a>

## MacosStaticFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosStaticFrameworkBundleInfo")

MacosStaticFrameworkBundleInfo()
</pre>

Denotes that a target is an macOS static framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically an macOS static framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is an macOS static framework should use this provider to describe
that requirement.


<a id="MacosXPCServiceBundleInfo"></a>

## MacosXPCServiceBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosXPCServiceBundleInfo")

MacosXPCServiceBundleInfo(<a href="#MacosXPCServiceBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a macOS XPC Service bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS XPC service
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS XPC service should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="MacosXPCServiceBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="MacosXcTestBundleInfo"></a>

## MacosXcTestBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "MacosXcTestBundleInfo")

MacosXcTestBundleInfo(<a href="#MacosXcTestBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes a target that is a macOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a macOS .xctest bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a macOS .xctest bundle should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="MacosXcTestBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="TvosApplicationBundleInfo"></a>

## TvosApplicationBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "TvosApplicationBundleInfo")

TvosApplicationBundleInfo(<a href="#TvosApplicationBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a tvOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS application bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a tvOS application should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="TvosApplicationBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="TvosExtensionBundleInfo"></a>

## TvosExtensionBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "TvosExtensionBundleInfo")

TvosExtensionBundleInfo(<a href="#TvosExtensionBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a tvOS application extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS application
extension bundle (and not some other Apple bundle). Rule authors who wish to
require that a dependency is a tvOS application extension should use this
provider to describe that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="TvosExtensionBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="TvosFrameworkBundleInfo"></a>

## TvosFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "TvosFrameworkBundleInfo")

TvosFrameworkBundleInfo(<a href="#TvosFrameworkBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a tvOS dynamic framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS dynamic framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a tvOS dynamic framework should use this provider to describe
that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="TvosFrameworkBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="TvosStaticFrameworkBundleInfo"></a>

## TvosStaticFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "TvosStaticFrameworkBundleInfo")

TvosStaticFrameworkBundleInfo(<a href="#TvosStaticFrameworkBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a tvOS static framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS static framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a tvOS static framework should use this provider to describe
that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="TvosStaticFrameworkBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="TvosXcTestBundleInfo"></a>

## TvosXcTestBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "TvosXcTestBundleInfo")

TvosXcTestBundleInfo(<a href="#TvosXcTestBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes a target that is a tvOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a tvOS .xctest bundle (and
not some other Apple bundle). Rule authors who wish to require that a dependency
is a tvOS .xctest bundle should use this provider to describe that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="TvosXcTestBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="VisionosApplicationBundleInfo"></a>

## VisionosApplicationBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "VisionosApplicationBundleInfo")

VisionosApplicationBundleInfo(<a href="#VisionosApplicationBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a visionOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a visionOS application
bundle (and not some other Apple bundle). Rule authors who wish to require that a
dependency is a visionOS application should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="VisionosApplicationBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="VisionosExtensionBundleInfo"></a>

## VisionosExtensionBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "VisionosExtensionBundleInfo")

VisionosExtensionBundleInfo(<a href="#VisionosExtensionBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a visionOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a visionOS application
bundle (and not some other Apple bundle). Rule authors who wish to require that a
dependency is a visionOS application should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="VisionosExtensionBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="VisionosFrameworkBundleInfo"></a>

## VisionosFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "VisionosFrameworkBundleInfo")

VisionosFrameworkBundleInfo(<a href="#VisionosFrameworkBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is visionOS dynamic framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a visionOS dynamic framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a visionOS dynamic framework should use this provider to describe
that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="VisionosFrameworkBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="VisionosXcTestBundleInfo"></a>

## VisionosXcTestBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "VisionosXcTestBundleInfo")

VisionosXcTestBundleInfo(<a href="#VisionosXcTestBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes a target that is a visionOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a visionOS .xctest bundle
(and not some other Apple bundle). Rule authors who wish to require that a
dependency is a visionOS .xctest bundle  should use this provider to describe
that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="VisionosXcTestBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="WatchosApplicationBundleInfo"></a>

## WatchosApplicationBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "WatchosApplicationBundleInfo")

WatchosApplicationBundleInfo(<a href="#WatchosApplicationBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a watchOS application.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a watchOS application
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a watchOS application should use this provider to describe that
requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="WatchosApplicationBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="WatchosExtensionBundleInfo"></a>

## WatchosExtensionBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "WatchosExtensionBundleInfo")

WatchosExtensionBundleInfo(<a href="#WatchosExtensionBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes that a target is a watchOS application extension.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a watchOS application
extension bundle (and not some other Apple bundle). Rule authors who wish to
require that a dependency is a watchOS application extension should use this
provider to describe that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="WatchosExtensionBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="WatchosFrameworkBundleInfo"></a>

## WatchosFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "WatchosFrameworkBundleInfo")

WatchosFrameworkBundleInfo()
</pre>

Denotes that a target is watchOS dynamic framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a watchOS dynamic framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a watchOS dynamic framework should use this provider to describe
that requirement.


<a id="WatchosStaticFrameworkBundleInfo"></a>

## WatchosStaticFrameworkBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "WatchosStaticFrameworkBundleInfo")

WatchosStaticFrameworkBundleInfo()
</pre>

Denotes that a target is an watchOS static framework.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a watchOS static framework
bundle (and not some other Apple bundle). Rule authors who wish to require that
a dependency is a watchOS static framework should use this provider to describe
that requirement.


<a id="WatchosXcTestBundleInfo"></a>

## WatchosXcTestBundleInfo

<pre>
load("@rules_apple//apple:providers.bzl", "WatchosXcTestBundleInfo")

WatchosXcTestBundleInfo(<a href="#WatchosXcTestBundleInfo-_init-kwargs">*kwargs</a>)
</pre>

Denotes a target that is a watchOS .xctest bundle.

This provider does not contain any fields of its own at this time but is used as
a "marker" to indicate that a target is specifically a watchOS .xctest bundle (and
not some other Apple bundle). Rule authors who wish to require that a dependency
is a watchOS .xctest bundle should use this provider to describe that requirement.

**CONSTRUCTOR PARAMETERS**

| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="WatchosXcTestBundleInfo-_init-kwargs"></a>kwargs | <p align="center">-</p> | none |


<a id="apple_provider.make_apple_bundle_version_info"></a>

## apple_provider.make_apple_bundle_version_info

<pre>
load("@rules_apple//apple:providers.bzl", "apple_provider")

apple_provider.make_apple_bundle_version_info(*, <a href="#apple_provider.make_apple_bundle_version_info-version_file">version_file</a>)
</pre>

Creates a new instance of the `AppleBundleVersionInfo` provider.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="apple_provider.make_apple_bundle_version_info-version_file"></a>version_file |  Required. See the docs on `AppleBundleVersionInfo`.   |  none |

**RETURNS**

A new `AppleBundleVersionInfo` provider based on the supplied arguments.


<a id="apple_provider.make_apple_test_runner_info"></a>

## apple_provider.make_apple_test_runner_info

<pre>
load("@rules_apple//apple:providers.bzl", "apple_provider")

apple_provider.make_apple_test_runner_info(<a href="#apple_provider.make_apple_test_runner_info-kwargs">**kwargs</a>)
</pre>

Creates a new instance of the AppleTestRunnerInfo provider.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="apple_provider.make_apple_test_runner_info-kwargs"></a>kwargs |  A set of keyword arguments expected to match the fields of `AppleTestRunnerInfo`. See the documentation for `AppleTestRunnerInfo` for what these must be.   |  none |

**RETURNS**

A new `AppleTestRunnerInfo` provider based on the supplied arguments.


<a id="apple_provider.merge_apple_framework_import_info"></a>

## apple_provider.merge_apple_framework_import_info

<pre>
load("@rules_apple//apple:providers.bzl", "apple_provider")

apple_provider.merge_apple_framework_import_info(<a href="#apple_provider.merge_apple_framework_import_info-apple_framework_import_infos">apple_framework_import_infos</a>)
</pre>

Merges multiple `AppleFrameworkImportInfo` into one.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="apple_provider.merge_apple_framework_import_info-apple_framework_import_infos"></a>apple_framework_import_infos |  List of `AppleFrameworkImportInfo` to be merged.   |  none |

**RETURNS**

Result of merging all the received framework infos.


