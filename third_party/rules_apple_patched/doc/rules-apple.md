<!-- Generated with Stardoc: http://skydoc.bazel.build -->

# Rules that apply to all Apple platforms.

<a id="apple_dynamic_framework_import"></a>

## apple_dynamic_framework_import

<pre>
load("@rules_apple//apple:apple.bzl", "apple_dynamic_framework_import")

apple_dynamic_framework_import(<a href="#apple_dynamic_framework_import-name">name</a>, <a href="#apple_dynamic_framework_import-deps">deps</a>, <a href="#apple_dynamic_framework_import-bundle_only">bundle_only</a>, <a href="#apple_dynamic_framework_import-dsym_imports">dsym_imports</a>, <a href="#apple_dynamic_framework_import-framework_imports">framework_imports</a>)
</pre>

This rule encapsulates an already-built dynamic framework. It is defined by a list of
files in exactly one `.framework` directory. `apple_dynamic_framework_import` targets
need to be added to library targets through the `deps` attribute.
### Examples

```python
apple_dynamic_framework_import(
    name = "my_dynamic_framework",
    framework_imports = glob(["my_dynamic_framework.framework/**"]),
)

objc_library(
    name = "foo_lib",
    ...,
    deps = [
        ":my_dynamic_framework",
    ],
)
```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_dynamic_framework_import-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_dynamic_framework_import-deps"></a>deps |  A list of targets that are dependencies of the target being built, which will be linked into that target.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_dynamic_framework_import-bundle_only"></a>bundle_only |  Avoid linking the dynamic framework, but still include it in the app. This is useful when you want to manually dlopen the framework at runtime.   | Boolean | optional |  `False`  |
| <a id="apple_dynamic_framework_import-dsym_imports"></a>dsym_imports |  The list of files under a .dSYM directory, that is the imported framework's dSYM bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_dynamic_framework_import-framework_imports"></a>framework_imports |  The list of files under a .framework directory which are provided to Apple based targets that depend on this target.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |


<a id="apple_dynamic_xcframework_import"></a>

## apple_dynamic_xcframework_import

<pre>
load("@rules_apple//apple:apple.bzl", "apple_dynamic_xcframework_import")

apple_dynamic_xcframework_import(<a href="#apple_dynamic_xcframework_import-name">name</a>, <a href="#apple_dynamic_xcframework_import-deps">deps</a>, <a href="#apple_dynamic_xcframework_import-bundle_only">bundle_only</a>, <a href="#apple_dynamic_xcframework_import-library_identifiers">library_identifiers</a>, <a href="#apple_dynamic_xcframework_import-xcframework_imports">xcframework_imports</a>)
</pre>

This rule encapsulates an already-built XCFramework. Defined by a list of files in a .xcframework
directory. apple_xcframework_import targets need to be added as dependencies to library targets
through the `deps` attribute.

### Example

```bzl
apple_dynamic_xcframework_import(
    name = "my_dynamic_xcframework",
    xcframework_imports = glob(["my_dynamic_framework.xcframework/**"]),
)

objc_library(
    name = "foo_lib",
    ...,
    deps = [
        ":my_dynamic_xcframework",
    ],
)
```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_dynamic_xcframework_import-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_dynamic_xcframework_import-deps"></a>deps |  List of targets that are dependencies of the target being built, which will provide headers and be linked into that target.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_dynamic_xcframework_import-bundle_only"></a>bundle_only |  Avoid linking the dynamic framework, but still include it in the app. This is useful when you want to manually dlopen the framework at runtime.   | Boolean | optional |  `False`  |
| <a id="apple_dynamic_xcframework_import-library_identifiers"></a>library_identifiers |  Unnecssary and ignored, will be removed in the future.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="apple_dynamic_xcframework_import-xcframework_imports"></a>xcframework_imports |  List of files under a .xcframework directory which are provided to Apple based targets that depend on this target.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |


<a id="apple_static_framework_import"></a>

## apple_static_framework_import

<pre>
load("@rules_apple//apple:apple.bzl", "apple_static_framework_import")

apple_static_framework_import(<a href="#apple_static_framework_import-name">name</a>, <a href="#apple_static_framework_import-deps">deps</a>, <a href="#apple_static_framework_import-data">data</a>, <a href="#apple_static_framework_import-alwayslink">alwayslink</a>, <a href="#apple_static_framework_import-framework_imports">framework_imports</a>, <a href="#apple_static_framework_import-has_swift">has_swift</a>,
                              <a href="#apple_static_framework_import-sdk_dylibs">sdk_dylibs</a>, <a href="#apple_static_framework_import-sdk_frameworks">sdk_frameworks</a>, <a href="#apple_static_framework_import-weak_sdk_frameworks">weak_sdk_frameworks</a>)
</pre>

This rule encapsulates an already-built static framework. It is defined by a list of
files in exactly one `.framework` directory. `apple_static_framework_import` targets
need to be added to library targets through the `deps` attribute.
### Examples

```python
apple_static_framework_import(
    name = "my_static_framework",
    framework_imports = glob(["my_static_framework.framework/**"]),
)

objc_library(
    name = "foo_lib",
    ...,
    deps = [
        ":my_static_framework",
    ],
)
```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_static_framework_import-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_static_framework_import-deps"></a>deps |  A list of targets that are dependencies of the target being built, which will provide headers and be linked into that target.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_framework_import-data"></a>data |  List of files needed by this target at runtime.<br><br>Files and targets named in the `data` attribute will appear in the `*.runfiles` area of this target, if it has one. This may include data files needed by a binary or library, or other programs needed by it.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_framework_import-alwayslink"></a>alwayslink |  If true, any binary that depends (directly or indirectly) on this framework will link in all the object files for the framework file, even if some contain no symbols referenced by the binary. This is useful if your code isn't explicitly called by code in the binary; for example, if you rely on runtime checks for protocol conformances added in extensions in the library but do not directly reference any other symbols in the object file that adds that conformance.   | Boolean | optional |  `False`  |
| <a id="apple_static_framework_import-framework_imports"></a>framework_imports |  The list of files under a .framework directory which are provided to Apple based targets that depend on this target.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="apple_static_framework_import-has_swift"></a>has_swift |  A boolean indicating if the target has Swift source code. This helps flag Apple frameworks that do not include Swift interface or Swift module files.   | Boolean | optional |  `False`  |
| <a id="apple_static_framework_import-sdk_dylibs"></a>sdk_dylibs |  Names of SDK .dylib libraries to link with. For instance, `libz` or `libarchive`. `libc++` is included automatically if the binary has any C++ or Objective-C++ sources in its dependency tree. When linking a binary, all libraries named in that binary's transitive dependency graph are used.   | List of strings | optional |  `[]`  |
| <a id="apple_static_framework_import-sdk_frameworks"></a>sdk_frameworks |  Names of SDK frameworks to link with (e.g. `AddressBook`, `QuartzCore`). `UIKit` and `Foundation` are always included when building for the iOS, tvOS, visionOS, and watchOS platforms. For macOS, only `Foundation` is always included. When linking a top level binary, all SDK frameworks listed in that binary's transitive dependency graph are linked.   | List of strings | optional |  `[]`  |
| <a id="apple_static_framework_import-weak_sdk_frameworks"></a>weak_sdk_frameworks |  Names of SDK frameworks to weakly link with. For instance, `MediaAccessibility`. In difference to regularly linked SDK frameworks, symbols from weakly linked frameworks do not cause an error if they are not present at runtime.   | List of strings | optional |  `[]`  |


<a id="apple_static_library"></a>

## apple_static_library

<pre>
load("@rules_apple//apple:apple.bzl", "apple_static_library")

apple_static_library(<a href="#apple_static_library-name">name</a>, <a href="#apple_static_library-deps">deps</a>, <a href="#apple_static_library-data">data</a>, <a href="#apple_static_library-additional_linker_inputs">additional_linker_inputs</a>, <a href="#apple_static_library-avoid_deps">avoid_deps</a>, <a href="#apple_static_library-linkopts">linkopts</a>,
                     <a href="#apple_static_library-minimum_os_version">minimum_os_version</a>, <a href="#apple_static_library-platform_type">platform_type</a>, <a href="#apple_static_library-sdk_dylibs">sdk_dylibs</a>, <a href="#apple_static_library-sdk_frameworks">sdk_frameworks</a>,
                     <a href="#apple_static_library-weak_sdk_frameworks">weak_sdk_frameworks</a>)
</pre>

This rule produces single- or multi-architecture ("fat") static libraries targeting
Apple platforms.

The `lipo` tool is used to combine files of multiple architectures. One of
several flags may control which architectures are included in the output,
depending on the value of the `platform_type` attribute.

NOTE: In most situations, users should prefer the platform- and
product-type-specific rules, such as `apple_static_xcframework`. This
rule is being provided for the purpose of transitioning users from the built-in
implementation of `apple_static_library` in Bazel core so that it can be removed.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_static_library-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_static_library-deps"></a>deps |  A list of dependencies targets that will be linked into this target's binary. Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_library-data"></a>data |  Files to be made available to the library archive upon execution.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_library-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_library-avoid_deps"></a>avoid_deps |  A list of library targets on which this framework depends in order to compile, but the transitive closure of which will not be linked into the framework's binary.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_library-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="apple_static_library-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="apple_static_library-platform_type"></a>platform_type |  The target Apple platform for which to create a binary. This dictates which SDK is used for compilation/linking and which flag is used to determine the architectures to target. For example, if `ios` is specified, then the output binaries/libraries will be created combining all architectures specified by `--ios_multi_cpus`. Options are:<br><br>*   `ios`: architectures gathered from `--ios_multi_cpus`. *   `macos`: architectures gathered from `--macos_cpus`. *   `tvos`: architectures gathered from `--tvos_cpus`. *   `visionos`: architectures gathered from `--visionos_cpus`. *   `watchos`: architectures gathered from `--watchos_cpus`.   | String | required |  |
| <a id="apple_static_library-sdk_dylibs"></a>sdk_dylibs |  Names of SDK `.dylib` libraries to link with (e.g., `libz` or `libarchive`). `libc++` is included automatically if the binary has any C++ or Objective-C++ sources in its dependency tree. When linking a binary, all libraries named in that binary's transitive dependency graph are used.   | List of strings | optional |  `[]`  |
| <a id="apple_static_library-sdk_frameworks"></a>sdk_frameworks |  Names of SDK frameworks to link with (e.g., `AddressBook`, `QuartzCore`). `UIKit` and `Foundation` are always included, even if this attribute is provided and does not list them.<br><br>This attribute is discouraged; in general, targets should list system framework dependencies in the library targets where that framework is used, not in the top-level bundle.   | List of strings | optional |  `[]`  |
| <a id="apple_static_library-weak_sdk_frameworks"></a>weak_sdk_frameworks |  Names of SDK frameworks to weakly link with (e.g., `MediaAccessibility`). Unlike regularly linked SDK frameworks, symbols from weakly linked frameworks do not cause the binary to fail to load if they are not present in the version of the framework available at runtime.<br><br>This attribute is discouraged; in general, targets should list system framework dependencies in the library targets where that framework is used, not in the top-level bundle.   | List of strings | optional |  `[]`  |


<a id="apple_static_xcframework"></a>

## apple_static_xcframework

<pre>
load("@rules_apple//apple:apple.bzl", "apple_static_xcframework")

apple_static_xcframework(<a href="#apple_static_xcframework-name">name</a>, <a href="#apple_static_xcframework-deps">deps</a>, <a href="#apple_static_xcframework-avoid_deps">avoid_deps</a>, <a href="#apple_static_xcframework-bundle_id">bundle_id</a>, <a href="#apple_static_xcframework-bundle_name">bundle_name</a>, <a href="#apple_static_xcframework-executable_name">executable_name</a>,
                         <a href="#apple_static_xcframework-families_required">families_required</a>, <a href="#apple_static_xcframework-infoplists">infoplists</a>, <a href="#apple_static_xcframework-ios">ios</a>, <a href="#apple_static_xcframework-macos">macos</a>, <a href="#apple_static_xcframework-minimum_deployment_os_versions">minimum_deployment_os_versions</a>,
                         <a href="#apple_static_xcframework-minimum_os_versions">minimum_os_versions</a>, <a href="#apple_static_xcframework-public_hdrs">public_hdrs</a>, <a href="#apple_static_xcframework-umbrella_header">umbrella_header</a>, <a href="#apple_static_xcframework-version">version</a>)
</pre>

Generates an XCFramework with static libraries for third-party distribution.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_static_xcframework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_static_xcframework-deps"></a>deps |  A list of files directly referencing libraries to be represented for each given platform split in the XCFramework. These libraries will be embedded within each platform split.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="apple_static_xcframework-avoid_deps"></a>avoid_deps |  A list of library targets on which this framework depends in order to compile, but the transitive closure of which will not be linked into the framework's binary, nor bundled into final XCFramework.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_xcframework-bundle_id"></a>bundle_id |  Optional bundle ID (reverse-DNS path followed by framework name) for each of the embedded frameworks. If present, this value will be embedded in an Info.plist within each framework bundle, similar to apple_xcframework (dynamic frameworks).   | String | optional |  `""`  |
| <a id="apple_static_xcframework-bundle_name"></a>bundle_name |  The desired name of the XCFramework bundle (without the extension) and the binaries for all embedded static libraries. If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="apple_static_xcframework-executable_name"></a>executable_name |  The desired name of the executable, if the bundle has an executable. If this attribute is not set, then the name of the `bundle_name` attribute will be used if it is set; if not, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="apple_static_xcframework-families_required"></a>families_required |  A list of device families supported by this extension, with platforms such as `ios` as keys. Valid values are `iphone` and `ipad` for `ios`; at least one must be specified if a platform is defined. Currently, this only affects processing of `ios` resources.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> List of strings</a> | optional |  `{}`  |
| <a id="apple_static_xcframework-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for each of the embedded frameworks. Only used if bundle_id is provided.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_xcframework-ios"></a>ios |  A dictionary of strings indicating which platform variants should be built for the `ios` platform ( `device` or `simulator`) as keys, and arrays of strings listing which architectures should be built for those platform variants (for example, `x86_64`, `arm64`) as their values.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> List of strings</a> | optional |  `{}`  |
| <a id="apple_static_xcframework-macos"></a>macos |  A list of strings indicating which architecture should be built for the macOS platform (for example, `x86_64`, `arm64`).   | List of strings | optional |  `[]`  |
| <a id="apple_static_xcframework-minimum_deployment_os_versions"></a>minimum_deployment_os_versions |  A dictionary of strings indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0") as values, with their respective platforms such as `ios` as keys. This is different from `minimum_os_versions`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="apple_static_xcframework-minimum_os_versions"></a>minimum_os_versions |  A dictionary of strings indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "8.0") as values, with their respective platforms such as `ios` as keys.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | required |  |
| <a id="apple_static_xcframework-public_hdrs"></a>public_hdrs |  A list of files directly referencing header files to be used as the publicly visible interface for each of these embedded libraries. These header files will be embedded within each platform split, typically in a subdirectory such as `Headers`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_xcframework-umbrella_header"></a>umbrella_header |  An optional single .h file to use as the umbrella header for this framework. Usually, this header will have the same name as this target, so that clients can load the header using the #import <MyFramework/MyFramework.h> format. If this attribute is not specified (the common use case), an umbrella header will be generated under the same name as this target.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="apple_static_xcframework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-versioning.md#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="apple_static_xcframework_import"></a>

## apple_static_xcframework_import

<pre>
load("@rules_apple//apple:apple.bzl", "apple_static_xcframework_import")

apple_static_xcframework_import(<a href="#apple_static_xcframework_import-name">name</a>, <a href="#apple_static_xcframework_import-deps">deps</a>, <a href="#apple_static_xcframework_import-data">data</a>, <a href="#apple_static_xcframework_import-alwayslink">alwayslink</a>, <a href="#apple_static_xcframework_import-has_swift">has_swift</a>, <a href="#apple_static_xcframework_import-includes">includes</a>,
                                <a href="#apple_static_xcframework_import-library_identifiers">library_identifiers</a>, <a href="#apple_static_xcframework_import-linkopts">linkopts</a>, <a href="#apple_static_xcframework_import-sdk_dylibs">sdk_dylibs</a>, <a href="#apple_static_xcframework_import-sdk_frameworks">sdk_frameworks</a>,
                                <a href="#apple_static_xcframework_import-weak_sdk_frameworks">weak_sdk_frameworks</a>, <a href="#apple_static_xcframework_import-xcframework_imports">xcframework_imports</a>)
</pre>

This rule encapsulates an already-built XCFramework with static libraries. Defined by a list of
files in a .xcframework directory. apple_xcframework_import targets need to be added as dependencies
to library targets through the `deps` attribute.
### Examples

```slarlark
apple_static_xcframework_import(
    name = "my_static_xcframework",
    xcframework_imports = glob(["my_static_framework.xcframework/**"]),
)

objc_library(
    name = "foo_lib",
    ...,
    deps = [
        ":my_static_xcframework",
    ],
)
```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_static_xcframework_import-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_static_xcframework_import-deps"></a>deps |  List of targets that are dependencies of the target being built, which will provide headers and be linked into that target.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_xcframework_import-data"></a>data |  List of files needed by this target at runtime.<br><br>Files and targets named in the `data` attribute will appear in the `*.runfiles` area of this target, if it has one. This may include data files needed by a binary or library, or other programs needed by it.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_static_xcframework_import-alwayslink"></a>alwayslink |  If true, any binary that depends (directly or indirectly) on this XCFramework will link in all the object files for the XCFramework bundle, even if some contain no symbols referenced by the binary. This is useful if your code isn't explicitly called by code in the binary; for example, if you rely on runtime checks for protocol conformances added in extensions in the library but do not directly reference any other symbols in the object file that adds that conformance.   | Boolean | optional |  `False`  |
| <a id="apple_static_xcframework_import-has_swift"></a>has_swift |  A boolean indicating if the target has Swift source code. This helps flag XCFrameworks that do not include Swift interface files.   | Boolean | optional |  `False`  |
| <a id="apple_static_xcframework_import-includes"></a>includes |  List of `#include/#import` search paths to add to this target and all depending targets.<br><br>The paths are interpreted relative to the single platform directory inside the XCFramework for the platform being built.<br><br>These flags are added for this rule and every rule that depends on it. (Note: not the rules it depends upon!) Be very careful, since this may have far-reaching effects.   | List of strings | optional |  `[]`  |
| <a id="apple_static_xcframework_import-library_identifiers"></a>library_identifiers |  Unnecssary and ignored, will be removed in the future.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="apple_static_xcframework_import-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="apple_static_xcframework_import-sdk_dylibs"></a>sdk_dylibs |  Names of SDK .dylib libraries to link with. For instance, `libz` or `libarchive`. `libc++` is included automatically if the binary has any C++ or Objective-C++ sources in its dependency tree. When linking a binary, all libraries named in that binary's transitive dependency graph are used.   | List of strings | optional |  `[]`  |
| <a id="apple_static_xcframework_import-sdk_frameworks"></a>sdk_frameworks |  Names of SDK frameworks to link with (e.g. `AddressBook`, `QuartzCore`). `UIKit` and `Foundation` are always included when building for the iOS, tvOS, visionOS and watchOS platforms. For macOS, only `Foundation` is always included. When linking a top level binary, all SDK frameworks listed in that binary's transitive dependency graph are linked.   | List of strings | optional |  `[]`  |
| <a id="apple_static_xcframework_import-weak_sdk_frameworks"></a>weak_sdk_frameworks |  Names of SDK frameworks to weakly link with. For instance, `MediaAccessibility`. In difference to regularly linked SDK frameworks, symbols from weakly linked frameworks do not cause an error if they are not present at runtime.   | List of strings | optional |  `[]`  |
| <a id="apple_static_xcframework_import-xcframework_imports"></a>xcframework_imports |  List of files under a .xcframework directory which are provided to Apple based targets that depend on this target.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |


<a id="apple_universal_binary"></a>

## apple_universal_binary

<pre>
load("@rules_apple//apple:apple.bzl", "apple_universal_binary")

apple_universal_binary(<a href="#apple_universal_binary-name">name</a>, <a href="#apple_universal_binary-binary">binary</a>, <a href="#apple_universal_binary-forced_cpus">forced_cpus</a>, <a href="#apple_universal_binary-minimum_deployment_os_version">minimum_deployment_os_version</a>, <a href="#apple_universal_binary-minimum_os_version">minimum_os_version</a>,
                       <a href="#apple_universal_binary-platform_type">platform_type</a>)
</pre>

This rule produces a multi-architecture ("fat") binary targeting Apple platforms.
The `lipo` tool is used to combine built binaries of multiple architectures.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_universal_binary-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_universal_binary-binary"></a>binary |  Target to generate a 'fat' binary from.   | <a href="https://bazel.build/concepts/labels">Label</a> | required |  |
| <a id="apple_universal_binary-forced_cpus"></a>forced_cpus |  An optional list of target CPUs for which the universal binary should be built.<br><br>If this attribute is present, the value of the platform-specific CPU flag (`--ios_multi_cpus`, `--macos_cpus`, `--tvos_cpus`, `--visionos_cpus`, or `--watchos_cpus`) will be ignored and the binary will be built for all of the specified architectures instead.<br><br>This is primarily useful to force macOS tools to be built as universal binaries using `forced_cpus = ["x86_64", "arm64"]`, without requiring the user to pass additional flags when invoking Bazel.   | List of strings | optional |  `[]`  |
| <a id="apple_universal_binary-minimum_deployment_os_version"></a>minimum_deployment_os_version |  A required string indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0"). This is different from `minimum_os_version`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | String | optional |  `""`  |
| <a id="apple_universal_binary-minimum_os_version"></a>minimum_os_version |  A required string indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "9.0").   | String | required |  |
| <a id="apple_universal_binary-platform_type"></a>platform_type |  The target Apple platform for which to create a binary. This dictates which SDK is used for compilation/linking and which flag is used to determine the architectures to target. For example, if `ios` is specified, then the output binaries/libraries will be created combining all architectures specified by `--ios_multi_cpus`. Options are:<br><br>*   `ios`: architectures gathered from `--ios_multi_cpus`. *   `macos`: architectures gathered from `--macos_cpus`. *   `tvos`: architectures gathered from `--tvos_cpus`. *   `watchos`: architectures gathered from `--watchos_cpus`.   | String | required |  |


<a id="apple_xcframework"></a>

## apple_xcframework

<pre>
load("@rules_apple//apple:apple.bzl", "apple_xcframework")

apple_xcframework(<a href="#apple_xcframework-name">name</a>, <a href="#apple_xcframework-deps">deps</a>, <a href="#apple_xcframework-data">data</a>, <a href="#apple_xcframework-additional_linker_inputs">additional_linker_inputs</a>, <a href="#apple_xcframework-bundle_id">bundle_id</a>, <a href="#apple_xcframework-bundle_name">bundle_name</a>,
                  <a href="#apple_xcframework-codesign_inputs">codesign_inputs</a>, <a href="#apple_xcframework-codesignopts">codesignopts</a>, <a href="#apple_xcframework-exported_symbols_lists">exported_symbols_lists</a>, <a href="#apple_xcframework-extension_safe">extension_safe</a>,
                  <a href="#apple_xcframework-families_required">families_required</a>, <a href="#apple_xcframework-infoplists">infoplists</a>, <a href="#apple_xcframework-ios">ios</a>, <a href="#apple_xcframework-linkopts">linkopts</a>, <a href="#apple_xcframework-macos">macos</a>, <a href="#apple_xcframework-minimum_deployment_os_versions">minimum_deployment_os_versions</a>,
                  <a href="#apple_xcframework-minimum_os_versions">minimum_os_versions</a>, <a href="#apple_xcframework-public_hdrs">public_hdrs</a>, <a href="#apple_xcframework-stamp">stamp</a>, <a href="#apple_xcframework-tvos">tvos</a>, <a href="#apple_xcframework-umbrella_header">umbrella_header</a>, <a href="#apple_xcframework-version">version</a>)
</pre>

Builds and bundles an XCFramework for third-party distribution.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_xcframework-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_xcframework-deps"></a>deps |  A list of dependent targets that will be linked into this target's binary(s). Any resources, such as asset catalogs, that are referenced by those targets will also be transitively included in the final bundle(s).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_xcframework-data"></a>data |  A list of resources or files bundled with the bundle. The resources will be stored in the appropriate resources location within each of the embedded framework bundles.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_xcframework-additional_linker_inputs"></a>additional_linker_inputs |  A list of input files to be passed to the linker.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_xcframework-bundle_id"></a>bundle_id |  The bundle ID (reverse-DNS path followed by app name) for each of the embedded frameworks. If present, this value will be embedded in an Info.plist within each framework bundle.   | String | required |  |
| <a id="apple_xcframework-bundle_name"></a>bundle_name |  The desired name of the xcframework bundle (without the extension) and the bundles for all embedded frameworks. If this attribute is not set, then the name of the target will be used instead.   | String | optional |  `""`  |
| <a id="apple_xcframework-codesign_inputs"></a>codesign_inputs |  A list of dependencies targets that provide inputs that will be used by `codesign` (referenced with `codesignopts`).   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_xcframework-codesignopts"></a>codesignopts |  A list of strings representing extra flags that should be passed to `codesign`.   | List of strings | optional |  `[]`  |
| <a id="apple_xcframework-exported_symbols_lists"></a>exported_symbols_lists |  A list of targets containing exported symbols lists files for the linker to control symbol resolution.<br><br>Each file is expected to have a list of global symbol names that will remain as global symbols in the compiled binary owned by this framework. All other global symbols will be treated as if they were marked as `__private_extern__` (aka `visibility=hidden`) and will not be global in the output file.<br><br>See the man page documentation for `ld(1)` on macOS for more details.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_xcframework-extension_safe"></a>extension_safe |  If true, compiles and links this framework with `-application-extension`, restricting the binary to use only extension-safe APIs.   | Boolean | optional |  `False`  |
| <a id="apple_xcframework-families_required"></a>families_required |  A list of device families supported by this extension, with platforms such as `ios` as keys. Valid values are `iphone` and `ipad` for `ios`; at least one must be specified if a platform is defined. Currently, this only affects processing of `ios` resources.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> List of strings</a> | optional |  `{}`  |
| <a id="apple_xcframework-infoplists"></a>infoplists |  A list of .plist files that will be merged to form the Info.plist for each of the embedded frameworks. At least one file must be specified. Please see [Info.plist Handling](https://github.com/bazelbuild/rules_apple/blob/main/doc/common_info.md#infoplist-handling) for what is supported.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="apple_xcframework-ios"></a>ios |  A dictionary of strings indicating which platform variants should be built for the iOS platform ( `device` or `simulator`) as keys, and arrays of strings listing which architectures should be built for those platform variants (for example, `x86_64`, `arm64`) as their values.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> List of strings</a> | optional |  `{}`  |
| <a id="apple_xcframework-linkopts"></a>linkopts |  A list of strings representing extra flags that should be passed to the linker.   | List of strings | optional |  `[]`  |
| <a id="apple_xcframework-macos"></a>macos |  A list of strings indicating which architecture should be built for the macOS platform (for example, `x86_64`, `arm64`).   | List of strings | optional |  `[]`  |
| <a id="apple_xcframework-minimum_deployment_os_versions"></a>minimum_deployment_os_versions |  A dictionary of strings indicating the minimum deployment OS version supported by the target, represented as a dotted version number (for example, "9.0") as values, with their respective platforms such as `ios` as keys. This is different from `minimum_os_versions`, which is effective at compile time. Ensure version specific APIs are guarded with `available` clauses.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="apple_xcframework-minimum_os_versions"></a>minimum_os_versions |  A dictionary of strings indicating the minimum OS version supported by the target, represented as a dotted version number (for example, "8.0") as values, with their respective platforms such as `ios`, or `tvos` as keys:<br><br>    minimum_os_versions = {         "ios": "13.0",         "tvos": "15.0",     }   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | required |  |
| <a id="apple_xcframework-public_hdrs"></a>public_hdrs |  A list of files directly referencing header files to be used as the publicly visible interface for each of these embedded frameworks. These header files will be embedded within each bundle, typically in a subdirectory such as `Headers`.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_xcframework-stamp"></a>stamp |  Enable link stamping. Whether to encode build information into the binary. Possible values:<br><br>*   `stamp = 1`: Stamp the build information into the binary. Stamped binaries are only rebuilt     when their dependencies change. Use this if there are tests that depend on the build     information. *   `stamp = 0`: Always replace build information by constant values. This gives good build     result caching. *   `stamp = -1`: Embedding of build information is controlled by the `--[no]stamp` flag.   | Integer | optional |  `-1`  |
| <a id="apple_xcframework-tvos"></a>tvos |  A dictionary of strings indicating which platform variants should be built for the tvOS platform ( `device` or `simulator`) as keys, and arrays of strings listing which architectures should be built for those platform variants (for example, `x86_64`, `arm64`) as their values.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> List of strings</a> | optional |  `{}`  |
| <a id="apple_xcframework-umbrella_header"></a>umbrella_header |  An optional single .h file to use as the umbrella header for this framework. Usually, this header will have the same name as this target, so that clients can load the header using the #import <MyFramework/MyFramework.h> format. If this attribute is not specified (the common use case), an umbrella header will be generated under the same name as this target.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |
| <a id="apple_xcframework-version"></a>version |  An `apple_bundle_version` target that represents the version for this target. See [`apple_bundle_version`](https://github.com/bazelbuild/rules_apple/blob/main/doc/rules-versioning.md#apple_bundle_version).   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


<a id="local_provisioning_profile"></a>

## local_provisioning_profile

<pre>
load("@rules_apple//apple:apple.bzl", "local_provisioning_profile")

local_provisioning_profile(<a href="#local_provisioning_profile-name">name</a>, <a href="#local_provisioning_profile-profile_extension">profile_extension</a>, <a href="#local_provisioning_profile-profile_name">profile_name</a>, <a href="#local_provisioning_profile-team_id">team_id</a>)
</pre>

This rule declares a bazel target that you can pass to the
`provisioning_profile` attribute of rules that support it. It discovers a
provisioning profile for the given attributes either on the user's local
machine, or with the optional `fallback_profiles` passed to
`provisioning_profile_repository`.

This rule will automatically pick the newest
profile if there are multiple profiles matching the given criteria.

By default this rule will search for a `.mobileprovision` file with the same name
as the rule itself, you can pass `profile_name` to use a different name, you
can pass `team_id` if you'd like to disambiguate between 2 Apple developer accounts
that have the same profile name. You may also pass `.provisionprofile` to
`profile_extension` to search for a macOS provisioning profile instead.

## Example

```starlark
load("//apple:apple.bzl", "local_provisioning_profile")

local_provisioning_profile(
    name = "app_debug_profile",
    profile_name = "Development App",
    team_id = "abc123",
)

ios_application(
    name = "app",
    ...
    provisioning_profile = ":app_debug_profile",
)

local_provisioning_profile(
    name = "app_release_profile",
)

ios_application(
    name = "release_app",
    ...
    provisioning_profile = ":app_release_profile",
)
```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="local_provisioning_profile-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="local_provisioning_profile-profile_extension"></a>profile_extension |  The extension for the provisioning profile which may differ by platform.   | String | optional |  `".mobileprovision"`  |
| <a id="local_provisioning_profile-profile_name"></a>profile_name |  Name of the profile to use, if it's not provided the name of the rule is used   | String | optional |  `""`  |
| <a id="local_provisioning_profile-team_id"></a>team_id |  Team ID of the profile to find. This is useful for disambiguating between multiple profiles with the same name on different developer accounts.   | String | optional |  `""`  |


<a id="provisioning_profile_repository"></a>

## provisioning_profile_repository

<pre>
load("@rules_apple//apple:apple.bzl", "provisioning_profile_repository")

provisioning_profile_repository(<a href="#provisioning_profile_repository-name">name</a>, <a href="#provisioning_profile_repository-fallback_profiles">fallback_profiles</a>)
</pre>

This rule declares an external repository for discovering locally installed
provisioning profiles. This is consumed by `local_provisioning_profile`.
You can optionally set 'fallback_profiles' to point at a stable location of
profiles if a newer version of the desired profile does not exist on the local
machine. This is useful for checking in the current version of the profile, but
not having to update it every time a new device or certificate is added.

## Example

### In your `MODULE.bazel` file:

You only need this in the case you want to setup fallback profiles, otherwise
it can be ommitted when using bzlmod.

```bzl
provisioning_profile_repository = use_extension("@build_bazel_rules_apple//apple:apple.bzl", "provisioning_profile_repository_extension")
provisioning_profile_repository.setup(
    fallback_profiles = "//path/to/some:filegroup", # Profiles to use if one isn't found locally
)
```

### In your `WORKSPACE` file:

```starlark
load("//apple:apple.bzl", "provisioning_profile_repository")

provisioning_profile_repository(
    name = "local_provisioning_profiles",
    fallback_profiles = "//path/to/some:filegroup", # Optional profiles to use if one isn't found locally
)
```

### In your `BUILD` files (see `local_provisioning_profile` for more examples):

```starlark
load("//apple:apple.bzl", "local_provisioning_profile")

local_provisioning_profile(
    name = "app_debug_profile",
    profile_name = "Development App",
    team_id = "abc123",
)

ios_application(
    name = "app",
    ...
    provisioning_profile = ":app_debug_profile",
)
```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="provisioning_profile_repository-name"></a>name |  A unique name for this repository.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="provisioning_profile_repository-fallback_profiles"></a>fallback_profiles |  -   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |

**ENVIRONMENT VARIABLES**

This repository rule depends on the following environment variables:

* `HOME`


<a id="provisioning_profile_repository_extension"></a>

## provisioning_profile_repository_extension

<pre>
provisioning_profile_repository_extension = use_extension("@rules_apple//apple:apple.bzl", "provisioning_profile_repository_extension")
provisioning_profile_repository_extension.setup(<a href="#provisioning_profile_repository_extension.setup-fallback_profiles">fallback_profiles</a>)
</pre>

See [`provisioning_profile_repository`](#provisioning_profile_repository) for more information and examples.


**TAG CLASSES**

<a id="provisioning_profile_repository_extension.setup"></a>

### setup

**Attributes**

| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="provisioning_profile_repository_extension.setup-fallback_profiles"></a>fallback_profiles |  -   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


