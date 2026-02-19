<!-- Generated with Stardoc: http://skydoc.bazel.build -->

# Rules related to Apple bundle versioning.

<a id="apple_bundle_version"></a>

## apple_bundle_version

<pre>
load("@rules_apple//apple:versioning.bzl", "apple_bundle_version")

apple_bundle_version(<a href="#apple_bundle_version-name">name</a>, <a href="#apple_bundle_version-build_label_pattern">build_label_pattern</a>, <a href="#apple_bundle_version-build_version">build_version</a>, <a href="#apple_bundle_version-capture_groups">capture_groups</a>, <a href="#apple_bundle_version-fallback_build_label">fallback_build_label</a>,
                     <a href="#apple_bundle_version-short_version_string">short_version_string</a>)
</pre>

Produces a target that contains versioning information for an Apple bundle.

This rule allows version numbers to be hard-coded into the BUILD file or
extracted from the build label passed into Bazel using the `--embed_label`
command line flag.

Targets created by this rule do not generate outputs themselves, but instead
should be used in the `version` attribute of an Apple application or extension
bundle target to set the version keys in that bundle's Info.plist file.

### Examples

```python
# A version scheme that uses hard-coded versions checked into your
# BUILD files.
apple_bundle_version(
    name = "simple",
    build_version = "1.0.134",
    short_version_string = "1.0",
)

ios_application(
    name = "foo_app",
    ...,
    version = ":simple",
)

# A version scheme that parses version information out of the build
# label and uses a fallback for developers' builds. For example, the
# following command
#
#    bazel build //myapp:myapp --embed_label=MyApp_1.2_build_345
#
# would yield the Info.plist values:
#
#    CFBundleVersion = "1.2.345"
#    CFBundleShortVersionString = "1.2"
#
# and the development builds using the command:
#
#    bazel build //myapp:myapp
#
# would yield the values:
#
#    CFBundleVersion = "99.99.99"
#    CFBundleShortVersionString = "99.99"
#
apple_bundle_version(
    name = "build_label_version",
    build_label_pattern = "MyApp_{version}_build_{build}",
    build_version = "{version}.{build}",
    capture_groups = {
        "version": "\\d+\\.\\d+",
        "build": "\\d+",
    },
    short_version_string = "{version}",
    fallback_build_label = "MyApp_99.99_build_99",
)

ios_application(
    name = "bar_app",
    ...,
    version = ":build_label_version",
)
```

Provides:
  AppleBundleVersionInfo: Contains a reference to the JSON file that holds the
      version information for a bundle.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_bundle_version-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_bundle_version-build_label_pattern"></a>build_label_pattern |  A pattern that should contain placeholders inside curly braces (e.g., `"foo_{version}_bar"`) that is used to parse the build label that is generated in the build info file with the `--embed_label` option passed to Bazel. Each of the placeholders is expected to match one of the keys in the `capture_groups` attribute.   | String | optional |  `""`  |
| <a id="apple_bundle_version-build_version"></a>build_version |  A string that will be used as the value for the `CFBundleVersion` key in a depending bundle's Info.plist. If this string contains placeholders, then they will be replaced by strings captured out of `build_label_pattern`.   | String | required |  |
| <a id="apple_bundle_version-capture_groups"></a>capture_groups |  A dictionary where each key is the name of a placeholder found in `build_label_pattern` and the corresponding value is the regular expression that should match that placeholder. If this attribute is provided, then `build_label_pattern` must also be provided.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> String</a> | optional |  `{}`  |
| <a id="apple_bundle_version-fallback_build_label"></a>fallback_build_label |  A build label to use when the no `--embed_label` was provided on the build. Used to provide a version that will be used during development.   | String | optional |  `""`  |
| <a id="apple_bundle_version-short_version_string"></a>short_version_string |  A string that will be used as the value for the `CFBundleShortVersionString` key in a depending bundle's Info.plist. If this string contains placeholders, then they will be replaced by strings captured out of `build_label_pattern`. This attribute is optional; if it is omitted, then the value of `build_version` will be used for this key as well.   | String | optional |  `""`  |


