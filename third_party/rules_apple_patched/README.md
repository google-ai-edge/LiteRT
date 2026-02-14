# Apple Rules for [Bazel](https://bazel.build)

This repository contains rules for [Bazel](https://bazel.build) that can be
used to bundle applications for Apple platforms.

These rules handle the linking and bundling of applications and extensions
(that is, the formation of an `.app` with an executable and resources,
archived in an `.ipa`). Compilation is still performed by the existing
[`objc_library` rule](https://bazel.build/reference/be/objective-c#objc_library)
in Bazel, and by the
[`swift_library` rule](https://github.com/bazelbuild/rules_swift/blob/main/doc/rules.md#swift_library)
available from [rules_swift](https://github.com/bazelbuild/rules_swift).

If you are looking for an easy way to build mixed language frameworks, check out [rules_swift's `mixed_language_library`](https://github.com/bazelbuild/rules_swift/blob/main/doc/rules.md#mixed_language_library).

## Reference documentation

[Click here](https://github.com/bazelbuild/rules_apple/tree/main/doc)
for the reference documentation for the rules and other definitions in this
repository.

## Quick setup

Copy the latest `MODULE.bazel` or `WORKSPACE` snippet from [the releases
page](https://github.com/bazelbuild/rules_apple/releases).

## Examples

Minimal example:

```python
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "MyLibrary",
    srcs = glob(["**/*.swift"]),
    data = [":Main.storyboard"],
)

# Links code from "deps" into an executable, collects and compiles resources
# from "deps" and places them with the executable in an .app bundle, and then
# outputs an .ipa with the bundle in its Payload directory.
ios_application(
    name = "App",
    bundle_id = "com.example.app",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [":Info.plist"],
    minimum_os_version = "15.0",
    deps = [":MyLibrary"],
)
```

See the [examples](https://github.com/bazelbuild/rules_apple/tree/main/examples)
directory for sample applications.

## Supported bazel versions

rules_apple and rules_swift are often affected by changes in bazel
itself. This means you generally need to update these rules as you
update bazel.

You can also see the supported bazel versions in the notes for each
release on the [releases
page](https://github.com/bazelbuild/rules_apple/releases).

Besides these constraints this repo follows
[semver](https://semver.org/) as best as we can since the 1.0.0 release.

| Bazel release | Minimum supported rules version | Final supported rules version | Supporting Branch |
|:-------------------:|:-------------------:|:-------------------------:|:-------------------------:|
| 8.x (most recent rolling) | 2.* | current | `main` |
| 7.x | 2.* | current | `main` |
| 6.x | 2.* | 3.13.0 | `main` |
| 5.x | 0.33.0 | 1.* | `bazel/5.x` |
| 4.x | 0.30.0 | 0.32.0 | N/A |
| 3.x | 0.20.0 | 0.21.2 | N/A |
