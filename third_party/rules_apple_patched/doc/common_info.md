# Apple Bazel definitions - Common Information

## Build limits

According to
[Apple's documentation](https://help.apple.com/app-store-connect/#/dev611e0a21f),
iOS and tvOS bundle sizes cannot exceed 4GB in size (limit for watchOS is 75MB).
Because of this, rules_apple does not support building zipped archives that
would be larger than 4GB. If your build outputs would be larger than 4GB (e.g.
test bundles) you'll need to reduce the number of dependencies to fit this limit
(e.g. for test bundles, you can split the targets so that output size is smaller
than 4GB).

## Build outputs

Most aspects of your builds should be controlled via the attributes you set on
the rules. However there are some things where Bazel and/or the rules allow you
to opt in/out of something on a per-build basis without the need to express it
in a BUILD file.

### Output Groups {#output_groups}

[Output groups](https://docs.bazel.build/versions/master/skylark/rules.html#requesting-output-files)
are an interface provided by Bazel to signal which files are to be built. By
default, Bazel will build all files in the `default` output group. In addition
to this, rules_apple supports other output groups that can be used to control
which files are requested:

*   `dsyms`: This output group contains all dSYM files generated during the
    build, for the top level target **and** its embedded dependencies. To
    request this output group to be built, use the `--output_groups=+dsyms`
    flag. In order to generate the dSYM files you still need to pass the
    `--apple_generate_dsym` flag.
*   `linkmaps`: This output group contains all linkmap files generated during
    the build, for the top level target **and** its embedded dependencies. To
    request this output group to be built, use the `--output_groups=+linkmaps`
    flag. In order to generate the linkmap files you still need to pass the
    `--objc_generate_linkmap` flag.

### dSYMs Generation {#apple_generate_dsym}

dSYMs are needed for debugging, decode crash logs, etc.; but they can take a
while to generate and aren't always needed. All of the Apple rules support
generating a dSYM bundle via `--apple_generate_dsym` when doing a `bazel build`.

```shell
bazel build --apple_generate_dsym //your/target
```

By default, only the top level dSYM bundles is built when this flag is
specified. If you require the dSYM bundles of the top level target dependencies,
you'll need to specify the `--output_groups=+dsyms` flag.

<!-- Begin-External -->

### Codesigning identity

When building for devices, by default the codesigning step will use the first
codesigning identity present in the given provisioning profile. This should
accommodate most cases, but there are certain scenarios when the provisioning
profile will have more than one allowed signing identity, and developers may
have different development certificates installed on their devices. For these
cases you can use the `--ios_signing_cert_name` flag to force the signing
identity to be used when codesigning your app.

```shell
bazel build //your/target --ios_signing_cert_name="iPhone Developer: [CERT OWNER NAME]"
```

To make this easier to use, we recommend adding the following to the
`~/.bazelrc` file, which will configure bazel to pass this flag to all
invocations:

```text
build --ios_signing_cert_name="iPhone Developer: [CERT OWNER NAME]"
```

<!-- End-External -->

<!-- Blocked on b/73547309

### Sanitizers {#sanitizers}

Sanitizers are useful for validating your code by detecting runtime corruptions
in memory (Address Sanitizer), data race conditions (Thread Sanitizer), or
undefined behavior (Undefined Behavior Sanitizer). When running an application,
or its tests, using Bazel, you can enable these sanitizers using the following
flags:

```shell
# Address Sanitizer
bazel test --features=asan //your/target

# Thread Sanitizer
bazel test --features=tsan //your/target

# Undefined Behavior Sanitizer
bazel test --features=ubsan //your/target
```

When you enable these features, the appropriate compilation and linking flags
will be added to the build, and the rules will package the corresponding dylibs
into your output bundle.

Similar to what you can find in Xcode, the Address and Thread sanitizers are
mutually exclusive, i.e. you can only specify one or the other for a particular
build.

In case you want to have different compiler and linker flags, you can use
`--features=include_clang_rt` and specify the required compiler and linker
flags yourself.

-->

### linkmap Generation {#objc_generate_linkmap}

Linkmaps can be useful for figuring out how the `deps` going into a target are
contributing to the final size of the binary. Bazel will generate a link map
when linking by adding `--objc_generate_linkmap` to a `bazel build`.

```shell
bazel build --objc_generate_linkmap //your/target
```

By default, only the top level linkmap file is built when this flag is
specified. If you require the linkmap file of the top level target dependencies,
you'll need to specify the `--output_groups=+linkmaps` flag.

### Debugging Entitlement Support {#apple.add_debugger_entitlement}

Some Apple platforms require an entitlement (`get-task-allow`) to support
debugging tools. The rules will auto add the entitlement for non optimized
builds (i.e. - anything that isn't `-c opt`). However when looking at specific
issues (performance of a release build via Instruments), the entitlement is also
needed.

The rules support direct control over the inclusion/exclusion of any bundle
being built by
`--define=apple.add_debugger_entitlement=(yes|true|1|no|false|0)`.

Add `get-task-allow` entitlement:

```shell
bazel build --define=apple.add_debugger_entitlement=yes //your/target
```

Ensure `get-task-allow` entitlement is *not* added (even if the default would
have added it):

```shell
bazel build --define=apple.add_debugger_entitlement=no //your/target
```

### Force ipa compression {#apple.compress_ipa}

By default the final `App.ipa` produced from building an app is uncompressed,
unless you're building with `--compilation_mode=opt`. This flag allows you to
force compression if the size is more important than the CPU time for your
build. To use this pass `--define=apple.compress_ipa=(yes|true|1)` to `bazel
build`.

### Include Embedded Bundles in Rule Output {#apple.propagate_embedded_extra_outputs}

**Deprecated: Please see the [Output Groups](#output_groups) section.**

Some Apple bundles include other bundles within them (for example, an
application extension inside an iOS application). When you build a top-level
application target and ask for extra outputs such as linkmaps or dSYM bundles,
Bazel typically only produces the extra outputs for the top-level application
but not for the embedded extension.

In order to produce those extra outputs for all embedded bundles as well, you
can pass `--define=apple.propagate_embedded_extra_outputs=(yes|true|1)` to
`bazel build`.

```shell
bazel build --define=apple.propagate_embedded_extra_outputs=yes //your/target
```

### Disable `SwiftSupport` in ipas

The SwiftSupport directory in a final ipa is only necessary if you're shipping
the build to Apple. If you want to disable bundling SwiftSupport in your ipa for
other device or enterprise builds, you can pass
`--define=apple.package_swift_support=no` to `bazel build`

### Codesign Bundles for the Simulator {#apple.skip_codesign_simulator_bundles}

The simulators are far more lax about a lot of things compared to working on
real devices. One of these areas is the codesigning of bundles (applications,
extensions, etc.). As of Xcode 9.3.x on macOS High Sierra, the Simulator will
run any bundle just fine as long as its Frameworks are signed (if it has any),
the main bundle does *not* appear to need to be signed.

However, if the binary makes use of entitlement-protected APIs, they may not
work. The entitlements for a simulator build are added as a Mach-O segment, so
they are still provided; but for some entitlements (at a minimum, Shared App
Groups), the simulator appears to also require the bundle be signed for the APIs
to work.

By default, the rules will do what Xcode would otherwise do and *will* sign the
main bundle (with an adhoc signature) when targeting the Simulator. However,
this feature can be used to opt out of this if you are more concerned with
build speed vs. potential correctness.

Remember, at any time, Apple could do a macOS point release and/or an Xcode
release that changes this and opting out of could mean your binary doesn't run
under the simulator.

The rules support direct control over this signing via
`--features=apple.skip_codesign_simulator_bundles`.

Disable the signing of simulator bundles:

```shell
bazel build --features=apple.skip_codesign_simulator_bundles //your/target
```

More likely you'll want to do this on a per-target basis such as with:

```bzl
ios_unit_test(
    ...
    features = ["apple.skip_codesign_simulator_bundles"],
)
```

### Codesigning performance

For larger applications, codesigning the final binary might be a
bottleneck in incremental build performance. [Michael Eisel
discovered](https://eisel.me/signing) a few clever optimizations for
improving this performance for debug builds. To use these with bazel you
can add something like this to your top level app rule, for example with
`ios_application`:

```bzl
config_setting(
    name = "dbg",
    values = {"compilation_mode": "dbg"},
)

ios_application(
    ...
    codesignopts = select({
        ":dbg": [
            "--digest-algorithm=sha1",
            "--resource-rules=$(RESOURCE_RULES)",
        ],
        "//conditions:default": [],
    }),
    codesign_inputs = select({
        ":dbg": ["@build_bazel_rules_apple//tools/codesigningtool:disable_signing_resource_rules"],
        "//conditions:default": [],
    }),
    toolchains = select({
        "//:dbg": ["@build_bazel_rules_apple//tools/codesigningtool:disable_signing_resource_rules"],
        "//conditions:default": [],
    }),
)
```

### Localization Handling

The Apple bundling rules have two flags for limiting which \*.lproj directories
are copied as part of your build. Without specifying either the
`apple.locales_to_include` flag or the `apple.trim_lproj_locales` flag, all
locales are copied.

Locale names are explicitly matched; for example `pt` may not be sufficient as
`pt_BR` or `pt_PT` is likely the name of the lproj folder. Note that
`Base.lproj` is always included if it exists.

#### Explicitly Listing Locales

Use `--define "apple.locales_to_include=foo,bar,bam"` where `foo,bar,bam` are
the exact names of the locales to be included.

This can be used to improve compile/debug/test cycles because most developers
only work/test in one language.

#### Explicitly Excluding Locales

Use `--define "apple.locales_to_exclude=foo,bar,bam"` where `foo,bar,bam` are
the exact names of the locales to be excluded.

#### Automatically Trimming Locales

Use `--define "apple.trim_lproj_locales=(yes|true|1)"` to strip any `.lproj`
folders that don't have a matching `.lproj` folder in the base of the resource
folder(e.g. bundles from Frameworks that are localized).

If a product is pulling in some other component(s) that support more
localizations than the product does, this can be used to strip away those extra
localizations thereby shrinking the final product sent to the end users.

## Info.plist Handling

### Merging

The `infoplists` attribute on the Apple rules takes a list of files that will be
merged. This allows developers to provide fragments of the final Info.plist and
use `select()` statements to pull in different bits based on different
configuration settings. For example, a `select()` on some `config_setting` could
allow the rule to pull in a file with a different `CFBundleDisplayName` pair for
the TestFlight build.

The merging though is done only at the root level of these plists. If two files
being merged have the same key, their values must match. That means if two files
both have a value for something (e.g., `CFBundleDisplayName`), then as long as
the values are the same, the build will succeed; but if they have different
values, then the build will fail. If a key's value is an array or a dictionary,
those values won't be merged and must match to avoid the failure.

### Variable Substitution

As the files are merged, the values are recursively checked for variable
references and substitutions are made. A reference is made using `${NAME}` or
`$(NAME)` notation. Similar to Xcode, the variable reference can get the
`rfc1034identifier` qualifier (i.e., `${NAME:rfc1034identifier}`); this will
transform the value such that any characters besides `0-9A-Za-z.` will be
replaced with the `-` character (i.e., `Foo Bar` will become `Foo-Bar`).

<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Variables Supported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>BUNDLE_NAME</code></td>
      <td>
        <p>This is the same value as <code>PRODUCT_NAME</code>, but with the
        an extension appended. If the target supports a
        <code>bundle_extension</code> attribute, that is used. If it does
        not, or it is not set, then the Apple default is used based on the
        target's product type (i.e., <code>.app</code>, <code>.appex</code>,
        <code>.bundle</code>, etc.).</p>
      </td>
    </tr>
    <tr>
      <td><code>DEVELOPMENT_LANGUAGE</code></td>
      <td>
        <p>This is currently hardcoded to <code>en</code> if exists. This is
        done to support the default Info.plists come from Xcode.</p>
      </td>
    </tr>
    <tr>
      <td><code>EXECUTABLE_NAME</code></td>
      <td>
        <p>The value of the rule's <code>executable_name</code> attribute if it
        was given; if not, then the name of the <code>bundle_name</code>
        attribute if it was given; if not, then the <code>name</code>of the
        target.</p>
      </td>
    </tr>
    <tr>
      <td><code>PRODUCT_BUNDLE_IDENTIFIER</code></td>
      <td>
        <p>The value of the rule's <code>bundle_id</code> attribute. If the rule
        does not have the attribute, it is not supported.</p>
      </td>
    </tr>
    <tr>
      <td><code>PRODUCT_NAME</code></td>
      <td>
        <p>If the rule supports a <code>bundle_name</code> attribute, it is
        that value. If the rule doesn't have the attribute or the attribute
        isn't set, then it is the <code>name</code> of the target.</p>
      </td>
    </tr>
    <tr>
      <td><code>TARGET_NAME</code></td>
      <td>
        <p>This is an alias for the same value as <code>PRODUCT_NAME</code>.
        This is done to match some developer expectations from Xcode. Note
        that, despite the word "TARGET" in the name, it may not always
        correspond to the <code>BUILD</code> target if the
        <code>bundle_name</code> attribute is provided.</p>
      </td>
    </tr>
  </tbody>
</table>

<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Variables Explicitly Not Supported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>PRODUCT_MODULE_NAME</code></td>
      <td>
        <p>This is a variable commonly used in extensions to specify the
        <code>NSExtensionPrincipalClass</code> attribute, which signals which
        class is the entry point to the extension (e.g.
        <code>$(PRODUCT_MODULE_NAME).ServiceExtension</code>). When not
        explicitly set, Xcode sets it to the same value as
        <code>PRODUCT_NAME</code>, which is the name of the Xcode target. This
        is mostly safe to do when using plain Xcode, as by default it defines
        one module per target, and all classes belong to that module. When using
        Bazel to build an extension, through <code>objc_library</code> and/or
        <code>swift_library</code>, each of the targets defines a new module,
        which makes it harder to automatically detect the module name which
        contains the principal class. Because of this reason, this value is not
        supported and therefore should be explicitly set in your targets'
        Info.plist files.</p>
      </td>
    </tr>
  </tbody>
</table>

## Tests

### Runfiles location for test data

Most likely, test related resources will be bundled within the `.xctest` bundle
itself, but there may be use cases where the test resources are not wanted in
the bundle, but instead are needed in the Bazel runfiles location. These
resources should be placed into the `data` attribute of the
`<platform>_unit_test` or `<platform>_ui_test` targets.

Within the tests you can retrieve the runfiles resources through the
`TEST_SRCDIR` environment variable following this template:

```
$TEST_SRCDIR/<workspace_name>/<workspace_relative_path_to_resource>
```

Take for example this `BUILD` file:

```
# my/package/BUILD

...
ios_unit_test(
    name = "MyTest",
    ...
    data = ["my_test_resource.txt"],
    ...
)
```

To read this file from, for example, a Swift test, you'd get the path with
something similar to:

``` swift
// MyTest.swift

...
  guard let runfilesPath = ProcessInfo.processInfo.environment["TEST_SRCDIR"],
        let workspaceName = ProcessInfo.processInfo.environment["TEST_WORKSPACE"],
        let binaryPath = ProcessInfo.processInfo.environment["TEST_BINARY"]
  else {
    fatalError("Unable to determine runfiles path")
  }
  let resourceFullPath = "\(runfilesPath)/\(workspaceName)/\(binaryPath)\(resourcePath)"
...

```

Note: If your test target's name shares the same name as part of its subpath, this will not
work i.e. naming your tests something like `ModelsTests` residing at `src/ModelsTests` then
runfiles will break. To fix this rename the test target to something like `ModelsUnitTests`

This issue is tracked [here](https://github.com/bazelbuild/bazel/issues/12312)

### Xcode's Issue navigator

If integrating with Xcode, the relative paths in test binaries can prevent the
Issue navigator from working for test failures. To work around this, you can
have the paths made absolute via swizzling by enabling the
`"apple.swizzle_absolute_xcttestsourcelocation"` feature. You'll also need to
set the `BUILD_WORKSPACE_DIRECTORY` environment variable in your scheme to the
root of your workspace (i.e. `$(SRCROOT)`).

### Xcode Version Selection and Invalidation

There are a few steps required to properly make Bazel use the right Xcode version. Moreover, a few tricks are needed to make sure that the Bazel server is restarted and certain caches cleared when changing Xcode version using `xcode-select`.

1. The first thing you should think about is to enforce a single Xcode version for all your builds to ensure remote cache hits. On top of that, enforcing a single Xcode version speeds up repository setup time. You can achieve this by passing a specific Xcode version config via the `--xcode_version_config` flag. More details are available in [Locking Xcode versions in Bazel](https://www.smileykeith.com/2021/03/08/locking-xcode-in-bazel).
2. If your configuration supports multiple Xcode versions, you should pass `--xcode_version` to specify which version should be used.
3. In your Bazel wrapper (`tools/bazel` in your repository, read more [here](https://github.com/bazelbuild/bazelisk#ensuring-that-your-developers-use-bazelisk-rather-than-bazel)), you should pass a few flags to every invocation or generate and import a `bazelrc`:
    * Capture `xcode-select -p` or use the value of `DEVELOPER_DIR` if available and forward it to repository rules for invalidation when changed: `--repo_env=DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
    * Pass `--repo_env=XCODE_VERSION=$xcode_version` where `xcode_version` should be the value of `xcodebuild -version | tail -1 | cut -d " " -f3` which is unique to each version. This will make sure the `apple_support` toolchain and `xcode_configure` repository rule are re-evaluated when your version changes.
    * To invalidate the repository rule when Xcode's path changes but the version doesn't, pass the `--host_jvm_args=-Xdock:name=$developer_dir` startup flag. This forwards an argument to the JVM which is ignored except for causing the server to restart when its value changes.

The above flags can be passed either directly to each invocation or by generating a `bazelrc` which is imported from your main `.bazelrc`. This snippet shows the latter option:

```bash
#!/bin/bash

bazel_real="$BAZEL_REAL"
bazelrc_lines=()

if [[ $OSTYPE == darwin* ]]; then
  xcode_path=$(xcode-select -p)
  xcode_version=$(xcodebuild -version | tail -1 | cut -d " " -f3)
  xcode_build_number=$(/usr/bin/xcodebuild -version 2>/dev/null | tail -1 | cut -d " " -f3)

  bazelrc_lines+=("startup --host_jvm_args=-Xdock:name=$xcode_path")
  bazelrc_lines+=("build --xcode_version=$xcode_version")
  bazelrc_lines+=("build --repo_env=XCODE_VERSION=$xcode_version")
  bazelrc_lines+=("build --repo_env=DEVELOPER_DIR=$xcode_path")
fi

printf '%s\n' "${bazelrc_lines[@]}" > xcode.bazelrc

exec "$bazel_real" "$@"
```

In your main `.bazelrc` add `import xcode.bazelrc` at the very bottom.

## Optimizing remote cache and build execution performance

When using Bazel's remote cache and/or build execution, there are a few flags you can pass to optimize performance. One of those flags is [`--modify_execution_info`](https://bazel.build/reference/command-line-reference#flag--modify_execution_info), which allows adding or removing execution info for specific [mnemonics](https://bazel.build/reference/glossary#mnemonic), which in turn allows you to configure what is cached or built remotely.

We recommend adding the following to your `.bazelrc`:

```shell
common --modify_execution_info=^(BundleApp|BundleTreeApp|DsymDwarf|DsymLipo|GenerateAppleSymbolsFile|ObjcBinarySymbolStrip|CppArchive|CppLink|ObjcLink|ProcessAndSign|SignBinary|SwiftArchive|SwiftStdlibCopy)$=+no-remote,^(BundleResources|ImportedDynamicFrameworkProcessor)$=+no-remote-exec
```

The following table provides a rationale for each mnemonic and tag. In general though, the mnemonics that are excluded in `--modify_execution_info` are excluded because they produce or work on large outputs which change frequently and as such are faster when run locally, or they are not generally configured for remote execution (such as signing).

| Mnemonics | Tag | Rationale |
| --- | --- | --- |
| `BundleApp`, `BundleTreeApp`, `ProcessAndSign` | `no-remote` | Produces a large bundle, which is inefficient to upload and download |
| `CppArchive`, `CppLink`, `ObjcLink`, `SwiftArchive` | `no-remote` | Linked binaries have local paths, and it's slower to download them versus linking locally |
| `SwiftStdlibCopy` | `no-remote` | Processing Swift stdlib is a quick file copy of a locally available resource, so it's not worth uploading or downloading |
| `DsymDwarf`, `DsymLipo`, `GenerateAppleSymbolsFile`| `no-remote-exec` | Processing dSYMs/Symbols remotely requires uploading the linked binary; this could go away if you switch to uploading linked binaries |
| `ImportedDynamicFrameworkProcessor` | `no-remote-exec` | Processing dynamic frameworks remotely incurs an upload and download of the same blob |
| `ObjcBinarySymbolStrip` | `no-remote-exec` | Stripping binaries remotely requires uploading the linked binary; this could go away if you switch to uploading linked binaries |
| `ProcessAndSign`, `SignBinary` | `no-remote-exec` | RBE is not generally configured for code signing |
| `BundleApp`, `BundleResources`, `BundleTreeApp`, `ImportedDynamicFrameworkProcessor`, `ProcessAndSign`, `SignBinary` | `no-remote-exec` | These actions are inefficient to do remotely, but in large numbers downloading can be efficient |
