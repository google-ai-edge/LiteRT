# Related types

This page documents other special types used by the Apple Bazel rules.

<a name="entitlements_validation_mode"></a>
## entitlements_validation_mode

A `struct` containing the possible modes for `entitlements_validation`
attribute on the bundling rules.

Provisioning profiles are not bundled into "simulator" builds. With a native
Xcode project, Xcode's UI will show errors/issues if you navigate to the
capabilities UI, but it doesn't actually prevent these simluator builds.
However, with "device" builds, Xcodes does validation before starting the build
steps for a target and raises errors (preventing the build) if there is any
issues.

These values can be used for the rules that support the
`entitlements_validation` attribute to set the desired validation behavior
during builds.

<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Modes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>error</code></td>
      <td>
        <p>Perform the checks and raise errors for any issues found.</p>
      </td>
    </tr>
    <tr>
      <td><code>warn</code></td>
      <td>
        <p>Perform the checks and only raise warnings for any issues found.</p>
      </td>
    </tr>
    <tr>
      <td><code>loose</code></td>
      <td>
        <p>Perform the checks and raise warnings for "simulator" builds, but
        raise errors for "device" builds.</p>
      </td>
    </tr>
    <tr>
      <td><code>skip</code></td>
      <td>
        <p>Perform no checks at all.</p>
        <p>This should only be need in cases where the build result is
        <i>not</i> used directly, and some external process resigns the build
        after the fact correcting the entitlements/provisioning.</p>
      </td>
    </tr>
  </tbody>
</table>

The `loose` value is closest to what a native Xcode project would do in that
it doesn't prevevent a simulator build, but does stop a device build. The main
difference being it atleast raises warnings to let you know of the issue with
the target since they would prevent one from deploying the build to a device.

Example usage:

```python
load(
    "@build_bazel_rules_apple//apple:common.bzl",
    "entitlements_validation_mode",
)
load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_application",
)

ios_application(
    name = "MyApp",
    entitlements = "MyApp.entitlements",
    entitlements_validation = entitlements_validation_mode.error,
    # other attributes...
)
```
