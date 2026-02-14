# Apple Bazel definitions

## Guides

*   [Common Information](common_info.md)
*   [Frameworks Guide](frameworks.md)
*   [Resources Guide](resources.md)

## Platform-specific rules

Each Apple platform has its own rules for building bundles (applications,
extensions, and frameworks) and for running unit tests and UI tests.

<table class="table table-condensed table-bordered table-params">
  <thead>
    <tr>
      <th>Platform</th>
      <th><code>.bzl</code> file</th>
      <th>Bundling rules</th>
      <th>Testing rules</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th align="left" valign="top">iOS</th>
      <td valign="top"><code>@build_bazel_rules_apple//apple:ios.bzl</code></td>
      <td valign="top">
        <code><a href="rules-ios.md#ios_app_clip">ios_app_clip</a></code><br/>
        <code><a href="rules-ios.md#ios_application">ios_application</a></code><br/>
        <code><a href="rules-ios.md#ios_dynamic_framework">ios_dynamic_framework</a></code><br/>
        <code><a href="rules-ios.md#ios_extension">ios_extension</a></code><br/>
        <code><a href="rules-ios.md#ios_framework">ios_framework</a></code><br/>
        <code><a href="rules-ios.md#ios_imessage_application">ios_imessage_application</a></code><br/>
        <code><a href="rules-ios.md#ios_imessage_extension">ios_imessage_extension</a></code><br/>
        <code><a href="rules-ios.md#ios_static_framework">ios_static_framework</a></code><br/>
        <code><a href="rules-ios.md#ios_sticker_pack_extension">ios_sticker_pack_extension</a></code><br/>
      </td>
      <td valign="top">
        <code><a href="rules-ios.md#ios_build_test">ios_build_test</a></code><br/>
        <code><a href="rules-ios.md#ios_ui_test_suite">ios_ui_test_suite</a></code><br/>
        <code><a href="rules-ios.md#ios_ui_test">ios_ui_test</a></code><br/>
        <code><a href="rules-ios.md#ios_unit_test_suite">ios_unit_test_suite</a></code><br/>
        <code><a href="rules-ios.md#ios_unit_test">ios_unit_test</a></code><br/>
        <code><a href="rules-ios.md#ios_xctestrun_runner">ios_xctestrun_runner</a></code><br/>
      </td>
    </tr>
    <tr>
      <th align="left" valign="top">macOS</th>
      <td valign="top"><code>@build_bazel_rules_apple//apple:macos.bzl</code></td>
      <td valign="top">
        <code><a href="rules-macos.md#macos_application">macos_application</a></code><br/>
        <code><a href="rules-macos.md#macos_bundle">macos_bundle</a></code><br/>
        <code><a href="rules-macos.md#macos_command_line_application">macos_command_line_application</a></code><br/>
        <code><a href="rules-macos.md#macos_extension">macos_extension</a></code><br/>
      </td>
      <td valign="top">
        <code><a href="rules-macos.md#macos_build_test">macos_build_test</a></code><br/>
        <code><a href="rules-macos.md#macos_unit_test">macos_unit_test</a></code><br/>
      </td>
    <tr>
      <th align="left" valign="top">tvOS</th>
      <td valign="top"><code>@build_bazel_rules_apple//apple:tvos.bzl</code></td>
      <td valign="top">
        <code><a href="rules-tvos.md#tvos_application">tvos_application</a></code><br/>
        <code><a href="rules-tvos.md#tvos_dynamic_framework">tvos_dynamic_framework</a></code><br/>
        <code><a href="rules-tvos.md#tvos_extension">tvos_extension</a></code><br/>
        <code><a href="rules-tvos.md#tvos_framework">tvos_framework</a></code><br/>
        <code><a href="rules-tvos.md#tvos_static_framework">tvos_static_framework</a></code><br/>
      </td>
      <td valign="top">
        <code><a href="rules-tvos.md#tvos_build_test">tvos_build_test</a></code><br/>
        <code><a href="rules-tvos.md#tvos_ui_test">tvos_ui_test</a></code><br/>
        <code><a href="rules-tvos.md#tvos_unit_test">tvos_unit_test</a></code><br/>
      </td>
    </tr>
    <tr>
      <th align="left" valign="top">watchOS</th>
      <td valign="top"><code>@build_bazel_rules_apple//apple:watchos.bzl</code></td>
      <td valign="top">
        <code><a href="rules-watchos.md#watchos_application">watchos_application</a></code><br/>
        <code><a href="rules-watchos.md#watchos_dynamic_framework">watchos_dynamic_framework</a></code><br/>
        <code><a href="rules-watchos.md#watchos_extension">watchos_extension</a></code><br/>
        <code><a href="rules-watchos.md#watchos_framework">watchos_framework</a></code><br/>
        <code><a href="rules-watchos.md#watchos_static_framework">watchos_static_framework</a></code><br/>
      </td>
      <td valign="top">
        <code><a href="rules-watchos.md#watchos_build_test">watchos_build_test</a></code><br/>
        <code><a href="rules-watchos.md#watchos_ui_test">watchos_ui_test</a></code><br/>
        <code><a href="rules-watchos.md#watchos_unit_test">watchos_unit_test</a></code><br/>
      </td>
    </tr>
  </tbody>
</table>

## Other rules

General rules that are not specific to a particular Apple platform are listed
below.

<table class="table table-condensed table-bordered table-params">
  <thead>
    <tr>
      <th>Category</th>
      <th><code>.bzl</code> file</th>
      <th>Rules</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th align="left" valign="top" rowspan="6">General</th>
      <tr>
        <td valign="top"><code>@build_bazel_rules_apple//apple:apple.bzl</code></td>
        <td valign="top">
          <code><a href="rules-apple.md#apple_dynamic_framework_import">apple_dynamic_framework_import</a></code><br/>
          <code><a href="rules-apple.md#apple_dynamic_xcframework_import">apple_dynamic_xcframework_import</a></code><br/>
          <code><a href="rules-apple.md#apple_static_framework_import">apple_static_framework_import</a></code><br/>
          <code><a href="rules-apple.md#apple_static_library">apple_static_library</a></code><br/>
          <code><a href="rules-apple.md#apple_static_xcframework_import">apple_static_xcframework_import</a></code><br/>
          <code><a href="rules-apple.md#apple_static_xcframework">apple_static_xcframework</a></code><br/>
          <code><a href="rules-apple.md#apple_universal_binary">apple_universal_binary</a></code><br/>
          <code><a href="rules-apple.md#apple_xcframework">apple_xcframework</a></code><br/>
          <code><a href="rules-apple.md#local_provisioning_profile">local_provisioning_profile</a></code><br/>
          <code><a href="rules-apple.md#provisioning_profile_repository_extension">provisioning_profile_repository_extension</a></code><br/>
          <code><a href="rules-apple.md#provisioning_profile_repository">provisioning_profile_repository</a></code><br/>
        </td>
      </tr>
      <tr>
        <td valign="top"><code>@build_bazel_rules_apple//apple:docc.bzl</code></td>
        <td valign="top">
          <code><a href="rules-docc.md#docc_archive">docc_archive</a></code>
        </td>
      </tr>
      <tr>
        <td valign="top"><code>@build_bazel_rules_apple//apple:header_map.bzl</code></td>
        <td valign="top">
          <code><a href="rules-header_map.md#header_map">header_map</a></code>
        </td>
      </tr>
      <tr>
        <td valign="top"><code>@build_bazel_rules_apple//apple:xcarchive.bzl</code></td>
        <td valign="top"><code><a href="rules-xcarchive.md#xcarchive">xcarchive</a></code></td>
      </tr>
      </tr>
        <td valign="top"><code>@build_bazel_rules_apple//apple:versioning.bzl</code></td>
        <td valign="top"><code><a href="rules-versioning.md#apple_bundle_version">apple_bundle_version</a></code><br/></td>
      </tr>
      <tr>
        <td valign="top"><code>@build_bazel_rules_apple//apple:xctrunner.bzl</code></td>
        <td valign="top"><code><a href="rules-xctrunner.md#xctrunner">xctrunner</a></code></td>
      </tr>
    </tr>
    <tr>
      <th align="left" valign="top" rowspan="1">Resources</th>
      <td valign="top"><code>@build_bazel_rules_apple//apple:resources.bzl</code></td>
      <td valign="top">
        <code><a href="rules-resources.md#apple_bundle_import">apple_bundle_import</a></code><br/>
        <code><a href="rules-resources.md#apple_core_data_model">apple_core_data_model</a></code><br/>
        <code><a href="rules-resources.md#apple_core_ml_library">apple_core_ml_library</a></code><br/>
        <code><a href="rules-resources.md#apple_precompiled_resource_bundle">apple_precompiled_resource_bundle</a></code><br/>
        <code><a href="rules-resources.md#apple_resource_bundle">apple_resource_bundle</a></code><br/>
        <code><a href="rules-resources.md#apple_resource_group">apple_resource_group</a></code><br/>
        <code><a href="rules-resources.md#swift_apple_core_ml_library">swift_apple_core_ml_library</a></code><br/>
        <code><a href="rules-resources.md#swift_intent_library">swift_intent_library</a></code><br/>
      </td>
    </tr>
  </tbody>
</table>

## Extending or integrating with these rules

If you want to write custom rules that integrate with these Apple platform rules
(for example, write a rule that provides resources to an application or takes an
application as a dependency), then please refer to the documentation on
[providers](providers.md) to see the data that these rules propagate as output
and expect as input.
