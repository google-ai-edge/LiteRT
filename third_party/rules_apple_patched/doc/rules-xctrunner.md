<!-- Generated with Stardoc: http://skydoc.bazel.build -->

Rule for creating a XCTRunner.app with one or more .xctest bundles. Retains same
platform and architectures as the given `tests` bundles.

<a id="xctrunner"></a>

## xctrunner

<pre>
load("@rules_apple//apple:xctrunner.bzl", "xctrunner")

xctrunner(<a href="#xctrunner-name">name</a>, <a href="#xctrunner-tests">tests</a>, <a href="#xctrunner-verbose">verbose</a>)
</pre>

Packages one or more .xctest bundles into a XCTRunner.app. Retains same
platform and architectures as the given `tests` bundles.

Example:

````starlark
load("//apple:xctrunner.bzl", "xctrunner")

ios_ui_test(
    name = "HelloWorldSwiftUITests",
    minimum_os_version = "15.0",
    runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_ordered_runner",
    test_host = ":HelloWorldSwift",
    deps = [":UITests"],
)

xctrunner(
    name = "HelloWorldSwiftXCTRunner",
    tests = [":HelloWorldSwiftUITests"],
    testonly = True,
)
````

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="xctrunner-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="xctrunner-tests"></a>tests |  List of test targets and suites to include.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="xctrunner-verbose"></a>verbose |  Print logs from xctrunnertool to console.   | Boolean | optional |  `False`  |


