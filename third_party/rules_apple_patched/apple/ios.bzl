# Copyright 2017 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bazel rules for creating iOS applications and bundles."""

load(
    "//apple/internal:ios_rules.bzl",
    _ios_app_clip = "ios_app_clip",
    _ios_application = "ios_application",
    _ios_dynamic_framework = "ios_dynamic_framework",
    _ios_extension = "ios_extension",
    _ios_framework = "ios_framework",
    _ios_imessage_application = "ios_imessage_application",
    _ios_imessage_extension = "ios_imessage_extension",
    _ios_static_framework = "ios_static_framework",
    _ios_sticker_pack_extension = "ios_sticker_pack_extension",
)
load(
    "//apple/internal/testing:apple_test_assembler.bzl",
    "apple_test_assembler",
)
load(
    "//apple/internal/testing:build_test_rules.bzl",
    "apple_build_test_rule",
)
load(
    "//apple/internal/testing:ios_rules.bzl",
    _ios_internal_ui_test_bundle = "ios_internal_ui_test_bundle",
    _ios_internal_unit_test_bundle = "ios_internal_unit_test_bundle",
    _ios_ui_test = "ios_ui_test",
    _ios_unit_test = "ios_unit_test",
)

# TODO(b/118104491): Remove these re-exports and move the rule definitions into this file.
ios_application = _ios_application
ios_app_clip = _ios_app_clip
ios_dynamic_framework = _ios_dynamic_framework
ios_extension = _ios_extension
ios_framework = _ios_framework
ios_imessage_application = _ios_imessage_application
ios_sticker_pack_extension = _ios_sticker_pack_extension
ios_imessage_extension = _ios_imessage_extension
ios_static_framework = _ios_static_framework

_DEFAULT_TEST_RUNNER = str(Label("//apple/testing/default_runner:ios_default_runner"))

def ios_unit_test(name, **kwargs):
    runner = kwargs.pop("runner", None) or _DEFAULT_TEST_RUNNER
    apple_test_assembler.assemble(
        name = name,
        bundle_rule = _ios_internal_unit_test_bundle,
        test_rule = _ios_unit_test,
        runner = runner,
        **kwargs
    )

def ios_ui_test(name, **kwargs):
    runner = kwargs.pop("runner", None) or _DEFAULT_TEST_RUNNER
    apple_test_assembler.assemble(
        name = name,
        bundle_rule = _ios_internal_ui_test_bundle,
        test_rule = _ios_ui_test,
        runner = runner,
        **kwargs
    )

def ios_unit_test_suite(name, runners = None, **kwargs):
    """Generates a [test_suite] containing an [ios_unit_test] for each of the given `runners`.

`ios_unit_test_suite` takes the same parameters as [ios_unit_test], except `runner` is replaced by `runners`.

[test_suite]: https://docs.bazel.build/versions/master/be/general.html#test_suite
[ios_unit_test]: #ios_unit_test

Args:
    runners: a list of runner targets
    **kwargs: passed to the [ios_unit_test]
"""
    apple_test_assembler.assemble(
        name = name,
        bundle_rule = _ios_internal_unit_test_bundle,
        test_rule = _ios_unit_test,
        runners = runners,
        **kwargs
    )

def ios_ui_test_suite(name, runners = None, **kwargs):
    """Generates a [test_suite] containing an [ios_ui_test] for each of the given `runners`.

`ios_ui_test_suite` takes the same parameters as [ios_ui_test], except `runner` is replaced by `runners`.

[test_suite]: https://docs.bazel.build/versions/master/be/general.html#test_suite
[ios_ui_test]: #ios_ui_test

Args:
    runners: a list of runner targets
    **kwargs: passed to the [ios_ui_test]
"""
    apple_test_assembler.assemble(
        name = name,
        bundle_rule = _ios_internal_ui_test_bundle,
        test_rule = _ios_ui_test,
        runners = runners,
        **kwargs
    )

ios_build_test = apple_build_test_rule(
    doc = """\
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
""",
    platform_type = "ios",
)
