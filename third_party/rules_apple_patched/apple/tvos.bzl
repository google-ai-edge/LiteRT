# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""Bazel rules for creating tvOS applications and bundles."""

load(
    "//apple/internal:tvos_rules.bzl",
    _tvos_application = "tvos_application",
    _tvos_dynamic_framework = "tvos_dynamic_framework",
    _tvos_extension = "tvos_extension",
    _tvos_framework = "tvos_framework",
    _tvos_static_framework = "tvos_static_framework",
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
    "//apple/internal/testing:tvos_rules.bzl",
    _tvos_internal_ui_test_bundle = "tvos_internal_ui_test_bundle",
    _tvos_internal_unit_test_bundle = "tvos_internal_unit_test_bundle",
    _tvos_ui_test = "tvos_ui_test",
    _tvos_unit_test = "tvos_unit_test",
)

# TODO(b/118104491): Remove these re-exports and move the rule definitions into this file.
tvos_application = _tvos_application
tvos_dynamic_framework = _tvos_dynamic_framework
tvos_extension = _tvos_extension
tvos_framework = _tvos_framework
tvos_static_framework = _tvos_static_framework

_DEFAULT_TEST_RUNNER = str(Label("//apple/testing/default_runner:tvos_default_runner"))

def tvos_unit_test(name, **kwargs):
    runner = kwargs.pop("runner", _DEFAULT_TEST_RUNNER)
    apple_test_assembler.assemble(
        name = name,
        bundle_rule = _tvos_internal_unit_test_bundle,
        test_rule = _tvos_unit_test,
        runner = runner,
        **kwargs
    )

def tvos_ui_test(name, **kwargs):
    runner = kwargs.pop("runner", _DEFAULT_TEST_RUNNER)
    apple_test_assembler.assemble(
        name = name,
        bundle_rule = _tvos_internal_ui_test_bundle,
        test_rule = _tvos_ui_test,
        runner = runner,
        **kwargs
    )

tvos_build_test = apple_build_test_rule(
    doc = """\
Test rule to check that the given library targets (Swift, Objective-C, C++)
build for tvOS.

Typical usage:

```starlark
tvos_build_test(
    name = "my_build_test",
    minimum_os_version = "12.0",
    targets = [
        "//some/package:my_library",
    ],
)
```
""",
    platform_type = "tvos",
)
