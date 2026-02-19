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

"""Helper methods for assembling the test targets."""

# Attributes belonging to the bundling rules that should be removed from the test targets.
#
# The bundling rules link binaries while the test target impls do not. As a rule of thumb, if an
# attr is required of linking and does not have a benefit to being exposed to cquery results,
# grouping tests or BUILD visibility, it should be in this list rather than the shared list.
_BUNDLE_ATTRS = {
    x: None
    for x in [
        "additional_contents",
        "additional_linker_inputs",
        "deps",
        "base_bundle_id",
        "bundle_id",
        "bundle_id_suffix",
        "bundle_name",
        "families",
        "frameworks",
        "infoplists",
        "linkopts",
        "provisioning_profile",
        "resources",
        "stamp",
    ]
}

# Attributes that should be explicitly shared between test targets and bundle rules without changes.
_SHARED_TEST_BUNDLE_ATTRS = {
    x: None
    for x in [
        "features",
        "minimum_os_version",
        "tags",
        "test_host",
        "test_host_is_bundle_loader",
        "visibility",
    ]
}

_SHARED_SUITE_TEST_ATTRS = {
    x: None
    for x in [
        "compatible_with",
        "deprecation",
        "distribs",
        "features",
        "licenses",
        "restricted_to",
        "tags",
        "testonly",
        "visibility",
    ]
}

def _assemble(name, bundle_rule, test_rule, runner = None, runners = None, **kwargs):
    """Assembles the test bundle and test targets.

    This method expects that either `runner` or `runners` is populated, but never both. If `runner`
    is given, then a single test target will be created under the given name. If `runners` is given
    then a test target will be created for each runner and a single `test_suite` target will be
    created under the given name, wrapping the created targets.

    The `kwargs` dictionary will contain both bundle and test attributes that this method will split
    accordingly.

    Attrs:
        name: The name of the test target or test suite to create.
        bundle_rule: The bundling rule to instantiate.
        test_rule: The test rule to instantiate.
        runner: A single runner target to use for the test target. Mutually exclusive with
            `runners`.
        runners: A list of runner targets to use for the test targets. Mutually exclusive with
            `runner`.
        **kwargs: The complete list of attributes to distribute between the bundle and test targets.
    """
    if runner != None and runners != None:
        fail("Can't specify both runner and runners.")
    elif not runner and not runners:
        fail("Must specify one of runner or runners.")

    test_bundle_name = name + ".__internal__.__test_bundle"

    test_attrs = {k: v for (k, v) in kwargs.items() if k not in _BUNDLE_ATTRS}
    bundle_attrs = {k: v for (k, v) in kwargs.items() if k in _BUNDLE_ATTRS}

    # Args to apply to the test and the bundle.
    for x in _SHARED_TEST_BUNDLE_ATTRS:
        if x in test_attrs:
            bundle_attrs[x] = test_attrs[x]

    # `bundle_name` is either provided or the default is `name`.
    bundle_name = bundle_attrs.pop("bundle_name", name)

    # `//...` shouldn't try to build the bundle rule directly.
    bundle_tags = bundle_attrs.pop("tags", [])
    if "manual" not in bundle_tags:
        bundle_tags = bundle_tags + ["manual"]

    # Ideally this target should be private, but the outputs should not be private, so we're
    # explicitly using the same visibility as the test (or None if none was set).
    bundle_rule(
        name = test_bundle_name,
        bundle_name = bundle_name,
        tags = bundle_tags,
        test_bundle_output = "{}.zip".format(bundle_name),
        testonly = True,
        **bundle_attrs
    )

    if runner:
        test_rule(
            name = name,
            runner = runner,
            deps = [":{}".format(test_bundle_name)],
            **test_attrs
        )
    elif runners:
        tests = []
        for runner in runners:
            test_name = "{}_{}".format(name, runner.rsplit(":", 1)[-1])
            tests.append(":{}".format(test_name))
            test_rule(
                name = test_name,
                runner = runner,
                deps = [":{}".format(test_bundle_name)],
                **test_attrs
            )
        shared_test_suite_attrs = {k: v for (k, v) in test_attrs.items() if k in _SHARED_SUITE_TEST_ATTRS}
        native.test_suite(
            name = name,
            tests = tests,
            **shared_test_suite_attrs
        )

apple_test_assembler = struct(
    assemble = _assemble,
)
