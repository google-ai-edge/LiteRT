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

"""# Rules related to Apple resources and resource bundles."""

load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//apple/internal:resources.bzl",
    _resources_common = "resources",
)
load(
    "//apple/internal/resource_rules:apple_bundle_import.bzl",
    _apple_bundle_import = "apple_bundle_import",
)
load(
    "//apple/internal/resource_rules:apple_core_data_model.bzl",
    _apple_core_data_model = "apple_core_data_model",
)
load(
    "//apple/internal/resource_rules:apple_core_ml_library.bzl",
    _apple_core_ml_library = "apple_core_ml_library",
)
load(
    "//apple/internal/resource_rules:apple_intent_library.bzl",
    _apple_intent_library = "apple_intent_library",
)
load(
    "//apple/internal/resource_rules:apple_metal_library.bzl",
    _apple_metal_library = "apple_metal_library",
)
load(
    "//apple/internal/resource_rules:apple_precompiled_resource_bundle.bzl",
    _apple_precompiled_resource_bundle = "apple_precompiled_resource_bundle",
)
load(
    "//apple/internal/resource_rules:apple_resource_bundle.bzl",
    _apple_resource_bundle = "apple_resource_bundle",
)
load(
    "//apple/internal/resource_rules:apple_resource_group.bzl",
    _apple_resource_group = "apple_resource_group",
)

visibility("public")

apple_bundle_import = _apple_bundle_import
apple_intent_library = _apple_intent_library
apple_metal_library = _apple_metal_library
apple_precompiled_resource_bundle = _apple_precompiled_resource_bundle
apple_resource_bundle = _apple_resource_bundle
apple_resource_group = _apple_resource_group
apple_core_data_model = _apple_core_data_model

resources_common = _resources_common

# TODO(b/124103649): Create a proper rule when ObjC compilation is available in Starlark.
def apple_core_ml_library(name, mlmodel, **kwargs):
    # buildifier: disable=function-docstring-args
    """Macro to orchestrate an objc_library with generated sources for mlmodel files."""

    # List of allowed attributes for the apple_core_ml_library rule. Do not want to expose the
    # underlying objc_library attributes which might slow down migration once we're able to create a
    # proper rule.
    allowed_attributes = [
        "tags",
        "testonly",
        "visibility",
    ]

    for attr, _ in kwargs.items():
        if attr not in allowed_attributes:
            fail("Unknown attribute '{}' in rule 'apple_core_ml_library'".format(attr))

    core_ml_name = "{}.CoreML".format(name)

    # Remove visibility from the internal target, to avoid misuse.
    core_ml_args = dict(kwargs)
    core_ml_args.pop("visibility", None)

    objc_source = ":{}.m".format(core_ml_name)
    objc_header = ":{}".format(core_ml_name)
    objc_public_header = "{}.h".format(name)
    _apple_core_ml_library(
        name = core_ml_name,
        mlmodel = mlmodel,
        language = "Objective-C",
        objc_source = objc_source,
        objc_header = objc_header,
        objc_public_header = objc_public_header,
        visibility = ["//visibility:private"],
        **core_ml_args
    )

    objc_library(
        name = name,
        srcs = [objc_source],
        hdrs = [objc_header],
        sdk_frameworks = ["CoreML"],
        data = [mlmodel],
        **kwargs
    )

def swift_apple_core_ml_library(name, mlmodel, **kwargs):
    # buildifier: disable=function-docstring-args
    """Macro to orchestrate a swift_library with generated sources for mlmodel files."""

    # List of allowed attributes for the apple_core_ml_library rule. Do not want to expose the
    # underlying swift_library attributes which might slow down migration once we're able to create a
    # proper rule.
    allowed_attributes = [
        "tags",
        "testonly",
        "visibility",
    ]

    for attr, _ in kwargs.items():
        if attr not in allowed_attributes:
            fail("Unknown attribute '{}' in rule 'swift_apple_core_ml_library'".format(attr))

    core_ml_name = "{}.CoreML".format(name)

    # Remove visibility from the internal target, to avoid misuse.
    core_ml_args = dict(kwargs)
    core_ml_args.pop("visibility", None)

    swift_source = "{}.swift".format(core_ml_name)
    _apple_core_ml_library(
        name = core_ml_name,
        mlmodel = mlmodel,
        language = "Swift",
        swift_source = swift_source,
        visibility = ["//visibility:private"],
        **core_ml_args
    )
    swift_library(
        name = name,
        srcs = [swift_source],
        data = [mlmodel],
        **kwargs
    )

def objc_intent_library(
        name,
        src,
        class_prefix = None,
        testonly = False,
        **kwargs):
    # buildifier: disable=function-docstring-args
    """Macro to orchestrate an objc_library with generated sources for intentdefiniton files."""
    intent_name = "{}.Intent".format(name)
    intent_srcs = "{}.srcs".format(intent_name)
    intent_hdrs = "{}.hdrs".format(intent_name)
    _apple_intent_library(
        name = intent_name,
        src = src,
        language = "Objective-C",
        class_prefix = class_prefix,
        header_name = name,
        tags = ["manual"],
        testonly = testonly,
    )
    native.filegroup(
        name = intent_srcs,
        srcs = [intent_name],
        output_group = "srcs",
        tags = ["manual"],
        testonly = testonly,
    )
    native.filegroup(
        name = intent_hdrs,
        srcs = [intent_name],
        output_group = "hdrs",
        tags = ["manual"],
        testonly = testonly,
    )
    objc_library(
        name = name,
        srcs = [intent_srcs],
        hdrs = [intent_hdrs],
        sdk_frameworks = ["Intents"],
        data = [src],
        testonly = testonly,
        **kwargs
    )

# Note: rules_apples depends on rules_swift, not the other way around. This means
# that apple_intent_library could not be imported in rules_swift and thus this
# macro must live here in rules_apple.
def swift_intent_library(
        name,
        src,
        class_prefix = None,
        class_visibility = None,
        swift_version = None,
        testonly = False,
        **kwargs):
    """
This macro supports the integration of Intents `.intentdefinition` files into Apple rules.

It takes a single `.intentdefinition` file and creates a target that can be added as a dependency from `objc_library` or
`swift_library` targets.

It accepts the regular `swift_library` attributes too.

Args:
    name: A unique name for the target.
    src: Reference to the `.intentdefiniton` file to process.
    class_prefix: Class prefix to use for the generated classes.
    class_visibility: Visibility attribute for the generated classes (`public`, `private`, `project`).
    swift_version: Version of Swift to use for the generated classes.
    testonly: Set to True to enforce that this library is only used from test code.
"""
    intent_name = "{}.Intent".format(name)
    _apple_intent_library(
        name = intent_name,
        src = src,
        language = "Swift",
        class_prefix = class_prefix,
        class_visibility = class_visibility,
        swift_version = swift_version,
        tags = ["manual"],
        testonly = testonly,
    )
    swift_library(
        name = name,
        srcs = [intent_name],
        data = [src],
        testonly = testonly,
        **kwargs
    )
