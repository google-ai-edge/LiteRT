# Copyright 2018 The Bazel Authors. All rights reserved.
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

"""Core resource propagation logic.

Resource are propagated using AppleResourceInfo, in which each field (or bucket) contains data
for resources that should be bundled inside top-level Apple bundles (e.g. ios_application).

Each bucket contains a list of tuples with the following schema:

    (parent_dir, swift_module, resource_files)

    - parent_dir: This is the target path relative to the root of the bundle that will embed the
        resource_files. Each of the resource_files will be copied into a directory structure that
        matches parent_dir. If parent_dir is None, the resources will be placed in the root level.
        For structured resources where the relative path to the target must be preserved,
        parent_dir might look like "some/dir/path". For bundles, parent_dir might look like
        "Resource.bundle".
    - swift_module: This is the name of the Swift module, should the resources had been added
        through a swift_library rule. This is needed as some resource types require this value when
        being compiled (e.g. xibs).
    - resource_files: This is a depset of all the files that should be placed under parent_dir.

During propagation, each target will need to merge multiple AppleResourceInfo providers coming
from dependencies. Merging will then aggressively minimize the tuples in order to only have one
tuple per parent_dir per swift_module per bucket.

AppleResourceInfo also has a `owners` field which contains a map with the short paths of every
resource in the buckets as keys, and a depset of the targets that declare usage as owner of that
resource as values. This dictionary is meant to be used during the deduplication phase, to account
for each usage of the resources in the dependency graph and avoid deduplication if the resource is
used in code higher level bundles. With this, every target is certain that the resource they
reference will be packaged in the same bundle as the code they implement, ensuring that
`[NSBundle bundleForClass:[self class]]` will always return a bundle containing the requested
resource.

In some cases the value for certain keys in `owners` may be None. This value is used to signal that
the target referencing the resource should not be considered the owner, and that the next target in
the dependency chain that can own resources should set itself as the owner. A good example of this
is is the apple_bundle_import rule. This rule doesn't contain any code, so the resources represented
by these targets should not be bound to the apple_bundle_import target, as they should be marked as
being owned by the objc_library or swift_library targets that reference them.

The None values in the `owners` dictionary are then replaced with a default owner in the
`merge_providers` method, which should be called to merge a list of providers into a single
AppleResourceInfo provider to be returned as the provider of the target, and to bundle the
resources contained within in the top-level bundling rules.

This file provides methods to easily:
    - collect all resource files from the different rules and their attributes
    - bucketize each of the resources into specific buckets depending on their path
    - minimize the resulting tuples in order to minimize memory usage
"""

load(
    "@bazel_skylib//lib:partial.bzl",
    "partial",
)
load(
    "@bazel_skylib//lib:paths.bzl",
    "paths",
)
load(
    "@bazel_skylib//lib:types.bzl",
    "types",
)
load(
    "//apple:providers.bzl",
    "AppleFrameworkBundleInfo",
)
load(
    "//apple/internal:providers.bzl",
    "new_appleresourceinfo",
)
load(
    "//apple/internal/partials/support:resources_support.bzl",
    "PROVIDER_TO_FIELD_ACTION",
)
load(
    "//apple/internal/utils:bundle_paths.bzl",
    "bundle_paths",
)

_CACHEABLE_PROVIDER_FIELDS = [
    "infoplists",
    "plists",
    "pngs",
    "strings",
]

def _get_attr_using_list(*, attr, nested_attr, split_attr_key = None):
    """Helper method to always get an attribute as a list within an existing list.

     Args:
        attr: The attributes object on the current context. Can be either a `ctx.attr/ctx.rule.attr`
            -like struct that has targets/lists as its values, or a `ctx.split_attr`-like struct
            with the dictionary fan-out corresponding to split key.
        nested_attr: List of nested attributes to collect values from.
        split_attr_keys: If defined, a 1:2+ transition key to merge values from.

    Returns:
        The found attribute's value within a list, if it is not already a list. Otherwise returns
            the attribute's value if it is a list, or an empty list if the attribute had no value.
    """
    value = getattr(attr, nested_attr)

    value_is_dict = types.is_dict(value)
    if not split_attr_key and value_is_dict:
        fail("Internal Error: Value returned for this attribute is a dictionary, but no split " +
             "attribute key was provided. Attribute was %s." % nested_attr)

    if split_attr_key and value:
        if not value_is_dict:
            fail("Internal Error: Found a split attribute key but the value returned is not a " +
                 "dictionary. Attribute was %s, split key was %s." % (nested_attr, split_attr_key))
        value = value.get(split_attr_key)
    if not value:
        return []
    elif types.is_list(value):
        return value
    else:
        return [value]

def _get_attr_as_list(*, attr, nested_attr, split_attr_keys):
    """Helper method to always get an attribute as a list, supporting 1:2+ transitions.

     Args:
        attr: The attributes object on the current context. Can be either a `ctx.attr/ctx.rule.attr`
            -like struct that has targets/lists as its values, or a `ctx.split_attr`-like struct
            with the dictionary fan-out corresponding to split key.
        nested_attr: List of nested attributes to collect values from.
        split_attr_keys: If `attr` is a 1:2+ transition, a list of 1:2+ transition keys to merge
            values from. Otherwise this must be an empty list.

    Returns:
        The found attribute's value as a list, if a value was found. Otherwise returns an empty
            list if no value was found.
    """
    attr_as_list = []

    if len(split_attr_keys) == 0:
        # If no split keys were defined, search the attribute directly. This is expected to
        # aggregate values across all keys if a 1:2+ transition has been applied to the attribute.
        attr_as_list.extend(_get_attr_using_list(
            attr = attr,
            nested_attr = nested_attr,
        ))
    else:
        # Search the attribute within each split key if any split keys were defined.
        for split_attr_key in split_attr_keys:
            attr_as_list.extend(_get_attr_using_list(
                attr = attr,
                nested_attr = nested_attr,
                split_attr_key = split_attr_key,
            ))
    return attr_as_list

def _bucketize_data(
        *,
        allowed_buckets = None,
        owner = None,
        parent_dir_param = None,
        resources,
        swift_module = None):
    """Separates the given resources into resource bucket types.

    This method takes a list of resources and constructs a tuple object for each, placing it inside
    the correct bucket.

    The parent_dir is calculated from the parent_dir_param object. This object can either be None
    (the default), a string object, or a function object. If a function is provided, it should
    accept only 1 parameter, which will be the File object representing the resource to bucket. This
    mechanism gives us a simpler way to manage multiple use cases. For example, when used to
    bucketize structured resources, the parent_dir_param can be a function that returns the relative
    path to the owning package; or in an objc_library it can be None, signaling that these resources
    should be placed in the root level.

    If no bucket was detected based on the short path for a specific resource, it will be placed
    into the "unprocessed" bucket. Resources in this bucket will not be processed and will be copied
    as is. Once all resources have been placed in buckets, each of the lists will be minimized.

    Finally, it will return a AppleResourceInfo provider with the resources bucketed per type.

    Args:
        allowed_buckets: List of buckets allowed for bucketing. Files that do not fall into these
            buckets will instead be placed into the "unprocessed" bucket. Defaults to `None` which
            means all buckets are allowed.
        owner: An optional string that has a unique identifier to the target that should own the
            resources. If an owner should be passed, it's usually equal to `str(ctx.label)`.
        parent_dir_param: Either a string/None or a struct used to calculate the value of
            parent_dir for each resource. If it is a struct, it will be considered a partial
            context, and will be invoked with partial.call().
        resources: List of resources to bucketize.
        swift_module: The Swift module name to associate to these resources.

    Returns:
        A tuple with a list of owners, a list of "unowned" resources, and a dictionary with
            bucketized resources organized by resource type.
    """
    buckets = {}
    owners = []
    unowned_resources = []

    # Transform the list of buckets to avoid into a set for faster lookup.
    allowed_bucket_set = {}
    if allowed_buckets:
        allowed_bucket_set = {k: None for k in allowed_buckets}

    for target, target_resources in resources.items():
        for resource in target_resources:
            # Local cache of the resource short path since it gets used quite a bit below.
            resource_short_path = resource.short_path

            if owner:
                owners.append((resource_short_path, owner))
            else:
                unowned_resources.append(resource_short_path)

            if types.is_string(parent_dir_param) or parent_dir_param == None:
                parent = parent_dir_param
            else:
                parent = partial.call(partial = parent_dir_param, resource = resource)

            # Special case for localized. If .lproj/ is in the path of the resource (and the parent
            # doesn't already have it) append the lproj component to the current parent.
            if ".lproj/" in resource_short_path and (not parent or ".lproj" not in parent):
                lproj_path = bundle_paths.farthest_parent(resource_short_path, "lproj")
                parent = paths.join(parent or "", paths.basename(lproj_path))

            resource_swift_module = None
            resource_depset = depset([resource])

            # For each type of resource, place in the appropriate bucket.
            if AppleFrameworkBundleInfo in target:
                if "framework.dSYM/" in resource_short_path or resource.extension == "linkmap":
                    # Ignore dSYM bundle and linkmap since the debug symbols partial is
                    # responsible for propagating this up the dependency graph.
                    continue
                bucket_name = "framework"
            elif resource_short_path.endswith(".strings") or resource_short_path.endswith(".stringsdict"):
                bucket_name = "strings"
            elif resource_short_path.endswith(".xcstrings"):
                bucket_name = "xcstrings"
            elif resource_short_path.endswith(".storyboard"):
                bucket_name = "storyboards"
                resource_swift_module = swift_module
            elif resource_short_path.endswith(".xib"):
                bucket_name = "xibs"
                resource_swift_module = swift_module
            elif ".alticon/" in resource_short_path:
                bucket_name = "asset_catalogs"
            elif (".icon/" in resource_short_path or
                  ".xcassets/" in resource_short_path or
                  ".xcstickers/" in resource_short_path):
                bucket_name = "asset_catalogs"
            elif ".xcdatamodel" in resource_short_path or ".xcmappingmodel/" in resource_short_path:
                bucket_name = "datamodels"
                resource_swift_module = swift_module
            elif ".atlas" in resource_short_path:
                bucket_name = "texture_atlases"
            elif resource_short_path.endswith(".png"):
                # Process standalone pngs after asset_catalogs and texture_atlases so the latter can
                # bucketed correctly.
                bucket_name = "pngs"
            elif resource_short_path.endswith(".plist"):
                bucket_name = "plists"

            elif resource_short_path.endswith(".metal") or resource_short_path.endswith(".h"):
                # Assume that any bundled headers are Metal headers for now.
                # Although this will break if anyone wants to bundle headers
                # files as is, it's unlikely that there is such a use case at
                # all.
                bucket_name = "metals"
            elif resource_short_path.endswith(".mlmodel") or resource_short_path.endswith(".mlpackage"):
                bucket_name = "mlmodels"
            else:
                bucket_name = "unprocessed"

            # If the allowed bucket list is not empty, and the bucket is not allowed, change the
            # to unprocessed instead.	            # bucket to unprocessed instead.
            if allowed_bucket_set and bucket_name not in allowed_bucket_set:
                bucket_name = "unprocessed"
                resource_swift_module = None

            buckets.setdefault(bucket_name, []).append(
                (parent, resource_swift_module, resource_depset),
            )

    return (
        owners,
        unowned_resources,
        dict([(k, _minimize(bucket = b)) for k, b in buckets.items()]),
    )

def _bucketize(
        *,
        allowed_buckets = None,
        owner = None,
        parent_dir_param = None,
        resources,
        swift_module = None):
    """Separates the given resources into resource bucket types and returns an AppleResourceInfo.

    This method wraps _bucketize_data and returns its tuple as an immutable Starlark structure to
    help propagate the structure of the Apple bundle resources to the bundler.

    Args:
        allowed_buckets: List of buckets allowed for bucketing. Files that do not fall into these
            buckets will instead be placed into the "unprocessed" bucket. Defaults to `None` which
            means all buckets are allowed.
        owner: An optional string that has a unique identifier to the target that should own the
            resources. If an owner should be passed, it's usually equal to `str(ctx.label)`.
        parent_dir_param: Either a string/None or a struct used to calculate the value of
            parent_dir for each resource. If it is a struct, it will be considered a partial
            context, and will be invoked with partial.call().
        resources: List of resources to bucketize.
        swift_module: The Swift module name to associate to these resources.

    Returns:
        An AppleResourceInfo provider with resources bucketized according to type.
    """
    owners, unowned_resources, buckets = _bucketize_data(
        resources = resources,
        swift_module = swift_module,
        owner = owner,
        parent_dir_param = parent_dir_param,
        allowed_buckets = allowed_buckets,
    )
    return new_appleresourceinfo(
        owners = depset(owners),
        unowned_resources = depset(unowned_resources),
        **buckets
    )

def _bucketize_typed_data(*, bucket_type, owner = None, parent_dir_param = None, resources):
    """Collects and bucketizes a specific type of resource.

    Adds the given resources directly into a tuple under the field named in bucket_type. This avoids
    the sorting mechanism that `bucketize` does, while grouping resources together using
    parent_dir_param when available.

    Args:
        bucket_type: The AppleResourceInfo field under which to collect the resources.
        owner: An optional string that has a unique identifier to the target that should own the
            resources. If an owner should be passed, it's usually equal to `str(ctx.label)`.
        parent_dir_param: Either a string/None or a struct used to calculate the value of
            parent_dir for each resource. If it is a struct, it will be considered a partial
            context, and will be invoked with partial.call().
        resources: List of resources to place in bucket_type or Dictionary of resources keyed by
            target to place in bucket_type. This dictionary is supported by the
            `resources.collect()` API.

    Returns:
        A tuple with a list of owners, a list of "unowned" resources, and a dictionary with
            bucketized resources that are all placed within a single bucket defined by bucket_type.
    """
    typed_bucket = []
    owners = []
    unowned_resources = []

    all_resources = []
    if type(resources) == "list":
        all_resources = resources
    elif type(resources) == "dict":
        for target_resources in resources.values():
            all_resources.extend(target_resources)
    else:
        fail("Internal error: 'resources' should be either a list or dictionary.\n" +
             "This is most likely a rules_apple bug, please file a bug with reproduction steps")

    for resource in all_resources:
        resource_short_path = resource.short_path
        if owner:
            owners.append((resource_short_path, owner))
        else:
            unowned_resources.append(resource_short_path)

        if types.is_string(parent_dir_param) or parent_dir_param == None:
            parent = parent_dir_param
        else:
            parent = partial.call(partial = parent_dir_param, resource = resource)

        if ".lproj/" in resource_short_path and (not parent or ".lproj" not in parent):
            lproj_path = bundle_paths.farthest_parent(resource_short_path, "lproj")
            parent = paths.join(parent or "", paths.basename(lproj_path))

        typed_bucket.append((parent, None, depset(direct = [resource])))

    return (
        owners,
        unowned_resources,
        dict([(bucket_type, _minimize(bucket = typed_bucket))]),
    )

def _bucketize_typed(resources, bucket_type, *, owner = None, parent_dir_param = None):
    """Collects and bucketizes a specific type of resource and returns an AppleResourceInfo.

    Adds the given resources directly into a tuple under the field named in bucket_type. This avoids
    the sorting mechanism that `bucketize` does, while grouping resources together using
    parent_dir_param when available.

    Args:
        bucket_type: The AppleResourceInfo field under which to collect the resources.
        owner: An optional string that has a unique identifier to the target that should own the
            resources. If an owner should be passed, it's usually equal to `str(ctx.label)`.
        parent_dir_param: Either a string/None or a struct used to calculate the value of
            parent_dir for each resource. If it is a struct, it will be considered a partial
            context, and will be invoked with partial.call().
        resources: List of resources to place in bucket_type or Dictionary of resources keyed by
            target to place in bucket_type. This dictionary is supported by the
            `resources.collect()` API.

    Returns:
        An AppleResourceInfo provider with resources in the given bucket.
    """
    owners, unowned_resources, buckets = _bucketize_typed_data(
        bucket_type = bucket_type,
        owner = owner,
        parent_dir_param = parent_dir_param,
        resources = resources,
    )

    return new_appleresourceinfo(
        owners = depset(owners),
        unowned_resources = depset(unowned_resources),
        **buckets
    )

def _process_bucketized_data(
        *,
        actions,
        apple_mac_toolchain_info,
        bucketized_owners = [],
        buckets,
        bundle_id,
        output_discriminator = None,
        platform_prerequisites,
        processing_owner = None,
        product_type,
        resource_types_to_process = _CACHEABLE_PROVIDER_FIELDS,
        rule_label,
        unowned_resources = []):
    """Registers actions for select resource types, given bucketized groupings of data.

    This method performs the same actions as bucketize_data, and further
    iterates through a subset of resource types to register actions to process
    them as necessary before returning an AppleResourceInfo. This
    AppleResourceInfo has an additional field, called "processed", featuring the
    expected outputs for each of the actions declared in this method.

    Args:
        actions: The actions provider from `ctx.actions`.
        apple_mac_toolchain_info: `struct` of tools from the shared Apple
            toolchain.
        bucketized_owners: A list of tuples indicating the owner of each
            bucketized resource.
        buckets: A dictionary with bucketized resources organized by resource
            type.
        bundle_id: The bundle ID to configure for this target.
        output_discriminator: A string to differentiate between different target
            intermediate files or `None`.
        platform_prerequisites: Struct containing information on the platform
            being targeted.
        processing_owner: An optional string that has a unique identifier to the
            target that should own the resources. If an owner should be passed,
            it's usually equal to `str(ctx.label)`.
        product_type: The product type identifier used to describe the current
            bundle type.
        resource_types_to_process: A list of bucket types to process.
        rule_label: The label of the target being analyzed.
        unowned_resources: A list of "unowned" resources.

    Returns:
        An AppleResourceInfo provider with resources bucketized according to
        type.
    """

    # Keep a list to reference what the processed files are based from.
    processed_origins = []

    field_to_action_map = {}
    for field in resource_types_to_process:
        action = PROVIDER_TO_FIELD_ACTION.get(field)
        if action:
            field_to_action_map[field] = action

    for bucket_name, bucket_action in field_to_action_map.items():
        processed_field = buckets.pop(bucket_name, default = None)
        if not processed_field:
            continue
        for parent_dir, swift_module, files in processed_field:
            processing_func, requires_swift_module = bucket_action

            processing_args = {
                "actions": actions,
                "apple_mac_toolchain_info": apple_mac_toolchain_info,
                "bundle_id": bundle_id,
                "files": files,
                "output_discriminator": output_discriminator,
                "parent_dir": parent_dir,
                "platform_prerequisites": platform_prerequisites,
                # If asset catalogs are being processed, it's not the top-level
                # target, so we don't support setting the primary icon name.
                "primary_icon_name": None,
                "product_type": product_type,
                "rule_label": rule_label,
            }

            # Only pass the Swift module name if the resource to process requires it.
            if requires_swift_module:
                processing_args["swift_module"] = swift_module

            # Execute the processing function.
            result = processing_func(**processing_args)

            # Store each origin as a tuple in an array, to keep this knowledge as a low-memory
            # reference within a depset.
            for processed_resource, processed_origin in getattr(result, "processed_origins", {}).items():
                processed_origins.append((processed_resource, tuple(processed_origin)))

            processed_field = {}
            for _, processed_parent_dir, processed_files in result.files:
                processed_field.setdefault(
                    processed_parent_dir or "",
                    [],
                ).append(processed_files)

            # Save files to the "processed" field for copying in the bundling phase.
            for processed_parent_dir, processed_files in processed_field.items():
                buckets.setdefault(
                    "processed",
                    default = [],
                ).append((
                    processed_parent_dir,
                    swift_module,
                    depset(transitive = processed_files),
                ))

            # Add owners information for each of the processed files.
            for _, _, processed_files in result.files:
                for processed_file in processed_files.to_list():
                    if processing_owner:
                        bucketized_owners.append((processed_file.short_path, processing_owner))
                    else:
                        unowned_resources.append(processed_file.short_path)

    return new_appleresourceinfo(
        owners = depset(bucketized_owners),
        unowned_resources = depset(unowned_resources),
        processed_origins = depset(processed_origins),
        **buckets
    )

def _bundle_relative_parent_dir(resource, extension):
    """Returns the bundle relative path to the resource rooted at the bundle.

    Looks for the first instance of a folder with the suffix specified by `extension`, and then
    returns the directory path to the file within the bundle. For example, for a resource with path
    my/package/Contents.bundle/directory/foo.txt and `extension` equal to `"bundle"`, it would
    return Contents.bundle/directory.

    Args:
        resource: The resource for which to calculate the bundle relative path.
        extension: The bundle extension to use when finding the relative path.

    Returns:
        The bundle relative path, rooted at the outermost bundle.
    """
    bundle_path = bundle_paths.farthest_parent(resource.short_path, extension)
    bundle_relative_path = paths.relativize(resource.short_path, bundle_path)

    parent_dir = paths.basename(bundle_path)
    bundle_relative_dir = paths.dirname(bundle_relative_path).strip("/")
    if bundle_relative_dir:
        parent_dir = paths.join(parent_dir, bundle_relative_dir)
    return parent_dir

def _collect(*, attr, res_attrs = [], split_attr_keys = []):
    """Collects all resource attributes present in the given attributes.

    Iterates over the given res_attrs attributes collecting files to be processed as resources.
    These are all placed into a list, and then returned.

    Args:
        attr: The attributes object on the current context. Can be either a `ctx.attr/ctx.rule.attr`
            -like struct that has targets/lists as its values, or a `ctx.split_attr`-like struct
            with the dictionary fan-out corresponding to split key.
        res_attrs: List of attributes to iterate over collecting resources.
        split_attr_keys: If defined, a list of 1:2+ transition keys to merge values from.

    Returns:
        A dictionary keyed by target from the rule attr with the list of all collected resources.
    """
    if not res_attrs:
        return []

    files_by_target = {}
    for res_attr in res_attrs:
        if not hasattr(attr, res_attr):
            continue

        targets_for_attr = _get_attr_as_list(
            attr = attr,
            nested_attr = res_attr,
            split_attr_keys = split_attr_keys,
        )
        for target in targets_for_attr:
            if not target.files:
                # Target does not export any file, ignore.
                continue
            files_by_target.setdefault(target, []).extend(target.files.to_list())

    return files_by_target

def _merge_providers(*, default_owner = None, providers, validate_all_resources_owned = False):
    """Merges multiple AppleResourceInfo providers into one.

    Args:
        default_owner: The default owner to be used for resources which have a None value in the
            `owners` dictionary. May be None, in which case no owner is marked.
        providers: The list of providers to merge. This method will fail unless there is at least 1
            provider in the list.
        validate_all_resources_owned: Whether to validate that all resources are owned. This is
            useful for top-level rules to ensure that the resources in AppleResourceInfo that
            they are propagating are fully owned. If default_owner is set, this attribute does
            nothing, as by definition the resources will all have a default owner.

    Returns:
        A AppleResourceInfo provider with the results of the merge of the given providers.
    """
    if not providers:
        fail(
            "merge_providers should be called with a non-empty list of providers. This is most " +
            "likely a bug in rules_apple, please file a bug with reproduction steps.",
        )

    if not default_owner and validate_all_resources_owned == False and len(providers) == 1:
        # Short path to avoid the merging and validation loops if the loop won't change the owners
        # mapping nor validate that all resources are marked as owned.
        return providers[0]

    buckets = {}

    for provider in providers:
        # Get the initialized fields in the provider, with the exception of to_json and to_proto,
        # which are not desireable for our use case.
        fields = _populated_resource_fields(provider)
        for field in fields:
            buckets.setdefault(
                field,
                [],
            ).extend(getattr(provider, field))

    # unowned_resources is a depset of resource paths.
    unowned_resources = depset(transitive = [provider.unowned_resources for provider in providers])

    # processed_origins is a depset of processed resources to lists of resources.
    processed_origins_list = [
        provider.processed_origins
        for provider in providers
        if getattr(provider, "processed_origins", None)
    ]
    if processed_origins_list:
        processed_origins = depset(transitive = processed_origins_list)
    else:
        processed_origins = None

    # owners is a depset of (resource_path, owner) pairs.
    transitive_owners = [provider.owners for provider in providers]

    # If owner is set, this rule now owns all previously unowned resources.
    if default_owner:
        transitive_owners.append(
            depset([(resource, default_owner) for resource in unowned_resources.to_list()]),
        )
        unowned_resources = depset()
    elif validate_all_resources_owned:
        if unowned_resources.to_list():
            fail(
                "The given providers have a resource that doesn't have an owner, and " +
                "validate_all_resources_owned was set. This is most likely a bug in " +
                "rules_apple, please file a bug with reproduction steps.",
            )

    return new_appleresourceinfo(
        owners = depset(transitive = transitive_owners),
        unowned_resources = unowned_resources,
        processed_origins = processed_origins,
        **dict([(k, _minimize(bucket = v)) for (k, v) in buckets.items()])
    )

def _minimize(*, bucket):
    """Minimizes the given list of tuples into the smallest subset possible.

    Takes the list of tuples that represent one resource bucket, and minimizes it so that 2 tuples
    with resources that should be placed under the same location are merged into 1 tuple.

    For tuples to be merged, they need to have the same parent_dir and swift_module.

    Args:
        bucket: List of tuples to be minimized.

    Returns:
        A list of minimized tuples.
    """
    resources_by_key = {}

    # Use these maps to keep track of the parent_dir and swift_module values.
    parent_dir_by_key = {}
    swift_module_by_key = {}

    for parent_dir, swift_module, resources in bucket:
        # TODO(b/275385433): Audit Starlark performance of different string interpolation methods.
        # Particularly for the resource aspect, using the '%' operator yielded better results than
        # using `str.format` and string concatenation.
        key = "%s_%s" % (parent_dir or "@root", swift_module or "@root")

        if parent_dir:
            parent_dir_by_key[key] = parent_dir
        if swift_module:
            swift_module_by_key[key] = swift_module

        # TODO(b/275385433): Audit Starlark performance of `dict.setdefault` vs. if/else statements.
        # Particularly for the resource aspect, using if/else statements yielded better results than
        # using `dict.setdefault` (from apple/internal/resources.bzl).
        if key in resources_by_key:
            resources_by_key[key].append(resources)
        else:
            resources_by_key[key] = [resources]

    return [
        (parent_dir_by_key.get(k, None), swift_module_by_key.get(k, None), depset(transitive = r))
        for k, r in resources_by_key.items()
    ]

def _nest_in_bundle(*, provider_to_nest, nesting_bundle_dir):
    """Nests resources in a AppleResourceInfo provider under a new parent bundle directory.

    This method is mostly used by rules that create resource bundles in order to nest other resource
    bundle targets within themselves. For instance, apple_resource_bundle supports nesting other
    bundles through the resources attribute. In these use cases, the dependency bundles are added as
    nested bundles into the dependent bundle.

    This method prepends the parent_dir field in the buckets with the given
    nesting_bundle_dir argument.

    Args:
        provider_to_nest: A AppleResourceInfo provider with the resources to nest.
        nesting_bundle_dir: The new bundle directory under which to bundle the resources.

    Returns:
        A new AppleResourceInfo provider with the resources nested under nesting_bundle_dir.
    """
    nested_provider_fields = {}
    for field in _populated_resource_fields(provider_to_nest):
        for parent_dir, swift_module, files in getattr(provider_to_nest, field):
            if parent_dir:
                nested_parent_dir = paths.join(nesting_bundle_dir, parent_dir)
            else:
                nested_parent_dir = nesting_bundle_dir
            nested_provider_fields.setdefault(field, []).append(
                (nested_parent_dir, swift_module, files),
            )

    return new_appleresourceinfo(
        owners = provider_to_nest.owners,
        unowned_resources = provider_to_nest.unowned_resources,
        processed_origins = getattr(provider_to_nest, "processed_origins", None),
        **nested_provider_fields
    )

def _populated_resource_fields(provider):
    """Returns a list of field names of the provider's resource buckets that are non empty."""

    # TODO(b/36412967): Remove the to_json and to_proto elements of this list.
    return [
        f
        for f in dir(provider)
        if f not in ["owners", "unowned_resources", "processed_origins", "to_json", "to_proto"]
    ]

def _structured_resources_parent_dir(
        *,
        parent_dir = None,
        resource,
        strip_prefixes = []):
    """Returns the package relative path for the parent directory of a resource.

    Args:
        parent_dir: Parent directory to prepend to the package relative path.
        resource: The resource for which to calculate the package relative path.
        strip_prefixes: A list of prefixes to strip from the package relative
            path. The first prefix that matches will be used.

    Returns:
        The package relative path to the parent directory of the resource.
    """
    package_relative = bundle_paths.owner_relative_path(resource)
    if resource.is_directory:
        path = package_relative
    else:
        path = paths.dirname(package_relative).rstrip("/")

    for prefix in strip_prefixes:
        if path.startswith(prefix):
            path = path[(len(prefix) + 1):]
            break

    return paths.join(parent_dir or "", path or "") or None

def _runfiles_resources_parent_dir(*, resource):
    """Returns the parent directory of the file.

    Args:
        resource: The resource for which to calculate the package relative path.

    Returns:
        The package relative path to the parent directory of the resource.
    """
    return paths.dirname(resource.path)

def _expand_owners(*, owners):
    """Converts a depset of (path, owner) to a dict of paths to dict of owners.

    Args:
      owners: A depset of (path, owner) pairs.
    """
    dict = {}
    for resource, owner in owners.to_list():
        if owner:
            dict.setdefault(resource, {})[owner] = None
    return dict

def _expand_processed_origins(*, processed_origins):
    """Converts a depset of (processed_resource, resource) to a dict.

    Args:
      processed_origins: A depset of (processed_resource, resource) pairs.
    """
    processed_origins_dict = {}
    for processed_resource, resource in processed_origins.to_list():
        processed_origins_dict[processed_resource] = resource
    return processed_origins_dict

def _deduplicate_field(
        *,
        avoid_owners,
        avoid_provider,
        field,
        owners,
        processed_origins,
        processed_deduplication_map,
        resources_provider):
    """Deduplicates and returns resources between 2 providers for a given field.

    Deduplication happens by comparing the target path of a file and the files
    themselves. If there are 2 resources with the same target path but different
    contents, the files will not be deduplicated.

    This approach is naïve in the sense that it deduplicates resources too
    aggressively. We also need to compare the target that references the
    resources so that they are not deduplicated if they are referenced within
    multiple binary-containing bundles.

    Args:
      avoid_owners: The owners map for avoid_provider computed by _expand_owners.
      avoid_provider: The provider with the resources to avoid bundling.
      field: The field to deduplicate resources on.
      resources_provider: The provider with the resources to be bundled.
      owners: The owners map for resources_provider computed by _expand_owners.
      processed_origins: The processed resources map for resources_provider computed by
          _expand_processed_origins.
      processed_deduplication_map: A dictionary of keys to lists of short paths referencing already-
          deduplicated resources that can be referenced by the resource processing aspect to avoid
          duplicating files referenced by library targets and top level targets.

    Returns:
      A list of tuples with the resources present in avoid_provider removed from
      resources_provider.
    """

    avoid_dict = {}
    if avoid_provider and hasattr(avoid_provider, field):
        for parent_dir, swift_module, files in getattr(avoid_provider, field):
            key = "%s_%s" % (parent_dir or "root", swift_module or "root")
            avoid_dict[key] = {x.short_path: None for x in files.to_list()}

    # Get the resources to keep, compare them to the avoid_dict under the same
    # key, and remove the duplicated file references. Then recreate the original
    # tuple with only the remaining files, if any.
    deduped_tuples = []

    for parent_dir, swift_module, files in getattr(resources_provider, field):
        key = "%s_%s" % (parent_dir or "root", swift_module or "root")

        # Dictionary used as a set to mark files as processed by short_path to deduplicate generated
        # files that may appear more than once if multiple architectures are being built.
        multi_architecture_deduplication_set = {}

        # Update the deduplication map for this key, representing the domain of this library
        # processable resource in bundling, and use that as our deduplication list for library
        # processable resources.
        if not processed_deduplication_map.get(key, None):
            processed_deduplication_map[key] = []
        processed_deduplication_list = processed_deduplication_map[key]

        deduped_files = []
        for to_bundle_file in files.to_list():
            short_path = to_bundle_file.short_path
            if short_path in multi_architecture_deduplication_set:
                continue
            multi_architecture_deduplication_set[short_path] = None
            if key in avoid_dict and short_path in avoid_dict[key]:
                # If the resource file is present in the provider of resources to avoid, we compare
                # the owners of the resource through the owners dictionaries of the providers. If
                # there are owners present in resources_provider which are not present in
                # avoid_provider, it means that there is at least one target that declares usage of
                # the resource which is not accounted for in avoid_provider. If this is the case, we
                # add the resource to be bundled in the bundle represented by resources_provider.
                deduped_owners = [
                    o
                    for o in owners[short_path]
                    if o not in avoid_owners[short_path]
                ]
                if not deduped_owners:
                    continue

            if field == "processed":
                # Check for duplicates referencing our map of where the processed resources were
                # based from.
                path_origins = processed_origins[short_path]
                if path_origins in processed_deduplication_list:
                    continue
                processed_deduplication_list.append(path_origins)
            elif field in _CACHEABLE_PROVIDER_FIELDS:
                # Check for duplicates across fields that can be processed by a resource aspect, to
                # avoid dupes between top-level fields and fields processed by the resource aspect.
                all_path_origins = [
                    path_origin
                    for path_origins in processed_deduplication_list
                    for path_origin in path_origins
                ]
                if short_path in all_path_origins:
                    continue
                processed_deduplication_list.append([short_path])

            deduped_files.append(to_bundle_file)

        if deduped_files:
            deduped_tuples.append((parent_dir, swift_module, depset(deduped_files)))

    return deduped_tuples

def _deduplicate(*, resources_provider, avoid_providers, field_handler, default_owner = None):
    avoid_provider = None
    if avoid_providers:
        # Call merge_providers with validate_all_resources_owned set, to ensure that all the
        # resources from dependency bundles have an owner.
        avoid_provider = _merge_providers(
            default_owner = default_owner,
            providers = avoid_providers,
            validate_all_resources_owned = True,
        )

    fields = _populated_resource_fields(resources_provider)

    # Precompute owners, avoid_owners and processed_origins to avoid duplicate work in
    # _deduplicate_field.
    # Build a dictionary with the file paths under each key for the avoided resources.
    avoid_owners = {}
    if avoid_provider:
        avoid_owners = _expand_owners(owners = avoid_provider.owners)
    owners = _expand_owners(owners = resources_provider.owners)
    if resources_provider.processed_origins:
        processed_origins = _expand_processed_origins(
            processed_origins = resources_provider.processed_origins,
        )
    else:
        processed_origins = {}

    # Create the deduplication map for library processable resources to be referenced across fields
    # for the purposes of deduplicating top level resources and multiple library scoped resources.
    processed_deduplication_map = {}

    for field in fields:
        deduplicated = _deduplicate_field(
            avoid_owners = avoid_owners,
            avoid_provider = avoid_provider,
            field = field,
            owners = owners,
            processed_origins = processed_origins,
            processed_deduplication_map = processed_deduplication_map,
            resources_provider = resources_provider,
        )
        field_handler(field, deduplicated)

resources = struct(
    bucketize = _bucketize,
    bucketize_data = _bucketize_data,
    bucketize_typed = _bucketize_typed,
    bucketize_typed_data = _bucketize_typed_data,
    bundle_relative_parent_dir = _bundle_relative_parent_dir,
    collect = _collect,
    deduplicate = _deduplicate,
    merge_providers = _merge_providers,
    minimize = _minimize,
    nest_in_bundle = _nest_in_bundle,
    populated_resource_fields = _populated_resource_fields,
    process_bucketized_data = _process_bucketized_data,
    runfiles_resources_parent_dir = _runfiles_resources_parent_dir,
    structured_resources_parent_dir = _structured_resources_parent_dir,
)
