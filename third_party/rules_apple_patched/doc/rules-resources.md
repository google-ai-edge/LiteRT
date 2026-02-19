<!-- Generated with Stardoc: http://skydoc.bazel.build -->

# Rules related to Apple resources and resource bundles.

<a id="apple_bundle_import"></a>

## apple_bundle_import

<pre>
load("@rules_apple//apple:resources.bzl", "apple_bundle_import")

apple_bundle_import(<a href="#apple_bundle_import-name">name</a>, <a href="#apple_bundle_import-bundle_imports">bundle_imports</a>)
</pre>

This rule encapsulates an already-built bundle. It is defined by a list of files
in exactly one `.bundle` directory. `apple_bundle_import` targets need to be
added to library targets through the `data` attribute, or to other resource
targets (i.e. `apple_resource_bundle` and `apple_resource_group`) through the
`resources` attribute.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_bundle_import-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_bundle_import-bundle_imports"></a>bundle_imports |  The list of files under a `.bundle` directory to be propagated to the top-level bundling target.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |


<a id="apple_core_data_model"></a>

## apple_core_data_model

<pre>
load("@rules_apple//apple:resources.bzl", "apple_core_data_model")

apple_core_data_model(<a href="#apple_core_data_model-name">name</a>, <a href="#apple_core_data_model-srcs">srcs</a>, <a href="#apple_core_data_model-outs">outs</a>, <a href="#apple_core_data_model-swift_version">swift_version</a>)
</pre>

This rule takes one or more Core Data model definitions from .xcdatamodeld
bundles and generates Swift or Objective-C source files that can be added
as srcs of a swift_library target.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_core_data_model-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_core_data_model-srcs"></a>srcs |  -   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="apple_core_data_model-outs"></a>outs |  A dictionary where the key is the name of a data model and the value is a list of source files expected to be generated from that data model. For example, if srcs contains one data model called "Taxonomy.xcdatamodeld" with a single entity called "Animal," you might provide this value: <pre><code>outs = {&#10;    "Taxonomy": [&#10;        "taxonomy+CoreDataModel.swift",&#10;        "Animal+CoreDataProperties.swift",&#10;    ],&#10;},</code></pre> If one or more files are provided for a data model, the rule will return these files individually as outputs. Otherwise, the rule will return the directory containing the sources for the data model.   | <a href="https://bazel.build/rules/lib/dict">Dictionary: String -> List of strings</a> | optional |  `{}`  |
| <a id="apple_core_data_model-swift_version"></a>swift_version |  Target Swift version for generated classes.   | String | optional |  `""`  |


<a id="apple_intent_library"></a>

## apple_intent_library

<pre>
load("@rules_apple//apple:resources.bzl", "apple_intent_library")

apple_intent_library(<a href="#apple_intent_library-name">name</a>, <a href="#apple_intent_library-src">src</a>, <a href="#apple_intent_library-class_prefix">class_prefix</a>, <a href="#apple_intent_library-class_visibility">class_visibility</a>, <a href="#apple_intent_library-header_name">header_name</a>, <a href="#apple_intent_library-language">language</a>,
                     <a href="#apple_intent_library-swift_version">swift_version</a>)
</pre>

This rule supports the integration of Intents `.intentdefinition` files into Apple rules.
It takes a single `.intentdefinition` file and creates a target that can be added as a dependency from `objc_library` or
`swift_library` targets. It accepts the regular `objc_library` attributes too.
This target generates a header named `<target_name>.h` that can be imported from within the package where this target
resides. For example, if this target's label is `//my/package:intent`, you can import the header as
`#import "my/package/intent.h"`.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_intent_library-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_intent_library-src"></a>src |  Label to a single `.intentdefinition` files from which to generate sources files.   | <a href="https://bazel.build/concepts/labels">Label</a> | required |  |
| <a id="apple_intent_library-class_prefix"></a>class_prefix |  Class prefix to use for the generated classes.   | String | optional |  `""`  |
| <a id="apple_intent_library-class_visibility"></a>class_visibility |  Visibility attribute for the generated classes ("public", "private", "project").   | String | optional |  `""`  |
| <a id="apple_intent_library-header_name"></a>header_name |  Name of the public header file (only when using Objective-C).   | String | optional |  `""`  |
| <a id="apple_intent_library-language"></a>language |  Language of generated classes ("Objective-C", "Swift")   | String | required |  |
| <a id="apple_intent_library-swift_version"></a>swift_version |  Version of Swift to use for the generated classes.   | String | optional |  `""`  |


<a id="apple_metal_library"></a>

## apple_metal_library

<pre>
load("@rules_apple//apple:resources.bzl", "apple_metal_library")

apple_metal_library(<a href="#apple_metal_library-name">name</a>, <a href="#apple_metal_library-srcs">srcs</a>, <a href="#apple_metal_library-out">out</a>, <a href="#apple_metal_library-hdrs">hdrs</a>, <a href="#apple_metal_library-copts">copts</a>, <a href="#apple_metal_library-includes">includes</a>)
</pre>

Compiles Metal shader language sources into a Metal library.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_metal_library-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_metal_library-srcs"></a>srcs |  A list of `.metal` source files that will be compiled into the library.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | required |  |
| <a id="apple_metal_library-out"></a>out |  An output `.metallib` filename. Defaults to `default.metallib` if unspecified.   | String | optional |  `"default.metallib"`  |
| <a id="apple_metal_library-hdrs"></a>hdrs |  A list of headers to make importable when compiling the metal library.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_metal_library-copts"></a>copts |  A list of compiler options passed to the `metal` compiler for each source.   | List of strings | optional |  `[]`  |
| <a id="apple_metal_library-includes"></a>includes |  A list of header search paths.   | List of strings | optional |  `[]`  |


<a id="apple_precompiled_resource_bundle"></a>

## apple_precompiled_resource_bundle

<pre>
load("@rules_apple//apple:resources.bzl", "apple_precompiled_resource_bundle")

apple_precompiled_resource_bundle(<a href="#apple_precompiled_resource_bundle-name">name</a>, <a href="#apple_precompiled_resource_bundle-resources">resources</a>, <a href="#apple_precompiled_resource_bundle-bundle_id">bundle_id</a>, <a href="#apple_precompiled_resource_bundle-bundle_name">bundle_name</a>, <a href="#apple_precompiled_resource_bundle-infoplists">infoplists</a>,
                                  <a href="#apple_precompiled_resource_bundle-strip_structured_resources_prefixes">strip_structured_resources_prefixes</a>, <a href="#apple_precompiled_resource_bundle-structured_resources">structured_resources</a>)
</pre>

This rule encapsulates a target which is provided to dependers as a bundle. An
`apple_precompiled_resource_bundle`'s resources are put in a resource bundle in the top
level Apple bundle dependent. `apple_precompiled_resource_bundle` targets need to be added to
library targets through the `data` attribute.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_precompiled_resource_bundle-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_precompiled_resource_bundle-resources"></a>resources |  Files to include in the resource bundle. Files that are processable resources, like .xib, .storyboard, .strings, .png, and others, will be processed by the Apple bundling rules that have those files as dependencies. Other file types that are not processed will be copied verbatim. These files are placed in the root of the resource bundle (e.g. `Payload/foo.app/bar.bundle/...`) in most cases. However, if they appear to be localized (i.e. are contained in a directory called *.lproj), they will be placed in a directory of the same name in the app bundle.<br><br>You can also add other `apple_precompiled_resource_bundle` and `apple_bundle_import` targets into `resources`, and the resource bundle structures will be propagated into the final bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_precompiled_resource_bundle-bundle_id"></a>bundle_id |  The bundle ID for this target. It will replace `$(PRODUCT_BUNDLE_IDENTIFIER)` found in the files from defined in the `infoplists` paramter.   | String | optional |  `""`  |
| <a id="apple_precompiled_resource_bundle-bundle_name"></a>bundle_name |  The desired name of the bundle (without the `.bundle` extension). If this attribute is not set, then the `name` of the target will be used instead.   | String | optional |  `""`  |
| <a id="apple_precompiled_resource_bundle-infoplists"></a>infoplists |  A list of `.plist` files that will be merged to form the `Info.plist` that represents the extension. At least one file must be specified. Please see [Info.plist Handling](/doc/common_info.md#infoplist-handling") for what is supported.<br><br>Duplicate keys between infoplist files will cause an error if and only if the values conflict. Bazel will perform variable substitution on the Info.plist file for the following values (if they are strings in the top-level dict of the plist):<br><br>${BUNDLE_NAME}: This target's name and bundle suffix (.bundle or .app) in the form name.suffix. ${PRODUCT_NAME}: This target's name. ${TARGET_NAME}: This target's name. The key in ${} may be suffixed with :rfc1034identifier (for example ${PRODUCT_NAME::rfc1034identifier}) in which case Bazel will replicate Xcode's behavior and replace non-RFC1034-compliant characters with -.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_precompiled_resource_bundle-strip_structured_resources_prefixes"></a>strip_structured_resources_prefixes |  A list of prefixes to strip from the paths of structured resources. For each structured resource, if the path starts with one of these prefixes, the first matching prefix will be removed from the path when the resource is placed in the bundle root. This is useful for removing intermediate directories from the resource paths.<br><br>For example, if `structured_resources` contains `["intermediate/res/foo.png"]`, and `strip_structured_resources_prefixes` contains `["intermediate"]`, `res/foo.png` will end up inside the bundle.   | List of strings | optional |  `[]`  |
| <a id="apple_precompiled_resource_bundle-structured_resources"></a>structured_resources |  Files to include in the final resource bundle. They are not processed or compiled in any way besides the processing done by the rules that actually generate them. These files are placed in the bundle root in the same structure passed to this argument, so `["res/foo.png"]` will end up in `res/foo.png` inside the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |


<a id="apple_resource_bundle"></a>

## apple_resource_bundle

<pre>
load("@rules_apple//apple:resources.bzl", "apple_resource_bundle")

apple_resource_bundle(<a href="#apple_resource_bundle-name">name</a>, <a href="#apple_resource_bundle-resources">resources</a>, <a href="#apple_resource_bundle-bundle_id">bundle_id</a>, <a href="#apple_resource_bundle-bundle_name">bundle_name</a>, <a href="#apple_resource_bundle-infoplists">infoplists</a>,
                      <a href="#apple_resource_bundle-strip_structured_resources_prefixes">strip_structured_resources_prefixes</a>, <a href="#apple_resource_bundle-structured_resources">structured_resources</a>)
</pre>

This rule encapsulates a target which is provided to dependers as a bundle. An
`apple_resource_bundle`'s resources are put in a resource bundle in the top
level Apple bundle dependent. apple_resource_bundle targets need to be added to
library targets through the `data` attribute.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_resource_bundle-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_resource_bundle-resources"></a>resources |  Files to include in the resource bundle. Files that are processable resources, like .xib, .storyboard, .strings, .png, and others, will be processed by the Apple bundling rules that have those files as dependencies. Other file types that are not processed will be copied verbatim. These files are placed in the root of the resource bundle (e.g. `Payload/foo.app/bar.bundle/...`) in most cases. However, if they appear to be localized (i.e. are contained in a directory called *.lproj), they will be placed in a directory of the same name in the app bundle.<br><br>You can also add other `apple_resource_bundle` and `apple_bundle_import` targets into `resources`, and the resource bundle structures will be propagated into the final bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_resource_bundle-bundle_id"></a>bundle_id |  The bundle ID for this target. It will replace `$(PRODUCT_BUNDLE_IDENTIFIER)` found in the files from defined in the `infoplists` paramter.   | String | optional |  `""`  |
| <a id="apple_resource_bundle-bundle_name"></a>bundle_name |  The desired name of the bundle (without the `.bundle` extension). If this attribute is not set, then the `name` of the target will be used instead.   | String | optional |  `""`  |
| <a id="apple_resource_bundle-infoplists"></a>infoplists |  A list of `.plist` files that will be merged to form the `Info.plist` that represents the extension. At least one file must be specified. Please see [Info.plist Handling](/doc/common_info.md#infoplist-handling") for what is supported.<br><br>Duplicate keys between infoplist files will cause an error if and only if the values conflict. Bazel will perform variable substitution on the Info.plist file for the following values (if they are strings in the top-level dict of the plist):<br><br>${BUNDLE_NAME}: This target's name and bundle suffix (.bundle or .app) in the form name.suffix. ${PRODUCT_NAME}: This target's name. ${TARGET_NAME}: This target's name. The key in ${} may be suffixed with :rfc1034identifier (for example ${PRODUCT_NAME::rfc1034identifier}) in which case Bazel will replicate Xcode's behavior and replace non-RFC1034-compliant characters with -.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_resource_bundle-strip_structured_resources_prefixes"></a>strip_structured_resources_prefixes |  A list of prefixes to strip from the paths of structured resources. For each structured resource, if the path starts with one of these prefixes, the first matching prefix will be removed from the path when the resource is placed in the bundle root. This is useful for removing intermediate directories from the resource paths.<br><br>For example, if `structured_resources` contains `["intermediate/res/foo.png"]`, and `strip_structured_resources_prefixes` contains `["intermediate"]`, `res/foo.png` will end up inside the bundle.   | List of strings | optional |  `[]`  |
| <a id="apple_resource_bundle-structured_resources"></a>structured_resources |  Files to include in the final resource bundle. They are not processed or compiled in any way besides the processing done by the rules that actually generate them. These files are placed in the bundle root in the same structure passed to this argument, so `["res/foo.png"]` will end up in `res/foo.png` inside the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |


<a id="apple_resource_group"></a>

## apple_resource_group

<pre>
load("@rules_apple//apple:resources.bzl", "apple_resource_group")

apple_resource_group(<a href="#apple_resource_group-name">name</a>, <a href="#apple_resource_group-resources">resources</a>, <a href="#apple_resource_group-strip_structured_resources_prefixes">strip_structured_resources_prefixes</a>, <a href="#apple_resource_group-structured_resources">structured_resources</a>)
</pre>

This rule encapsulates a target which provides resources to dependents. An
`apple_resource_group`'s `resources` and `structured_resources` are put in the
top-level Apple bundle target. `apple_resource_group` targets need to be added
to library targets through the `data` attribute, or to other
`apple_resource_bundle` or `apple_resource_group` targets through the
`resources` attribute.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="apple_resource_group-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="apple_resource_group-resources"></a>resources |  Files to include in the final bundle that depends on this target. Files that are processable resources, like .xib, .storyboard, .strings, .png, and others, will be processed by the Apple bundling rules that have those files as dependencies. Other file types that are not processed will be copied verbatim. These files are placed in the root of the final bundle (e.g. Payload/foo.app/...) in most cases. However, if they appear to be localized (i.e. are contained in a directory called *.lproj), they will be placed in a directory of the same name in the app bundle.<br><br>You can also add apple_resource_bundle and apple_bundle_import targets into `resources`, and the resource bundle structures will be propagated into the final bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="apple_resource_group-strip_structured_resources_prefixes"></a>strip_structured_resources_prefixes |  A list of prefixes to strip from the paths of structured resources. For each structured resource, if the path starts with one of these prefixes, the first matching prefix will be removed from the path when the resource is placed in the bundle root. This is useful for removing intermediate directories from the resource paths.<br><br>For example, if `structured_resources` contains `["intermediate/res/foo.png"]`, and `strip_structured_resources_prefixes` contains `["intermediate"]`, `res/foo.png` will end up inside the bundle.   | List of strings | optional |  `[]`  |
| <a id="apple_resource_group-structured_resources"></a>structured_resources |  Files to include in the final application bundle. They are not processed or compiled in any way besides the processing done by the rules that actually generate them. These files are placed in the bundle root in the same structure passed to this argument, so `["res/foo.png"]` will end up in `res/foo.png` inside the bundle.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |


<a id="apple_core_ml_library"></a>

## apple_core_ml_library

<pre>
load("@rules_apple//apple:resources.bzl", "apple_core_ml_library")

apple_core_ml_library(<a href="#apple_core_ml_library-name">name</a>, <a href="#apple_core_ml_library-mlmodel">mlmodel</a>, <a href="#apple_core_ml_library-kwargs">**kwargs</a>)
</pre>

Macro to orchestrate an objc_library with generated sources for mlmodel files.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="apple_core_ml_library-name"></a>name |  <p align="center"> - </p>   |  none |
| <a id="apple_core_ml_library-mlmodel"></a>mlmodel |  <p align="center"> - </p>   |  none |
| <a id="apple_core_ml_library-kwargs"></a>kwargs |  <p align="center"> - </p>   |  none |


<a id="objc_intent_library"></a>

## objc_intent_library

<pre>
load("@rules_apple//apple:resources.bzl", "objc_intent_library")

objc_intent_library(<a href="#objc_intent_library-name">name</a>, <a href="#objc_intent_library-src">src</a>, <a href="#objc_intent_library-class_prefix">class_prefix</a>, <a href="#objc_intent_library-testonly">testonly</a>, <a href="#objc_intent_library-kwargs">**kwargs</a>)
</pre>

Macro to orchestrate an objc_library with generated sources for intentdefiniton files.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="objc_intent_library-name"></a>name |  <p align="center"> - </p>   |  none |
| <a id="objc_intent_library-src"></a>src |  <p align="center"> - </p>   |  none |
| <a id="objc_intent_library-class_prefix"></a>class_prefix |  <p align="center"> - </p>   |  `None` |
| <a id="objc_intent_library-testonly"></a>testonly |  <p align="center"> - </p>   |  `False` |
| <a id="objc_intent_library-kwargs"></a>kwargs |  <p align="center"> - </p>   |  none |


<a id="resources_common.bucketize"></a>

## resources_common.bucketize

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.bucketize(*, <a href="#resources_common.bucketize-allowed_buckets">allowed_buckets</a>, <a href="#resources_common.bucketize-owner">owner</a>, <a href="#resources_common.bucketize-parent_dir_param">parent_dir_param</a>, <a href="#resources_common.bucketize-resources">resources</a>, <a href="#resources_common.bucketize-swift_module">swift_module</a>)
</pre>

Separates the given resources into resource bucket types and returns an AppleResourceInfo.

This method wraps _bucketize_data and returns its tuple as an immutable Starlark structure to
help propagate the structure of the Apple bundle resources to the bundler.


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.bucketize-allowed_buckets"></a>allowed_buckets |  List of buckets allowed for bucketing. Files that do not fall into these buckets will instead be placed into the "unprocessed" bucket. Defaults to `None` which means all buckets are allowed.   |  `None` |
| <a id="resources_common.bucketize-owner"></a>owner |  An optional string that has a unique identifier to the target that should own the resources. If an owner should be passed, it's usually equal to `str(ctx.label)`.   |  `None` |
| <a id="resources_common.bucketize-parent_dir_param"></a>parent_dir_param |  Either a string/None or a struct used to calculate the value of parent_dir for each resource. If it is a struct, it will be considered a partial context, and will be invoked with partial.call().   |  `None` |
| <a id="resources_common.bucketize-resources"></a>resources |  List of resources to bucketize.   |  none |
| <a id="resources_common.bucketize-swift_module"></a>swift_module |  The Swift module name to associate to these resources.   |  `None` |

**RETURNS**

An AppleResourceInfo provider with resources bucketized according to type.


<a id="resources_common.bucketize_data"></a>

## resources_common.bucketize_data

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.bucketize_data(*, <a href="#resources_common.bucketize_data-allowed_buckets">allowed_buckets</a>, <a href="#resources_common.bucketize_data-owner">owner</a>, <a href="#resources_common.bucketize_data-parent_dir_param">parent_dir_param</a>, <a href="#resources_common.bucketize_data-resources">resources</a>,
                                <a href="#resources_common.bucketize_data-swift_module">swift_module</a>)
</pre>

Separates the given resources into resource bucket types.

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


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.bucketize_data-allowed_buckets"></a>allowed_buckets |  List of buckets allowed for bucketing. Files that do not fall into these buckets will instead be placed into the "unprocessed" bucket. Defaults to `None` which means all buckets are allowed.   |  `None` |
| <a id="resources_common.bucketize_data-owner"></a>owner |  An optional string that has a unique identifier to the target that should own the resources. If an owner should be passed, it's usually equal to `str(ctx.label)`.   |  `None` |
| <a id="resources_common.bucketize_data-parent_dir_param"></a>parent_dir_param |  Either a string/None or a struct used to calculate the value of parent_dir for each resource. If it is a struct, it will be considered a partial context, and will be invoked with partial.call().   |  `None` |
| <a id="resources_common.bucketize_data-resources"></a>resources |  List of resources to bucketize.   |  none |
| <a id="resources_common.bucketize_data-swift_module"></a>swift_module |  The Swift module name to associate to these resources.   |  `None` |

**RETURNS**

A tuple with a list of owners, a list of "unowned" resources, and a dictionary with
      bucketized resources organized by resource type.


<a id="resources_common.bucketize_typed"></a>

## resources_common.bucketize_typed

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.bucketize_typed(<a href="#resources_common.bucketize_typed-resources">resources</a>, <a href="#resources_common.bucketize_typed-bucket_type">bucket_type</a>, *, <a href="#resources_common.bucketize_typed-owner">owner</a>, <a href="#resources_common.bucketize_typed-parent_dir_param">parent_dir_param</a>)
</pre>

Collects and bucketizes a specific type of resource and returns an AppleResourceInfo.

Adds the given resources directly into a tuple under the field named in bucket_type. This avoids
the sorting mechanism that `bucketize` does, while grouping resources together using
parent_dir_param when available.


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.bucketize_typed-resources"></a>resources |  List of resources to place in bucket_type or Dictionary of resources keyed by target to place in bucket_type. This dictionary is supported by the `resources.collect()` API.   |  none |
| <a id="resources_common.bucketize_typed-bucket_type"></a>bucket_type |  The AppleResourceInfo field under which to collect the resources.   |  none |
| <a id="resources_common.bucketize_typed-owner"></a>owner |  An optional string that has a unique identifier to the target that should own the resources. If an owner should be passed, it's usually equal to `str(ctx.label)`.   |  `None` |
| <a id="resources_common.bucketize_typed-parent_dir_param"></a>parent_dir_param |  Either a string/None or a struct used to calculate the value of parent_dir for each resource. If it is a struct, it will be considered a partial context, and will be invoked with partial.call().   |  `None` |

**RETURNS**

An AppleResourceInfo provider with resources in the given bucket.


<a id="resources_common.bucketize_typed_data"></a>

## resources_common.bucketize_typed_data

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.bucketize_typed_data(*, <a href="#resources_common.bucketize_typed_data-bucket_type">bucket_type</a>, <a href="#resources_common.bucketize_typed_data-owner">owner</a>, <a href="#resources_common.bucketize_typed_data-parent_dir_param">parent_dir_param</a>, <a href="#resources_common.bucketize_typed_data-resources">resources</a>)
</pre>

Collects and bucketizes a specific type of resource.

Adds the given resources directly into a tuple under the field named in bucket_type. This avoids
the sorting mechanism that `bucketize` does, while grouping resources together using
parent_dir_param when available.


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.bucketize_typed_data-bucket_type"></a>bucket_type |  The AppleResourceInfo field under which to collect the resources.   |  none |
| <a id="resources_common.bucketize_typed_data-owner"></a>owner |  An optional string that has a unique identifier to the target that should own the resources. If an owner should be passed, it's usually equal to `str(ctx.label)`.   |  `None` |
| <a id="resources_common.bucketize_typed_data-parent_dir_param"></a>parent_dir_param |  Either a string/None or a struct used to calculate the value of parent_dir for each resource. If it is a struct, it will be considered a partial context, and will be invoked with partial.call().   |  `None` |
| <a id="resources_common.bucketize_typed_data-resources"></a>resources |  List of resources to place in bucket_type or Dictionary of resources keyed by target to place in bucket_type. This dictionary is supported by the `resources.collect()` API.   |  none |

**RETURNS**

A tuple with a list of owners, a list of "unowned" resources, and a dictionary with
      bucketized resources that are all placed within a single bucket defined by bucket_type.


<a id="resources_common.bundle_relative_parent_dir"></a>

## resources_common.bundle_relative_parent_dir

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.bundle_relative_parent_dir(<a href="#resources_common.bundle_relative_parent_dir-resource">resource</a>, <a href="#resources_common.bundle_relative_parent_dir-extension">extension</a>)
</pre>

Returns the bundle relative path to the resource rooted at the bundle.

Looks for the first instance of a folder with the suffix specified by `extension`, and then
returns the directory path to the file within the bundle. For example, for a resource with path
my/package/Contents.bundle/directory/foo.txt and `extension` equal to `"bundle"`, it would
return Contents.bundle/directory.


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.bundle_relative_parent_dir-resource"></a>resource |  The resource for which to calculate the bundle relative path.   |  none |
| <a id="resources_common.bundle_relative_parent_dir-extension"></a>extension |  The bundle extension to use when finding the relative path.   |  none |

**RETURNS**

The bundle relative path, rooted at the outermost bundle.


<a id="resources_common.collect"></a>

## resources_common.collect

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.collect(*, <a href="#resources_common.collect-attr">attr</a>, <a href="#resources_common.collect-res_attrs">res_attrs</a>, <a href="#resources_common.collect-split_attr_keys">split_attr_keys</a>)
</pre>

Collects all resource attributes present in the given attributes.

Iterates over the given res_attrs attributes collecting files to be processed as resources.
These are all placed into a list, and then returned.


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.collect-attr"></a>attr |  The attributes object on the current context. Can be either a `ctx.attr/ctx.rule.attr` -like struct that has targets/lists as its values, or a `ctx.split_attr`-like struct with the dictionary fan-out corresponding to split key.   |  none |
| <a id="resources_common.collect-res_attrs"></a>res_attrs |  List of attributes to iterate over collecting resources.   |  `[]` |
| <a id="resources_common.collect-split_attr_keys"></a>split_attr_keys |  If defined, a list of 1:2+ transition keys to merge values from.   |  `[]` |

**RETURNS**

A dictionary keyed by target from the rule attr with the list of all collected resources.


<a id="resources_common.deduplicate"></a>

## resources_common.deduplicate

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.deduplicate(*, <a href="#resources_common.deduplicate-resources_provider">resources_provider</a>, <a href="#resources_common.deduplicate-avoid_providers">avoid_providers</a>, <a href="#resources_common.deduplicate-field_handler">field_handler</a>, <a href="#resources_common.deduplicate-default_owner">default_owner</a>)
</pre>



**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.deduplicate-resources_provider"></a>resources_provider |  <p align="center"> - </p>   |  none |
| <a id="resources_common.deduplicate-avoid_providers"></a>avoid_providers |  <p align="center"> - </p>   |  none |
| <a id="resources_common.deduplicate-field_handler"></a>field_handler |  <p align="center"> - </p>   |  none |
| <a id="resources_common.deduplicate-default_owner"></a>default_owner |  <p align="center"> - </p>   |  `None` |


<a id="resources_common.merge_providers"></a>

## resources_common.merge_providers

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.merge_providers(*, <a href="#resources_common.merge_providers-default_owner">default_owner</a>, <a href="#resources_common.merge_providers-providers">providers</a>, <a href="#resources_common.merge_providers-validate_all_resources_owned">validate_all_resources_owned</a>)
</pre>

Merges multiple AppleResourceInfo providers into one.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.merge_providers-default_owner"></a>default_owner |  The default owner to be used for resources which have a None value in the `owners` dictionary. May be None, in which case no owner is marked.   |  `None` |
| <a id="resources_common.merge_providers-providers"></a>providers |  The list of providers to merge. This method will fail unless there is at least 1 provider in the list.   |  none |
| <a id="resources_common.merge_providers-validate_all_resources_owned"></a>validate_all_resources_owned |  Whether to validate that all resources are owned. This is useful for top-level rules to ensure that the resources in AppleResourceInfo that they are propagating are fully owned. If default_owner is set, this attribute does nothing, as by definition the resources will all have a default owner.   |  `False` |

**RETURNS**

A AppleResourceInfo provider with the results of the merge of the given providers.


<a id="resources_common.minimize"></a>

## resources_common.minimize

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.minimize(*, <a href="#resources_common.minimize-bucket">bucket</a>)
</pre>

Minimizes the given list of tuples into the smallest subset possible.

Takes the list of tuples that represent one resource bucket, and minimizes it so that 2 tuples
with resources that should be placed under the same location are merged into 1 tuple.

For tuples to be merged, they need to have the same parent_dir and swift_module.


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.minimize-bucket"></a>bucket |  List of tuples to be minimized.   |  none |

**RETURNS**

A list of minimized tuples.


<a id="resources_common.nest_in_bundle"></a>

## resources_common.nest_in_bundle

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.nest_in_bundle(*, <a href="#resources_common.nest_in_bundle-provider_to_nest">provider_to_nest</a>, <a href="#resources_common.nest_in_bundle-nesting_bundle_dir">nesting_bundle_dir</a>)
</pre>

Nests resources in a AppleResourceInfo provider under a new parent bundle directory.

This method is mostly used by rules that create resource bundles in order to nest other resource
bundle targets within themselves. For instance, apple_resource_bundle supports nesting other
bundles through the resources attribute. In these use cases, the dependency bundles are added as
nested bundles into the dependent bundle.

This method prepends the parent_dir field in the buckets with the given
nesting_bundle_dir argument.


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.nest_in_bundle-provider_to_nest"></a>provider_to_nest |  A AppleResourceInfo provider with the resources to nest.   |  none |
| <a id="resources_common.nest_in_bundle-nesting_bundle_dir"></a>nesting_bundle_dir |  The new bundle directory under which to bundle the resources.   |  none |

**RETURNS**

A new AppleResourceInfo provider with the resources nested under nesting_bundle_dir.


<a id="resources_common.populated_resource_fields"></a>

## resources_common.populated_resource_fields

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.populated_resource_fields(<a href="#resources_common.populated_resource_fields-provider">provider</a>)
</pre>

Returns a list of field names of the provider's resource buckets that are non empty.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.populated_resource_fields-provider"></a>provider |  <p align="center"> - </p>   |  none |


<a id="resources_common.process_bucketized_data"></a>

## resources_common.process_bucketized_data

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.process_bucketized_data(*, <a href="#resources_common.process_bucketized_data-actions">actions</a>, <a href="#resources_common.process_bucketized_data-apple_mac_toolchain_info">apple_mac_toolchain_info</a>, <a href="#resources_common.process_bucketized_data-bucketized_owners">bucketized_owners</a>,
                                         <a href="#resources_common.process_bucketized_data-buckets">buckets</a>, <a href="#resources_common.process_bucketized_data-bundle_id">bundle_id</a>, <a href="#resources_common.process_bucketized_data-output_discriminator">output_discriminator</a>,
                                         <a href="#resources_common.process_bucketized_data-platform_prerequisites">platform_prerequisites</a>, <a href="#resources_common.process_bucketized_data-processing_owner">processing_owner</a>, <a href="#resources_common.process_bucketized_data-product_type">product_type</a>,
                                         <a href="#resources_common.process_bucketized_data-resource_types_to_process">resource_types_to_process</a>, <a href="#resources_common.process_bucketized_data-rule_label">rule_label</a>, <a href="#resources_common.process_bucketized_data-unowned_resources">unowned_resources</a>)
</pre>

Registers actions for select resource types, given bucketized groupings of data.

This method performs the same actions as bucketize_data, and further
iterates through a subset of resource types to register actions to process
them as necessary before returning an AppleResourceInfo. This
AppleResourceInfo has an additional field, called "processed", featuring the
expected outputs for each of the actions declared in this method.


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.process_bucketized_data-actions"></a>actions |  The actions provider from `ctx.actions`.   |  none |
| <a id="resources_common.process_bucketized_data-apple_mac_toolchain_info"></a>apple_mac_toolchain_info |  `struct` of tools from the shared Apple toolchain.   |  none |
| <a id="resources_common.process_bucketized_data-bucketized_owners"></a>bucketized_owners |  A list of tuples indicating the owner of each bucketized resource.   |  `[]` |
| <a id="resources_common.process_bucketized_data-buckets"></a>buckets |  A dictionary with bucketized resources organized by resource type.   |  none |
| <a id="resources_common.process_bucketized_data-bundle_id"></a>bundle_id |  The bundle ID to configure for this target.   |  none |
| <a id="resources_common.process_bucketized_data-output_discriminator"></a>output_discriminator |  A string to differentiate between different target intermediate files or `None`.   |  `None` |
| <a id="resources_common.process_bucketized_data-platform_prerequisites"></a>platform_prerequisites |  Struct containing information on the platform being targeted.   |  none |
| <a id="resources_common.process_bucketized_data-processing_owner"></a>processing_owner |  An optional string that has a unique identifier to the target that should own the resources. If an owner should be passed, it's usually equal to `str(ctx.label)`.   |  `None` |
| <a id="resources_common.process_bucketized_data-product_type"></a>product_type |  The product type identifier used to describe the current bundle type.   |  none |
| <a id="resources_common.process_bucketized_data-resource_types_to_process"></a>resource_types_to_process |  A list of bucket types to process.   |  `["infoplists", "plists", "pngs", "strings"]` |
| <a id="resources_common.process_bucketized_data-rule_label"></a>rule_label |  The label of the target being analyzed.   |  none |
| <a id="resources_common.process_bucketized_data-unowned_resources"></a>unowned_resources |  A list of "unowned" resources.   |  `[]` |

**RETURNS**

An AppleResourceInfo provider with resources bucketized according to
  type.


<a id="resources_common.runfiles_resources_parent_dir"></a>

## resources_common.runfiles_resources_parent_dir

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.runfiles_resources_parent_dir(*, <a href="#resources_common.runfiles_resources_parent_dir-resource">resource</a>)
</pre>

Returns the parent directory of the file.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.runfiles_resources_parent_dir-resource"></a>resource |  The resource for which to calculate the package relative path.   |  none |

**RETURNS**

The package relative path to the parent directory of the resource.


<a id="resources_common.structured_resources_parent_dir"></a>

## resources_common.structured_resources_parent_dir

<pre>
load("@rules_apple//apple:resources.bzl", "resources_common")

resources_common.structured_resources_parent_dir(*, <a href="#resources_common.structured_resources_parent_dir-parent_dir">parent_dir</a>, <a href="#resources_common.structured_resources_parent_dir-resource">resource</a>, <a href="#resources_common.structured_resources_parent_dir-strip_prefixes">strip_prefixes</a>)
</pre>

Returns the package relative path for the parent directory of a resource.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="resources_common.structured_resources_parent_dir-parent_dir"></a>parent_dir |  Parent directory to prepend to the package relative path.   |  `None` |
| <a id="resources_common.structured_resources_parent_dir-resource"></a>resource |  The resource for which to calculate the package relative path.   |  none |
| <a id="resources_common.structured_resources_parent_dir-strip_prefixes"></a>strip_prefixes |  A list of prefixes to strip from the package relative path. The first prefix that matches will be used.   |  `[]` |

**RETURNS**

The package relative path to the parent directory of the resource.


<a id="swift_apple_core_ml_library"></a>

## swift_apple_core_ml_library

<pre>
load("@rules_apple//apple:resources.bzl", "swift_apple_core_ml_library")

swift_apple_core_ml_library(<a href="#swift_apple_core_ml_library-name">name</a>, <a href="#swift_apple_core_ml_library-mlmodel">mlmodel</a>, <a href="#swift_apple_core_ml_library-kwargs">**kwargs</a>)
</pre>

Macro to orchestrate a swift_library with generated sources for mlmodel files.

**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="swift_apple_core_ml_library-name"></a>name |  <p align="center"> - </p>   |  none |
| <a id="swift_apple_core_ml_library-mlmodel"></a>mlmodel |  <p align="center"> - </p>   |  none |
| <a id="swift_apple_core_ml_library-kwargs"></a>kwargs |  <p align="center"> - </p>   |  none |


<a id="swift_intent_library"></a>

## swift_intent_library

<pre>
load("@rules_apple//apple:resources.bzl", "swift_intent_library")

swift_intent_library(<a href="#swift_intent_library-name">name</a>, <a href="#swift_intent_library-src">src</a>, <a href="#swift_intent_library-class_prefix">class_prefix</a>, <a href="#swift_intent_library-class_visibility">class_visibility</a>, <a href="#swift_intent_library-swift_version">swift_version</a>, <a href="#swift_intent_library-testonly">testonly</a>, <a href="#swift_intent_library-kwargs">**kwargs</a>)
</pre>

This macro supports the integration of Intents `.intentdefinition` files into Apple rules.

It takes a single `.intentdefinition` file and creates a target that can be added as a dependency from `objc_library` or
`swift_library` targets.

It accepts the regular `swift_library` attributes too.


**PARAMETERS**


| Name  | Description | Default Value |
| :------------- | :------------- | :------------- |
| <a id="swift_intent_library-name"></a>name |  A unique name for the target.   |  none |
| <a id="swift_intent_library-src"></a>src |  Reference to the `.intentdefiniton` file to process.   |  none |
| <a id="swift_intent_library-class_prefix"></a>class_prefix |  Class prefix to use for the generated classes.   |  `None` |
| <a id="swift_intent_library-class_visibility"></a>class_visibility |  Visibility attribute for the generated classes (`public`, `private`, `project`).   |  `None` |
| <a id="swift_intent_library-swift_version"></a>swift_version |  Version of Swift to use for the generated classes.   |  `None` |
| <a id="swift_intent_library-testonly"></a>testonly |  Set to True to enforce that this library is only used from test code.   |  `False` |
| <a id="swift_intent_library-kwargs"></a>kwargs |  <p align="center"> - </p>   |  none |


