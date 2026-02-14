# Apple Rules - Resources

## Background

This document describes the philosophy around depending on resources for Apple
targets using the Bazel Apple rules.

Apple rule targets have two main mechanisms to depend on resources so that they
are present in the output bundles:

*   __Resources used by library targets__: These are resources that library
    targets require to be present at runtime. For example, a `UIViewController`
    presenting an icon inside of a button, or a data layer library that requires
    CoreData model resources. These resources should be depended directly from
    the library targets.
*   __Top level resources__: These are resources that are not referenced by the
    bundle's binary, but are instead used by the Apple platform's operating
    system. For example, an iOS application icons are used to present the
    application in user device, or the translation files used by the operating
    system to inject into an application based on the users device settings.
    These resources should be depended directly from the top level targets.

These two mechanisms have different means of depending on the resources:

*   For library targets, resources should be depended through the `data`
    attribute.
*   For top level targets, resources should be depended through rule specific
    attributes. These attributes are used to declare specific use cases for
    otherwise generic resources. For example, a `storyboard` file in
    `launch_storyboard` is processed a bit differently from a regular storyboard
    resource, or `infoplists` files are merged together to create the bundle's
    `Info.plist` file.

Some resources require to be preprocessed before being usable in an Apple
bundle. For example, `.storyboard` files need to be compiled using `ibtoold`
before being packaged. The decision on how to preprocess resources is based on
the resource file extension. For a complete list of supported resources, please
refer to the [appendix](#appendix).

Resource files can either be grouped inside resource bundles or be placed
standalone at the root of the bundle. Resource bundles are useful for libraries
that are used across multiple application targets, as the bundle namespace
prevents collision with standalone resources that the application might want to
include. There are two ways to create a resource bundle:

*   `apple_bundle_import`: Used to import an already checked in .bundle
    directory inside the workspace.
*   `apple_resource_bundle`: Used to construct a resource bundle from multiple
    files imported from the workspace

By default, the transitive closure of resources depended on by a top-level
target will be packaged at the root of the bundle (or in the `Resources`
directory for macOS bundles). In some rare cases, you may need to place the
resources not at the root of the bundle, but inside a subdirectory structure. In
order to depend on these structured resources, you'll need to use the
`apple_resources_group`'s `structured_resources` attribute, and then depend on
this target on your library target.

## Described Use Cases

### Simple Resources

The most common use case for resources is to just add them as `data` to the
library targets:

```build
ios_application(
  name = "MyApplication",
  ...
  deps = [":MyLibrary"],
)

objc_library(
  name = "MyLibrary",
  srcs = [...],
  data = [
    "MyStoryboard.storyboard",
    "MyText.txt",
    "Subdirectory/MyPlist.plist",
  ],
)
```

This will generate an application bundle with the following resource structure:

```
MyApplication.app/MyStoryboard.storyboardc/...
MyApplication.app/MyText.txt
MyApplication.app/MyPlist.plist
```

Notice that even though the plist file was referenced from a subdirectory, it is
still placed at the root of the application bundle. Also note that the
storyboard file was compiled into a `.storyboardc` directory. This is the output
of invoking `ibtoold` on the source storyboard.

### Structured Resources

If there's a requirement that some resources need to maintain a specific
structure within the application bundle, you'll need to use
`apple_resource_group` instead:

```build
ios_application(
  name = "MyApplication",
  ...
  deps = [":MyLibrary"],
)

objc_library(
  name = "MyLibrary",
  srcs = [...],
  data = [":MyResources"],
)

apple_resource_group(
  name = "MyResources",
  resources = [
    "MyStoryboard.storyboard",
    "Subdirectory/FlattenedResource.txt",
    "MyText.txt",
  ],
  structured_resources = [
    "Subdirectory/MyPlist.plist",
  ],
)
```

This generates an application bundle with the following resource structure:

```
MyApplication.app/MyStoryboard.storyboardc/...
MyApplication.app/MyText.txt
MyApplication.app/FlattenedResource.txt
MyApplication.app/Subdirectory/MyPlist.plist
```

Notice that the plist file is now placed inside the `Subdirectory` directory.
Also notice that the `FlattenedResource.txt` file, which is checked in
underneath the `Subdirectory` tree, is not placed inside `Subdirectory` in the
application bundle, since it was not referenced in `structured_resources`.

### Bundled resources

In some cases, it's recommended to collect resources inside a resource bundle.
For example, shared libraries might find it easier to manage their resources if
they are namespaced inside a bundle, so that they don't collide with resources
that clients might want to bundle in their applications. To create a resource
bundle, use the `apple_resource_bundle` rule:

```build
ios_application(
  name = "MyApplication",
  ...
  deps = [":MyLibrary"],
)

objc_library(
  name = "MyLibrary",
  srcs = [...],
  data = [":SharedLibrary"],
)

objc_library(
  name = "SharedLibrary",
  data = [":SharedResources"],
)

apple_resource_bundle(
  name = "SharedResources",
  resources = [
    "SharedStoryboard.storyboard",
    "SharedText.txt",
  ],
  structured_resources = [
    "Subdirectory/SharedPlist.plist",
  ],
)
```

This generates an application bundle with the following resource structure:

```
MyApplication.app/SharedResources.bundle/SharedStoryboard.storyboardc/...
MyApplication.app/SharedResources.bundle/SharedText.txt
MyApplication.app/SharedResources.bundle/Subdirectory/SharedPlist.plist
```

## Recommendations

There are multiple ways to reference resources for library and/or top-level
targets.

*   Add the resource files directly into the `data` attribute.
*   Wrap the resources in an `apple_resource_group` target.
*   Wrap the resources in an `apple_resource_bundle` target.

Each approach has its benefits and drawbacks and depends on the use case which
one is better. For example, as mentioned before, `apple_resource_bundle` is
useful for shared libraries so that their resources do not collide with the
application resources.

Direct `data` usage of resources is useful if the resources are used in single
libraries. If there is a collection of resources that is used by multiple
libraries, it's best to encapsulate them into an `apple_resource_group` target
so that it's easier to share.

## Resources and Frameworks

The Bazel Apple rules track which library targets reference which resources, and
makes sure to package the resources in the same bundle that contains the binary
that linked the library code. Take for example this setup:

```build
ios_application(
  name = "MyApplication",
  deps = [":MyApplicationLibrary"],
)

objc_library(
  name = "MyApplicationLibrary",
  srcs = [...],
  deps = [":MySharedLibrary"],
)

objc_library(
  name = "MySharedLibrary",
  srcs = [...],
  data = ["MySharedResource.txt"],
)
```

In this case, the application binary would statically link the MySharedLibrary
code, so the `MySharedResource.txt` file would be placed in the application
bundle:

```
MyApplication.app/MyApplication
MyApplication.app/MySharedResource.txt
```

If a Dynamic Framework is now introduced into the BUILD graph to contain
`MySharedLibrary`,

```build
ios_application(
  name = "MyApplication",
  ...
  frameworks = [":MyFramework"],
  deps = [":MyApplicationLibrary"],
)

ios_framework(
  name = "MyFramework",
  ...
  deps = [":MySharedLibrary"],
)

objc_library(
  name = "MyApplicationLibrary",
  srcs = [...],
  deps = [":MySharedLibrary"],
)

objc_library(
  name = "MySharedLibrary",
  srcs = [...],
  data = ["MySharedResource.txt"],
)
```

... the bundle structure would change so that the resource would be instead
packaged inside the framework bundle:

```
MyApplication.app/MyApplication
MyApplication.app/Frameworks/MyFramework.framework/MyFramework
MyApplication.app/Frameworks/MyFramework.framework/MySharedResource.txt
```

Because `MySharedLibrary` is now linked into the Dynamic Framework,
`MySharedResource.txt` is also packaged inside the framework bundle.

Now, consider the case where `MyApplicationLibrary` also has a dependency on
`MySharedResource.txt`:

```build

ios_application(
  name = "MyApplication",
  ...
  frameworks = [":MyFramework"],
  deps = [":MyLibrary"],
)

ios_framework(
  name = "MyFramework",
  ...
  deps = [":MySharedLibrary"],
)

objc_library(
  name = "MyApplicationLibrary",
  srcs = [...],
  deps = [":MySharedLibrary"],
  data = ["MySharedResource.txt"],
)

objc_library(
  name = "MySharedLibrary",
  srcs = [...],
  data = ["MySharedResource.txt"],
)
```

In this setup, `MySharedLibrary` would be linked into `MyFramework`, while
`MyApplicationLibrary` would be linked into `MyApplication`. Because the
`MySharedResource.txt` file is required by both libraries, it will be packaged
in both bundles:

```
MyApplication.app/MyApplication
MyApplication.app/MySharedResource.txt
MyApplication.app/Frameworks/MyFramework.framework/MyFramework
MyApplication.app/Frameworks/MyFramework.framework/MySharedResource.txt
```

The philosophy behind this feature is that library code declares ownership of
the resources that it depends on, and that it should always be able to find its
declared resources by using the `[NSBundle bundleForClass:[MyClass class]]` API.
By using this approach, clients avoid having to implement resource locator
functionalities that look for the resources across different frameworks/bundles.
This approach also makes testing with resources easier, since if the library
ends up being linked inside of a `.xctest` bundle, its resources will also be
packaged in that bundle, and thus avoid requiring the `[NSBundle mainBundle]`
API to retrieve resources. This is especially useful when using logic tests
(i.e. test that do not require a test host).

Note that with careful planning of which dependencies are linked into the
framework, you can avoid having multiple copies of the resources across the
application bundle. If you find that a resource is duplicated in an application
and framework bundle, check which of your dependencies references the resources,
and try to merge them into the framework. That way only 1 copy of the resources
will exist in the application bundle.

## Migration

Historically, the `objc_library` and `swift_library` rules have had Apple
specific resource attributes, like `resources`, `structured_resources`,
`datamodels`, `asset_catalogs`, and so on. In order to unify these library APIs
with the Bazel concept of runtime files that are added through the `data`
attribute, the resource attributes in `objc_library` and `swift_library` are
being removed.

For `swift_library` targets, these means changing from:

```build
swift_library(
  name = "MySwiftLibrary",
  resources = ["MyResourceA", "MyResourceB"],
  structured_resources = ["Subdirectory/MyStructuredResource"],
)
```

to:

```build
swift_library(
  name = "MySwiftLibrary",
  data = [":MyResourceGroup"],
)

apple_resource_group(
  name = "MyResourceGroup",
  resources = ["MyResourceA", "MyResourceB"],
  structured_resources = ["Subdirectory/MyStructuredResource"],
)
```

For `objc_library` targets, this means changing from:

```build
objc_library(
  name = "MyObjCLibrary",
  asset_catalogs = glob(["MyAssets.xcassets/**"]),
  bundles = [":MyResourceBundle"],
  datamodels = glob(["MyDatamodels.xcdatamodel/**"]),
  resources = [],
  storyboards = ["MyStoryboard.storyboard"],
)
```

to:

```build
objc_library(
  name = "MyObjCLibrary",
  data = [
    ":MyResourceBundle",
    "MyResource.txt",
    "MyStoryboard.storyboard",
  ] + glob(["MyDatamodels.xcdatamodel/**", "MyAssets.xcassets/**"]),
)
```

Keep in mind the [recommendations](#recommendations) above when migrating your
targets. You might not need to create `apple_resource_group` targets depending
on your use case for the resources.

## Appendix

### Resource Processing by type

#### Library Supported Resources

*   `.storyboard` files: These files are processed with `ibtoold`.
*   `.xcassets` files: Since these files require a specific directory structure,
    they are usually added as `glob(["MyXCAssets.xcassets/**"]`). At the top
    level, all .xcassets files are grouped and processed with `actool` to
    generate the `Assets.car` file. In some cases, the `actool` command will
    generate an `Info.plist` fragment file. This file is then also merged into
    the root `Info.plist` file.
*   `.strings` and `.plist` files. string and generic plist (i.e. non Info.plist
    files) files are processed using plutil to convert them into binary format,
    to reduce their size.
*   `.png` files: PNG files are processed using `copypng` to optimize them for
    iOS devices. This is disabled by default on macOS, but can be enabled with
    the `apple.macos_compress_png_files` feature.
*   `.xib` files: XIB files are processed using `ibtoold`.
*   `.xcdatamodel` and `.xcmappingmodel` files: These files are processed with
    `momc` and `mapc` respectively.
*   `.atlas` files: These files are processed with `TextureAtlas`.
*   `.mlpackage` and `.mlmodel` files: These files are processed with `coremlc`
    aka `coremlcompiler`) into `.mlmodelc` bundles.
*   `.metal`: These files are individually processed with `metal` into `.air`
    files then linked into a `.metallib`.
*   Any other file type: These files are not processed and are copied as is into
    the bundle.

#### Bundling Rule Supported Resources

*   `.plist` `Info.plist` files: Files added through the `infoplists` attribute
    are all merged together into a single `Info.plist` file that will be placed
    at the root of the bundle.
*   `.xib` and `.storyboard` files: XIB files added through the
    `launch_storyboard` attribute are processed using `ibtoold`.
*   `.xcasset/*.appicon` files: App Icon files added through the `app_icons`
    attribute are processed in a similar manner to the `.xcassets` file in the
    above section.

#### Structured Resources

Resource files added through `structured_resources` are not processed and will
be copied as is.
