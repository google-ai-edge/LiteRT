<!-- Generated with Stardoc: http://skydoc.bazel.build -->

Rules for creating header maps.

<a id="header_map"></a>

## header_map

<pre>
load("@rules_apple//apple:header_map.bzl", "header_map")

header_map(<a href="#header_map-name">name</a>, <a href="#header_map-deps">deps</a>, <a href="#header_map-hdrs">hdrs</a>, <a href="#header_map-module_name">module_name</a>)
</pre>

Creates a binary `.hmap` file from the given headers suitable for passing to clang.

Headermaps can be used in `-I` and `-iquote` compile flags (as well as in `includes`) to tell clang where to find headers.
This can be used to allow headers to be imported at a consistent path regardless of the package structure being used.

For example, if you have a package structure like this:

    ```
    MyLib/
        headers/
            MyLib.h
        MyLib.c
        BUILD
    ```

And you want to import `MyLib.h` from `MyLib.c` using angle bracket imports: `#import <MyLib/MyLib.h>`
You can create a header map like this:

    ```bzl
    header_map(
        name = "MyLib.hmap",
        hdrs = ["headers/MyLib.h"],
    )
    ```

This generates a binary headermap that looks like:

    ```
    MyLib.h -> headers/MyLib.h
    MyLib/MyLib.h -> headers/MyLib.h
    ```

Then update `deps`, `copts` and `includes` to use the header map:

    ```bzl
    objc_library(
        name = "MyLib",
        module_name = "MyLib",
        srcs = ["MyLib.c"],
        hdrs = ["headers/MyLib.h"],
        deps = [":MyLib.hmap"],
        copts = ["-I.]
        includes = ["MyLib.hmap"]
    )
    ```

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="header_map-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="header_map-deps"></a>deps |  Targets whose direct headers should be added to the list of hdrs and rooted at the module_name   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="header_map-hdrs"></a>hdrs |  The list of headers included in the header_map   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="header_map-module_name"></a>module_name |  The prefix to be used for header imports   | String | optional |  `""`  |


