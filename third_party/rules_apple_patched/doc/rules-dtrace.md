<!-- Generated with Stardoc: http://skydoc.bazel.build -->

# Bazel rules for working with dtrace.

<a id="dtrace_compile"></a>

## dtrace_compile

<pre>
load("@rules_apple//apple:dtrace.bzl", "dtrace_compile")

dtrace_compile(<a href="#dtrace_compile-name">name</a>, <a href="#dtrace_compile-srcs">srcs</a>, <a href="#dtrace_compile-dtrace">dtrace</a>)
</pre>

Compiles
[dtrace files with probes](https://www.ibm.com/developerworks/aix/library/au-dtraceprobes.html)
to generate header files to use those probes in C languages. The header files
generated will have the same name as the source files but with a `.h`
extension. Headers will be generated in a label scoped workspace relative file
structure. For example with a directory structure of

```
  Workspace
  foo/
    bar.d
```
and a target named `dtrace_gen` the header path would be
`<GENFILES>/dtrace_gen/foo/bar.h`.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| <a id="dtrace_compile-name"></a>name |  A unique name for this target.   | <a href="https://bazel.build/concepts/labels#target-names">Name</a> | required |  |
| <a id="dtrace_compile-srcs"></a>srcs |  dtrace(.d) source files to be compiled.   | <a href="https://bazel.build/concepts/labels">List of labels</a> | optional |  `[]`  |
| <a id="dtrace_compile-dtrace"></a>dtrace |  dtrace binary to use.   | <a href="https://bazel.build/concepts/labels">Label</a> | optional |  `None`  |


