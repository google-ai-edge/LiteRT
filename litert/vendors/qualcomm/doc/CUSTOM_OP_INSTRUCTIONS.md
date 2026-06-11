# Custom Op Package Instructions

This document explains how to build a QNN custom op package and use it with the
LiteRT Qualcomm compiler and dispatch plugins.

An example XML OpDef config is provided under
[`examples/custom_op_package/`](../examples/custom_op_package).

## Prerequisites

- Please follow [the instructions to install proper version of Hexagon SDK and Hexagon Tools.](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/linux_setup.html#htp-and-dsp)
- This example is verified with SM8850 (Qualcomm Snapdragon 8 Elite Gen 5).
- Install hexagon-sdk-6.4.0, hexagon-sdk-6.5.0, and hexagon tool 21.0.01.1

```bash
# install hexagon sdk 6.4.0
qpm-cli --install hexagonsdk6.x --version 6.4.0.1 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.4.0
# install hexagon sdk 6.5.0
qpm-cli --install hexagonsdk6.x --version 6.5.0.1 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.5.0
# install hexagon tool 21.0
qpm-cli --extract hexagon21.0 --version 21.0.01.1 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.5.0/tools/HEXAGON_Tools/21.0.01.1
```

## Setup environment variables

| Variable | Description |
| --- | --- |
| `$QNN_SDK_ROOT` | Root of the Qualcomm AI Engine Direct SDK. |
| `$HEXAGON_SDK_ROOT` | Root of the specified version of Hexagon SDK, i.e., the directory containing `readme.txt`. |
| `$X86_CXX` | The clang++ compiler, verified with clang++17. |

```bash
export HEXAGON_SDK_ROOT=/path/to/Qualcomm/Hexagon_SDK
export X86_CXX=/path/to/clang-17.0.x/bin/clang++

# Source the QNN environment setup script to make op package tools available
source $QNN_SDK_ROOT/bin/envsetup.sh
```

## Implement a QNN op package

An op package consists of an XML config file and C++ implementation files.

### Define the XML OpDef config

Create an XML file describing the package name, domain, version, and the operations it contains. The `PackageName` in the XML determines the library name (`libQnn<PackageName>.so`).

```xml
<OpDefCollection
    PackageName="ExampleOpPackage"
    Domain="aisw"
    Version="1.0.0">
  <OpDefList>
    <OpDef>
      <Name>ExampleCustomOp</Name>
      ...
    </OpDef>
  </OpDefList>
</OpDefCollection>
```

#### Mapping TFLite custom options to QNN parameters

If the TFLite custom op carries custom options (i.e., `custom_initial_data` in the FlatBuffer, stored as a flexbuffer map), the LiteRT Qualcomm compiler plugin **decodes the flexbuffer in `BuildCustomOp` and turns each map entry into its own named QNN op parameter**.

For each key/value pair in the flexbuffer map, the plugin emits a QNN parameter whose name is exactly the flexbuffer key:

| Flexbuffer value | QNN parameter | Datatype |
| --- | --- | --- |
| `bool` scalar | scalar param | `QNN_DATATYPE_BOOL_8` |
| `int` scalar | scalar param | `QNN_DATATYPE_INT_32` |
| `uint` scalar | scalar param | `QNN_DATATYPE_UINT_32` |
| `float` scalar | scalar param | `QNN_DATATYPE_FLOAT_32` |
| vector (uniform scalar type) | static tensor param, shape inferred from the vector | matching `BOOL_8` / `INT_32` / `UINT_32` / `FLOAT_32` |

Ragged vectors and non-uniform vector element types are not supported. The flexbuffer keys and the value types must match the `<Parameter>` entries you declare in the XML OpDef.

For example, a leaky ReLU whose custom options encode `{"alpha": 0.01}` is exposed to the op package as a scalar `float` parameter named `alpha`:

```xml
<Parameter>
    <Name>alpha</Name>
    <Description>
        <Content>Negative slope for the leaky ReLU.</Content>
    </Description>
    <Mandatory>true</Mandatory>
    <Datatype>QNN_DATATYPE_FLOAT_32</Datatype>
    <Shape>
        <Rank>SCALAR</Rank>
        <Text>scalar</Text>
    </Shape>
</Parameter>
```

If the TFLite custom op has no custom options, simply omit all parameter entries.

Refer to [the example XML config](../examples/custom_op_package/htp_example_custom_op.xml) for a complete example. Consult the [Qualcomm AI Engine Direct op package documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/op_def_schema.html) for the full schema.

### Generate skeleton code

Pass the XML to `qnn-op-package-generator` to generate the C++ skeleton:

```bash
qnn-op-package-generator --config_path path/to/config.xml
```

Detailed instructions to use `qnn-op-package-generator` can be found here: https://docs.qualcomm.com/doc/80-63442-10/topic/op_package_gen_example.html

### Implement the op

Fill in the generated C++ source files. The interface file generally does not require changes. The op source file (e.g., `src/ops/ExampleCustomOp.cpp`) contains the kernel implementation, which reads the QNN parameters declared above (e.g., the `alpha` scalar) directly — no flexbuffer parsing is required inside the kernel.

## Build

```bash
cd <op_package_dir>

make htp_x86      # Build for x86 offline preparation
make htp_aarch64  # Build for Android online preparation
make htp_v81      # Build for Android online execution
```

## Use the op package with the Qualcomm plugin

The op package name and library paths are supplied through the Qualcomm options
rather than being embedded in the model. The compiler plugin uses
`name` / `interface_provider` / `compile_package_path`; the dispatch plugin uses
`name` / `interface_provider` / `dispatch_package_path` / `target`. `target` must
match the chosen QNN backend (e.g., `HTP`).

Via the `apply_plugin_main` / `run_model` CLI flag:

```bash
--qualcomm_custom_op_package="name:ExampleOpPackage;\
interface_provider:ExampleOpPackageInterfaceProvider;\
compile_package_path:libQnnExampleOpPackage.so;\
dispatch_package_path:libQnnExampleOpPackage.so;\
target:HTP;"
```

Or programmatically through the C++ options API:

```cpp
litert::qualcomm::QualcommOptions::CustomOpPackage pkg;
pkg.name = "ExampleOpPackage";
pkg.interface_provider = "ExampleOpPackageInterfaceProvider";
pkg.compile_package_path = "libQnnExampleOpPackage.so";
pkg.dispatch_package_path = "libQnnExampleOpPackage.so";
pkg.target = "HTP";
options.SetCustomOpPackage(pkg);
```

- When the package `name` is left empty, the plugin falls back to the built-in
`qti.aisw` package. 
- The runtime registers the package by calling
`backendRegisterOpPackage` with the `dispatch_package_path`, `interface_provider`,
and `target`. 
- On the IR/LPAI backend, custom op packages are unsupported and will be ignored.
