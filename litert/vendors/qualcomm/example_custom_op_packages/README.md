# Example Custom Op Packages

This directory contains example QNN custom op packages for use with the LiteRT Qualcomm compiler plugin.

## Prerequisites

- Please follow [the instructions to install proper version of Hexagon SDK and Hexagon Tools.](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/linux_setup.html#htp-and-dsp)
- This example is verified with SM8850 (Qualcomm Snapdragon 8 Elite Gen 5).
- Install hexagon-sdk-5.4.0, hexagon-sdk-6.0.0, and hexagon tool 21.0.01.1

```bash
# install hexagon sdk 6.4.0
qpm-cli --install hexagonsdk6.x --version 6.4.0.1 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.4.0
# install hexagon sdk 6.5.0
qpm-cli --install hexagonsdk6.x --version 6.5.0.1 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.5.0
# install hexagon tool 21.0
qpm-cli --extract hexagon21.0 --version 21.0.01.1 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.5.0/tools/HEXAGON_Tools/21.0.01.1
```

## Setup environment variables

`$QNN_SDK_ROOT` refers to the root of the Qualcomm AI Engine Direct SDK.

`$HEXAGON_SDK_ROOT` refers to the root of the specified version of Hexagon SDK, i.e., the directory containing `readme.txt`.

`$X86_CXX` refers to the clang++ compiler, verified with clang++17.

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

If the TFLite custom op defines custom options (i.e., `custom_initial_data` in the FlatBuffer), the XML OpDef should include a `CustomInitialData` parameter with the exact name, datatype, and rank shown below. If the TFLite custom op has no custom options, simply omit this parameter.

`CustomInitialData` is a binary blob representing `TfLiteNode::custom_initial_data`. The delegate treats this blob as opaque and passes it through as-is. It is the responsibility of the op package to properly parse this blob.

- **Mandatory**: `true` (only include this parameter if your op has custom data; otherwise, don't define it at all)
- **Datatype**: `QNN_DATATYPE_UINT_8`
- **Shape**: `[b]`, where `b` is the size in bytes of the blob

```xml
<Parameter>
    <Name>CustomInitialData</Name>
    <Description>
        <Content>A binary blob representing TfLiteNode::custom_initial_data.</Content>
    </Description>
    <Mandatory>true</Mandatory>
    <Datatype>QNN_DATATYPE_UINT_8</Datatype>
    <Shape>
        <Rank>1D</Rank>
        <Text>[b], b = size in bytes of the blob</Text>
    </Shape>
</Parameter>
```

Refer to [the example XML config](ExampleOpPackage/config/htp_example_custom_op.xml) for a complete example. Consult the [Qualcomm AI Engine Direct op package documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/op_def_schema.html) for the full schema.

### Generate skeleton code

Pass the XML to `qnn-op-package-generator` to generate the C++ skeleton:

```bash
qnn-op-package-generator --config_path path/to/config.xml
```

Detailed instructions to use `qnn-op-package-generator` can be found here: https://docs.qualcomm.com/doc/80-63442-10/topic/op_package_gen_example.html

### Implement the op

Fill in the generated C++ source files. The interface file generally does not require changes. The op source file (e.g., `src/ops/ExampleCustomOp.cpp`) contains the kernel implementation. Refer to [the example implementation](ExampleOpPackage/src/ops/ExampleCustomOp.cpp) for details.

## Build

```bash
cd ExampleOpPackage

make htp_x86      # Build for x86 offline preparation
make htp_aarch64  # Build for Android online preparation
make htp_v81      # Build for Android online execution
```