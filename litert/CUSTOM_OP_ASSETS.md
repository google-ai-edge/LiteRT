# LiteRT Custom Op Assets Guide

This guide explains how to use the Custom Op Assets workflow in LiteRT NPU
accelerator.
This feature allows users to bundle auxiliary asset files such as custom op
implementations with their custom NPU operations at
compile time, and have them delivered to the vendor driver at runtime.

## 1. Model Expectations

The workflow expects a TFLite model containing custom operations that require
external assets. These operations are typically represented as `tfl.custom` ops
in the MLIR dialect or `Custom` ops in the TFLite FlatBuffer.

Example MLIR representation:
```mlir
%1 = "tfl.custom"(%0, %arg1) {
  custom_code = "my_custom_op"
} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
```
In this example, the custom op is identified by the `custom_code`
"my_custom_op". The asset file you provide will be associated with this
identifier.

## 2. User Journey: Compiling the Model

As a user, you need to compile your model using the `apply_plugin_main` tool
and specify which asset files belong to which custom operations.

### Using the Command Line Interface

You can use the `apply_plugin` tool with the `--npu_custom_op_info` flag. The
flag accepts a list of strings in the format `op_name,path/to/asset`.

**Example Command:**

```bash
bazel run //third_party/odml/litert/litert/tools:apply_plugin -- \
  --models="MySocModel" \
  --soc_manufacturer="MyVendor" \
  --npu_custom_op_info="my_custom_op,/path/to/my_asset.bin" \
  --npu_custom_op_info="another_op,/path/to/config.json" \
  /path/to/input.tflite \
  /path/to/output.bin
```

In this example:
*   The file `/path/to/my_asset.bin` will be embedded in the compiled model and
    associated with the custom op `my_custom_op`.
*   The file `/path/to/config.json` will be associated with `another_op`.

### How it Works

1.  The compiler reads the specified asset files.
2.  It embeds the file contents into the output model's metadata with a special
    key prefix (`litert_npu_asset_<op_name>`).
3.  The resulting model file is self-contained and holds both the graph and the
    custom assets.

## 3. Vendor Journey: Supporting Custom Assets

As a vendor, you need to handle these assets both in your compiler plugin (to
recognize and compile the custom ops) and in your Dispatch API implementation
(to receive the asset data at runtime).

### Compiler Plugin Implementation

Your compiler plugin can inspect the `LiteRtCompilerOptions` passed during
initialization to dynamically support custom ops that the user has provided
assets for.

1.  **Retrieve Custom Op Info:** In `LiteRtCreateCompilerPlugin`, use
    `LiteRtGetCompilerOptionsNumCustomOpInfo` and
    `LiteRtGetCompilerOptionsCustomOpInfo` to list the custom ops provided by
    the user. Store these names.
2.  **Partitioning:** In your `LiteRtCompilerPluginPartition` function, check
    if a `tfl.custom` op's custom code matches one of the names retrieved from
    the options. If it matches, mark it for compilation.

**Example Code Snippet:**

```cpp
// In LiteRtCreateCompilerPlugin
LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* plugin_out,
                                        LiteRtEnvironmentOptions env,
                                        LiteRtOptions options) {
  // ...

  // Read user provided custom op info
  LiteRtOpaqueOptions opaque_opts;
  if (LiteRtGetOpaqueOptions(options, &opaque_opts) == kLiteRtStatusOk) {
    LiteRtCompilerOptions compiler_opts;
    if (LiteRtFindCompilerOptions(opaque_opts, &compiler_opts) ==
        kLiteRtStatusOk) {
      LiteRtParamIndex num_custom_ops;
      LiteRtGetCompilerOptionsNumCustomOpInfo(compiler_opts, &num_custom_ops);

      for (int i = 0; i < num_custom_ops; ++i) {
        const char* name;
        const char* path;
        LiteRtGetCompilerOptionsCustomOpInfo(compiler_opts, i, &name, &path);
        plugin->supported_custom_ops.push_back(name);
      }
    }
  }
  // ...
}

// In LiteRtCompilerPluginPartition
LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin plugin, ...,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  // Iterate ops
  // ...
  if (op.Code() == kLiteRtOpCodeTflCustom) {
    const char* custom_code;
    LiteRtGetCustomCode(op.Get(), &custom_code);

    // Check if custom_code is in plugin->supported_custom_ops
    for (const auto& name : plugin->supported_custom_ops) {
      if (name == custom_code) {
        LiteRtPushOp(selected_ops, op.Get(), ...);
        break;
      }
    }
  }
  // ...
}
```

### Dispatch API Implementation (Runtime)

You must implement the `register_asset` function in your `LiteRtDispatchApi`
table. This function will be called by the LiteRT runtime during initialization
for each asset found in the model.

**Function Signature:**

```c
LiteRtStatus LiteRtDispatchRegisterAsset(
    LiteRtDispatchDeviceContext device_context,
    const char* asset_name,
    const LiteRtMemBuffer* asset_buffer);
```

**Implementation Steps:**

1.  **Store the Asset:** In your implementation, store the provided
    `asset_buffer` (which contains the pointer and size, or a file descriptor)
    in your device context. Use `asset_name` as the key.
    *   *Note:* The memory pointed to by `base_addr` is guaranteed to be valid
        for the lifetime of the `device_context` (or until the model is
        unloaded).

2.  **Access During Invocation:** When `LiteRtDispatchInvocationContextCreate`
    or `LiteRtDispatchInvoke` is called for a custom op, look up the stored
    asset using the op's name (which matches the `asset_name`).

3.  **Use the Asset:** Pass the asset data to your driver or hardware as needed.

**Example Code Snippet:**

```cpp
// In your dispatch implementation
LiteRtStatus MyRegisterAsset(LiteRtDispatchDeviceContext device_context,
                             const char* asset_name,
                             const LiteRtMemBuffer* asset_buffer) {
  auto* ctx = reinterpret_cast<MyDeviceContext*>(device_context);

  // Store the view of the asset.
  // Ideally, copy the metadata or keep a reference if the buffer is long-lived.
  ctx->assets[asset_name] = *asset_buffer;

  return kLiteRtStatusOk;
}

// In your API table definition
LiteRtDispatchApi MyApi = {
    // ... other functions ...
    .register_asset = MyRegisterAsset,
};
```

By implementing this interface, your compiler plugin (which generated the
`tfl.custom` op) and your runtime driver can coordinate to execute custom
operations with external data dependencies seamlessly.
