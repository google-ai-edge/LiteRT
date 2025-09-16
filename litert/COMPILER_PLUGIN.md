# LiteRt Compiler Plugins

-----

## When Should I Create a Compiler Plugin?

A **LiteRt Compiler Plugin** is necessary when you need to **integrate a
specific hardware accelerator** with a compiler dependency into the LiteRt
framework.

You should create a compiler plugin if:

1.  You are targeting a **new hardware backend** that is not currently
    supported.
2.  You want to **offload specific model operations** to that hardware
    accelerator for performance or power efficiency.
3.  You require support for AOT compilation (on workstation) and/or on-device
    compilation.

The plugin acts as a bridge, taking portions of the machine learning model and
converting them into a format that your target hardware can execute, via a call
to the backend's compiler. LiteRt embeds the output of said plugin call within
the native model format (`.tflite`) and it becomes executable through the LiteRt
runtime.

-----

## How Do Compiler Plugins Work?

The LiteRt framework uses the compiler plugin during the model loading or
offline pre-processing phase to identify and prepare model subgraphs for
execution on the target hardware.

The process involves two main phases orchestrated by the framework using the
plugin's exported functions:

1.  **Partitioning:** The plugin inspects the entire model graph and identifies
    subsets of operations that it supports and can efficiently accelerate on the
    target hardware. These supported subgraphs are "partitioned" (marked) for
    compilation and outlined.
2.  **Compilation:** The LiteRt framework passes the partitioned subgraphs back
    to the plugin. The plugin then uses its internal logic and possibly external
    toolchains (compilers) to generate one or more **hardware-specific bytecode
    modules** implementing the partitions. This bytecode is what the target
    hardware's runtime (HAL/driver) will eventually load and execute.

The framework replaces the original subgraphs with custom operations that invoke
the hardware driver, passing along the compiled bytecode created by the plugin.

**LiteRt Dispatch** is the runtime analog for compiler plugin. They provide the
means of calling into the HAL given compiler output. See the dispatch
documentation here #TODO.

### AOT vs On-Device

LiteRt can leverage compiler plugins to support AOT compilation through our
native tooling, as well as on-device compilation. on-device compilation is more
flexible, fully internalized within the LiteRt runtime API's and only requires
the management of a single backend agnostic model. The AOT flow can unblock
compilation when it is too resource intensive to run on-device which may be
the case with many contemporary large models.

### Fallback

LiteRt is built with first-class support for heterogenous graphs. Any operation
not selected by the plugin will be left to the CPU or made available for
acceleration on another backend.

-----

## Implementing a Compiler Plugin

A LiteRt compiler plugin is implemented as a shared library that exports a
specific set of C functions defined in the LiteRt C API.

### Essential Interface Functions

The core functionality revolves around two key compilation steps:
`LiteRtCompilerPluginPartition` and `LiteRtCompilerPluginCompile`.

| Function | Purpose |
| :--- | :--- |
| **LiteRtCompilerPluginPartition** | Selects and marks all supported operations within a given model subgraph (the **Partition** step). |
| **LiteRtCompilerPluginCompile$** | Generates the hardware-specific bytecode for the pre-selected partitions (the **Compile** step). |

### C API Snippets

```c
// Name associated with the manufacturer this plugin relates to.
LITERT_CAPI_EXPORT const char* LiteRtGetCompilerPluginSocManufacturer();

// Create and initialize the plugin instance.
LITERT_CAPI_EXPORT LiteRtStatus
LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin,
                           LiteRtEnvironmentOptions env, LiteRtOptions options);

// Select desired ops for compilation.
// This is the PARTITION step.
LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginPartition(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraph subgraph, LiteRtOpList selected_ops);

// Prepare result to pass to the runtime for given model containing partitioned
// subgraphs. This is the COMPILE step.
LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result);
```

### 1\. The Partition Function

The function signature is:

```c
LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginPartition(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraph subgraph, LiteRtOpList selected_ops);
```

**What the `partition` function does:** This is the **selection** phase. The
plugin iterates over the operations in the input `LiteRtSubgraph`. For every
operation that the target hardware supports and can accelerate, the plugin
**adds that operation to the LiteRtOpList$** provided in the `selected_ops`
parameter. The LiteRt framework uses this list to define the boundaries of the
partitions that will be sent for the final compilation step.

By default, LiteRt will group all selected ops into the largest possible
sub-DAGs. For more fine grained partitioning, an index can be associated when
selecting ops which serves to further break up these subgraphs.

### 2\. The Compile Function

The function signature is:

```c
LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result);
```

**What the `compile` function does:** This is the **generation** phase. The
input `partitions` represents a model where **all** the selected subgraphs have
been isolated. The plugin processes these partitions, invoking it's specific
toolchain to generate the **bytecode** for the target hardware. It is expected
that the plugin's output provides an entry point for each subgraph passed for
compilation. In most cases this is either individual byte code modules for each
input subgraph, or a single byte code module with multiple entry points.

**Type of the data returned by `compile`:** The `LiteRtCompilerPluginCompile`
function returns its output via the out-parameter **`LiteRtCompiledResult`**.

The `LiteRtCompiledResult` is an opaque (with respect to LiteRt) handle to a
structure managed by the plugin. It represents the **output of the compilation**
and contains two main pieces of information:

1.  **Byte Code Modules:** One or more raw memory buffers containing the
    **hardware-specific executable bytecode** (i.e., compiled instructions).
2.  **Call Information:** Metadata for each partition. This provides the mapping
    from `i`th input subgraph to a result byte code module and entry point
    identifier into that module.

-----

## Example Implementation

The following snippets illustrate how a basic plugin might implement the core
functions. This example is taken from a fully functional example in
`litert/vendors/examples/`

### Plugin Identification and Setup

These functions provide the framework with basic information about the plugin
and hardware.

```
// Define the plugin's internal state structure
struct LiteRtCompilerPluginT {};

// Identify the manufacturer
const char* LiteRtGetCompilerPluginSocManufacturer() {
  return "AcmeCorp"; // Example manufacturer name
}

// Specify the supported hardware (in this example, it supports kLiteRtHwAcceleratorNpu)
LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  // ... argument checking ...
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}
```

### Partitioning Logic (`LiteRtCompilerPluginPartition`)

This example shows the plugin selecting a limited set of operations (`mul`,
`sub`, and a specific composite op) only if all inputs/outputs are 32bit floats.
Usually determining whether or not an operation should be selected will include
a call to a validation hook in backend's compiler toolchain.

```
LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                          const char* soc_model,
                                          LiteRtSubgraph subgraph,
                                          LiteRtOpList selected_ops) {

  // Iterate over ops and check criteria for selection
  // (using a C++ wrapper namespace '::litert' for convenience).
  // `subgraph` is a single subgraph from the original model, as such
  // this function will be called for each subgraph in the original model.

  ::litert::Subgraph main_subgraph(subgraph);
  for (const auto& op : main_subgraph.Ops()) {
    // 1. Check a constraint: require all tensors to be Float32
    bool only_f32 = true;
    // ... logic to check input/output types ...
    if (!only_f32) {
      continue;
    }

    // 2. Check op codes and push to selected_ops list
    if (op.Code() == kLiteRtOpCodeTflMul) {
      LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
    } else if (op.Code() == kLiteRtOpCodeTflSub) {
      LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
    } else if (op.Code() == kLiteRtOpCodeShloComposite) {
      // Example of checking composite op options
      // ... logic to check for "odml.rms_norm" name ...
      LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
    }
  }
  return kLiteRtStatusOk;
}
```

Before calling compilation, LiteRt will validate and "outline" all of the
selected ops into new subgraphs in a new intermediate model. This intermedaite
model is what is passed to compilation.

### Compilation Logic (`LiteRtCompilerPluginCompile`)

This function takes the partitioned subgraphs and generates a custom
`LiteRtCompiledResult`. This example generates a standalone bytecode module for
each partition to be compiled. In real cases, this usually involves converting
LiteRt ops to types native to the backend compiler library. The functional
example plugin's "compilation" simply creates a human readable string which
encodes the graph.

```
// Internal structure defining the compiled output
struct LiteRtCompiledResultT {
  std::vector<std::string> byte_code;   // The hardware bytecode buffers
  std::vector<std::string> per_op_data; // Per-call metadata (CallInfo)
};

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {

  // 1. Create the internal result structure
  auto model = litert::Model::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();
  auto result = std::make_unique<LiteRtCompiledResultT>();
  result->byte_code.resize(num_partitions);
  result->per_op_data.resize(num_partitions);

  // 2. Iterate and compile each partition
  for (auto i = 0; i < num_partitions; ++i) {
    // CompileSinglePartition is an internal helper that converts the subgraph
    // into the target hardware's format and stores it in result->byte_code.
    // In the case of the example this is just a stringification of the graph.

    // ... internal call to CompileSinglePartition ...
    // Example: result.byte_code[i] = generated_hw_code;
    // Example: result.per_op_data[i] = absl::StrFormat("Partition_%d", i);

    // The "per_op_data" is a unique identifier associated to the `ith` partition.
    // This is analagous to the name of a function in a library.
    // This is only meaningful when the plugin is preparing single modules with multiple entry points.
  }

  // 3. Pass ownership of the result back to the framework
  *compiled_result = result.release();

  return kLiteRtStatusOk;
}

// Functions to expose the compiled result data to the framework
LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  // ... implementation reads from compiled_result->byte_code ...
}
// ... other LiteRtGetCompiledResult* functions ...
```

-----

## Usage and Validation

LiteRt provides various toolings for applying compiler plugins to model files,
executing the result, and validating/benchmarking. Please refer to the
documentation for the LiteRt's tooling: #TODO.