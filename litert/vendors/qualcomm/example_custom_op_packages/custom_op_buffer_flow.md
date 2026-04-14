## How Custom Op Options Flow from TFLite Model to Hexagon DSP

### The Example

Consider a simple model with one custom op — a leaky ReLU with a configurable `alpha` parameter:

```
┌─────────────────────────────────────────────────┐
│  TFLite Model (.tflite)                         │
│                                                 │
│  Subgraph 0:                                    │
│    input_tensor [1,2,3,4] float32               │
│        │                                        │
│    ┌───▼──────────────────────────┐             │
│    │ CustomOp                     │             │
│    │   custom_code: "ExampleCustomOp"           │
│    │   custom_options: flexbuf{alpha: 0.01}     │
│    └───┬──────────────────────────┘             │
│        │                                        │
│    output_tensor [1,2,3,4] float32              │
└─────────────────────────────────────────────────┘
```

The `custom_options` field is a **flexbuffer-encoded byte blob** stored directly in the FlatBuffer model. For this example it encodes `{"alpha": 0.01}`.

### Stage 1: Model Loading (LiteRT Core)

When the `.tflite` file is loaded, the model IR stores two fields on the op:

- **`custom_code_`** = `"ExampleCustomOp"` — identifies the op type
- **`custom_options_`** = raw bytes of the flexbuffer blob

These are accessible via:

- `litert::Op::CustomCode()` → `"ExampleCustomOp"`
- `litert::Op::CustomOptions()` → `absl::Span<const uint8_t>` (the raw flexbuffer bytes)

### Stage 2: Partitioning (Compiler Plugin)

The Qualcomm compiler plugin's `Partition()` selects the custom op. `OutlinePartition()` clones it into a new subgraph. `CloneTo()` copies all fields including `custom_code_` (after our fix). `model.Yank()` extracts the subgraph into a new model for compilation.

```
Original model          After partition + yank
┌──────────┐            ┌──────────────────┐
│ Subgraph │            │ Sliced model     │
│  input   │            │  Subgraph 0:     │
│    │     │            │   input          │
│ CustomOp │  ──yank──> │     │            │
│    │     │            │  CustomOp        │
│  output  │            │   (custom_code + │
└──────────┘            │    custom_opts)  │
                        │     │            │
                        │   output         │
                        └──────────────────┘
```

### Stage 3: Compilation — Building the QNN Graph

The sliced model is passed to `MapGraph()` → `ConvertOp()` → `BuildCustomOp()` in `qnn_compose_graph.cc`.

`BuildCustomOp()` constructs a QNN op:

```cpp
// 1. Create op with package name and type
op = OpWrapper("ExampleCustomOp_0",       // name
               "ExampleOpPackage",         // package (from options)
               "ExampleCustomOp",          // type (from custom_code)
               QnnOpCode::kUnknown);

// 2. Attach input/output tensors
op.AddInputTensor(input_tensor);
op.AddOutputTensor(output_tensor);

// 3. Attach custom options as a STATIC TENSOR PARAMETER
op.AddTensorParam("CustomInitialData",
    tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_8,              // dtype: raw bytes
        {},                                // no quantization
        {options_bytes},                   // shape: [N] where N = byte count
        options_bytes,                     // size
        custom_options.data()));           // the flexbuffer blob
```

The key insight: **the flexbuffer blob becomes a 1-D `uint8` static tensor parameter** named `"CustomInitialData"`. In the QNN graph it looks like:

```
QNN Op: "ExampleCustomOp_0"
  Package:    "ExampleOpPackage"
  Type:       "ExampleCustomOp"
  Inputs:     [input_tensor (float32, [1,2,3,4])]
  Outputs:    [output_tensor (float32, [1,2,3,4])]
  Params:     [CustomInitialData (uint8, [27])]  ← the flexbuffer blob
```

### Stage 4: QNN Compilation → Hexagon Binary

QNN's `QnnGraph_finalize()` takes this graph, loads `libQnnExampleOpPackage.so`, and compiles it into a Hexagon binary (`.bin` / context blob). The static tensor parameter `CustomInitialData` is **baked into the binary** as constant data.

### Stage 5: On-Device Execution (Hexagon DSP)

When the compiled graph runs on the Hexagon DSP, QNN calls the QHPI execute function. **Parameters are delivered as extra inputs after the regular inputs:**

```
inputs[0] = input_tensor       (float32, the actual activation data)
inputs[1] = CustomInitialData  (uint8, the flexbuffer blob)
outputs[0] = output_tensor     (float32, result)
```

The Hexagon kernel in `ExampleCustomOp.cpp` then:

```cpp
// 1. Get the parameter tensor (it's input[1])
const QHPI_Tensor* custom_initial_data = inputs[1];

// 2. Read the raw bytes
const uint8_t* custom_data_ptr = qhpi_tensor_raw_data(custom_initial_data);
uint32_t custom_data_size = shape.dims[rank - 1];  // = 27 bytes

// 3. Parse the flexbuffer
auto map = flexbuffers::GetRoot(custom_data_ptr, custom_data_size).AsMap();
float alpha = map["alpha"].AsFloat();  // = 0.01

// 4. Execute: leaky ReLU
for (i = 0; i < num_elements; i++)
    output[i] = (input[i] < 0) ? input[i] * alpha : input[i];
```

### End-to-End Summary

```
TFLite Model (.tflite)
  │  custom_code:    "ExampleCustomOp"
  │  custom_options: flexbuf bytes {alpha: 0.01}
  │
  │  load (LiteRtCreateModelFromFile)
  ▼
LiteRT Model IR (LiteRtOpT)
  │  custom_code_:    "ExampleCustomOp"
  │  custom_options_: Span<uint8_t> (27 bytes)
  │
  │  partition (CloneTo + Yank)
  ▼
Sliced Model (new subgraph with cloned op)
  │  custom_code_:    "ExampleCustomOp"    ← preserved by CloneTo fix
  │  custom_options_: Span<uint8_t> (27 bytes)
  │
  │  build QNN graph (BuildCustomOp)
  ▼
QNN Graph
  │  Op type:    "ExampleCustomOp"         ← from custom_code
  │  Package:    "ExampleOpPackage"         ← from Qualcomm options
  │  Param:      "CustomInitialData" uint8[27]  ← from custom_options
  │
  │  compile (QnnGraph_finalize → libQnnHtp.so)
  ▼
Hexagon Binary (.bin / context blob)
  │  CustomInitialData baked in as constant data
  │
  │  execute on DSP (QHPI kernel)
  ▼
ExampleCustomOp.cpp
    inputs[0] = activation tensor (float32)
    inputs[1] = CustomInitialData (uint8[27])  ← the flexbuf blob
                    │
                    ▼
              flexbuffers::GetRoot() → map["alpha"].AsFloat() → 0.01
                    │
                    ▼
              output[i] = (input[i] < 0) ? input[i] * 0.01 : input[i]
```

The flexbuffer bytes are **never decoded by LiteRT or the compiler plugin**. They pass through opaquely as a raw byte tensor. Only the Hexagon kernel knows how to interpret them.
