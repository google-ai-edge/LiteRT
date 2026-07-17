# LiteRT Swift API

The LiteRT Swift API provides a clean, type-safe Swifty wrapper layer over the
core LiteRT C API (`litert/c`).

It allows iOS, macOS, and cross-platform Swift applications to load, compile,
and execute TensorFlow Lite models with hardware acceleration (CPU, GPU, NPU)
utilizing the highly optimized LiteRT runtime.

## Package Structure

The package is organized into `Sources/` and `Tests/` directories to match
standard Swift project conventions:

- `BUILD`: Target compilation and testing rules.
- **`Sources/`**: Production API source wrappers and headers.
-   `LiteRtC.h`: Bridging C-API umbrella header.
-   `Environment.swift`: LiteRT Environment configuration.
-   `CompiledModel.swift`: Model loader and graph compilation/executor.
-   `Options.swift`: Accelerator selection and opaque option settings.
-   `OpaqueOptions.swift`: Type-erased options for accelerator-specific settings.
-   `TensorBuffer.swift`: Arithmetic heap read/write memory (supporting Host and Metal backing).
-   `TensorType.swift`: Elements and dimensions structures.
-   `LiteRtError.swift`: Swifty runtime status boundaries.
- **`Tests/`**: Component-focused unit tests.
-   `EnvironmentTests.swift`: Environment configuration tests.
-   `TensorBufferTests.swift`: Host/Metal buffer and locking tests.
-   `OptionsTests.swift`: Accelerator and kernel configuration tests.
-   `OpaqueOptionsTests.swift`: Opaque options tests.
-   `CompiledModelTests.swift`: Model loading, graph execution and cancellation tests.
-   `TensorTypeTests.swift`: Layout tests.
-   `IntegrationTests.swift`: End-to-end integration test.

## Key Classes

- **`Environment`**: Holds and configures runtime environment options (such as
  custom compiler or dispatch plugin library paths) and registered hardware
  accelerators.
- **`CompiledModel`**: Loads, compiles, and represents the executable LiteRT
  model.
  Supports loading from a local file path, raw binary `Data` buffers, or a file
  descriptor. Exposes inference execution (`run`, `dispatch`) and buffer
  requirements query APIs.
- **`Options`**: Configures hardware backends and compilation settings (such as
  selecting executing hardware accelerators like CPU, GPU, NPU). Supports adding
  opaque configurations.
- **`OpaqueOptions`**: Manages type-erased configuration payloads for advanced,
  accelerator-specific settings.
- **`TensorBuffer`**: Manages raw memory allocations where tensor data resides.
  Supports wrapping host memory and Metal buffers (MTLBuffer).
  Provides generic read and write methods supporting Float, Int, Int8, Bool, and
  other numeric arrays.
- **`TensorType` & `Layout`**: Exposes tensor metadata such as element data
  types and dimension/shape layout.

---

## Usage Example

Below is a complete end-to-end example loading a simple add model, compiling it
for CPU execution via `XNNPACK`, doing synchronous inference, and verifying the
results.

```swift
import Foundation
import LiteRT

func runInference() throws {
  // 1. Initialize the LiteRT Environment
  let environment = try Environment()

  // 2. Configure Compilation Options (CPU acceleration)
  let options = try Options()
  try options.setHardwareAccelerators([.cpu])

  // 3. Load and compiles the Model into CompiledModel
  let compiledModel = try CompiledModel(
    filePath: "path/to/simple_add_model.tflite",
    environment: environment,
    options: options
  )

  // 4. Allocate Input and Output Buffers from Model Requirements
  let inputBuffers = try compiledModel.createInputBuffers()
  let outputBuffers = try compiledModel.createOutputBuffers()

  // 5. Populate Input Buffers with Data
  let input0Data: [Float] = [1.0, 2.0]
  let input1Data: [Float] = [10.0, 20.0]
  try inputBuffers[0].write(input0Data)
  try inputBuffers[1].write(input1Data)

  // 6. Execute Inference
  try compiledModel.run(inputs: inputBuffers, outputs: outputBuffers)

  // 7. Retrieve and Read the Output Data
  let outputData: [Float] = try outputBuffers[0].read()
  print("Output elements: \(outputData)") // Expected: [11.0, 22.0]
}
```

---

## BUILD Integration

To depend on the LiteRT Swift library in google3, add it to your build target's
`deps` list:

```starlark
swift_library(
    name = "my_target",
    srcs = ["MyFile.swift"],
    deps = [
        "//litert/swift:litert_swift",
    ],
)
```
