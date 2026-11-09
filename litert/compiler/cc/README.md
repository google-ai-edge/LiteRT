<!-- Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LiteRT Compiler C++ Library (`compiler/cc`)

`litert/compiler/cc` provides idiomatic C++ abstractions
and utilities for authoring LiteRT compiler plugins, graph transformations,
optimizations, and compiler passes.

## Overview

The LiteRT compiler plugin interface exposes a C ABI (`LiteRtCompilerContext`,
`LiteRtModel`, `LiteRtSubgraph`, `LiteRtOp`, `LiteRtTensor`) for ABI stability
across shared libraries. The `compiler/cc` library wraps these C constructs in
modern, type-safe C++ abstractions to make compiler plugin passes and graph
transformations easier to write, safer, and maintainable.

## Key Components

### 1. Model Abstractions (`:litert_model`, `litert_model.h`)

Lightweight, non-owning view wrappers around LiteRT compiler objects:

- `Model`: Represents a compiler model containing subgraphs.
- `Subgraph`: Provides access to subgraph inputs, outputs, and ops.
- `Op`: Represents an operation in the graph; access inputs, outputs, opcode,
  and op options.
- `Tensor`: Represents a tensor; access names, element types, ranked/unranked
  tensor types, layouts, weights, quantization info, and use-def chains
  (`Uses()`, `DefiningOp()`, `GetDefiningOp()`).
- `Weights`: Wrapper for accessing constant tensor data and buffer payloads.

### 2. Graph Builder (`:litert_builder`, `litert_builder.h`)

API for creating and modifying graph elements:

- `Builder`: Interface to construct new tensors (`BuildTensor`), create
  operations (`BuildOp`), attach weights (`BuildWeights`), and configure op
  options (`SetOpOptions`).
- `RankedTensorSpec` & `RankedTensorSpecBuilder`: Fluent builder for specifying
  ranked tensor properties, including shapes, element types, weights, and
  quantization configurations (per-tensor, per-channel, block-wise).

### 3. Strongly Typed Op Options (`:litert_op_options`, `litert_op_options.h`)

Type-safe C++ representations of FlatBuffer and built-in op options:

- Structs such as `AddOptions`, `BatchMatmulOptions`, `ConcatenationOptions`,
  `ReshapeOptions`, `SoftmaxOptions`, `StridedSliceOptions`, etc.
- `GetOptionsAs<OptionsT>(ctx, op)`: Extracts and parses options from an op
  handle into a typed struct.
- `Builder::SetOpOptions`: Serializes and sets options onto newly created ops.

### 4. Graph Matchers (`:litert_matchers`, `litert_matchers.h`)

A declarative pattern-matching DSL (analogous to LLVM/MLIR matchers) for graph
rewrite rules:

- Pattern primitives: `m_Op(opcode, ...)`, `m_Shape(...)`,
  `m_ElementType(...)`, `m_Constant()`, `m_HasOneUse()`.
- Combinators: `m_AllOf(...)`, `m_AnyOf(...)`, `m_Any()`.
- Capture helpers: `m_CaptureOrSameAs(...)`, `m_Tensor(...)`.

### 5. Quantization Utilities (`:litert_quantization`, `litert_quantization.h`)

Helpers for inspecting and manipulating quantization metadata:

- Supports per-tensor (`LiteRtQuantizationPerTensor`), per-channel
  (`LiteRtQuantizationPerChannel`), and block-wise
  (`LiteRtQuantizationBlockWise`) quantization.
- Utilities to check quantization validity, extract scaling factors, and copy
  or preserve quantization across transformations.

### 6. Transformation Utilities (`:transformation_utils`, `transformation_utils.h`)

Common, high-level helper functions for writing graph transformations:

- **Naming & Namespaces**:
  - `GetTensorNamespace`: Extracts namespace prefixes (e.g.
    `"model/layers.0/attn"` from `"model/layers.0/attn/logits"`).
  - `MakeNamespacedTensorName`: Combines namespace prefixes with tag names and
    optional indices.
- **Inspection**:
  - `HasDims`: Checks if a tensor matches an expected shape.
  - `HasPerm`: Validates whether a Transpose op uses a constant permutation
    matching expected axis order.
  - `SingleUser`: Finds the unique consuming op if a tensor has exactly one
    use matching opcode and argument index.
  - `SingleOutput`: Returns the unique output tensor of an op, or `std::nullopt`
    otherwise.
- **Op & Tensor Construction**:
  - `ConstI32`: Creates a constant 1D `int32` tensor from an
    `absl::Span<const int32_t>`.
  - `NewLike`: Constructs a tensor copying type and per-tensor quantization
    parameters from a prototype tensor.
  - `BuildSlice`: Builds a Slice operation and output tensor.
  - `BuildConcat` / `BuildConcatInto`: Builds Concatenation operations with
    specified axis.
  - `BuildSoftmax`: Builds a Softmax operation with specified beta.
  - `BuildAdd`: Builds an Add operation.
  - `BuildReshape` / `BuildReshapeInto`: Builds Reshape operations with target
    dimensions.
  - `BuildBatchMatmul`: Builds BatchMatmul operations with `adj_y` and
    `asymmetric_quantize_input` options.

## Example: Pattern Matching and Graph Rewriting

```cpp
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/transformation_utils.h"

using namespace litert::compiler;

// Example: Match an Add operation with one use and rewrite with a new Reshape
void MaybeTransform(Builder& builder, const Op& op) {
  Tensor in1, in2, out;
  if (!Match(op, m_Op(kLiteRtOpCodeTflAdd,
                      m_Tensor(&in1),
                      m_Tensor(&in2),
                      m_Tensor(&out).With(m_HasOneUse())))) {
    return;
  }

  // Construct replacement ops using transformation utils
  auto reshaped = BuildReshape(builder, out, out, {1, -1}, "Reshaped");
  if (!reshaped.HasValue()) {
    return;
  }
  // ...
}
```

## Testing

Build and run all unit tests in this directory:

```bash
blaze test //litert/compiler/cc/...
```
