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
modern, type-safe C++ abstractions so graph transformations and vendor compiler
passes remain concise, declarative, and maintainable.

## Key Components

### 1. Model Abstractions (`:litert_model`, `litert_model.h`)

Lightweight, non-owning view wrappers around LiteRT compiler objects:

- `Model`: Represents a compiler model containing subgraphs.
- `Subgraph`: Provides access to subgraph inputs, outputs, and ops.
- `Op`: Represents an operation in the graph; access inputs, outputs, opcode,
  and op options.
- `Tensor`: Represents a tensor; access names, element types, ranked/unranked
  tensor types, layouts, weights, quantization info, owning subgraph and I/O
  predicates (`GetSubgraph()`, `IsConstant()`, `IsSubgraphInput()`,
  `IsSubgraphOutput()`), and use-def chains (`Uses()`, `DefiningOp()`,
  `GetDefiningOp()`).
- `Weights`: Wrapper for accessing constant tensor data and buffer payloads.

### 2. Graph Builder (`:litert_builder`, `litert_builder.h`)

API for creating and modifying graph elements:

- `Builder`: Interface to construct new tensors (`BuildTensor`), create
  operations (`BuildOp`), attach weights (`BuildWeights`), configure op options
  (`SetOpOptions`), and erase replaced ops (`EraseOp`).
- `RankedTensorSpec` & `RankedTensorSpecBuilder`: C++17-compatible fluent
  builder for specifying ranked tensor shapes, element types, weights, and
  quantization parameters.

### 3. Strongly Typed Op Options (`:litert_op_options`, `litert_op_options.h`)

Type-safe C++ representations of FlatBuffer and built-in op options:

- Structs such as `AddOptions`, `BatchMatmulOptions`, `ConcatenationOptions`,
  `DivOptions`, `MulOptions`, `ReshapeOptions`, `SoftmaxOptions`, `SubOptions`,
  `StridedSliceOptions`, etc.
- `GetOptionsAs<OptionsT>(ctx, op)`: Extracts and parses options from an op
  handle into a typed struct.
- `Builder::SetOpOptions`: Serializes and sets options onto newly created ops.

### 4. Graph Matchers (`:litert_matchers`, `litert_matchers.h`)

A header-only declarative pattern-matching DSL (with optional `MatchTracer` /
`DebugMatch` diagnostics) for graph rewrite rules:

- **Op & Use-Def Primitives**: `m_Op<Code>(...)`, `m_OpVariadic<Code>(...)`,
  `m_CommutativeOp<Code>(m1, m2)`, `m_OpCode<Code>()`,
  `m_OpCodeAnyOf<Codes...>()`, `m_Outputs(...)`, `m_SingleOutput(m)`,
  `m_Output(idx, m)`, `m_OutputIndex(idx, m)`, `m_Options<OptionsT>(pred)`,
  `m_CustomOp(...)`, `m_CompositeOp(...)`.
- **Tensor, Shape & Quantization Matchers**: `m_Shape(...)`, `m_Rank(r)`,
  `m_RankAtLeast(r)`, `m_HasStaticPositiveShape()`, `m_SameShapeAs(&other)`,
  `m_ElementType(type)`, `m_IsQuantized()`, `m_IsUnquantized()`,
  `m_UnquantizedElementType(type)`, `m_QType(qtype)`, `m_HasUsers(n)`,
  `m_HasOneUse()`, `m_HasNoUses()`, `m_IsConstant()`,
  `m_ConstantValue<T>(val)`, `m_ConstantElements<T>(pred)`,
  `m_IsSubgraphInput()`, `m_IsSubgraphOutput()`, `m_Name(name)`.
- **Combinators & Captures**: `m_AllOf(...)`, `m_AnyOf(...)`, `m_Not(m)`,
  `m_Any()`, `m_AnyOp()`, `m_Capture(&dst, [inner])` (unconditional capture safe
  inside `m_CommutativeOp` / `m_AnyOf`), `m_CaptureOrSameAs(&dst, inner)`
  (capture on first sight, equality check on subsequent occurrences).

### 5. Quantization Utilities (`:litert_quantization`, `litert_quantization.h`)

Helpers for inspecting and manipulating per-tensor, per-channel, and block-wise
quantization metadata across transformations.

### 6. Transformation Utilities (`:transformation_utils`, `transformation_utils.h`)

Common vendor-agnostic helper functions for writing graph transformations:

- **Naming & Namespaces**:
  - `GetTensorNamespace`: Extracts namespace prefixes (e.g.,
    `"model/layers.0/attn"` from `"model/layers.0/attn/logits"`).
  - `MakeNamespacedTensorName`: Combines namespace prefixes with tag names and
    optional indices.
- **Shape, Constant & Graph Inspection**:
  - `HasDims` / `HasPerm`: Checks expected tensor dimensions or `Transpose`
    constant permutation.
  - `GetStaticPositiveDims`: Returns static positive dimensions (`{}` for
    scalars) or `std::nullopt` for unranked/dynamic/zero-sized tensors.
  - `CheckedNumElements`: Overflow-safe element count for static positive dims.
  - `BroadcastShapes`: NumPy/TFLite right-aligned broadcast shape calculation.
  - `ReadIntConstValues`: Reads unquantized `Int32` or `Int64` constant weights
    as `int64_t` values.
  - `SingleUser` / `SingleOutput`: Inspects single-consumer tensors and
    single-output ops.
- **Op & Tensor Construction**:
  - `BuildTypedTensor` / `NewLike`: Builds unquantized typed tensors or clones
    prototype element type and per-tensor quantization.
  - `ConstI32`, `BuildConst<T>`, `BuildScalarConst<T>`: Builds constant tensors
    with attached weights.
  - `BuildCast` / `BuildCastInto`: Builds `Cast` ops.
  - `BuildBinaryOp` / `BuildBinaryOpInto` / `BuildAdd`: Builds elementwise
    binary ops (configuring `AddOptions`, `SubOptions`, `MulOptions`, or
    `DivOptions` when applicable).
  - `BuildSlice`, `BuildConcat` / `BuildConcatInto`, `BuildSoftmax`,
    `BuildReshape` / `BuildReshapeInto`, `BuildBatchMatmul`: High-level op
    builders.
  - `CopyOpOptions<OptionsT>`: Copies typed op options from a source `Op` onto a
    replacement `Op`.

## Example: Pattern Matching and Graph Rewriting

```cpp
#include "litert/c/litert_op_code.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/transformation_utils.h"

namespace litert::compiler {

// Match an Add op whose unique output has one consumer, then append a Reshape.
void MaybeTransform(const Builder& builder, const Op& op) {
  Tensor in1;
  Tensor in2;
  Tensor out;
  if (!Match(op, m_AllOf(m_Op<kLiteRtOpCodeTflAdd>(m_Capture(&in1),
                                                   m_Capture(&in2)),
                         m_SingleOutput(m_Capture(&out, m_HasOneUse()))))) {
    return;
  }

  auto reshaped = BuildReshape(builder, out, out, {1, 8}, "Reshaped");
  if (!reshaped.HasValue()) {
    return;
  }
}

}  // namespace litert::compiler
```

## Testing

Build and run all unit tests in this directory:

```bash
bazel test //litert/compiler/cc/...
```
