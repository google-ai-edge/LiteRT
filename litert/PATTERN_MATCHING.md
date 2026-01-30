# LiteRT Transformation Pattern Matching Guide

This guide describes the declarative C++ matching utilities in LiteRT. These
tools are designed to simplify model transformations and optimizations by
replacing verbose nested logic with clear, structured patterns.

## 1. Using the LiteRT Matcher API

The Matcher API (defined in `litert_matchers.h`) provides a set of functional
combinators (prefixed with `m_`) to describe expected subgraphs.

### Basic Op Matching
 *   `m_OpCode<code>()`: Matches an operation by its `LiteRtOpCode` template argument.
 *   `m_Op<code>(...inputs)`: Matches an operation by code and requires its
    inputs to match the provided sub-matchers (exact count).

```cpp
using namespace litert;

// Matches any Add operation.
Match(op, m_OpCode<kLiteRtOpCodeTflAdd>());

// Matches an Add operation with exactly two inputs.
Match(op, m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any()));
```

### Structural and Topology Matching
 *   `m_OpVariadic<code>(...inputs)`: Matches an op and its *prefix* inputs.
    Useful for ops with optional or many inputs like Concatenation.
 *   `m_CommutativeOp<code>(m1, m2)`: Matches a binary op where inputs can match
    `(m1, m2)` or `(m2, m1)`.
 *   `m_OutputIndex(index, m_Op)`: Matches a tensor that is the N-th output of
 an op matching `m_Op`.
 *   `m_CaptureOrSameAs(&captured, m)`: Captures value into `captured` if it is
    null, otherwise checks if the current value is the same as `captured`.

```cpp
// Matches Mul(x, x).
Tensor x(nullptr);
Match(op, m_Op<kLiteRtOpCodeTflMul>(m_CaptureOrSameAs(&x, m_Any()), m_CaptureOrSameAs(&x, m_Any())));

// Matches the 1st output of a Split op.
Match(tensor, m_OutputIndex(1, m_OpCode<kLiteRtOpCodeTflSplit>()));
```

---

## 2. Matching Properties and Types

### Tensor Properties
 *   `m_Shape(vector<int32_t>)`: Matches tensor dimensions (use `-1` for
    wildcards).
 *   `m_Rank(n)`: Matches the number of dimensions.
 *   `m_ElementType(type)`: Matches the data type (e.g.,
    `kLiteRtElementTypeFloat32`).
 *   `m_IsConstant()`: Matches if the tensor has constant weights.
 *   `m_ConstantValue<T>(val)`: Matches a constant scalar or splat with a
    specific value.

### Op Properties
 *   `m_Options<OptionsT>(predicate)`: Matches based on operation-specific
    options (e.g., Conv2D stride, Add fusion).
 *   `m_CustomOpCode("name")`: Matches a custom operation by its string
    identifier.

```cpp
// Matches Add with ReLU activation.
auto is_relu = [](const AddOptions& o) {
  return o.fused_activation_function == kActivationFunctionTypeRelu;
};
Match(op, m_AllOf(m_OpCode<kLiteRtOpCodeTflAdd>(),
                  m_Options<AddOptions>(is_relu)));
```

### Connectivity and Usage
 *   `m_HasUsers(n)`: Matches if a tensor has exactly `n` users.
 *   `m_HasOneUse()`: Convenience for `m_HasUsers(1)`. Crucial for ensuring
    safety during fusion.

---

## 3. Logical Combinators and Control

Match patterns can be combined using boolean logic:
 *   `m_AllOf(...matchers)`: Logical AND.
 *   `m_AnyOf(...matchers)`: Logical OR.
 *   `m_Not(matcher)`: Logical NOT.
 *   `m_Any()`: Wildcard for Tensors (always true).
 *   `m_AnyOp()`: Wildcard for Ops (always true).
 *   `m_Predicate<T>(lambda)`: General purpose escape hatch for arbitrary logic.

---

## 4. Summary Table of Matchers

| Matcher | Description | Target |
| :--- | :--- | :--- |
| **Op Basics** | | |
| `m_OpCode<code>(label?)` | Matches opcode of an operation. | Op / Tensor |
| `m_Op<code>(..., label?)` | Matches opcode and exact input count. | Op / Tensor |
| `m_OpVariadic<code>(..., label?)` | Matches opcode and prefix inputs. | Op |
| `m_CommutativeOp<code>(m1, m2, label?)` | Binary op with inputs in any order. | Op / Tensor |
| `m_CustomOpCode(n, label?)` | Matches custom op by its string code. | Op / Tensor |
| **Topology** | | |
| `m_OutputIndex(i, m, label?)` | Matches tensor at index `i` of op `m`. | Tensor |
| `m_CaptureOrSameAs(&ptr, m, label?)` | Captures or matches equality with `ptr`. | Op / Tensor |
| `m_HasUsers(n, label?)` | Matches if tensor has exactly `n` users. | Tensor |
| `m_HasOneUse(label?)` | Matches if tensor has exactly 1 user. | Tensor |
| **Properties** | | |
| `m_Shape(dims, label?)` | Matches dimensions (use -1 for wildcard). | Tensor |
| `m_Rank(n, label?)` | Matches tensor rank. | Tensor |
| `m_ElementType(t, label?)` | Matches tensor data type. | Tensor |
| `m_IsConstant(label?)` | Matches if the tensor is a constant. | Tensor |
| `m_ConstantValue(v, label?)` | Matches constant scalar with value `v`. | Tensor |
| `m_Options<T>(p, label?)` | Matches op options via predicate `p`. | Op / Tensor |
| **Logic** | | |
| `m_AllOf(..., label?)` | Matches if all sub-matchers match. | Any |
| `m_AnyOf(..., label?)` | Matches if any sub-matcher matches. | Any |
| `m_Not(m, label?)` | Matches if `m` does NOT match. | Any |
| `m_Any()` / `m_AnyOp()` | Always matches. | Tensor / Op |
| `m_Predicate<T>(p, label?)` | Matches if predicate `p` returns true. | T |
| **Extension** | | |
| `m_Custom(p, label?)` | Fully customizable matcher.  | Any |

---

## 5. Comprehensive Example: RMS Norm Fragment

The following code matches a pattern for `sqrt(mean(x * x))`, typical of
Layer Norm or RMS Norm implementations. It uses capturing and usage checks
to ensure the transformation is safe.

```cpp
Op mean_op(nullptr);
Tensor input_x(nullptr);

// Match: Sqrt(Mean(Mul(x, x)))
// Ensure intermediate tensors are only used once before fusing.
auto pattern = m_Op<kLiteRtOpCodeTflSqrt>(
    m_AllOf(
        m_HasOneUse(),
        m_CaptureOrSameAs(&mean_op, m_Op<kLiteRtOpCodeTflMean>(
            m_AllOf(
                m_HasOneUse(),
                m_Op<kLiteRtOpCodeTflMul>(
                    m_CaptureOrSameAs(&input_x, m_Any()),
                    m_CaptureOrSameAs(&input_x, m_Any())
                )
            ),
            m_Any() // Match axis input of Mean
        ))
    )
);

if (Match(root_op, pattern)) {
    // We captured the root of the expression and the shared input 'x'.
    // We also verified it is safe to erase intermediate results.
}
```

---

## 6. Debugging with Labels and DebugMatch

When a complex pattern fails to match, it can be difficult to identify which
specific part of the subgraph caused the failure. LiteRT provides a labeling
system and a `DebugMatch` entry point to help.

### Adding Labels
Most matchers accept an optional trailing string argument as a `label`. These
labels are used in debug logs to identify matchers. If no label is provided,
a default name (like "OpMatcher" or "ShapeMatcher") is used.

```cpp
auto pattern = m_Op<kLiteRtOpCodeTflAdd>(
    m_IsConstant("MyWeight"),
    m_Any("MyInput"),
    "MyAddOp"
);
```

### Using DebugMatch
`DebugMatch` works like `Match` but prints a hierarchical trace of the matching
process to `LITERT_LOG` (INFO level) if the match is considered "significant"
(see Filtering Noise below).

```cpp
// Simply swap Match for DebugMatch to see why a pattern isn't working.
if (!DebugMatch(root_op, pattern)) {
  // Check LITERT_LOG for a trace like:
  // [Start] MyAddOp
  //   [Start] Input[0]
  //     [Fail] MyWeight: Tensor is not constant
  // [Fail] MyAddOp: Input count mismatch
}
```

### Filtering Noise
When matching a pattern against every op in a large graph, you usually only care
about cases where the pattern *almost* matched. You can use the
`log_depth_greater_than` parameter (default is 0) to filter the output.

This parameter specifies the minimum number of log entries (starts and fails)
that must be generated for the trace to be printed.

```cpp
// Only print the match trace if it generated more than 5 log entries.
// This helps skip trivial mismatches (e.g., wrong opcode at the root).
DebugMatch(root_op, pattern, /*log_depth_greater_than=*/5);
```
