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
 *   `m_SameAs(&captured)`: Matches if the current value is the same as a
    previously captured element.

```cpp
// Matches Mul(x, x).
Tensor x(nullptr);
Match(op, m_Op<kLiteRtOpCodeTflMul>(m_Capture(&x, m_Any()), m_SameAs(&x)));

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
| `m_OpCode<code>()` | Matches opcode of an operation. | Op / Tensor |
| `m_Op<code>(...)` | Matches opcode and exact input count. | Op / Tensor |
| `m_OpVariadic<code>(...)` | Matches opcode and prefix inputs. | Op |
| `m_CommutativeOp<code>(...)` | Binary op with inputs in any order. | Op / Tensor |
| `m_CustomOpCode(n)` | Matches custom op by its string code. | Op / Tensor |
| **Topology** | | |
| `m_OutputIndex(i, m)` | Matches tensor at index `i` of op `m`. | Tensor |
| `m_Capture(&ptr, m)` | Saves matched object into `ptr`. | Op / Tensor |
| `m_SameAs(&ptr)` | Matches if identical to handle in `ptr`. | Op / Tensor |
| `m_HasUsers(n)` | Matches if tensor has exactly `n` users. | Tensor |
| `m_HasOneUse()` | Matches if tensor has exactly 1 user. | Tensor |
| **Properties** | | |
| `m_Shape(dims)` | Matches dimensions (use -1 for wildcard). | Tensor |
| `m_Rank(n)` | Matches tensor rank. | Tensor |
| `m_ElementType(t)` | Matches tensor data type. | Tensor |
| `m_IsConstant()` | Matches if the tensor is a constant. | Tensor |
| `m_ConstantValue(v)` | Matches constant scalar with value `v`. | Tensor |
| `m_Options<T>(p)` | Matches op options via predicate `p`. | Op / Tensor |
| **Logic** | | |
| `m_AllOf(...)` | Matches if all sub-matchers match. | Any |
| `m_AnyOf(...)` | Matches if any sub-matcher matches. | Any |
| `m_Not(m)` | Matches if `m` does NOT match. | Any |
| `m_Any()` / `m_AnyOp()` | Always matches. | Tensor / Op |
| `m_Predicate<T>(p)` | Matches if predicate `p` returns true. | T |
| **Extension** | | |
| `m_Custom(p)` | Fully customizable matcher.  | Any |

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
        m_Capture(&mean_op, m_Op<kLiteRtOpCodeTflMean>(
            m_AllOf(
                m_HasOneUse(),
                m_Op<kLiteRtOpCodeTflMul>(
                    m_Capture(&input_x, m_Any()),
                    m_SameAs(&input_x)
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
