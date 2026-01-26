# LiteRT Transformation Pattern Matching Guide

This guide introduces the new C++ matching utilities and IR extensions designed
to simplify the development of model transformations and optimizations in
LiteRT.

##  1. Using the LiteRT Matcher API

The Matcher API (found in `litert_matchers.h`) allows you to replace verbose
nested `if` statements with concise, declarative patterns.

### Basic Matching: `m_OpCode` and `m_Op`
Use `m_OpCode` for simple opcode checks and `m_Op` when you want to match an
operation and its inputs.

```cpp
using namespace litert;

// Matches any Add operation
Match(op, m_OpCode(kLiteRtOpCodeTflAdd));

// Matches an Add operation with exactly 0 inputs
Match(op, m_Op(kLiteRtOpCodeTflAdd));
```

### Wildcards: `m_Any` and `m_AnyOp`
Wildcards match any element, allowing you to ignore specific branches of a
pattern.
 - `m_Any()`: Matches any `Tensor`.
 - `m_AnyOp()`: Matches any `Op` (or a `Tensor` produced by any `Op`).

```cpp
// Matches an Abs operation with any input tensor
Match(op, m_Op(kLiteRtOpCodeTflAbs, m_Any()));
```

---

## 2. Advanced Matching Patterns

### Recursive Matching
`m_Op` can be nested to match entire chains or trees of operations.

```cpp
// Matches Add(Mul(Any, Any), Sub(Any, Any))
auto pattern = m_Op(kLiteRtOpCodeTflAdd,
                    m_Op(kLiteRtOpCodeTflMul, m_Any(), m_Any()),
                    m_Op(kLiteRtOpCodeTflSub, m_Any(), m_Any()));

if (Match(root_op, pattern)) {
  // Pattern matched!
}
```

### Capturing Elements with `m_Capture`
Matching is often the first step before replacing an op. `m_Capture` allows you
to "save" parts of the matched pattern into pointers for later use.

```cpp
Op mul_op(nullptr);
Tensor input_tensor(nullptr);

auto pattern = m_Op(kLiteRtOpCodeTflAdd,
                    m_Capture(&mul_op, m_OpCode(kLiteRtOpCodeTflMul)),
                    m_Capture(&input_tensor, m_Any()));

if (Match(root_op, pattern)) {
  // You can now use mul_op and input_tensor directly
}
```

---

## 3. Matching Specific Properties

### Constants and Subgraph Inputs
You can specifically target constant tensors (e.g., weights) or inputs coming
from outside the subgraph.

```cpp
// Matches Mul(Tensor, Constant)
Match(op, m_Op(kLiteRtOpCodeTflMul, m_Any(), m_IsConstant()));

// Matches Mul(Tensor, SubgraphInput)
Match(op, m_Op(kLiteRtOpCodeTflMul, m_Any(), m_IsSubgraphInput()));
```

### Custom Predicates
For logic that can't be expressed with standard matchers, use `m_Predicate`.

```cpp
// Matches a tensor only if its name is "target_tensor"
auto name_matcher = m_Predicate<Tensor>([](const Tensor& t) {
  return t.Name() == "target_tensor";
});

Match(tensor, name_matcher);
```

---

## 4. Full Example: `sqrt(mean(square(x)))`

Here is how a complex transformation pattern can be simplified using the new
APIs.

**Old way (Nested `if` checks):**
```cpp
if (!root_op.Is(kLiteRtOpCodeTflSqrt)) return kLiteRtStatusPatternNoMatch;
auto mean_op = root_op.Input(0)->GetDefiningOp();
if (!mean_op || !mean_op->Is(kLiteRtOpCodeTflMean)) return kLiteRtStatusPatternNoMatch;
// ... and so on
```

**New way (Declarative Matching):**
```cpp
Op mean_op(nullptr);
Op square_op(nullptr);
Tensor input_x(nullptr);

auto pattern = m_Op(kLiteRtOpCodeTflSqrt,
                 m_Capture(&mean_op, m_Op(kLiteRtOpCodeTflMean,
                   m_Capture(&square_op, m_Op(kLiteRtOpCodeTflMul,
                     m_Capture(&input_x, m_Any()),
                     m_Any())))));

if (Match(root_op, pattern)) {
    // Verify square_op is actually x * x (same input)
    if (square_op.Input(0) == square_op.Input(1)) {
        // Proceed with transformation using mean_op, square_op, and input_x
    }
}
```
