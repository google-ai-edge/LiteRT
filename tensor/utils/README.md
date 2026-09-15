# Tensor status and test utilities

This directory provides shared error-handling macros, source-location capture,
and Google Test helpers used throughout the [Tensor API](../tensor.h), its
[graph internals](../internal), backends, and runners. All library
implementations here are headers.

Start with [macros.h](macros.h) for production error handling and
[matchers.h](matchers.h) for test assertions. These utilities carry status
information across layers without requiring the callers to repeat the same
success/error checks.

## Directory structure

| File | Responsibility |
| --- | --- |
| [macros.h](macros.h) | Defines return-on-error and abort-on-error macros, `ErrorStatusBuilder`, status conversions, and value forwarding. |
| [source_location.h](source_location.h) | Provides `litert::tensor::source_location` for recording a call site's file and line. |
| [matchers.h](matchers.h) | Provides `LRT_TENSOR_ASSERT_OK_AND_ASSIGN`, `IsOk()`, and `IsOkAndHolds()` for Google Test/Google Mock. |
| [macros_test.cc](macros_test.cc) | Tests status propagation, added context, references, move-only values, structured bindings, and abort behavior. |
| [BUILD](BUILD) | Declares `:macros`, `:source_location`, `:macros_test`, and the test-only `:matchers` and `:matchers_no_g3` libraries. Default visibility is public. |

## Propagating errors

| Macro | Behavior |
| --- | --- |
| `LRT_TENSOR_RETURN_IF_ERROR(expr)` | Evaluates `expr` once and returns an error builder when it indicates failure. Execution continues on success. |
| `LRT_TENSOR_ASSIGN_OR_RETURN(decl, expr)` | Evaluates a status-or-value expression once, returns on failure, and assigns or declares the unwrapped value on success. |
| `LRT_TENSOR_ABORT_IF_ERROR(expr)` | Logs the failure and calls `std::abort()` instead of returning it. |
| `LRT_TENSOR_ASSIGN_OR_ABORT(decl, expr)` | Unwraps a successful value or logs and aborts on failure. |

`ErrorStatusBuilder` preserves an Abseil status code and appends a source
location to its message. Its `<<` operator adds context, and implicit
conversions produce `absl::Status`, `absl::StatusOr<T>`, or a compatible type
constructible from an Abseil status. Passing an error through multiple macro
call sites accumulates file/line context. Ordinary propagation adds that
context to the returned status; `Log()` explicitly writes an INFO log, while
the abort helpers log before terminating.

Error detection supports `absl::Status`, `absl::StatusOr<T>`, booleans,
pointers, and arithmetic values. False, null, and zero indicate failure and
convert to `Unknown`; true, non-null, and nonzero values indicate success.
For a new error representation, extend
`ErrorStatusBuilder::ErrorConversion` and test its failure detection and
status conversion.

The return macros accept an optional custom return expression. Within that
expression, `_` names the error builder. For example, inside a function with
an appropriate return type:

```cpp
LRT_TENSOR_RETURN_IF_ERROR(ValidateInput()) << "While validating input";
LRT_TENSOR_ASSIGN_OR_RETURN(auto value, LoadValue(),
                            _ << "While loading input");
```

Here `ValidateInput()` and `LoadValue()` stand for application functions
returning a status and a status-or-value, respectively. The custom return
expression can also be another value accepted by the enclosing function.
`LRT_TENSOR_ASSIGN_OR_ABORT` takes an optional error-builder expression for
customizing the log before aborting.

Assignment macros move values from `absl::StatusOr<T>` and preserve references
from `absl::StatusOr<T&>`. This permits move-only values and reference bindings;
do not assume the original status-or-value still contains an untouched value
after extraction. Declarations remain visible in the enclosing scope. When a
declaration contains commas, protect it with parentheses:

```cpp
LRT_TENSOR_ASSIGN_OR_RETURN((auto [first, second]), LoadPair());
```

The assignment macros expand into multiple statements and generate temporary
names using `__LINE__`. Use braces around surrounding conditional/loop bodies
and place each invocation on its own source line.

## Source locations

`source_location::current()` captures the caller's file and line when used as
a default argument. The header aliases `std::source_location` when
`__cpp_lib_source_location` is defined. Otherwise it supplies a smaller
implementation with `current()`, `file_name()`, and `line()`, using compiler
builtins when available. Its fallback values are `"unknown"` and zero; the
default-constructed fallback object also uses those values.

Graph construction forwards these locations into tensor groups, and
`ErrorStatusBuilder` records the location where an error is wrapped. Forward
an existing location when writing an API wrapper that needs to preserve its
caller's location rather than reporting the wrapper's implementation line.

## Test helpers

`LRT_TENSOR_ASSERT_OK_AND_ASSIGN(decl, expr)` evaluates a status-or-value
expression, reports a fatal Google Test assertion with its status on failure,
and unwraps the value on success. It uses the same move/reference forwarding
and parenthesized-declaration support as the production assignment helpers.
Use it in a test body or another context where a fatal assertion can return
from a `void` function.

`IsOk()` matches both `absl::Status` and `absl::StatusOr<T>`.
`IsOkAndHolds(matcher)` first checks success, then applies the supplied matcher
to the contained value. These are defined in `testing::tensor` and re-exported
as convenience functions in `litert::tensor`. For example:

```cpp
EXPECT_THAT(status, litert::tensor::IsOk());
EXPECT_THAT(result, litert::tensor::IsOkAndHolds(::testing::Eq(42)));
LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto value, MakeValue());
```

Use `:matchers` only in test targets. `:matchers_no_g3` exposes the same header
and dependencies for compatibility with internal targets built exclusively
for open source; keep its definition synchronized with `:matchers`.

## Tests and build targets

Run these commands from the repository root; `bazelisk` can be used in place
of `bazel`:

```sh
# Build the production utility headers and their dependencies.
bazelisk build //tensor/utils:macros \
  //tensor/utils:source_location

# Run propagation, forwarding, source-context, and death tests.
bazelisk test //tensor/utils:macros_test --test_output=errors
```

`macros_test.cc` covers success and error paths, custom return values, added
messages, reference and move-only extraction, structured bindings, and abort
death tests. It also checks accumulated source locations through several
return-on-error calls. There are no separate test targets for `matchers.h` or
`source_location.h` in this directory. Extend the existing tests when changing
macro expansion or error conversions, and keep Google Test dependencies in
the test-only libraries.
