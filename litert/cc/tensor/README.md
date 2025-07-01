# LiteRT Tensor API Implementation

This directory contains a complete implementation of the proposed LiteRT Tensor API that provides NumPy-like syntax for mobile ML developers.

## 🎯 Implementation Overview

The API provides three equivalent programming styles:

1. **Operator Overloading** (C++ idiomatic): `auto c = a + b;`
2. **Fluent Style** (TensorFlow.js-like): `auto c = a.add(b);` 
3. **Functional Style** (NumPy/TensorFlow-like): `auto c = add(a, b);`

## 📁 Files

- **`litert_tensor.h`** - Core tensor class template with NumPy-like interface
- **`litert_tensor.cc`** - Complete implementation with TensorBuffer integration
- **`BUILD`** - Bazel build configuration
- **`simple_test.cc`** - Basic tensor creation test
- **`minimal_demo.cc`** - Core operations demo  
- **`safe_demo.cc`** - Full API feature demonstration
- **`litert_tensor_test.cc`** - Comprehensive unit tests

## 🚀 Quick Start

### Build the library:
```bash
bazel build //litert/cc/tensor:litert_tensor
```

### Run demos:
```bash
# Basic functionality
bazel run //litert/cc/tensor:simple_test

# Core operations  
bazel run //litert/cc/tensor:minimal_demo

# All API features
bazel run //litert/cc/tensor:safe_demo
```

### Run tests:
```bash
bazel test //litert/cc/tensor:litert_tensor_test
```

## 💡 Key Features

### ✅ Working Features
- **Tensor Creation**: `zeros()`, `ones()`, `full()` factory functions
- **Three API Styles**: All working identically for arithmetic operations
- **Element Access**: Multi-dimensional indexing with `tensor(i, j, k)`
- **Shape Operations**: `reshape()`, `size()`, `shape()`, `expand_dims()`, `squeeze()`
- **Arithmetic**: Addition, subtraction, multiplication, division (tensor-tensor and tensor-scalar)
- **Universal Functions**: `sin()`, `cos()`, `sqrt()`, `exp()`, `log()`
- **Zero-Copy Integration**: Built on LiteRT's TensorBuffer system
- **Move Semantics**: Efficient C++ memory management

### 🔧 Implementation Highlights
- Template-based design supporting different data types
- Integrates seamlessly with existing LiteRT TensorBuffer infrastructure  
- Proper error handling with LiteRT's Expected<T> pattern
- Memory-efficient operations with move semantics
- Comprehensive unit test coverage

## 📊 API Examples

### Three Equivalent Arithmetic Operations
```cpp
auto a = full<float>({2, 2}, 2.0f);
auto b = full<float>({2, 2}, 3.0f);

// All three produce identical results:
auto c1 = a + b;         // Style 1: Operator overloading
auto c2 = a.add(b);      // Style 2: Fluent style  
auto c3 = add(a, b);     // Style 3: Functional style
```

### Fluent API Chaining
```cpp
auto result = tensor
    .add(1.0f)           // Add scalar
    .mul(2.0f)           // Multiply by scalar
    .sqrt()              // Take square root
    .reshape({3, 2});    // Reshape
```

### Shape Manipulation
```cpp
auto tensor = zeros<float>({2, 3});
auto reshaped = tensor.reshape({3, 2});
auto flattened = tensor.reshape({6});
auto expanded = tensor.expand_dims(0);  // Add dimension
auto squeezed = expanded.squeeze();     // Remove size-1 dimensions
```

## 🔬 Architecture

The implementation is built on top of LiteRT's existing infrastructure:

- **TensorBuffer**: Zero-copy memory management
- **Expected<T>**: Error handling without exceptions
- **ElementType**: Type-safe element type system
- **Layout**: Dimension and stride management

This ensures seamless integration with existing LiteRT models and operations while providing the requested NumPy-like developer experience.

## 📈 Status

**✅ WORKING**: Core tensor API with all three programming styles  
**✅ TESTED**: Comprehensive unit test suite  
**✅ INTEGRATED**: Built on LiteRT TensorBuffer infrastructure  
**✅ DEMONSTRATED**: Multiple working demos showing API capabilities

The implementation successfully demonstrates the exact API proposed in the original design document.