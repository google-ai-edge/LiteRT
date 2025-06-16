// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_RUNTIME_DISPATCH_NODE_OPERATIONS_H_
#define ODML_LITERT_LITERT_RUNTIME_DISPATCH_NODE_OPERATIONS_H_

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "tflite/c/c_api_opaque.h"

namespace litert::internal {

// Forward declarations
class DispatchDelegateKernel;

namespace node_ops {

// Direction enum for compile-time dispatch
enum class Direction { kInput, kOutput };

// Base template for node tensor operations (static polymorphism)
template <typename Derived>
class NodeOperation {
 public:
  // Process a single input tensor
  Expected<void> ProcessInput(TfLiteOpaqueContext* ctx, TfLiteOpaqueNode* node,
                              int index, TfLiteOpaqueTensor* tensor) {
    return static_cast<Derived*>(this)->ProcessInputImpl(ctx, node, index,
                                                         tensor);
  }

  // Process a single output tensor
  Expected<void> ProcessOutput(TfLiteOpaqueContext* ctx, TfLiteOpaqueNode* node,
                               int index, TfLiteOpaqueTensor* tensor) {
    return static_cast<Derived*>(this)->ProcessOutputImpl(ctx, node, index,
                                                          tensor);
  }
};

// Generic node tensor visitor with compile-time optimizations
template <typename Operation>
class NodeTensorVisitor {
 private:
  Operation op_;

  // Select appropriate TFLite API function based on direction
  template <Direction Dir>
  static constexpr auto SelectCountFn() {
    if constexpr (Dir == Direction::kInput) {
      return &TfLiteOpaqueNodeNumberOfInputs;
    } else {
      return &TfLiteOpaqueNodeNumberOfOutputs;
    }
  }

  template <Direction Dir>
  static constexpr auto SelectGetFn() {
    if constexpr (Dir == Direction::kInput) {
      return &TfLiteOpaqueNodeGetInput;
    } else {
      return &TfLiteOpaqueNodeGetOutput;
    }
  }

 public:
  explicit NodeTensorVisitor(Operation op) : op_(std::move(op)) {}

  // Visit all tensors of given type (input or output)
  template <Direction Dir>
  Expected<void> Visit(TfLiteOpaqueContext* ctx, TfLiteOpaqueNode* node) {
    constexpr auto count_fn = SelectCountFn<Dir>();
    constexpr auto get_fn = SelectGetFn<Dir>();

    const int count = count_fn(node);

    // Process each tensor
    for (int i = 0; i < count; ++i) {
      auto* tensor = const_cast<TfLiteOpaqueTensor*>(get_fn(ctx, node, i));
      if (!tensor) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Tensor not found at index " + std::to_string(i));
      }

      Expected<void> result;
      if constexpr (Dir == Direction::kInput) {
        LITERT_RETURN_IF_ERROR(op_.ProcessInput(ctx, node, i, tensor));
      } else {
        LITERT_RETURN_IF_ERROR(op_.ProcessOutput(ctx, node, i, tensor));
      }
    }

    return {};
  }

  // Visit both inputs and outputs
  Expected<void> VisitAll(TfLiteOpaqueContext* ctx, TfLiteOpaqueNode* node) {
    LITERT_RETURN_IF_ERROR(Visit<Direction::kInput>(ctx, node));
    return Visit<Direction::kOutput>(ctx, node);
  }
};

// Composite operation that chains multiple operations
template <typename... Operations>
class CompositeOperation
    : public NodeOperation<CompositeOperation<Operations...>> {
 private:
  std::tuple<Operations...> operations_;

  template <size_t... Is>
  Expected<void> ProcessInputHelper(TfLiteOpaqueContext* ctx,
                                    TfLiteOpaqueNode* node, int index,
                                    TfLiteOpaqueTensor* tensor,
                                    std::index_sequence<Is...>) {
    Expected<void> result;
    ((result = std::get<Is>(operations_).ProcessInput(ctx, node, index, tensor),
      result.HasValue()) &&
     ...);
    return result;
  }

  template <size_t... Is>
  Expected<void> ProcessOutputHelper(TfLiteOpaqueContext* ctx,
                                     TfLiteOpaqueNode* node, int index,
                                     TfLiteOpaqueTensor* tensor,
                                     std::index_sequence<Is...>) {
    Expected<void> result;
    ((result =
          std::get<Is>(operations_).ProcessOutput(ctx, node, index, tensor),
      result.HasValue()) &&
     ...);
    return result;
  }

 public:
  explicit CompositeOperation(Operations... ops)
      : operations_(std::move(ops)...) {}

  Expected<void> ProcessInputImpl(TfLiteOpaqueContext* ctx,
                                  TfLiteOpaqueNode* node, int index,
                                  TfLiteOpaqueTensor* tensor) {
    return ProcessInputHelper(ctx, node, index, tensor,
                              std::index_sequence_for<Operations...>{});
  }

  Expected<void> ProcessOutputImpl(TfLiteOpaqueContext* ctx,
                                   TfLiteOpaqueNode* node, int index,
                                   TfLiteOpaqueTensor* tensor) {
    return ProcessOutputHelper(ctx, node, index, tensor,
                               std::index_sequence_for<Operations...>{});
  }
};

// Lambda-based operation adapter
template <typename InputFn, typename OutputFn>
class LambdaOperation
    : public NodeOperation<LambdaOperation<InputFn, OutputFn>> {
 private:
  InputFn input_fn_;
  OutputFn output_fn_;

 public:
  LambdaOperation(InputFn in_fn, OutputFn out_fn)
      : input_fn_(std::move(in_fn)), output_fn_(std::move(out_fn)) {}

  Expected<void> ProcessInputImpl(TfLiteOpaqueContext* ctx,
                                  TfLiteOpaqueNode* node, int index,
                                  TfLiteOpaqueTensor* tensor) {
    return input_fn_(ctx, node, index, tensor);
  }

  Expected<void> ProcessOutputImpl(TfLiteOpaqueContext* ctx,
                                   TfLiteOpaqueNode* node, int index,
                                   TfLiteOpaqueTensor* tensor) {
    return output_fn_(ctx, node, index, tensor);
  }
};

// Factory functions
template <typename Op>
NodeTensorVisitor<std::remove_reference_t<Op>> MakeVisitor(Op&& op) {
  return NodeTensorVisitor<std::remove_reference_t<Op>>(std::forward<Op>(op));
}

template <typename InputFn, typename OutputFn>
auto MakeLambdaOperation(InputFn&& in_fn, OutputFn&& out_fn) {
  return LambdaOperation<std::decay_t<InputFn>, std::decay_t<OutputFn>>(
      std::forward<InputFn>(in_fn), std::forward<OutputFn>(out_fn));
}

template <typename... Ops>
auto MakeComposite(Ops&&... ops) {
  return CompositeOperation<std::decay_t<Ops>...>(std::forward<Ops>(ops)...);
}

// Helper to iterate over all nodes with index
template <typename Fn>
Expected<void> ForEachNodeIndexed(const std::vector<TfLiteOpaqueNode*>& nodes,
                                  TfLiteOpaqueContext* ctx, Fn&& fn) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto result = fn(static_cast<int>(i), nodes[i]);
    if (!result) return result;
  }
  return {};
}

}  // namespace node_ops
}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_DISPATCH_NODE_OPERATIONS_H_
