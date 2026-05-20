/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_TENSOR_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_TENSOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/source_location.h"

namespace litert::tensor {

using Shape = graph::Shape;
using Quantization = graph::Quantization;
using PerChannelAffineQuantization = graph::PerChannelAffineQuantization;

struct TensorInit {
  std::string name;
  Type type = Type::kUnknown;
  Shape shape;
  std::variant<std::shared_ptr<Buffer>, std::vector<float>,
               std::vector<int32_t>, std::vector<int8_t>>
      buffer;
  std::shared_ptr<Quantization> quantization;
};

class TensorHandle {
 public:
  explicit TensorHandle(source_location loc = source_location::current());
  explicit TensorHandle(TensorInit init,
                        source_location loc = source_location::current());
  explicit TensorHandle(graph::Tensor impl);

  TensorHandle(absl::Status status,
               source_location loc = source_location::current())
      : TensorHandle(graph::ErrorTensor(std::move(status), loc)) {}

  // Creates an invalid tensor handle.
  //
  // The validity of a Tensor can be checked using the `GetStatus()` function.
  static TensorHandle Invalid() { return TensorHandle(graph::Tensor()); };

  TensorHandle& Set(TensorInit init,
                    source_location loc = source_location::current()) &;

  // Clones the underlying information into a new tensor.
  //
  // Warning: the tensor `buffer` and `quantization` are shared with the
  // original tensor.
  TensorHandle ShallowClone() const;

  // Clones the underlying information into an existing tensor.
  //
  // Warning: the tensor `buffer` and `quantization` are shared with the
  // original tensor.
  void ShallowCloneTo(TensorHandle& other) const;

  // Sets the tensor name.
  //
  // Tensors are nameless by default.
  TensorHandle& SetName(std::string name) &;
  TensorHandle&& SetName(std::string name) &&;

  // Gets the tensor name.
  //
  // A nameless tensor will return an empty string.
  absl::string_view GetName() const;

  // Sets the tensor quantization.
  TensorHandle& SetQuantization(std::shared_ptr<Quantization> quantization) &;
  TensorHandle&& SetQuantization(std::shared_ptr<Quantization> quantization) &&;

  std::shared_ptr<const Quantization> GetQuantization() const;
  std::shared_ptr<Quantization> GetQuantization();

  TensorHandle& SetBuffer(std::shared_ptr<Buffer> buffer) &;
  TensorHandle&& SetBuffer(std::shared_ptr<Buffer> buffer) &&;

  absl::StatusOr<Buffer&> GetBuffer() const;

  // Sets the tensor type.
  TensorHandle& SetType(Type t) &;
  TensorHandle&& SetType(Type t) &&;

  // Gets the tensor type.
  Type GetType() const;

  // Sets the tensor type.
  TensorHandle& SetShape(Shape shape) &;
  TensorHandle&& SetShape(Shape shape) &&;

  // Gets the tensor type.
  const Shape& GetShape() const;

  // Gets the tensor status.
  absl::Status GetStatus() const;

  // Gets the underlying graph tensor.
  //
  // Warning: This is an implementation detail. Do not call this unless you are
  // directly manipulating the graph.
  graph::Tensor& GetRaw() { return impl_; }

  // Gets the underlying graph tensor.
  //
  // Warning: This is an implementation detail. Do not call this unless you are
  // directly manipulating the graph.
  const graph::Tensor& GetRaw() const { return impl_; }

  // Checks for handle equality.
  //
  // Warning: This checks whether two Tensor handles point to the same
  // underlying data, not that the tensor properties are equal.
  friend bool operator==(const TensorHandle& a, const TensorHandle& b) {
    return a.impl_ == b.impl_;
  }

 private:
  graph::Tensor impl_;
};

template <class... Mixins>
class Tensor : public TensorHandle, public Mixins... {
 public:
  // NOLINTNEXTLINE(*-explicit-constructor)
  Tensor(const TensorHandle& t) : TensorHandle(t) {}

  Tensor(const Tensor& t) = default;

  explicit Tensor(source_location loc = source_location::current())
      : TensorHandle(loc) {};
  explicit Tensor(const TensorInit& t,
                  source_location loc = source_location::current())
      : TensorHandle(t, loc) {}
  explicit Tensor(graph::Tensor impl) : TensorHandle(impl) {};
};

template <class... Mixins>
Tensor() -> Tensor<Mixins...>;

// For placeholders with no buffer
inline TensorHandle Create(const char* name, Type type, Shape shape) {
  return TensorHandle({.name = name, .type = type, .shape = std::move(shape)});
}
// For placeholders with no buffer
inline TensorHandle Create(std::string name, Type type, Shape shape) {
  return TensorHandle(
      {.name = std::move(name), .type = type, .shape = std::move(shape)});
}
// For placeholders with no buffer
inline TensorHandle Create(absl::string_view name, Type type, Shape shape) {
  return TensorHandle(
      {.name = std::string(name), .type = type, .shape = std::move(shape)});
}
// For tensors with buffer
template <typename Buffer>
inline TensorHandle Create(const char* name, Type type, Shape shape,
                           Buffer&& buffer) {
  return TensorHandle({.name = name,
                       .type = type,
                       .shape = std::move(shape),
                       .buffer = std::forward<Buffer>(buffer)});
}
// For tensors with buffer
template <typename Buffer>
inline TensorHandle Create(std::string name, Type type, Shape shape,
                           Buffer&& buffer) {
  return TensorHandle({.name = std::move(name),
                       .type = type,
                       .shape = std::move(shape),
                       .buffer = std::forward<Buffer>(buffer)});
}
// For tensors with buffer
template <typename Buffer>
inline TensorHandle Create(absl::string_view name, Type type, Shape shape,
                           Buffer&& buffer) {
  return TensorHandle({.name = std::string(name),
                       .type = type,
                       .shape = std::move(shape),
                       .buffer = std::forward<Buffer>(buffer)});
}

template <>
struct ErrorStatusBuilder::ErrorConversion<TensorHandle> {
  static bool IsError(const TensorHandle& value) {
    return !value.GetStatus().ok();
  };
  static absl::Status AsError(const TensorHandle& value) {
    return value.GetStatus();
  }
  static TensorHandle& Forward(TensorHandle& value) { return value; }
};

template <class... Mixins>
struct ErrorStatusBuilder::ErrorConversion<Tensor<Mixins...>>
    : ErrorStatusBuilder::ErrorConversion<TensorHandle> {
  static Tensor<Mixins...>& Forward(Tensor<Mixins...>& value) { return value; }
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_TENSOR_H_
