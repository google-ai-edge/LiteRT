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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_H_

// The LiterT Tensor API underlying graph is represented using `TensorGroup` and
// `Operation` objects linked together. A `Tensor` is an `(index, tensor group)`
// pair.
//
// Because this API is used to build graph incrementaly, this representation
// allows us to avoid having to carry around a builder object that would hold
// all of the information.
//
// The ownership strategy is the following:
//
// - `Operation`s own their inputs.
// - `TensorGroup`s own the operation that produces them.
// - `Tensor`s own the tensor group they refer to (that ownership is shared).
//
// The links are bidirectional, with a weak pointer used from the inputs of a
// graph towards its outputs.
//
// This ensures that a long as a `Tensor` is alive, all the graph leading to
// that tensor is also kept alive. In turn that means that we can then serialize
// a model from its outputs.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/type_id.h"
#include "tensor/utils/source_location.h"

namespace litert::tensor::graph {

struct Operation;

struct Quantization {
  virtual ~Quantization() = default;
  virtual internal::TypeId GetTypeId() const = 0;
  virtual bool IsA(internal::TypeId id) const = 0;

  template <class T>
  absl::StatusOr<T&> As() {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<T*>(this);
    }
    return absl::InvalidArgumentError("Wrong quantization type.");
  }

  template <class T>
  absl::StatusOr<const T&> As() const {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<const T*>(this);
    }
    return absl::InvalidArgumentError("Wrong quantization type.");
  }
};

// Holds quantization parameters for a tensor.
struct PerChannelAffineQuantization : Quantization {
  PerChannelAffineQuantization() = default;

  PerChannelAffineQuantization(std::vector<float> scales,
                               std::vector<int64_t> zero_points,
                               int quantized_dimension = 0)
      : scales(std::move(scales)),
        zero_points(std::move(zero_points)),
        quantized_dimension(quantized_dimension) {}

  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<PerChannelAffineQuantization>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<PerChannelAffineQuantization>();
  }

  std::vector<float> scales;
  std::vector<int64_t> zero_points;
  int quantized_dimension;
};

using Shape = std::vector<int32_t>;

// Holds information about a tensor.
struct TensorInformation {
  std::string name;
  std::vector<std::weak_ptr<Operation>> consumers;
  Type type = Type::kUnknown;
  Shape shape;
  std::shared_ptr<Buffer> buffer;
  std::shared_ptr<Quantization> quantization;
  size_t GetSize() const {
    size_t size = 1;
    for (int32_t dim : shape) {
      size *= dim;
    }
    return size;
  }
};

// Holds tensors that must share their lifetime.
//
// This is used to model the fact that an operation's output tensors should all
// live as long a one of them is still in use.
struct TensorGroup {
  std::shared_ptr<Operation> producer;
  std::vector<TensorInformation> tensor_infos;

  // Use for debugging. This should hold the location of the op call that
  // creates this object.
  source_location loc;
  // Keeps track of the validity of this group. If an operation detects a
  // precondition violation, it should return an invalid group.
  //
  // An invalid group's tensors are not guaranteed to exist. The producer should
  // still be set.
  absl::Status status = absl::OkStatus();
};

// Links to a specific tensor in a tensor group.
//
// This is used to simplify single tensor manipulation.
struct Tensor {
  std::shared_ptr<TensorGroup> group;
  size_t index;

  // Checks for handle equality.
  //
  // Warning: This checks whether two Tensor objects point to the same
  // underlying data, not that the tensor properties are equal.
  bool operator==(const Tensor& b) const {
    return group == b.group && index == b.index;
  }

  template <class H>
  friend H AbslHashValue(H h, const Tensor& t) {
    return H::combine(std::move(h), t.group, t.index);
  }
};

class BackendExtension {
 public:
  virtual ~BackendExtension() = default;
  virtual internal::TypeId GetTypeId() const = 0;
};

#define LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION                        \
  internal::TypeId GetTypeId() const override {                                \
    return internal::TypeId::Get<                                              \
        std::decay_t<std::remove_pointer_t<decltype(this)>>>();                \
  }                                                                            \
  bool IsA(internal::TypeId id) const override {                               \
    return id == internal::TypeId::Get<                                        \
                     std::decay_t<std::remove_pointer_t<decltype(this)>>>() || \
           Operation::IsA(id);                                                 \
  }

// Represents an ML operation.
struct Operation {
  virtual ~Operation();

  virtual internal::TypeId GetTypeId() const {
    return internal::TypeId::Get<Operation>();
  }
  virtual bool IsA(internal::TypeId id) const {
    return id == internal::TypeId::Get<Operation>();
  }

  template <class T>
  absl::StatusOr<T&> As() {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<T*>(this);
    }
    return absl::InvalidArgumentError("Wrong operation type.");
  }

  template <class T>
  absl::StatusOr<const T&> As() const {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<const T*>(this);
    }
    return absl::InvalidArgumentError("Wrong operation type.");
  }

  template <typename T>
  const T* GetExtension() const {
    for (const auto& ext : extensions) {
      if (ext->GetTypeId() == internal::TypeId::Get<T>()) {
        return static_cast<const T*>(ext.get());
      }
    }
    return nullptr;
  }

  std::vector<Tensor> inputs;
  std::weak_ptr<TensorGroup> outputs_group;

  // For testing and debugging purposes only.
  virtual absl::string_view GetName() const = 0;

  absl::InlinedVector<std::unique_ptr<BackendExtension>, 4> extensions;
};

class OpDebugger {
 public:
  static void DebugOp(const Operation& op);
  static bool Enabled();

 private:
  static std::string GetTensorInfoDebugString(size_t index,
                                              const TensorInformation& info,
                                              const absl::Status& status);
};

// Creates a tensor holding the given status.
Tensor ErrorTensor(absl::Status status,
                   source_location op_loc = source_location::current());

// Creates a tensor group with a single tensor.
//
// `op_loc` should track the location of the function that creates this tensor.
//
// Note: `op_loc` is intentionally non defaulted to `source_location::current()`
// to force you to pass in the location of the function that creates this
// tensor.
Tensor NewTensor(source_location op_loc);

// Adds a tensor to an existing tensor group.
Tensor NewTensor(std::shared_ptr<TensorGroup>& group);

// Creates a new tensor group holding `count` tensors.
//
// `op_loc` should track the location of the function that creates this tensor.
//
// Note: `op_loc` is intentionally non defaulted to `source_location::current()`
// to force you to pass in the location of the function that creates this
// tensor.
std::shared_ptr<TensorGroup> NewTensorGroup(size_t count,
                                            source_location op_loc);

// Creates a tensor handle from an existing tensor group.
//
// Warning: The tensor may be invalid and should be checked with `GetStatus()`.
Tensor GetTensor(size_t index, std::shared_ptr<TensorGroup> group);

// Materializes tensors held by a tensor group.
absl::StatusOr<std::vector<Tensor>> GetTensors(
    const std::shared_ptr<TensorGroup>& group);

// Locks an operation's outputs and materializes tensors for them.
absl::StatusOr<std::vector<Tensor>> GetOutputs(const Operation& operation);

// Gets a tensor information.
absl::StatusOr<TensorInformation&> GetInfo(Tensor& tensor);

// Gets a tensor information.
absl::StatusOr<const TensorInformation&> GetInfo(const Tensor& tensor);

// Gets the `consumers` of a tensor.
absl::StatusOr<std::vector<std::weak_ptr<Operation>>&> GetConsumers(
    Tensor& tensor);

// Gets the `producer` of a tensor.
absl::StatusOr<std::shared_ptr<Operation>&> GetProducer(const Tensor& tensor);

// Gets the `name` of a tensor.
absl::StatusOr<const std::string&> GetName(const Tensor& tensor);

// Sets the `name` of a tensor.
absl::Status SetName(Tensor& tensor, std::string name);

// Sets the `buffer` of a tensor.
absl::Status SetBuffer(Tensor& tensor, std::shared_ptr<Buffer> buffer);

// Gets the `buffer` of a tensor.
absl::StatusOr<Buffer&> GetBuffer(Tensor& tensor);

// Gets the status of a tensor.
//
// This reflects whether the tensor can still be used to create a graph. An
// operation that detects invalid preconditions may return an invalid tensor.
absl::Status GetStatus(const Tensor& tensor);

// Gets the location where a tensor was created.
source_location GetLocation(const Tensor& tensor);

}  // namespace litert::tensor::graph

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_H_
