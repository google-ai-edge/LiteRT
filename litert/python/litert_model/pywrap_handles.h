// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_PYTHON_LITERT_MODEL_PYWRAP_HANDLES_H_
#define ODML_LITERT_LITERT_PYTHON_LITERT_MODEL_PYWRAP_HANDLES_H_

// Ownership model of the `litert_model` Python bindings.
//
// One `ModelState` owns one `LiteRtModelT`. Python objects for subgraphs, ops,
// tensors and signatures are small value handles holding a shared pointer to
// the state (so the model outlives every handle) plus the raw IR pointer and a
// registration id. IR objects deallocated by a binding (op removal, DCE,
// outlining, signature removal) are forgotten by the state, so a stale handle
// raises instead of dereferencing freed memory, even if the allocator reuses
// the address for a new object. Serializing the model marks the state as
// consumed and every later access raises.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"

namespace litert::litert_model {

// Kind of Python exception raised for an `IrError`.
enum class ErrorKind {
  kLiteRt,          // litert_model.LiteRtError
  kShapeInference,  // litert_model.ShapeInferenceError
  kValidation,      // litert_model.ValidationError
};

struct PyOp;

// Removes the source location lines that LiteRT error propagation macros add
// ("ERROR: [file:line]" followed by "└ message") and joins the remaining parts.
inline std::string CleanErrorMessage(absl::string_view message) {
  std::string result;
  size_t begin = 0;
  while (begin <= message.size()) {
    size_t end = message.find('\n', begin);
    if (end == absl::string_view::npos) end = message.size();
    absl::string_view line = message.substr(begin, end - begin);
    begin = end + 1;
    static constexpr absl::string_view kTreePrefix = "\xe2\x94\x94 ";  // "└ "
    line = absl::StripPrefix(line, kTreePrefix);
    if (absl::StartsWith(line, "ERROR: [") && absl::EndsWith(line, "]")) {
      continue;
    }
    if (line.empty()) continue;
    if (!result.empty()) result += ": ";
    result += line;
  }
  return result.empty() ? std::string(message) : result;
}

// Status of an error without the side effect of `Error::Status()` (which saves
// the message as the thread's "last error").
inline LiteRtStatus StatusOf(const Error& error) {
  return static_cast<LiteRtStatus>(error.StatusValue());
}

// C++ exception translated to a `litert_model.LiteRtError` (or subclass)
// carrying the LiteRtStatus.
class IrError : public std::exception {
 public:
  IrError(LiteRtStatus status, absl::string_view message,
          ErrorKind kind = ErrorKind::kLiteRt)
      : status_(status), message_(CleanErrorMessage(message)), kind_(kind) {}

  const char* what() const noexcept override { return message_.c_str(); }
  LiteRtStatus status() const { return status_; }
  ErrorKind kind() const { return kind_; }

  // Failing op of a shape inference error, if known.
  const std::shared_ptr<PyOp>& op() const { return op_; }
  void set_op(std::shared_ptr<PyOp> op) { op_ = std::move(op); }

 private:
  LiteRtStatus status_;
  std::string message_;
  ErrorKind kind_;
  std::shared_ptr<PyOp> op_;
};

// Throws an `IrError` for a failed `Expected`.
inline void ThrowIfError(const Expected<void>& result,
                         ErrorKind kind = ErrorKind::kLiteRt) {
  if (!result) {
    throw IrError(StatusOf(result.Error()), result.Error().Message(), kind);
  }
}

template <class T>
T ValueOrThrow(Expected<T>&& result, ErrorKind kind = ErrorKind::kLiteRt) {
  if (!result) {
    throw IrError(StatusOf(result.Error()), result.Error().Message(), kind);
  }
  return std::move(*result);
}

// Owner of a `LiteRtModelT` and registry of the IR objects exposed to Python.
class ModelState {
 public:
  explicit ModelState(std::unique_ptr<LiteRtModelT> model)
      : model_(std::move(model)) {}

  ModelState(const ModelState&) = delete;
  ModelState& operator=(const ModelState&) = delete;

  // The model; throws once the model has been serialized.
  LiteRtModelT& Model() {
    if (serialized_) {
      throw IrError(kLiteRtStatusErrorRuntimeFailure,
                    "model has been serialized");
    }
    return *model_;
  }

  // Marks the model as consumed by `SerializeModel`. The `LiteRtModelT` object
  // itself stays allocated so zero-copy weight views remain valid.
  void MarkSerialized() { serialized_ = true; }
  bool serialized() const { return serialized_; }

  // Returns the registration id of `object`, registering it if needed.
  uint64_t Register(const void* object) {
    auto [it, inserted] = ids_.try_emplace(object, next_id_);
    if (inserted) {
      ++next_id_;
    }
    return it->second;
  }

  // True if `object` is still registered under `id`.
  bool IsLive(const void* object, uint64_t id) const {
    const auto it = ids_.find(object);
    return it != ids_.end() && it->second == id;
  }

  // Forgets a deallocated object.
  void Forget(const void* object) { ids_.erase(object); }

  // Takes ownership of `buffer` for the lifetime of the state and returns a
  // view of it, to be registered with the model's `BufferManager` as a
  // NON-owned buffer.
  //
  // `BufferManager` stores buffers in a `std::vector` of variants whose
  // `OwningBufferRef` alternative has a throwing move constructor, so growing
  // the vector COPIES every owned buffer and frees the original. Memory of
  // buffers registered as owned therefore moves whenever another buffer is
  // registered, which would leave zero-copy weight views dangling. Buffers kept
  // here never move.
  BufferRef<uint8_t> KeepBuffer(OwningBufferRef<uint8_t>&& buffer) {
    kept_buffers_.push_back(std::move(buffer));
    OwningBufferRef<uint8_t>& kept = kept_buffers_.back();
    const uint8_t* data = kept.Data();
    stable_data_.insert(data);
    return BufferRef<uint8_t>(data, kept.Size());
  }

  // True if `data` is the start of a buffer kept by `KeepBuffer`.
  bool IsKeptBuffer(const uint8_t* data) const {
    return stable_data_.contains(data);
  }

 private:
  // Declared before `model_` so that the model (which refers to these buffers)
  // is destroyed first.
  std::list<OwningBufferRef<uint8_t>> kept_buffers_;
  absl::flat_hash_set<const uint8_t*> stable_data_;
  std::unique_ptr<LiteRtModelT> model_;
  bool serialized_ = false;
  absl::flat_hash_map<const void*, uint64_t> ids_;
  uint64_t next_id_ = 1;
};

using State = std::shared_ptr<ModelState>;

// Handle to a subgraph.
struct PySubgraph {
  State state;
  LiteRtSubgraphT* subgraph = nullptr;
  uint64_t id = 0;

  static PySubgraph Make(const State& state, LiteRtSubgraphT* subgraph) {
    return PySubgraph{state, subgraph, state->Register(subgraph)};
  }

  LiteRtSubgraphT& Get() const {
    state->Model();
    if (!state->IsLive(subgraph, id)) {
      throw IrError(kLiteRtStatusErrorNotFound,
                    "subgraph has been removed from the model");
    }
    return *subgraph;
  }
};

// Handle to an op (and its owning subgraph).
struct PyOp {
  PySubgraph parent;
  LiteRtOpT* op = nullptr;
  uint64_t id = 0;

  static PyOp Make(const PySubgraph& parent, LiteRtOpT* op) {
    return PyOp{parent, op, parent.state->Register(op)};
  }

  LiteRtOpT& Get() const {
    parent.Get();
    if (!parent.state->IsLive(op, id)) {
      throw IrError(kLiteRtStatusErrorNotFound,
                    "op has been removed from the model");
    }
    return *op;
  }
};

// Handle to a tensor (and its owning subgraph).
struct PyTensor {
  PySubgraph parent;
  LiteRtTensorT* tensor = nullptr;
  uint64_t id = 0;

  static PyTensor Make(const PySubgraph& parent, LiteRtTensorT* tensor) {
    return PyTensor{parent, tensor, parent.state->Register(tensor)};
  }

  LiteRtTensorT& Get() const {
    parent.Get();
    if (!parent.state->IsLive(tensor, id)) {
      throw IrError(kLiteRtStatusErrorNotFound,
                    "tensor has been removed from the model");
    }
    return *tensor;
  }
};

// Handle to a signature.
struct PySignature {
  State state;
  LiteRtSignatureT* signature = nullptr;
  uint64_t id = 0;

  static PySignature Make(const State& state, LiteRtSignatureT* signature) {
    return PySignature{state, signature, state->Register(signature)};
  }

  LiteRtSignatureT& Get() const {
    state->Model();
    if (!state->IsLive(signature, id)) {
      throw IrError(kLiteRtStatusErrorNotFound,
                    "signature has been removed from the model");
    }
    return *signature;
  }
};

// Records the ops and tensors of a subgraph before an operation that may
// deallocate some of them, and forgets the deallocated ones when destroyed (or
// on an explicit `Commit`). The operation must not allocate IR objects after
// freeing some: a new object at a freed address would keep the stale id.
class FreedObjectTracker {
 public:
  FreedObjectTracker(ModelState& state, const LiteRtSubgraphT& subgraph)
      : state_(state), subgraph_(subgraph) {
    for (const LiteRtOpT* op : subgraph.Ops()) before_.insert(op);
    for (const LiteRtTensorT* tensor : subgraph.Tensors()) {
      before_.insert(tensor);
    }
  }

  FreedObjectTracker(const FreedObjectTracker&) = delete;
  FreedObjectTracker& operator=(const FreedObjectTracker&) = delete;

  ~FreedObjectTracker() { Commit(); }

  void Commit() {
    for (const LiteRtOpT* op : subgraph_.Ops()) before_.erase(op);
    for (const LiteRtTensorT* tensor : subgraph_.Tensors()) {
      before_.erase(tensor);
    }
    for (const void* freed : before_) state_.Forget(freed);
    before_.clear();
  }

 private:
  ModelState& state_;
  const LiteRtSubgraphT& subgraph_;
  absl::flat_hash_set<const void*> before_;
};

}  // namespace litert::litert_model

#endif  // ODML_LITERT_LITERT_PYTHON_LITERT_MODEL_PYWRAP_HANDLES_H_
