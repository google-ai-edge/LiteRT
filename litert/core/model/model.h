// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_MODEL_H_
#define ODML_LITERT_LITERT_CORE_MODEL_MODEL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/internal/litert_logging.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/build_stamp.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/ir_allocator.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

////////////////////////////////////////////////////////////////////////////////
// Internal LiteRtIR
//
// These are the backing definitions for the opaque types in the c api
// (c/litert_model.h).
//
// < STORAGE DETAIL >
//
// Unless deleted as a result of calls c api client, the lifetime of all "IR
// Objects" (definitions of opaque types) are designed to be transitively owned
// by the LiteRtModelT which is generally the longset living object. See various
// "Emplace" methods.
//
// Since c api clients interface with pointers to IR Ojbects, a form of pointer
// stability is desirable. Classes in this file enforce that pointers to IR
// Objects are valid for their entire life time. Thus a c api client may store
// pointers and depend on referential equality of IR Objects thoughout different
// calls. This also facilitates storing edge/parent-references as pointers
// within IR Objects.
//
// Direct copying is generally not allowed for IR Objects since copying
// instances of mutually recursive types is not entirely well-defined.
//
// IR Objects are generally default constructible to facilitate stable storage
// and iterative construction.
//
// < EXPOSING TFLITE SCHEMA >
//
// Direct access to tflite schema types is limited to the "detail" namespace.
// This indicates that encapsulating all the details of the flatbuffer is a WIP.
// Future implementations may use different data forms (new litert serialized
// format, tflite runtime types etc).
//
// < USAGE NOTE >
//
// The classes here contain only simple getters & setters. Care should be taken
// to leave the IR in a valid state when using setters since the graph is
// doubly-linked. Higher-level functionality for correct graph mutation can be
// found in "model_graph.h".
////////////////////////////////////////////////////////////////////////////////

// All tflite schema type usage.
namespace litert::internal {

// OP

// Placeholder for the ind of the dispatch op code added during serialization.
static constexpr auto kDispatchOpCodeTflInd = -1;

void SetTflOpCodeInd(LiteRtOpT& litert_op, int32_t tfl_op_code_ind);

int32_t GetTflOpCodeInd(const LiteRtOpT& litert_op);

template <class Arg>
void SetTflOptions(LiteRtOpT& litert_op, Arg&& arg);

template <class Arg>
void SetTflOptions2(LiteRtOpT& litert_op, Arg&& arg);

const ::litert::internal::TflOptions& GetTflOptions(const LiteRtOpT& litert_op);

const ::litert::internal::TflOptions2& GetTflOptions2(
    const LiteRtOpT& litert_op);

::litert::internal::TflOptions&& TakeTflOptions(LiteRtOpT& litert_op);

::litert::internal::TflOptions2&& TakeTflOptions2(LiteRtOpT& litert_op);

void ClearTflOptions(LiteRtOpT& litert_op);

// MODEL

const std::vector<::litert::internal::TflOpCodePtr>& GetTflOpCodes(
    const LiteRtModelT& litert_model);

template <class Arg>
void SetTflOpCodes(LiteRtModelT& litert_model, Arg&& arg);

std::vector<::litert::internal::TflOpCodePtr>&& TakeTflOpCodes(
    LiteRtModelT& litert_model);

void SetTflFlatbuffer(LiteRtModelT& litert_model,
                      ::litert::internal::FlatbufferWrapper&& tfl_flatbuffer);

const ::litert::internal::FlatbufferWrapper& GetTflFlatbuffer(
    const LiteRtModelT& litert_model);

}  // namespace litert::internal

//
// Helpers for conceptual unions from C api.
//

// // For requesting opaque data stored within IR.
using ScratchBufferProvider = std::function<uint8_t*(size_t size)>;

// TENSOR TYPE

// Detail convenience type for tensor type union.
typedef union {
  LiteRtUnrankedTensorType unranked_tensor_type;
  LiteRtRankedTensorType ranked_tensor_type;
} TensorTypeDetail;

// Union and identifier for tensor types.
using TensorType = std::pair<LiteRtTensorTypeId, TensorTypeDetail>;

// Construct tensor type union as ranked tensor. NOTE: Copies data in `dims`.
TensorType MakeRankedTensorType(LiteRtElementType element_type,
                                absl::Span<const int32_t> dims);

// QUANTIZATION TYPE

// Detail convenience type for quantization type union.
typedef union {
  LiteRtQuantizationPerTensor per_tensor;
  LiteRtQuantizationPerChannel per_channel;
} QuantizationDetail;

// Union and identifier for quantization types.
using Quantization = std::pair<LiteRtQuantizationTypeId, QuantizationDetail>;

// Make default type with quantization info.
inline Quantization MakeEmptyQuantization() {
  return Quantization(kLiteRtQuantizationNone, QuantizationDetail());
}

// Construct quantization type as per tensor.
Quantization MakePerTensorQuantization(float scale, int64_t zero_point);

// Construct quantization type as per channel, requires buffer callback to
// store data.
template <class Scales, class ZeroPoints>
Quantization MakePerChannelQuantization(const Scales& scales,
                                        const ZeroPoints& zero_points,
                                        int32_t quantized_dim,
                                        ScratchBufferProvider buffer_provider) {
  const auto size = std::size(scales);
  ABSL_DCHECK_EQ(size, std::size(zero_points));

  Quantization res;
  res.first = kLiteRtQuantizationPerChannel;

  res.second.per_channel.num_channels = size;
  res.second.per_channel.quantized_dimension = quantized_dim;

  const size_t scales_buf_size = size * sizeof(float);
  const size_t zeros_buf_size = size * sizeof(int64_t);
  auto* scales_buf = reinterpret_cast<float*>(buffer_provider(scales_buf_size));
  auto* zeros_buf = reinterpret_cast<int64_t*>(buffer_provider(zeros_buf_size));
  std::copy(std::cbegin(scales), std::cend(scales), scales_buf);
  std::copy(std::cbegin(zero_points), std::cend(zero_points), zeros_buf);

  res.second.per_channel.scales = scales_buf;
  res.second.per_channel.zero_points = zeros_buf;

  return res;
}

//
// Tensor
//

// Constant data associated with a tensor.
class LiteRtWeightsT {
 private:
  using OwnedBuffer = ::litert::OwningBufferRef<uint8_t>;

 public:
  using BufferId = ::litert::internal::BufferManager::BufferId;
  using BufferManager = ::litert::internal::BufferManager;

  // Underlying data.
  ::litert::BufferRef<uint8_t> Buffer() const {
    auto buf = GetBufferManager()->GetBuffer(buffer_id_);
    ABSL_DCHECK(buf.HasValue());
    return *buf;
  }

  // Set the buffer manager, expects a stable pointer. A default buffer manager
  // will be initialized for convenience but most cases will share a single
  // buffer manager owned by the model.
  void SetBufferManager(BufferManager* buffer_manager) {
    buffer_manager_ = buffer_manager;
  }

  // Get the underlying buffer manager.
  BufferManager* GetBufferManager() const {
    if (std::holds_alternative<BufferManager*>(buffer_manager_)) {
      return std::get<BufferManager*>(buffer_manager_);
    } else {
      return std::get<BufferManager::Ptr>(buffer_manager_).get();
    }
  }

  // Set from a pre-registered buffer. This expects buffer was registered
  // with the same manager.
  void SetBufferId(BufferId buffer_id) { buffer_id_ = buffer_id; }

  // Get the id generated for the buffer by the manager.
  BufferId GetBufferId() const { return buffer_id_; }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtWeightsT() = default;
  explicit LiteRtWeightsT(BufferManager* buffer_manager)
      : buffer_manager_(buffer_manager) {}
  LiteRtWeightsT(const LiteRtWeightsT&) = delete;
  LiteRtWeightsT(LiteRtWeightsT&&) = default;
  LiteRtWeightsT& operator=(const LiteRtWeightsT&) = delete;
  LiteRtWeightsT& operator=(LiteRtWeightsT&&) = default;

 private:
  BufferId buffer_id_ = BufferManager::kEmptyBufferId;
  std::variant<BufferManager*, BufferManager::Ptr> buffer_manager_ =
      std::make_unique<BufferManager>();
};

// Set weights via an unowned buffer. Caller is responsible for ensuring the
// buffer outlives the weights. Registers the buffer with the manager.
inline void SetWeightsFromUnownedBuffer(
    LiteRtWeightsT& weights, ::litert::BufferRef<uint8_t> buffer,
    std::optional<litert::internal::BufferContext> context = std::nullopt) {
  auto* manager = weights.GetBufferManager();
  auto buf_id = manager->RegisterNonOwnedBuffer(buffer, context);
  weights.SetBufferId(buf_id);
}

// Set weights via an unowned buffer. Caller is responsible for ensuring the
// buffer outlives the weights. Registers the buffer with the manager.
inline void SetWeightsFromOwnedBuffer(
    LiteRtWeightsT& weights, ::litert::OwningBufferRef<uint8_t>&& buffer,
    std::optional<litert::internal::BufferContext> context = std::nullopt) {
  auto* manager = weights.GetBufferManager();
  auto buf_id = manager->RegisterOwnedBuffer(std::move(buffer), context);
  weights.SetBufferId(buf_id);
}

// Fundamental value in a litert program, "edges" in the graph.
class LiteRtTensorT {
 private:
  using UserData = std::unique_ptr<uint8_t[]>;

 public:
  using Ref = std::reference_wrapper<LiteRtTensorT>;
  using Use = std::pair<LiteRtOp, LiteRtParamIndex>;
  using UseVec = std::vector<Use>;
  using Alloc = ::litert::internal::IrAllocator<LiteRtTensorT>;

  // The ops that take this tensor as input.
  const std::vector<LiteRtOp>& Users() const { return users_; }
  std::vector<LiteRtOp>& Users() { return users_; }

  // Which operand index users take this tensor on, respects the ordering of
  // users..
  const std::vector<LiteRtParamIndex>& UserArgInds() const {
    return user_arg_inds_;
  }
  std::vector<LiteRtParamIndex>& UserArgInds() { return user_arg_inds_; }

  // Number of uses, same as number of user arg inds.
  size_t NumUses() const { return users_.size(); }

  // Get the ith use.
  Use GetUse(size_t ind) const {
    return {users_.at(ind), user_arg_inds_.at(ind)};
  }

  // Remove the use at the given index.
  void RemoveUse(size_t ind) {
    users_.erase(users_.begin() + ind);
    user_arg_inds_.erase(user_arg_inds_.begin() + ind);
  }

  // Get the op that outputs this tensor, null if constant or subgraph input.
  LiteRtOp DefiningOp() const { return defining_op_; }

  // Get the output index of the op that defines this tensor, only meaningful
  // if it has a defining op.
  LiteRtParamIndex DefiningOpOutInd() const { return defining_op_out_ind_; }

  // Update the defining op of this tensor. The caller is required to update the
  // given op's output if not already correct.
  void SetDefiningOp(LiteRtOpT& defining_op, LiteRtParamIndex out_ind) {
    defining_op_ = &defining_op;
    defining_op_out_ind_ = out_ind;
  }

  // Set the defining op to none.
  void ClearDefiningOp() {
    defining_op_ = nullptr;
    defining_op_out_ind_ = 0;
  }

  // Any constant data associated with this tensor.
  const LiteRtWeightsT& Weights() const { return weights_; }
  LiteRtWeightsT& Weights() { return weights_; }

  // Authored name associated with this tensor. May be empty.
  absl::string_view Name() const { return name_; }

  // Update the name associated with this tensor.
  void SetName(std::string name) { name_ = std::move(name); }

  // Get tensor index associated with this tensor.
  uint32_t TensorIndex() const { return tensor_index_; }

  // Update the index associated with this tensor.
  void SetTensorIndex(uint32_t tensor_index) { tensor_index_ = tensor_index; }

  // Get quantization information for this tensor.
  const Quantization& Qparams() const { return quantization_; }
  Quantization& Qparams() { return quantization_; }

  // Set quantization information.
  template <class Arg>
  void SetQarams(Arg&& arg) {
    quantization_ = std::forward<Arg>(arg);
  }

  // Get the tensor type of this tensor.
  const TensorType& Type() const { return tensor_type_; }
  TensorType& Type() { return tensor_type_; }

  // Get ranked type directly.
  ::litert::Expected<LiteRtRankedTensorType> Ranked() const {
    if (Type().first == kLiteRtRankedTensorType) {
      return Type().second.ranked_tensor_type;
    }
    return ::litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Tensor type is not ranked");
  }

  // Number of elements in the tensor.
  size_t NumElements() const {
    auto ranked = Ranked();
    if (!ranked) {
      return 0;
    }
    const auto& dims = ranked->layout.dimensions;
    return static_cast<size_t>(std::accumulate(
        std::cbegin(dims), std::cend(dims), 1,
        std::multiplies<std::remove_reference_t<decltype(dims[0])>>()));
  }

  // Set the tensor type.
  template <class Arg>
  void SetType(Arg&& arg) {
    tensor_type_ = std::forward<Arg>(arg);
  }

  // Get a new buffer that will live as long as this tensor. Used for storing
  // various buffers passed through c-api (dims, quantization etc).
  // NOTE: This is just scratch data unrelated to weights buffer.
  uint8_t* RequestScratchBuffer(size_t size) {
    user_data_.push_back(std::make_unique<uint8_t[]>(size));
    return user_data_.back().get();
  }

  // Allow for implicit conversion to scratch buffer provider.
  // NOTE: This is just scratch data unrelated to weights buffer.
  // NOLINTNEXTLINE
  operator ScratchBufferProvider() & {
    return [this](auto s) { return this->RequestScratchBuffer(s); };
  }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtTensorT() = default;
  LiteRtTensorT(::litert::internal::BufferManager* buffer_manager)
      : weights_(buffer_manager) {}
  LiteRtTensorT(const LiteRtTensorT&) = delete;
  LiteRtTensorT(LiteRtTensorT&&) = default;
  LiteRtTensorT& operator=(const LiteRtTensorT&) = delete;
  LiteRtTensorT& operator=(LiteRtTensorT&&) = default;

 private:
  std::vector<LiteRtOp> users_;
  std::vector<LiteRtParamIndex> user_arg_inds_;

  LiteRtOp defining_op_ = nullptr;
  LiteRtParamIndex defining_op_out_ind_;

  LiteRtWeightsT weights_;
  Quantization quantization_;
  TensorType tensor_type_;

  std::string name_;

  std::uint32_t tensor_index_;

  std::vector<UserData> user_data_;
};

// Helper to get multiple uses at once.
template <class Inds>
LiteRtTensorT::UseVec GetTensorUses(const LiteRtTensorT& tensor,
                                    const Inds& inds) {
  auto start = std::cbegin(inds);
  auto end = std::cend(inds);
  LiteRtTensorT::UseVec uses(end - start);
  auto get = [&tensor = std::as_const(tensor)](auto i) {
    return tensor.GetUse(i);
  };
  std::transform(start, end, uses.begin(), get);
  return uses;
}

//
// Op
//

// Fundamental unit of compute of a litert program, or "nodes" in the graph.
class LiteRtOpT {
 public:
  using Ref = std::reference_wrapper<LiteRtOpT>;
  using Alloc = ::litert::internal::IrAllocator<LiteRtOpT>;

  // Input tensors for this op.
  const std::vector<LiteRtTensor>& Inputs() const { return inputs_; }
  std::vector<LiteRtTensor>& Inputs() { return inputs_; }

  // Access input at given ind.
  LiteRtTensorT& Input(size_t ind) { return *Inputs().at(ind); }
  const LiteRtTensorT& Input(size_t ind) const { return *Inputs().at(ind); }

  // Number of input tensors.
  size_t NumInputs() const { return inputs_.size(); }

  // Output tensors for this op.
  const std::vector<LiteRtTensor>& Outputs() const { return outputs_; }
  std::vector<LiteRtTensor>& Outputs() { return outputs_; }

  // Number of output tensors.
  size_t NumOutputs() const { return outputs_.size(); }

  // Access output at given ind.
  LiteRtTensorT& Output(size_t ind) { return *Outputs().at(ind); }
  const LiteRtTensorT& Output(size_t ind) const { return *Outputs().at(ind); }

  // Remove the ith entry of input list.
  void RemoveInput(size_t ind) { inputs_.erase(inputs_.begin() + ind); }

  // Remove the ith entry of output list.
  void RemoveOutput(size_t ind) { outputs_.erase(outputs_.begin() + ind); }

  // Get any custom options attached to this op. Empty if there are none.
  litert::BufferRef<uint8_t> CustomOptions() const { return custom_options_; }

  // Op index. (For internal debugging only)
  void SetOpIndex(uint32_t op_index) { op_index_ = op_index; }
  uint32_t OpIndex() const { return op_index_; }

  // Attach custom opaque optins to this op.
  template <class... Args>
  void SetCustomOptions(Args&&... args) {
    custom_options_ =
        ::litert::OwningBufferRef<uint8_t>(std::forward<Args>(args)...);
  }

  // Sets the custom options to zero length buffer.
  void ClearCustomOptions() { custom_options_.Reset(); }

  // Get the op code.
  LiteRtOpCode OpCode() const { return litert_op_code_; }

  // Set the op code.
  void SetOpCode(LiteRtOpCode litert_op_code) {
    litert_op_code_ = litert_op_code;
  }

  // Get the custom code if the op is a custom op.
  ::litert::Expected<absl::string_view> CustomCode() const {
    if (OpCode() != kLiteRtOpCodeTflCustom) {
      return ::litert::Error(kLiteRtStatusErrorInvalidArgument,
                             "Op code is not custom");
    }
    return absl::string_view(custom_code_);
  }

  // Set the custom code.
  void SetCustomCode(std::string custom_code) {
    custom_code_ = std::move(custom_code);
  }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtOpT() = default;
  LiteRtOpT(const LiteRtOpT&) = delete;
  LiteRtOpT(LiteRtOpT&&) = default;
  LiteRtOpT& operator=(const LiteRtOpT&) = delete;
  LiteRtOpT& operator=(LiteRtOpT&&) = default;

  // Friendship for internal tflite details.
  friend void litert::internal::SetTflOpCodeInd(LiteRtOpT& litert_op,
                                                int32_t tfl_op_code_ind);

  friend int32_t litert::internal::GetTflOpCodeInd(const LiteRtOpT& litert_op);

  template <class Arg>
  friend void litert::internal::SetTflOptions(LiteRtOpT& litert_op, Arg&& arg);

  template <class Arg>
  friend void litert::internal::SetTflOptions2(LiteRtOpT& litert_op, Arg&& arg);

  friend const ::litert::internal::TflOptions& litert::internal::GetTflOptions(
      const LiteRtOpT& litert_op);

  friend const ::litert::internal::TflOptions2&
  litert::internal::GetTflOptions2(const LiteRtOpT& litert_op);

  friend ::litert::internal::TflOptions&& litert::internal::TakeTflOptions(
      LiteRtOpT& litert_op);

  friend ::litert::internal::TflOptions2&& litert::internal::TakeTflOptions2(
      LiteRtOpT& litert_op);

  friend void litert::internal::ClearTflOptions(LiteRtOpT& litert_op);

 private:
  LiteRtOpCode litert_op_code_;

  ::litert::OwningBufferRef<uint8_t> custom_options_;

  std::vector<LiteRtTensor> inputs_;
  std::vector<LiteRtTensor> outputs_;

  std::string custom_code_;

  uint32_t op_index_;

  // TFLITE
  int32_t tfl_op_code_ind_ = litert::internal::kDispatchOpCodeTflInd;
  ::litert::internal::TflOptions tfl_option_;
  ::litert::internal::TflOptions2 tfl_option_2_;
};

// Clears any attribute data and sets the op to be a dispatch op.
inline void MakeDispatchOp(LiteRtOpT& op) {
  litert::internal::ClearTflOptions(op);
  op.ClearCustomOptions();
  op.SetOpCode(kLiteRtOpCodeTflCustom);
  litert::internal::SetTflOpCodeInd(op,
                                    litert::internal::kDispatchOpCodeTflInd);
}

//
// Subgraph
//

// Fundamental block of a litert program. Manages the storage of all
// ops and tensor within.
class LiteRtSubgraphT {
 public:
  using Ref = std::reference_wrapper<LiteRtSubgraphT>;
  using Alloc = ::litert::internal::IrAllocator<LiteRtSubgraphT>;

  // Get a stable pointer for all of the tensors in this subgraph.
  absl::Span<LiteRtTensor> Tensors() { return tensors_.Elements(); }
  absl::Span<const LiteRtTensor> Tensors() const { return tensors_.Elements(); }

  // Access the tensor at given ind.
  LiteRtTensorT& Tensor(size_t ind) { return *Tensors().at(ind); }
  const LiteRtTensorT& Tensor(size_t ind) const { return *Tensors().at(ind); }

  // Get a stable pointer for all of the ops in this subgraph. Will
  // be a valid toplological order.
  absl::Span<LiteRtOp> Ops() { return ops_.Elements(); }
  absl::Span<const LiteRtOp> Ops() const { return ops_.Elements(); }

  // Access op at the given ind.
  LiteRtOpT& Op(size_t ind) { return *Ops().at(ind); }
  const LiteRtOpT& Op(size_t ind) const { return *Ops().at(ind); }

  // All the subgraph input tensors, these also exist in Tensors.
  const std::vector<LiteRtTensor>& Inputs() const { return inputs_; }
  std::vector<LiteRtTensor>& Inputs() { return inputs_; }

  // Number of inputs tensors.
  size_t NumInputs() const { return inputs_.size(); }

  // Access the subgraph input at given ind.
  LiteRtTensorT& Input(size_t ind) { return *Inputs().at(ind); }
  const LiteRtTensorT& Input(size_t ind) const { return *Inputs().at(ind); }

  // All the subgraph output tensors, these also exist in Tensors.
  const std::vector<LiteRtTensor>& Outputs() const { return outputs_; }
  std::vector<LiteRtTensor>& Outputs() { return outputs_; }

  // Number of outputs tensors.
  size_t NumOutputs() const { return outputs_.size(); }

  // Access the subgraph output at given ind.
  LiteRtTensorT& Output(size_t ind) { return *Outputs().at(ind); }
  const LiteRtTensorT& Output(size_t ind) const { return *Outputs().at(ind); }

  // Clear the entry for the ith input.
  void ClearInput(size_t ind) { inputs_.erase(inputs_.begin() + ind); }

  // Clear the entry for the ith output.
  void ClearOutput(size_t ind) { outputs_.erase(outputs_.begin() + ind); }

  // Construct a new tensor which will be owned by this subgraph and get a
  // reference to it.
  template <class... Args>
  LiteRtTensorT& EmplaceTensor(Args&&... args) {
    if (buffer_manager_ == nullptr) {
      return tensors_.EmplaceBack(std::forward<Args>(args)...);
    } else {
      // std::cerr << "Emplacing tensor with buffer manager \n";
      return tensors_.EmplaceBack(buffer_manager_, std::forward<Args>(args)...);
    }
  }

  // Construct a new op which will be owned by this subgraph and get a
  // reference to it.
  template <class... Args>
  LiteRtOpT& EmplaceOp(Args&&... args) {
    return ops_.EmplaceBack(std::forward<Args>(args)...);
  }

  // Construct a new op which will be owned by this subgraph and get a
  // reference to it.
  template <class... Args>
  LiteRtOpT& EmplaceOpAt(int index, Args&&... args) {
    return ops_.EmplaceAt(index, std::forward<Args>(args)...);
  }

  // De-allocates ops that pass given predicate. Returns number of ops removed.
  size_t RemoveOpIf(std::function<bool(const LiteRtOpT& op)> pred) {
    return ops_.RemoveIf(pred);
  }

  // De-allocates tensors that pass given predicate. Returns number of tensors
  // removed.
  size_t RemoveTensorIf(std::function<bool(const LiteRtTensorT& tensor)> pred) {
    return tensors_.RemoveIf(pred);
  }

  // Transfers the ownership of ops from the given allocator to this subgraph.
  // Ops are appended in order.
  void TransferOpsFrom(LiteRtOpT::Alloc& other, size_t index) {
    ops_.TransferFrom(other, index);
  }
  // Transfers the ownership of tensors from the given allocator to this
  // subgraph. Tensors are appended in order.
  void TransferTensorsFrom(LiteRtTensorT::Alloc& other) {
    tensors_.TransferFrom(other);
  }

  LiteRtOpT::Alloc& OpsAllocation() { return ops_; }
  LiteRtTensorT::Alloc& TensorsAllocation() { return tensors_; }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtSubgraphT() = default;
  LiteRtSubgraphT(::litert::internal::BufferManager* buffer_manager)
      : buffer_manager_(buffer_manager) {};
  LiteRtSubgraphT(const LiteRtSubgraphT&) = delete;
  LiteRtSubgraphT(LiteRtSubgraphT&&) = default;
  LiteRtSubgraphT& operator=(const LiteRtSubgraphT&) = delete;
  LiteRtSubgraphT& operator=(LiteRtSubgraphT&&) = default;

  // Get the buffer manager for this subgraph.
  ::litert::internal::BufferManager* GetBufferManager() const {
    return buffer_manager_;
  }

 private:
  // If null, tensors emplaced will own their own buffer managers.
  ::litert::internal::BufferManager* buffer_manager_ = nullptr;

  LiteRtTensorT::Alloc tensors_;

  LiteRtOpT::Alloc ops_;

  std::vector<LiteRtTensor> inputs_;
  std::vector<LiteRtTensor> outputs_;
};

//
// Signature
//

class LiteRtSignatureT {
 private:
  using StrVec = std::vector<std::string>;

 public:
  using Ptr = std::unique_ptr<LiteRtSignatureT>;
  using Ref = std::reference_wrapper<LiteRtSignatureT>;
  using Alloc = ::litert::internal::IrAllocator<LiteRtSignatureT>;

  static constexpr absl::string_view kDefaultSignatureKey =
      "<placeholder signature>";

  LiteRtSignatureT(LiteRtSubgraph subgraph, StrVec input_names,
                   std::vector<LiteRtTensor> input_tensors, StrVec output_names,
                   std::vector<LiteRtTensor> output_tensors, std::string key)
      : key_(std::move(key)),
        subgraph_(subgraph),
        input_names_(std::move(input_names)),
        input_tensors_(std::move(input_tensors)),
        output_names_(std::move(output_names)),
        output_tensors_(std::move(output_tensors)) {
    ABSL_DCHECK_EQ(input_names_.size(), input_tensors_.size());
    ABSL_DCHECK_EQ(output_names_.size(), output_tensors_.size());
    for (size_t i = 0; i < input_names_.size(); ++i) {
      input_name_to_tensor_.emplace(input_names_[i], input_tensors_[i]);
    }
    for (size_t i = 0; i < output_names_.size(); ++i) {
      output_name_to_tensor_.emplace(output_names_[i], output_tensors_[i]);
    }
  }

  // String named inputs for called subgraph.
  const StrVec& InputNames() const { return input_names_; }

  // String named outputs for called subgraph.
  const StrVec& OutputNames() const { return output_names_; }

  // Get the input tensor at the given index.
  LiteRtTensor GetInputTensor(size_t index) const {
    return input_tensors_.at(index);
  }

  // Get the output tensor at the given index.
  LiteRtTensor GetOutputTensor(size_t index) const {
    return output_tensors_.at(index);
  }

  // Find the input tensor with the given name.
  ::litert::Expected<LiteRtTensor> FindInputTensor(
      absl::string_view name) const {
    if (auto it = input_name_to_tensor_.find(std::string(name));
        it != input_name_to_tensor_.end()) {
      return it->second;
    }
    return ::litert::Unexpected(kLiteRtStatusErrorNotFound,
                                "Signature input alias not found");
  }

  // Find the output tensor with the given name.
  ::litert::Expected<LiteRtTensor> FindOutputTensor(
      absl::string_view name) const {
    if (auto it = output_name_to_tensor_.find(std::string(name));
        it != output_name_to_tensor_.end()) {
      return it->second;
    }
    return ::litert::Unexpected(kLiteRtStatusErrorNotFound,
                                "Signature output alias not found");
  }

  // Get the callable subgraph.
  const LiteRtSubgraphT& GetSubgraph() const { return *subgraph_; }
  LiteRtSubgraphT& GetSubgraph() { return *subgraph_; }

  // Name of the callable signature.
  absl::string_view Key() const { return key_; }

  bool operator==(const LiteRtSignatureT& other) const {
    const auto key_eq = key_ == other.key_;
    const auto subgraph_eq = subgraph_ == other.subgraph_;
    const auto input_names_eq = input_names_ == other.input_names_;
    const auto input_tensors_eq = input_tensors_ == other.input_tensors_;
    const auto output_names_eq = output_names_ == other.output_names_;
    const auto output_tensors_eq = output_tensors_ == other.output_tensors_;
    return key_eq && subgraph_eq && input_names_eq && input_tensors_eq &&
           output_names_eq && output_tensors_eq;
  }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtSignatureT() = default;
  LiteRtSignatureT(const LiteRtSignatureT&) = delete;
  LiteRtSignatureT(LiteRtSignatureT&&) = default;
  LiteRtSignatureT& operator=(const LiteRtSignatureT&) = delete;
  LiteRtSignatureT& operator=(LiteRtSignatureT&&) = default;

 private:
  std::string key_;

  LiteRtSubgraph subgraph_;

  StrVec input_names_;
  std::vector<LiteRtTensor> input_tensors_;
  std::unordered_map<std::string, LiteRtTensor> input_name_to_tensor_;
  StrVec output_names_;
  std::vector<LiteRtTensor> output_tensors_;
  std::unordered_map<std::string, LiteRtTensor> output_name_to_tensor_;
};

// Make a basic signature from information in the given subgraph. Used with the
// main subgraph when no explicit signatures have been authored.
LiteRtSignatureT MakeDefaultSignature(LiteRtSubgraph subgraph);

// Rewriter (Experimental feature)

// The LiteRtRewriterT class provides an interface to build and modify
// LiteRtSubgraphT instances in a transactional manner. It allows for the
// creation of new tensors and operators, cloning existing ones, and marking
// operators for erasure. Changes are accumulated within the rewriter and
// applied to a target subgraph only when ApplyChanges() is called. This ensures
// atomic updates to the graph structure.

class LiteRtRewriterT {
 public:
  LiteRtRewriterT() = default;
  LiteRtRewriterT(const LiteRtRewriterT&) = delete;
  LiteRtRewriterT(LiteRtRewriterT&&) = default;
  LiteRtRewriterT& operator=(const LiteRtRewriterT&) = delete;
  LiteRtRewriterT& operator=(LiteRtRewriterT&&) = default;

  // Get the subgraph that is being rewritten.
  LiteRtSubgraphT& Subgraph() { return subgraph_; }

  // Returns the set of ops that are marked for erases.
  absl::flat_hash_set<LiteRtOp> Erases() const { return erases_; }

  // Builds a new LiteRt tensor owned by the rewriter.
  LiteRtTensorT& BuildTensor(const LiteRtWeightsT& weights,
                             Quantization quantization, TensorType tensor_type,
                             std::optional<std::string> name = std::nullopt);

  // Builds a new LiteRt tensor owned by the rewriter, clone of src.
  LiteRtTensorT& BuildTensor(const LiteRtTensorT& src);

  // Builds a new LiteRt op owned by the rewriter.
  LiteRtOpT& BuildOp(LiteRtOpCode code, std::vector<LiteRtTensor> inputs,
                     std::vector<LiteRtTensor> outputs);

  // Builds a new LiteRt op owned by the rewriter, clone of src.
  LiteRtOpT& BuildOp(LiteRtOpT& src, std::vector<LiteRtTensor> inputs,
                     std::vector<LiteRtTensor> outputs);

  // Checks if op is allocated in rewriter.
  bool IsOpAllocated(LiteRtOp op) const { return allocated_ops_.contains(op); }

  // Checks if tensor is allocated in rewriter.
  bool IsTensorAllocated(LiteRtTensor tensor) const {
    return allocated_tensors_.contains(tensor);
  }

  // Transactionally erases an op, changes won't be applied until ApplyChanges
  // is called.
  void EraseOp(LiteRtOp opToErase);

  // Applies all changes to the given subgraph, that was recorded by the
  // rewriter.
  //
  // Note: This internal function is intentionally not exposed to the public
  // API, to avoid users from accidentally applying changes mid-computation.
  void ApplyChanges(LiteRtSubgraphT* subgraph_to_apply);

 private:
  // Subgraph to hold the IR.
  LiteRtSubgraphT subgraph_;

  // Records of transactions.
  absl::flat_hash_set<LiteRtOp> erases_;
  absl::flat_hash_set<LiteRtTensor> allocated_tensors_;
  absl::flat_hash_set<LiteRtOp> allocated_ops_;
};

//
// Model
//

// Root-level graph object for litert programs. Manages the storage
// of all litert graph objects within.
class LiteRtModelT {
 public:
  using Ref = std::reference_wrapper<LiteRtModelT>;
  using Ptr = std::unique_ptr<LiteRtModelT>;
  using TflOpCodes = std::vector<litert::internal::TflOpCodePtr>;

  using BufferManager = ::litert::internal::BufferManager;
  using StoredBufferManager = std::variant<BufferManager::Ptr, BufferManager*>;
  using BufferId = BufferManager::BufferId;

  using OpAssetReference = std::pair<BufferId, std::string>;
  using OpAssetMap = std::unordered_map<LiteRtOp, OpAssetReference>;

  using MetadataMap = std::unordered_map<std::string, BufferId>;

  using TflFlatbuffer = ::litert::internal::FlatbufferWrapper;

  // TODO replace this with the index of the default signature.
  static constexpr const size_t kMainSubgraphIndex = 0;

  // SUBGRAPHS

  // Get a stable pointer for all of the subgraphs within this model.
  absl::Span<LiteRtSubgraph> Subgraphs() { return subgraphs_.Elements(); }
  absl::Span<const LiteRtSubgraph> Subgraphs() const {
    return subgraphs_.Elements();
  }

  // Access subgraph at given ind.
  LiteRtSubgraphT& Subgraph(size_t ind) { return *Subgraphs().at(ind); }
  const LiteRtSubgraphT& Subgraph(size_t ind) const {
    return *Subgraphs().at(ind);
  }

  // Number of subraphs.
  size_t NumSubgraphs() const { return subgraphs_.Elements().size(); }

  // Default entry point of this model.
  const LiteRtSubgraphT* MainSubgraph() const {
    return &Subgraph(kMainSubgraphIndex);
  }
  LiteRtSubgraph MainSubgraph() { return &Subgraph(kMainSubgraphIndex); }

  // Look up signature by key.
  litert::Expected<LiteRtSignatureT::Ref> FindSignature(
      absl::string_view signature_key) const {
    for (LiteRtSignature sig : signatures_.Elements()) {
      if (sig->Key() == signature_key) {
        return std::ref(*sig);
      }
    }
    return ::litert::Error(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  // Build a new subgraph and get a stable reference to it.
  template <class... Args>
  LiteRtSubgraphT& EmplaceSubgraph(Args&&... args) {
    return subgraphs_.EmplaceBack(Buffers(), std::forward<Args>(args)...);
  }

  // Transfers given subgraphs into this model. New subgraphs are appended.
  void TransferSubgraphsFrom(LiteRtSubgraphT::Alloc&& subgraphs) {
    // TODO: Consider merging buffer managers here.
    subgraphs_.TransferFrom(std::move(subgraphs));
  }

  // Cut all by the first `size` subgraphs. Does nothing if given size is
  // greater or equal to current.
  void ResizeSubgraphsDown(size_t size) { subgraphs_.ResizeDown(size); }

  // Transfers the subgraph at the given index to the back of the given
  // allocator. Also updates any IR owned by the model that refers to subgraphs
  // by index (e.g. composites). Does not update any IR in the subgraphs being
  // transferred.
  void TransferSubgraphTo(LiteRtSubgraphT::Alloc& dest,
                          std::vector<size_t> indices);

  // Splits a model along the given subgraph indices. Returns a new model with
  // the specified subgraphs that were moved from the model. Similar to
  // TransferSubgraphTo but also handles op codes and buffer manager.
  //
  // NOTE: This only copies enough IR to build a valid model. It does not handle
  // signatures, metadata etc.
  LiteRtModelT Yank(std::vector<size_t> indices);

  // SIGNATURES

  // All signatures registered with this model.
  absl::Span<LiteRtSignature> Signatures() const {
    return signatures_.Elements();
  }

  // Construct a new signature for this model.
  template <class... Args>
  LiteRtSignatureT& EmplaceSignature(Args&&... args) {
    return signatures_.EmplaceBack(std::forward<Args>(args)...);
  }

  // METADATA

  // Look up metadata by key, getting a view of its buffer as a string
  // if it exists.
  litert::Expected<litert::BufferRef<uint8_t>> FindMetadata(
      absl::string_view key) const {
    if (auto it = metadata_.find(std::string(key)); it != metadata_.end()) {
      const auto buf_id = it->second;
      return Buffers()->GetBuffer(buf_id);
    }
    return ::litert::Error(kLiteRtStatusErrorNotFound);
  }

  // Metadata key-val pair iterator.
  MetadataMap::iterator MetadataBegin() { return metadata_.begin(); }
  MetadataMap::iterator MetadataEnd() { return metadata_.end(); }

  // Adds a new metadata buffer to the model. Fails if it already exists.
  template <class... Args>
  LiteRtStatus PushMetadata(absl::string_view key, Args&&... args) {
    if (metadata_.find(std::string(key)) != metadata_.end()) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    const auto buf_id = Buffers()->RegisterOwnedBuffer(
        ::litert::OwningBufferRef<uint8_t>(std::forward<Args>(args)...));
    metadata_.emplace(std::make_pair(std::string(key), buf_id));
    return kLiteRtStatusOk;
  }

  // BUFFERS

  // Get stable pointer to buffer manager object.
  BufferManager* Buffers() const {
    if (std::holds_alternative<BufferManager::Ptr>(buffer_manager_)) {
      return std::get<BufferManager::Ptr>(buffer_manager_).get();
    } else {
      return std::get<BufferManager*>(buffer_manager_);
    }
  }

  // Records the original source path of the model, if known.
  void SetSourcePath(std::string path) { source_path_ = path; }

  const std::optional<std::string>& SourcePath() const { return source_path_; }

  // Attach an asset to the given op. An asset is a non-tensor buffer
  // that is used by the op. Assets may be referenced by multiple ops.
  // Each edge from an op to an asset is identified by a name. All buffers
  // are appended to the model upon serialization and referenced by offset
  // relative to the start of the model within the referring op's custom
  // options.
  void AttachAssetToOp(LiteRtOp op, BufferId buf_id, std::string name) {
    OpAssetReference ref = {buf_id, std::move(name)};
    external_buffer_map_.emplace(op, std::move(ref));
  }

  // Returns an immutable view of the external buffer and the name of the edge
  // if the given op has one attached.
  litert::Expected<OpAssetReference> FindOpAsset(LiteRtOp op) {
    if (auto it = external_buffer_map_.find(op);
        it != external_buffer_map_.end()) {
      return it->second;
    }
    return ::litert::Error(kLiteRtStatusErrorNotFound);
  }

  // Contains details about the compiler used if this model was compiled.
  struct BuildStamp {
    absl::string_view soc_manufacturer;
    absl::string_view soc_model;
  };

  // IR is generally, default constructible and movable but not copyable.
  LiteRtModelT() = default;
  LiteRtModelT(const LiteRtModelT&) = delete;
  LiteRtModelT(LiteRtModelT&&) = default;
  LiteRtModelT& operator=(const LiteRtModelT&) = delete;
  LiteRtModelT& operator=(LiteRtModelT&&) = default;

  // TFLITE

  // Friendship for internal tflite details.
  friend const TflOpCodes& litert::internal::GetTflOpCodes(
      const LiteRtModelT& litert_model);

  template <class Arg>
  friend void litert::internal::SetTflOpCodes(LiteRtModelT& litert_model,
                                              Arg&& arg);

  friend TflOpCodes&& litert::internal::TakeTflOpCodes(
      LiteRtModelT& litert_model);

  friend void litert::internal::SetTflFlatbuffer(
      LiteRtModelT& litert_model, TflFlatbuffer&& tfl_flatbuffer);

  friend const TflFlatbuffer& litert::internal::GetTflFlatbuffer(
      const LiteRtModelT& litert_model);

  explicit LiteRtModelT(TflFlatbuffer&& tfl_flatbuffer)
      : tfl_flatbuffer_(std::move(tfl_flatbuffer)) {}

 private:
  LiteRtSubgraphT::Alloc subgraphs_;
  LiteRtSignatureT::Alloc signatures_;

  MetadataMap metadata_;
  OpAssetMap external_buffer_map_;

  // Use unique ptr here to keep stable. Optionally non-owned.
  StoredBufferManager buffer_manager_ = std::make_unique<BufferManager>();

  // TFLITE
  TflOpCodes tfl_operator_codes_;
  TflFlatbuffer tfl_flatbuffer_;
  std::optional<std::string> source_path_;
};

// Get the build stamp from the model if it exists.
// TODO: Consider a setter and internalizeing all build stamp stuff behind model
// interface.
std::optional<LiteRtModelT::BuildStamp> GetBuildStamp(
    const LiteRtModelT& model);

// Returns true if this model contains any ops compiled for NPU.
bool IsCompiled(const LiteRtModelT& model);

// Get the custom op code from a given op if it is a custom op.
std::optional<std::string> GetCustomOpCode(const LiteRtModelT& model,
                                           const LiteRtOpT& op);

// Lookup subgraph by signature name.
::litert::Expected<LiteRtSubgraph> LookupSubgraph(
    const LiteRtModelT& model, absl::string_view signature_key);

namespace litert::internal {

template <class Arg>
void SetTflOptions(LiteRtOpT& litert_op, Arg&& arg) {
  litert_op.tfl_option_ = std::forward<Arg>(arg);
}

template <class Arg>
void SetTflOptions2(LiteRtOpT& litert_op, Arg&& arg) {
  litert_op.tfl_option_2_ = std::forward<Arg>(arg);
}

inline void ClearTflOptions(LiteRtOpT& litert_op) {
  litert_op.tfl_option_2_.Reset();
  litert_op.tfl_option_.Reset();
}

template <class Arg>
void SetTflOpCodes(LiteRtModelT& litert_model, Arg&& arg) {
  litert_model.tfl_operator_codes_ = std::forward<Arg>(arg);
}

// Model graph stuff
// using IrMapping = absl::flat_hash_map<LiteRtTensor, LiteRtTensor>;

// CLONING

// Clones the basic data between tensors (like name and data) but not
// things related to incoming/outgoing edges (users, defining op) or weights.
void CloneTo(const LiteRtTensorT& src, LiteRtTensorT& dest);

// Clones the basic data between ops (like op code and options) but
// things related to incoming/outgoing edges (input/output tensors).
void CloneTo(const LiteRtOpT& src, LiteRtOpT& dest);

// Same as clone to, but allocates a the dest tensor into given subgraph.
LiteRtTensorT& MakeClone(LiteRtSubgraphT& parent, const LiteRtTensorT& src);

// Same as clone to, but allocates a the dest op into given subgraph.
LiteRtOpT& MakeClone(LiteRtSubgraphT& parent, const LiteRtOpT& src);

// OBSERVERS

// Checks if tensor is input to given op, return its index if so.
std::optional<LiteRtParamIndex> FindInput(const LiteRtOpT& op,
                                          const LiteRtTensorT& tensor);

// Checks if tensor is output to given op, return its index if so.
std::optional<LiteRtParamIndex> FindOutput(const LiteRtOpT& op,
                                           const LiteRtTensorT& tensor);

// Checks if tensor is input to given subgraph, return its index if so.
std::optional<LiteRtParamIndex> FindInput(const LiteRtSubgraphT& subgraph,
                                          const LiteRtTensorT& tensor);

// Checks if tensor is output to given subgraph, return its index if so.
std::optional<LiteRtParamIndex> FindOutput(const LiteRtSubgraphT& subgraph,
                                           const LiteRtTensorT& tensor);

// Check if tensor is part of subgraph IO.
bool IsIO(const LiteRtSubgraphT& subgraph, const LiteRtTensorT& tensor);

using UseIndices =
    absl::InlinedVector<LiteRtParamIndex, kExpectedMaxNumOfTensorUses>;

// Checks if tensor is used by op, return the use inds for each use of tensor by
// op (there may be multiple). These are the indexes to call
// LiteRtTensorT::GetUse with.
UseIndices FindUseInds(const LiteRtTensorT& tensor, const LiteRtOpT& op);

// Is this tensor a constant tensor?
bool IsConstant(const LiteRtTensorT& tensor);

// Is this tensor a subgraph input tensor?
bool IsSubgraphInput(const LiteRtTensorT& tensor);

// MUTATORS

// Attaches the pre-allocated tensor to be an input of given op.
void AttachInput(LiteRtTensor tensor, LiteRtOpT& op);

// Attaches the pre-allocated tensor to be an output of given op.
void AttachOutput(LiteRtTensor tensor, LiteRtOpT& op);

// Remove the input edge from an op. Return the disconnected tensor.
LiteRtTensor DisconnectInput(LiteRtOpT& op, LiteRtParamIndex input_ind);

// Remove an output edge from an op. Return the disconnected tensor.
LiteRtTensor DisconnectOutput(LiteRtOpT& op, LiteRtParamIndex output_ind);

// Remove all incoming and outgoing edges from this op. This can prep nodes
// for removal in DCE.
void Drop(LiteRtOpT& litert_op);

// Run very naive dead code elimination. Removes only ops/tensors that have no
// in/out edges. Ops are handled first. Ignores subgraph IO. Not recursive and
// does only one pass. Returns if the graph was modified.
// NOTE: This de-allocates removed objects, only use when references to these
// objects will not be used.
// TODO: Update this with complete work-list based approach.
bool DCE(LiteRtSubgraphT& subgraph);

}  // namespace litert::internal

//
// Misc Ir Containers
//

using LiteRtOpWithPartitionIndex = std::pair<LiteRtOp, LiteRtParamIndex>;

// Used for communicating selections of ops in when partitioning.
class LiteRtOpListT {
 public:
  void Push(LiteRtOp op, LiteRtParamIndex partition_index = 0) {
    values_.push_back(LiteRtOpWithPartitionIndex(op, partition_index));
  }

  std::vector<LiteRtOpWithPartitionIndex> Values() const {
    std::vector<LiteRtOpWithPartitionIndex> ops;
    ops.reserve(values_.size());
    ops.assign(values_.begin(), values_.end());

    return ops;
  }

 private:
  // Investigate if this is possible with vector (hit some issues).
  std::list<LiteRtOpWithPartitionIndex> values_;
};

//
// Traversal Utils
//

namespace litert::internal {

// Does graph consist of only disptach ops.
bool IsFullyCompiled(const LiteRtModelT& graph);

// Does graph consist of any ops compiled for NPU.
bool HasAnyCompiled(const LiteRtModelT& graph);

}  // namespace litert::internal

// Apply func to all the IR in the given model. Iteration behavior is determined
// by the callback signature.
template <class F>
void ForEachIr(LiteRtModel model, F func) {
  // Per subgraph callbacks.
  using SgF1 = std::function<void(LiteRtSubgraph)>;
  using SgF2 = std::function<void(LiteRtSubgraph, int32_t subgraph_ind)>;

  // Per op callbacks.
  using OpF1 = std::function<void(LiteRtOp)>;
  using OpF2 = std::function<void(LiteRtSubgraph, LiteRtOp)>;
  using OpF3 =
      std::function<void(LiteRtSubgraph, int32_t subgraph_ind, LiteRtOp)>;

  constexpr bool kIsSgOpF1 = std::is_convertible_v<F, SgF1>;
  constexpr bool kIsSgF2 = std::is_convertible_v<F, SgF2>;
  constexpr bool kIsOpF1 = std::is_convertible_v<F, OpF1>;
  constexpr bool kIsOpF2 = std::is_convertible_v<F, OpF2>;
  constexpr bool kIsOpF3 = std::is_convertible_v<F, OpF3>;

  for (int i = 0; i < model->NumSubgraphs(); ++i) {
    auto subgraph = model->Subgraphs()[i];

    if constexpr (kIsSgF2) {
      func(subgraph, i);
    } else if constexpr (kIsSgOpF1) {
      func(subgraph);
    } else {
      for (int j = 0; j < subgraph->Ops().size(); ++j) {
        auto* op = subgraph->Ops()[j];
        if constexpr (kIsOpF1) {
          func(op);
        } else if constexpr (kIsOpF2) {
          func(subgraph, op);
        } else if constexpr (kIsOpF3) {
          func(subgraph, i, op);
        }
      }
    }
  }
}
template <class F>
void ForEachIr(const LiteRtModelT& model, F func) {
  return ForEachIr(const_cast<LiteRtModel>(&model), func);
}

//
// Printing
//

// TODO(@lukeboyer): Migrate dump.h to use absl printing.

// TENSOR PRINTING

template <class Sink>
void AbslStringify(Sink& sink, const TensorType& type) {
  const auto& [id, detail] = type;
  if (id == kLiteRtRankedTensorType) {
    absl::Format(&sink, "%v", detail.ranked_tensor_type);
  } else {
    absl::Format(&sink, "%s", ::litert::kNoPrinterTag);
  }
}

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtTensorT& tensor) {
  auto weights = tensor.Weights().Buffer();
  std::string weights_str = "";
  if (weights.Size() > 0) {
    weights_str = absl::StrFormat("_cst[%s]",
                                  ::litert::HumanReadableSize(weights.Size()));
  }
  absl::Format(&sink, "%v%s", tensor.Type(), weights_str);
}

template <class Sink>
void AbslStringify(Sink& sink, const std::vector<LiteRtTensor>& tensors) {
  sink.Append("(");
  for (auto it = tensors.begin(); it < tensors.end() - 1; ++it) {
    sink.Append(absl::StrFormat("%v", **it));
    sink.Append(",");
  }
  sink.Append(absl::StrFormat("%v", *tensors.back()));
  sink.Append(")");
}

// OP PRINTING

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtOpT& op) {
  static constexpr auto kFmt = "%v%v%v->%v";
  const auto& opts = ::litert::internal::GetTflOptions(op);
  if (opts.type != ::tflite::BuiltinOptions_NONE) {
    absl::Format(&sink, kFmt, op.OpCode(), opts, op.Inputs(), op.Outputs());
    return;
  }
  absl::Format(&sink, kFmt, op.OpCode(), ::litert::internal::GetTflOptions2(op),
               op.Inputs(), op.Outputs());
}

template <class Sink>
void AbslStringify(Sink& sink, const LiteRtOpT* op) {
  absl::Format(&sink, "null");
}

namespace absl {

template <class Sink>
void StringifyLiteRtOpImpl(Sink& sink, const absl::Span<const LiteRtOp>& ops) {
  for (auto it = ops.begin(); it < ops.end() - 1; ++it) {
    sink.Append(absl::StrFormat("%v", **it));
    sink.Append("/");
  }
  sink.Append(absl::StrFormat("%v", *ops.back()));
}

template <class Sink>
void AbslStringify(Sink& sink, const absl::Span<const LiteRtOp>& ops) {
  StringifyLiteRtOpImpl(sink, ops);
}

template <class Sink>
void AbslStringify(Sink& sink, const absl::Span<LiteRtOp>& ops) {
  StringifyLiteRtOpImpl(sink, ops);
}

}  // namespace absl

// OPTIONS PRINTING

namespace litert::internal {

// Since options have a common structure, we provide this builder to format
// sets of options in a consistent manner.
template <typename Sink>
struct OptionStrBuilder {
  OptionStrBuilder(Sink& sink) : sink_(sink) { sink_.Append("{"); }

  template <typename Val>
  void operator()(const std::string& key, const Val& value) {
    if (num_opts_++ > 0) {
      sink_.Append(",");
    }
    if constexpr (std::is_convertible_v<Val, absl::string_view>) {
      absl::Format(&sink_, "%s=%s", key, value);
    } else {
      absl::Format(&sink_, "%s=%v", key, value);
    }
  }

  ~OptionStrBuilder() { sink_.Append("}"); }

 private:
  size_t num_opts_ = 0;
  Sink& sink_;
};
template <typename Sink>
OptionStrBuilder(Sink& sink) -> OptionStrBuilder<Sink>;

template <typename Sink, typename Options>
void PrintNullableOpts(Sink& sink, const Options* opts) {
  if (!opts) {
    absl::Format(&sink, "{null}");
    return;
  }
  absl::Format(&sink, "%v", *opts);
}

}  // namespace litert::internal

namespace tflite {

template <class Sink>
void AbslStringify(Sink& sink, const ::litert::internal::TflOptions& opts) {
  // NOTE: Printers for specific options will be added on an as needed basis.
  const auto type = opts.type;
  switch (type) {
    case tflite::BuiltinOptions_AddOptions: {
      const auto* add_opts = opts.AsAddOptions();
      absl::Format(&sink, "%v", add_opts);
      break;
    }
    default:
      absl::Format(&sink, "{%s}", ::litert::kNoPrinterTag);
      break;
  }
}

template <class Sink>
void AbslStringify(Sink& sink, const ::litert::internal::TflOptions2& opts) {
  // NOTE: Printers for specific options will be added on an as needed basis.
  const auto type = opts.type;
  switch (type) {
    default:
      absl::Format(&sink, "{%s}", ::litert::kNoPrinterTag);
      break;
  }
}

// AddOptionsT

template <typename Sink>
void AbslStringify(Sink& sink, const ActivationFunctionType& type) {
  switch (type) {
    case ActivationFunctionType_NONE:
      sink.Append("NONE");
      break;
    case ActivationFunctionType_RELU6:
      sink.Append("RELU6");
      break;
    case ActivationFunctionType_RELU:
      sink.Append("RELU");
      break;
    case ActivationFunctionType_RELU_N1_TO_1:
      sink.Append("RELU_N1_TO_1");
      break;
    case ActivationFunctionType_TANH:
      sink.Append("TANH");
      break;
    case ActivationFunctionType_SIGN_BIT:
      sink.Append("SIGN_BIT");
      break;
    default:
      sink.Append(::litert::kNoPrinterTag);
      break;
  }
}

template <class Sink>
void AbslStringify(Sink& sink, const AddOptionsT& opts) {
  ::litert::internal::OptionStrBuilder b(sink);
  const auto faf = opts.fused_activation_function;
  b("fa", faf);
  const auto pot = opts.pot_scale_int16;
  if (pot) {
    b("pot", pot);
  }
}

template <class Sink>
void AbslStringify(Sink& sink, const AddOptionsT* opts) {
  ::litert::internal::PrintNullableOpts(sink, opts);
}

}  // namespace tflite

#endif  // ODML_LITERT_LITERT_CORE_MODEL_MODEL_H_
