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

#include "litert/experimental/custom_ops/bidirectional_lstm/bidirectional_lstm_custom_op.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/experimental/custom_ops/bidirectional_lstm/bidirectional_lstm_impl.h"

namespace litert {
namespace custom_ops {
namespace {

constexpr int kNumInputs = 9;
constexpr int kNumOutputs = 1;

// Operand positions, matching `fused_bidirectional_lstm` in the exporter.
constexpr int kInputX = 0;
constexpr int kInputSeqLengths = 1;
constexpr int kInputMask = 2;
constexpr int kInputWeightIhFwd = 3;
constexpr int kInputWeightHhFwd = 4;
constexpr int kInputBiasFwd = 5;
constexpr int kInputWeightIhBwd = 6;
constexpr int kInputWeightHhBwd = 7;
constexpr int kInputBiasBwd = 8;

// The only gate packing the kernel implements. PyTorch's nn.LSTM order.
constexpr char kSupportedGateOrder[] = "ifgo";

}  // namespace

const std::string& BidirectionalLstmCustomOpKernel::OpName() const {
  return kOpName;
}

int BidirectionalLstmCustomOpKernel::OpVersion() const { return 1; }

Expected<void> BidirectionalLstmCustomOpKernel::Destroy() { return {}; }

Expected<void> BidirectionalLstmCustomOpKernel::Init(const void* init_data,
                                                     size_t init_data_size) {
  // Nothing is retained: `Init` runs once per node against a shared kernel
  // object, so this is purely a validation hook. `hidden_size` is ignored
  // because the operand shapes already carry it and cannot go stale.
  if (init_data == nullptr || init_data_size == 0) {
    return {};
  }
  const flexbuffers::Map attrs =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(init_data),
                           init_data_size)
          .AsMap();
  if (attrs["gate_order"].IsNull()) {
    return {};
  }
  const std::string gate_order = attrs["gate_order"].AsString().str();
  if (gate_order != kSupportedGateOrder) {
    return Unexpected(
        Status::kErrorInvalidArgument,
        absl::StrCat("bidirectional_lstm: unsupported gate_order '", gate_order,
                     "', only '", kSupportedGateOrder, "' is implemented"));
  }
  return {};
}

Expected<void> BidirectionalLstmCustomOpKernel::GetOutputLayouts(
    const std::vector<Layout>& input_layouts,
    std::vector<Layout>& output_layouts) {
  if (input_layouts.size() != kNumInputs ||
      output_layouts.size() != kNumOutputs) {
    return Unexpected(
        Status::kErrorInvalidArgument,
        absl::StrCat("bidirectional_lstm: expected ", kNumInputs,
                     " inputs and ", kNumOutputs, " output, got ",
                     input_layouts.size(), " and ", output_layouts.size()));
  }
  const auto x_dims = input_layouts[kInputX].Dimensions();
  const auto w_hh_dims = input_layouts[kInputWeightHhFwd].Dimensions();
  if (x_dims.size() != 3 || w_hh_dims.size() != 2) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "bidirectional_lstm: x must be rank 3 and w_hh rank 2");
  }
  // Shapes are authoritative. The `hidden_size` composite attribute is
  // deliberately not consulted: it would have to be remembered from `Init`,
  // which a shared kernel object cannot do per node.
  const int32_t hidden_size = w_hh_dims[1];
  if (w_hh_dims[0] != 4 * hidden_size) {
    return Unexpected(
        Status::kErrorInvalidArgument,
        absl::StrCat("bidirectional_lstm: w_hh must be [4H, H], got [",
                     w_hh_dims[0], ", ", hidden_size, "]"));
  }
  output_layouts[0] = Layout(
      litert::Dimensions{x_dims[0], x_dims[1], 2 * hidden_size});
  return {};
}

Expected<void> BidirectionalLstmCustomOpKernel::Run(
    const std::vector<TensorBuffer>& inputs,
    std::vector<TensorBuffer>& outputs) {
  if (inputs.size() != kNumInputs || outputs.size() != kNumOutputs) {
    return Unexpected(
        Status::kErrorInvalidArgument,
        absl::StrCat("bidirectional_lstm: expected ", kNumInputs,
                     " inputs and ", kNumOutputs, " output, got ",
                     inputs.size(), " and ", outputs.size()));
  }

  LITERT_ASSIGN_OR_RETURN(auto x_type, inputs[kInputX].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto w_hh_type,
                          inputs[kInputWeightHhFwd].TensorType());
  const auto x_dims = x_type.Layout().Dimensions();
  const auto w_hh_dims = w_hh_type.Layout().Dimensions();
  if (x_dims.size() != 3 || w_hh_dims.size() != 2) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "bidirectional_lstm: x must be rank 3 and w_hh rank 2");
  }
  const int batch = x_dims[0];
  const int t_steps = x_dims[1];
  const int input_size = x_dims[2];
  const int hidden_size = w_hh_dims[1];

  LITERT_ASSIGN_OR_RETURN(auto x_lock,
                          TensorBufferScopedLock::Create<const float>(
                              inputs[kInputX], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto seq_lengths_lock,
      TensorBufferScopedLock::Create<const int32_t>(
          inputs[kInputSeqLengths], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto mask_lock, TensorBufferScopedLock::Create<const float>(
                          inputs[kInputMask], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto w_ih_fwd_lock,
      TensorBufferScopedLock::Create<const float>(
          inputs[kInputWeightIhFwd], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto w_hh_fwd_lock,
      TensorBufferScopedLock::Create<const float>(
          inputs[kInputWeightHhFwd], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto b_fwd_lock,
      TensorBufferScopedLock::Create<const float>(
          inputs[kInputBiasFwd], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto w_ih_bwd_lock,
      TensorBufferScopedLock::Create<const float>(
          inputs[kInputWeightIhBwd], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto w_hh_bwd_lock,
      TensorBufferScopedLock::Create<const float>(
          inputs[kInputWeightHhBwd], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto b_bwd_lock,
      TensorBufferScopedLock::Create<const float>(
          inputs[kInputBiasBwd], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto out_lock,
                          TensorBufferScopedLock::Create<float>(
                              outputs[0], TensorBuffer::LockMode::kWrite));

  // Local, not a member: the kernel object is shared by every node and every
  // interpreter, so a member here would be a data race. One allocation of
  // T * (D + 2H) floats per call is negligible against the GEMMs below.
  std::vector<float> scratch(static_cast<size_t>(t_steps) *
                             (input_size + 2 * hidden_size));

  ComputeBidirectionalLstm(
      x_lock.second, seq_lengths_lock.second, mask_lock.second,
      w_ih_fwd_lock.second, w_hh_fwd_lock.second, b_fwd_lock.second,
      w_ih_bwd_lock.second, w_hh_bwd_lock.second, b_bwd_lock.second,
      out_lock.second, scratch.data(), batch, t_steps, input_size,
      hidden_size);
  return {};
}

Expected<void> RegisterBidirectionalLstmCustomOp(Options& options) {
  // LiteRtOptions keeps a raw pointer to the kernel and the CompiledModel
  // outlives this call, so the kernel must have static storage duration.
  static auto* const kernel = new BidirectionalLstmCustomOpKernel();
  LITERT_RETURN_IF_ERROR(options.AddCustomOpKernel(*kernel));
  return {};
}

}  // namespace custom_ops
}  // namespace litert
