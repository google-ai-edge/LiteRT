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

#include "litert/core/model/model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/build_stamp.h"
#include "litert/core/util/flatbuffer_tools.h"

using ::litert::internal::AttachInput;
using ::litert::internal::AttachOutput;
using ::litert::internal::DCE;
using ::litert::internal::Drop;
using ::litert::internal::TflBuffer;
using ::litert::internal::TflBufferPtr;
using ::litert::internal::TflOpCode;
using ::litert::internal::TflOpCodePtr;
using ::litert::internal::TflOptions;
using ::litert::internal::TflOptions2;

std::optional<LiteRtModelT::BuildStamp> GetBuildStamp(
    const LiteRtModelT& model) {
  using ::litert::internal::kLiteRtBuildStampKey;
  using ::litert::internal::ParseBuildStamp;

  auto stamp_meta = model.FindMetadata(kLiteRtBuildStampKey);
  if (!stamp_meta) {
    return std::nullopt;
  }
  auto parsed_stamp = ParseBuildStamp(*stamp_meta);
  if (!parsed_stamp) {
    return std::nullopt;
  }
  auto [soc_manufacturer, soc_model] = *parsed_stamp;
  return LiteRtModelT::BuildStamp{soc_manufacturer, soc_model};
}

bool IsCompiled(const LiteRtModelT& model) {
  return GetBuildStamp(model).has_value();
}

std::optional<std::string> GetCustomOpCode(const LiteRtModelT& model,
                                           const LiteRtOpT& op) {
  if (op.OpCode() != kLiteRtOpCodeTflCustom) {
    return {};
  }
  const auto& tfl_op_codes = litert::internal::GetTflOpCodes(model);
  const auto tfl_op_code_ind = litert::internal::GetTflOpCodeInd(op);
  return tfl_op_codes[tfl_op_code_ind]->custom_code;
}

TensorType MakeRankedTensorType(LiteRtElementType element_type,
                                absl::Span<const int32_t> dims) {
  TensorType tensor_type;
  tensor_type.first = kLiteRtRankedTensorType;
  auto& ranked = tensor_type.second.ranked_tensor_type;
  ranked.element_type = element_type;
  ABSL_DCHECK_LE(dims.size(), LITERT_TENSOR_MAX_RANK);
  ranked.layout.rank = dims.size();
  std::copy(dims.begin(), dims.end(), ranked.layout.dimensions);
  // Strides not yet supported.
  ranked.layout.has_strides = false;
  return tensor_type;
}

Quantization MakePerTensorQuantization(float scale, int64_t zero_point) {
  Quantization quantization;
  quantization.first = kLiteRtQuantizationPerTensor;
  quantization.second.per_tensor.scale = scale;
  quantization.second.per_tensor.zero_point = zero_point;
  return quantization;
}

LiteRtSignatureT MakeDefaultSignature(LiteRtSubgraph subgraph) {
  std::vector<std::string> input_names;
  std::vector<LiteRtTensor> input_tensors;
  input_names.reserve(subgraph->NumInputs());
  input_tensors.reserve(subgraph->NumInputs());
  for (auto* tensor : subgraph->Inputs()) {
    input_names.push_back(std::string(tensor->Name()));
    input_tensors.push_back(tensor);
  }

  std::vector<std::string> output_names;
  std::vector<LiteRtTensor> output_tensors;
  output_names.reserve(subgraph->NumOutputs());
  output_tensors.reserve(subgraph->NumOutputs());
  for (auto* tensor : subgraph->Outputs()) {
    output_names.push_back(std::string(tensor->Name()));
    output_tensors.push_back(tensor);
  }

  std::string name(LiteRtSignatureT::kDefaultSignatureKey);
  return LiteRtSignatureT(subgraph, std::move(input_names),
                          std::move(input_tensors), std::move(output_names),
                          std::move(output_tensors), std::move(name));
}

::litert::Expected<LiteRtSubgraph> LookupSubgraph(
    const LiteRtModelT& model, absl::string_view signature_key) {
  auto sig = model.FindSignature(signature_key);
  if (!sig) {
    return sig.Error();
  }
  return &sig->get().GetSubgraph();
}

void LiteRtModelT::TransferSubgraphTo(LiteRtSubgraphT::Alloc& dest,
                                      std::vector<size_t> indices) {
  if (indices.empty()) {
    return;
  }
  std::sort(indices.begin(), indices.end());
  std::vector<int> new_inds(subgraphs_.Size(), 0);
  auto num_removed = 0;
  auto i = indices.begin();
  for (size_t j = 0; j < new_inds.size(); ++j) {
    if (i != indices.end() && *i == j) {
      ++num_removed;
      // Keep track of removed sgs just for dcheck.
      new_inds[j] = -1;
      ++i;
      continue;
    }
    new_inds[j] = j - num_removed;
  }

  ForEachIr(
      this, [&](LiteRtSubgraph subgraph, int32_t subgraph_index, LiteRtOp op) {
        if (op->OpCode() != kLiteRtOpCodeShloComposite) {
          return;
        }
        auto opts = litert::internal::TakeTflOptions2(*op);
        auto& decomp_ind =
            opts.AsStableHLOCompositeOptions()->decomposition_subgraph_index;
        // Skip update for decomposition ops that are already removed.
        if (decomp_ind != -1) {
          const auto new_ind = new_inds[decomp_ind];
          decomp_ind = new_ind;
        }
        litert::internal::SetTflOptions2(*op, std::move(opts));
      });
  subgraphs_.TransferTo(dest, std::move(indices));
}

LiteRtModelT LiteRtModelT::Yank(std::vector<size_t> indices) {
  LiteRtSubgraphT::Alloc yanked;
  TransferSubgraphTo(yanked, std::move(indices));

  LiteRtModelT res;
  res.TransferSubgraphsFrom(std::move(yanked));

  // Copy op codes.
  const auto& op_codes = litert::internal::GetTflOpCodes(*this);

  LiteRtModelT::TflOpCodes codes;
  codes.reserve(op_codes.size());
  for (const auto& op_code : op_codes) {
    codes.emplace_back(
        std::make_unique<::litert::internal::TflOpCode>(*op_code));
  }

  litert::internal::SetTflOpCodes(res, std::move(codes));

  res.buffer_manager_ = LiteRtModelT::StoredBufferManager(Buffers());

  return res;
}

LiteRtTensorT& LiteRtRewriterT::BuildTensor(const LiteRtWeightsT& weights,
                                            Quantization quantization,
                                            TensorType tensor_type,
                                            std::optional<std::string> name) {
  LiteRtTensorT& tensor = subgraph_.EmplaceTensor();
  tensor.SetType(tensor_type);
  tensor.SetQarams(quantization);
  tensor.Weights().SetBufferId(weights.GetBufferId());
  tensor.Weights().SetBufferManager(weights.GetBufferManager());
  if (name.has_value()) {
    tensor.SetName(*name);
  }
  allocated_tensors_.insert(&tensor);
  return tensor;
}

LiteRtTensorT& LiteRtRewriterT::BuildTensor(const LiteRtTensorT& src) {
  return BuildTensor(src.Weights(), src.Qparams(), src.Type(),
                     std::string(src.Name()));
}

LiteRtOpT& LiteRtRewriterT::BuildOp(LiteRtOpCode code,
                                    std::vector<LiteRtTensor> inputs,
                                    std::vector<LiteRtTensor> outputs) {
  LiteRtOpT& op = subgraph_.EmplaceOp();
  op.SetOpCode(code);
  // Only use AttachInput/AttachOutput for tensors that are allocated in the
  // rewriter.
  for (const LiteRtTensor& input : inputs) {
    ABSL_DCHECK(input != nullptr) << "Input tensor is null";
    if (IsTensorAllocated(input)) {
      AttachInput(input, op);
    } else {
      op.Inputs().push_back(input);
      subgraph_.Inputs().push_back(input);
    }
  }
  for (const LiteRtTensor& output : outputs) {
    ABSL_DCHECK(output != nullptr) << "Output tensor is null";
    if (IsTensorAllocated(output)) {
      AttachOutput(output, op);
    } else {
      op.Outputs().push_back(output);
      subgraph_.Outputs().push_back(output);
    }
  }
  // LITERT_LOG(LITERT_INFO, "BuildOp: %d", code);
  allocated_ops_.insert(&op);
  return op;
}

LiteRtOpT& LiteRtRewriterT::BuildOp(LiteRtOpT& src,
                                    std::vector<LiteRtTensor> inputs,
                                    std::vector<LiteRtTensor> outputs) {
  return BuildOp(src.OpCode(), inputs, outputs);
}

void LiteRtRewriterT::EraseOp(LiteRtOp opToErase) {
  ABSL_DCHECK(opToErase != nullptr) << "Op to erase is null";
  erases_.insert(opToErase);
}

void LiteRtRewriterT::ApplyChanges(LiteRtSubgraphT* subgraph_to_apply) {
  // Remove all ops that are marked for erases.
  for (LiteRtOp op : erases_) {
    Drop(*op);
  }

  // Clear dead ops and tensors in user defined subgraph.
  DCE(subgraph_);

  // Recover the graph connectivity after transferring, for inputs and
  // outputs of the rewriter subgraph.
  auto const is_a_io_tensor = [](const LiteRtTensor& tensor,
                                 const std::vector<LiteRtTensor>& io_Tensors) {
    return std::any_of(
        io_Tensors.begin(), io_Tensors.end(),
        [&tensor](const LiteRtTensor& input) { return input == tensor; });
  };

  for (auto& op : subgraph_.Ops()) {
    for (auto& input : op->Inputs()) {
      if (is_a_io_tensor(input, subgraph_.Inputs())) {
        input->Users().push_back(op);
        input->UserArgInds().push_back(op->Inputs().size() - 1);
      }
    }
    for (auto& output : op->Outputs()) {
      if (is_a_io_tensor(output, subgraph_.Outputs())) {
        output->SetDefiningOp(*op, op->Outputs().size() - 1);
      }
    }
  }

  // Transfer ownership of tensors and ops to the root subgraph.
  // Note: Maintain the original topological order of the ops.
  size_t splice_index = subgraph_to_apply->Ops().size() - 1;
  LITERT_LOG(LITERT_DEBUG, "splice_index starting: %zu", splice_index);
  for (size_t original_op_index = 0;
       original_op_index < subgraph_to_apply->Ops().size();
       ++original_op_index) {
    for (LiteRtOp op_to_erase : erases_) {
      if (subgraph_to_apply->Ops().at(original_op_index) == op_to_erase) {
        splice_index = std::min(splice_index, original_op_index);
      }
    }
  }
  DCE(*subgraph_to_apply);
  subgraph_to_apply->TransferOpsFrom(subgraph_.OpsAllocation(), splice_index);
  subgraph_to_apply->TransferTensorsFrom(subgraph_.TensorsAllocation());
}

namespace litert::internal {

void SetTflOpCodeInd(LiteRtOpT& litert_op, int32_t tfl_op_code_ind) {
  litert_op.tfl_op_code_ind_ = tfl_op_code_ind;
}

int32_t GetTflOpCodeInd(const LiteRtOpT& litert_op) {
  return litert_op.tfl_op_code_ind_;
}

const TflOptions& GetTflOptions(const LiteRtOpT& litert_op) {
  return litert_op.tfl_option_;
}

const TflOptions2& GetTflOptions2(const LiteRtOpT& litert_op) {
  return litert_op.tfl_option_2_;
}

TflOptions&& TakeTflOptions(LiteRtOpT& litert_op) {
  return std::move(litert_op.tfl_option_);
}

TflOptions2&& TakeTflOptions2(LiteRtOpT& litert_op) {
  return std::move(litert_op.tfl_option_2_);
}

const std::vector<TflOpCodePtr>& GetTflOpCodes(
    const LiteRtModelT& litert_model) {
  return litert_model.tfl_operator_codes_;
}

std::vector<TflOpCodePtr>&& TakeTflOpCodes(LiteRtModelT& litert_model) {
  return std::move(litert_model.tfl_operator_codes_);
}

// new stuff start
void SetTflFlatbuffer(LiteRtModelT& litert_model,
                      LiteRtModelT::TflFlatbuffer&& tfl_flatbuffer) {
  litert_model.tfl_flatbuffer_ = std::move(tfl_flatbuffer);
}

const LiteRtModelT::TflFlatbuffer& GetTflFlatbuffer(
    const LiteRtModelT& litert_model) {
  return litert_model.tfl_flatbuffer_;
}
// new stuff end

namespace {

bool IsOpDead(const LiteRtOpT& op) {
  return op.Inputs().empty() && op.Outputs().empty();
}

bool IsTensorDead(const LiteRtTensorT& tensor) {
  return tensor.DefiningOp() == nullptr && tensor.NumUses() == 0;
}

}  // namespace

void CloneTo(const LiteRtTensorT& src, LiteRtTensorT& dest) {
  dest.SetName({src.Name().cbegin(), src.Name().cend()});
  dest.SetQarams(src.Qparams());
  dest.SetType(src.Type());
  dest.SetTensorIndex(src.TensorIndex());

  // Manully copy per-channel quantization params,quant array is owned by
  // tensor.
  if (src.Qparams().first == kLiteRtQuantizationPerChannel) {
    std::vector<float> scales(
        src.Qparams().second.per_channel.scales,
        src.Qparams().second.per_channel.scales +
            src.Qparams().second.per_channel.num_channels);
    std::vector<int64_t> zero_points(
        src.Qparams().second.per_channel.zero_points,
        src.Qparams().second.per_channel.zero_points +
            src.Qparams().second.per_channel.num_channels);
    Quantization dest_qparams = MakePerChannelQuantization(
        scales, zero_points,
        src.Qparams().second.per_channel.quantized_dimension,
        [&dest](auto s) { return dest.RequestScratchBuffer(s); });
    dest.SetQarams(std::move(dest_qparams));
  }

  // Move weight buffer from src to dest.
  const auto& src_weights = src.Weights();
  auto& dest_weights = dest.Weights();

  const auto same_manager =
      src_weights.GetBufferManager() == dest_weights.GetBufferManager();

  if (same_manager) {
    dest_weights.SetBufferId(src_weights.GetBufferId());
  } else {
    OwningBufferRef<uint8_t> weights_buffer(src_weights.Buffer().Data(),
                                            src_weights.Buffer().Size());
    SetWeightsFromOwnedBuffer(dest_weights, std::move(weights_buffer));
  }
}

void CloneTo(const LiteRtOpT& src, LiteRtOpT& dest) {
  dest.SetCustomOptions(src.CustomOptions().Data(), src.CustomOptions().Size());
  litert::internal::SetTflOptions(dest, litert::internal::GetTflOptions(src));
  litert::internal::SetTflOptions2(dest, litert::internal::GetTflOptions2(src));
  litert::internal::SetTflOpCodeInd(dest,
                                    litert::internal::GetTflOpCodeInd(src));
  dest.SetOpCode(src.OpCode());
}

LiteRtTensorT& MakeClone(LiteRtSubgraphT& parent, const LiteRtTensorT& src) {
  auto& new_tensor = parent.EmplaceTensor();
  CloneTo(src, new_tensor);
  return new_tensor;
}

LiteRtOpT& MakeClone(LiteRtSubgraphT& parent, const LiteRtOpT& src) {
  auto& new_op = parent.EmplaceOp();
  CloneTo(src, new_op);
  return new_op;
}

std::optional<LiteRtParamIndex> FindInput(const LiteRtOpT& op,
                                          const LiteRtTensorT& tensor) {
  return FindInd(op.Inputs().cbegin(), op.Inputs().cend(), &tensor);
}

std::optional<LiteRtParamIndex> FindOutput(const LiteRtOpT& op,
                                           const LiteRtTensorT& tensor) {
  return FindInd(op.Outputs().cbegin(), op.Outputs().cend(), &tensor);
}

std::optional<LiteRtParamIndex> FindInput(const LiteRtSubgraphT& subgraph,
                                          const LiteRtTensorT& tensor) {
  return FindInd(subgraph.Inputs().cbegin(), subgraph.Inputs().cend(), &tensor);
}

std::optional<LiteRtParamIndex> FindOutput(const LiteRtSubgraphT& subgraph,
                                           const LiteRtTensorT& tensor) {
  return FindInd(subgraph.Outputs().cbegin(), subgraph.Outputs().cend(),
                 &tensor);
}

UseIndices FindUseInds(const LiteRtTensorT& tensor, const LiteRtOpT& op) {
  UseIndices res;
  for (auto i = 0; i < tensor.NumUses(); ++i) {
    if (tensor.Users().at(i) == &op) {
      res.push_back(i);
    }
  }
  return res;
}

bool IsConstant(const LiteRtTensorT& tensor) {
  bool is_zero_sized = false;
  auto layout = tensor.Type().second.ranked_tensor_type.layout;
  if (layout.rank == 1) {
    if (layout.dimensions[0] == 0) {
      is_zero_sized = true;
    }
  }
  const auto is_const = tensor.Weights().Buffer().Size() > 0 || is_zero_sized;
  ABSL_DCHECK(!is_const || tensor.DefiningOp() == nullptr)
      << "Constant tensors should not be defined by an op";
  return is_const;
}

void AttachInput(LiteRtTensor tensor, LiteRtOpT& op) {
  op.Inputs().push_back(tensor);
  tensor->Users().push_back(&op);
  tensor->UserArgInds().push_back(op.Inputs().size() - 1);
}

void AttachOutput(LiteRtTensor tensor, LiteRtOpT& op) {
  ABSL_DCHECK(tensor->DefiningOp() == nullptr)
      << "Cannot add an already defined tensor as op output";
  op.Outputs().push_back(tensor);
  tensor->SetDefiningOp(op, op.Outputs().size() - 1);
}

LiteRtTensor DisconnectInput(LiteRtOpT& op, LiteRtParamIndex input_ind) {
  ABSL_DCHECK(input_ind < op.Inputs().size()) << "Removing tensor index oob";
  auto& input = op.Input(input_ind);

  // Find the index of the use for the given in edge.
  auto target_use_ind = -1;
  for (auto i = 0; i < input.NumUses(); ++i) {
    if (input.Users().at(i) == &op && input.UserArgInds().at(i) == input_ind) {
      target_use_ind = i;
    }
  }
  ABSL_DCHECK_GE(target_use_ind, 0) << "Malformed graph";

  // Slide latter input use arg inds to the left.
  for (auto i = input_ind + 1; i < op.Inputs().size(); ++i) {
    auto& r_in = op.Input(i);
    for (auto u = 0; u < r_in.NumUses(); ++u) {
      auto& r_arg_ind = r_in.UserArgInds().at(u);
      if (r_in.Users().at(u) == &op && r_arg_ind > input_ind) {
        r_arg_ind -= 1;
      }
    }
  }

  // Update the edges.
  input.RemoveUse(target_use_ind);
  op.RemoveInput(input_ind);

  return &input;
}

bool IsIO(const LiteRtSubgraphT& subgraph, const LiteRtTensorT& tensor) {
  return FindInput(subgraph, tensor) || FindOutput(subgraph, tensor);
}

namespace {

bool IsCompiledOp(const LiteRtModelT& graph, LiteRtOpT& op) {
  // If there hasn't been a round of serialization,
  // since dispatches were added, they won't be in the code
  // table.
  // TODO: Fix this once the code table is updateded
  // dynamically.
  return litert::internal::GetTflOpCodeInd(op) ==
             litert::internal::kDispatchOpCodeTflInd ||
         GetCustomOpCode(graph, op) ==
             litert::internal::kLiteRtDispatchOpCustomName;
}

}  // namespace

bool IsFullyCompiled(const LiteRtModelT& graph) {
  bool res = true;
  ForEachIr(graph,
            [&res, &graph](LiteRtOp op) { res &= IsCompiledOp(graph, *op); });
  return res;
}

bool HasAnyCompiled(const LiteRtModelT& graph) {
  bool res = false;
  ForEachIr(graph,
            [&res, &graph](LiteRtOp op) { res |= IsCompiledOp(graph, *op); });
  return res;
}

LiteRtTensor DisconnectOutput(LiteRtOpT& op, LiteRtParamIndex output_ind) {
  ABSL_DCHECK(output_ind < op.Outputs().size()) << "Removing tensor index oob";
  auto& output = op.Output(output_ind);
  output.ClearDefiningOp();
  op.RemoveOutput(output_ind);
  return &output;
}

void Drop(LiteRtOpT& litert_op) {
  while (!litert_op.Inputs().empty()) {
    DisconnectInput(litert_op, 0);
  }
  while (!litert_op.Outputs().empty()) {
    DisconnectOutput(litert_op, 0);
  }
}

bool DCE(LiteRtSubgraphT& subgraph) {
  const auto ops_removed = subgraph.RemoveOpIf(IsOpDead);

  auto rm_tensor = [&subgraph = std::as_const(subgraph)](const auto& t) {
    return IsTensorDead(t) && !IsIO(subgraph, t);
  };
  const auto tensors_removed = subgraph.RemoveTensorIf(rm_tensor);
  LITERT_LOG(LITERT_INFO, "DCE removed %d ops, %d tensors", ops_removed,
             tensors_removed);

  return (ops_removed + tensors_removed) > 0;
}

}  // namespace litert::internal
