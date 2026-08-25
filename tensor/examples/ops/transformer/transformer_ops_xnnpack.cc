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

#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/backends/xnnpack/utils.h"  // IWYU pragma: keep
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/ops/transformer/transformer_ops_graph.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::graph {

namespace {

absl::Status ValidateFp32Tensor(const Tensor& tensor,
                                absl::string_view op_name) {
  LRT_TENSOR_RETURN_IF_ERROR(graph::GetStatus(tensor))
      << op_name << " missing tensor information.";
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
  if (info.type != Type::kFP32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s only supports FP32 tensors. Got type id %d.",
                        op_name, static_cast<int>(info.type)));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status OpMixin<FillAttentionMaskOperation, XnnpackMixinTag>::ToXnnpack(
    const graph::Operation& op, XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "FillAttentionMask";

  // This op is materialized as a constant FP32 tensor holding the causal (and
  // optional sliding-window) attention mask. Since XNNPACK doesn't provide a
  // "fill" operator, we populate the output buffer on the host at build time.
  //
  // This is appropriate for our current Gemma3 flow where we rebuild the graph
  // for each sequence length.

  LRT_TENSOR_ASSIGN_OR_RETURN(const FillAttentionMaskOperation& data,
                              op.As<FillAttentionMaskOperation>());

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  if (outputs.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 1 output, got %d", op_name, outputs.size()));
  }
  Tensor output = outputs.front();
  LRT_TENSOR_RETURN_IF_ERROR(ValidateFp32Tensor(output, op_name));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto& output_info, graph::GetInfo(output));
  if (output_info.shape.size() < 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s output shape must have at least 2 dims, got %d",
                        op_name, output_info.shape.size()));
  }

  const int32_t seq_q = output_info.shape[output_info.shape.size() - 2];
  const int32_t seq_k = output_info.shape[output_info.shape.size() - 1];
  if (seq_q <= 0 || seq_k <= 0 || seq_q != seq_k) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects output shape [..., S, S] with S>0; got [%s]", op_name,
        absl::StrJoin(output_info.shape, ", ")));
  }

  const int seq_len = static_cast<int>(seq_q);

  int64_t leading = 1;
  for (size_t i = 0; i + 2 < output_info.shape.size(); ++i) {
    const int32_t dim = output_info.shape[i];
    if (dim <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s does not support non-positive leading dims. Got shape [%s]",
          op_name, absl::StrJoin(output_info.shape, ", ")));
    }
    leading *= static_cast<int64_t>(dim);
  }

  const int64_t elems_per_matrix =
      static_cast<int64_t>(seq_len) * static_cast<int64_t>(seq_len);
  const int64_t total_elems = leading * elems_per_matrix;
  if (total_elems <= 0 || total_elems > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s output too large (%lld elements) for current implementation",
        op_name, static_cast<int64_t>(total_elems)));
  }

  const float neg_inf = std::numeric_limits<float>::lowest();
  std::vector<float> values(static_cast<size_t>(total_elems));

  const bool is_local = data.is_local;
  const int sliding_window_size = data.sliding_window_size;
  for (int64_t b = 0; b < leading; ++b) {
    const int64_t base = b * elems_per_matrix;
    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        const bool is_causal_masked = (j > i);
        const bool is_sliding_masked = is_local && (sliding_window_size > 0) &&
                                       (i - j >= sliding_window_size);
        values[static_cast<size_t>(base + i * seq_len + j)] =
            (is_causal_masked || is_sliding_masked) ? neg_inf : 0.0f;
      }
    }
  }

  output_info.buffer = OwningCpuBuffer::Copy<Type::kFP32>(values);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto _, ctx.DefineValue(output));
  return absl::OkStatus();
}

absl::Status OpMixin<FillRopeCosSinOperation, XnnpackMixinTag>::ToXnnpack(
    const graph::Operation& op, XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "FillRopeCosSin";

  LRT_TENSOR_ASSIGN_OR_RETURN(const FillRopeCosSinOperation& data,
                              op.As<FillRopeCosSinOperation>());

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  if (outputs.size() != 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 2 outputs, got %d", op_name, outputs.size()));
  }

  Tensor cos_tensor = outputs[0];
  Tensor sin_tensor = outputs[1];
  LRT_TENSOR_RETURN_IF_ERROR(ValidateFp32Tensor(cos_tensor, op_name));
  LRT_TENSOR_RETURN_IF_ERROR(ValidateFp32Tensor(sin_tensor, op_name));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto& cos_info, graph::GetInfo(cos_tensor));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& sin_info, graph::GetInfo(sin_tensor));
  if (cos_info.shape.size() != 4 || sin_info.shape.size() != 4 ||
      cos_info.shape != sin_info.shape) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects matching cos/sin shape [1,1,seq,dim]; got "
                        "cos=[%s] sin=[%s]",
                        op_name, absl::StrJoin(cos_info.shape, ", "),
                        absl::StrJoin(sin_info.shape, ", ")));
  }

  const int32_t seq_len = cos_info.shape[2];
  const int32_t head_dim = cos_info.shape[3];
  if (seq_len <= 0 || head_dim <= 0 || (head_dim % 2) != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s invalid shape [1,1,%d,%d]; requires seq>0, dim>0, dim even",
        op_name, seq_len, head_dim));
  }

  const float rope_base = data.rope_base;
  std::vector<float> cos_values(static_cast<size_t>(seq_len) *
                                static_cast<size_t>(head_dim));
  std::vector<float> sin_values(static_cast<size_t>(seq_len) *
                                static_cast<size_t>(head_dim));

  const int half_dim = head_dim / 2;
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int i = 0; i < half_dim; ++i) {
      const float freq = 1.0f / std::pow(rope_base, 2.0f * i / head_dim);
      const float angle = static_cast<float>(pos) * freq;
      const float cos_val = std::cos(angle);
      const float sin_val = std::sin(angle);
      // Duplicate both halves to match ApplyRotaryEmbedding implementation.
      cos_values[static_cast<size_t>(pos) * head_dim + i] = cos_val;
      cos_values[static_cast<size_t>(pos) * head_dim + half_dim + i] = cos_val;
      sin_values[static_cast<size_t>(pos) * head_dim + i] = sin_val;
      sin_values[static_cast<size_t>(pos) * head_dim + half_dim + i] = sin_val;
    }
  }

  cos_info.buffer = OwningCpuBuffer::Copy<Type::kFP32>(cos_values);
  sin_info.buffer = OwningCpuBuffer::Copy<Type::kFP32>(sin_values);

  LRT_TENSOR_RETURN_IF_ERROR(ctx.DefineValue(cos_tensor).status());
  LRT_TENSOR_RETURN_IF_ERROR(ctx.DefineValue(sin_tensor).status());
  return absl::OkStatus();
}

absl::Status OpMixin<RmsNormOperation, XnnpackMixinTag>::ToXnnpack(
    const graph::Operation& op, XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "RmsNorm";
  if (op.inputs.size() < 2 || op.inputs.size() > 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 2 or 3 inputs (input, scale[, epsilon])", op_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  if (outputs.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 1 output, got %d", op_name, outputs.size()));
  }
  const graph::Tensor& original_output = outputs.front();

  // Define original output first to ensure it has an ID.
  LRT_TENSOR_RETURN_IF_ERROR(ctx.DefineValue(original_output).status());

  using XnnTensor = litert::tensor::Tensor<XnnpackMixinTag>;

  XnnTensor input(XnnTensor(op.inputs[0]).ShallowClone());

  if (input.GetShape().empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s input tensor must have at least 1 dimension.", op_name));
  }

  XnnTensor scale = TensorHandle::Invalid();
  bool has_scale = op.inputs[1].group != nullptr;
  if (has_scale) {
    LRT_TENSOR_RETURN_IF_ERROR(graph::GetStatus(op.inputs[1]));
    scale = XnnTensor(op.inputs[1]).ShallowClone();
  }

  XnnTensor eps;
  bool has_epsilon = op.inputs.size() > 2 && op.inputs[2].group != nullptr;
  if (has_epsilon) {
    LRT_TENSOR_RETURN_IF_ERROR(graph::GetStatus(op.inputs[2]));
    eps = XnnTensor(op.inputs[2]).ShallowClone();
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(const RmsNormOperation& data,
                                op.As<RmsNormOperation>());
    eps = XnnTensor(
        {.name = "epsilon",
         .type = litert::tensor::Type::kFP32,
         .shape = {},
         .buffer =
             litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kFP32>(
                 {data.epsilon})});
  }

  // Decompose using Tensor API.
  XnnTensor x2 = Square(input);
  int last_axis = input.GetShape().size() - 1;
  XnnTensor mean_x2 = Mean(x2, {last_axis}, /*keep_dims=*/true);
  XnnTensor mean_x2_eps = Add(mean_x2, eps);
  XnnTensor rsqrt_mean_x2_eps = Rsqrt(mean_x2_eps);
  XnnTensor norm_input = Mul(input, rsqrt_mean_x2_eps);

  XnnTensor composite_output;
  if (has_scale) {
    composite_output = Mul(std::move(norm_input), scale);
  } else {
    composite_output = std::move(norm_input);
  }

  return InlineImplementationGraphFor(
      op, {input.GetRaw(), scale.GetRaw(), eps.GetRaw()},
      {composite_output.GetRaw()}, ctx);
}

}  // namespace litert::tensor::graph
