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

#include "ml_drift_delegate/delegate/composite/short_conv_step_kernel.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor_handle.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/short_conv_step_parser.h"

namespace litert::ml_drift {

namespace {

class FusedShortConvStepOp : public ::ml_drift::GPUOperation {
 public:
  FusedShortConvStepOp() = default;
  // One work item per (token, channel slice) of the output.
  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(dst_[0]->Width(), dst_[0]->Height(),
                            dst_[0]->Slices());
  }

  // Move only
  FusedShortConvStepOp(FusedShortConvStepOp&&) = default;
  FusedShortConvStepOp& operator=(FusedShortConvStepOp&&) = default;
  FusedShortConvStepOp(const FusedShortConvStepOp&) = delete;
  FusedShortConvStepOp& operator=(const FusedShortConvStepOp&) = delete;
};

// Vector component that holds conv tap (or channel lane) `k`.
absl::string_view Component(int k) {
  static constexpr absl::string_view kComponents[] = {"x", "y", "z", "w"};
  return kComponents[k];
}

// Shader expression of B * x at token `token` in the slice `S`.
std::string BxExpr(absl::string_view token) {
  return absl::Substitute(
      "(ucl::Convert<float4>(args.in_proj.Read($0, 0, S)) * "
      "ucl::Convert<float4>(args.in_proj.Read($0, 0, S + 2 * "
      "args.num_slices)))",
      token);
}

// Shader expression of C at token `token` in the slice `S`.
std::string CExpr(absl::string_view token) {
  return absl::Substitute(
      "ucl::Convert<float4>(args.in_proj.Read($0, 0, S + args.num_slices))",
      token);
}

}  // namespace

std::unique_ptr<::ml_drift::GPUOperation> CreateFusedShortConvStep(
    const ::ml_drift::TensorDescriptor& in_proj_desc,
    const ::ml_drift::TensorDescriptor& conv_state_desc,
    const ::ml_drift::TensorDescriptor& conv_weight_desc,
    const ::ml_drift::TensorDescriptor* conv_bias_desc,
    const ::ml_drift::TensorDescriptor& dst_desc,
    const ::ml_drift::TensorDescriptor& next_state_desc, int num_slices,
    int hidden_size, int conv_L_cache,
    const ::ml_drift::TensorDescriptor* num_valid_tokens_desc) {
  const int clamped_L_cache = std::clamp(conv_L_cache, 2, 4);
  const int num_state_taps = clamped_L_cache - 1;

  FusedShortConvStepOp custom_op;
  custom_op.args_.AddInt("num_slices", num_slices);
  custom_op.args_.AddInt("hidden_size", hidden_size);
  custom_op.args_.AddInt("conv_L_cache", clamped_L_cache);
  custom_op.args_.AddInt("has_bias", conv_bias_desc != nullptr ? 1 : 0);

  custom_op.AddSrcTensor("in_proj", in_proj_desc);
  custom_op.AddSrcTensor("conv_state", conv_state_desc);
  custom_op.AddSrcTensor("conv_weight", conv_weight_desc);
  if (conv_bias_desc != nullptr) {
    custom_op.AddSrcTensor("conv_bias", *conv_bias_desc);
  }
  if (num_valid_tokens_desc != nullptr) {
    custom_op.AddSrcTensor("num_valid_tokens", *num_valid_tokens_desc);
  }
  custom_op.AddDstTensor("out", dst_desc);
  custom_op.AddDstTensor("next_state", next_state_desc);

  auto w_shape = conv_weight_desc.GetBHWCShape();
  std::string weight_read_expr;
  if (w_shape.b == hidden_size) {
    weight_read_expr = "args.conv_weight.Read(0, 0, 0, $0)";
  } else if (w_shape.w == hidden_size) {
    weight_read_expr = "args.conv_weight.Read($0, 0, 0)";
  } else {
    weight_read_expr = "args.conv_weight.Read(0, 0, 0, $0)";
  }

  auto state_shape = conv_state_desc.GetBHWCShape();
  std::string state_read_expr;
  std::string next_state_write_expr;
  if (state_shape.w == hidden_size) {
    state_read_expr = "args.conv_state.Read($0, 0, 0)";
    next_state_write_expr =
        "args.next_state.Write(ucl::Convert<args.next_state::type>($0), $1, 0, "
        "0)";
  } else {
    state_read_expr = "args.conv_state.Read(0, 0, 0, $0)";
    next_state_write_expr =
        "args.next_state.Write(ucl::Convert<args.next_state::type>($0), 0, 0, "
        "0, $1)";
  }

  std::string bias_expr = "ucl::Init<float4>(0.0f, 0.0f, 0.0f, 0.0f)";
  if (conv_bias_desc != nullptr) {
    const ::ml_drift::BHWC bias_shape = conv_bias_desc->GetBHWCShape();
    if (bias_shape.b == hidden_size && bias_shape.c != hidden_size) {
      bias_expr = "ucl::Init<float4>(";
      for (int i = 0; i < 4; ++i) {
        absl::StrAppend(&bias_expr, i > 0 ? ", " : "",
                        "ucl::Convert<float4>(args.conv_bias.Read(0, 0, 0, 4 * "
                        "S + ",
                        i, ")).x");
      }
      bias_expr += ")";
    } else {
      bias_expr = "ucl::Convert<float4>(args.conv_bias.Read(0, 0, S))";
    }
  }

  ABSL_LOG(INFO) << "ShortConvStep: hidden_size=" << hidden_size
                 << ", seq_len=" << dst_desc.GetBHWCShape().w
                 << ", num_slices=" << num_slices
                 << ", conv_L_cache=" << clamped_L_cache
                 << ", has_bias=" << (conv_bias_desc != nullptr)
                 << ", has_num_valid_tokens="
                 << (num_valid_tokens_desc != nullptr);

  // With L = conv_L_cache and padded = concat(conv_state, B * x) along the
  // sequence, padded[p] is conv_state tap p for p < L - 1 and B * x of token
  // p - (L - 1) otherwise. Token t convolves the window padded[t : t + L].
  std::string c = "MAIN_FUNCTION($0) {\n";
  c += "  args.in_proj.SetBatchRef(0);\n";
  c += "  args.conv_state.SetBatchRef(0);\n";
  c += "  args.conv_weight.SetBatchRef(0);\n";
  if (conv_bias_desc != nullptr) {
    c += "  args.conv_bias.SetBatchRef(0);\n";
  }
  if (num_valid_tokens_desc != nullptr) {
    c += "  args.num_valid_tokens.SetBatchRef(0);\n";
  }
  c += "  args.out.SetBatchRef(0);\n";
  c += "  args.next_state.SetBatchRef(0);\n";
  c += R"(
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.out.Width() || Y >= args.out.Height() ||
      S >= args.out.Slices()) {
    return;
  }
)";
  // Work item 0 computes all tokens whose window reaches into conv_state
  // (tokens [0, L - 2]) and then writes next_state. Keeping every conv_state
  // read and next_state write in one work item keeps the kernel correct when
  // the runtime binds the same buffer to conv_state and next_state, as
  // LiteRT-LM does for in-place state updates.
  if (num_state_taps > 1) {
    absl::StrAppend(&c, "  if (X > 0 && X < ", num_state_taps,
                    ") {\n    return;\n  }\n");
  }

  // Weight w<i>.<k> is tap k of channel i of the slice, and wt<k> holds tap k
  // of the 4 channels.
  for (int i = 0; i < 4; ++i) {
    absl::StrAppend(
        &c, "  float4 w", i, " = ucl::Convert<float4>(",
        absl::Substitute(weight_read_expr, absl::StrCat("4 * S + ", i)),
        ");\n");
  }
  for (int k = 0; k < clamped_L_cache; ++k) {
    const absl::string_view tap = Component(k);
    absl::StrAppend(&c, "  float4 wt", k, " = ucl::Init<float4>(w0.", tap,
                    ", w1.", tap, ", w2.", tap, ", w3.", tap, ");\n");
  }
  absl::StrAppend(&c, "  float4 bias = ", bias_expr, ";\n");

  // Tokens [L - 1, S): the whole window comes from in_proj.
  absl::StrAppend(&c, "  if (X >= ", num_state_taps, ") {\n");
  c += "    float4 acc = bias;\n";
  for (int k = 0; k < clamped_L_cache; ++k) {
    const int offset = num_state_taps - k;
    const std::string token = offset == 0 ? "X" : absl::StrCat("X - ", offset);
    absl::StrAppend(&c, "    acc += wt", k, " * ", BxExpr(token), ";\n");
  }
  absl::StrAppend(&c, "    args.out.Write(ucl::Convert<args.out::type>(",
                  CExpr("X"), " * acc), X, 0, S);\n");
  c += "    return;\n  }\n";

  // Work item 0. State s<i>.<j> is tap j of channel i of the slice, and st<j>
  // holds tap j of the 4 channels.
  for (int i = 0; i < 4; ++i) {
    absl::StrAppend(
        &c, "  float4 s", i, " = ucl::Convert<float4>(",
        absl::Substitute(state_read_expr, absl::StrCat("4 * S + ", i)), ");\n");
  }
  for (int j = 0; j < num_state_taps; ++j) {
    const absl::string_view tap = Component(j);
    absl::StrAppend(&c, "  float4 st", j, " = ucl::Init<float4>(s0.", tap,
                    ", s1.", tap, ", s2.", tap, ", s3.", tap, ");\n");
  }
  c += "  int seq_len = args.out.Width();\n";
  for (int t = 0; t < num_state_taps; ++t) {
    absl::StrAppend(&c, "  if (", t, " < seq_len) {\n");
    c += "    float4 acc = bias;\n";
    for (int k = 0; k < clamped_L_cache; ++k) {
      const int p = t + k;
      const std::string input = p < num_state_taps
                                    ? absl::StrCat("st", p)
                                    : BxExpr(absl::StrCat(p - num_state_taps));
      absl::StrAppend(&c, "    acc += wt", k, " * ", input, ";\n");
    }
    absl::StrAppend(&c, "    args.out.Write(ucl::Convert<args.out::type>(",
                    CExpr(absl::StrCat(t)), " * acc), ", t, ", 0, S);\n");
    c += "  }\n";
  }

  // next_state tap j = padded[n + j] with n = num_valid_tokens (or seq_len when
  // num_valid_tokens is omitted), clamped to [0, seq_len].
  if (num_valid_tokens_desc != nullptr) {
    c += R"(  int n = ucl::Convert<int4>(args.num_valid_tokens.Read(0, 0, 0)).x;
  if (n < 0) {
    n = 0;
  }
  if (n > seq_len) {
    n = seq_len;
  }
)";
  } else {
    c += "  int n = seq_len;\n";
  }
  for (int j = 0; j < num_state_taps; ++j) {
    absl::StrAppend(&c, "  int p", j, " = n + ", j, ";\n");
    absl::StrAppend(&c, "  float4 ns", j, ";\n");
    // p<j> >= j, so conv_state taps below j are never selected.
    for (int i = j; i < num_state_taps; ++i) {
      absl::StrAppend(&c, i == j ? "  if (p" : "  } else if (p", j, " == ", i,
                      ") {\n    ns", j, " = st", i, ";\n");
    }
    absl::StrAppend(&c, "  } else {\n    ns", j, " = ",
                    BxExpr(absl::StrCat("p", j, " - ", num_state_taps)),
                    ";\n  }\n");
  }
  for (int i = 0; i < 4; ++i) {
    std::string taps;
    for (int j = 0; j < 4; ++j) {
      absl::StrAppend(&taps, j > 0 ? ", " : "",
                      j < num_state_taps
                          ? absl::StrCat("ns", j, ".", Component(i))
                          : std::string("0.0f"));
    }
    absl::StrAppend(
        &c, "  ",
        absl::Substitute(next_state_write_expr,
                         absl::StrCat("ucl::Init<float4>(", taps, ")"),
                         absl::StrCat("4 * S + ", i)),
        ";\n");
  }
  c += "}\n";

  custom_op.code_ = std::move(c);
  return std::make_unique<FusedShortConvStepOp>(std::move(custom_op));
}

namespace {

absl::Status BuildShortConvStepGpuGraph(
    const std::vector<uint32_t>& input_ids,
    const std::vector<uint32_t>& output_ids,
    const ShortConvStepAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() < 3 || input_ids.size() > 5) {
    return absl::InvalidArgumentError("ShortConvStep expects 3 to 5 inputs.");
  }
  if (output_ids.size() != 2) {
    return absl::InvalidArgumentError("ShortConvStep expects 2 outputs.");
  }
  if (attr.conv_L_cache < 2 || attr.conv_L_cache > 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("ShortConvStep supports conv_L_cache in [2, 4], but got ",
                     attr.conv_L_cache));
  }

  ABSL_ASSIGN_OR_RETURN(auto in_proj, model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(auto conv_state,
                        model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(auto conv_weight,
                        model_builder->GetTensor(input_ids[2]));

  std::optional<::ml_drift::TensorHandle> conv_bias;
  const ::ml_drift::TensorDescriptor* conv_bias_desc = nullptr;
  std::optional<::ml_drift::TensorHandle> num_valid_tokens;
  const ::ml_drift::TensorDescriptor* num_valid_tokens_desc = nullptr;
  for (size_t i = 3; i < input_ids.size(); ++i) {
    ABSL_ASSIGN_OR_RETURN(auto extra_input,
                          model_builder->GetTensor(input_ids[i]));
    if (extra_input.tensor_desc.GetDataType() == ::ml_drift::DataType::kInt32) {
      num_valid_tokens = std::move(extra_input);
      num_valid_tokens_desc = &num_valid_tokens->tensor_desc;
    } else {
      conv_bias = std::move(extra_input);
      conv_bias_desc = &conv_bias->tensor_desc;
    }
  }

  auto in_proj_shape = in_proj.tensor_desc.GetBHWCShape();
  int hidden_size = in_proj_shape.c / 3;
  int num_slices = (hidden_size + 3) / 4;

  auto dst_desc = in_proj.tensor_desc;
  dst_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, in_proj_shape.w, hidden_size));
  auto dst = model_builder->AddTensor(dst_desc);

  auto next_state_desc = conv_state.tensor_desc;
  auto next_state = model_builder->AddTensor(next_state_desc);

  auto op = CreateFusedShortConvStep(
      in_proj.tensor_desc, conv_state.tensor_desc, conv_weight.tensor_desc,
      conv_bias_desc, dst_desc, next_state_desc, num_slices, hidden_size,
      attr.conv_L_cache, num_valid_tokens_desc);

  std::vector<::ml_drift::TensorHandle> src_tensors = {in_proj, conv_state,
                                                       conv_weight};
  if (conv_bias.has_value()) {
    src_tensors.push_back(*conv_bias);
  }
  if (num_valid_tokens.has_value()) {
    src_tensors.push_back(*num_valid_tokens);
  }
  std::vector<::ml_drift::TensorHandle> dst_tensors = {dst, next_state};
  model_builder->AddGpuOperation(src_tensors, dst_tensors, std::move(op),
                                 "short_conv_step");

  ABSL_RETURN_IF_ERROR(model_builder->UpdateOutputTensor(dst, output_ids[0]));
  if (model_builder->gpu_info().IsApiWebGpu()) {
    // WebGPU forbids binding the same buffer as both read-only (`conv_state`)
    // and read-write (`next_state`) in a single compute dispatch when the
    // runtime aliases input and output state buffers in-place. Write
    // `next_state` to an intermediate tensor and copy it to the graph output.
    ABSL_ASSIGN_OR_RETURN(auto out_state_ref,
                          model_builder->GetTensor(output_ids[1]));
    auto out_state = model_builder->AddTensor(out_state_ref.tensor_desc);
    model_builder->Copy(next_state, out_state);
    return model_builder->UpdateOutputTensor(out_state, output_ids[1]);
  }
  return model_builder->UpdateOutputTensor(next_state, output_ids[1]);
}

}  // namespace

absl::Status CreateShortConvStepFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const ShortConvStepAttributes& attr =
      std::any_cast<const ShortConvStepAttributes&>(node.operation.attributes);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* output : outputs) output_ids.push_back(output->id);
  return BuildShortConvStepGpuGraph(input_ids, output_ids, attr, model_builder);
}

absl::Status CreateShortConvStepFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const ShortConvStepAttributes& attr =
      std::any_cast<const ShortConvStepAttributes&>(node.attr);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* output : outputs) output_ids.push_back(output->id);
  return BuildShortConvStepGpuGraph(input_ids, output_ids, attr, model_builder);
}

}  // namespace litert::ml_drift
