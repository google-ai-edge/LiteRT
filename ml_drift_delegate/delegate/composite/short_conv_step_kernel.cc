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

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
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

}  // namespace

std::unique_ptr<::ml_drift::GPUOperation> CreateFusedShortConvStep(
    const ::ml_drift::TensorDescriptor& in_proj_desc,
    const ::ml_drift::TensorDescriptor& conv_state_desc,
    const ::ml_drift::TensorDescriptor& conv_weight_desc,
    const ::ml_drift::TensorDescriptor* conv_bias_desc,
    const ::ml_drift::TensorDescriptor& dst_desc,
    const ::ml_drift::TensorDescriptor& next_state_desc,
    int num_slices, int hidden_size, int conv_L_cache,
    bool is_gated, bool use_silu) {
  FusedShortConvStepOp custom_op;
  custom_op.args_.AddInt("num_slices", num_slices);
  custom_op.args_.AddInt("hidden_size", hidden_size);
  custom_op.args_.AddInt("conv_L_cache", conv_L_cache);
  custom_op.args_.AddInt("has_bias", conv_bias_desc != nullptr ? 1 : 0);

  custom_op.AddSrcTensor("in_proj", in_proj_desc);
  custom_op.AddSrcTensor("conv_state", conv_state_desc);
  custom_op.AddSrcTensor("conv_weight", conv_weight_desc);
  if (conv_bias_desc != nullptr) {
    custom_op.AddSrcTensor("conv_bias", *conv_bias_desc);
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

  std::string input_read_code;
  if (is_gated) {
    input_read_code = R"(
  float4 b = ucl::Convert<float4>(args.in_proj.Read(0, 0, S));
  float4 c = ucl::Convert<float4>(args.in_proj.Read(0, 0, S + args.num_slices));
  float4 x = ucl::Convert<float4>(args.in_proj.Read(0, 0, S + 2 * args.num_slices));
  float4 p = b * x;
)";
  } else {
    input_read_code = R"(
  float4 p = ucl::Convert<float4>(args.in_proj.Read(0, 0, S));
)";
  }

  auto make_channel_code = [&](int idx, const char* comp) {
    std::string ch = absl::Substitute("ch$0", idx);
    std::string s = absl::Substitute("s$0", idx);
    std::string w = absl::Substitute("w$0", idx);
    std::string dot_expr;
    std::string next_state_val;
    if (conv_L_cache == 4) {
      dot_expr = absl::Substitute(
          "conv_out.$0 = $1.x * $2.x + $1.y * $2.y + $1.z * $2.z + "
          "p.$0 * $2.w;",
          comp, s, w);
      next_state_val = absl::Substitute(
          "ucl::Init<float4>($0.y, $0.z, p.$1, 0.0f)", s, comp);
    } else {
      dot_expr = absl::Substitute(
          "conv_out.$0 = $1.x * $2.x + $1.y * $2.y + p.$0 * $2.z;", comp, s, w);
      next_state_val = absl::Substitute(
          "ucl::Init<float4>($0.y, p.$1, 0.0f, 0.0f)", s, comp);
    }
    return absl::Substitute(
        R"(
  // Channel $0
  int $1 = 4 * S + $0;
  float4 $2 = ucl::Convert<float4>($4);
  float4 $3 = ucl::Convert<float4>($5);
  $6
  $7;
)",
        idx, ch, s, w, absl::Substitute(state_read_expr, ch),
        absl::Substitute(weight_read_expr, ch), dot_expr,
        absl::Substitute(next_state_write_expr, next_state_val, ch));
  };

  std::string channels_code = make_channel_code(0, "x") +
                              make_channel_code(1, "y") +
                              make_channel_code(2, "z") +
                              make_channel_code(3, "w");

  std::string bias_code = "";
  if (conv_bias_desc != nullptr) {
    bias_code = R"(
  args.conv_bias.SetBatchRef(0);
  float4 bias = ucl::Convert<float4>(args.conv_bias.Read(0, 0, S));
  conv_out += bias;
)";
  }

  std::string silu_code = "";
  if (use_silu) {
    silu_code = R"(
  conv_out = conv_out / (ucl::Init<float4>(1.0f) + exp(-conv_out));
)";
  }

  std::string output_code =
      is_gated ? "  float4 y = c * conv_out;\n" : "  float4 y = conv_out;\n";

  std::string op_code = R"(
MAIN_FUNCTION($0) {
  args.in_proj.SetBatchRef(0);
  args.conv_state.SetBatchRef(0);
  args.conv_weight.SetBatchRef(0);
  args.out.SetBatchRef(0);
  args.next_state.SetBatchRef(0);

  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.out.Width() || Y >= args.out.Height() || S >= args.out.Slices()) {
    return;
  }
)" + input_read_code + R"(
  float4 conv_out;
)" + channels_code + bias_code +
                        silu_code + output_code + R"(
  args.out.Write(ucl::Convert<args.out::type>(y), 0, 0, S);
}
)";
  custom_op.code_ = std::move(op_code);
  return std::make_unique<FusedShortConvStepOp>(std::move(custom_op));
}

namespace {

absl::Status BuildShortConvStepGpuGraph(
    const std::vector<uint32_t>& input_ids,
    const std::vector<uint32_t>& output_ids,
    const ShortConvStepAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() < 3 || input_ids.size() > 4) {
    return absl::InvalidArgumentError("ShortConvStep expects 3 or 4 inputs.");
  }
  if (output_ids.size() != 2) {
    return absl::InvalidArgumentError("ShortConvStep expects 2 outputs.");
  }

  ABSL_ASSIGN_OR_RETURN(auto in_proj, model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(auto conv_state,
                        model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(auto conv_weight,
                        model_builder->GetTensor(input_ids[2]));

  bool has_bias = (input_ids.size() == 4);
  std::optional<::ml_drift::TensorHandle> conv_bias;
  const ::ml_drift::TensorDescriptor* conv_bias_desc = nullptr;
  if (has_bias) {
    ABSL_ASSIGN_OR_RETURN(conv_bias, model_builder->GetTensor(input_ids[3]));
    conv_bias_desc = &conv_bias->tensor_desc;
  }

  auto in_proj_shape = in_proj.tensor_desc.GetBHWCShape();
  int hidden_size = attr.is_gated ? (in_proj_shape.c / 3) : in_proj_shape.c;
  int num_slices = (hidden_size + 3) / 4;

  auto dst_desc = in_proj.tensor_desc;
  dst_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, hidden_size));
  auto dst = model_builder->AddTensor(dst_desc);

  auto next_state_desc = conv_state.tensor_desc;
  auto next_state = model_builder->AddTensor(next_state_desc);

  auto op = CreateFusedShortConvStep(
      in_proj.tensor_desc, conv_state.tensor_desc, conv_weight.tensor_desc,
      conv_bias_desc, dst_desc, next_state_desc, num_slices, hidden_size,
      attr.conv_L_cache, attr.is_gated, attr.use_silu);

  std::vector<::ml_drift::TensorHandle> src_tensors = {in_proj, conv_state,
                                                      conv_weight};
  if (has_bias) {
    src_tensors.push_back(*conv_bias);
  }
  std::vector<::ml_drift::TensorHandle> dst_tensors = {dst, next_state};
  model_builder->AddGpuOperation(src_tensors, dst_tensors, std::move(op),
                                 "short_conv_step");

  ABSL_RETURN_IF_ERROR(model_builder->UpdateOutputTensor(dst, output_ids[0]));
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
