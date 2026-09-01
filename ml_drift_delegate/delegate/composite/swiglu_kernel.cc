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

#include "ml_drift_delegate/delegate/composite/swiglu_kernel.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/swiglu_parser.h"

namespace litert::ml_drift {

namespace {

class FusedSwiGLUOp : public ::ml_drift::GPUOperation {
 public:
  FusedSwiGLUOp() = default;
  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(dst_[0]->Width(), dst_[0]->Height(),
                            dst_[0]->Slices());
  }

  // Move only
  FusedSwiGLUOp(FusedSwiGLUOp&&) = default;
  FusedSwiGLUOp& operator=(FusedSwiGLUOp&&) = default;
  FusedSwiGLUOp(const FusedSwiGLUOp&) = delete;
  FusedSwiGLUOp& operator=(const FusedSwiGLUOp&) = delete;
};

std::unique_ptr<::ml_drift::GPUOperation> CreateFusedSwiGLU(
    const ::ml_drift::TensorDescriptor& src_desc,
    const ::ml_drift::TensorDescriptor& dst_desc, int gate_slices) {
  FusedSwiGLUOp custom_op;
  custom_op.args_.AddInt("gate_slices", gate_slices);
  custom_op.AddSrcTensor("src_tensor", src_desc);
  custom_op.AddDstTensor("dst_tensor", dst_desc);

  std::string op_code = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || S >= args.dst_tensor.Slices()) {
    return;
  }
  float4 gate = ucl::Convert<float4>(args.src_tensor.Read(X, Y, S));
  float4 up = ucl::Convert<float4>(args.src_tensor.Read(X, Y, S + args.gate_slices));
  float4 sig = 1.0f / (1.0f + exp(-gate));
  float4 silu = gate * sig;
  float4 res = silu * up;
  args.dst_tensor.Write(ucl::Convert<args.dst_tensor::type>(res), X, Y, S);
}
)";
  custom_op.code_ = std::move(op_code);
  return std::make_unique<FusedSwiGLUOp>(std::move(custom_op));
}

absl::Status BuildSwigluGpuGraph(
    const std::vector<uint32_t>& input_ids, uint32_t output_id,
    const SwigluAttributes& attr, ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.empty()) {
    return absl::InvalidArgumentError("SwiGLU expects 1 or 2 inputs.");
  }
  if (input_ids.size() == 1) {
    ABSL_ASSIGN_OR_RETURN(auto gate_up, model_builder->GetTensor(input_ids[0]));
    auto dst_shape = gate_up.tensor_desc.GetBHWCShape();
    int total_channels = dst_shape.c;
    int gate_size = attr.gate_size > 0 ? attr.gate_size : total_channels / 2;
    int gate_slices = (gate_size + 3) / 4;
    dst_shape.c = gate_size;

    ::ml_drift::TensorDescriptor dst_desc = gate_up.tensor_desc;
    dst_desc.SetBHWCShape(dst_shape);
    auto dst = model_builder->AddTensor(dst_desc);

    auto op =
        CreateFusedSwiGLU(gate_up.tensor_desc, dst.tensor_desc, gate_slices);
    model_builder->AddGpuOperation({gate_up}, {dst}, std::move(op), "swiglu");
    return model_builder->UpdateOutputTensor(dst, output_id);
  } else {
    ABSL_ASSIGN_OR_RETURN(auto gate, model_builder->GetTensor(input_ids[0]));
    ABSL_ASSIGN_OR_RETURN(auto up, model_builder->GetTensor(input_ids[1]));
    auto sigmoid_gate =
        model_builder->Elementwise(gate, ::ml_drift::OperationType::SIGMOID);
    auto silu_gate = model_builder->Multiplication(gate, sigmoid_gate);
    auto output = model_builder->Multiplication(silu_gate, up);
    return model_builder->UpdateOutputTensor(output, output_id);
  }
}

}  // namespace

absl::Status CreateSwigluFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const SwigluAttributes& attr =
      std::any_cast<const SwigluAttributes&>(node.operation.attributes);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSwigluGpuGraph(input_ids, outputs[0]->id, attr, model_builder);
}

absl::Status CreateSwigluFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const SwigluAttributes& attr =
      std::any_cast<const SwigluAttributes&>(node.attr);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSwigluGpuGraph(input_ids, outputs[0]->id, attr, model_builder);
}

}  // namespace litert::ml_drift
