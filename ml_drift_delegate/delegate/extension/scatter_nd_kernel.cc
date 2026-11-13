// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/extension/scatter_nd_kernel.h"

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
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

// output[b] = sum of updates[n] over all n with indices[n] == b.
std::unique_ptr<::ml_drift::GPUOperation> CreateScatterNdOp(
    const ::ml_drift::TensorDescriptor& indices_desc,
    const ::ml_drift::TensorDescriptor& updates_desc,
    const ::ml_drift::TensorDescriptor& dst_desc) {
  ::ml_drift::GPUOperation op;
  op.AddSrcTensor("indices", indices_desc);
  op.AddSrcTensor("updates", updates_desc);
  op.AddDstTensor("dst_tensor", dst_desc);
  op.tensor_to_grid_ = ::ml_drift::TensorToGrid::kWBToX_HDToY_SToZ;

  const bool indices_batch = indices_desc.HasAxis(::ml_drift::Axis::kBatch);
  const bool updates_batch = updates_desc.HasAxis(::ml_drift::Axis::kBatch);
  const bool dst_batch = dst_desc.HasAxis(::ml_drift::Axis::kBatch);

  std::string c = "MAIN_FUNCTION($0) {\n";
  if (dst_batch) {
    c += "  int linear_xb = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_xb / args.dst_tensor.Batch();\n";
    c += "  int B = linear_xb % args.dst_tensor.Batch();\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
    c += "  int B = 0;\n";
  }
  c += R"(  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || S >= args.dst_tensor.Slices()) {
    return;
  }
  float4 acc = ucl::Init<float4>(0.0f);
)";
  c += std::string("  for (int n = 0; n < ") +
       (updates_batch ? "args.updates.Batch()" : "1") + "; n++) {\n";
  c += std::string("    int index = args.indices.Read<int>(0, 0, 0") +
       (indices_batch ? ", n" : "") + ").x;\n";
  c += "    if (index == B) {\n";
  c += std::string(
           "      acc += ucl::Convert<float4>(args.updates.Read(X, Y, S") +
       (updates_batch ? ", n" : "") + "));\n";
  c += "    }\n";
  c += "  }\n";
  c += "  args.dst_tensor::type res = "
       "ucl::Convert<args.dst_tensor::type>(acc);\n";
  c += std::string("  args.dst_tensor.Write(res, X, Y, S") +
       (dst_batch ? ", B" : "") + ");\n";
  c += "}\n";
  op.code_ = std::move(c);
  return std::make_unique<::ml_drift::GPUOperation>(std::move(op));
}

absl::Status BuildScatterNd(uint32_t indices_id, uint32_t updates_id,
                            uint32_t output_id,
                            const ::ml_drift::BHWC& output_shape,
                            ::ml_drift::GpuModelBuilder* model_builder) {
  ABSL_ASSIGN_OR_RETURN(auto indices, model_builder->GetTensor(indices_id));
  ABSL_ASSIGN_OR_RETURN(auto updates, model_builder->GetTensor(updates_id));

  ::ml_drift::TensorDescriptor dst_desc = updates.tensor_desc;
  dst_desc.SetBHWCShape(output_shape);
  auto dst = model_builder->AddTensor(dst_desc);

  auto op = CreateScatterNdOp(indices.tensor_desc, updates.tensor_desc,
                              dst.tensor_desc);
  model_builder->AddGpuOperation({indices, updates}, {dst}, std::move(op),
                                 kScatterNdType);
  return model_builder->UpdateOutputTensor(dst, output_id);
}

}  // namespace

absl::Status CreateScatterNdFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  if (inputs.size() != 2 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "scatter_nd expects 2 inputs and 1 output.");
  }
  return BuildScatterNd(inputs[0]->id, inputs[1]->id, outputs[0]->id,
                        outputs[0]->tensor.shape, model_builder);
}

absl::Status CreateScatterNdFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& op,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (inputs.size() != 2 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "scatter_nd expects 2 inputs and 1 output.");
  }
  return BuildScatterNd(inputs[0]->id, inputs[1]->id, outputs[0]->id,
                        outputs[0]->desc.GetBHWCShape(), model_builder);
}

}  // namespace litert::ml_drift
