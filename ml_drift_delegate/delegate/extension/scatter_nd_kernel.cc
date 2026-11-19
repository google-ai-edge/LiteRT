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
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

// Index depth 3: each index vector (b, h, w) addresses an output cell, and the
// matching updates row supplies its channel vector. Cells that are not
// addressed stay zero. A [N, 3] indices tensor is the case Bi = N, Hi = Wi = 1.
//
// All index vectors that address a cell contribute to it: float updates are
// summed and bool updates are OR-ed. Out-of-range index vectors are ignored.
//
// Each thread computes one output element (X, Y, S, B) by scanning `indices`.
std::unique_ptr<::ml_drift::GPUOperation> CreateScatterNdOp(
    const ::ml_drift::TensorDescriptor& indices_desc,
    const ::ml_drift::TensorDescriptor& updates_desc,
    const ::ml_drift::TensorDescriptor& dst_desc) {
  ::ml_drift::GPUOperation op;
  op.AddSrcTensor("indices", indices_desc);
  op.AddSrcTensor("updates", updates_desc);
  op.AddDstTensor("dst_tensor", dst_desc);
  op.tensor_to_grid_ = ::ml_drift::TensorToGrid::kWBToX_HDToY_SToZ;

  const bool dst_batch = dst_desc.HasAxis(::ml_drift::Axis::kBatch);
  const bool is_float = ::ml_drift::IsFloatType(updates_desc.GetDataType());

  std::string c = "MAIN_FUNCTION($0) {\n";
  if (dst_batch) {
    c += R"(  int linear_xb = ucl::GetGlobalId<0>();
  int X = linear_xb / args.dst_tensor.Batch();
  int B = linear_xb % args.dst_tensor.Batch();
)";
  } else {
    c += R"(  int X = ucl::GetGlobalId<0>();
  int B = 0;
)";
  }
  c += R"(  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() ||
      S >= args.dst_tensor.Slices()) {
    return;
  }
)";
  // Every matching index vector contributes to its cell.
  std::string init;
  std::string combine;
  std::string convert;
  if (is_float) {
    init = "  float4 acc = ucl::Init<float4>(0.0f);\n";
    combine =
        "          acc += ucl::Convert<float4>("
        "args.updates.Read(iw, ih, S, ib));\n";
    convert =
        "  args.dst_tensor::type res = "
        "ucl::Convert<args.dst_tensor::type>(acc);\n";
  } else {
    // Bool updates.
    init = "  args.updates::type acc = args.updates::zero_value;\n";
    combine = R"(          args.updates::type v = args.updates.Read(iw, ih, S, ib);
          acc.x = acc.x || v.x;
          acc.y = acc.y || v.y;
          acc.z = acc.z || v.z;
          acc.w = acc.w || v.w;
)";
    convert = R"(  args.dst_tensor::type res;
  res.x = ucl::Convert<args.dst_tensor::scalar_type>(acc.x);
  res.y = ucl::Convert<args.dst_tensor::scalar_type>(acc.y);
  res.z = ucl::Convert<args.dst_tensor::scalar_type>(acc.z);
  res.w = ucl::Convert<args.dst_tensor::scalar_type>(acc.w);
)";
  }
  c += init;
  c += R"(  for (int ib = 0; ib < args.indices.Batch(); ib++) {
    for (int ih = 0; ih < args.indices.Height(); ih++) {
      for (int iw = 0; iw < args.indices.Width(); iw++) {
        int4 coord = args.indices.Read<int>(iw, ih, 0, ib);
        if (coord.x == B && coord.y == Y && coord.z == X) {
)";
  c += combine;
  c += R"(        }
      }
    }
  }
)";
  c += convert;
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
  const ::ml_drift::DataType updates_type = updates.tensor_desc.GetDataType();
  if (!::ml_drift::IsFloatType(updates_type) &&
      updates_type != ::ml_drift::DataType::kBool) {
    return absl::InvalidArgumentError(
        "scatter_nd supports only float and bool updates.");
  }

  ::ml_drift::TensorDescriptor dst_desc = updates.tensor_desc;
  dst_desc.SetBHWCShape(output_shape);
  // The layout of `updates` does not tell whether the output has a batch axis.
  dst_desc.SetLayout(output_shape.b == 1 ? ::ml_drift::Layout::kHWC
                                         : ::ml_drift::Layout::kBHWC);
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
