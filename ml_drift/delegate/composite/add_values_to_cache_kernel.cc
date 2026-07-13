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

#include "third_party/odml/litert/ml_drift/delegate/composite/add_values_to_cache_kernel.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/delegate/composite/add_values_to_cache_parser.h"

namespace litert::ml_drift {
namespace {

class AddValuesToCacheOp : public ::ml_drift::GPUOperation {
 public:
  AddValuesToCacheOp() = default;
  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(src_[0]->Width(), batch_size_, src_[0]->Slices());
  }

  // Move only
  AddValuesToCacheOp(AddValuesToCacheOp&& operation) = default;
  AddValuesToCacheOp& operator=(AddValuesToCacheOp&& operation) = default;
  AddValuesToCacheOp(const AddValuesToCacheOp&) = delete;
  AddValuesToCacheOp& operator=(const AddValuesToCacheOp&) = delete;

  int batch_size_ = 1;
};

std::unique_ptr<::ml_drift::GPUOperation> CreateAddValuesToCache(
    const ::ml_drift::TensorDescriptor& src_k,
    const ::ml_drift::TensorDescriptor& src_v,
    const ::ml_drift::TensorDescriptor& cache_k,
    const ::ml_drift::TensorDescriptor& cache_v, int cache_size,
    int head_dimension, int kv_cache_batch_size, float scale_k, float scale_v) {
  AddValuesToCacheOp custom_op;
  custom_op.batch_size_ = kv_cache_batch_size;
  custom_op.args_.AddInt("batch_size", custom_op.batch_size_);
  custom_op.args_.AddInt("cache_size", cache_size);
  custom_op.args_.AddInt("head_size", head_dimension);
  custom_op.AddSrcTensor("src_k", src_k);
  custom_op.AddSrcTensor("src_v", src_v);
  ::ml_drift::BufferDescriptor params_buffer;
  params_buffer.element_type = ::ml_drift::DataType::INT32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);
  custom_op.AddDstTensor("cache_k", cache_k);
  custom_op.AddDstTensor("cache_v", cache_v);

  // quantized cache case. Expects src_k and src_v to be float and quantize
  // inside the shader.
  bool quantized_cache = false;
  if ((src_k.GetDataType() == ::ml_drift::DataType::FLOAT32 ||
       src_k.GetDataType() == ::ml_drift::DataType::FLOAT16) &&
      cache_k.GetDataType() == ::ml_drift::DataType::UINT8) {
    float max_value_k = static_cast<float>(INT8_MAX) * scale_k;
    float min_value_k = static_cast<float>(INT8_MIN) * scale_k;
    float inverse_scale_k = 1.0f / scale_k;
    custom_op.args_.AddFloat("min_k", min_value_k, src_k.GetDataType());
    custom_op.args_.AddFloat("max_k", max_value_k, src_k.GetDataType());
    custom_op.args_.AddFloat("inverse_scale_k", inverse_scale_k,
                             src_k.GetDataType());
    float max_value_v = static_cast<float>(INT8_MAX) * scale_v;
    float min_value_v = static_cast<float>(INT8_MIN) * scale_v;
    float inverse_scale_v = 1.0f / scale_v;
    custom_op.args_.AddFloat("min_v", min_value_v, src_v.GetDataType());
    custom_op.args_.AddFloat("max_v", max_value_v, src_v.GetDataType());
    custom_op.args_.AddFloat("inverse_scale_v", inverse_scale_v,
                             src_v.GetDataType());
    quantized_cache = true;
  }

  std::string op_code;
  op_code = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.src_k.Width() || Y >= args.batch_size || S >= args.src_k.Slices()) {
    return;
  }
  int token_index_offset = args.params.Read(0);
  int active_tokens = args.params.Read(1);
  int token_index = token_index_offset + X;
  if (token_index >= args.cache_size || token_index >= active_tokens) {
    return;
  }

  int src_y = Y % args.src_k.Height();  // broadcast Height dim used as Batch
  args.src_k::type value_k = args.src_k.Read(X, src_y, S);
  args.src_v::type value_v = args.src_v.Read(X, src_y, S);
  )";

  if (quantized_cache) {
    const std::string src_k_type = ToUclDataType(src_k.GetDataType(), 4);
    op_code += absl::Substitute(R"(
  // quantize the slices
  $0 clamped_value_k = min(ucl::Init<$0>(args.max_k), max(ucl::Init<$0>(args.min_k), value_k));
  $0 quantized_value_k = round((clamped_value_k - ucl::Init<$0>(args.min_k)) * ucl::Init<$0>(args.inverse_scale_k));
  uchar4 final_value_k = ucl::Convert<uchar4>(quantized_value_k);

  $0 clamped_value_v = min(ucl::Init<$0>(args.max_v), max(ucl::Init<$0>(args.min_v), value_v));
  $0 quantized_value_v = round((clamped_value_v - ucl::Init<$0>(args.min_v)) * ucl::Init<$0>(args.inverse_scale_v));
  uchar4 final_value_v = ucl::Convert<uchar4>(quantized_value_v);
  )",
                                src_k_type);
  } else {
    op_code += R"(
  args.src_k::type final_value_k = value_k;
  args.src_v::type final_value_v = value_v;
  )";
  }
  op_code += R"(
  {
    // cache_k - kOSpatialIOGroupO4I4 layout, O - cache_size, I - head_size
    // OGroup = o_slices -> actual layout SpatialIOGroupO4I4
    // SpatialIOGroupO4I4 Spatial ISlice OSlice O4 I4
    // index of vec4(I4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + O4Index
    int k_o = token_index;
    int k_o_slice = k_o / 4;
    int k_o4_index = k_o % 4;
    int k_i = S * 4;
    int k_i_slice = k_i / 4;
    int k_sp = Y;
    int i_slices = (args.head_size + 3) / 4;
    int o_slices = (args.cache_size + 3) / 4;
    int k_index = ((k_sp * i_slices + k_i_slice) * o_slices + k_o_slice) * 4 + k_o4_index;
    args.cache_k.WriteLinear(final_value_k, k_index);
  }
  {
    // cache_v - kOSpatialIOGroupI4O4 layout, O - head_size, I - cache_size
    // OGroup = o_slices -> actual layout SpatialIOGroupI4O4
    // SpatialIOI4O4 Spatial ISlice OSlice I4 O4
    // index of vec4(O4) =
    //   ((Spatial * ISlices + ISlice) * OSlices + Oslice) * 4 + I4Index
    int v_o = S * 4;
    int v_o_slice = v_o / 4;
    int v_i = token_index;
    int v_i_slice = v_i / 4;
    int v_i4_index = v_i % 4;
    int v_sp = Y;
    int i_slices = (args.cache_size + 3) / 4;
    int o_slices = (args.head_size + 3) / 4;
    int v_index = ((v_sp * i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    args.cache_v.WriteLinear(final_value_v, v_index);
  }
})";

  custom_op.code_ = std::move(op_code);
  return std::make_unique<AddValuesToCacheOp>(std::move(custom_op));
}

}  // namespace

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateAddValuesToCacheFromNode(const ::ml_drift::OperationDef& op_def,
                               const ::ml_drift::Node& node) {
  if (op_def.src_tensors.size() != 3 || op_def.dst_tensors.size() != 2) {
    return absl::InvalidArgumentError(
        "AddValuesToCache operation expects 3 inputs and 2 outputs.");
  }

  const auto& attr = std::any_cast<const AddValuesToCacheAttributes&>(
      node.operation.attributes);
  float scale_k = attr.scale_k.value_or(1.0);
  float scale_v = attr.scale_v.value_or(1.0);
  return CreateAddValuesToCache(op_def.src_tensors[0], op_def.src_tensors[1],
                                op_def.dst_tensors[0], op_def.dst_tensors[1],
                                attr.cache_size, attr.head_size,
                                attr.kv_cache_batch_size, scale_k, scale_v);
}

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateAddValuesToCacheFromNode(const ::ml_drift::OperationDef& op_def,
                               const ::ml_drift::ir::IrOp& ir_op) {
  if (op_def.src_tensors.size() != 3 || op_def.dst_tensors.size() != 2) {
    return absl::InvalidArgumentError(
        "AddValuesToCache operation expects 3 inputs and 2 outputs.");
  }

  const auto& attr =
      std::any_cast<const AddValuesToCacheAttributes&>(ir_op.attr);
  float scale_k = attr.scale_k.value_or(1.0);
  float scale_v = attr.scale_v.value_or(1.0);
  return CreateAddValuesToCache(op_def.src_tensors[0], op_def.src_tensors[1],
                                op_def.dst_tensors[0], op_def.dst_tensors[1],
                                attr.cache_size, attr.head_size,
                                attr.kv_cache_batch_size, scale_k, scale_v);
}

}  // namespace litert::ml_drift
