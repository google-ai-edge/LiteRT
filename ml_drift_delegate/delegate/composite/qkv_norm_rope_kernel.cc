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

#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_kernel.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_parser.h"

namespace litert::ml_drift {

namespace {

class FusedQkvNormRoPEOp : public ::ml_drift::GPUOperation {
 public:
  explicit FusedQkvNormRoPEOp(int total_heads, int half_head_slices)
      : total_heads_(total_heads), half_head_slices_(half_head_slices) {}
  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(dst_[0]->Width(), total_heads_, half_head_slices_);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    // TODO(b/552147487): Optimize this kernel to utilize simd more.
    return {::ml_drift::int3(1, 1, half_head_slices_)};
  }

  FusedQkvNormRoPEOp(FusedQkvNormRoPEOp&&) = default;
  FusedQkvNormRoPEOp& operator=(FusedQkvNormRoPEOp&&) = default;
  FusedQkvNormRoPEOp(const FusedQkvNormRoPEOp&) = delete;
  FusedQkvNormRoPEOp& operator=(const FusedQkvNormRoPEOp&) = delete;

 private:
  int total_heads_;
  int half_head_slices_;
};

std::unique_ptr<::ml_drift::GPUOperation> CreateFusedQkvNormRoPE(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::TensorDescriptor& qkv_desc,
    const ::ml_drift::TensorDescriptor& pos_desc,
    const ::ml_drift::TensorDescriptor& q_weight_desc,
    const ::ml_drift::TensorDescriptor& k_weight_desc,
    const ::ml_drift::TensorDescriptor& q_out_desc,
    const ::ml_drift::TensorDescriptor& k_out_desc,
    const ::ml_drift::TensorDescriptor& v_out_desc,
    const QkvNormRopeAttributes& attr) {
  int total_slices_per_head = (attr.head_dim + 3) / 4;
  int half_slices = total_slices_per_head / 2;
  int total_heads = attr.num_heads + 2 * attr.num_kv_heads;

  FusedQkvNormRoPEOp custom_op(total_heads, half_slices);
  custom_op.work_group_size_ = ::ml_drift::int3(1, 1, half_slices);
  custom_op.args_.AddInt("num_heads", attr.num_heads);
  custom_op.args_.AddInt("num_kv_heads", attr.num_kv_heads);
  custom_op.args_.AddInt("total_heads", total_heads);
  custom_op.args_.AddInt("head_dim", attr.head_dim);
  custom_op.args_.AddInt("total_slices_per_head", total_slices_per_head);
  custom_op.args_.AddInt("half_slices", half_slices);
  custom_op.args_.AddFloat("min_timescale", attr.min_timescale);
  custom_op.args_.AddFloat("max_timescale", attr.max_timescale);
  custom_op.args_.AddFloat("proportion", attr.proportion);
  custom_op.args_.AddFloat("epsilon", attr.epsilon);

  custom_op.AddSrcTensor("qkv", qkv_desc);
  custom_op.AddSrcTensor("position", pos_desc);
  custom_op.AddSrcTensor("q_weight", q_weight_desc);
  custom_op.AddSrcTensor("k_weight", k_weight_desc);
  custom_op.AddDstTensor("q_out", q_out_desc);
  custom_op.AddDstTensor("k_out", k_out_desc);
  custom_op.AddDstTensor("v_out", v_out_desc);

  std::string pow_func_name = "pow";
  std::string sin_func_name = "sin";
  std::string cos_func_name = "cos";
  if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
    pow_func_name = "native_powr";
    sin_func_name = "native_sin";
    cos_func_name = "native_cos";
  }

  std::string reduction_code;
  for (int offset = half_slices / 2; offset > 0; offset >>= 1) {
    absl::StrAppend(&reduction_code,
                    "    if (tid < ", offset,
                    ") { shared_sum[tid] += shared_sum[tid + ", offset,
                    "]; }\n    ucl::SyncThreads<WorkGroup, Local>();\n");
  }

  std::string op_code = absl::StrCat(R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  int tid = ucl::GetLocalId<2>();
  if (X >= args.q_out.Width() || Y >= args.total_heads || S >= args.half_slices) {
    return;
  }

  int base_slice = Y * args.total_slices_per_head;
  float4 val0 = ucl::Convert<float4>(args.qkv.Read(X, 0, base_slice + S, 0));
  float4 val1 = ucl::Convert<float4>(args.qkv.Read(X, 0, base_slice + S + args.half_slices, 0));

  __local float shared_sum[)", half_slices, R"(];

  if (Y < args.num_heads) {
    // === QUERY HEAD ===
    shared_sum[tid] = dot(val0, val0) + dot(val1, val1);
    ucl::SyncThreads<WorkGroup, Local>();
)", reduction_code, R"(
    float inv_std = rsqrt(shared_sum[0] / ucl::Convert<float>(args.head_dim) + args.epsilon);
    float w0_x = args.q_weight.Read(0, 0, 0, S * 4 + 0).x;
    float w0_y = args.q_weight.Read(0, 0, 0, S * 4 + 1).x;
    float w0_z = args.q_weight.Read(0, 0, 0, S * 4 + 2).x;
    float w0_w = args.q_weight.Read(0, 0, 0, S * 4 + 3).x;
    float4 w0 = float4(w0_x, w0_y, w0_z, w0_w);

    int s1_base = (S + args.half_slices) * 4;
    float w1_x = args.q_weight.Read(0, 0, 0, s1_base + 0).x;
    float w1_y = args.q_weight.Read(0, 0, 0, s1_base + 1).x;
    float w1_z = args.q_weight.Read(0, 0, 0, s1_base + 2).x;
    float w1_w = args.q_weight.Read(0, 0, 0, s1_base + 3).x;
    float4 w1 = float4(w1_x, w1_y, w1_z, w1_w);

    val0 = val0 * inv_std * w0;
    val1 = val1 * inv_std * w1;

    // RoPE
    float pos_scalar;
    if (args.position.Channels() > 1) {
      args.position.ReadPerChannel<float>(pos_scalar, 0, 0, X % args.position.Channels());
    } else {
      pos_scalar = args.position.Read<float>(X % args.position.Width(), 0, 0).x;
    }
    float4 pos_val = ucl::Init<float4>(pos_scalar);
    float inv_dst_ch = 1.0f / ucl::Convert<float>(args.head_dim);
    int4 p = S * 4 + ucl::Init<int4>(0, 1, 2, 3);
    float4 fraction = 2.0f * ucl::Convert<float4>(p) * inv_dst_ch;

    float4 min_timescale = ucl::Init<float4>(args.min_timescale);
    float4 max_timescale = ucl::Init<float4>(args.max_timescale);
    float4 timescale = min_timescale * )", pow_func_name, R"((max_timescale / min_timescale, fraction);
    float4 sinusoid_inp = pos_val / timescale;
    Type sin_val = ucl::Convert<Type>()", sin_func_name, R"((sinusoid_inp));
    Type cos_val = ucl::Convert<Type>()", cos_func_name, R"((sinusoid_inp));

    Type v0 = ucl::Convert<Type>(val0);
    Type v1 = ucl::Convert<Type>(val1);
    Type out0 = v0 * cos_val - v1 * sin_val;
    Type out1 = v1 * cos_val + v0 * sin_val;

    args.q_out.Write(out0, X, Y, S);
    args.q_out.Write(out1, X, Y, S + args.half_slices);
  } else if (Y < args.num_heads + args.num_kv_heads) {
    // === KEY HEAD ===
    int kv_head = Y - args.num_heads;
    shared_sum[tid] = dot(val0, val0) + dot(val1, val1);
    ucl::SyncThreads<WorkGroup, Local>();
)", reduction_code, R"(
    float inv_std = rsqrt(shared_sum[0] / ucl::Convert<float>(args.head_dim) + args.epsilon);
    float w0_x = args.k_weight.Read(0, 0, 0, S * 4 + 0).x;
    float w0_y = args.k_weight.Read(0, 0, 0, S * 4 + 1).x;
    float w0_z = args.k_weight.Read(0, 0, 0, S * 4 + 2).x;
    float w0_w = args.k_weight.Read(0, 0, 0, S * 4 + 3).x;
    float4 w0 = float4(w0_x, w0_y, w0_z, w0_w);

    int s1_base = (S + args.half_slices) * 4;
    float w1_x = args.k_weight.Read(0, 0, 0, s1_base + 0).x;
    float w1_y = args.k_weight.Read(0, 0, 0, s1_base + 1).x;
    float w1_z = args.k_weight.Read(0, 0, 0, s1_base + 2).x;
    float w1_w = args.k_weight.Read(0, 0, 0, s1_base + 3).x;
    float4 w1 = float4(w1_x, w1_y, w1_z, w1_w);

    val0 = val0 * inv_std * w0;
    val1 = val1 * inv_std * w1;

    // RoPE
    float pos_scalar;
    if (args.position.Channels() > 1) {
      args.position.ReadPerChannel<float>(pos_scalar, 0, 0, X % args.position.Channels());
    } else {
      pos_scalar = args.position.Read<float>(X % args.position.Width(), 0, 0).x;
    }
    float4 pos_val = ucl::Init<float4>(pos_scalar);
    float inv_dst_ch = 1.0f / ucl::Convert<float>(args.head_dim);
    int4 p = S * 4 + ucl::Init<int4>(0, 1, 2, 3);
    float4 fraction = 2.0f * ucl::Convert<float4>(p) * inv_dst_ch;

    float4 min_timescale = ucl::Init<float4>(args.min_timescale);
    float4 max_timescale = ucl::Init<float4>(args.max_timescale);
    float4 timescale = min_timescale * )", pow_func_name, R"((max_timescale / min_timescale, fraction);
    float4 sinusoid_inp = pos_val / timescale;
    Type sin_val = ucl::Convert<Type>()", sin_func_name, R"((sinusoid_inp));
    Type cos_val = ucl::Convert<Type>()", cos_func_name, R"((sinusoid_inp));

    Type v0 = ucl::Convert<Type>(val0);
    Type v1 = ucl::Convert<Type>(val1);
    Type out0 = v0 * cos_val - v1 * sin_val;
    Type out1 = v1 * cos_val + v0 * sin_val;

    args.k_out.Write(out0, X, kv_head, S);
    args.k_out.Write(out1, X, kv_head, S + args.half_slices);
  } else {
    // === VALUE HEAD ===
    int kv_head = Y - (args.num_heads + args.num_kv_heads);
    Type out0 = ucl::Convert<Type>(val0);
    Type out1 = ucl::Convert<Type>(val1);
    args.v_out.Write(out0, X, kv_head, S);
    args.v_out.Write(out1, X, kv_head, S + args.half_slices);
  }
}
)");

  absl::StrReplaceAll(
      {{"Type", ::ml_drift::ToUclDataType(q_out_desc.GetDataType(), 4)}},
      &op_code);
  custom_op.code_ = std::move(op_code);
  return std::make_unique<FusedQkvNormRoPEOp>(std::move(custom_op));
}

absl::Status BuildQkvNormRopeGpuGraph(
    const std::vector<uint32_t>& input_ids,
    const std::vector<uint32_t>& output_ids,
    const QkvNormRopeAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() != 4) {
    return absl::InvalidArgumentError("QkvNormRoPE expects 4 inputs.");
  }
  if (output_ids.size() != 3) {
    return absl::InvalidArgumentError("QkvNormRoPE expects 3 outputs.");
  }
  ABSL_ASSIGN_OR_RETURN(auto qkv, model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(auto pos, model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(auto q_weight, model_builder->GetTensor(input_ids[2]));
  ABSL_ASSIGN_OR_RETURN(auto k_weight, model_builder->GetTensor(input_ids[3]));

  QkvNormRopeAttributes resolved_attr = attr;
  auto q_target = model_builder->GetTensor(output_ids[0]);
  if (q_target.ok()) {
    const auto q_shape = q_target->tensor_desc.GetBHWCShape();
    if (q_shape.h > 0) {
      if (resolved_attr.num_heads > 0 && resolved_attr.num_heads != q_shape.h) {
        return absl::InvalidArgumentError(absl::StrCat(
            "QkvNormRoPE num_heads in attr (", resolved_attr.num_heads,
            ") does not match output shape (", q_shape.h, ")."));
      }
      resolved_attr.num_heads = q_shape.h;
    }
    if (q_shape.c > 0) {
      if (resolved_attr.head_dim > 0 && resolved_attr.head_dim != q_shape.c) {
        return absl::InvalidArgumentError(absl::StrCat(
            "QkvNormRoPE head_dim in attr (", resolved_attr.head_dim,
            ") does not match output shape (", q_shape.c, ")."));
      }
      resolved_attr.head_dim = q_shape.c;
    }
  }
  auto k_target = model_builder->GetTensor(output_ids[1]);
  if (k_target.ok()) {
    const auto k_shape = k_target->tensor_desc.GetBHWCShape();
    if (k_shape.h > 0) {
      if (resolved_attr.num_kv_heads > 0 &&
          resolved_attr.num_kv_heads != k_shape.h) {
        return absl::InvalidArgumentError(absl::StrCat(
            "QkvNormRoPE num_kv_heads in attr (", resolved_attr.num_kv_heads,
            ") does not match output shape (", k_shape.h, ")."));
      }
      resolved_attr.num_kv_heads = k_shape.h;
    }
  }

  int T = qkv.tensor_desc.GetBHWCShape().w;
  auto q_out = model_builder->AddTensor(
      ::ml_drift::BHWC(1, resolved_attr.num_heads, T, resolved_attr.head_dim),
      qkv.tensor_desc.GetDataType());
  auto k_out =
      model_builder->AddTensor(::ml_drift::BHWC(1, resolved_attr.num_kv_heads,
                                                T, resolved_attr.head_dim),
                               qkv.tensor_desc.GetDataType());
  auto v_out =
      model_builder->AddTensor(::ml_drift::BHWC(1, resolved_attr.num_kv_heads,
                                                T, resolved_attr.head_dim),
                               qkv.tensor_desc.GetDataType());

  auto op = CreateFusedQkvNormRoPE(
      model_builder->gpu_info(), qkv.tensor_desc, pos.tensor_desc,
      q_weight.tensor_desc, k_weight.tensor_desc, q_out.tensor_desc,
      k_out.tensor_desc, v_out.tensor_desc, resolved_attr);
  model_builder->AddGpuOperation(
      {qkv, pos, q_weight, k_weight}, {q_out, k_out, v_out}, std::move(op),
      "qkv_norm_rope");

  ABSL_RETURN_IF_ERROR(model_builder->UpdateOutputTensor(q_out, output_ids[0]));
  ABSL_RETURN_IF_ERROR(model_builder->UpdateOutputTensor(k_out, output_ids[1]));
  ABSL_RETURN_IF_ERROR(model_builder->UpdateOutputTensor(v_out, output_ids[2]));
  return absl::OkStatus();
}

}  // namespace

absl::Status CreateQkvNormRopeFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const QkvNormRopeAttributes& attr =
      std::any_cast<const QkvNormRopeAttributes&>(node.operation.attributes);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* output : outputs) output_ids.push_back(output->id);
  return BuildQkvNormRopeGpuGraph(input_ids, output_ids, attr, model_builder);
}

absl::Status CreateQkvNormRopeFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const QkvNormRopeAttributes& attr =
      std::any_cast<const QkvNormRopeAttributes&>(node.attr);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* output : outputs) output_ids.push_back(output->id);
  return BuildQkvNormRopeGpuGraph(input_ids, output_ids, attr, model_builder);
}

}  // namespace litert::ml_drift
