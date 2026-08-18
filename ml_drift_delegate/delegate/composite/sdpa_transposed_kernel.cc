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

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.h"

#include <algorithm>
#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

namespace litert::ml_drift {
namespace {

int GetOptimalWorkgroupSize(const ::ml_drift::GpuInfo& gpu_info,
                            int dst_slices) {
  if (gpu_info.IsAdreno()) {
    return 64;
  }
  return 32;
}

class SdpaFlashDecodeOp : public ::ml_drift::GPUOperation {
 public:
  explicit SdpaFlashDecodeOp(int wg_size = 32) : wg_size_(wg_size) {
    work_group_size_ = ::ml_drift::int3(wg_size_, 1, 1);
  }

  ::ml_drift::int3 GetGridSize() const override {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch() * wg_size_;
    const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
    const int grid_z = 1;
    return ::ml_drift::int3(grid_x, grid_y, grid_z);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

 private:
  int wg_size_ = 32;
};

std::unique_ptr<::ml_drift::GPUOperation> CreateSdpaSingleKernelWithFlashDecode(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::OperationDef& definition, bool has_mask, bool has_param,
    int src_end_ch_index, int k_cache_slices, std::optional<float> softcap) {
  const int dst_channels = definition.dst_tensors[0].GetBHWCShape().c;
  const int dst_slices = (dst_channels + 3) / 4;
  const int wg_size = GetOptimalWorkgroupSize(gpu_info, dst_slices);
  const int s_iters = std::max(1, (dst_slices + wg_size - 1) / wg_size);

  auto op = std::make_unique<SdpaFlashDecodeOp>(wg_size);
  op->AddSrcTensor("q", definition.src_tensors[0]);
  op->AddSrcTensor("k", definition.src_tensors[1]);
  op->AddSrcTensor("v", definition.src_tensors[2]);
  if (has_mask) {
    op->AddSrcTensor("mask", definition.src_tensors[3]);
  }
  if (has_param) {
    ::ml_drift::BufferDescriptor buffer_desc;
    buffer_desc.element_type = ::ml_drift::DataType::INT32;
    buffer_desc.element_size = 1;
    op->AddSrcBuffer("params", buffer_desc);
    op->args_.AddInt("src_end_ch_index", src_end_ch_index);
  }
  op->args_.AddInt("k_cache_slices", k_cache_slices);
  if (softcap.has_value()) {
    op->args_.AddFloat("softcap", *softcap);
  }
  op->AddDstTensor("dst", definition.dst_tensors[0]);

  std::string code;
  code += "#define WG_SIZE " + std::to_string(wg_size) + "\n";
  code += "#define S_ITERS " + std::to_string(s_iters) + "\n";
  code += "#define Q_SLICES " + std::to_string(dst_slices) + "\n";
  code += "MAIN_FUNCTION($0) {\n";
  code += "  int X = ucl::GetGroupId<0>();\n";
  code += "  int Y = ucl::GetGroupId<1>();\n";
  code += "  int tid = ucl::GetLocalId<0>();\n";
  code += "  if (X >= args.dst.Width() || Y >= args.dst.Height()) return;\n";

  if (has_param) {
    code += "  int active_kv_len = args.params.Read(args.src_end_ch_index);\n";
  } else {
    code += "  int active_kv_len = args.k.Width();\n";
  }

  code += R"(
  int k_i_slices = args.k.Slices();
  int k_o_slices = args.k_cache_slices;
  int v_i_slices = args.k_cache_slices;
  int v_o_slices = args.v.Width() / 4;
  int dst_slices = args.dst.Slices();

  int base_k_offset = Y * k_i_slices * k_o_slices * 4;
  int base_v_offset = Y * v_i_slices * v_o_slices * 4;
  int k_stride = k_o_slices * 4;

  __local args.q::type shared_q[Q_SLICES];
  __local float shared_scores[WG_SIZE];

  for (int s = tid; s < k_i_slices; s += WG_SIZE) {
    shared_q[s] = args.q.Read(X, Y, s);
  }
  ucl::SyncThreads<WorkGroup, Local>();

  float max_score = -10000.0f;
  float sum_exp = 0.0f;
  args.dst::type out_val[S_ITERS];
  for (int s_iter = 0; s_iter < S_ITERS; ++s_iter) {
    out_val[s_iter] = ucl::Init<args.dst::type>(ucl::Convert<args.dst::scalar_type>(0.0f));
  }

  for (int chunk_start = 0; chunk_start < active_kv_len; chunk_start += WG_SIZE) {
    int k_pos = chunk_start + tid;
    float score = 0.0f;

    if (k_pos < active_kv_len) {
      int k_o_slice = k_pos / 4;
      int k_o4_index = k_pos % 4;

      int base_k_index = base_k_offset + k_o_slice * 4 + k_o4_index;

      for (int s = 0; s < k_i_slices; ++s) {
        args.k::type k_vec = args.k.Read(base_k_index);
        score += ucl::Convert<float>(dot(shared_q[s], k_vec));
        base_k_index += k_stride;
      }
)";

  if (softcap.has_value()) {
    code += R"(
      score = tanh(score / args.softcap) * args.softcap;
)";
  }

  if (has_mask) {
    if (definition.src_tensors[3].GetDataType() == ::ml_drift::DataType::BOOL) {
      code += R"(
      int mask_slice = k_pos / 4;
      int mask_comp = k_pos % 4;
      args.mask::type mask_vec = args.mask.Read(X, 0, mask_slice);
      bool valid = mask_vec.x;
      if (mask_comp == 1) valid = mask_vec.y;
      if (mask_comp == 2) valid = mask_vec.z;
      if (mask_comp == 3) valid = mask_vec.w;
      if (!valid) {
        score = -10000.0f;
      }
)";
    } else {
      code += R"(
      int mask_slice = k_pos / 4;
      int mask_comp = k_pos % 4;
      float4 mask_vec = ucl::Convert<float4>(args.mask.Read(X, 0, mask_slice));
      float m_elem = mask_vec.x;
      if (mask_comp == 1) m_elem = mask_vec.y;
      if (mask_comp == 2) m_elem = mask_vec.z;
      if (mask_comp == 3) m_elem = mask_vec.w;
      score += m_elem;
)";
    }
  }

  code += R"(
      shared_scores[tid] = score;
    } else {
      shared_scores[tid] = -10000.0f;
    }
    ucl::SyncThreads<WorkGroup, Local>();

    float chunk_max = -10000.0f;
    for (int i = 0; i < WG_SIZE; ++i) {
      chunk_max = max(chunk_max, shared_scores[i]);
    }

    if (chunk_max > -9999.0f) {
      float new_max = max(max_score, chunk_max);
      float alpha_old = exp(max_score - new_max);

      sum_exp *= alpha_old;
      for (int s_iter = 0; s_iter < S_ITERS; ++s_iter) {
        out_val[s_iter] *= ucl::Init<args.dst::type>(ucl::Convert<args.dst::scalar_type>(alpha_old));
      }
      max_score = new_max;
    }

    for (int i = 0; i < WG_SIZE; ++i) {
      float sc = shared_scores[i];
      if (sc > -9999.0f) {
        float p = exp(sc - max_score);
        if (tid == 0) {
          sum_exp += p;
        }

        int v_k_pos = chunk_start + i;
        int v_i_slice = v_k_pos / 4;
        int v_i4_index = v_k_pos % 4;
        int cur_v_offset = base_v_offset + v_i_slice * v_o_slices * 4 + v_i4_index;

        args.dst::type p_vec = ucl::Init<args.dst::type>(ucl::Convert<args.dst::scalar_type>(p));

        for (int s_iter = 0; s_iter < S_ITERS; ++s_iter) {
          int S = tid + s_iter * WG_SIZE;
          if (S < dst_slices) {
            int v_index = cur_v_offset + S * 4;
            args.v::type v_elem = args.v.Read(v_index);
            out_val[s_iter] += p_vec * v_elem;
          }
        }
      }
    }
    ucl::SyncThreads<WorkGroup, Local>();
  }

  __local float shared_sum_exp;
  if (tid == 0) {
    shared_sum_exp = sum_exp;
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (shared_sum_exp > 0.0f) {
    args.dst::type inv_sum = ucl::Init<args.dst::type>(ucl::Convert<args.dst::scalar_type>(1.0f / shared_sum_exp));
    for (int s_iter = 0; s_iter < S_ITERS; ++s_iter) {
      int S = tid + s_iter * WG_SIZE;
      if (S < dst_slices) {
        out_val[s_iter] *= inv_sum;
        args.dst.Write(out_val[s_iter], X, Y, S);
      }
    }
  } else {
    for (int s_iter = 0; s_iter < S_ITERS; ++s_iter) {
      int S = tid + s_iter * WG_SIZE;
      if (S < dst_slices) {
        args.dst.Write(out_val[s_iter], X, Y, S);
      }
    }
  }
}
)";

  op->code_ = code;
  return op;
}

enum class SdpaImplementationStrategy {
  kSingleKernelFlashDecode,
  kCompositeMultiKernelFallback,
};

inline SdpaImplementationStrategy SelectSdpaStrategy(
    bool is_decode, bool allow_single_kernel, bool request_flash_decoding) {
  if (is_decode && allow_single_kernel && request_flash_decoding) {
    return SdpaImplementationStrategy::kSingleKernelFlashDecode;
  }
  return SdpaImplementationStrategy::kCompositeMultiKernelFallback;
}

struct SdpaInputs {
  ::ml_drift::GpuModelBuilder::TensorHandle q;
  ::ml_drift::GpuModelBuilder::TensorHandle k;
  ::ml_drift::GpuModelBuilder::TensorHandle v;
  std::optional<::ml_drift::GpuModelBuilder::TensorHandle> mask;
  std::optional<::ml_drift::GpuModelBuilder::TensorHandle> param_tensor;
  bool is_decode = false;
  int cache_size = 0;
  int k_cache_slices = 0;
  int src_end_ch_index = 2;
  std::optional<float> softcap;

  bool has_mask() const { return mask.has_value(); }
  bool has_param() const { return param_tensor.has_value(); }
  const ::ml_drift::GpuModelBuilder::TensorHandle* param_tensor_ptr() const {
    return param_tensor.has_value() ? &*param_tensor : nullptr;
  }
};

absl::StatusOr<SdpaInputs> ExtractSdpaInputs(
    const std::vector<uint32_t>& input_ids,
    const SdpaTransposedAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() < 3) {
    return absl::InvalidArgumentError(
        "SDPA transposed expects at least 3 inputs (Q, K, V).");
  }

  SdpaInputs inputs;
  ABSL_ASSIGN_OR_RETURN(inputs.q, model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(inputs.k, model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(inputs.v, model_builder->GetTensor(input_ids[2]));

  if (input_ids.size() == 4) {
    ABSL_ASSIGN_OR_RETURN(auto tensor_4,
                          model_builder->GetTensor(input_ids[3]));
    if (tensor_4.tensor_desc.GetDataType() == ::ml_drift::DataType::INT32) {
      inputs.param_tensor = tensor_4;
    } else {
      inputs.mask = tensor_4;
    }
  } else if (input_ids.size() > 4) {
    ABSL_ASSIGN_OR_RETURN(inputs.mask, model_builder->GetTensor(input_ids[3]));
    ABSL_ASSIGN_OR_RETURN(inputs.param_tensor,
                          model_builder->GetTensor(input_ids[4]));
  }

  // Some model (Qwen3) has 2 seqlen, real & imaginary (complex number) in RoPE.
  inputs.is_decode = inputs.q.tensor_desc.GetBHWCShape().w <= 2;
  inputs.cache_size = inputs.k.tensor_desc.GetBHWCShape().w;
  inputs.k_cache_slices = (inputs.cache_size + 3) / 4;
  inputs.src_end_ch_index = attr.runtime_check.src_end_ch_index.value_or(2);
  inputs.softcap = attr.softcap;

  return inputs;
}

absl::Status BuildSingleKernelFlashDecodeSdpaGraph(
    const SdpaInputs& inputs, uint32_t output_id,
    ::ml_drift::GpuModelBuilder* model_builder) {
  ::ml_drift::OperationDef op_def;
  op_def.src_tensors.push_back(inputs.q.tensor_desc);
  op_def.src_tensors.push_back(inputs.k.tensor_desc);
  op_def.src_tensors.push_back(inputs.v.tensor_desc);
  if (inputs.has_mask()) {
    op_def.src_tensors.push_back(inputs.mask->tensor_desc);
  }

  ::ml_drift::BHWC q_shape = inputs.q.tensor_desc.GetBHWCShape();
  ::ml_drift::BHWC dst_shape(1, q_shape.h, q_shape.w, q_shape.c);
  auto dst_handle =
      model_builder->AddTensor(dst_shape, inputs.q.tensor_desc.GetDataType());
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto gpu_op = CreateSdpaSingleKernelWithFlashDecode(
      model_builder->gpu_info(), op_def, inputs.has_mask(), inputs.has_param(),
      inputs.src_end_ch_index, inputs.k_cache_slices, inputs.softcap);

  std::vector<::ml_drift::GpuModelBuilder::TensorHandle> srcs = {
      inputs.q, inputs.k, inputs.v};
  if (inputs.has_mask()) srcs.push_back(*inputs.mask);
  if (inputs.has_param()) srcs.push_back(*inputs.param_tensor);

  model_builder->AddGpuOperation(srcs, dst_handle, std::move(gpu_op),
                                 "sdpa_single_kernel");

  return model_builder->UpdateOutputTensor(dst_handle, output_id);
}

absl::Status BuildCompositeMultiKernelSdpaGraph(
    const SdpaInputs& inputs, const SdpaTransposedAttributes& attr,
    uint32_t output_id, ::ml_drift::GpuModelBuilder* model_builder) {
  // Fallback implementation using FullyConnectedExternalWeights.
  ::ml_drift::WeightsDescription bmm1_desc = attr.bmm1_weights.desc;
  bmm1_desc.type = inputs.q.tensor_desc.GetDataType();
  const ::ml_drift::GpuModelBuilder::Weights bmm1_external_weights =
      ::ml_drift::CreateExternalWeights(inputs.k, bmm1_desc,
                                        attr.bmm1_weights.weights_shape);
  ::ml_drift::ConvRuntimeCheckDesc bmm1_runtime_check = {
      .dst_end_ch_index = attr.runtime_check.src_end_ch_index,
  };

  ABSL_ASSIGN_OR_RETURN(
      auto logits,
      model_builder->FullyConnectedExternalWeights(
          inputs.q, bmm1_external_weights, /*biases=*/nullptr,
          /*src_exp=*/nullptr, bmm1_runtime_check, inputs.param_tensor_ptr()));

  if (inputs.has_mask()) {
    if (inputs.mask->tensor_desc.GetDataType() == ::ml_drift::DataType::BOOL) {
      ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::BHWC>,
                         ::ml_drift::DataType::FLOAT32>
          fill_tensor;
      fill_tensor.shape = ::ml_drift::BHWC(1, 1, 1, 1);
      // Use a large negative value to simulate -inf. std::limit<float>::min()
      // causes regression.
      fill_tensor.data = {-10000.0f};
      auto neg_val = model_builder->AddConstantTensor(
          fill_tensor, logits.tensor_desc.GetDataType());
      logits = model_builder->SelectV2(*inputs.mask, logits, neg_val);
    } else {
      logits = model_builder->Add(logits, *inputs.mask);
    }
  }

  ::ml_drift::SoftmaxRuntimeCheckDesc softmax_runtime_check = {
      .end_ch_index = attr.runtime_check.src_end_ch_index,
  };
  auto sfmx_partial = model_builder->SoftmaxReduce(
      logits, softmax_runtime_check, inputs.param_tensor_ptr());

  ::ml_drift::WeightsDescription bmm2_desc = attr.bmm2_weights.desc;
  bmm2_desc.type = logits.tensor_desc.GetDataType();
  const ::ml_drift::GpuModelBuilder::Weights bmm2_external_weights =
      ::ml_drift::CreateExternalWeights(inputs.v, bmm2_desc,
                                        attr.bmm2_weights.weights_shape);
  ::ml_drift::ConvRuntimeCheckDesc bmm2_runtime_check = {
      .src_end_ch_index = attr.runtime_check.src_end_ch_index,
  };

  ABSL_ASSIGN_OR_RETURN(
      auto output,
      model_builder->FullyConnectedExternalWeights(
          logits, bmm2_external_weights, /*biases=*/nullptr, &sfmx_partial,
          bmm2_runtime_check, inputs.param_tensor_ptr()));

  return model_builder->UpdateOutputTensor(output, output_id);
}

}  // namespace

absl::Status BuildSdpaTransposedGpuGraph(
    const std::vector<uint32_t>& input_ids, uint32_t output_id,
    const SdpaTransposedAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder,
    bool allow_single_kernel_implementation, bool request_flash_decoding) {
  ABSL_ASSIGN_OR_RETURN(const SdpaInputs inputs,
                        ExtractSdpaInputs(input_ids, attr, model_builder));

  const SdpaImplementationStrategy strategy =
      SelectSdpaStrategy(inputs.is_decode, allow_single_kernel_implementation,
                         request_flash_decoding);

  switch (strategy) {
    case SdpaImplementationStrategy::kSingleKernelFlashDecode:
      return BuildSingleKernelFlashDecodeSdpaGraph(inputs, output_id,
                                                   model_builder);
    case SdpaImplementationStrategy::kCompositeMultiKernelFallback:
      return BuildCompositeMultiKernelSdpaGraph(inputs, attr, output_id,
                                                model_builder);
  }
}

absl::Status CreateSdpaTransposedFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const SdpaTransposedAttributes& attr =
      std::any_cast<const SdpaTransposedAttributes&>(node.operation.attributes);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSdpaTransposedGpuGraph(input_ids, outputs[0]->id, attr,
                                     model_builder);
}

absl::Status CreateSdpaTransposedFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const SdpaTransposedAttributes& attr =
      std::any_cast<const SdpaTransposedAttributes&>(node.attr);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSdpaTransposedGpuGraph(input_ids, outputs[0]->id, attr,
                                     model_builder);
}

}  // namespace litert::ml_drift
