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

#include "ml_drift_delegate/delegate/composite/litert_op_selector.h"

#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/tensor_handle.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"
#include "ml_drift_delegate/delegate/composite/gated_delta_update_parser.h"
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"
#include "ml_drift_delegate/delegate/composite/short_conv_step_parser.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::BHWC;
using ::ml_drift::CalculationsPrecision;
using ::ml_drift::CreateGpuModelInfo;
using ::ml_drift::DataType;
using ::ml_drift::GpuApi;
using ::ml_drift::GpuInfo;
using ::ml_drift::GpuModel;
using ::ml_drift::GpuModelBuilder;
using ::ml_drift::GpuModelBuilderOptions;
using ::ml_drift::GraphFloat32;
using ::ml_drift::Layout;
using ::ml_drift::Node;
using ::ml_drift::OHWI;
using ::ml_drift::OperationDef;
using ::ml_drift::SliceAttributes;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using ::ml_drift::Value;
using ::ml_drift::ValueId;
using ::ml_drift::WeightsLayout;
using ::testing::HasSubstr;
using ::testing::Not;

GpuInfo GetTestGpuInfo() {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  gpu_info.opencl_info.supports_fp16 = true;
  gpu_info.opencl_info.supports_images = true;
  gpu_info.opencl_info.compute_units_count = 4;
  gpu_info.opencl_info.max_allocation_size = 256 * 1024 * 1024;
  gpu_info.opencl_info.buffer_max_size = 256 * 1024 * 1024;
  gpu_info.opencl_info.image2d_max_width = 16384;
  gpu_info.opencl_info.image2d_max_height = 16384;
  gpu_info.opencl_info.image_buffer_max_size = 65536;
  gpu_info.opencl_info.max_work_group_size_x = 256;
  gpu_info.opencl_info.max_work_group_size_y = 256;
  gpu_info.opencl_info.max_work_group_size_z = 256;
  gpu_info.opencl_info.max_work_group_total_size = 256;
  for (auto type : {DataType::kFloat32, DataType::kFloat16, DataType::kInt32}) {
    gpu_info.opencl_info.supported_images_2d.r_layout.insert(type);
    gpu_info.opencl_info.supported_images_2d.rg_layout.insert(type);
    gpu_info.opencl_info.supported_images_2d.rgb_layout.insert(type);
    gpu_info.opencl_info.supported_images_2d.rgba_layout.insert(type);
  }
  return gpu_info;
}

// Regression test for b/565413009: when a runtime-param tensor (produced by an
// earlier op with TEXTURE_2D storage) is consumed both by a composite op that
// requires BUFFER storage (such as `add_values_to_cache`) and by another op
// that expects the original TEXTURE_2D tensor (such as a channel slice in the
// ring-buffer SDPA path), `ParamTensorToBuffer` must not orphan the original
// tensor by re-pointing its producer. Every consumer's input tensor must remain
// produced by some node in the model.
TEST(LiteRtOpSelectorTest,
     ParamTensorToBufferPreservesProducerForOtherConsumers) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kTexture2D;

  GpuModelBuilderOptions options;
  options.storage = TensorStorageType::kTexture2D;
  GpuModelBuilder builder(gpu_info, options);

  // Graph input and an internal producer that writes `param_tensor` with
  // TEXTURE_2D storage (modeling the in-graph concat that builds `param'`).
  const TensorDescriptor param_tex_desc(
      DataType::kInt32, TensorStorageType::kTexture2D, Layout::kHWC);
  TensorDescriptor raw_param_desc = param_tex_desc;
  raw_param_desc.SetBHWCShape(BHWC(1, 1, 1, 7));
  TensorDescriptor param_desc = param_tex_desc;
  param_desc.SetBHWCShape(BHWC(1, 1, 1, 7));

  auto raw_param = builder.AddTensor(raw_param_desc);
  auto param = builder.AddTensor(param_desc);
  builder.Copy(raw_param, param);

  // Tensors for `add_values_to_cache` (inputs: src_k, src_v, param; outputs:
  // dst_k, dst_v).
  TensorDescriptor kv_desc(DataType::kFloat32, TensorStorageType::kTexture2D,
                           Layout::kHWC);
  kv_desc.SetBHWCShape(BHWC(1, 1, 8, 64));
  auto src_k = builder.AddTensor(kv_desc);
  auto src_v = builder.AddTensor(kv_desc);
  auto dst_k = builder.AddTensor(kv_desc);
  auto dst_v = builder.AddTensor(kv_desc);

  Value v_src_k{src_k.id, {DataType::kFloat32, BHWC(1, 1, 8, 64)}};
  Value v_src_v{src_v.id, {DataType::kFloat32, BHWC(1, 1, 8, 64)}};
  Value v_param{param.id, {DataType::kInt32, BHWC(1, 1, 1, 7)}};
  Value v_dst_k{dst_k.id, {DataType::kFloat32, BHWC(1, 1, 8, 64)}};
  Value v_dst_v{dst_v.id, {DataType::kFloat32, BHWC(1, 1, 8, 64)}};

  OperationDef cache_op_def;
  cache_op_def.src_tensors = {src_k.tensor_desc, src_v.tensor_desc,
                              param.tensor_desc};
  cache_op_def.dst_tensors = {dst_k.tensor_desc, dst_v.tensor_desc};

  GraphFloat32 graph;
  Node* cache_node = graph.NewNode();
  cache_node->operation.type = kAddValuesToCacheType;
  AddValuesToCacheAttributes attr;
  attr.kv_cache_batch_size = 1;
  attr.cache_size = 64;
  attr.head_size = 64;
  cache_node->operation.attributes = attr;

  LiteRtOpSelector selector(&create_info, &gpu_info);
  ASSERT_OK(selector.GPUOperationFromNode(cache_op_def,
                                          {&v_src_k, &v_src_v, &v_param},
                                          {&v_dst_k, &v_dst_v}, *cache_node,
                                          &builder));

  // Second consumer of the original `param` tensor (as TEXTURE_2D), modeling
  // `rest = param'[3:7]` in the ring-buffer SDPA path.
  SliceAttributes slice_attr;
  slice_attr.starts = BHWC(0, 0, 0, 3);
  slice_attr.ends = BHWC(1, 1, 1, 7);
  slice_attr.strides = BHWC(1, 1, 1, 1);
  auto sliced_param = builder.StridedSlice(param, slice_attr);

  GpuModel gpu_model;
  ASSERT_OK(builder.GetGpuModel({raw_param.id, src_k.id, src_v.id},
                                {dst_k.id, dst_v.id, sliced_param.id},
                                &gpu_model));

  // Build the set of tensors that have a valid source (graph input, constant,
  // or produced by a node).
  absl::flat_hash_set<ValueId> defined_tensors;
  for (const auto& [in_id, ref_id] : gpu_model.input_ids_and_refs) {
    defined_tensors.insert(in_id);
  }
  for (const auto& [const_id, desc] : gpu_model.const_tensors) {
    defined_tensors.insert(const_id);
  }
  for (const auto& node : gpu_model.nodes) {
    for (ValueId out_id : node.outputs) {
      defined_tensors.insert(out_id);
    }
  }

  // Every node input must be defined. Before the fix, `param.id` had its
  // producer hijacked to `new_param_tensor.id`, leaving the StridedSlice node
  // reading an unproduced tensor.
  for (const auto& node : gpu_model.nodes) {
    for (ValueId in_id : node.inputs) {
      EXPECT_TRUE(defined_tensors.contains(in_id))
          << "Node '" << node.name << "' reads tensor " << in_id
          << " which has no producer in the GpuModel.";
    }
  }

  // And the `add_values_to_cache` node must receive a BUFFER-storage param.
  bool found_cache_node = false;
  for (const auto& node : gpu_model.nodes) {
    if (node.name == kAddValuesToCacheType) {
      found_cache_node = true;
      ASSERT_EQ(node.inputs.size(), 3);
      const ValueId cache_param_id = node.inputs[2];
      ASSERT_TRUE(gpu_model.tensors.contains(cache_param_id));
      EXPECT_EQ(gpu_model.tensors.at(cache_param_id).GetStorageType(),
                TensorStorageType::kBuffer);
    }
  }
  EXPECT_TRUE(found_cache_node);
}

TEST(LiteRtOpSelectorTest, FusedSdpaCacheUpdateConvertsParamTensorToBuffer) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kTexture2D;

  GpuModelBuilderOptions options;
  options.storage = TensorStorageType::kTexture2D;
  GpuModelBuilder builder(gpu_info, options);

  auto add_tex_tensor = [&](DataType dtype, const BHWC& shape) {
    TensorDescriptor desc(dtype, TensorStorageType::kTexture2D, Layout::kHWC);
    desc.SetBHWCShape(shape);
    return builder.AddTensor(desc);
  };
  auto add_buf_tensor = [&](DataType dtype, const BHWC& shape) {
    TensorDescriptor desc(dtype, TensorStorageType::kBuffer, Layout::kHWC);
    desc.SetBHWCShape(shape);
    return builder.AddTensor(desc);
  };

  const int Hkv = 2, G = 1, T = 4, W = 16, D = 8;
  auto q = add_tex_tensor(DataType::kFloat32, BHWC(1, Hkv, G * T, D));
  auto kc = add_buf_tensor(DataType::kFloat32, BHWC(1, Hkv, W, D));
  auto vc = add_buf_tensor(DataType::kFloat32, BHWC(1, Hkv, D, W));
  auto kn = add_tex_tensor(DataType::kFloat32, BHWC(1, Hkv, T, D));
  auto vn = add_tex_tensor(DataType::kFloat32, BHWC(1, Hkv, D, T));
  auto mask = add_tex_tensor(DataType::kFloat32, BHWC(1, 1, T, W + T));
  auto param = add_tex_tensor(DataType::kInt32, BHWC(1, 1, 1, 7));
  auto attn_out = add_tex_tensor(DataType::kFloat32, BHWC(1, Hkv, G * T, D));
  auto kc_out = add_buf_tensor(DataType::kFloat32, BHWC(1, Hkv, W, D));
  auto vc_out = add_buf_tensor(DataType::kFloat32, BHWC(1, Hkv, D, W));

  Value v_q{q.id, {DataType::kFloat32, BHWC(1, Hkv, G * T, D)}};
  Value v_kc{kc.id, {DataType::kFloat32, BHWC(1, Hkv, W, D)}};
  Value v_vc{vc.id, {DataType::kFloat32, BHWC(1, Hkv, D, W)}};
  Value v_kn{kn.id, {DataType::kFloat32, BHWC(1, Hkv, T, D)}};
  Value v_vn{vn.id, {DataType::kFloat32, BHWC(1, Hkv, D, T)}};
  Value v_mask{mask.id, {DataType::kFloat32, BHWC(1, 1, T, W + T)}};
  Value v_param{param.id, {DataType::kInt32, BHWC(1, 1, 1, 7)}};
  Value v_attn{attn_out.id, {DataType::kFloat32, BHWC(1, Hkv, G * T, D)}};
  Value v_kc_out{kc_out.id, {DataType::kFloat32, BHWC(1, Hkv, W, D)}};
  Value v_vc_out{vc_out.id, {DataType::kFloat32, BHWC(1, Hkv, D, W)}};

  OperationDef op_def;
  op_def.src_tensors = {q.tensor_desc,    kc.tensor_desc, vc.tensor_desc,
                        kn.tensor_desc,   vn.tensor_desc, mask.tensor_desc,
                        param.tensor_desc};
  op_def.dst_tensors = {attn_out.tensor_desc, kc_out.tensor_desc,
                        vc_out.tensor_desc};

  GraphFloat32 graph;
  Node* sdpa_node = graph.NewNode();
  sdpa_node->operation.type = kFusedSdpaCacheUpdateType;
  FusedSdpaCacheUpdateAttributes attr;
  attr.update_cache = true;
  sdpa_node->operation.attributes = attr;

  LiteRtOpSelector selector(&create_info, &gpu_info);
  ASSERT_OK(selector.GPUOperationFromNode(
      op_def, {&v_q, &v_kc, &v_vc, &v_kn, &v_vn, &v_mask, &v_param},
      {&v_attn, &v_kc_out, &v_vc_out}, *sdpa_node, &builder));

  GpuModel gpu_model;
  ASSERT_OK(
      builder.GetGpuModel({q.id, kc.id, vc.id, kn.id, vn.id, mask.id, param.id},
                          {attn_out.id, kc_out.id, vc_out.id}, &gpu_model));

  // Verify that the softmax and ring-cache write ops inside
  // fused_sdpa_cache_update receive a BUFFER-storage param tensor.
  int checked_nodes = 0;
  for (const auto& node : gpu_model.nodes) {
    if (node.name == "fused_sdpa_masked_softmax") {
      ASSERT_EQ(node.inputs.size(), 4);
      const ValueId param_id = node.inputs[3];
      ASSERT_TRUE(gpu_model.tensors.contains(param_id));
      EXPECT_EQ(gpu_model.tensors.at(param_id).GetStorageType(),
                TensorStorageType::kBuffer);
      ++checked_nodes;
    } else if (node.name == "fused_sdpa_ring_cache_write") {
      ASSERT_EQ(node.inputs.size(), 4);
      const ValueId param_id = node.inputs[2];
      ASSERT_TRUE(gpu_model.tensors.contains(param_id));
      EXPECT_EQ(gpu_model.tensors.at(param_id).GetStorageType(),
                TensorStorageType::kBuffer);
      ++checked_nodes;
    }
  }
  EXPECT_EQ(checked_nodes, 2);
}

TEST(LiteRtOpSelectorTest,
     AddValuesToCacheIntermediateOutputsUseBufferForSdpaTransposed) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kTexture2D;

  GpuModelBuilderOptions options;
  options.storage = TensorStorageType::kTexture2D;
  GpuModelBuilder builder(gpu_info, options);

  TensorDescriptor q_desc(DataType::kFloat32, TensorStorageType::kTexture2D,
                          Layout::kHWC);
  q_desc.SetBHWCShape(BHWC(1, 4, 1, 64));
  TensorDescriptor k_desc(DataType::kFloat32, TensorStorageType::kTexture2D,
                          Layout::kHWC);
  k_desc.SetBHWCShape(BHWC(1, 1, 32, 64));
  TensorDescriptor v_in_desc = k_desc;
  TensorDescriptor v_cache_desc(DataType::kFloat32,
                                TensorStorageType::kTexture2D, Layout::kHWC);
  v_cache_desc.SetBHWCShape(BHWC(1, 1, 64, 32));
  TensorDescriptor param_desc(DataType::kInt32, TensorStorageType::kTexture2D,
                              Layout::kHWC);
  param_desc.SetBHWCShape(BHWC(1, 1, 1, 7));

  auto q = builder.AddTensor(q_desc);
  auto src_k = builder.AddTensor(k_desc);
  auto src_v = builder.AddTensor(v_in_desc);
  auto param = builder.AddTensor(param_desc);
  auto cache_k = builder.AddTensor(k_desc);
  auto cache_v = builder.AddTensor(v_cache_desc);
  auto sdpa_out = builder.AddTensor(q_desc);

  Value v_q{q.id, {DataType::kFloat32, BHWC(1, 4, 1, 64)}};
  Value v_src_k{src_k.id, {DataType::kFloat32, BHWC(1, 1, 32, 64)}};
  Value v_src_v{src_v.id, {DataType::kFloat32, BHWC(1, 1, 32, 64)}};
  Value v_param{param.id, {DataType::kInt32, BHWC(1, 1, 1, 7)}};
  Value v_cache_k{cache_k.id, {DataType::kFloat32, BHWC(1, 1, 32, 64)}};
  Value v_cache_v{cache_v.id, {DataType::kFloat32, BHWC(1, 1, 64, 32)}};
  Value v_sdpa_out{sdpa_out.id, {DataType::kFloat32, BHWC(1, 4, 1, 64)}};

  OperationDef cache_op_def;
  cache_op_def.src_tensors = {src_k.tensor_desc, src_v.tensor_desc,
                              param.tensor_desc};
  cache_op_def.dst_tensors = {cache_k.tensor_desc, cache_v.tensor_desc};

  GraphFloat32 graph;
  Node* cache_node = graph.NewNode();
  cache_node->operation.type = kAddValuesToCacheType;
  AddValuesToCacheAttributes cache_attr;
  cache_attr.kv_cache_batch_size = 1;
  cache_attr.cache_size = 32;
  cache_attr.head_size = 64;
  cache_node->operation.attributes = cache_attr;

  LiteRtOpSelector selector(&create_info, &gpu_info);
  ASSERT_OK(selector.GPUOperationFromNode(
      cache_op_def, {&v_src_k, &v_src_v, &v_param}, {&v_cache_k, &v_cache_v},
      *cache_node, &builder));

  Node* sdpa_node = graph.NewNode();
  sdpa_node->operation.type = kSdpaTransposedType;
  SdpaTransposedAttributes sdpa_attr;
  sdpa_attr.from_cache_update = true;
  sdpa_attr.runtime_check.src_end_ch_index = 2;
  sdpa_attr.bmm1_weights.weights_shape = OHWI(32, 1, 1, 64);
  sdpa_attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      DataType::kFloat32, sdpa_attr.bmm1_weights.weights_shape);
  sdpa_attr.bmm1_weights.desc.layout = WeightsLayout::kOSpatialIOGroupO4I4;
  sdpa_attr.bmm2_weights.weights_shape = OHWI(64, 1, 1, 32);
  sdpa_attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      DataType::kFloat32, sdpa_attr.bmm2_weights.weights_shape);
  sdpa_node->operation.attributes = sdpa_attr;

  OperationDef sdpa_op_def;
  sdpa_op_def.src_tensors = {q.tensor_desc, cache_k.tensor_desc,
                             cache_v.tensor_desc, param.tensor_desc};
  sdpa_op_def.dst_tensors = {sdpa_out.tensor_desc};

  ASSERT_OK(selector.GPUOperationFromNode(
      sdpa_op_def, {&v_q, &v_cache_k, &v_cache_v, &v_param}, {&v_sdpa_out},
      *sdpa_node, &builder));

  GpuModel gpu_model;
  ASSERT_OK(builder.GetGpuModel({q.id, src_k.id, src_v.id, param.id},
                                {sdpa_out.id}, &gpu_model));

  bool found_cache_node = false;
  for (const auto& node : gpu_model.nodes) {
    if (node.name == kAddValuesToCacheType) {
      found_cache_node = true;
      ASSERT_EQ(node.outputs.size(), 2);
      for (ValueId out_id : node.outputs) {
        ASSERT_TRUE(gpu_model.tensors.contains(out_id));
        EXPECT_EQ(gpu_model.tensors.at(out_id).GetStorageType(),
                  TensorStorageType::kBuffer);
      }
    }
  }
  EXPECT_TRUE(found_cache_node);
}

TEST(LiteRtOpSelectorTest, ShortConvStepAllOptionCombinationsAndConvLengths) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kTexture2D;

  constexpr int kHiddenSize = 6144;
  for (int filter_size : {2, 3, 4}) {
    const int state_cache_size = filter_size - 1;
    for (bool is_gated : {false, true}) {
      for (bool use_silu : {false, true}) {
        GpuModelBuilderOptions options;
        options.storage = TensorStorageType::kTexture2D;
        GpuModelBuilder builder(gpu_info, options);

        const int in_channels = is_gated ? (3 * kHiddenSize) : kHiddenSize;
        TensorDescriptor in_proj_desc(
            DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWC);
        in_proj_desc.SetBHWCShape(BHWC(1, 1, 1, in_channels));
        TensorDescriptor conv_weight_desc(
            DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWC);
        conv_weight_desc.SetBHWCShape(BHWC(1, 1, filter_size, kHiddenSize));
        TensorDescriptor out_desc(
            DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWC);
        out_desc.SetBHWCShape(BHWC(1, 1, 1, kHiddenSize));
        TensorDescriptor next_state_desc(
            DataType::kFloat32, TensorStorageType::kBuffer, Layout::kHWC);
        next_state_desc.SetBHWCShape(BHWC(1, 1, state_cache_size, kHiddenSize));

        auto in_proj = builder.AddTensor(in_proj_desc);
        auto conv_weight = builder.AddTensor(conv_weight_desc);
        auto out = builder.AddTensor(out_desc);
        auto next_state = builder.AddTensor(next_state_desc);

        Value v_in_proj{
            in_proj.id, {DataType::kFloat32, BHWC(1, 1, 1, in_channels)}};
        Value v_conv_weight{
            conv_weight.id,
            {DataType::kFloat32, BHWC(1, 1, filter_size, kHiddenSize)}};
        Value v_out{out.id, {DataType::kFloat32, BHWC(1, 1, 1, kHiddenSize)}};
        Value v_next_state{
            next_state.id,
            {DataType::kFloat32, BHWC(1, 1, state_cache_size, kHiddenSize)}};

        OperationDef op_def;
        op_def.src_tensors = {in_proj.tensor_desc, conv_weight.tensor_desc};
        op_def.dst_tensors = {out.tensor_desc, next_state.tensor_desc};

        GraphFloat32 graph;
        Node* conv_node = graph.NewNode();
        conv_node->operation.type = kShortConvStepType;
        ShortConvStepAttributes attr;
        attr.conv_L_cache = filter_size;
        attr.is_gated = is_gated;
        attr.use_silu = use_silu;
        conv_node->operation.attributes = attr;

        LiteRtOpSelector selector(&create_info, &gpu_info);
        ASSERT_OK(selector.GPUOperationFromNode(
            op_def, {&v_in_proj, &v_conv_weight}, {&v_out, &v_next_state},
            *conv_node, &builder));

        GpuModel gpu_model;
        ASSERT_OK(builder.GetGpuModel({in_proj.id, conv_weight.id},
                                      {out.id, next_state.id}, &gpu_model));

        ASSERT_EQ(gpu_model.nodes.size(), 1);
        EXPECT_EQ(gpu_model.nodes[0].name, "short_conv_step");
        const std::string& code = gpu_model.nodes[0].gpu_operation->code_;
        EXPECT_THAT(code,
                    HasSubstr("convert_float4(args.state_in_out_buffer["));
        EXPECT_THAT(
            code,
            HasSubstr("acc0 += wt" + std::to_string(filter_size - 1) +
                      " * bx0;"));
        if (is_gated) {
          EXPECT_THAT(code, HasSubstr("S + 2 * args.num_slices"));
          EXPECT_THAT(code, HasSubstr("S + args.num_slices"));
        } else {
          EXPECT_THAT(code, Not(HasSubstr("args.num_slices")));
        }
        if (use_silu) {
          EXPECT_THAT(code, HasSubstr("exp(-acc0)"));
        } else {
          EXPECT_THAT(code, Not(HasSubstr("exp(-acc0)")));
        }
        EXPECT_THAT(code, HasSubstr("args.state_in_out_buffer[(((S) * "
                                    "args.state_in_out_height + (0)) * "
                                    "args.state_in_out_width + (0))] ="));
      }
    }
  }
}

TEST(LiteRtOpSelectorTest, ShortConvStepRejectsChannelsFirstStateOrWeight) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kTexture2D;

  constexpr int kHiddenSize = 64;
  constexpr int kFilterSize = 3;
  constexpr int kStateCacheSize = kFilterSize - 1;
  const BHWC channels_last_weight(1, 1, kFilterSize, kHiddenSize);
  const BHWC channels_last_state(1, 1, kStateCacheSize, kHiddenSize);
  // Channels-first shapes of older exports: weight [H, 1, L] and state
  // [1, H, L - 1].
  const BHWC channels_first_weight(kHiddenSize, 1, 1, kFilterSize);
  const BHWC channels_first_state(1, 1, kHiddenSize, kStateCacheSize);
  for (const auto& [weight_shape, state_shape] :
       {std::pair(channels_first_weight, channels_last_state),
        std::pair(channels_last_weight, channels_first_state)}) {
    GpuModelBuilderOptions options;
    options.storage = TensorStorageType::kTexture2D;
    GpuModelBuilder builder(gpu_info, options);

    TensorDescriptor in_proj_desc(DataType::kFloat32,
                                  TensorStorageType::kTexture2D, Layout::kHWC);
    in_proj_desc.SetBHWCShape(BHWC(1, 1, 1, kHiddenSize));
    TensorDescriptor conv_weight_desc(
        DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWC);
    conv_weight_desc.SetBHWCShape(weight_shape);
    TensorDescriptor out_desc(DataType::kFloat32, TensorStorageType::kTexture2D,
                              Layout::kHWC);
    out_desc.SetBHWCShape(BHWC(1, 1, 1, kHiddenSize));
    TensorDescriptor next_state_desc(DataType::kFloat32,
                                     TensorStorageType::kBuffer, Layout::kHWC);
    next_state_desc.SetBHWCShape(state_shape);

    auto in_proj = builder.AddTensor(in_proj_desc);
    auto conv_weight = builder.AddTensor(conv_weight_desc);
    auto out = builder.AddTensor(out_desc);
    auto next_state = builder.AddTensor(next_state_desc);

    Value v_in_proj{in_proj.id,
                    {DataType::kFloat32, BHWC(1, 1, 1, kHiddenSize)}};
    Value v_conv_weight{conv_weight.id, {DataType::kFloat32, weight_shape}};
    Value v_out{out.id, {DataType::kFloat32, BHWC(1, 1, 1, kHiddenSize)}};
    Value v_next_state{next_state.id, {DataType::kFloat32, state_shape}};

    OperationDef op_def;
    op_def.src_tensors = {in_proj.tensor_desc, conv_weight.tensor_desc};
    op_def.dst_tensors = {out.tensor_desc, next_state.tensor_desc};

    GraphFloat32 graph;
    Node* conv_node = graph.NewNode();
    conv_node->operation.type = kShortConvStepType;
    ShortConvStepAttributes attr;
    attr.conv_L_cache = kFilterSize;
    attr.is_gated = false;
    conv_node->operation.attributes = attr;

    LiteRtOpSelector selector(&create_info, &gpu_info);
    EXPECT_FALSE(selector
                     .GPUOperationFromNode(op_def, {&v_in_proj, &v_conv_weight},
                                           {&v_out, &v_next_state}, *conv_node,
                                           &builder)
                     .ok());
  }
}

TEST(LiteRtOpSelectorTest, ScaledDotProductAttentionUsesFlashSdpaKernel) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuModelBuilderOptions options;
  options.storage = TensorStorageType::kBuffer;
  GpuModelBuilder builder(gpu_info, options);

  TensorDescriptor desc(DataType::kFloat16, TensorStorageType::kBuffer,
                        Layout::kHWC);
  desc.SetBHWCShape(BHWC(1, 10080, 12, 64));
  auto q = builder.AddTensor(desc);
  auto k = builder.AddTensor(desc);
  auto v = builder.AddTensor(desc);
  auto out = builder.AddTensor(desc);

  Value v_q{q.id, {DataType::kFloat16, BHWC(1, 10080, 12, 64)}};
  Value v_k{k.id, {DataType::kFloat16, BHWC(1, 10080, 12, 64)}};
  Value v_v{v.id, {DataType::kFloat16, BHWC(1, 10080, 12, 64)}};
  Value v_out{out.id, {DataType::kFloat16, BHWC(1, 10080, 12, 64)}};

  OperationDef op_def;
  op_def.src_tensors = {q.tensor_desc, k.tensor_desc, v.tensor_desc};
  op_def.dst_tensors = {out.tensor_desc};

  GraphFloat32 graph;
  Node* sdpa_node = graph.NewNode();
  sdpa_node->operation.type =
      ToString(::ml_drift::OperationType::kScaledDotProductAttention);
  ::ml_drift::ScaledDotProductAttentionAttributes attr;
  attr.scale = 0.125f;
  sdpa_node->operation.attributes = attr;

  LiteRtOpSelector selector(&create_info, &gpu_info);
  ASSERT_OK(selector.GPUOperationFromNode(op_def, {&v_q, &v_k, &v_v}, {&v_out},
                                          *sdpa_node, &builder));

  GpuModel gpu_model;
  ASSERT_OK(builder.GetGpuModel({q.id, k.id, v.id}, {out.id}, &gpu_model));
  ASSERT_EQ(gpu_model.nodes.size(), 5);
  EXPECT_EQ(gpu_model.nodes[3].name, "flash_sdpa");
}

TEST(LiteRtOpSelectorTest,
     ScaledDotProductAttentionFallsBackWhenUnsupported) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuModelBuilderOptions options;
  options.storage = TensorStorageType::kBuffer;
  GpuModelBuilder builder(gpu_info, options);

  TensorDescriptor desc(DataType::kFloat16, TensorStorageType::kBuffer,
                        Layout::kHWC);
  desc.SetBHWCShape(BHWC(1, 64, 8, 256));
  auto q = builder.AddTensor(desc);
  auto k = builder.AddTensor(desc);
  auto v = builder.AddTensor(desc);
  auto out = builder.AddTensor(desc);

  Value v_q{q.id, {DataType::kFloat16, BHWC(1, 64, 8, 256)}};
  Value v_k{k.id, {DataType::kFloat16, BHWC(1, 64, 8, 256)}};
  Value v_v{v.id, {DataType::kFloat16, BHWC(1, 64, 8, 256)}};
  Value v_out{out.id, {DataType::kFloat16, BHWC(1, 64, 8, 256)}};

  OperationDef op_def;
  op_def.src_tensors = {q.tensor_desc, k.tensor_desc, v.tensor_desc};
  op_def.dst_tensors = {out.tensor_desc};

  GraphFloat32 graph;
  Node* sdpa_node = graph.NewNode();
  sdpa_node->operation.type =
      ToString(::ml_drift::OperationType::kScaledDotProductAttention);
  ::ml_drift::ScaledDotProductAttentionAttributes attr;
  attr.scale = 0.0625f;
  sdpa_node->operation.attributes = attr;

  LiteRtOpSelector selector(&create_info, &gpu_info);
  ASSERT_OK(selector.GPUOperationFromNode(op_def, {&v_q, &v_k, &v_v}, {&v_out},
                                          *sdpa_node, &builder));

  GpuModel gpu_model;
  ASSERT_OK(builder.GetGpuModel({q.id, k.id, v.id}, {out.id}, &gpu_model));
  EXPECT_FALSE(gpu_model.nodes.empty());
  for (const auto& node : gpu_model.nodes) {
    EXPECT_NE(node.name, "flash_sdpa");
  }
}
TEST(LiteRtOpSelectorTest,
     FullyConnectedInt4SymmetricExternalWeightsOmitsZeroPoint) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuModelBuilderOptions options;
  options.storage = TensorStorageType::kBuffer;
  GpuModelBuilder builder(gpu_info, options);

  TensorDescriptor act_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                            Layout::kHWC);
  act_desc.SetBHWCShape(BHWC(1, 1, 1, 64));
  auto src = builder.AddTensor(act_desc);

  TensorDescriptor out_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                            Layout::kHWC);
  out_desc.SetBHWCShape(BHWC(1, 1, 1, 16));
  auto out = builder.AddTensor(out_desc);

  TensorDescriptor weights_desc(DataType::kUint32, TensorStorageType::kBuffer,
                                Layout::kHWC);
  weights_desc.SetBHWCShape(BHWC(1, 1, 4, 32));
  auto weights = builder.AddTensor(weights_desc);

  TensorDescriptor scale_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                              Layout::kHWC);
  scale_desc.SetBHWCShape(BHWC(1, 1, 4, 8));
  auto scale = builder.AddTensor(scale_desc);

  Value v_src{src.id, {DataType::kFloat16, BHWC(1, 1, 1, 64)}};
  Value v_weights{weights.id, {DataType::kUint32, BHWC(1, 1, 4, 32)}};
  Value v_scale{scale.id, {DataType::kFloat16, BHWC(1, 1, 4, 8)}};
  Value v_out{out.id, {DataType::kFloat16, BHWC(1, 1, 1, 16)}};

  OperationDef op_def;
  op_def.src_tensors = {src.tensor_desc, weights.tensor_desc,
                        scale.tensor_desc};
  op_def.dst_tensors = {out.tensor_desc};

  GraphFloat32 graph;
  Node* fc_node = graph.NewNode();
  fc_node->operation.type =
      ToString(::ml_drift::OperationType::kFullyConnectedInt4);
  ::ml_drift::FullyConnectedInt4Attributes attr;
  ::ml_drift::Tensor<OHWI, DataType::kInt4> w_tensor;
  w_tensor.shape = OHWI(16, 1, 1, 64);
  attr.weights = w_tensor;
  attr.scale.shape = OHWI(16, 1, 1, 2);
  attr.zero_point.shape = OHWI(0, 0, 0, 0);
  fc_node->operation.attributes = attr;

  LiteRtOpSelector selector(&create_info, &gpu_info);
  ASSERT_OK(selector.GPUOperationFromNode(
      op_def, {&v_src, &v_weights, &v_scale}, {&v_out}, *fc_node, &builder));

  GpuModel gpu_model;
  ASSERT_OK(builder.GetGpuModel({src.id, weights.id, scale.id}, {out.id},
                                &gpu_model));
  ASSERT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].inputs.size(), 3);
}

GpuInfo GetApple10Metal4GpuInfo() {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kMetal;
  gpu_info.vendor = ::ml_drift::GpuVendor::kApple;
  gpu_info.apple_info = ::ml_drift::AppleInfo("Apple M5 Max");
  gpu_info.apple_info.gpu_family = ::ml_drift::AppleInfo::Family::kApple10;
  gpu_info.apple_info.SetComputeUnits(40);
  gpu_info.supported_wave_sizes = {32};
  gpu_info.metal_info.language_version =
      ::ml_drift::MetalLanguageVersion::kMetal4_0;
  gpu_info.metal_info.max_work_group_size_x = 1024;
  gpu_info.metal_info.max_work_group_size_y = 1024;
  gpu_info.metal_info.max_work_group_size_z = 1024;
  gpu_info.metal_info.buffer_max_size = 256 * 1024 * 1024;
  return gpu_info;
}

// Adds a `FullyConnectedExternalWeights` projection (`out_proj`) consuming
// `src`. When `total_spatial_size > GetRecommendedMaxTotalSpatialSize` (16) and
// `gpu_info` satisfies `SupportsConvAppleMPP` (`Apple10` + `Metal 4.0` with
// `DivideRoundUp(c, 4) % 8 == 0`), `GpuModelBuilder` inserts a linkable
// `ToDHWBCC4` (`to_dhwbcc4`) layout copy immediately after `src`, which
// `LinkNodes` fuses into the preceding composite kernel's `.Write(...)`
// expression via `ResolveLinking`.
absl::StatusOr<::ml_drift::TensorHandle> AppendOutProj(
    GpuModelBuilder* builder, const ::ml_drift::TensorHandle& src) {
  const int in_ch = src.tensor_desc.GetBHWCShape().c;
  const OHWI w_shape(32, 1, 1, in_ch);
  const auto w_desc =
      builder->GetFullyConnectedWeightsDesc(DataType::kFloat16, w_shape);
  TensorDescriptor w_td(w_desc.type, TensorStorageType::kBuffer, Layout::kHWC);
  w_td.SetBHWCShape(BHWC(1, 1, 1, w_shape.DimensionsProduct()));
  GpuModelBuilder::Weights weights;
  weights.weights = builder->AddTensor(w_td);
  weights.desc = w_desc;
  weights.shape = w_shape;
  return builder->FullyConnectedExternalWeights(src, weights);
}

// Regression test for Apple10 + Metal 4.0 (A19 Pro / M5):
// `FullyConnectedExternalWeights` inserts a linkable `ToDHWBCC4` layout copy
// immediately after `sdpa_transposed` when `SupportsConvAppleMPP` holds, which
// `LinkNodes` fuses into the SDPA kernel's `args.dst.Write(...)`. Passing an
// inline `ucl::Convert<...>(...)` expression as the first argument to `Write`
// previously caused `ResolveLinking` to emit invalid Metal
// (`half4 half4(res)_final;`).
TEST(LiteRtOpSelectorTest,
     Apple10Metal4SdpaTransposedDecodePostOpLinksValidIdentifier) {
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kBuffer;

  auto build_sdpa_model = [&](const GpuInfo& gpu_info, int head_dim,
                              GpuModel* gpu_model) {
    GpuModelBuilderOptions options;
    options.storage = TensorStorageType::kBuffer;
    GpuModelBuilder builder(gpu_info, options);

    auto add_buf = [&](DataType dtype, const BHWC& shape) {
      TensorDescriptor desc(dtype, TensorStorageType::kBuffer, Layout::kHWC);
      desc.SetBHWCShape(shape);
      return builder.AddTensor(desc);
    };

    const int q_heads = 24;
    const int kv_heads = 4;
    const int cache_size = 2048;
    auto q = add_buf(DataType::kFloat16, BHWC(1, q_heads, 1, head_dim));
    auto k =
        add_buf(DataType::kFloat16, BHWC(1, kv_heads, cache_size, head_dim));
    auto v =
        add_buf(DataType::kFloat16, BHWC(1, kv_heads, head_dim, cache_size));
    auto param = add_buf(DataType::kInt32, BHWC(1, 1, 1, 4));
    auto sdpa_out = add_buf(DataType::kFloat16, BHWC(1, q_heads, 1, head_dim));

    Value v_q{q.id, {DataType::kFloat16, BHWC(1, q_heads, 1, head_dim)}};
    Value v_k{k.id,
              {DataType::kFloat16, BHWC(1, kv_heads, cache_size, head_dim)}};
    Value v_v{v.id,
              {DataType::kFloat16, BHWC(1, kv_heads, head_dim, cache_size)}};
    Value v_param{param.id, {DataType::kInt32, BHWC(1, 1, 1, 4)}};
    Value v_sdpa_out{sdpa_out.id,
                     {DataType::kFloat16, BHWC(1, q_heads, 1, head_dim)}};

    OperationDef op_def;
    op_def.src_tensors = {q.tensor_desc, k.tensor_desc, v.tensor_desc,
                          param.tensor_desc};
    op_def.dst_tensors = {sdpa_out.tensor_desc};

    GraphFloat32 graph;
    Node* sdpa_node = graph.NewNode();
    sdpa_node->operation.type = kSdpaTransposedType;
    SdpaTransposedAttributes attr;
    attr.from_cache_update = true;
    attr.is_causal = true;
    attr.is_prefill = false;
    attr.runtime_check.src_end_ch_index = 0;
    sdpa_node->operation.attributes = attr;

    LiteRtOpSelector selector(&create_info, &gpu_info);
    ASSERT_OK(
        selector.GPUOperationFromNode(op_def, {&v_q, &v_k, &v_v, &v_param},
                                      {&v_sdpa_out}, *sdpa_node, &builder));
    ASSERT_OK_AND_ASSIGN(auto proj_out, AppendOutProj(&builder, sdpa_out));
    ASSERT_OK(builder.GetGpuModel({q.id, k.id, v.id, param.id}, {proj_out.id},
                                  gpu_model));
  };

  const GpuInfo apple10_info = GetApple10Metal4GpuInfo();
  for (int head_dim : {256, 128}) {
    GpuModel gpu_model;
    build_sdpa_model(apple10_info, head_dim, &gpu_model);
    bool found_linked_sdpa = false;
    for (auto& node : gpu_model.nodes) {
      if (node.name == "flash_decode_sdpa_combine -> to_dhwbcc4" ||
          node.name == "flash_decode_sdpa -> to_dhwbcc4") {
        found_linked_sdpa = true;
        ASSERT_OK(node.gpu_operation->AssembleCode(apple10_info));
        EXPECT_THAT(node.gpu_operation->code_, HasSubstr("res_value_final"));
        EXPECT_THAT(node.gpu_operation->code_, Not(HasSubstr(")_final")));
      }
    }
    EXPECT_TRUE(found_linked_sdpa) << "head_dim=" << head_dim;
  }

  // Confirm that on pre-Apple10 / pre-Metal 4.0 GPUs, `SupportsConvAppleMPP` is
  // false and `to_dhwbcc4` is not emitted.
  GpuInfo apple9_info = apple10_info;
  apple9_info.apple_info.gpu_family = ::ml_drift::AppleInfo::Family::kApple9;
  apple9_info.metal_info.language_version =
      ::ml_drift::MetalLanguageVersion::kMetal3_2;
  GpuModel apple9_model;
  build_sdpa_model(apple9_info, 128, &apple9_model);
  for (const auto& node : apple9_model.nodes) {
    EXPECT_THAT(node.name, Not(HasSubstr("to_dhwbcc4")));
  }
}

TEST(LiteRtOpSelectorTest,
     Apple10Metal4GatedDeltaBlockPostOpLinksValidIdentifier) {
  const GpuInfo gpu_info = GetApple10Metal4GpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kBuffer;

  for (int seq_len : {1, 32}) {
    GpuModelBuilderOptions options;
    options.storage = TensorStorageType::kBuffer;
    GpuModelBuilder builder(gpu_info, options);

    auto add_buf = [&](DataType dtype, const BHWC& shape) {
      TensorDescriptor desc(dtype, TensorStorageType::kBuffer, Layout::kHWC);
      desc.SetBHWCShape(shape);
      return builder.AddTensor(desc);
    };

    const int num_k_heads = 16;
    const int num_v_heads = 16;
    const int head_k_dim = 128;
    const int head_v_dim = 128;
    const int conv_dim =
        2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
    const int zba_dim = num_v_heads * head_v_dim + 2 * num_v_heads;
    const int value_dim = num_v_heads * head_v_dim;

    auto conv_out = add_buf(DataType::kFloat16, BHWC(1, 1, seq_len, conv_dim));
    auto zba = add_buf(DataType::kFloat16, BHWC(1, 1, seq_len, zba_dim));
    auto a_log = add_buf(DataType::kFloat32, BHWC(1, 1, num_v_heads, 1));
    auto dt_bias = add_buf(DataType::kFloat32, BHWC(1, 1, num_v_heads, 1));
    auto norm_w = add_buf(DataType::kFloat32, BHWC(1, 1, head_v_dim, 1));
    auto gdn_out = add_buf(DataType::kFloat16, BHWC(1, 1, seq_len, value_dim));
    auto state_out = add_buf(DataType::kFloat32,
                             BHWC(1, num_v_heads, head_k_dim, head_v_dim));

    Value v_conv{conv_out.id,
                 {DataType::kFloat16, BHWC(1, 1, seq_len, conv_dim)}};
    Value v_zba{zba.id, {DataType::kFloat16, BHWC(1, 1, seq_len, zba_dim)}};
    Value v_a_log{a_log.id, {DataType::kFloat32, BHWC(1, 1, num_v_heads, 1)}};
    Value v_dt_bias{dt_bias.id,
                    {DataType::kFloat32, BHWC(1, 1, num_v_heads, 1)}};
    Value v_norm_w{norm_w.id, {DataType::kFloat32, BHWC(1, 1, head_v_dim, 1)}};
    Value v_gdn_out{gdn_out.id,
                    {DataType::kFloat16, BHWC(1, 1, seq_len, value_dim)}};
    Value v_state_out{
        state_out.id,
        {DataType::kFloat32, BHWC(1, num_v_heads, head_k_dim, head_v_dim)}};

    OperationDef op_def;
    op_def.src_tensors = {conv_out.tensor_desc, zba.tensor_desc,
                          a_log.tensor_desc, dt_bias.tensor_desc,
                          norm_w.tensor_desc};
    op_def.dst_tensors = {gdn_out.tensor_desc, state_out.tensor_desc};

    GraphFloat32 graph;
    Node* gdn_node = graph.NewNode();
    gdn_node->operation.type = kGatedDeltaUpdateType;
    GatedDeltaUpdateAttributes attr;
    GatedDeltaNetAttributes block;
    block.num_k_heads = num_k_heads;
    block.num_v_heads = num_v_heads;
    block.head_k_dim = head_k_dim;
    block.head_v_dim = head_v_dim;
    attr.block = block;
    gdn_node->operation.attributes = attr;

    LiteRtOpSelector selector(&create_info, &gpu_info);
    ASSERT_OK(selector.GPUOperationFromNode(
        op_def, {&v_conv, &v_zba, &v_a_log, &v_dt_bias, &v_norm_w},
        {&v_gdn_out, &v_state_out}, *gdn_node, &builder));
    ::ml_drift::TensorHandle final_out;
    std::string expected_suffix;
    if (seq_len > 16) {
      // Prefill (`total_spatial_size > 16`) triggers Apple10 + Metal 4.0
      // `SupportsConvAppleMPP` -> `to_dhwbcc4` post-op fusion.
      ASSERT_OK_AND_ASSIGN(final_out, AppendOutProj(&builder, gdn_out));
      expected_suffix = " -> to_dhwbcc4";
    } else {
      // Single-token decode (`seq_len == 1`) links any downstream elementwise
      // post-op into the decode specialization's `.Write(...)`.
      final_out = add_buf(DataType::kFloat16, BHWC(1, 1, seq_len, value_dim));
      builder.Copy(gdn_out, final_out);
      expected_suffix = " -> copy";
    }

    GpuModel gpu_model;
    ASSERT_OK(builder.GetGpuModel(
        {conv_out.id, zba.id, a_log.id, dt_bias.id, norm_w.id},
        {final_out.id, state_out.id}, &gpu_model));
    bool found_linked_gdn = false;
    for (auto& node : gpu_model.nodes) {
      if (node.name == "gated_delta_update" + expected_suffix ||
          node.name == "gated_delta_update_norm" + expected_suffix) {
        found_linked_gdn = true;
        ASSERT_OK(node.gpu_operation->AssembleCode(gpu_info));
        EXPECT_THAT(node.gpu_operation->code_,
                    HasSubstr("gated_out_act_final"));
        EXPECT_THAT(node.gpu_operation->code_, Not(HasSubstr(")_final")));
      }
    }
    EXPECT_TRUE(found_linked_gdn) << "seq_len=" << seq_len;
  }
}

TEST(LiteRtOpSelectorTest,
     Apple10Metal4ShortConvStepPostOpLinksValidIdentifier) {
  const GpuInfo gpu_info = GetApple10Metal4GpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuModelBuilderOptions options;
  options.storage = TensorStorageType::kBuffer;
  GpuModelBuilder builder(gpu_info, options);

  auto add_buf = [&](DataType dtype, const BHWC& shape) {
    TensorDescriptor desc(dtype, TensorStorageType::kBuffer, Layout::kHWC);
    desc.SetBHWCShape(shape);
    return builder.AddTensor(desc);
  };

  const int seq_len = 32;
  const int filter_size = 4;
  const int hidden_size = 32;
  auto in_proj = add_buf(DataType::kFloat16, BHWC(1, 1, seq_len, hidden_size));
  auto conv_weight =
      add_buf(DataType::kFloat16, BHWC(1, 1, filter_size, hidden_size));
  auto conv_state =
      add_buf(DataType::kFloat16, BHWC(1, 1, filter_size - 1, hidden_size));
  auto conv_out = add_buf(DataType::kFloat16, BHWC(1, 1, seq_len, hidden_size));
  auto next_state =
      add_buf(DataType::kFloat16, BHWC(1, 1, filter_size - 1, hidden_size));

  Value v_in{in_proj.id,
             {DataType::kFloat16, BHWC(1, 1, seq_len, hidden_size)}};
  Value v_w{conv_weight.id,
            {DataType::kFloat16, BHWC(1, 1, filter_size, hidden_size)}};
  Value v_st{conv_state.id,
             {DataType::kFloat16, BHWC(1, 1, filter_size - 1, hidden_size)}};
  Value v_out{conv_out.id,
              {DataType::kFloat16, BHWC(1, 1, seq_len, hidden_size)}};
  Value v_nst{next_state.id,
              {DataType::kFloat16, BHWC(1, 1, filter_size - 1, hidden_size)}};

  OperationDef op_def;
  op_def.src_tensors = {in_proj.tensor_desc, conv_weight.tensor_desc,
                        conv_state.tensor_desc};
  op_def.dst_tensors = {conv_out.tensor_desc, next_state.tensor_desc};

  GraphFloat32 graph;
  Node* conv_node = graph.NewNode();
  conv_node->operation.type = kShortConvStepType;
  ShortConvStepAttributes attr;
  attr.conv_L_cache = filter_size;
  attr.is_gated = false;
  conv_node->operation.attributes = attr;

  LiteRtOpSelector selector(&create_info, &gpu_info);
  ASSERT_OK(selector.GPUOperationFromNode(
      op_def, {&v_in, &v_w, &v_st}, {&v_out, &v_nst}, *conv_node, &builder));
  ASSERT_OK_AND_ASSIGN(auto proj_out, AppendOutProj(&builder, conv_out));

  GpuModel gpu_model;
  ASSERT_OK(builder.GetGpuModel({in_proj.id, conv_weight.id, conv_state.id},
                                {proj_out.id, next_state.id}, &gpu_model));
  bool found_linked_conv = false;
  for (auto& node : gpu_model.nodes) {
    if (node.name == "short_conv_step -> to_dhwbcc4") {
      found_linked_conv = true;
      ASSERT_OK(node.gpu_operation->AssembleCode(gpu_info));
      EXPECT_THAT(node.gpu_operation->code_, HasSubstr("out_val_final"));
      EXPECT_THAT(node.gpu_operation->code_, HasSubstr("out_val0_final"));
      EXPECT_THAT(node.gpu_operation->code_, HasSubstr("out_val1_final"));
      EXPECT_THAT(node.gpu_operation->code_, HasSubstr("out_val2_final"));
      EXPECT_THAT(node.gpu_operation->code_, Not(HasSubstr(")_final")));
    }
  }
  EXPECT_TRUE(found_linked_conv);
}

TEST(LiteRtOpSelectorTest,
     Apple10Metal4FusedSdpaCacheUpdatePrefillLinksValidIdentifier) {
  const GpuInfo gpu_info = GetApple10Metal4GpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuModelBuilderOptions options;
  options.storage = TensorStorageType::kBuffer;
  GpuModelBuilder builder(gpu_info, options);

  auto add_buf = [&](DataType dtype, const BHWC& shape) {
    TensorDescriptor desc(dtype, TensorStorageType::kBuffer, Layout::kHWC);
    desc.SetBHWCShape(shape);
    return builder.AddTensor(desc);
  };

  const int Hkv = 2, G = 1, T = 32, W = 64, D = 64;
  auto q = add_buf(DataType::kFloat16, BHWC(1, Hkv, G * T, D));
  auto kc = add_buf(DataType::kFloat16, BHWC(1, Hkv, W, D));
  auto vc = add_buf(DataType::kFloat16, BHWC(1, Hkv, D, W));
  auto kn = add_buf(DataType::kFloat16, BHWC(1, Hkv, T, D));
  auto vn = add_buf(DataType::kFloat16, BHWC(1, Hkv, D, T));
  auto mask = add_buf(DataType::kFloat16, BHWC(1, 1, T, W + T));
  auto param = add_buf(DataType::kInt32, BHWC(1, 1, 1, 7));
  auto attn_out = add_buf(DataType::kFloat16, BHWC(1, Hkv, G * T, D));
  auto kc_out = add_buf(DataType::kFloat16, BHWC(1, Hkv, W, D));
  auto vc_out = add_buf(DataType::kFloat16, BHWC(1, Hkv, D, W));

  Value v_q{q.id, {DataType::kFloat16, BHWC(1, Hkv, G * T, D)}};
  Value v_kc{kc.id, {DataType::kFloat16, BHWC(1, Hkv, W, D)}};
  Value v_vc{vc.id, {DataType::kFloat16, BHWC(1, Hkv, D, W)}};
  Value v_kn{kn.id, {DataType::kFloat16, BHWC(1, Hkv, T, D)}};
  Value v_vn{vn.id, {DataType::kFloat16, BHWC(1, Hkv, D, T)}};
  Value v_mask{mask.id, {DataType::kFloat16, BHWC(1, 1, T, W + T)}};
  Value v_param{param.id, {DataType::kInt32, BHWC(1, 1, 1, 7)}};
  Value v_attn{attn_out.id, {DataType::kFloat16, BHWC(1, Hkv, G * T, D)}};
  Value v_kc_out{kc_out.id, {DataType::kFloat16, BHWC(1, Hkv, W, D)}};
  Value v_vc_out{vc_out.id, {DataType::kFloat16, BHWC(1, Hkv, D, W)}};

  OperationDef op_def;
  op_def.src_tensors = {q.tensor_desc,    kc.tensor_desc, vc.tensor_desc,
                        kn.tensor_desc,   vn.tensor_desc, mask.tensor_desc,
                        param.tensor_desc};
  op_def.dst_tensors = {attn_out.tensor_desc, kc_out.tensor_desc,
                        vc_out.tensor_desc};

  GraphFloat32 graph;
  Node* sdpa_node = graph.NewNode();
  sdpa_node->operation.type = kFusedSdpaCacheUpdateType;
  FusedSdpaCacheUpdateAttributes attr;
  attr.update_cache = true;
  sdpa_node->operation.attributes = attr;

  LiteRtOpSelector selector(&create_info, &gpu_info);
  ASSERT_OK(selector.GPUOperationFromNode(
      op_def, {&v_q, &v_kc, &v_vc, &v_kn, &v_vn, &v_mask, &v_param},
      {&v_attn, &v_kc_out, &v_vc_out}, *sdpa_node, &builder));

  GpuModel gpu_model;
  ASSERT_OK(
      builder.GetGpuModel({q.id, kc.id, vc.id, kn.id, vn.id, mask.id, param.id},
                          {attn_out.id, kc_out.id, vc_out.id}, &gpu_model));
  bool found_linked_softmax = false;
  for (auto& node : gpu_model.nodes) {
    if (absl::StrContains(node.name, "fused_sdpa_masked_softmax ->")) {
      found_linked_softmax = true;
      ASSERT_OK(node.gpu_operation->AssembleCode(gpu_info));
      EXPECT_THAT(node.gpu_operation->code_, HasSubstr("prob_val_final"));
      EXPECT_THAT(node.gpu_operation->code_, HasSubstr("zero_prob_final"));
      EXPECT_THAT(node.gpu_operation->code_, Not(HasSubstr(")_final")));
    }
  }
  EXPECT_TRUE(found_linked_softmax);
}
}  // namespace
}  // namespace litert::ml_drift
