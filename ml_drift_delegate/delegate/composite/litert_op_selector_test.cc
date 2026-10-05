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

#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
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
using ::ml_drift::OperationDef;
using ::ml_drift::SliceAttributes;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using ::ml_drift::Value;
using ::ml_drift::ValueId;
using ::testing::HasSubstr;

GpuInfo GetTestGpuInfo() {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  gpu_info.opencl_info.supports_fp16 = true;
  gpu_info.opencl_info.supports_images = true;
  gpu_info.opencl_info.max_allocation_size = 256 * 1024 * 1024;
  gpu_info.opencl_info.buffer_max_size = 256 * 1024 * 1024;
  gpu_info.opencl_info.image2d_max_width = 16384;
  gpu_info.opencl_info.image2d_max_height = 16384;
  gpu_info.opencl_info.image_buffer_max_size = 65536;
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

TEST(LiteRtOpSelectorTest, ShortConvStepUngatedSiluConv4) {
  const GpuInfo gpu_info = GetTestGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kTexture2D;

  GpuModelBuilderOptions options;
  options.storage = TensorStorageType::kTexture2D;
  GpuModelBuilder builder(gpu_info, options);

  constexpr int kHiddenSize = 6144;
  constexpr int kFilterSize = 4;
  constexpr int kStateCacheSize = kFilterSize - 1;

  TensorDescriptor in_proj_desc(DataType::kFloat32,
                                TensorStorageType::kTexture2D, Layout::kHWC);
  in_proj_desc.SetBHWCShape(BHWC(1, 1, 1, kHiddenSize));
  TensorDescriptor conv_state_desc(DataType::kFloat32,
                                   TensorStorageType::kTexture2D, Layout::kHWC);
  conv_state_desc.SetBHWCShape(BHWC(1, 1, kHiddenSize, kStateCacheSize));
  TensorDescriptor conv_weight_desc(
      DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWC);
  conv_weight_desc.SetBHWCShape(BHWC(1, 1, kHiddenSize, kFilterSize));
  TensorDescriptor out_desc(DataType::kFloat32, TensorStorageType::kTexture2D,
                            Layout::kHWC);
  out_desc.SetBHWCShape(BHWC(1, 1, 1, kHiddenSize));
  TensorDescriptor next_state_desc(DataType::kFloat32,
                                   TensorStorageType::kTexture2D, Layout::kHWC);
  next_state_desc.SetBHWCShape(BHWC(1, 1, kHiddenSize, kStateCacheSize));

  auto in_proj = builder.AddTensor(in_proj_desc);
  auto conv_state = builder.AddTensor(conv_state_desc);
  auto conv_weight = builder.AddTensor(conv_weight_desc);
  auto out = builder.AddTensor(out_desc);
  auto next_state = builder.AddTensor(next_state_desc);

  Value v_in_proj{in_proj.id, {DataType::kFloat32, BHWC(1, 1, 1, kHiddenSize)}};
  Value v_conv_state{
      conv_state.id,
      {DataType::kFloat32, BHWC(1, 1, kHiddenSize, kStateCacheSize)}};
  Value v_conv_weight{
      conv_weight.id,
      {DataType::kFloat32, BHWC(1, 1, kHiddenSize, kFilterSize)}};
  Value v_out{out.id, {DataType::kFloat32, BHWC(1, 1, 1, kHiddenSize)}};
  Value v_next_state{
      next_state.id,
      {DataType::kFloat32, BHWC(1, 1, kHiddenSize, kStateCacheSize)}};

  OperationDef op_def;
  op_def.src_tensors = {in_proj.tensor_desc, conv_state.tensor_desc,
                        conv_weight.tensor_desc};
  op_def.dst_tensors = {out.tensor_desc, next_state.tensor_desc};

  GraphFloat32 graph;
  Node* conv_node = graph.NewNode();
  conv_node->operation.type = kShortConvStepType;
  ShortConvStepAttributes attr;
  attr.conv_L_cache = kFilterSize;
  attr.is_gated = false;
  attr.use_silu = true;
  conv_node->operation.attributes = attr;

  LiteRtOpSelector selector(&create_info, &gpu_info);
  ASSERT_OK(selector.GPUOperationFromNode(
      op_def, {&v_in_proj, &v_conv_state, &v_conv_weight},
      {&v_out, &v_next_state}, *conv_node, &builder));

  GpuModel gpu_model;
  ASSERT_OK(builder.GetGpuModel({in_proj.id, conv_state.id, conv_weight.id},
                                {out.id, next_state.id}, &gpu_model));

  ASSERT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].name, "short_conv_step");
  const std::string& code = gpu_model.nodes[0].gpu_operation->code_;
  EXPECT_THAT(
      code,
      HasSubstr("conv_out.x = s0.x * w0.x + s0.y * w0.y + s0.z * w0.z + p.x * "
                "w0.w;"));
  EXPECT_THAT(code, HasSubstr("(s0.y, s0.z, p.x, 0.0f)"));
  EXPECT_THAT(code, HasSubstr("conv_out = conv_out / ("));
  EXPECT_THAT(code, HasSubstr("+ exp(-conv_out));"));
}

}  // namespace
}  // namespace litert::ml_drift
