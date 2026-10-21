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

#include "ml_drift_delegate/delegate/composite/ir/litert_op_selector.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

namespace litert::ml_drift::ir {
namespace {

using ::ml_drift::BHWC;
using ::ml_drift::CalculationsPrecision;
using ::ml_drift::CreateGpuModelInfo;
using ::ml_drift::DataType;
using ::ml_drift::GpuApi;
using ::ml_drift::GpuInfo;
using ::ml_drift::GpuModel;
using ::ml_drift::GpuModelBuilder;
using ::ml_drift::Layout;
using ::ml_drift::OHWI;
using ::ml_drift::OperationDef;
using ::ml_drift::SliceAttributes;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using ::ml_drift::ValueId;
using ::ml_drift::WeightsLayout;
using ::ml_drift::ir::IrModel;
using ::ml_drift::ir::IrOp;
using ::ml_drift::ir::IrTensor;

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

class LiteRtOpSelectorIrTest : public ::testing::Test {
 protected:
  LiteRtOpSelectorIrTest()
      : gpu_info_(GetTestGpuInfo()),
        create_info_{.precision = CalculationsPrecision::kF32,
                     .storage_type = TensorStorageType::kTexture2D},
        builder_(gpu_info_, {.storage = TensorStorageType::kTexture2D}),
        selector_(&create_info_, &gpu_info_) {}

  const IrTensor* AddTensor(
      const BHWC& shape, DataType dtype = DataType::kFloat32,
      TensorStorageType storage = TensorStorageType::kTexture2D) {
    TensorDescriptor desc(dtype, storage, Layout::kHWC);
    desc.SetBHWCShape(shape);
    auto handle = builder_.AddTensor(desc);
    IrTensor* tensor =
        tensors_.emplace_back(std::make_unique<IrTensor>(handle.id)).get();
    tensor->desc = handle.tensor_desc;
    return tensor;
  }

  absl::Status SelectOp(absl::string_view name, std::any attr,
                        const std::vector<const IrTensor*>& inputs,
                        const std::vector<const IrTensor*>& outputs) {
    OperationDef op_def;
    for (const IrTensor* in : inputs) op_def.src_tensors.push_back(in->desc);
    for (const IrTensor* out : outputs) op_def.dst_tensors.push_back(out->desc);
    IrOp* op = ir_model_.add_op();
    op->name = std::string(name);
    op->attr = std::move(attr);
    return selector_.GPUOperationFromNode(op_def, inputs, outputs, *op,
                                          &builder_);
  }

  GpuModelBuilder::TensorHandle Handle(const IrTensor* tensor) {
    return {tensor->desc, static_cast<ValueId>(tensor->id)};
  }

  absl::Status BuildModel(const std::vector<const IrTensor*>& inputs,
                          const std::vector<const IrTensor*>& outputs,
                          GpuModel* gpu_model) {
    std::vector<ValueId> in_ids;
    std::vector<ValueId> out_ids;
    for (const IrTensor* in : inputs) in_ids.push_back(in->id);
    for (const IrTensor* out : outputs) out_ids.push_back(out->id);
    return builder_.GetGpuModel(in_ids, out_ids, gpu_model);
  }

  GpuInfo gpu_info_;
  CreateGpuModelInfo create_info_;
  GpuModelBuilder builder_;
  LiteRtOpSelector selector_;
  IrModel ir_model_;
  std::vector<std::unique_ptr<IrTensor>> tensors_;
};

// When a runtime-param tensor (produced by an earlier op with TEXTURE_2D
// storage) is consumed both by a composite op that requires BUFFER storage
// (such as `add_values_to_cache`) and by another op that expects the original
// TEXTURE_2D tensor (such as a channel slice in the ring-buffer SDPA path),
// `TensorToBuffer` must not orphan the original tensor by re-pointing its
// producer. Every consumer's input tensor must remain produced by some node in
// the model.
TEST_F(LiteRtOpSelectorIrTest,
       ParamTensorToBufferPreservesProducerForOtherConsumers) {
  const IrTensor* raw_param = AddTensor(BHWC(1, 1, 1, 7), DataType::kInt32);
  const IrTensor* param = AddTensor(BHWC(1, 1, 1, 7), DataType::kInt32);
  builder_.Copy(Handle(raw_param), Handle(param));

  const IrTensor* src_k = AddTensor(BHWC(1, 1, 8, 64));
  const IrTensor* src_v = AddTensor(BHWC(1, 1, 8, 64));
  const IrTensor* dst_k = AddTensor(BHWC(1, 1, 8, 64));
  const IrTensor* dst_v = AddTensor(BHWC(1, 1, 8, 64));

  AddValuesToCacheAttributes attr{
      .kv_cache_batch_size = 1, .cache_size = 64, .head_size = 64};
  ASSERT_OK(SelectOp(kAddValuesToCacheType, attr, {src_k, src_v, param},
                     {dst_k, dst_v}));

  SliceAttributes slice_attr{.starts = BHWC(0, 0, 0, 3),
                             .ends = BHWC(1, 1, 1, 7),
                             .strides = BHWC(1, 1, 1, 1)};
  auto sliced_handle = builder_.StridedSlice(Handle(param), slice_attr);
  IrTensor sliced_param(sliced_handle.id);

  GpuModel gpu_model;
  ASSERT_OK(BuildModel({raw_param, src_k, src_v},
                       {dst_k, dst_v, &sliced_param}, &gpu_model));

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
  for (const auto& node : gpu_model.nodes) {
    for (ValueId in_id : node.inputs) {
      EXPECT_TRUE(defined_tensors.contains(in_id))
          << "Node '" << node.name << "' reads tensor " << in_id
          << " which has no producer in the GpuModel.";
    }
  }

  bool found_cache_node = false;
  for (const auto& node : gpu_model.nodes) {
    if (node.name == kAddValuesToCacheType) {
      found_cache_node = true;
      ASSERT_EQ(node.inputs.size(), 3);
      EXPECT_EQ(gpu_model.tensors.at(node.inputs[2]).GetStorageType(),
                TensorStorageType::kBuffer);
    }
  }
  EXPECT_TRUE(found_cache_node);
}

TEST_F(LiteRtOpSelectorIrTest,
       AddValuesToCacheIntermediateOutputsUseBufferForSdpaTransposed) {
  const IrTensor* q = AddTensor(BHWC(1, 4, 1, 64));
  const IrTensor* src_k = AddTensor(BHWC(1, 1, 32, 64));
  const IrTensor* src_v = AddTensor(BHWC(1, 1, 32, 64));
  const IrTensor* param = AddTensor(BHWC(1, 1, 1, 7), DataType::kInt32);
  const IrTensor* cache_k = AddTensor(BHWC(1, 1, 32, 64));
  const IrTensor* cache_v = AddTensor(BHWC(1, 1, 64, 32));
  const IrTensor* sdpa_out = AddTensor(BHWC(1, 4, 1, 64));

  AddValuesToCacheAttributes cache_attr{
      .kv_cache_batch_size = 1, .cache_size = 32, .head_size = 64};
  ASSERT_OK(SelectOp(kAddValuesToCacheType, cache_attr, {src_k, src_v, param},
                     {cache_k, cache_v}));

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
  ASSERT_OK(SelectOp(kSdpaTransposedType, sdpa_attr,
                     {q, cache_k, cache_v, param}, {sdpa_out}));

  GpuModel gpu_model;
  ASSERT_OK(BuildModel({q, src_k, src_v, param}, {sdpa_out}, &gpu_model));

  bool found_cache_node = false;
  for (const auto& node : gpu_model.nodes) {
    if (node.name == kAddValuesToCacheType) {
      found_cache_node = true;
      ASSERT_EQ(node.outputs.size(), 2);
      for (ValueId out_id : node.outputs) {
        EXPECT_EQ(gpu_model.tensors.at(out_id).GetStorageType(),
                  TensorStorageType::kBuffer);
      }
    }
  }
  EXPECT_TRUE(found_cache_node);
}

}  // namespace
}  // namespace litert::ml_drift::ir
