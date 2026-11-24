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

#include "ml_drift_delegate/delegate/composite/ir/block_ffw_parser.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/block_ffw_parser.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

using ::absl_testing::StatusIs;
using ::ml_drift::BHWC;
using ::ml_drift::OperationType;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

struct IntArrayHolder {
  IntArrayHolder(std::initializer_list<int> vals) {
    ptr = TfLiteIntArrayCreate(vals.size());
    int i = 0;
    for (int v : vals) ptr->data[i++] = v;
  }
  ~IntArrayHolder() { TfLiteIntArrayFree(ptr); }
  IntArrayHolder(const IntArrayHolder&) = delete;
  IntArrayHolder& operator=(const IntArrayHolder&) = delete;
  IntArrayHolder(IntArrayHolder&& other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }
  TfLiteIntArray* ptr = nullptr;
};

struct SingleNodeContextState {
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
};

TfLiteStatus GetSingleNodeAndRegistration(TfLiteContext* context, int node_idx,
                                          TfLiteNode** node,
                                          TfLiteRegistration** registration) {
  if (node_idx != 0) return kTfLiteError;
  auto* state = static_cast<SingleNodeContextState*>(context->impl_);
  *node = state->node;
  *registration = state->registration;
  return kTfLiteOk;
}

class SyntheticCompositeArena {
 public:
  int AddRuntimeTensor(TfLiteType type, std::initializer_list<int> dims) {
    dims_.emplace_back(dims);
    TfLiteTensor t{};
    t.type = type;
    t.dims = dims_.back().ptr;
    t.allocation_type = kTfLiteArenaRw;
    tensors_.push_back(t);
    return static_cast<int>(tensors_.size()) - 1;
  }

  int AddConstFloatTensor(std::initializer_list<int> dims, float fill_val) {
    dims_.emplace_back(dims);
    size_t count = 1;
    for (int d : dims) count *= d;
    float_buffers_.push_back(std::vector<float>(count, fill_val));
    TfLiteTensor t{};
    t.type = kTfLiteFloat32;
    t.dims = dims_.back().ptr;
    t.data.raw_const =
        reinterpret_cast<const char*>(float_buffers_.back().data());
    t.bytes = count * sizeof(float);
    t.allocation_type = kTfLiteMmapRo;
    tensors_.push_back(t);
    return static_cast<int>(tensors_.size()) - 1;
  }

  int AddBlockwiseV2Tensor(TfLiteType type, std::initializer_list<int> dims,
                           std::initializer_list<int> block_shape,
                           std::initializer_list<int> param_dims,
                           float scale_val, float zp_val) {
    const int scale_idx = AddConstFloatTensor(param_dims, scale_val);
    const int zp_idx = AddConstFloatTensor(param_dims, zp_val);
    dims_.emplace_back(dims);
    dims_.emplace_back(block_shape);
    size_t count = 1;
    for (int d : dims) count *= d;
    size_t packed_bytes = count;
    if (type == kTfLiteInt2) {
      packed_bytes = (count + 3) / 4;
    } else if (type == kTfLiteInt4) {
      packed_bytes = (count + 1) / 2;
    }
    byte_buffers_.push_back(std::vector<int8_t>(packed_bytes, 0x4E));
    auto q = std::make_unique<TfLiteBlockwiseQuantizationV2>();
    q->scale = scale_idx;
    q->zero_point = zp_idx;
    q->block_shape = dims_.back().ptr;
    quant_params_.push_back(std::move(q));

    TfLiteTensor t{};
    t.type = type;
    t.dims = dims_[dims_.size() - 2].ptr;
    t.data.raw_const =
        reinterpret_cast<const char*>(byte_buffers_.back().data());
    t.bytes = packed_bytes;
    t.allocation_type = kTfLiteMmapRo;
    t.quantization.type = kTfLiteBlockwiseQuantizationV2;
    t.quantization.params = quant_params_.back().get();
    tensors_.push_back(t);
    return static_cast<int>(tensors_.size()) - 1;
  }

  int AddBlockwiseV2Int2Tensor(std::initializer_list<int> dims,
                               std::initializer_list<int> block_shape,
                               std::initializer_list<int> param_dims,
                               float scale_val, float zp_val) {
    return AddBlockwiseV2Tensor(kTfLiteInt2, dims, block_shape, param_dims,
                                scale_val, zp_val);
  }

  TfLiteContext BuildContext(SingleNodeContextState* state = nullptr) {
    TfLiteContext ctx{};
    ctx.tensors = tensors_.data();
    ctx.tensors_size = tensors_.size();
    ctx.impl_ = state;
    ctx.GetNodeAndRegistration = GetSingleNodeAndRegistration;
    return ctx;
  }

 private:
  std::vector<IntArrayHolder> dims_;
  std::vector<std::vector<float>> float_buffers_;
  std::vector<std::vector<int8_t>> byte_buffers_;
  std::vector<std::unique_ptr<TfLiteBlockwiseQuantizationV2>> quant_params_;
  std::vector<TfLiteTensor> tensors_;
};

std::vector<uint8_t> BuildFlexAttrs(const std::string& activation,
                                    bool include_scales,
                                    bool distinct_adjacent_scales = false,
                                    bool dynamic_spec = false) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.String("activation", activation);
    if (dynamic_spec) {
      fbb.String("u1_quant_spec", "unsupported_dynamic_spec");
    }
    if (include_scales) {
      fbb.Float("u1_in_scale", 0.02f);
      fbb.Float("u1_out_scale", 0.05f);
      fbb.Float("u2_in_scale", distinct_adjacent_scales ? 0.06f : 0.05f);
      fbb.Float("u2_out_scale", 0.10f);
      fbb.Float("d1_in_scale", 0.04f);
      fbb.Float("d1_out_scale", 0.03f);
      fbb.Float("d2_in_scale", distinct_adjacent_scales ? 0.025f : 0.03f);
      fbb.Float("d2_out_scale", 0.02f);
    }
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::unique_ptr<::ml_drift::ir::IrModel> RunBuildIrModel(
    TfLiteContext& ctx, TfLiteNode& node, TfLiteRegistration& reg,
    SharedConstTensorsMap* shared_tensors,
    const TensorIndexToBufferIdMap* buffer_map) {
  SingleNodeContextState state{.node = &node, .registration = &reg};
  ctx.impl_ = &state;
  IntArrayHolder nodes_to_replace({0});
  TfLiteDelegateParams delegate_params{};
  delegate_params.nodes_to_replace = nodes_to_replace.ptr;
  delegate_params.input_tensors = node.inputs;
  delegate_params.output_tensors = node.outputs;
  CustomIrOpMap custom_parsers;
  custom_parsers[kBlockedFfwCompositeName] = GetBlockFfwParser();
  custom_parsers[kBlockFfwCompositeName] = GetBlockFfwParser();
  return std::unique_ptr<::ml_drift::ir::IrModel>(
      BuildIrModel(ctx, delegate_params, /*options=*/{}, &custom_parsers,
                   shared_tensors, buffer_map));
}

TEST(BlockFfwIrParserTest, ConvertsSharedInt2WeightsAndInfers3dLayouts) {
  // K1 = 2, K2 = 2, S = 4 -> 7 inputs (x, u1_0, u1_1, u2, d1, d2_0, d2_1).
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int w_u1_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u1_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  // U2 stored as [K2, I, O] (oi_transposed = true), D1 stored as
  // [K2, O, I] (oi_transposed = false).
  const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 8, 12}, {1, 8, 1},
                                                  {2, 1, 12}, 0.1f, -0.5f);
  const int w_d1 = arena.AddBlockwiseV2Int2Tensor({2, 8, 12}, {1, 1, 12},
                                                  {2, 8, 1}, 0.1f, -0.5f);
  const int w_d2_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_d2_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  std::vector<uint8_t> fb_attrs =
      BuildFlexAttrs("silu", /*include_scales=*/true,
                     /*distinct_adjacent_scales=*/true);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockedFfwCompositeName.data();
  params.subgraph_index = 0;
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinStablehloComposite;

  TfLiteContext ctx = arena.BuildContext();
  CustomIrOpParser parser = GetBlockFfwParser();
  ASSERT_OK(parser.is_supported(&ctx, &node, &reg));

  TensorIndexToBufferIdMap buffer_map = {{w_u1_0, 1}, {w_u1_1, 2}, {w_u2, 3},
                                         {w_d1, 4},   {w_d2_0, 5}, {w_d2_1, 6}};
  SharedConstTensorsMap shared_tensors;
  auto ir_model = RunBuildIrModel(ctx, node, reg, &shared_tensors, &buffer_map);
  ASSERT_NE(ir_model, nullptr);

  ASSERT_EQ(shared_tensors.size(), 6);
  bool saw_u2 = false;
  bool saw_d1 = false;
  for (const auto& [ir_tid, shared_info] : shared_tensors) {
    EXPECT_FALSE(shared_info.dequant_forced);
    const auto* t = ir_model->tensor(ir_tid);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->consumers.size(), 1);
    if (shared_info.tflite_tensor_id == w_u2) {
      saw_u2 = true;
      EXPECT_TRUE(shared_info.oi_transposed);
      EXPECT_EQ(t->desc.GetBHWCShape(), BHWC(12, 2, 1, 8));
    } else if (shared_info.tflite_tensor_id == w_d1) {
      saw_d1 = true;
      EXPECT_FALSE(shared_info.oi_transposed);
      EXPECT_EQ(t->desc.GetBHWCShape(), BHWC(8, 2, 1, 12));
    } else {
      EXPECT_TRUE(shared_info.oi_transposed);
      EXPECT_EQ(t->desc.GetBHWCShape(), BHWC(8, 1, 1, 8));
    }
  }
  EXPECT_TRUE(saw_u2);
  EXPECT_TRUE(saw_d1);

  std::vector<std::string> op_names;
  int fc2_count = 0;
  for (const auto& op : ir_model->ops()) {
    ASSERT_NE(op, nullptr);
    op_names.push_back(op->name);
    if (op->name == ToString(OperationType::kFullyConnectedInt2)) {
      ++fc2_count;
      const auto* attr =
          std::any_cast<::ml_drift::FullyConnectedInt2Attributes>(&op->attr);
      ASSERT_NE(attr, nullptr);
      EXPECT_GT(attr->scale.shape.o, 0);
      EXPECT_EQ(attr->zero_point.shape, attr->scale.shape);
    }
  }
  EXPECT_EQ(fc2_count, 6);

  const std::string qdq = ToString(OperationType::kQuantizeAndDequantize);
  const std::string fc2 = ToString(OperationType::kFullyConnectedInt2);
  EXPECT_THAT(op_names, ElementsAre(kBlockFfwUnpackPartType, qdq, fc2, qdq,
                                    kBlockFfwUnpackPartType, qdq, fc2, qdq,
                                    kBlockFfwPackPartsType, qdq, fc2, qdq,
                                    kBlockFfwSiluType, qdq, fc2, qdq,
                                    kBlockFfwUnpackPartType, qdq, fc2, qdq,
                                    kBlockFfwUnpackPartType, qdq, fc2, qdq,
                                    kBlockFfwPackPartsType));
}

TEST(BlockFfwIrParserTest, DeduplicatesIdenticalAdjacentScales) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int w_u1_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u1_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 8, 12}, {1, 8, 1},
                                                  {2, 1, 12}, 0.1f, -0.5f);
  const int w_d1 = arena.AddBlockwiseV2Int2Tensor({2, 12, 8}, {1, 12, 1},
                                                  {2, 1, 8}, 0.1f, -0.5f);
  const int w_d2_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_d2_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  std::vector<uint8_t> fb_attrs =
      BuildFlexAttrs("silu", /*include_scales=*/true,
                     /*distinct_adjacent_scales=*/false);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockFfwCompositeName.data();
  params.subgraph_index = 0;
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinStablehloComposite;

  TfLiteContext ctx = arena.BuildContext();
  TensorIndexToBufferIdMap buffer_map = {{w_u1_0, 1}, {w_u1_1, 2}, {w_u2, 3},
                                         {w_d1, 4},   {w_d2_0, 5}, {w_d2_1, 6}};
  SharedConstTensorsMap shared_tensors;
  auto ir_model = RunBuildIrModel(ctx, node, reg, &shared_tensors, &buffer_map);
  ASSERT_NE(ir_model, nullptr);

  std::vector<std::string> op_names;
  for (const auto& op : ir_model->ops()) {
    op_names.push_back(op->name);
  }
  const std::string qdq = ToString(OperationType::kQuantizeAndDequantize);
  const std::string fc2 = ToString(OperationType::kFullyConnectedInt2);
  // u2_in_scale == u1_out_scale and d2_in_scale == d1_out_scale are elided.
  EXPECT_THAT(
      op_names,
      ElementsAre(kBlockFfwUnpackPartType, qdq, fc2, qdq,
                  kBlockFfwUnpackPartType, qdq, fc2, qdq,
                  kBlockFfwPackPartsType, fc2, qdq, kBlockFfwSiluType, qdq, fc2,
                  qdq, kBlockFfwUnpackPartType, fc2, qdq,
                  kBlockFfwUnpackPartType, fc2, qdq, kBlockFfwPackPartsType));
}

TEST(BlockFfwIrParserTest, ConvertsNonSharedWeightsViaHostFallback) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 3, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 3, 16});
  const int w_u1_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u1_1 = arena.AddConstFloatTensor({8, 8}, 0.25f);
  const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 8, 12}, {1, 8, 1},
                                                  {2, 1, 12}, 0.1f, -0.5f);
  const int w_d1 = arena.AddConstFloatTensor({2, 12, 8}, 0.125f);
  const int w_d2_0 = arena.AddConstFloatTensor({8, 8}, 0.5f);
  const int w_d2_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  std::vector<uint8_t> fb_attrs =
      BuildFlexAttrs("silu", /*include_scales=*/false);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockedFfwCompositeName.data();
  params.subgraph_index = 0;
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinStablehloComposite;

  TfLiteContext ctx = arena.BuildContext();
  SharedConstTensorsMap shared_tensors;
  auto ir_model =
      RunBuildIrModel(ctx, node, reg, &shared_tensors, /*buffer_map=*/nullptr);
  ASSERT_NE(ir_model, nullptr);
  EXPECT_THAT(shared_tensors, IsEmpty());

  std::vector<std::string> op_names;
  for (const auto& op : ir_model->ops()) {
    op_names.push_back(op->name);
  }
  const std::string fc = ToString(OperationType::kFullyConnected);
  const std::string cnst = ToString(OperationType::kConstant);
  const std::string bmm = ToString(OperationType::kBatchedMatmul);
  EXPECT_THAT(
      op_names,
      ElementsAre(kBlockFfwUnpackPartType, fc, kBlockFfwUnpackPartType, fc,
                  kBlockFfwPackPartsType, cnst, bmm, kBlockFfwSiluType, cnst,
                  bmm, kBlockFfwUnpackPartType, fc, kBlockFfwUnpackPartType, fc,
                  kBlockFfwPackPartsType));

  for (const auto& op : ir_model->ops()) {
    if (op->name != fc) continue;
    const auto* attr =
        std::any_cast<::ml_drift::FullyConnectedAttributes>(&op->attr);
    ASSERT_NE(attr, nullptr);
    EXPECT_GE(attr->weights.data.size(),
              attr->weights.shape.DimensionsProduct() +
                  XNN_EXTRA_BYTES / sizeof(float));
  }
}

TEST(BlockFfwIrParserTest, DuplicateSharedWeightGetsDistinctIrHandles) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int w_edge =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 8, 12}, {1, 8, 1},
                                                  {2, 1, 12}, 0.1f, -0.5f);
  const int w_d1 = arena.AddBlockwiseV2Int2Tensor({2, 12, 8}, {1, 12, 1},
                                                  {2, 1, 8}, 0.1f, -0.5f);

  // Reuse `w_edge` across all four edge weight slots.
  IntArrayHolder inputs({x_id, w_edge, w_edge, w_u2, w_d1, w_edge, w_edge});
  IntArrayHolder outputs({out_id});
  std::vector<uint8_t> fb_attrs =
      BuildFlexAttrs("silu", /*include_scales=*/false);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockedFfwCompositeName.data();
  params.subgraph_index = 0;
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinStablehloComposite;

  TfLiteContext ctx = arena.BuildContext();
  TensorIndexToBufferIdMap buffer_map = {{w_edge, 10}, {w_u2, 20}, {w_d1, 30}};
  SharedConstTensorsMap shared_tensors;
  auto ir_model = RunBuildIrModel(ctx, node, reg, &shared_tensors, &buffer_map);
  ASSERT_NE(ir_model, nullptr);
  ASSERT_EQ(shared_tensors.size(), 6);
  for (const auto& [ir_tid, shared_info] : shared_tensors) {
    EXPECT_FALSE(shared_info.dequant_forced);
    EXPECT_FALSE(shared_info.layout.has_value());
    const auto* t = ir_model->tensor(ir_tid);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->consumers.size(), 1);
    EXPECT_FALSE(t->buffer_source.dequant_forced);
    EXPECT_FALSE(t->buffer_source.force_linear_layout);
  }
}

TEST(BlockFfwIrParserTest, RejectsUnsupportedActivationAndDynamicQuantSpec) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int w_u1_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u1_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 8, 12}, {1, 8, 1},
                                                  {2, 1, 12}, 0.1f, -0.5f);
  const int w_d1 = arena.AddBlockwiseV2Int2Tensor({2, 12, 8}, {1, 12, 1},
                                                  {2, 1, 8}, 0.1f, -0.5f);
  const int w_d2_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_d2_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  TfLiteContext ctx = arena.BuildContext();
  TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinStablehloComposite;
  CustomIrOpParser parser = GetBlockFfwParser();

  {
    std::vector<uint8_t> bad_act = BuildFlexAttrs("gelu", false);
    TfLiteStablehloCompositeParams params{};
    params.name = kBlockedFfwCompositeName.data();
    params.attributes = bad_act.data();
    params.attributes_size = bad_act.size();
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    node.builtin_data = &params;
    EXPECT_THAT(parser.is_supported(&ctx, &node, &reg),
                StatusIs(absl::StatusCode::kUnimplemented));
  }
  {
    std::vector<uint8_t> dyn_spec = BuildFlexAttrs("silu", false, false, true);
    TfLiteStablehloCompositeParams params{};
    params.name = kBlockedFfwCompositeName.data();
    params.attributes = dyn_spec.data();
    params.attributes_size = dyn_spec.size();
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    node.builtin_data = &params;
    EXPECT_THAT(parser.is_supported(&ctx, &node, &reg),
                StatusIs(absl::StatusCode::kUnimplemented));
  }
}

TEST(BlockFfwIrParserTest,
     K2OneNonSharedMiddleWeightsUseBatchedMatmulFallback) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 2, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 2, 16});
  const int w_u1_0 = arena.AddConstFloatTensor({4, 8}, 1.0f);
  const int w_u1_1 = arena.AddConstFloatTensor({4, 8}, 1.0f);
  // K2 = 1 middle weights: U2 is `[1, O=12, I=8]` (HOI), D1 is `[1, I=12, O=8]`
  // (HIO). Both must lower to Constant + BatchedMatmul rather than FC.
  const int w_u2 = arena.AddConstFloatTensor({1, 12, 8}, 1.0f);
  const int w_d1 = arena.AddConstFloatTensor({1, 12, 8}, 1.0f);
  const int w_d2_0 = arena.AddConstFloatTensor({8, 4}, 1.0f);
  const int w_d2_1 = arena.AddConstFloatTensor({8, 4}, 1.0f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  std::vector<uint8_t> fb_attrs =
      BuildFlexAttrs("silu", /*include_scales=*/false);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockedFfwCompositeName.data();
  params.subgraph_index = 0;
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinStablehloComposite;

  TfLiteContext ctx = arena.BuildContext();
  SharedConstTensorsMap shared_tensors;
  auto ir_model =
      RunBuildIrModel(ctx, node, reg, &shared_tensors, /*buffer_map=*/nullptr);
  ASSERT_NE(ir_model, nullptr);

  std::vector<std::string> op_names;
  for (const auto& op : ir_model->ops()) {
    op_names.push_back(op->name);
  }
  const std::string fc = ToString(OperationType::kFullyConnected);
  const std::string cnst = ToString(OperationType::kConstant);
  const std::string bmm = ToString(OperationType::kBatchedMatmul);
  EXPECT_THAT(
      op_names,
      ElementsAre(kBlockFfwUnpackPartType, fc, kBlockFfwUnpackPartType, fc,
                  kBlockFfwPackPartsType, cnst, bmm, kBlockFfwSiluType, cnst,
                  bmm, kBlockFfwUnpackPartType, fc, kBlockFfwUnpackPartType, fc,
                  kBlockFfwPackPartsType));
}

TEST(BlockFfwIrParserTest, ConvertsSharedInt4AndInt8Weights) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int w_u1_0 = arena.AddBlockwiseV2Tensor(kTfLiteInt8, {8, 8}, {1, 8},
                                                {8, 1}, 0.1f, 0.0f);
  const int w_u1_1 = arena.AddBlockwiseV2Tensor(kTfLiteInt4, {8, 8}, {1, 8},
                                                {8, 1}, 0.1f, -0.5f);
  // U2 INT4 stored as `[H=2, I=8, O=12]`, D1 INT8 stored as `[H=2, O=8, I=12]`.
  const int w_u2 = arena.AddBlockwiseV2Tensor(kTfLiteInt4, {2, 8, 12},
                                              {1, 8, 1}, {2, 1, 12}, 0.1f,
                                              -0.5f);
  const int w_d1 = arena.AddBlockwiseV2Tensor(kTfLiteInt8, {2, 8, 12},
                                              {1, 1, 12}, {2, 8, 1}, 0.1f,
                                              0.0f);
  const int w_d2_0 = arena.AddBlockwiseV2Tensor(kTfLiteInt4, {8, 8}, {1, 8},
                                                {8, 1}, 0.1f, -0.5f);
  const int w_d2_1 = arena.AddBlockwiseV2Tensor(kTfLiteInt8, {8, 8}, {1, 8},
                                                {8, 1}, 0.1f, 0.0f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  std::vector<uint8_t> fb_attrs =
      BuildFlexAttrs("silu", /*include_scales=*/false);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockedFfwCompositeName.data();
  params.subgraph_index = 0;
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinStablehloComposite;

  TfLiteContext ctx = arena.BuildContext();
  TensorIndexToBufferIdMap buffer_map = {{w_u1_0, 1}, {w_u1_1, 2}, {w_u2, 3},
                                         {w_d1, 4},   {w_d2_0, 5}, {w_d2_1, 6}};
  SharedConstTensorsMap shared_tensors;
  auto ir_model = RunBuildIrModel(ctx, node, reg, &shared_tensors, &buffer_map);
  ASSERT_NE(ir_model, nullptr);
  ASSERT_EQ(shared_tensors.size(), 6);
  for (const auto& [ir_tid, shared_info] : shared_tensors) {
    EXPECT_FALSE(shared_info.dequant_forced);
    if (shared_info.tflite_tensor_id == w_u2) {
      EXPECT_TRUE(shared_info.oi_transposed);
    } else if (shared_info.tflite_tensor_id == w_d1) {
      EXPECT_FALSE(shared_info.oi_transposed);
    }
  }

  std::vector<std::string> op_names;
  for (const auto& op : ir_model->ops()) {
    ASSERT_NE(op, nullptr);
    op_names.push_back(op->name);
    if (op->name == ToString(OperationType::kFullyConnectedInt4)) {
      const auto* attr =
          std::any_cast<::ml_drift::FullyConnectedInt4Attributes>(&op->attr);
      ASSERT_NE(attr, nullptr);
      EXPECT_GT(attr->scale.shape.o, 0);
      EXPECT_EQ(attr->zero_point.shape, attr->scale.shape);
    } else if (op->name == ToString(OperationType::kFullyConnectedInt8)) {
      const auto* attr =
          std::any_cast<::ml_drift::FullyConnectedInt8Attributes>(&op->attr);
      ASSERT_NE(attr, nullptr);
      EXPECT_GT(attr->scale.shape.o, 0);
      EXPECT_EQ(attr->zero_point.shape, attr->scale.shape);
    }
  }

  const std::string fc4 = ToString(OperationType::kFullyConnectedInt4);
  const std::string fc8 = ToString(OperationType::kFullyConnectedInt8);
  EXPECT_THAT(
      op_names,
      ElementsAre(kBlockFfwUnpackPartType, fc8, kBlockFfwUnpackPartType, fc4,
                  kBlockFfwPackPartsType, fc4, kBlockFfwSiluType, fc8,
                  kBlockFfwUnpackPartType, fc4, kBlockFfwUnpackPartType, fc8,
                  kBlockFfwPackPartsType));
}

TEST(BlockFfwIrParserTest,
     ClearsStaleBufferSourceFlagsOnEmptyConsumerWeightTensor) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int w_u1_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u1_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 8, 12}, {1, 8, 1},
                                                  {2, 1, 12}, 0.1f, -0.5f);
  const int w_d1 = arena.AddBlockwiseV2Int2Tensor({2, 12, 8}, {1, 12, 1},
                                                  {2, 1, 8}, 0.1f, -0.5f);
  const int w_d2_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_d2_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  std::vector<uint8_t> fb_attrs =
      BuildFlexAttrs("silu", /*include_scales=*/false);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockedFfwCompositeName.data();
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinStablehloComposite;
  TfLiteContext ctx = arena.BuildContext();

  // Simulate `PrepareSharedConstantInputsForNode` handing `BlockFfwConvert`
  // cloned empty-consumer weight IrTensors carrying stale `dequant_forced` and
  // `force_linear_layout` flags from an earlier non-quantized consumer.
  ::ml_drift::ir::IrModel ir_model;
  absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId> tensor_map;
  tensor_map[x_id] =
      ir_model.add_tensor(::ml_drift::DataType::kFloat16, BHWC(1, 1, 1, 16))
          ->id;
  tensor_map[out_id] =
      ir_model.add_tensor(::ml_drift::DataType::kFloat16, BHWC(1, 1, 1, 16))
          ->id;
  for (int tfl_id : {w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1}) {
    ::ml_drift::ir::IrTensor* w_tensor =
        ir_model.add_tensor(::ml_drift::DataType::kInt2, BHWC(1, 1, 1, 1));
    w_tensor->buffer_source.is_shared = true;
    w_tensor->buffer_source.tensor_id = tfl_id;
    w_tensor->buffer_source.dequant_forced = true;
    w_tensor->buffer_source.force_linear_layout = true;
    ASSERT_THAT(w_tensor->consumers, IsEmpty());
    tensor_map[tfl_id] = w_tensor->id;
  }

  CustomIrOpParser parser = GetBlockFfwParser();
  parser.convert(ctx, node, reg, tensor_map, /*options=*/{}, ir_model);

  int quantized_fc_count = 0;
  for (const auto& op : ir_model.ops()) {
    ASSERT_NE(op, nullptr);
    if (op->name != ToString(OperationType::kFullyConnectedInt2)) continue;
    ++quantized_fc_count;
    ASSERT_EQ(op->inputs.size(), 2);
    const ::ml_drift::ir::IrTensor* w = ir_model.tensor(op->inputs[1]);
    ASSERT_NE(w, nullptr);
    EXPECT_TRUE(w->buffer_source.is_shared);
    EXPECT_FALSE(w->buffer_source.dequant_forced);
    EXPECT_FALSE(w->buffer_source.force_linear_layout);
  }
  EXPECT_EQ(quantized_fc_count, 6);
}

}  // namespace
}  // namespace litert::ml_drift::ir
