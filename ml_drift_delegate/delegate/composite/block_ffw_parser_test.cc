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

#include "ml_drift_delegate/delegate/composite/block_ffw_parser.h"

#include <any>
#include <cstdint>
#include <cstring>
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
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/custom_parsers.h"
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

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

struct FloatArrayHolder {
  FloatArrayHolder(std::initializer_list<float> vals) {
    ptr = TfLiteFloatArrayCreate(vals.size());
    int i = 0;
    for (float v : vals) ptr->data[i++] = v;
  }
  ~FloatArrayHolder() { TfLiteFloatArrayFree(ptr); }
  FloatArrayHolder(const FloatArrayHolder&) = delete;
  FloatArrayHolder& operator=(const FloatArrayHolder&) = delete;
  FloatArrayHolder(FloatArrayHolder&& other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }
  TfLiteFloatArray* ptr = nullptr;
};

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

  int AddBlockwiseV2Int2Tensor(std::initializer_list<int> dims,
                               std::initializer_list<int> block_shape,
                               std::initializer_list<int> param_dims,
                               float scale_val, float zp_val) {
    const int scale_idx = AddConstFloatTensor(param_dims, scale_val);
    const int zp_idx = AddConstFloatTensor(param_dims, zp_val);
    dims_.emplace_back(dims);
    dims_.emplace_back(block_shape);
    size_t count = 1;
    for (int d : dims) count *= d;
    const size_t packed_bytes = (count + 3) / 4;
    byte_buffers_.push_back(std::vector<int8_t>(packed_bytes, 0x4E));
    auto q = std::make_unique<TfLiteBlockwiseQuantizationV2>();
    q->scale = scale_idx;
    q->zero_point = zp_idx;
    q->block_shape = dims_.back().ptr;
    quant_params_.push_back(std::move(q));

    TfLiteTensor t{};
    t.type = kTfLiteInt2;
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

  int AddConstFloat16Tensor(std::initializer_list<int> dims,
                            uint16_t fill_val) {
    dims_.emplace_back(dims);
    size_t count = 1;
    for (int d : dims) count *= d;
    fp16_buffers_.push_back(std::vector<uint16_t>(count, fill_val));
    TfLiteTensor t{};
    t.type = kTfLiteFloat16;
    t.dims = dims_.back().ptr;
    t.data.raw_const =
        reinterpret_cast<const char*>(fp16_buffers_.back().data());
    t.bytes = count * sizeof(uint16_t);
    t.allocation_type = kTfLiteMmapRo;
    tensors_.push_back(t);
    return static_cast<int>(tensors_.size()) - 1;
  }

  int AddAffineQuantizedInt8Tensor(std::initializer_list<int> dims,
                                   int quantized_dimension = 0) {
    dims_.emplace_back(dims);
    size_t count = 1;
    for (int d : dims) count *= d;
    byte_buffers_.push_back(std::vector<int8_t>(count, 1));
    float_arrays_.emplace_back(std::initializer_list<float>{0.1f});
    dims_.emplace_back(std::initializer_list<int>{0});
    auto q = std::make_unique<TfLiteAffineQuantization>();
    q->scale = float_arrays_.back().ptr;
    q->zero_point = dims_.back().ptr;
    q->quantized_dimension = quantized_dimension;
    affine_quant_params_.push_back(std::move(q));

    TfLiteTensor t{};
    t.type = kTfLiteInt8;
    t.dims = dims_[dims_.size() - 2].ptr;
    t.data.raw_const =
        reinterpret_cast<const char*>(byte_buffers_.back().data());
    t.bytes = count;
    t.allocation_type = kTfLiteMmapRo;
    t.quantization.type = kTfLiteAffineQuantization;
    t.quantization.params = affine_quant_params_.back().get();
    tensors_.push_back(t);
    return static_cast<int>(tensors_.size()) - 1;
  }

  TfLiteTensor& MutableTensor(int idx) { return tensors_[idx]; }

  TfLiteContext BuildContext() {
    TfLiteContext ctx{};
    ctx.tensors = tensors_.data();
    ctx.tensors_size = tensors_.size();
    return ctx;
  }

 private:
  std::vector<IntArrayHolder> dims_;
  std::vector<FloatArrayHolder> float_arrays_;
  std::vector<std::vector<float>> float_buffers_;
  std::vector<std::vector<uint16_t>> fp16_buffers_;
  std::vector<std::vector<int8_t>> byte_buffers_;
  std::vector<std::unique_ptr<TfLiteBlockwiseQuantizationV2>> quant_params_;
  std::vector<std::unique_ptr<TfLiteAffineQuantization>> affine_quant_params_;
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
      fbb.String("u1_quant_spec", "cint2_fp32_int4_e8m0_drq");
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

TEST(BlockFfwParserTest, CustomFactoryRegistersBothCompositeNames) {
  ::ml_drift::GpuInfo gpu_info;
  CustomOperationParserFactory factory(&gpu_info);
  EXPECT_TRUE(factory.SupportsIntegerTypes(kBlockedFfwCompositeName));
  EXPECT_TRUE(factory.SupportsIntegerTypes(kBlockFfwCompositeName));
  EXPECT_NE(factory.Create(kBlockedFfwCompositeName), nullptr);
  EXPECT_NE(factory.Create(kBlockFfwCompositeName), nullptr);
}

TEST(BlockFfwParserTest, ParsesSharedCint2Weights) {
  // K1 = 2, K2 = 2, S = 4 -> 7 inputs (x, u1_0, u1_1, u2, d1, d2_0, d2_1).
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int w_u1_0 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  const int w_u1_1 =
      arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
  // U2 and D1 are stored as [K2, I, O] with block shape [1, I, 1].
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
                     /*distinct_adjacent_scales=*/true);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockFfwCompositeName;
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteContext ctx = arena.BuildContext();
  TfLiteRegistration reg{};
  BlockFfwOperationParser parser;
  ASSERT_OK(parser.IsSupported(&ctx, &node, &reg));

  ::ml_drift::GraphFloat32 graph;
  absl::flat_hash_map<int, ::ml_drift::Value*> tensor_to_value;
  TensorIndexToBufferIdMap buffer_map = {{w_u1_0, 1}, {w_u1_1, 2}, {w_u2, 3},
                                         {w_d1, 4},   {w_d2_0, 5}, {w_d2_1, 6}};
  SharedConstTensorsMap shared_map;
  ObjectReader reader(&graph, &ctx, &node, &tensor_to_value,
                      /*quant_conversion_map=*/nullptr, &buffer_map,
                      /*tensor_to_external_buffer_id_map=*/nullptr,
                      &shared_map);
  parser.Parse(&node, &reg, &graph, &reader);

  ASSERT_EQ(shared_map.size(), 6);
  for (const auto& [val_id, shared_info] : shared_map) {
    EXPECT_TRUE(shared_info.oi_transposed);
  }

  std::vector<std::string> op_types;
  for (const auto* n : graph.nodes()) {
    op_types.push_back(n->operation.type);
  }
  const std::string qdq =
      ToString(::ml_drift::OperationType::kQuantizeAndDequantize);
  const std::string fc2 =
      ToString(::ml_drift::OperationType::kFullyConnectedInt2);
  // Distinct adjacent scales (u2_in != u1_out, d2_in != d1_out) keep their own
  // QuantizeAndDequantize nodes.
  EXPECT_THAT(op_types,
              ElementsAre(kBlockFfwUnpackPartType, qdq, fc2, qdq,
                          kBlockFfwUnpackPartType, qdq, fc2, qdq,
                          kBlockFfwPackPartsType, qdq, fc2, qdq,
                          kBlockFfwSiluType, qdq, fc2, qdq,
                          kBlockFfwUnpackPartType, qdq, fc2, qdq,
                          kBlockFfwUnpackPartType, qdq, fc2, qdq,
                          kBlockFfwPackPartsType));
}

TEST(BlockFfwParserTest, HostFallbackPadsFullyConnectedWeightsForXnnpack) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 2, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 2, 16});
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
  params.name = kBlockedFfwCompositeName;
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteContext ctx = arena.BuildContext();
  TfLiteRegistration reg{};
  BlockFfwOperationParser parser;
  ASSERT_OK(parser.IsSupported(&ctx, &node, &reg));

  ::ml_drift::GraphFloat32 graph;
  absl::flat_hash_map<int, ::ml_drift::Value*> tensor_to_value;
  ObjectReader reader(&graph, &ctx, &node, &tensor_to_value,
                      /*quant_conversion_map=*/nullptr,
                      /*tensor_to_buffer_id_map=*/nullptr,
                      /*tensor_to_external_buffer_id_map=*/nullptr,
                      /*shared_tensor_map=*/nullptr);
  parser.Parse(&node, &reg, &graph, &reader);

  int fc_count = 0;
  const std::string fc = ToString(::ml_drift::OperationType::kFullyConnected);
  for (const auto* n : graph.nodes()) {
    if (n->operation.type != fc) continue;
    ++fc_count;
    const auto* attr = std::any_cast<::ml_drift::FullyConnectedAttributes>(
        &n->operation.attributes);
    ASSERT_NE(attr, nullptr);
    EXPECT_GE(attr->weights.data.size(),
              attr->weights.shape.DimensionsProduct() +
                  XNN_EXTRA_BYTES / sizeof(float));
  }
  EXPECT_EQ(fc_count, 4);
}

TEST(BlockFfwParserTest, Rejects3dEdgeWeights) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int w_u1 = arena.AddBlockwiseV2Int2Tensor({2, 8, 8}, {1, 1, 8},
                                                  {2, 8, 1}, 0.1f, -0.5f);
  const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 8, 12}, {1, 8, 1},
                                                  {2, 1, 12}, 0.1f, -0.5f);
  const int w_d1 = arena.AddBlockwiseV2Int2Tensor({2, 12, 8}, {1, 12, 1},
                                                  {2, 1, 8}, 0.1f, -0.5f);
  const int w_d2 = arena.AddBlockwiseV2Int2Tensor({2, 8, 8}, {1, 1, 8},
                                                  {2, 8, 1}, 0.1f, -0.5f);

  IntArrayHolder inputs({x_id, w_u1, w_u2, w_d1, w_d2});
  IntArrayHolder outputs({out_id});
  std::vector<uint8_t> fb_attrs =
      BuildFlexAttrs("silu", /*include_scales=*/true);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockedFfwCompositeName;
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteContext ctx = arena.BuildContext();
  TfLiteRegistration reg{};
  BlockFfwOperationParser parser;
  EXPECT_THAT(parser.IsSupported(&ctx, &node, &reg),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BlockFfwParserTest, RejectsUnsupportedActivationAndDynamicQuantSpec) {
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
  BlockFfwOperationParser parser;

  {
    std::vector<uint8_t> bad_act = BuildFlexAttrs("gelu", false);
    TfLiteStablehloCompositeParams params{};
    params.attributes = bad_act.data();
    params.attributes_size = bad_act.size();
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    node.builtin_data = &params;
    EXPECT_THAT(parser.IsSupported(&ctx, &node, &reg),
                StatusIs(absl::StatusCode::kUnimplemented));
  }
  {
    std::vector<uint8_t> dyn_spec = BuildFlexAttrs("silu", false, false, true);
    TfLiteStablehloCompositeParams params{};
    params.attributes = dyn_spec.data();
    params.attributes_size = dyn_spec.size();
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    node.builtin_data = &params;
    EXPECT_THAT(parser.IsSupported(&ctx, &node, &reg),
                StatusIs(absl::StatusCode::kUnimplemented));
  }
}

TEST(BlockFfwParserTest, RejectsOutOfRangeWeightTensorIndices) {
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

  TfLiteContext ctx = arena.BuildContext();
  const int oob_id = static_cast<int>(ctx.tensors_size);
  IntArrayHolder outputs({out_id});
  ResolvedBlockFfwPlan plan;

  // Null tensors array.
  {
    IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    EXPECT_THAT(ResolveBlockFfwPlan(/*tensors=*/nullptr, ctx.tensors_size, node,
                                    &plan),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Negative (`kTfLiteOptionalTensor == -1`) and out-of-range weight indices in
  // every weight slot (U1, U2, D1, D2).
  for (const int bad_id : {kTfLiteOptionalTensor, oob_id}) {
    for (const int slot : {1, 2, 3, 4, 5, 6}) {
      IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
      inputs.ptr->data[slot] = bad_id;
      TfLiteNode node{};
      node.inputs = inputs.ptr;
      node.outputs = outputs.ptr;
      EXPECT_THAT(
          ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node, &plan),
          StatusIs(absl::StatusCode::kInvalidArgument));
    }
  }
}

TEST(BlockFfwParserTest, RejectsInvalidAndTruncatedWeightBuffers) {
  auto run_with_u1_0 = [](auto&& make_u1_0) {
    SyntheticCompositeArena arena;
    const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
    const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
    const int w_u1_0 = make_u1_0(arena);
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
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    TfLiteContext ctx = arena.BuildContext();
    ResolvedBlockFfwPlan plan;
    return ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node, &plan);
  };

  // Float32 weight with null host pointer.
  EXPECT_THAT(
      run_with_u1_0([](SyntheticCompositeArena& arena) {
        const int id = arena.AddConstFloatTensor({8, 8}, 0.5f);
        arena.MutableTensor(id).data.raw_const = nullptr;
        return id;
      }),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Truncated float32 weight buffer.
  EXPECT_THAT(
      run_with_u1_0([](SyntheticCompositeArena& arena) {
        const int id = arena.AddConstFloatTensor({8, 8}, 0.5f);
        arena.MutableTensor(id).bytes -= 1;
        return id;
      }),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Truncated float16 weight buffer.
  EXPECT_THAT(
      run_with_u1_0([](SyntheticCompositeArena& arena) {
        const int id = arena.AddConstFloat16Tensor({8, 8}, 0x3C00);
        arena.MutableTensor(id).bytes -= 1;
        return id;
      }),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Truncated BlockwiseV2 quantized weight buffer.
  EXPECT_THAT(
      run_with_u1_0([](SyntheticCompositeArena& arena) {
        const int id =
            arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
        arena.MutableTensor(id).bytes -= 1;
        return id;
      }),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Affine quantized weight with non-zero quantized_dimension.
  EXPECT_THAT(
      run_with_u1_0([](SyntheticCompositeArena& arena) {
        return arena.AddAffineQuantizedInt8Tensor({8, 8},
                                                  /*quantized_dimension=*/1);
      }),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Truncated affine quantized weight buffer.
  EXPECT_THAT(
      run_with_u1_0([](SyntheticCompositeArena& arena) {
        const int id = arena.AddAffineQuantizedInt8Tensor({8, 8});
        arena.MutableTensor(id).bytes -= 1;
        return id;
      }),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Valid affine quantized weight is accepted.
  EXPECT_OK(run_with_u1_0([](SyntheticCompositeArena& arena) {
    return arena.AddAffineQuantizedInt8Tensor({8, 8});
  }));

  // Externalized quantized weight (null host data, bytes == 0) is accepted for
  // sharing, while null host data with a non-zero truncated byte size is not.
  EXPECT_OK(run_with_u1_0([](SyntheticCompositeArena& arena) {
    const int id =
        arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
    arena.MutableTensor(id).data.raw_const = nullptr;
    arena.MutableTensor(id).bytes = 0;
    return id;
  }));
  EXPECT_THAT(
      run_with_u1_0([](SyntheticCompositeArena& arena) {
        const int id =
            arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
        arena.MutableTensor(id).data.raw_const = nullptr;
        arena.MutableTensor(id).bytes = 1;
        return id;
      }),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BlockFfwParserTest, IsSupportedRejectsMalformedWeightIndexAndBuffer) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int w_u1_0 = arena.AddConstFloatTensor({8, 8}, 0.5f);
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
  IntArrayHolder outputs({out_id});
  TfLiteRegistration reg{};
  BlockFfwOperationParser parser;

  // Negative weight index (`kTfLiteOptionalTensor == -1`).
  {
    IntArrayHolder inputs(
        {x_id, w_u1_0, w_u1_1, kTfLiteOptionalTensor, w_d1, w_d2_0, w_d2_1});
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    TfLiteContext ctx = arena.BuildContext();
    EXPECT_THAT(parser.IsSupported(&ctx, &node, &reg),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Float weight with truncated host buffer.
  {
    arena.MutableTensor(w_u1_0).bytes -= 1;
    IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    TfLiteContext ctx = arena.BuildContext();
    EXPECT_THAT(parser.IsSupported(&ctx, &node, &reg),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
}  // namespace litert::ml_drift
