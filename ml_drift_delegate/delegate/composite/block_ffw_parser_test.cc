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
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "fp16.h"  // from @FP16
#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;

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

  int AddAffineQuantizedTensor(TfLiteType type, std::initializer_list<int> dims,
                               std::vector<int8_t> packed_data,
                               std::initializer_list<float> scales,
                               std::initializer_list<int> zero_points,
                               int quantized_dimension = 0) {
    dims_.emplace_back(dims);
    byte_buffers_.push_back(std::move(packed_data));
    float_arrays_.emplace_back(scales);
    dims_.emplace_back(zero_points);
    auto q = std::make_unique<TfLiteAffineQuantization>();
    q->scale = float_arrays_.back().ptr;
    q->zero_point = dims_.back().ptr;
    q->quantized_dimension = quantized_dimension;
    affine_quant_params_.push_back(std::move(q));

    TfLiteTensor t{};
    t.type = type;
    t.dims = dims_[dims_.size() - 2].ptr;
    t.data.raw_const =
        reinterpret_cast<const char*>(byte_buffers_.back().data());
    t.bytes = byte_buffers_.back().size();
    t.allocation_type = kTfLiteMmapRo;
    t.quantization.type = kTfLiteAffineQuantization;
    t.quantization.params = affine_quant_params_.back().get();
    tensors_.push_back(t);
    return static_cast<int>(tensors_.size()) - 1;
  }

  int AddAffineQuantizedInt8Tensor(std::initializer_list<int> dims,
                                   int quantized_dimension = 0) {
    size_t count = 1;
    for (int d : dims) count *= d;
    return AddAffineQuantizedTensor(kTfLiteInt8, dims,
                                    std::vector<int8_t>(count, 1), {0.1f}, {0},
                                    quantized_dimension);
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

TEST(BlockFfwParserTest, ResolvesPlanAndExtractsHostFallback) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 2, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 2, 16});
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
                     /*distinct_adjacent_scales=*/true);
  TfLiteStablehloCompositeParams params{};
  params.name = kBlockFfwCompositeName.data();
  params.attributes = fb_attrs.data();
  params.attributes_size = fb_attrs.size();

  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;
  node.builtin_data = &params;

  TfLiteContext ctx = arena.BuildContext();
  absl::StatusOr<ResolvedBlockFfwPlan> plan =
      ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node);
  ASSERT_OK(plan);
  EXPECT_EQ(plan->tokens, 2);
  EXPECT_EQ(plan->model_dim, 16);
  EXPECT_EQ(plan->k1, 2);
  EXPECT_EQ(plan->k2, 2);
  EXPECT_FALSE(plan->u1_weights[0].is_batched);
  EXPECT_FALSE(plan->u1_weights[0].oi_transposed);
  EXPECT_TRUE(plan->u2_weight.is_batched);
  EXPECT_TRUE(plan->u2_weight.oi_transposed);
  EXPECT_TRUE(plan->d1_weight.is_batched);
  EXPECT_TRUE(plan->d1_weight.oi_transposed);

  // Byte 0x4E = 0b01_00_11_10 unpacks LSB-first to int2 values {-2, -1, 0, 1}.
  // With scale = 0.1f and zero_point = -0.5f, dequantized values cycle
  // {-0.15f, -0.05f, 0.05f, 0.15f}.
  const float kExpectedCycle[4] = {-0.15f, -0.05f, 0.05f, 0.15f};
  const std::vector<float> u1_fb = ExtractBlockFfwHostFallback(
      ctx.tensors, ctx.tensors_size, plan->u1_weights[0]);
  ASSERT_EQ(u1_fb.size(), 64);
  for (size_t i = 0; i < u1_fb.size(); ++i) {
    EXPECT_NEAR(u1_fb[i], kExpectedCycle[i % 4], 1e-6f);
  }
  const std::vector<float> u2_fb = ExtractBlockFfwHostFallback(
      ctx.tensors, ctx.tensors_size, plan->u2_weight);
  ASSERT_EQ(u2_fb.size(), 2 * 8 * 12);
  for (size_t i = 0; i < u2_fb.size(); ++i) {
    EXPECT_NEAR(u2_fb[i], kExpectedCycle[i % 4], 1e-6f);
  }
}

TEST(BlockFfwParserTest, ParsesSharedInt2Weights) {
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
  params.name = kBlockFfwCompositeName.data();
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
    const bool is_middle = shared_info.tflite_tensor_id == w_u2 ||
                           shared_info.tflite_tensor_id == w_d1;
    EXPECT_EQ(shared_info.oi_transposed, is_middle);
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
  params.name = kBlockedFfwCompositeName.data();
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
  int const_count = 0;
  int bmm_count = 0;
  std::vector<::ml_drift::BHWC> const_shapes;
  const std::string fc = ToString(::ml_drift::OperationType::kFullyConnected);
  const std::string constant = ToString(::ml_drift::OperationType::kConstant);
  const std::string bmm = ToString(::ml_drift::OperationType::kBatchedMatmul);
  for (const auto* n : graph.nodes()) {
    if (n->operation.type == fc) {
      ++fc_count;
      const auto* attr = std::any_cast<::ml_drift::FullyConnectedAttributes>(
          &n->operation.attributes);
      ASSERT_NE(attr, nullptr);
      EXPECT_GE(attr->weights.data.size(),
                attr->weights.shape.DimensionsProduct() +
                    XNN_EXTRA_BYTES / sizeof(float));
    } else if (n->operation.type == constant) {
      ++const_count;
      const auto* attr = std::any_cast<::ml_drift::ConstTensorAttributes>(
          &n->operation.attributes);
      ASSERT_NE(attr, nullptr);
      const auto* t = std::get_if<
          ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::kFloat32>>(
          &attr->tensor);
      ASSERT_NE(t, nullptr);
      const_shapes.push_back(t->shape);
    } else if (n->operation.type == bmm) {
      ++bmm_count;
    }
  }
  EXPECT_EQ(fc_count, 4);
  EXPECT_EQ(const_count, 2);
  EXPECT_EQ(bmm_count, 2);
  EXPECT_THAT(const_shapes, ElementsAre(::ml_drift::BHWC(1, 2, 8, 12),
                                        ::ml_drift::BHWC(1, 2, 12, 8)));
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
  params.name = kBlockedFfwCompositeName.data();
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

  // Null tensors array.
  {
    IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    EXPECT_THAT(
        ResolveBlockFfwPlan(/*tensors=*/nullptr, ctx.tensors_size, node),
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
      EXPECT_THAT(ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node),
                  StatusIs(absl::StatusCode::kInvalidArgument));
    }
  }
}

TEST(BlockFfwParserTest, TransposesHoi3dWeightsInHostFallback) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 1, 16});
  const int w_u1_0 = arena.AddConstFloatTensor({8, 8}, 1.0f);
  const int w_u1_1 = arena.AddConstFloatTensor({8, 8}, 1.0f);
  // U2 stored as `[H=2, O=12, I=8]` (`oi_transposed == false`).
  const int w_u2 = arena.AddConstFloatTensor({2, 12, 8}, 0.0f);
  auto* u2_data = const_cast<float*>(arena.MutableTensor(w_u2).data.f);
  for (int h = 0; h < 2; ++h) {
    for (int o = 0; o < 12; ++o) {
      for (int i = 0; i < 8; ++i) {
        u2_data[(h * 12 + o) * 8 + i] =
            static_cast<float>(h * 1000 + o * 10 + i);
      }
    }
  }
  const int w_d1 = arena.AddConstFloatTensor({2, 12, 8}, 1.0f);
  const int w_d2_0 = arena.AddConstFloatTensor({8, 8}, 1.0f);
  const int w_d2_1 = arena.AddConstFloatTensor({8, 8}, 1.0f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;

  TfLiteContext ctx = arena.BuildContext();
  absl::StatusOr<ResolvedBlockFfwPlan> plan =
      ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node);
  ASSERT_OK(plan);
  EXPECT_FALSE(plan->u2_weight.oi_transposed);

  const std::vector<float> u2_hio = ExtractBlockFfwHostFallback(
      ctx.tensors, ctx.tensors_size, plan->u2_weight);
  ASSERT_EQ(u2_hio.size(), 2 * 8 * 12);
  for (int h = 0; h < 2; ++h) {
    for (int i = 0; i < 8; ++i) {
      for (int o = 0; o < 12; ++o) {
        EXPECT_FLOAT_EQ(u2_hio[(h * 8 + i) * 12 + o],
                        static_cast<float>(h * 1000 + o * 10 + i));
      }
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
    return ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node).status();
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

TEST(BlockFfwParserTest, ExtractsExactHostFallbackValuesAcrossDtypes) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 1, 16});

  // U1[0]: float32 [8, 8] with distinct elements.
  const int w_u1_0 = arena.AddConstFloatTensor({8, 8}, 0.0f);
  auto* u1_0_data = const_cast<float*>(arena.MutableTensor(w_u1_0).data.f);
  std::vector<float> expected_f32(64);
  for (int i = 0; i < 64; ++i) {
    expected_f32[i] = 0.25f * static_cast<float>(i - 20);
    u1_0_data[i] = expected_f32[i];
  }

  // U1[1]: float16 [8, 8].
  const int w_u1_1 = arena.AddConstFloat16Tensor({8, 8}, 0);
  auto* u1_1_data = reinterpret_cast<uint16_t*>(
      const_cast<char*>(arena.MutableTensor(w_u1_1).data.raw_const));
  std::vector<float> expected_f16(64);
  for (int i = 0; i < 64; ++i) {
    const float v = 0.5f * static_cast<float>((i % 9) - 4);
    u1_1_data[i] = fp16_ieee_from_fp32_value(v);
    expected_f16[i] = fp16_ieee_to_fp32_value(u1_1_data[i]);
  }

  // U2: BlockwiseV2 int2 stored as `[H=2, O=12, I=8]` (`oi_transposed=false`).
  // Fill byte `(h * 12 + o) * 2 + (i / 4)` with distinct nibbles so transpose
  // `[H, O, I]` -> `[H, I, O]` is verified element-by-element.
  const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 12, 8}, {1, 1, 8},
                                                  {2, 12, 1}, 0.2f, -0.5f);
  auto* u2_bytes = const_cast<int8_t*>(arena.MutableTensor(w_u2).data.int8);
  std::vector<float> expected_u2_hio(2 * 8 * 12);
  for (int h = 0; h < 2; ++h) {
    for (int o = 0; o < 12; ++o) {
      for (int grp = 0; grp < 2; ++grp) {
        uint8_t byte = 0;
        for (int lane = 0; lane < 4; ++lane) {
          const int i = grp * 4 + lane;
          const int qval = ((h + o + i) % 4) - 2;  // in {-2, -1, 0, 1}
          byte |= static_cast<uint8_t>((qval & 0x03) << (2 * lane));
          expected_u2_hio[(h * 8 + i) * 12 + o] =
              0.2f * (static_cast<float>(qval) + 0.5f);
        }
        u2_bytes[(h * 12 + o) * 2 + grp] = static_cast<int8_t>(byte);
      }
    }
  }

  const int w_d1 = arena.AddBlockwiseV2Int2Tensor({2, 12, 8}, {1, 12, 1},
                                                  {2, 1, 8}, 0.1f, -0.5f);

  // D2[0]: per-channel affine int8 [8, 8].
  std::vector<int8_t> d2_0_q(64);
  std::vector<float> expected_affine_i8(64);
  const std::initializer_list<float> d2_0_scales = {0.1f, 0.2f, 0.3f, 0.4f,
                                                    0.5f, 0.6f, 0.7f, 0.8f};
  const std::initializer_list<int> d2_0_zps = {0, 1, -1, 2, -2, 0, 1, -1};
  for (int o = 0; o < 8; ++o) {
    const float sc = *(d2_0_scales.begin() + o);
    const int zp = *(d2_0_zps.begin() + o);
    for (int i = 0; i < 8; ++i) {
      const int8_t qv = static_cast<int8_t>((o * 3 + i) - 10);
      d2_0_q[o * 8 + i] = qv;
      expected_affine_i8[o * 8 + i] = sc * static_cast<float>(qv - zp);
    }
  }
  const int w_d2_0 = arena.AddAffineQuantizedTensor(
      kTfLiteInt8, {8, 8}, d2_0_q, d2_0_scales, d2_0_zps);

  // D2[1]: packed affine int4 [8, 8] (low nibble = even index, high = odd).
  std::vector<int8_t> d2_1_packed(32);
  std::vector<float> expected_affine_i4(64);
  for (int idx = 0; idx < 64; idx += 2) {
    const int q0 = ((idx / 2) % 15) - 7;
    const int q1 = (((idx / 2) + 3) % 15) - 7;
    d2_1_packed[idx / 2] =
        static_cast<int8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    expected_affine_i4[idx] = 0.25f * static_cast<float>(q0 - 1);
    expected_affine_i4[idx + 1] = 0.25f * static_cast<float>(q1 - 1);
  }
  const int w_d2_1 = arena.AddAffineQuantizedTensor(
      kTfLiteInt4, {8, 8}, d2_1_packed, {0.25f}, {1});

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;

  TfLiteContext ctx = arena.BuildContext();
  absl::StatusOr<ResolvedBlockFfwPlan> plan =
      ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node);
  ASSERT_OK(plan);
  EXPECT_FALSE(plan->u2_weight.oi_transposed);

  EXPECT_THAT(ExtractBlockFfwHostFallback(ctx.tensors, ctx.tensors_size,
                                          plan->u1_weights[0]),
              Pointwise(FloatNear(1e-6f), expected_f32));
  EXPECT_THAT(ExtractBlockFfwHostFallback(ctx.tensors, ctx.tensors_size,
                                          plan->u1_weights[1]),
              Pointwise(FloatNear(1e-6f), expected_f16));
  EXPECT_THAT(ExtractBlockFfwHostFallback(ctx.tensors, ctx.tensors_size,
                                          plan->u2_weight),
              Pointwise(FloatNear(1e-6f), expected_u2_hio));
  EXPECT_THAT(ExtractBlockFfwHostFallback(ctx.tensors, ctx.tensors_size,
                                          plan->d2_weights[0]),
              Pointwise(FloatNear(1e-6f), expected_affine_i8));
  EXPECT_THAT(ExtractBlockFfwHostFallback(ctx.tensors, ctx.tensors_size,
                                          plan->d2_weights[1]),
              Pointwise(FloatNear(1e-6f), expected_affine_i4));

  // Also test packed affine int2 [8, 8] directly via a second node.
  std::vector<int8_t> int2_packed(16);
  std::vector<float> expected_affine_i2(64);
  for (int b = 0; b < 16; ++b) {
    uint8_t byte = 0;
    for (int lane = 0; lane < 4; ++lane) {
      const int idx = b * 4 + lane;
      const int qv = ((b + lane) % 4) - 2;
      byte |= static_cast<uint8_t>((qv & 0x03) << (2 * lane));
      expected_affine_i2[idx] = 0.5f * static_cast<float>(qv);
    }
    int2_packed[b] = static_cast<int8_t>(byte);
  }
  const int w_u1_i2 = arena.AddAffineQuantizedTensor(
      kTfLiteInt2, {8, 8}, int2_packed, {0.5f}, {0});
  IntArrayHolder inputs_i2({x_id, w_u1_i2, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  node.inputs = inputs_i2.ptr;
  ctx = arena.BuildContext();
  absl::StatusOr<ResolvedBlockFfwPlan> plan_i2 =
      ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node);
  ASSERT_OK(plan_i2);
  EXPECT_THAT(ExtractBlockFfwHostFallback(ctx.tensors, ctx.tensors_size,
                                          plan_i2->u1_weights[0]),
              Pointwise(FloatNear(1e-6f), expected_affine_i2));
}

TEST(BlockFfwParserTest, HandlesK2OneHoiTransposeAndRejects3dAffine) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 1, 16});
  const int w_u1_0 = arena.AddConstFloatTensor({4, 8}, 1.0f);
  const int w_u1_1 = arena.AddConstFloatTensor({4, 8}, 1.0f);
  // K2 = 1, S = 4, I2 = K1 * S = 8, O2 = 12; U2 is `[1, O=12, I=8]` (HOI).
  const int w_u2 = arena.AddConstFloatTensor({1, 12, 8}, 0.0f);
  auto* u2_data = const_cast<float*>(arena.MutableTensor(w_u2).data.f);
  for (int o = 0; o < 12; ++o) {
    for (int i = 0; i < 8; ++i) {
      u2_data[o * 8 + i] = static_cast<float>(o * 10 + i);
    }
  }
  const int w_d1 = arena.AddConstFloatTensor({1, 12, 8}, 1.0f);
  const int w_d2_0 = arena.AddConstFloatTensor({8, 4}, 1.0f);
  const int w_d2_1 = arena.AddConstFloatTensor({8, 4}, 1.0f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;

  TfLiteContext ctx = arena.BuildContext();
  absl::StatusOr<ResolvedBlockFfwPlan> plan =
      ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node);
  ASSERT_OK(plan);
  EXPECT_EQ(plan->k2, 1);
  EXPECT_TRUE(plan->u2_weight.is_batched);
  EXPECT_FALSE(plan->u2_weight.oi_transposed);

  const std::vector<float> u2_hio = ExtractBlockFfwHostFallback(
      ctx.tensors, ctx.tensors_size, plan->u2_weight);
  ASSERT_EQ(u2_hio.size(), 8 * 12);
  for (int i = 0; i < 8; ++i) {
    for (int o = 0; o < 12; ++o) {
      EXPECT_FLOAT_EQ(u2_hio[i * 12 + o], static_cast<float>(o * 10 + i));
    }
  }

  // K2 = 1 3-D affine quantized weight must be rejected instead of falling
  // through to the 2-D affine branch.
  const int w_u2_affine = arena.AddAffineQuantizedInt8Tensor({1, 8, 12});
  IntArrayHolder bad_inputs(
      {x_id, w_u1_0, w_u1_1, w_u2_affine, w_d1, w_d2_0, w_d2_1});
  node.inputs = bad_inputs.ptr;
  ctx = arena.BuildContext();
  EXPECT_THAT(ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(BlockFfwParserTest, ResolvesSquare3dWeightLayoutTieBreakers) {
  auto resolve_square = [](std::initializer_list<int> u2_block,
                           std::initializer_list<int> u2_scale,
                           std::initializer_list<int> d1_block,
                           std::initializer_list<int> d1_scale) {
    SyntheticCompositeArena arena;
    const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
    const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
    const int w_u1_0 =
        arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
    const int w_u1_1 =
        arena.AddBlockwiseV2Int2Tensor({8, 8}, {1, 8}, {8, 1}, 0.1f, -0.5f);
    // Square `[H=2, 8, 8]` middle weights (`I2 = O2 = 8`).
    const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 8, 8}, u2_block,
                                                    u2_scale, 0.1f, -0.5f);
    const int w_d1 = arena.AddBlockwiseV2Int2Tensor({2, 8, 8}, d1_block,
                                                    d1_scale, 0.1f, -0.5f);
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
    return ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node);
  };

  // Disambiguated by block_shape (`[1, 8, 1]` -> HIO, `[1, 1, 8]` -> HOI).
  ASSERT_OK_AND_ASSIGN(
      const ResolvedBlockFfwPlan by_block,
      resolve_square({1, 8, 1}, {2, 1, 8}, {1, 1, 8}, {2, 8, 1}));
  EXPECT_TRUE(by_block.u2_weight.oi_transposed);
  EXPECT_FALSE(by_block.d1_weight.oi_transposed);

  // Disambiguated by scale tensor extents when block_shape is `[1, 1, 1]`.
  ASSERT_OK_AND_ASSIGN(
      const ResolvedBlockFfwPlan by_scale,
      resolve_square({1, 1, 1}, {2, 1, 8}, {1, 1, 1}, {2, 8, 1}));
  EXPECT_TRUE(by_scale.u2_weight.oi_transposed);
  EXPECT_FALSE(by_scale.d1_weight.oi_transposed);
}

TEST(BlockFfwParserTest, ParsesCustomInitialDataAndRejectsBadFlexBuffers) {
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

  // Valid FlexBuffers passed through `custom_initial_data`.
  {
    std::vector<uint8_t> attrs =
        BuildFlexAttrs("swish", /*include_scales=*/true);
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    node.custom_initial_data = attrs.data();
    node.custom_initial_data_size = static_cast<int>(attrs.size());
    ASSERT_OK_AND_ASSIGN(
        const ResolvedBlockFfwPlan plan,
        ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node));
    ASSERT_TRUE(plan.attrs.u1.in_scale.has_value());
    EXPECT_FLOAT_EQ(*plan.attrs.u1.in_scale, 0.02f);
  }

  // Negative `custom_initial_data_size`.
  {
    const uint8_t dummy = 0;
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    node.custom_initial_data = &dummy;
    node.custom_initial_data_size = -1;
    EXPECT_THAT(ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Malformed FlexBuffers bytes.
  {
    const std::vector<uint8_t> corrupt = {0xFF, 0xFF, 0xFF, 0xFF};
    TfLiteStablehloCompositeParams params{};
    params.attributes = corrupt.data();
    params.attributes_size = corrupt.size();
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    node.builtin_data = &params;
    EXPECT_THAT(ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Valid FlexBuffers whose root is not a map.
  {
    flexbuffers::Builder fbb;
    fbb.Int(42);
    fbb.Finish();
    const std::vector<uint8_t> non_map = fbb.GetBuffer();
    TfLiteStablehloCompositeParams params{};
    params.attributes = non_map.data();
    params.attributes_size = non_map.size();
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    node.builtin_data = &params;
    EXPECT_THAT(ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(BlockFfwParserTest, NullHostDataRulesFor2dAnd3dBlockwiseV2Weights) {
  auto make_arena = []() {
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
    return std::tuple{std::move(arena), x_id, out_id, w_u1_0, w_u1_1,
                      w_u2,             w_d1, w_d2_0, w_d2_1};
  };

  // 2-D externalized BlockwiseV2 weight (`raw_const == nullptr, bytes == 0`)
  // is allowed.
  {
    auto [arena, x_id, out_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1] =
        make_arena();
    arena.MutableTensor(w_u1_0).data.raw_const = nullptr;
    arena.MutableTensor(w_u1_0).bytes = 0;
    IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
    IntArrayHolder outputs({out_id});
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    TfLiteContext ctx = arena.BuildContext();
    EXPECT_OK(ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node));
  }

  // 2-D BlockwiseV2 weight with null host data and non-zero truncated `bytes`
  // is rejected.
  {
    auto [arena, x_id, out_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1] =
        make_arena();
    arena.MutableTensor(w_u1_0).data.raw_const = nullptr;
    arena.MutableTensor(w_u1_0).bytes = 1;
    IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
    IntArrayHolder outputs({out_id});
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    TfLiteContext ctx = arena.BuildContext();
    EXPECT_THAT(ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 3-D batched BlockwiseV2 weight (`U2`) with null host data must be
  // rejected so `IsSupported` falls back cleanly instead of failing later in
  // `SharedMemoryManager::CreateBlockwiseV2BatchedWeightsTensor`.
  {
    auto [arena, x_id, out_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1] =
        make_arena();
    arena.MutableTensor(w_u2).data.raw_const = nullptr;
    arena.MutableTensor(w_u2).bytes = 0;
    IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
    IntArrayHolder outputs({out_id});
    TfLiteNode node{};
    node.inputs = inputs.ptr;
    node.outputs = outputs.ptr;
    TfLiteContext ctx = arena.BuildContext();
    EXPECT_THAT(ResolveBlockFfwPlan(ctx.tensors, ctx.tensors_size, node),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
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

TEST(BlockFfwParserTest, DeduplicatesIdenticalAdjacentScales) {
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
  params.name = kBlockedFfwCompositeName.data();
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

  std::vector<std::string> op_types;
  for (const auto* n : graph.nodes()) {
    op_types.push_back(n->operation.type);
  }
  const std::string qdq =
      ToString(::ml_drift::OperationType::kQuantizeAndDequantize);
  const std::string fc2 =
      ToString(::ml_drift::OperationType::kFullyConnectedInt2);
  // Identical adjacent scales (u2_in == u1_out, d2_in == d1_out) omit the
  // redundant u2_in and d2_in QuantizeAndDequantize nodes.
  EXPECT_THAT(op_types,
              ElementsAre(kBlockFfwUnpackPartType, qdq, fc2, qdq,
                          kBlockFfwUnpackPartType, qdq, fc2, qdq,
                          kBlockFfwPackPartsType, fc2, qdq, kBlockFfwSiluType,
                          qdq, fc2, qdq, kBlockFfwUnpackPartType, fc2, qdq,
                          kBlockFfwUnpackPartType, fc2, qdq,
                          kBlockFfwPackPartsType));
}

TEST(BlockFfwParserTest, K2OneNonSharedMiddleWeightsUseBatchedMatmulFallback) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 2, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat32, {1, 2, 16});
  const int w_u1_0 = arena.AddConstFloatTensor({4, 8}, 1.0f);
  const int w_u1_1 = arena.AddConstFloatTensor({4, 8}, 1.0f);
  // K2 = 1 middle weights: U2 is `[1, O=12, I=8]` (HOI), D1 is `[1, I=12, O=8]`
  // (HIO). Both must route to Constant + BatchedMatmul, never kFullyConnected.
  const int w_u2 = arena.AddConstFloatTensor({1, 12, 8}, 1.0f);
  const int w_d1 = arena.AddConstFloatTensor({1, 12, 8}, 1.0f);
  const int w_d2_0 = arena.AddConstFloatTensor({8, 4}, 1.0f);
  const int w_d2_1 = arena.AddConstFloatTensor({8, 4}, 1.0f);

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;

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
  int bmm_count = 0;
  std::vector<::ml_drift::BHWC> const_shapes;
  const std::string fc = ToString(::ml_drift::OperationType::kFullyConnected);
  const std::string constant = ToString(::ml_drift::OperationType::kConstant);
  const std::string bmm = ToString(::ml_drift::OperationType::kBatchedMatmul);
  for (const auto* n : graph.nodes()) {
    if (n->operation.type == fc) {
      ++fc_count;
    } else if (n->operation.type == constant) {
      const auto* attr = std::any_cast<::ml_drift::ConstTensorAttributes>(
          &n->operation.attributes);
      ASSERT_NE(attr, nullptr);
      const auto* t = std::get_if<
          ::ml_drift::Tensor<::ml_drift::BHWC, ::ml_drift::DataType::kFloat32>>(
          &attr->tensor);
      ASSERT_NE(t, nullptr);
      const_shapes.push_back(t->shape);
    } else if (n->operation.type == bmm) {
      ++bmm_count;
    }
  }
  EXPECT_EQ(fc_count, 4);
  EXPECT_EQ(bmm_count, 2);
  EXPECT_THAT(const_shapes, ElementsAre(::ml_drift::BHWC(1, 1, 8, 12),
                                        ::ml_drift::BHWC(1, 1, 12, 8)));
}

TEST(BlockFfwParserTest, PropagatesHoiAndSupportsSharedInt4AndInt8Weights) {
  SyntheticCompositeArena arena;
  const int x_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  const int out_id = arena.AddRuntimeTensor(kTfLiteFloat16, {1, 1, 16});
  // Mix shared affine int8, affine int4, and BlockwiseV2 int2 `[H, O, I]`.
  const int w_u1_0 = arena.AddAffineQuantizedInt8Tensor({8, 8});
  const int w_u1_1 = arena.AddAffineQuantizedTensor(
      kTfLiteInt4, {8, 8}, std::vector<int8_t>(32, 0x12), {0.1f}, {0});
  // U2 stored as `[H=2, O=12, I=8]` (`oi_transposed == false`).
  const int w_u2 = arena.AddBlockwiseV2Int2Tensor({2, 12, 8}, {1, 1, 8},
                                                  {2, 12, 1}, 0.1f, -0.5f);
  // D1 stored as `[H=2, I=12, O=8]` (`oi_transposed == true`).
  const int w_d1 = arena.AddBlockwiseV2Int2Tensor({2, 12, 8}, {1, 12, 1},
                                                  {2, 1, 8}, 0.1f, -0.5f);
  const int w_d2_0 = arena.AddAffineQuantizedTensor(
      kTfLiteInt4, {8, 8}, std::vector<int8_t>(32, 0x12), {0.1f}, {0});
  const int w_d2_1 = arena.AddAffineQuantizedInt8Tensor({8, 8});

  IntArrayHolder inputs({x_id, w_u1_0, w_u1_1, w_u2, w_d1, w_d2_0, w_d2_1});
  IntArrayHolder outputs({out_id});
  TfLiteNode node{};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;

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
    if (shared_info.tflite_tensor_id == w_u2) {
      EXPECT_FALSE(shared_info.oi_transposed);
    } else if (shared_info.tflite_tensor_id == w_d1) {
      EXPECT_TRUE(shared_info.oi_transposed);
    }
  }

  std::vector<std::string> op_types;
  for (const auto* n : graph.nodes()) {
    op_types.push_back(n->operation.type);
  }
  const std::string fc8 =
      ToString(::ml_drift::OperationType::kFullyConnectedInt8);
  const std::string fc4 =
      ToString(::ml_drift::OperationType::kFullyConnectedInt4);
  const std::string fc2 =
      ToString(::ml_drift::OperationType::kFullyConnectedInt2);
  EXPECT_THAT(op_types,
              ElementsAre(kBlockFfwUnpackPartType, fc8, kBlockFfwUnpackPartType,
                          fc4, kBlockFfwPackPartsType, fc2, kBlockFfwSiluType,
                          fc2, kBlockFfwUnpackPartType, fc4,
                          kBlockFfwUnpackPartType, fc8,
                          kBlockFfwPackPartsType));
}

}  // namespace
}  // namespace litert::ml_drift
