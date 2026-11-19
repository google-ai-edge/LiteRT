// Copyright 2025 Google LLC.
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

#include "ml_drift_delegate/tflite/model_builder_helper.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/core/c/common.h"
#include "tflite/testing/matchers.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::tflite::SimpleConstTensor;

struct IntArrayWrapper {
  IntArrayWrapper() = delete;
  IntArrayWrapper(std::initializer_list<int> dims) {
    array = TfLiteIntArrayCreate(dims.size());
    int i = 0;
    for (int dim : dims) array->data[i++] = dim;
  }
  ~IntArrayWrapper() { TfLiteIntArrayFree(array); }
  TfLiteIntArray* array;
};

TEST(ModelBuilderHelperTest, CopyDataDifferentSize) {
  int32_t data[4] = {1, 2, 3, 4};
  SimpleConstTensor tensor(kTfLiteInt32, {4}, absl::MakeSpan(data));

  int32_t dst[4];
  CopyData(tensor, dst);

  EXPECT_THAT(dst, ElementsAre(1, 2, 3, 4));
}

TEST(ModelBuilderHelperTest, Broadcastable1d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_1d0 = /*         */{01};
  EXPECT_TRUE(IsBroadcastable(dims_1d0.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_1d0.array));

  IntArrayWrapper dims_1d1 = /*         */{32};
  EXPECT_TRUE(IsBroadcastable(dims_1d1.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_1d1.array));
}

TEST(ModelBuilderHelperTest, NotBroadcastable1d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_1d = /*          */{31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_1d.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_1d.array));
}

TEST(ModelBuilderHelperTest, Broadcastable2d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_2d0 = /*     */{01, 01};
  EXPECT_TRUE(IsBroadcastable(dims_2d0.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_2d0.array));

  IntArrayWrapper dims_2d1 = /*     */{01, 32};
  EXPECT_TRUE(IsBroadcastable(dims_2d1.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_2d1.array));

  IntArrayWrapper dims_2d2 = /*     */{64, 01};
  EXPECT_TRUE(IsBroadcastable(dims_2d2.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_2d2.array));

  IntArrayWrapper dims_2d3 = /*     */{64, 32};
  EXPECT_TRUE(IsBroadcastable(dims_2d3.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_2d3.array));
}

TEST(ModelBuilderHelperTest, NotBroadcastable2d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_2d0 = /*     */{01, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d0.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d0.array));

  IntArrayWrapper dims_2d1 = /*     */{63, 01};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d1.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d1.array));

  IntArrayWrapper dims_2d2 = /*     */{63, 31};
  //                                   ^^  ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d2.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d2.array));

  IntArrayWrapper dims_2d3 = /*     */{63, 32};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d3.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d3.array));

  IntArrayWrapper dims_2d4 = /*     */{64, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d4.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d4.array));
}

TEST(ModelBuilderHelperTest, Broadcastable3d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_3d0 = /**/{001, 01, 01};
  EXPECT_TRUE(IsBroadcastable(dims_3d0.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d0.array));

  IntArrayWrapper dims_3d1 = /**/{001, 01, 32};
  EXPECT_TRUE(IsBroadcastable(dims_3d1.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d1.array));

  IntArrayWrapper dims_3d2 = /**/{001, 64, 01};
  EXPECT_TRUE(IsBroadcastable(dims_3d2.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d2.array));

  IntArrayWrapper dims_3d3 = /**/{001, 64, 32};
  EXPECT_TRUE(IsBroadcastable(dims_3d3.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d3.array));

  IntArrayWrapper dims_3d4 = /**/{128, 01, 01};
  EXPECT_TRUE(IsBroadcastable(dims_3d4.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d4.array));

  IntArrayWrapper dims_3d5 = /**/{128, 01, 32};
  EXPECT_TRUE(IsBroadcastable(dims_3d5.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d5.array));

  IntArrayWrapper dims_3d6 = /**/{128, 64, 01};
  EXPECT_TRUE(IsBroadcastable(dims_3d6.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d6.array));

  IntArrayWrapper dims_3d7 = /**/{128, 64, 32};
  EXPECT_TRUE(IsBroadcastable(dims_3d7.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d7.array));
}

TEST(ModelBuilderHelperTest, NotBroadcastable3d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_3d0 = /**/{001, 01, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d0.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d0.array));

  IntArrayWrapper dims_3d1 = /**/{001, 63, 01};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d1.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d1.array));

  IntArrayWrapper dims_3d2 = /**/{001, 63, 31};
  //                                   ^^  ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d2.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d2.array));

  IntArrayWrapper dims_3d3 = /**/{001, 63, 32};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d3.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d3.array));

  IntArrayWrapper dims_3d4 = /**/{001, 64, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d4.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d4.array));

  IntArrayWrapper dims_3d5 = /**/{127, 01, 01};
  //                              ^^^         not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d5.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d5.array));

  IntArrayWrapper dims_3d6 = /**/{127, 01, 31};
  //                              ^^^      ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d6.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d6.array));

  IntArrayWrapper dims_3d7 = /**/{127, 01, 32};
  //                              ^^^         not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d7.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d7.array));

  IntArrayWrapper dims_3d8 = /**/{127, 63, 01};
  //                              ^^^  ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d8.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d8.array));

  IntArrayWrapper dims_3d9 = /**/{127, 63, 31};
  //                              ^^^  ^^  ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d9.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d9.array));

  IntArrayWrapper dims_3da = /**/{127, 63, 32};
  //                              ^^^  ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3da.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3da.array));

  IntArrayWrapper dims_3db = /**/{127, 64, 01};
  //                              ^^^         not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3db.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3db.array));

  IntArrayWrapper dims_3dc = /**/{127, 64, 31};
  //                              ^^^      ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3dc.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3dc.array));

  IntArrayWrapper dims_3dd = /**/{127, 64, 32};
  //                              ^^^         not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3dd.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3dd.array));

  IntArrayWrapper dims_3de = /**/{128, 01, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3de.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3de.array));

  IntArrayWrapper dims_3df = /**/{128, 63, 01};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3df.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3df.array));

  IntArrayWrapper dims_3dg = /**/{128, 63, 31};
  //                                   ^^  ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3dg.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3dg.array));

  IntArrayWrapper dims_3dh = /**/{128, 63, 32};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3dh.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3dh.array));

  IntArrayWrapper dims_3di = /**/{128, 64, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3di.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3di.array));
}

TfLiteTensor MakeConstTensor(TfLiteType dtype, TfLiteIntArray* dims,
                             const void* buf, size_t bytes) {
  TfLiteTensor t{};
  t.type = dtype;
  t.dims = dims;
  t.data.raw_const = static_cast<const char*>(buf);
  t.bytes = bytes;
  t.allocation_type = kTfLiteMmapRo;
  return t;
}

TEST(ModelBuilderHelperTest, BlockwiseV2BatchedWeightsHioAndHoi) {
  // 2 batches x 3 rows x 4 cols = 24 INT2 values (6 bytes).
  // Byte 0x4E packs (-2, -1, 0, 1) in 2-bit two's complement (0b10, 0b11, 0b00,
  // 0b01 from LSB to MSB).
  int8_t packed_int2[6] = {
      0x4E, 0x4E, 0x4E, 0x4E, 0x4E, 0x4E,
  };

  // Case 1: [H=2, I=4, O=3] with block_shape [1, 4, 1], scales [2, 1, 3].
  float scales_hio[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float zps_hio[6] = {-0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f};
  IntArrayWrapper w_dims_hio = {2, 4, 3};
  IntArrayWrapper s_dims_hio = {2, 1, 3};
  IntArrayWrapper bshape_hio = {1, 4, 1};
  TfLiteBlockwiseQuantizationV2 q_hio = {
      .scale = 1, .zero_point = 2, .block_shape = bshape_hio.array};
  TfLiteTensor tensors_hio[3] = {
      MakeConstTensor(kTfLiteInt2, w_dims_hio.array, packed_int2,
                      sizeof(packed_int2)),
      MakeConstTensor(kTfLiteFloat32, s_dims_hio.array, scales_hio,
                      sizeof(scales_hio)),
      MakeConstTensor(kTfLiteFloat32, s_dims_hio.array, zps_hio,
                      sizeof(zps_hio)),
  };
  tensors_hio[0].quantization.type = kTfLiteBlockwiseQuantizationV2;
  tensors_hio[0].quantization.params = &q_hio;

  ASSERT_OK_AND_ASSIGN(BlockwiseV2Params params_hio,
                       GetBlockwiseV2Params(tensors_hio, 3, tensors_hio[0]));
  BlockwiseV2WeightsInfo info_hio;
  ASSERT_OK(GetBlockwiseV2WeightsInfo(tensors_hio[0], params_hio, &info_hio,
                                      /*oi_transposed=*/true));
  EXPECT_EQ(info_hio.weights_shape, ::ml_drift::OHWI(3, 2, 1, 4));
  EXPECT_EQ(info_hio.scale_shape, ::ml_drift::OHWI(3, 2, 1, 1));
  EXPECT_TRUE(info_hio.has_zero_point);
  EXPECT_TRUE(info_hio.oi_transposed);
  EXPECT_THAT(
      ReadBlockwiseV2WeightsParam(tensors_hio[0], *params_hio.scale, info_hio),
      ElementsAre(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f));

  // Case 2: [H=2, O=3, I=4] with block_shape [1, 1, 4], scales [2, 3, 1].
  float scales_hoi[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float zps_hoi[6] = {-0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f};
  IntArrayWrapper w_dims_hoi = {2, 3, 4};
  IntArrayWrapper s_dims_hoi = {2, 3, 1};
  IntArrayWrapper bshape_hoi = {1, 1, 4};
  TfLiteBlockwiseQuantizationV2 q_hoi = {
      .scale = 1, .zero_point = 2, .block_shape = bshape_hoi.array};
  TfLiteTensor tensors_hoi[3] = {
      MakeConstTensor(kTfLiteInt2, w_dims_hoi.array, packed_int2,
                      sizeof(packed_int2)),
      MakeConstTensor(kTfLiteFloat32, s_dims_hoi.array, scales_hoi,
                      sizeof(scales_hoi)),
      MakeConstTensor(kTfLiteFloat32, s_dims_hoi.array, zps_hoi,
                      sizeof(zps_hoi)),
  };
  tensors_hoi[0].quantization.type = kTfLiteBlockwiseQuantizationV2;
  tensors_hoi[0].quantization.params = &q_hoi;

  ASSERT_OK_AND_ASSIGN(BlockwiseV2Params params_hoi,
                       GetBlockwiseV2Params(tensors_hoi, 3, tensors_hoi[0]));
  BlockwiseV2WeightsInfo info_hoi;
  ASSERT_OK(GetBlockwiseV2WeightsInfo(tensors_hoi[0], params_hoi, &info_hoi,
                                      /*oi_transposed=*/false));
  EXPECT_EQ(info_hoi.weights_shape, ::ml_drift::OHWI(3, 2, 1, 4));
  EXPECT_EQ(info_hoi.scale_shape, ::ml_drift::OHWI(3, 2, 1, 1));
  EXPECT_TRUE(info_hoi.has_zero_point);
  EXPECT_FALSE(info_hoi.oi_transposed);
  EXPECT_THAT(
      ReadBlockwiseV2WeightsParam(tensors_hoi[0], *params_hoi.scale, info_hoi),
      ElementsAre(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f));
  EXPECT_THAT(ReadBlockwiseV2WeightsParam(tensors_hoi[0],
                                          *params_hoi.zero_point, info_hoi),
              ElementsAre(-0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f));

  float dequant[24];
  DequantizeBlockwiseV2(tensors_hoi[0], params_hoi, dequant);
  // Row 0 of batch 0 has scale 1.0f and zp -0.5f: (-1.5, -0.5, 0.5, 1.5).
  EXPECT_FLOAT_EQ(dequant[0], -1.5f);
  EXPECT_FLOAT_EQ(dequant[1], -0.5f);
  EXPECT_FLOAT_EQ(dequant[2], 0.5f);
  EXPECT_FLOAT_EQ(dequant[3], 1.5f);
}

TEST(ModelBuilderHelperTest, BlockwiseV2Weights2dMultiBlockBroadcastAndNoZp) {
  // 2D [O=2, I=4] INT8 weights with block_shape [1, 2] -> 2 blocks along I.
  int8_t weights[8] = {1, 2, 3, 4, -1, -2, -3, -4};
  float scales[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  // Broadcast zero point [1, 1] -> -0.5 for every block.
  float zp_scalar[1] = {-0.5f};
  IntArrayWrapper w_dims = {2, 4};
  IntArrayWrapper s_dims = {2, 2};
  IntArrayWrapper zp_dims = {1, 1};
  IntArrayWrapper bshape = {1, 2};

  // With broadcast zero point.
  TfLiteBlockwiseQuantizationV2 q_with_zp = {
      .scale = 1, .zero_point = 2, .block_shape = bshape.array};
  TfLiteTensor tensors[3] = {
      MakeConstTensor(kTfLiteInt8, w_dims.array, weights, sizeof(weights)),
      MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales)),
      MakeConstTensor(kTfLiteFloat32, zp_dims.array, zp_scalar,
                      sizeof(zp_scalar)),
  };
  tensors[0].quantization.type = kTfLiteBlockwiseQuantizationV2;
  tensors[0].quantization.params = &q_with_zp;

  ASSERT_OK_AND_ASSIGN(BlockwiseV2Params params,
                       GetBlockwiseV2Params(tensors, 3, tensors[0]));
  BlockwiseV2WeightsInfo info;
  ASSERT_OK(GetBlockwiseV2WeightsInfo(tensors[0], params, &info));
  EXPECT_EQ(info.weights_shape, ::ml_drift::OHWI(2, 1, 1, 4));
  EXPECT_EQ(info.scale_shape, ::ml_drift::OHWI(2, 1, 1, 2));
  EXPECT_TRUE(info.has_zero_point);
  EXPECT_THAT(ReadBlockwiseV2WeightsParam(tensors[0], *params.scale, info),
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  EXPECT_THAT(ReadBlockwiseV2WeightsParam(tensors[0], *params.zero_point, info),
              ElementsAre(-0.5f, -0.5f, -0.5f, -0.5f));

  float dequant[8];
  DequantizeBlockwiseV2(tensors[0], params, dequant);
  EXPECT_THAT(dequant, ElementsAre(1.5f, 2.5f, 7.0f, 9.0f, -1.5f, -4.5f, -10.0f,
                                   -14.0f));

  // Absent zero point (kTfLiteOptionalTensor) and broadcast scale [1, 1].
  float scale_scalar[1] = {0.5f};
  TfLiteBlockwiseQuantizationV2 q_no_zp = {.scale = 1,
                                           .zero_point = kTfLiteOptionalTensor,
                                           .block_shape = bshape.array};
  TfLiteTensor tensors_no_zp[2] = {
      MakeConstTensor(kTfLiteInt8, w_dims.array, weights, sizeof(weights)),
      MakeConstTensor(kTfLiteFloat32, zp_dims.array, scale_scalar,
                      sizeof(scale_scalar)),
  };
  tensors_no_zp[0].quantization.type = kTfLiteBlockwiseQuantizationV2;
  tensors_no_zp[0].quantization.params = &q_no_zp;

  ASSERT_OK_AND_ASSIGN(
      BlockwiseV2Params params_no_zp,
      GetBlockwiseV2Params(tensors_no_zp, 2, tensors_no_zp[0]));
  EXPECT_EQ(params_no_zp.zero_point, nullptr);
  BlockwiseV2WeightsInfo info_no_zp;
  ASSERT_OK(
      GetBlockwiseV2WeightsInfo(tensors_no_zp[0], params_no_zp, &info_no_zp));
  EXPECT_EQ(info_no_zp.scale_shape, ::ml_drift::OHWI(2, 1, 1, 1));
  EXPECT_FALSE(info_no_zp.has_zero_point);
  EXPECT_THAT(ReadBlockwiseV2WeightsParam(tensors_no_zp[0], *params_no_zp.scale,
                                          info_no_zp),
              ElementsAre(0.5f, 0.5f));

  DequantizeBlockwiseV2(tensors_no_zp[0], params_no_zp, dequant);
  EXPECT_THAT(dequant,
              ElementsAre(0.5f, 1.0f, 1.5f, 2.0f, -0.5f, -1.0f, -1.5f, -2.0f));
}

TEST(ModelBuilderHelperTest, BlockwiseV2Weights4dHioAndHoi) {
  // 1 x 2 x 4 x 2 = 16 INT4 values (8 bytes).
  int8_t packed_int4[8] = {};
  float scales[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  // [1, H=2, I=4, O=2] with block_shape [1, 1, 2, 1] -> scale [1, 2, 2, 2].
  IntArrayWrapper w_dims_hio = {1, 2, 4, 2};
  IntArrayWrapper s_dims_hio = {1, 2, 2, 2};
  IntArrayWrapper bshape_hio = {1, 1, 2, 1};
  TfLiteBlockwiseQuantizationV2 q_hio = {.scale = 1,
                                         .zero_point = kTfLiteOptionalTensor,
                                         .block_shape = bshape_hio.array};
  TfLiteTensor tensors_hio[2] = {
      MakeConstTensor(kTfLiteInt4, w_dims_hio.array, packed_int4,
                      sizeof(packed_int4)),
      MakeConstTensor(kTfLiteFloat32, s_dims_hio.array, scales, sizeof(scales)),
  };
  tensors_hio[0].quantization.type = kTfLiteBlockwiseQuantizationV2;
  tensors_hio[0].quantization.params = &q_hio;

  ASSERT_OK_AND_ASSIGN(BlockwiseV2Params params_hio,
                       GetBlockwiseV2Params(tensors_hio, 2, tensors_hio[0]));
  BlockwiseV2WeightsInfo info_hio;
  ASSERT_OK(GetBlockwiseV2WeightsInfo(tensors_hio[0], params_hio, &info_hio,
                                      /*oi_transposed=*/true));
  EXPECT_EQ(info_hio.weights_shape, ::ml_drift::OHWI(2, 2, 1, 4));
  EXPECT_EQ(info_hio.scale_shape, ::ml_drift::OHWI(2, 2, 1, 2));
  EXPECT_THAT(
      ReadBlockwiseV2WeightsParam(tensors_hio[0], *params_hio.scale, info_hio),
      ElementsAre(1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f));

  // [1, H=2, O=2, I=4] with block_shape [1, 1, 1, 2] -> scale [1, 2, 2, 2].
  IntArrayWrapper w_dims_hoi = {1, 2, 2, 4};
  IntArrayWrapper s_dims_hoi = {1, 2, 2, 2};
  IntArrayWrapper bshape_hoi = {1, 1, 1, 2};
  TfLiteBlockwiseQuantizationV2 q_hoi = {.scale = 1,
                                         .zero_point = kTfLiteOptionalTensor,
                                         .block_shape = bshape_hoi.array};
  TfLiteTensor tensors_hoi[2] = {
      MakeConstTensor(kTfLiteInt4, w_dims_hoi.array, packed_int4,
                      sizeof(packed_int4)),
      MakeConstTensor(kTfLiteFloat32, s_dims_hoi.array, scales, sizeof(scales)),
  };
  tensors_hoi[0].quantization.type = kTfLiteBlockwiseQuantizationV2;
  tensors_hoi[0].quantization.params = &q_hoi;

  ASSERT_OK_AND_ASSIGN(BlockwiseV2Params params_hoi,
                       GetBlockwiseV2Params(tensors_hoi, 2, tensors_hoi[0]));
  BlockwiseV2WeightsInfo info_hoi;
  ASSERT_OK(GetBlockwiseV2WeightsInfo(tensors_hoi[0], params_hoi, &info_hoi,
                                      /*oi_transposed=*/false));
  EXPECT_EQ(info_hoi.weights_shape, ::ml_drift::OHWI(2, 2, 1, 4));
  EXPECT_EQ(info_hoi.scale_shape, ::ml_drift::OHWI(2, 2, 1, 2));
  EXPECT_THAT(
      ReadBlockwiseV2WeightsParam(tensors_hoi[0], *params_hoi.scale, info_hoi),
      ElementsAre(1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f));
}

TEST(ModelBuilderHelperTest, BlockwiseV2ParamsValidationErrors) {
  int8_t weights[8] = {};
  float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  int8_t zps[4] = {};
  IntArrayWrapper w_dims = {2, 4};
  IntArrayWrapper s_dims = {2, 2};
  IntArrayWrapper bshape = {1, 2};

  auto make_valid = [&]() {
    std::vector<TfLiteTensor> tensors = {
        MakeConstTensor(kTfLiteInt8, w_dims.array, weights, sizeof(weights)),
        MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales)),
        MakeConstTensor(kTfLiteInt8, s_dims.array, zps, sizeof(zps)),
    };
    return tensors;
  };

  // Invalid scale index.
  {
    auto tensors = make_valid();
    TfLiteBlockwiseQuantizationV2 q = {.scale = 5,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape.array};
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(
        GetBlockwiseV2Params(tensors.data(), tensors.size(), tensors[0]),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Invalid zero point index.
  {
    auto tensors = make_valid();
    TfLiteBlockwiseQuantizationV2 q = {
        .scale = 1, .zero_point = 9, .block_shape = bshape.array};
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(
        GetBlockwiseV2Params(tensors.data(), tensors.size(), tensors[0]),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Non-constant scale tensor.
  {
    auto tensors = make_valid();
    tensors[1].allocation_type = kTfLiteArenaRw;
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape.array};
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(
        GetBlockwiseV2Params(tensors.data(), tensors.size(), tensors[0]),
        StatusIs(absl::StatusCode::kUnimplemented));
  }

  // Mismatched rank on scale tensor.
  {
    auto tensors = make_valid();
    IntArrayWrapper bad_rank = {2, 2, 1};
    tensors[1].dims = bad_rank.array;
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape.array};
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(
        GetBlockwiseV2Params(tensors.data(), tensors.size(), tensors[0]),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Mismatched dim extent on scale tensor (neither 1 nor num_blocks).
  {
    auto tensors = make_valid();
    IntArrayWrapper bad_extent = {2, 3};
    tensors[1].dims = bad_extent.array;
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape.array};
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(
        GetBlockwiseV2Params(tensors.data(), tensors.size(), tensors[0]),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Undersized scale buffer.
  {
    auto tensors = make_valid();
    tensors[1].bytes = sizeof(float) * 3;
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape.array};
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(
        GetBlockwiseV2Params(tensors.data(), tensors.size(), tensors[0]),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Unsupported scale dtype (INT8 scale is rejected).
  {
    auto tensors = make_valid();
    tensors[1].type = kTfLiteInt8;
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape.array};
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(
        GetBlockwiseV2Params(tensors.data(), tensors.size(), tensors[0]),
        StatusIs(absl::StatusCode::kUnimplemented));
  }

  // Unsupported zero point dtype (FLOAT16 zero point is rejected).
  {
    auto tensors = make_valid();
    tensors[2].type = kTfLiteFloat16;
    TfLiteBlockwiseQuantizationV2 q = {
        .scale = 1, .zero_point = 2, .block_shape = bshape.array};
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(
        GetBlockwiseV2Params(tensors.data(), tensors.size(), tensors[0]),
        StatusIs(absl::StatusCode::kUnimplemented));
  }

  // Unsupported quantized tensor dtype (FLOAT32).
  {
    auto tensors = make_valid();
    tensors[0].type = kTfLiteFloat32;
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape.array};
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(
        GetBlockwiseV2Params(tensors.data(), tensors.size(), tensors[0]),
        StatusIs(absl::StatusCode::kUnimplemented));
  }

  // Overflow-safe num_blocks and num_elements calculation on huge dimensions.
  {
    constexpr int kMaxInt = std::numeric_limits<int>::max();
    IntArrayWrapper huge_dims = {kMaxInt, kMaxInt};
    IntArrayWrapper huge_bshape = {1, 1};
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = huge_bshape.array};
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, huge_dims.array, weights, sizeof(weights)),
        MakeConstTensor(kTfLiteFloat32, huge_dims.array, scales,
                        sizeof(scales)),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    EXPECT_THAT(GetBlockwiseV2Params(tensors, 2, tensors[0]),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(ModelBuilderHelperTest, BlockwiseV2WeightsInfoValidationErrors) {
  int8_t weights[16] = {};
  float scales[8] = {};
  float zps[8] = {};

  // Uneven blocks (in_channels % block != 0) when not broadcast.
  {
    IntArrayWrapper w_dims = {2, 5};
    IntArrayWrapper s_dims = {2, 2};
    IntArrayWrapper bshape = {1, 4};
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape.array};
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, w_dims.array, weights, 10),
        MakeConstTensor(kTfLiteFloat32, s_dims.array, scales,
                        sizeof(float) * 4),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    ASSERT_OK_AND_ASSIGN(BlockwiseV2Params params,
                         GetBlockwiseV2Params(tensors, 2, tensors[0]));
    BlockwiseV2WeightsInfo info;
    EXPECT_THAT(GetBlockwiseV2WeightsInfo(tensors[0], params, &info),
                StatusIs(absl::StatusCode::kUnimplemented));
  }

  // Unsupported rank (1-D) and 4-D with leading dim != 1.
  {
    IntArrayWrapper w_dims_1d = {4};
    IntArrayWrapper s_dims_1d = {1};
    IntArrayWrapper bshape_1d = {4};
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape_1d.array};
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, w_dims_1d.array, weights, 4),
        MakeConstTensor(kTfLiteFloat32, s_dims_1d.array, scales, sizeof(float)),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    ASSERT_OK_AND_ASSIGN(BlockwiseV2Params params,
                         GetBlockwiseV2Params(tensors, 2, tensors[0]));
    BlockwiseV2WeightsInfo info;
    EXPECT_THAT(GetBlockwiseV2WeightsInfo(tensors[0], params, &info),
                StatusIs(absl::StatusCode::kUnimplemented));
  }
  {
    IntArrayWrapper w_dims_4d = {2, 1, 2, 4};
    IntArrayWrapper s_dims_4d = {2, 1, 2, 1};
    IntArrayWrapper bshape_4d = {1, 1, 1, 4};
    TfLiteBlockwiseQuantizationV2 q = {.scale = 1,
                                       .zero_point = kTfLiteOptionalTensor,
                                       .block_shape = bshape_4d.array};
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, w_dims_4d.array, weights, 16),
        MakeConstTensor(kTfLiteFloat32, s_dims_4d.array, scales,
                        sizeof(float) * 4),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    ASSERT_OK_AND_ASSIGN(BlockwiseV2Params params,
                         GetBlockwiseV2Params(tensors, 2, tensors[0]));
    BlockwiseV2WeightsInfo info;
    EXPECT_THAT(GetBlockwiseV2WeightsInfo(tensors[0], params, &info),
                StatusIs(absl::StatusCode::kUnimplemented));
  }

  // Mismatched non-broadcast block counts between scale and zero_point.
  {
    IntArrayWrapper w_dims = {2, 8};
    IntArrayWrapper s_dims = {2, 4};
    IntArrayWrapper zp_dims = {2, 2};
    IntArrayWrapper bshape = {1, 2};
    TfLiteBlockwiseQuantizationV2 q = {
        .scale = 1, .zero_point = 2, .block_shape = bshape.array};
    TfLiteTensor w = MakeConstTensor(kTfLiteInt8, w_dims.array, weights, 16);
    w.quantization = {kTfLiteBlockwiseQuantizationV2, &q};
    TfLiteTensor s =
        MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales));
    TfLiteTensor zp =
        MakeConstTensor(kTfLiteFloat32, zp_dims.array, zps, sizeof(float) * 4);
    BlockwiseV2Params params = {.scale = &s, .zero_point = &zp};
    BlockwiseV2WeightsInfo info;
    EXPECT_THAT(GetBlockwiseV2WeightsInfo(w, params, &info),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
}  // namespace litert::ml_drift
