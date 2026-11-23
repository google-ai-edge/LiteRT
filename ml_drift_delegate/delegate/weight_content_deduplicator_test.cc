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

#include "ml_drift_delegate/delegate/weight_content_deduplicator.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "testing/base/public/gunit.h"
#include "tflite/core/c/common.h"

namespace litert::ml_drift {
namespace {

class ScopedTestContext {
 public:
  ~ScopedTestContext() {
    for (TfLiteTensor& t : tensors_) {
      if (t.dims != nullptr) {
        TfLiteIntArrayFree(t.dims);
      }
      TfLiteQuantizationFree(&t.quantization);
    }
  }

  int AddConstTensor(TfLiteType type, const std::vector<int>& dims, void* data,
                     size_t bytes) {
    TfLiteTensor t = {};
    t.type = type;
    t.allocation_type = kTfLiteMmapRo;
    t.dims = TfLiteIntArrayCreate(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      t.dims->data[i] = dims[i];
    }
    t.data.raw = reinterpret_cast<char*>(data);
    t.bytes = bytes;
    t.quantization.type = kTfLiteNoQuantization;
    tensors_.push_back(t);
    return static_cast<int>(tensors_.size()) - 1;
  }

  void SetAffineQuant(int tensor_idx, const std::vector<float>& scales,
                      const std::vector<int>& zero_points, int quant_dim) {
    TfLiteTensor& t = tensors_[tensor_idx];
    auto* params = reinterpret_cast<TfLiteAffineQuantization*>(
        calloc(1, sizeof(TfLiteAffineQuantization)));
    params->scale = TfLiteFloatArrayCreate(scales.size());
    for (size_t i = 0; i < scales.size(); ++i) {
      params->scale->data[i] = scales[i];
    }
    params->zero_point = TfLiteIntArrayCreate(zero_points.size());
    for (size_t i = 0; i < zero_points.size(); ++i) {
      params->zero_point->data[i] = zero_points[i];
    }
    params->quantized_dimension = quant_dim;
    t.quantization.type = kTfLiteAffineQuantization;
    t.quantization.params = params;
  }

  void SetBlockwiseQuant(int tensor_idx, int scale_idx, int zp_idx,
                         int blocksize, int quant_dim) {
    TfLiteTensor& t = tensors_[tensor_idx];
    auto* params = reinterpret_cast<TfLiteBlockwiseQuantization*>(
        calloc(1, sizeof(TfLiteBlockwiseQuantization)));
    params->scale = scale_idx;
    params->zero_point = zp_idx;
    params->blocksize = blocksize;
    params->quantized_dimension = quant_dim;
    t.quantization.type = kTfLiteBlockwiseQuantization;
    t.quantization.params = params;
  }

  TfLiteContext BuildContext() {
    TfLiteContext ctx = {};
    ctx.tensors = tensors_.data();
    ctx.tensors_size = tensors_.size();
    return ctx;
  }

 private:
  std::vector<TfLiteTensor> tensors_;
};

TEST(WeightContentDeduplicatorTest, DeduplicatesIdenticalLoopedBuffers) {
  ScopedTestContext test_ctx;
  std::vector<float> w_pass1 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> w_pass2 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> w_other = {1.0f, 2.0f, 3.0f, 5.0f};

  const int t0 = test_ctx.AddConstTensor(kTfLiteFloat32, {2, 2}, w_pass1.data(),
                                         w_pass1.size() * sizeof(float));
  const int t1 = test_ctx.AddConstTensor(kTfLiteFloat32, {2, 2}, w_pass2.data(),
                                         w_pass2.size() * sizeof(float));
  const int t2 = test_ctx.AddConstTensor(kTfLiteFloat32, {2, 2}, w_other.data(),
                                         w_other.size() * sizeof(float));

  TfLiteContext ctx = test_ctx.BuildContext();
  WeightContentDeduplicator dedup;
  WeightContentDeduplicator::BufferIdMap input_map = {
      {t0, 10}, {t1, 20}, {t2, 30}};

  WeightContentDeduplicator::BufferIdMap canonical =
      dedup.CanonicalizeBufferIdMap(ctx, input_map);

  EXPECT_EQ(canonical[t0], canonical[t1]);
  EXPECT_NE(canonical[t0], canonical[t2]);
  EXPECT_EQ(dedup.NumDeduplicatedBuffers(), 1);
}

TEST(WeightContentDeduplicatorTest,
     DoesNotMergeDifferentShapeOrQuantizationScales) {
  ScopedTestContext test_ctx;
  std::vector<int8_t> data_a = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int8_t> data_b = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int8_t> data_c = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int8_t> data_d = {1, 2, 3, 4, 5, 6, 7, 8};

  const int t_shape1 = test_ctx.AddConstTensor(kTfLiteInt8, {2, 4},
                                               data_a.data(), data_a.size());
  const int t_shape2 = test_ctx.AddConstTensor(kTfLiteInt8, {4, 2},
                                               data_b.data(), data_b.size());
  const int t_scale1 = test_ctx.AddConstTensor(kTfLiteInt8, {2, 4},
                                               data_c.data(), data_c.size());
  const int t_scale2 = test_ctx.AddConstTensor(kTfLiteInt8, {2, 4},
                                               data_d.data(), data_d.size());
  test_ctx.SetAffineQuant(t_shape1, {0.5f, 0.25f}, {0, 0}, 0);
  test_ctx.SetAffineQuant(t_shape2, {0.5f, 0.25f}, {0, 0}, 0);
  test_ctx.SetAffineQuant(t_scale1, {0.5f, 0.25f}, {0, 0}, 0);
  test_ctx.SetAffineQuant(t_scale2, {0.5f, 0.50f}, {0, 0}, 0);

  TfLiteContext ctx = test_ctx.BuildContext();
  WeightContentDeduplicator dedup;
  WeightContentDeduplicator::BufferIdMap canonical =
      dedup.CanonicalizeBufferIdMap(
          ctx, {{t_shape1, 1}, {t_shape2, 2}, {t_scale1, 3}, {t_scale2, 4}});

  EXPECT_EQ(canonical[t_shape1], canonical[t_scale1]);
  EXPECT_NE(canonical[t_shape1], canonical[t_shape2]);
  EXPECT_NE(canonical[t_shape1], canonical[t_scale2]);
  EXPECT_EQ(dedup.NumDeduplicatedBuffers(), 1);
}

TEST(WeightContentDeduplicatorTest, DeduplicatesBlockwiseAcrossSubgraphs) {
  ScopedTestContext sub1;
  std::vector<int8_t> w1 = {10, 20, 30, 40};
  std::vector<float> s1 = {0.1f, 0.2f};
  const int s1_idx = sub1.AddConstTensor(kTfLiteFloat32, {2, 1}, s1.data(),
                                         s1.size() * sizeof(float));
  const int w1_idx =
      sub1.AddConstTensor(kTfLiteInt8, {2, 2}, w1.data(), w1.size());
  sub1.SetBlockwiseQuant(w1_idx, s1_idx, -1, 2, 0);

  ScopedTestContext sub2;
  std::vector<int8_t> w2 = {10, 20, 30, 40};
  std::vector<float> s2 = {0.1f, 0.2f};
  const int s2_idx = sub2.AddConstTensor(kTfLiteFloat32, {2, 1}, s2.data(),
                                         s2.size() * sizeof(float));
  const int w2_idx =
      sub2.AddConstTensor(kTfLiteInt8, {2, 2}, w2.data(), w2.size());
  sub2.SetBlockwiseQuant(w2_idx, s2_idx, -1, 2, 0);

  WeightContentDeduplicator dedup;
  auto c1 = dedup.CanonicalizeBufferIdMap(sub1.BuildContext(), {{w1_idx, 100}});
  auto c2 = dedup.CanonicalizeBufferIdMap(sub2.BuildContext(), {{w2_idx, 200}});

  EXPECT_EQ(c1[w1_idx], 100);
  EXPECT_EQ(c2[w2_idx], 100);
  EXPECT_EQ(dedup.NumDeduplicatedBuffers(), 1);
}

TEST(WeightContentDeduplicatorTest, PreservesNullDataBuffers) {
  ScopedTestContext test_ctx;
  const int t0 = test_ctx.AddConstTensor(kTfLiteInt4, {4, 4}, nullptr, 8);
  const int t1 = test_ctx.AddConstTensor(kTfLiteInt4, {4, 4}, nullptr, 8);

  WeightContentDeduplicator dedup;
  auto canonical = dedup.CanonicalizeBufferIdMap(test_ctx.BuildContext(),
                                                 {{t0, 5}, {t1, 6}});
  EXPECT_EQ(canonical[t0], 5);
  EXPECT_EQ(canonical[t1], 6);
  EXPECT_EQ(dedup.NumDeduplicatedBuffers(), 0);
}

}  // namespace
}  // namespace litert::ml_drift
