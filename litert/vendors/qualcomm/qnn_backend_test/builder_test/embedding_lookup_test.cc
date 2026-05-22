// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"
#include "QnnTypes.h"  // from @qairt

using testing::Pointwise;
namespace litert::qnn {
namespace {
INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

// Table shape: (2, 10), output shape: (1, 10).
// Both table and output use ScaleOffset quant with scale=0.1 and offset=0.
//
// Table data (int8, row-major):
//   row 0: [ 1,  2,  3,  4,  5, -1, -2, -3, -4, -5]
//   row 1: [10, 20, 30, 40, 50,-10,-20,-30,-40,-50]
//
// Indices: [1]  →  gather row 1.
//
// Since table and output share the same quant params, BuildEmbeddingLookupOp
// emits a single Gather op (no Convert op).
//
// Expected output (int8): [10, 20, 30, 40, 50, -10, -20, -30, -40, -50]
TEST_P(QnnModelTest, EmbeddingLookupScaleOffsetSameQuantParams) {
  const std::vector<std::uint32_t> kTableDims{2, 10};
  const std::vector<std::uint32_t> kIndicesDims{1};
  const std::vector<std::uint32_t> kOutputDims{1, 10};

  // 10 per-column scales, all 0.1.
  const constexpr float kScales = 0.1f;
  const constexpr std::int32_t kZeroPoints = 0;
  const auto kQuant =
      ::qnn::ScaleOffsetQuantizeParamsWrapper{kScales, kZeroPoints};

  // Table: row 0 = [1..5, -1..-5], row 1 = [10..50, -10..-50] (int8).
  std::vector<std::int8_t> table_data = {
      1,  2,  3,  4,  5,  -1,  -2,  -3,  -4,  -5,  // row 0
      10, 20, 30, 40, 50, -10, -20, -30, -40, -50  // row 1
  };

  auto& table_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, kQuant, kTableDims,
      table_data.size() * sizeof(std::int8_t), table_data.data());

  auto& indices_tensor = tensor_pool_.CreateInputTensorWithName(
      "indices", QNN_DATATYPE_UINT_32, {}, kIndicesDims);

  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_8, kQuant, kOutputDims);

  auto ops = ::qnn::BuildEmbeddingLookupOp(
      tensor_pool_, {indices_tensor, table_tensor}, {output_tensor});
  // Same quant params: expect exactly one Gather op, no Convert op.
  ASSERT_EQ(ops.size(), 1u);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

  const auto indices_idx = qnn_model_.AddInputTensor(indices_tensor);
  const auto output_idx = qnn_model_.AddOutputTensor(output_tensor);

  // Select row 1.
  qnn_model_.SetInputData<std::uint32_t>(indices_idx, {1});
  ASSERT_TRUE(qnn_model_.Execute());

  const auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 10u);
  // Row 1 int8 values pass through unchanged.
  ASSERT_THAT(
      output_data.value(),
      Pointwise(testing::Eq(), std::vector<std::int8_t>{10, 20, 30, 40, 50, -10,
                                                        -20, -30, -40, -50}));
}

// Same table as above (scale=0.1), but the output uses
// ScaleOffset with scale=0.2.
//
// BuildEmbeddingLookupOp must emit Gather + Convert because the quant params
// differ.
//
// Gather picks row 1 (int8): [10, 20, 30, 40, 50, -10, -20, -30, -40, -50]
// Real values (scale 0.1):   [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0,
//                              -4.0, -5.0]
// Re-quantized to scale 0.2: round(real / 0.2)
//                           = [5, 10, 15, 20, 25, -5, -10, -15, -20, -25]
TEST_P(QnnModelTest, EmbeddingLookupScaleOffsetDifferentQuantParams) {
  const std::vector<std::uint32_t> kTableDims{2, 10};
  const std::vector<std::uint32_t> kIndicesDims{1};
  const std::vector<std::uint32_t> kOutputDims{1, 10};

  const auto kTableQuant = ::qnn::ScaleOffsetQuantizeParamsWrapper{0.1f, 0};
  const auto kOutputQuant = ::qnn::ScaleOffsetQuantizeParamsWrapper{0.2f, 0};

  std::vector<std::int8_t> table_data = {
      1,  2,  3,  4,  5,  -1,  -2,  -3,  -4,  -5,  // row 0
      10, 20, 30, 40, 50, -10, -20, -30, -40, -50  // row 1
  };

  auto& table_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, kTableQuant, kTableDims,
      table_data.size() * sizeof(std::int8_t), table_data.data());

  auto& indices_tensor = tensor_pool_.CreateInputTensorWithName(
      "indices", QNN_DATATYPE_UINT_32, {}, kIndicesDims);

  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_8, kOutputQuant, kOutputDims);

  auto ops = ::qnn::BuildEmbeddingLookupOp(
      tensor_pool_, {indices_tensor, table_tensor}, {output_tensor});
  // Different quant params: expect Gather + Convert = 2 ops.
  ASSERT_EQ(ops.size(), 2u);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

  const auto indices_idx = qnn_model_.AddInputTensor(indices_tensor);
  const auto output_idx = qnn_model_.AddOutputTensor(output_tensor);

  // Select row 1.
  qnn_model_.SetInputData<std::uint32_t>(indices_idx, {1});
  ASSERT_TRUE(qnn_model_.Execute());

  const auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 10u);
  // Real values from row 1 (scale 0.1) re-quantized to scale 0.2:
  //   round(v * 0.1 / 0.2) = round(v / 2)
  //   [10, 20, 30, 40, 50, -10, -20, -30, -40, -50] / 2 =
  //   [5, 10, 15, 20, 25, -5, -10, -15, -20, -25]
  ASSERT_THAT(
      output_data.value(),
      Pointwise(testing::Eq(), std::vector<std::int8_t>{5, 10, 15, 20, 25, -5,
                                                        -10, -15, -20, -25}));
}

}  // namespace
}  // namespace litert::qnn
