/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tflite/delegates/gpu/delegate.h"
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::Pointwise;

class BatchMatMulOpModel : public SingleOpModel {
 public:
  BatchMatMulOpModel(bool use_gpu, const std::vector<int>& lhs_shape,
                     const std::vector<int>& rhs_shape, bool adj_x,
                     bool adj_y,
                     const std::vector<float>* constant_rhs = nullptr)
      : rhs_is_constant_(constant_rhs != nullptr) {
    lhs_ = AddInput({TensorType_FLOAT32, lhs_shape});
    rhs_ = rhs_is_constant_
               ? AddConstInput({TensorType_FLOAT32, rhs_shape}, *constant_rhs)
               : AddInput({TensorType_FLOAT32, rhs_shape});
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_BATCH_MATMUL, BuiltinOptions_BatchMatMulOptions,
        CreateBatchMatMulOptions(builder_, adj_x, adj_y).Union());
    BuildInterpreter({GetShape(lhs_), GetShape(rhs_)}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false);

    if (use_gpu) {
      auto options = TfLiteGpuDelegateOptionsV2Default();
      options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
      SetDelegate(
          {TfLiteGpuDelegateV2Create(&options), TfLiteGpuDelegateV2Delete});
    }
  }

  void SetInputs(const std::vector<float>& lhs,
                 const std::vector<float>& rhs) {
    PopulateTensor(lhs_, lhs);
    if (!rhs_is_constant_) {
      PopulateTensor(rhs_, rhs);
    }
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  std::vector<int32_t> GetOutputShape() {
    return GetTensorShape(output_);
  }

  int CpuKernelCount() { return CountOpsExecutedByCpuKernel(); }

 private:
  int lhs_;
  int rhs_;
  int output_;
  const bool rhs_is_constant_;
};

struct BatchMatMulTestCase {
  const char* name;
  int rank;
  bool adj_x;
  bool adj_y;
  bool square = false;
  bool broadcast_rhs = false;
  bool constant_rhs = false;
};

std::vector<int> MatrixShape(int rank, int rows, int columns,
                             int batch_size) {
  if (rank == 2) {
    return {rows, columns};
  }
  if (rank == 3) {
    return {batch_size, rows, columns};
  }
  return {1, batch_size, rows, columns};
}

std::vector<float> MakeInput(const std::vector<int>& shape, int seed) {
  int size = 1;
  for (int dimension : shape) {
    size *= dimension;
  }
  std::vector<float> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = static_cast<float>((i * seed) % 17 - 8) * 0.125f;
  }
  return values;
}

class BatchMatMulOpenClTest
    : public ::testing::TestWithParam<BatchMatMulTestCase> {};

TEST_P(BatchMatMulOpenClTest, MatchesCpu) {
  const BatchMatMulTestCase& test = GetParam();
  const int rows = test.square ? 3 : 2;
  const int depth = 3;
  const int columns = test.square ? 3 : 4;
  const int batch_size = 2;
  const std::vector<int> lhs_shape =
      MatrixShape(test.rank, test.adj_x ? depth : rows,
                  test.adj_x ? rows : depth, batch_size);
  const std::vector<int> rhs_shape =
      MatrixShape(test.rank, test.adj_y ? columns : depth,
                  test.adj_y ? depth : columns,
                  test.broadcast_rhs ? 1 : batch_size);
  const std::vector<int> output_shape =
      MatrixShape(test.rank, rows, columns, batch_size);
  const std::vector<float> lhs = MakeInput(lhs_shape, 5);
  const std::vector<float> rhs = MakeInput(rhs_shape, 7);

  const std::vector<float>* constant_rhs =
      test.constant_rhs ? &rhs : nullptr;
  BatchMatMulOpModel cpu(/*use_gpu=*/false, lhs_shape, rhs_shape, test.adj_x,
                         test.adj_y, constant_rhs);
  cpu.SetInputs(lhs, rhs);
  ASSERT_EQ(cpu.Invoke(), kTfLiteOk);
  EXPECT_THAT(cpu.GetOutputShape(), ElementsAreArray(output_shape));

  BatchMatMulOpModel gpu(/*use_gpu=*/true, lhs_shape, rhs_shape, test.adj_x,
                         test.adj_y, constant_rhs);
  ASSERT_EQ(gpu.ApplyDelegate(), kTfLiteOk);
  gpu.SetInputs(lhs, rhs);
  ASSERT_EQ(gpu.Invoke(), kTfLiteOk);
  EXPECT_EQ(gpu.CpuKernelCount(), 0);
  EXPECT_THAT(gpu.GetOutput(),
              Pointwise(FloatNear(1.0e-4f), cpu.GetOutput()));
  EXPECT_THAT(gpu.GetOutputShape(), ElementsAreArray(output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    AllOptionsAndRanks, BatchMatMulOpenClTest,
    ::testing::Values(
        BatchMatMulTestCase{"Rank2NN", 2, false, false},
        BatchMatMulTestCase{"Rank2NT", 2, false, true},
        BatchMatMulTestCase{"Rank2TN", 2, true, false},
        BatchMatMulTestCase{"Rank2TT", 2, true, true},
        BatchMatMulTestCase{"Rank2ConstNN", 2, false, false,
                           /*square=*/false, /*broadcast_rhs=*/false,
                           /*constant_rhs=*/true},
        BatchMatMulTestCase{"Rank2ConstTT", 2, true, true,
                           /*square=*/false, /*broadcast_rhs=*/false,
                           /*constant_rhs=*/true},
        BatchMatMulTestCase{"Rank3NN", 3, false, false},
        BatchMatMulTestCase{"Rank3NT", 3, false, true},
        BatchMatMulTestCase{"Rank3TN", 3, true, false},
        BatchMatMulTestCase{"Rank3TT", 3, true, true},
        BatchMatMulTestCase{"Rank3TTBroadcastRhs", 3, true, true,
                           /*square=*/false, /*broadcast_rhs=*/true},
        BatchMatMulTestCase{"Rank4NN", 4, false, false, /*square=*/true},
        BatchMatMulTestCase{"Rank4NT", 4, false, true, /*square=*/true},
        BatchMatMulTestCase{"Rank4TN", 4, true, false, /*square=*/true},
        BatchMatMulTestCase{"Rank4TT", 4, true, true, /*square=*/true}),
    [](const ::testing::TestParamInfo<BatchMatMulTestCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace tflite
