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
  explicit BatchMatMulOpModel(bool use_gpu) {
    lhs_ = AddInput({TensorType_FLOAT32, {1, 2, 3}});
    rhs_ = AddInput({TensorType_FLOAT32, {1, 4, 3}});
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_BATCH_MATMUL, BuiltinOptions_BatchMatMulOptions,
        CreateBatchMatMulOptions(builder_, /*adj_x=*/false, /*adj_y=*/true)
            .Union());
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
    PopulateTensor(rhs_, rhs);
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
};

TEST(BatchMatMulOpenClTest, AdjointRhsMatchesCpu) {
  const std::vector<float> lhs = {1, 2, 3, 4, 5, 6};
  const std::vector<float> rhs = {
      1, 0, 0,  //
      0, 1, 0,  //
      0, 0, 1,  //
      1, 1, 1,
  };
  const std::vector<float> expected = {1, 2, 3, 6, 4, 5, 6, 15};

  BatchMatMulOpModel cpu(/*use_gpu=*/false);
  cpu.SetInputs(lhs, rhs);
  ASSERT_EQ(cpu.Invoke(), kTfLiteOk);
  EXPECT_THAT(cpu.GetOutput(), ElementsAreArray(expected));
  EXPECT_THAT(cpu.GetOutputShape(), ElementsAreArray({1, 2, 4}));

  BatchMatMulOpModel gpu(/*use_gpu=*/true);
  ASSERT_EQ(gpu.ApplyDelegate(), kTfLiteOk);
  gpu.SetInputs(lhs, rhs);
  ASSERT_EQ(gpu.Invoke(), kTfLiteOk);
  EXPECT_EQ(gpu.CpuKernelCount(), 0);
  EXPECT_THAT(gpu.GetOutput(),
              Pointwise(FloatNear(1.0e-5f), cpu.GetOutput()));
  EXPECT_THAT(gpu.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

}  // namespace
}  // namespace tflite
