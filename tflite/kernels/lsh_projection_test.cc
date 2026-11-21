/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

class LSHProjectionOpModel : public SingleOpModel {
 public:
  LSHProjectionOpModel(LSHProjectionType type,
                       std::initializer_list<int> hash_shape,
                       std::initializer_list<int> input_shape,
                       std::initializer_list<int> weight_shape) {
    hash_ = AddInput(TensorType_FLOAT32);
    input_ = AddInput(TensorType_INT32);
    if (weight_shape.size() > 0) {
      weight_ = AddInput(TensorType_FLOAT32);
    }
    output_ = AddOutput(TensorType_INT32);

    SetBuiltinOp(BuiltinOperator_LSH_PROJECTION,
                 BuiltinOptions_LSHProjectionOptions,
                 CreateLSHProjectionOptions(builder_, type).Union());
    if (weight_shape.size() > 0) {
      BuildInterpreter({hash_shape, input_shape, weight_shape});
    } else {
      BuildInterpreter({hash_shape, input_shape});
    }

    output_size_ = 1;
    for (int i : hash_shape) {
      output_size_ *= i;
      if (type == LSHProjectionType_SPARSE) {
        break;
      }
    }
  }
  void SetInput(std::initializer_list<int> data) {
    PopulateTensor(input_, data);
  }

  void SetHash(std::initializer_list<float> data) {
    PopulateTensor(hash_, data);
  }

  void SetWeight(std::initializer_list<float> f) { PopulateTensor(weight_, f); }

  std::vector<int> GetOutput() { return ExtractVector<int>(output_); }

 private:
  int input_;
  int hash_;
  int weight_;
  int output_;

  int output_size_;
};

TEST(LSHProjectionOpTest2, Dense1DInputs) {
  LSHProjectionOpModel m(LSHProjectionType_DENSE, {3, 2}, {5}, {5});

  m.SetInput({12345, 54321, 67890, 9876, -12345678});
  m.SetHash({0.123, 0.456, -0.321, 1.234, 5.678, -4.321});
  m.SetWeight({1.0, 1.0, 1.0, 1.0, 1.0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // Hash returns differently on machines with different endianness
  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 0, 1, 1, 1, 0));
#else
  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 0, 0, 1, 0, 0));
#endif
}

TEST(LSHProjectionOpTest2, Sparse1DInputs) {
  LSHProjectionOpModel m(LSHProjectionType_SPARSE, {3, 2}, {5}, {});

  m.SetInput({12345, 54321, 67890, 9876, -12345678});
  m.SetHash({0.123, 0.456, -0.321, 1.234, 5.678, -4.321});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // Hash returns differently on machines with different endianness
  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 0, 4 + 3, 8 + 2));
#else
  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 0, 4 + 1, 8 + 0));
#endif
}

TEST(LSHProjectionOpTest2, Sparse3DInputs) {
  LSHProjectionOpModel m(LSHProjectionType_SPARSE, {3, 2}, {5, 2, 2}, {5});

  m.SetInput({1234, 2345, 3456, 1234, 4567, 5678, 6789, 4567, 7891, 8912,
              9123, 7890, -987, -876, -765, -987, -543, -432, -321, -543});
  m.SetHash({0.123, 0.456, -0.321, 1.234, 5.678, -4.321});
  m.SetWeight({0.12, 0.34, 0.56, 0.67, 0.78});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // Hash returns differently on machines with different endianness
  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 0, 4 + 3, 8 + 2));
#else
  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 2, 4 + 1, 8 + 1));
#endif
}

class CustomTypeLSHProjectionOpModel : public SingleOpModel {
 public:
  CustomTypeLSHProjectionOpModel(
      TensorType hash_type, TensorType weight_type,
      const std::vector<int>& hash_shape,
      LSHProjectionType proj_type = LSHProjectionType_DENSE) {
    AddInput(hash_type);
    AddInput(TensorType_INT32);
    AddInput(weight_type);
    AddOutput(TensorType_INT32);
    SetBuiltinOp(
        BuiltinOperator_LSH_PROJECTION, BuiltinOptions_LSHProjectionOptions,
        CreateLSHProjectionOptions(builder_, proj_type).Union());
    BuildInterpreter({hash_shape, {5}, {5}}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true, /*allocate_and_delegate=*/false);
  }
};

TEST(LSHProjectionOpTest2, RejectsNonFloatHashAndWeightTypes) {
  CustomTypeLSHProjectionOpModel int8_hash(TensorType_INT8, TensorType_FLOAT32,
                                           {3, 2});
  EXPECT_EQ(int8_hash.AllocateTensors(), kTfLiteError);

  CustomTypeLSHProjectionOpModel int8_weight(TensorType_FLOAT32,
                                             TensorType_INT8, {3, 2});
  EXPECT_EQ(int8_weight.AllocateTensors(), kTfLiteError);

  CustomTypeLSHProjectionOpModel zero_bits(TensorType_FLOAT32,
                                           TensorType_FLOAT32, {3, 0});
  EXPECT_EQ(zero_bits.AllocateTensors(), kTfLiteError);

  CustomTypeLSHProjectionOpModel sparse_31_bits_2_hashes(
      TensorType_FLOAT32, TensorType_FLOAT32, {2, 31},
      LSHProjectionType_SPARSE);
  EXPECT_EQ(sparse_31_bits_2_hashes.AllocateTensors(), kTfLiteOk);

  CustomTypeLSHProjectionOpModel sparse_30_bits_4_hashes(
      TensorType_FLOAT32, TensorType_FLOAT32, {4, 30},
      LSHProjectionType_SPARSE);
  EXPECT_EQ(sparse_30_bits_4_hashes.AllocateTensors(), kTfLiteOk);

  CustomTypeLSHProjectionOpModel sparse_32_bits_1_hash(
      TensorType_FLOAT32, TensorType_FLOAT32, {1, 32},
      LSHProjectionType_SPARSE);
  EXPECT_EQ(sparse_32_bits_1_hash.AllocateTensors(), kTfLiteOk);

  CustomTypeLSHProjectionOpModel sparse_31_bits_4_hashes(
      TensorType_FLOAT32, TensorType_FLOAT32, {4, 31},
      LSHProjectionType_SPARSE);
  EXPECT_EQ(sparse_31_bits_4_hashes.AllocateTensors(), kTfLiteError);

  CustomTypeLSHProjectionOpModel sparse_32_bits_2_hashes(
      TensorType_FLOAT32, TensorType_FLOAT32, {2, 32},
      LSHProjectionType_SPARSE);
  EXPECT_EQ(sparse_32_bits_2_hashes.AllocateTensors(), kTfLiteError);
}

}  // namespace
}  // namespace tflite
