/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tflite/kernels/internal/utils/sparsity_format_converter.h"

#include <vector>

#include <gtest/gtest.h>
#include "tflite/c/common.h"

namespace tflite {
namespace internal {
namespace sparsity {
namespace {
TEST(FormatConverterTest, SimpleTestD0D1) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 4};
  const std::vector<int> traversal_order = {0, 1};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimDense};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0 = {3};
  const std::vector<int> dm1 = {4};
  EXPECT_EQ(dm0, dim_metadata[0]);
  EXPECT_EQ(dm1, dim_metadata[2]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, SimpleTestS0D1) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 4};
  const std::vector<int> traversal_order = {0, 1};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimSparseCSR,
                                                   kTfLiteDimDense};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0_0 = {0, 2};
  const std::vector<int> dm0_1 = {0, 2};
  const std::vector<int> dm1 = {4};
  EXPECT_EQ(dm0_0, dim_metadata[0]);
  EXPECT_EQ(dm0_1, dim_metadata[1]);
  EXPECT_EQ(dm1, dim_metadata[2]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 0, 9, 8, 5, 0, 0, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, SimpleTestD0S1) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 4};
  const std::vector<int> traversal_order = {0, 1};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimSparseCSR};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0 = {3};
  const std::vector<int> dm1_0 = {0, 3, 3, 5};
  const std::vector<int> dm1_1 = {0, 2, 3, 0, 3};
  EXPECT_EQ(dm0, dim_metadata[0]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 9, 8, 5, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, SimpleTestS0S1) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 4};
  const std::vector<int> traversal_order = {0, 1};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimSparseCSR,
                                                   kTfLiteDimSparseCSR};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0_0 = {0, 2};
  const std::vector<int> dm0_1 = {0, 2};
  const std::vector<int> dm1_0 = {0, 3, 5};
  const std::vector<int> dm1_1 = {0, 2, 3, 0, 3};
  EXPECT_EQ(dm0_0, dim_metadata[0]);
  EXPECT_EQ(dm0_1, dim_metadata[1]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 9, 8, 5, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, SimpleTestD1D0) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 4};
  const std::vector<int> traversal_order = {1, 0};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimDense};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0 = {4};
  const std::vector<int> dm1 = {3};
  EXPECT_EQ(dm0, dim_metadata[0]);
  EXPECT_EQ(dm1, dim_metadata[2]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 0, 5, 0, 0, 0, 9, 0, 0, 8, 0, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, SimpleTestS1D0) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 4};
  const std::vector<int> traversal_order = {1, 0};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimSparseCSR};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0_0 = {0, 3};
  const std::vector<int> dm0_1 = {0, 2, 3};
  const std::vector<int> dm1 = {3};
  EXPECT_EQ(dm0_0, dim_metadata[0]);
  EXPECT_EQ(dm0_1, dim_metadata[1]);
  EXPECT_EQ(dm1, dim_metadata[2]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 0, 5, 9, 0, 0, 8, 0, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, SimpleTestD1S0) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 4};
  const std::vector<int> traversal_order = {1, 0};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimSparseCSR,
                                                   kTfLiteDimDense};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0 = {4};
  const std::vector<int> dm1_0 = {0, 2, 2, 3, 5};
  const std::vector<int> dm1_1 = {0, 2, 0, 0, 2};
  EXPECT_EQ(dm0, dim_metadata[0]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 5, 9, 8, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, SimpleTestS1S0) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 4};
  const std::vector<int> traversal_order = {1, 0};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimSparseCSR,
                                                   kTfLiteDimSparseCSR};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0_0 = {0, 3};
  const std::vector<int> dm0_1 = {0, 2, 3};
  const std::vector<int> dm1_0 = {0, 2, 3, 5};
  const std::vector<int> dm1_1 = {0, 2, 0, 0, 2};
  EXPECT_EQ(dm0_0, dim_metadata[0]);
  EXPECT_EQ(dm0_1, dim_metadata[1]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 5, 9, 8, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, 3DTestS0D1S2) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 2, 2};
  const std::vector<int> traversal_order = {0, 1, 2};
  const std::vector<TfLiteDimensionType> format = {
      kTfLiteDimSparseCSR, kTfLiteDimDense, kTfLiteDimSparseCSR};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0_0 = {0, 2};
  const std::vector<int> dm0_1 = {0, 2};
  const std::vector<int> dm1 = {2};
  const std::vector<int> dm2_0 = {0, 1, 3, 4, 5};
  const std::vector<int> dm2_1 = {0, 0, 1, 0, 1};

  EXPECT_EQ(dm0_0, dim_metadata[0]);
  EXPECT_EQ(dm0_1, dim_metadata[1]);
  EXPECT_EQ(dm1, dim_metadata[2]);
  EXPECT_EQ(dm2_0, dim_metadata[4]);
  EXPECT_EQ(dm2_1, dim_metadata[5]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 9, 8, 5, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, 3DTestD0D1S2) {
  const std::vector<int> dense_values = {6, 0, 9, 8, 0, 0, 0, 0, 5, 0, 0, 7};
  const std::vector<int> dense_shape = {3, 2, 2};
  const std::vector<int> traversal_order = {0, 1, 2};
  const std::vector<TfLiteDimensionType> format = {
      kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimSparseCSR};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0 = {3};
  const std::vector<int> dm1 = {2};
  const std::vector<int> dm2_0 = {0, 1, 3, 3, 3, 4, 5};
  const std::vector<int> dm2_1 = {0, 0, 1, 0, 1};

  EXPECT_EQ(dm0, dim_metadata[0]);
  EXPECT_EQ(dm1, dim_metadata[2]);
  EXPECT_EQ(dm2_0, dim_metadata[4]);
  EXPECT_EQ(dm2_1, dim_metadata[5]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {6, 9, 8, 5, 7};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, 3DTestS0S1S2) {
  const std::vector<int> dense_values = {1, 7, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 2, 0, 0, 4, 8, 3, 9};
  const std::vector<int> dense_shape = {3, 4, 2};
  const std::vector<int> traversal_order = {0, 1, 2};
  const std::vector<TfLiteDimensionType> format = {
      kTfLiteDimSparseCSR, kTfLiteDimSparseCSR, kTfLiteDimSparseCSR};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0_0 = {0, 2};
  const std::vector<int> dm0_1 = {0, 2};
  const std::vector<int> dm1_0 = {0, 2, 5};
  const std::vector<int> dm1_1 = {0, 2, 0, 2, 3};
  const std::vector<int> dm2_0 = {0, 2, 3, 4, 6, 8};
  const std::vector<int> dm2_1 = {0, 1, 1, 1, 0, 1, 0, 1};
  EXPECT_EQ(dm0_0, dim_metadata[0]);
  EXPECT_EQ(dm0_1, dim_metadata[1]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);
  EXPECT_EQ(dm2_0, dim_metadata[4]);
  EXPECT_EQ(dm2_1, dim_metadata[5]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {1, 7, 5, 2, 4, 8, 3, 9};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, 3DTestS0S2S1) {
  const std::vector<int> dense_values = {1, 0, 0, 0, 7, 0, 5, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 4, 3, 2, 0, 8, 9};
  const std::vector<int> dense_shape = {3, 2, 4};
  const std::vector<int> traversal_order = {0, 2, 1};
  const std::vector<TfLiteDimensionType> format = {
      kTfLiteDimSparseCSR, kTfLiteDimSparseCSR, kTfLiteDimSparseCSR};
  FormatConverter<int> converter(dense_shape, traversal_order, format);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0_0 = {0, 2};
  const std::vector<int> dm0_1 = {0, 2};
  const std::vector<int> dm1_0 = {0, 2, 5};
  const std::vector<int> dm1_1 = {0, 2, 0, 2, 3};
  const std::vector<int> dm2_0 = {0, 2, 3, 4, 6, 8};
  const std::vector<int> dm2_1 = {0, 1, 1, 1, 0, 1, 0, 1};
  EXPECT_EQ(dm0_0, dim_metadata[0]);
  EXPECT_EQ(dm0_1, dim_metadata[1]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);
  EXPECT_EQ(dm2_0, dim_metadata[4]);
  EXPECT_EQ(dm2_1, dim_metadata[5]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {1, 7, 5, 2, 4, 8, 3, 9};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, BlockTestD0D1) {
  const std::vector<int> dense_values = {1, 0, 2, 3, 0, 4, 0, 0,
                                         0, 0, 5, 0, 0, 0, 0, 6};
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<int> traversal_order = {0, 1, 2, 3};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimDense};
  const std::vector<int> block_size = {2, 2};
  const std::vector<int> block_map = {0, 1};
  FormatConverter<int> converter(dense_shape, traversal_order, format,
                                 block_size, block_map);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm = {2};
  EXPECT_EQ(dm, dim_metadata[0]);
  EXPECT_EQ(dm, dim_metadata[2]);
  EXPECT_EQ(dm, dim_metadata[4]);
  EXPECT_EQ(dm, dim_metadata[6]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {1, 0, 0, 4, 2, 3, 0, 0,
                                          0, 0, 0, 0, 5, 0, 0, 6};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

// BCSR
TEST(FormatConverterTest, BlockTestD0S11DBlock) {
  const std::vector<int> dense_values = {1, 0, 2, 3, 0, 4, 0, 0,
                                         0, 0, 5, 0, 0, 0, 0, 6};
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<int> traversal_order = {0, 1, 2};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimSparseCSR};
  const std::vector<int> block_size = {2};
  const std::vector<int> block_map = {1};
  FormatConverter<int> converter(dense_shape, traversal_order, format,
                                 block_size, block_map);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm0 = {4};
  const std::vector<int> dm2 = {2};
  const std::vector<int> dm1_0 = {0, 2, 3, 4, 5};
  const std::vector<int> dm1_1 = {0, 1, 0, 1, 1};
  EXPECT_EQ(dm0, dim_metadata[0]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);
  EXPECT_EQ(dm2, dim_metadata[4]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {1, 0, 2, 3, 0, 4, 5, 0, 0, 6};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

// BCSR
TEST(FormatConverterTest, BlockTestD0S12DBlock) {
  const std::vector<int> dense_values = {1, 0, 2, 3, 0, 4, 0, 0,
                                         0, 0, 5, 0, 0, 0, 0, 6};
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<int> traversal_order = {0, 1, 2, 3};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimSparseCSR};
  const std::vector<int> block_size = {2, 2};
  const std::vector<int> block_map = {0, 1};
  FormatConverter<int> converter(dense_shape, traversal_order, format,
                                 block_size, block_map);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm = {2};
  const std::vector<int> dm1_0 = {0, 2, 3};
  const std::vector<int> dm1_1 = {0, 1, 1};
  EXPECT_EQ(dm, dim_metadata[0]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);
  EXPECT_EQ(dm, dim_metadata[4]);
  EXPECT_EQ(dm, dim_metadata[6]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {1, 0, 0, 4, 2, 3, 0, 0, 5, 0, 0, 6};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

// BCSC
TEST(FormatConverterTest, BlockTestD1S0) {
  const std::vector<int> dense_values = {1, 0, 2, 3, 0, 4, 0, 0,
                                         0, 0, 5, 0, 0, 0, 0, 6};
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<int> traversal_order = {1, 0, 3, 2};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimSparseCSR,
                                                   kTfLiteDimDense};
  const std::vector<int> block_size = {2, 2};
  const std::vector<int> block_map = {0, 1};
  FormatConverter<int> converter(dense_shape, traversal_order, format,
                                 block_size, block_map);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm = {2};
  const std::vector<int> dm1_0 = {0, 1, 3};
  const std::vector<int> dm1_1 = {0, 0, 1};
  EXPECT_EQ(dm, dim_metadata[0]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);
  EXPECT_EQ(dm, dim_metadata[4]);
  EXPECT_EQ(dm, dim_metadata[6]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {1, 0, 0, 4, 2, 0, 3, 0, 5, 0, 0, 6};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

// BCSR with last block being empty
TEST(FormatConverterTest, BlockTestD0S1LastBlockEmpty) {
  const std::vector<int> dense_values = {1, 0, 2, 3, 0, 4, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<int> traversal_order = {0, 1, 2, 3};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimSparseCSR};
  const std::vector<int> block_size = {2, 2};
  const std::vector<int> block_map = {0, 1};
  FormatConverter<int> converter(dense_shape, traversal_order, format,
                                 block_size, block_map);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm = {2};
  const std::vector<int> dm1_0 = {0, 2, 2};
  const std::vector<int> dm1_1 = {0, 1};
  EXPECT_EQ(dm, dim_metadata[0]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);
  EXPECT_EQ(dm, dim_metadata[4]);
  EXPECT_EQ(dm, dim_metadata[6]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {1, 0, 0, 4, 2, 3, 0, 0};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, BlockTestD0S1ColMajorBlock) {
  const std::vector<int> dense_values = {1, 0, 2, 3, 0, 4, 0, 0, 1, 0, 2,
                                         3, 0, 4, 0, 0, 0, 0, 5, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<int> dense_shape = {4, 8};
  const std::vector<int> traversal_order = {0, 1, 3, 2};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimSparseCSR};
  const std::vector<int> block_size = {2, 2};
  const std::vector<int> block_map = {0, 1};
  FormatConverter<int> converter(dense_shape, traversal_order, format,
                                 block_size, block_map);
  converter.DenseToSparse(dense_values.data());

  const auto& dim_metadata = converter.GetDimMetadata();
  const std::vector<int> dm = {2};
  const std::vector<int> dm1_0 = {0, 3, 4};
  const std::vector<int> dm1_1 = {0, 1, 2, 1};
  EXPECT_EQ(dm, dim_metadata[0]);
  EXPECT_EQ(dm1_0, dim_metadata[2]);
  EXPECT_EQ(dm1_1, dim_metadata[3]);
  EXPECT_EQ(dm, dim_metadata[4]);
  EXPECT_EQ(dm, dim_metadata[6]);

  const auto& data = converter.GetData();
  const std::vector<int> expected_data = {1, 1, 0, 0, 2, 2, 3, 3,
                                          0, 0, 4, 4, 5, 0, 0, 0};
  EXPECT_EQ(expected_data, data);

  converter.SparseToDense(expected_data.data());
  const auto& data_back = converter.GetData();
  EXPECT_EQ(data_back, dense_values);

  std::vector<int> dense_data(dense_values.size());
  converter.SparseToDense(expected_data.data(), dense_data.size(),
                          dense_data.data(), nullptr);
  EXPECT_EQ(dense_data, dense_values);
}

TEST(FormatConverterTest, OutOfBoundsArrayIndicesIgnored) {
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<int> traversal_order = {0, 1};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimSparseCSR};
  const std::vector<int> dense_size = {4, 0};
  const std::vector<std::vector<int>> segments = {{}, {0, 1, 2, 3, 4}};
  const std::vector<std::vector<int>> indices = {{}, {0, 1, 2, 99}};
  FormatConverter<float> converter(dense_shape, traversal_order, format,
                                   dense_size, segments, indices);
  const std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(converter.SparseToDense(src.data()), kTfLiteOk);
  std::vector<float> dest(256, 7.0f);
  EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
            kTfLiteOk);
  EXPECT_FLOAT_EQ(dest[0], 1.0f);
  EXPECT_FLOAT_EQ(dest[5], 2.0f);
  EXPECT_FLOAT_EQ(dest[10], 3.0f);
  EXPECT_FLOAT_EQ(dest[15], 0.0f);
  for (size_t i = 16; i < dest.size(); ++i) {
    EXPECT_FLOAT_EQ(dest[i], 7.0f);
  }
}

TEST(FormatConverterTest, OutOfBoundsAndShortTraversalOrderIgnored) {
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimSparseCSR};
  const std::vector<int> dense_size = {4, 0};
  const std::vector<std::vector<int>> segments = {{}, {0, 1, 2, 3, 4}};
  const std::vector<std::vector<int>> indices = {{}, {0, 1, 2, 3}};
  const std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};

  // Out-of-bounds traversal_order entry (9 >= orig_rank).
  {
    FormatConverter<float> converter(dense_shape, /*traversal_order=*/{0, 9},
                                     format, dense_size, segments, indices);
    EXPECT_EQ(converter.SparseToDense(src.data()), kTfLiteOk);
    std::vector<float> dest(256, 7.0f);
    EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
              kTfLiteOk);
    for (size_t i = 16; i < dest.size(); ++i) {
      EXPECT_FLOAT_EQ(dest[i], 7.0f);
    }
  }

  // Short traversal_order (size 1 < orig_rank 2).
  {
    FormatConverter<float> converter(dense_shape, /*traversal_order=*/{0},
                                     format, dense_size, segments, indices);
    EXPECT_EQ(converter.SparseToDense(src.data()), kTfLiteOk);
    std::vector<float> dest(256, 7.0f);
    EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
              kTfLiteOk);
    for (size_t i = 16; i < dest.size(); ++i) {
      EXPECT_FLOAT_EQ(dest[i], 7.0f);
    }
  }
}

TEST(FormatConverterTest, OutOfBoundsBlockMapAndTraversalIgnored) {
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<float> src(16, 1.0f);

  // Block traversal_order entry exceeding block_map size.
  {
    FormatConverter<float> converter(
        dense_shape, /*traversal_order=*/{0, 1, 2, 3},
        {kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense},
        /*dense_size=*/{2, 2, 2, 1}, /*segments=*/{{}, {}, {}, {}},
        /*indices=*/{{}, {}, {}, {}}, /*block_map=*/{1});
    EXPECT_EQ(converter.SparseToDense(src.data()), kTfLiteOk);
    std::vector<float> dest(256, 7.0f);
    EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
              kTfLiteOk);
    for (size_t i = 16; i < dest.size(); ++i) {
      EXPECT_FLOAT_EQ(dest[i], 7.0f);
    }
  }

  // block_map entry exceeding original rank.
  {
    FormatConverter<float> converter(
        dense_shape, /*traversal_order=*/{0, 1, 2},
        {kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense},
        /*dense_size=*/{4, 4, 1}, /*segments=*/{{}, {}, {}},
        /*indices=*/{{}, {}, {}}, /*block_map=*/{7});
    EXPECT_EQ(converter.SparseToDense(src.data()), kTfLiteOk);
    std::vector<float> dest(256, 7.0f);
    EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
              kTfLiteOk);
    for (size_t i = 16; i < dest.size(); ++i) {
      EXPECT_FLOAT_EQ(dest[i], 7.0f);
    }
  }

  // Block traversal_order entry out of range for dense_size in Init and Populate.
  {
    FormatConverter<float> converter(
        dense_shape, /*traversal_order=*/{0, 1, 9},
        {kTfLiteDimDense, kTfLiteDimSparseCSR, kTfLiteDimDense},
        /*dense_size=*/{4, 0, 2}, /*segments=*/{{}, {0, 1, 2, 3, 4}, {}},
        /*indices=*/{{}, {0, 1, 2, 3}, {}}, /*block_map=*/{1});
    EXPECT_EQ(converter.SparseToDense(src.data()), kTfLiteOk);
    std::vector<float> dest(256, 7.0f);
    EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
              kTfLiteOk);
    for (size_t i = 16; i < dest.size(); ++i) {
      EXPECT_FLOAT_EQ(dest[i], 7.0f);
    }
  }
}

TEST(FormatConverterTest, NonIdentityBlockTraversalOrder2x4Bounded) {
  const std::vector<int> dense_shape = {4, 8};
  const std::vector<TfLiteDimensionType> format = {
      kTfLiteDimDense, kTfLiteDimSparseCSR, kTfLiteDimDense, kTfLiteDimDense};
  const std::vector<int> dense_size = {2, 0, 2, 4};
  const std::vector<std::vector<int>> segments = {{}, {0, 2, 4}, {}, {}};
  const std::vector<std::vector<int>> indices = {{}, {0, 1, 0, 1}, {}, {}};
  const std::vector<int> block_map = {0, 1};
  std::vector<float> src(32);
  for (int i = 0; i < 32; ++i) {
    src[i] = static_cast<float>(i + 1);
  }

  // Identity block traversal_order [0, 1, 2, 3] densifies all 32 elements.
  {
    FormatConverter<float> control(dense_shape, /*traversal_order=*/{0, 1, 2, 3},
                                   format, dense_size, segments, indices,
                                   block_map);
    EXPECT_EQ(control.SparseToDense(src.data()), kTfLiteOk);
    std::vector<float> dest(256, 7.0f);
    EXPECT_EQ(control.SparseToDense(src.data(), 32u, dest.data(), nullptr),
              kTfLiteOk);
    for (size_t i = 32; i < dest.size(); ++i) {
      EXPECT_FLOAT_EQ(dest[i], 7.0f);
    }
  }

  // Non-identity block traversal_order [0, 1, 3, 2] with asymmetric 2x4 blocks
  // must not write past the 32-element destination buffer.
  {
    FormatConverter<float> converter(
        dense_shape, /*traversal_order=*/{0, 1, 3, 2}, format, dense_size,
        segments, indices, block_map);
    EXPECT_EQ(converter.SparseToDense(src.data()), kTfLiteOk);
    std::vector<float> dest(256, 7.0f);
    EXPECT_EQ(converter.SparseToDense(src.data(), 32u, dest.data(), nullptr),
              kTfLiteOk);
    for (size_t i = 32; i < dest.size(); ++i) {
      EXPECT_FLOAT_EQ(dest[i], 7.0f);
    }
  }
}
}  // namespace
}  // namespace sparsity
}  // namespace internal
}  // namespace tflite
