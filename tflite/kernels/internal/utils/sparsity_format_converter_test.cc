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

#include <limits>
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

TEST(FormatConverterTest, RejectsMismatchedDimMetadataAndTraversalOrderSize) {
  TfLiteIntArray* traversal_order = TfLiteIntArrayCreate(2);
  traversal_order->data[0] = 0;
  traversal_order->data[1] = 1;

  TfLiteDimensionMetadata metadata = {};
  metadata.format = kTfLiteDimDense;
  metadata.dense_size = 4;

  TfLiteSparsity sparsity = {};
  sparsity.traversal_order = traversal_order;
  sparsity.block_map = nullptr;
  sparsity.dim_metadata = &metadata;
  sparsity.dim_metadata_size = 1;

  const std::vector<int> dense_shape = {4, 4};
  const std::vector<int> sparse_data(16, 1);
  FormatConverter<int> converter(dense_shape, sparsity);
  EXPECT_EQ(converter.SparseToDense(sparse_data.data()), kTfLiteError);

  std::vector<int> dense_data(16);
  EXPECT_EQ(converter.SparseToDense(sparse_data.data(), dense_data.size(),
                                    dense_data.data(), nullptr),
            kTfLiteError);

  TfLiteIntArrayFree(traversal_order);
}

// Densifies into a buffer larger than the dense tensor so that a write past
// the first `dense_size` elements lands in the canary region instead of
// corrupting the heap.
constexpr float kCanary = 7.0f;
constexpr size_t kPaddedSize = 256;

bool CanaryIntact(const std::vector<float>& dest, size_t dense_size) {
  for (size_t i = dense_size; i < dest.size(); ++i) {
    if (dest[i] != kCanary) return false;
  }
  return true;
}

TEST(FormatConverterTest, NonIdentityBlockTraversalOrderOutOfBoundsRejected) {
  // [4, 8] tensor with 2x4 blocks: block_map {0, 1}, block dense sizes 2 and 4.
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

  // Identity block traversal order fills all 32 elements in bounds.
  {
    FormatConverter<float> converter(dense_shape, {0, 1, 2, 3}, format,
                                     dense_size, segments, indices, block_map);
    std::vector<float> dest(kPaddedSize, kCanary);
    EXPECT_EQ(converter.SparseToDense(src.data(), 32u, dest.data(), nullptr),
              kTfLiteOk);
    for (size_t i = 0; i < 32; ++i) {
      EXPECT_NE(dest[i], 0.0f) << "element " << i;
    }
    EXPECT_TRUE(CanaryIntact(dest, 32));
  }

  // Swapping the two block dimensions in traversal_order pairs the block of
  // dimension 0 with the size of the block of dimension 1, so the
  // reconstructed row index reaches 7 in a 4-row tensor.
  {
    FormatConverter<float> converter(dense_shape, {0, 1, 3, 2}, format,
                                     dense_size, segments, indices, block_map);
    std::vector<float> dest(kPaddedSize, kCanary);
    EXPECT_EQ(converter.SparseToDense(src.data(), 32u, dest.data(), nullptr),
              kTfLiteError);
    EXPECT_TRUE(CanaryIntact(dest, 32));
  }
}

TEST(FormatConverterTest, OutOfBoundsArrayIndicesRejected) {
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<TfLiteDimensionType> format = {kTfLiteDimDense,
                                                   kTfLiteDimSparseCSR};
  const std::vector<int> dense_size = {4, 0};
  const std::vector<std::vector<int>> segments = {{}, {0, 1, 2, 3, 4}};
  const std::vector<std::vector<int>> indices = {{}, {0, 1, 2, 99}};
  const std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
  FormatConverter<float> converter(dense_shape, {0, 1}, format, dense_size,
                                   segments, indices);
  std::vector<float> dest(kPaddedSize, kCanary);
  EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
            kTfLiteError);
  EXPECT_TRUE(CanaryIntact(dest, 16));
}

TEST(FormatConverterTest, InvalidTraversalOrderOrBlockMapRejectedAtInit) {
  const std::vector<int> dense_shape = {4, 4};
  const std::vector<float> src(16, 1.0f);

  // traversal_order shorter than the tensor rank.
  {
    FormatConverter<float> converter(
        dense_shape, {0}, {kTfLiteDimDense, kTfLiteDimSparseCSR}, {4, 0},
        {{}, {0, 1, 2, 3, 4}}, {{}, {0, 1, 2, 3}});
    std::vector<float> dest(kPaddedSize, kCanary);
    EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
              kTfLiteError);
    EXPECT_TRUE(CanaryIntact(dest, 16));
  }

  // traversal_order longer than rank plus block_map size.
  {
    FormatConverter<float> converter(
        dense_shape, {0, 1, 2, 3},
        {kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense},
        {2, 2, 2, 1}, {{}, {}, {}, {}}, {{}, {}, {}, {}}, /*block_map=*/{1});
    std::vector<float> dest(kPaddedSize, kCanary);
    EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
              kTfLiteError);
    EXPECT_TRUE(CanaryIntact(dest, 16));
  }

  // Block traversal_order entry outside dim_metadata.
  {
    FormatConverter<float> converter(
        dense_shape, {0, 1, 9},
        {kTfLiteDimDense, kTfLiteDimSparseCSR, kTfLiteDimDense}, {4, 0, 2},
        {{}, {0, 1, 2, 3, 4}, {}}, {{}, {0, 1, 2, 3}, {}}, /*block_map=*/{1});
    std::vector<float> dest(kPaddedSize, kCanary);
    EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
              kTfLiteError);
    EXPECT_TRUE(CanaryIntact(dest, 16));
  }
}

TEST(FormatConverterTest, OutOfRangeTraversalOrderValueRejected) {
  const std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
  FormatConverter<float> converter(
      {4, 4}, {0, 9}, {kTfLiteDimDense, kTfLiteDimSparseCSR}, {4, 0},
      {{}, {0, 1, 2, 3, 4}}, {{}, {0, 1, 2, 3}});
  std::vector<float> dest(kPaddedSize, kCanary);
  EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
            kTfLiteError);
  EXPECT_TRUE(CanaryIntact(dest, 16));
}

TEST(FormatConverterTest, OutOfRangeBlockMapValueRejected) {
  const std::vector<float> src(16, 1.0f);
  FormatConverter<float> converter(
      {4, 4}, {0, 1, 2}, {kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense},
      {4, 4, 1}, {{}, {}, {}}, {{}, {}, {}}, /*block_map=*/{7});
  std::vector<float> dest(kPaddedSize, kCanary);
  EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
            kTfLiteError);
  EXPECT_TRUE(CanaryIntact(dest, 16));
}

TEST(FormatConverterTest, OutOfRangeBlockTraversalIndexRejected) {
  // The block entry of traversal_order (3) maps to block index 1, but
  // block_map has a single entry.
  const std::vector<float> src(16, 1.0f);
  FormatConverter<float> converter(
      {4, 4}, {0, 1, 3}, {kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense},
      {4, 4, 1}, {{}, {}, {}}, {{}, {}, {}}, /*block_map=*/{7});
  std::vector<float> dest(kPaddedSize, kCanary);
  EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
            kTfLiteError);
  EXPECT_TRUE(CanaryIntact(dest, 16));
}

TEST(FormatConverterTest, OutOfDimensionIndexInsideBufferRejected) {
  // Column 5 of row 0 flattens to element 5, which is inside the 16-element
  // buffer but outside the 4-column dimension.
  const std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
  FormatConverter<float> converter(
      {4, 4}, {0, 1}, {kTfLiteDimDense, kTfLiteDimSparseCSR}, {4, 0},
      {{}, {0, 1, 2, 3, 4}}, {{}, {5, 0, 2, 3}});
  std::vector<float> dest(kPaddedSize, kCanary);
  EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
            kTfLiteError);
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(dest[i], 0.0f) << "element " << i;
  }
  EXPECT_TRUE(CanaryIntact(dest, 16));
}

TEST(FormatConverterTest, HugeArrayIndexRejected) {
  // Row index 0x20000000 in a [4, 8] tensor, whose flattened index does not
  // fit in an int.
  const std::vector<float> src(8, 1.0f);
  FormatConverter<float> converter({4, 8}, {0, 1},
                                   {kTfLiteDimSparseCSR, kTfLiteDimDense},
                                   {0, 8}, {{0, 1}, {}}, {{0x20000000}, {}});
  std::vector<float> dest(kPaddedSize, kCanary);
  EXPECT_EQ(converter.SparseToDense(src.data(), 32u, dest.data(), nullptr),
            kTfLiteError);
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(dest[i], 0.0f) << "element " << i;
  }
  EXPECT_TRUE(CanaryIntact(dest, 32));
}

TEST(FormatConverterTest, DenseLevelPositionOverflowRejected) {
  // [4, 4, 4] tensor stored as CSR, dense, CSR.
  const std::vector<int> dense_shape = {4, 4, 4};
  const std::vector<int> traversal_order = {0, 1, 2};
  const std::vector<TfLiteDimensionType> format = {
      kTfLiteDimSparseCSR, kTfLiteDimDense, kTfLiteDimSparseCSR};

  // Control: rows 1 and 3 are stored, and each of their 8 positions in the
  // dense level holds one element on the diagonal of the last two dimensions.
  {
    FormatConverter<float> converter(dense_shape, traversal_order, format,
                                     {0, 4, 0},
                                     {{0, 2}, {}, {0, 1, 2, 3, 4, 5, 6, 7, 8}},
                                     {{1, 3}, {}, {0, 1, 2, 3, 0, 1, 2, 3}});
    const std::vector<float> src = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> dest(kPaddedSize, kCanary);
    EXPECT_EQ(converter.SparseToDense(src.data(), 64u, dest.data(), nullptr),
              kTfLiteOk);
    std::vector<float> expected(64, 0.0f);
    for (int j = 0; j < 4; ++j) {
      expected[16 * 1 + 4 * j + j] = src[j];
      expected[16 * 3 + 4 * j + j] = src[4 + j];
    }
    for (size_t i = 0; i < 64; ++i) {
      EXPECT_EQ(dest[i], expected[i]) << "element " << i;
    }
    EXPECT_TRUE(CanaryIntact(dest, 64));
  }

  // The CSR level hands segment position 2 to a dense level of size
  // 2147483647, so the position handed to the last level,
  // 2 * 2147483647 + i, does not fit in an int.
  {
    FormatConverter<float> converter(dense_shape, traversal_order, format,
                                     {0, 2147483647, 0}, {{2, 3}, {}, {0}},
                                     {{0, 0, 0}, {}, {}});
    const std::vector<float> src(4, 1.0f);
    std::vector<float> dest(kPaddedSize, kCanary);
    EXPECT_EQ(converter.SparseToDense(src.data(), 64u, dest.data(), nullptr),
              kTfLiteError);
    for (size_t i = 0; i < 64; ++i) {
      EXPECT_EQ(dest[i], 0.0f) << "element " << i;
    }
    EXPECT_TRUE(CanaryIntact(dest, 64));
  }
}

TEST(FormatConverterTest, PositionPastLastSegmentRejected) {
  // Same layout as the control above, but the last level has segments for
  // only 4 of the 8 positions of the dense level.
  FormatConverter<float> converter(
      {4, 4, 4}, {0, 1, 2},
      {kTfLiteDimSparseCSR, kTfLiteDimDense, kTfLiteDimSparseCSR}, {0, 4, 0},
      {{0, 2}, {}, {0, 1, 2, 3, 4}}, {{1, 3}, {}, {0, 1, 2, 3}});
  const std::vector<float> src = {1, 2, 3, 4};
  std::vector<float> dest(kPaddedSize, kCanary);
  EXPECT_EQ(converter.SparseToDense(src.data(), 64u, dest.data(), nullptr),
            kTfLiteError);
  EXPECT_TRUE(CanaryIntact(dest, 64));
}

TEST(FormatConverterTest, BlockTraversalOrderValueBelowRankRejected) {
  // With block_map {1, 0}, InitSparseToDenseConverter checks only the first
  // block entry of traversal_order, so the second one reaches Populate.
  const std::vector<float> src(16, 1.0f);
  FormatConverter<float> converter(
      {4, 4}, {0, 1, 2, std::numeric_limits<int>::min()},
      {kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimDense},
      {2, 2, 2, 2}, {{}, {}, {}, {}}, {{}, {}, {}, {}}, /*block_map=*/{1, 0});
  std::vector<float> dest(kPaddedSize, kCanary);
  EXPECT_EQ(converter.SparseToDense(src.data(), 16u, dest.data(), nullptr),
            kTfLiteError);
  EXPECT_TRUE(CanaryIntact(dest, 16));
}
}  // namespace
}  // namespace sparsity
}  // namespace internal
}  // namespace tflite
