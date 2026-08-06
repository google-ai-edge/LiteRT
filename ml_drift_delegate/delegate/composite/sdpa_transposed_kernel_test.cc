// Copyright 2026 Google LLC.
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

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.h"

#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

namespace litert::ml_drift {
namespace {

TEST(SdpaTransposedKernelTest, BuildGpuGraphFloat32) {
  ::ml_drift::GpuInfo gpu_info;
  ::ml_drift::GpuModelBuilder builder(gpu_info, {},
                                      ::ml_drift::CalculationsPrecision::F32,
                                      ::ml_drift::TensorStorageType::BUFFER);

  auto q = builder.AddTensor(::ml_drift::BHWC(1, 1, 4, 64),
                             ::ml_drift::DataType::FLOAT32);
  auto k = builder.AddTensor(::ml_drift::BHWC(1, 1, 128, 64),
                             ::ml_drift::DataType::FLOAT32);
  auto v = builder.AddTensor(::ml_drift::BHWC(1, 1, 128, 64),
                             ::ml_drift::DataType::FLOAT32);
  auto mask = builder.AddTensor(::ml_drift::BHWC(1, 1, 4, 128),
                                ::ml_drift::DataType::FLOAT32);
  auto param_tensor = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 7),
                                        ::ml_drift::DataType::INT32);
  auto out_tensor = builder.AddTensor(::ml_drift::BHWC(1, 1, 4, 64),
                                      ::ml_drift::DataType::FLOAT32);

  SdpaTransposedAttributes attr;
  attr.runtime_check.src_end_ch_index = 2;
  attr.bmm1_weights.weights_shape = ::ml_drift::OHWI(128, 1, 1, 64);
  attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::FLOAT32, attr.bmm1_weights.weights_shape);
  attr.bmm1_weights.desc.layout =
      ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;

  attr.bmm2_weights.weights_shape = ::ml_drift::OHWI(128, 1, 1, 64);
  attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::FLOAT32, attr.bmm2_weights.weights_shape);

  auto status =
      BuildSdpaTransposedGpuGraph({q.id, k.id, v.id, mask.id, param_tensor.id},
                                  out_tensor.id, attr, &builder);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST(SdpaTransposedKernelTest, BuildGpuGraphFloat16) {
  ::ml_drift::GpuInfo gpu_info;
  ::ml_drift::GpuModelBuilder builder(gpu_info, {},
                                      ::ml_drift::CalculationsPrecision::F16,
                                      ::ml_drift::TensorStorageType::BUFFER);

  auto q = builder.AddTensor(::ml_drift::BHWC(1, 1, 4, 64),
                             ::ml_drift::DataType::FLOAT16);
  auto k = builder.AddTensor(::ml_drift::BHWC(1, 1, 128, 64),
                             ::ml_drift::DataType::FLOAT16);
  auto v = builder.AddTensor(::ml_drift::BHWC(1, 1, 128, 64),
                             ::ml_drift::DataType::FLOAT16);
  auto mask = builder.AddTensor(::ml_drift::BHWC(1, 1, 4, 128),
                                ::ml_drift::DataType::FLOAT16);
  auto param_tensor = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 7),
                                        ::ml_drift::DataType::INT32);
  auto out_tensor = builder.AddTensor(::ml_drift::BHWC(1, 1, 4, 64),
                                      ::ml_drift::DataType::FLOAT16);

  SdpaTransposedAttributes attr;
  attr.runtime_check.src_end_ch_index = 2;
  attr.bmm1_weights.weights_shape = ::ml_drift::OHWI(128, 1, 1, 64);
  attr.bmm1_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::FLOAT16, attr.bmm1_weights.weights_shape);
  attr.bmm1_weights.desc.layout =
      ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;

  attr.bmm2_weights.weights_shape = ::ml_drift::OHWI(128, 1, 1, 64);
  attr.bmm2_weights.desc = ::ml_drift::GetFullyConnectedWeightsDesc(
      ::ml_drift::DataType::FLOAT16, attr.bmm2_weights.weights_shape);

  auto status =
      BuildSdpaTransposedGpuGraph({q.id, k.id, v.id, mask.id, param_tensor.id},
                                  out_tensor.id, attr, &builder);
  EXPECT_TRUE(status.ok()) << status.message();
}

}  // namespace
}  // namespace litert::ml_drift
