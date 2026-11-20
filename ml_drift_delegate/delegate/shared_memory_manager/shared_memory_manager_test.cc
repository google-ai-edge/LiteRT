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

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/shared_memory_manager/gf32_graph_adapter.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "tflite/c/common.h"

namespace ml_drift {
namespace {

using ::absl_testing::StatusIs;

class FakeGpuSpatialTensor : public GpuSpatialTensor {
 public:
  explicit FakeGpuSpatialTensor(const TensorDescriptor& desc) : desc_(desc) {}

  int Width() const override { return desc_.GetBHWDCShape().w; }
  int Height() const override { return desc_.GetBHWDCShape().h; }
  int Depth() const override { return desc_.GetBHWDCShape().d; }
  int Channels() const override { return desc_.GetBHWDCShape().c; }
  int Slices() const override { return (Channels() + 3) / 4; }
  int Batch() const override { return desc_.GetBHWDCShape().b; }
  TensorDescriptor GetDescriptor() const override { return desc_; }

 private:
  TensorDescriptor desc_;
};

struct IntArrayGuard {
  IntArrayGuard(std::initializer_list<int> dims) {
    array = TfLiteIntArrayCreate(dims.size());
    int i = 0;
    for (int d : dims) array->data[i++] = d;
  }
  ~IntArrayGuard() { TfLiteIntArrayFree(array); }
  TfLiteIntArray* array;
};

TfLiteTensor MakeConstTensor(TfLiteType type, TfLiteIntArray* dims,
                             const void* data, size_t bytes) {
  TfLiteTensor tensor = {};
  tensor.type = type;
  tensor.dims = dims;
  tensor.data.raw_const = static_cast<const char*>(data);
  tensor.bytes = bytes;
  tensor.allocation_type = kTfLiteMmapRo;
  return tensor;
}

GpuInfo MakeDefaultGpuInfo() {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  gpu_info.vendor = GpuVendor::kNvidia;
  return gpu_info;
}

CreateGpuModelInfo MakeCreateInfo() {
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;
  return create_info;
}

struct BatchedFcGraph {
  GraphFloat32 graph;
  Value* input = nullptr;
  Value* weights = nullptr;
  Value* output = nullptr;
  Node* fc_node = nullptr;
};

BatchedFcGraph BuildBatchedFcGraph(const OHWI& weights_shape,
                                   OperationType fc_type) {
  BatchedFcGraph g;
  g.fc_node = g.graph.NewNode();
  g.fc_node->operation.type = ToString(fc_type);
  g.input = g.graph.NewValue();
  g.input->tensor.shape = BHWC(1, 1, 1, weights_shape.i);
  g.input->tensor.type = DataType::kFloat32;
  g.weights = g.graph.NewValue();
  g.weights->tensor.shape =
      BHWC(weights_shape.o, weights_shape.h, weights_shape.w, weights_shape.i);
  g.weights->tensor.type = DataType::kInt8;
  g.output = g.graph.NewValue();
  g.output->tensor.shape = BHWC(1, 1, 1, weights_shape.o);
  g.output->tensor.type = DataType::kFloat32;
  g.graph.AddConsumer(g.fc_node->id, g.input->id);
  g.graph.AddConsumer(g.fc_node->id, g.weights->id);
  g.graph.SetProducer(g.fc_node->id, g.output->id);
  return g;
}

TEST(SharedMemoryManagerHostTest,
     BlockwiseV2BatchedWeightsHioAndHoiProduceMatchingDescriptors) {
  // Logical weights in (O=3, H=2, 1, I=4).
  // For each (o, h, i), value = (o * 8 + h * 4 + i) - 12.
  int8_t hio_data[24];
  int8_t hoi_data[24];
  for (int h = 0; h < 2; ++h) {
    for (int o = 0; o < 3; ++o) {
      for (int i = 0; i < 4; ++i) {
        const int8_t val = static_cast<int8_t>(o * 8 + h * 4 + i - 12);
        hio_data[(h * 4 + i) * 3 + o] = val;
        hoi_data[(h * 3 + o) * 4 + i] = val;
      }
    }
  }
  // Scales for (o, h) = (o + 1) + 10 * h.
  float scales_hio[6];
  float scales_hoi[6];
  for (int h = 0; h < 2; ++h) {
    for (int o = 0; o < 3; ++o) {
      const float s = static_cast<float>((o + 1) + 10 * h);
      scales_hio[h * 3 + o] = s;
      scales_hoi[h * 3 + o] = s;
    }
  }
  float zp_scalar[1] = {-0.5f};

  auto run_case = [&](bool oi_transposed, const int8_t* w_data,
                      const float* s_data, std::vector<uint8_t>* out_weights,
                      std::vector<uint8_t>* out_scales,
                      std::vector<uint8_t>* out_zps) {
    IntArrayGuard w_dims =
        oi_transposed ? IntArrayGuard{2, 4, 3} : IntArrayGuard{2, 3, 4};
    IntArrayGuard s_dims =
        oi_transposed ? IntArrayGuard{2, 1, 3} : IntArrayGuard{2, 3, 1};
    IntArrayGuard zp_dims{1, 1, 1};
    IntArrayGuard bshape =
        oi_transposed ? IntArrayGuard{1, 4, 1} : IntArrayGuard{1, 1, 4};
    TfLiteBlockwiseQuantizationV2 quant = {
        .scale = 1, .zero_point = 2, .block_shape = bshape.array};
    TfLiteTensor tensors[3] = {
        MakeConstTensor(kTfLiteInt8, w_dims.array, w_data, 24),
        MakeConstTensor(kTfLiteFloat32, s_dims.array, s_data,
                        sizeof(float) * 6),
        MakeConstTensor(kTfLiteFloat32, zp_dims.array, zp_scalar,
                        sizeof(zp_scalar)),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &quant};

    TfLiteContext context = {};
    context.tensors = tensors;
    context.tensors_size = 3;

    BatchedFcGraph g = BuildBatchedFcGraph(OHWI(3, 2, 1, 4),
                                           OperationType::kFullyConnectedInt8);
    ValueIdToSharedTensorMap weights_map;
    ValueIdToSharedTensorMap params_map;
    auto create_fn = [](TensorDescriptor& td, size_t,
                        ::litert::ml_drift::ReleaseDataCallback,
                        std::unique_ptr<GpuSpatialTensor>& out) {
      out = std::make_unique<FakeGpuSpatialTensor>(td);
      return absl::OkStatus();
    };
    SharedMemoryManager manager(
        MakeDefaultGpuInfo(), MakeCreateInfo(),
        std::make_unique<GraphFloat32Adapter>(g.graph), create_fn, &context,
        weights_map, params_map, /*has_prepacked_tflite_tensors=*/false,
        /*serialization_cache=*/nullptr, /*madvise_original_tensors=*/false);

    ::litert::ml_drift::SharedTfliteTensor shared_info;
    shared_info.tflite_tensor_id = 0;
    shared_info.global_id = 42;
    shared_info.oi_transposed = oi_transposed;

    absl::flat_hash_map<ValueId, SharedMemoryManager::GlobalId> external_map;
    ASSERT_OK(manager.RegisterExternalConstantTensors(
        g.weights->id, shared_info, external_map));

    // Weights + scale + zero_point registered.
    ASSERT_EQ(external_map.size(), 3);
    ASSERT_EQ(g.graph.FindInputs(g.fc_node->id).size(), 4);

    const SharedConstTensor& entry = weights_map.at(42);
    ASSERT_NE(entry.weights, nullptr);
    ASSERT_TRUE(entry.scale_global_tensor_id.has_value());
    ASSERT_TRUE(entry.zero_point_global_tensor_id.has_value());

    const auto w_span = entry.weights->GetDescriptor().GetData();
    out_weights->assign(w_span.begin(), w_span.end());
    const auto s_span = params_map.at(*entry.scale_global_tensor_id)
                            .weights->GetDescriptor()
                            .GetData();
    out_scales->assign(s_span.begin(), s_span.end());
    const auto zp_span = params_map.at(*entry.zero_point_global_tensor_id)
                             .weights->GetDescriptor()
                             .GetData();
    out_zps->assign(zp_span.begin(), zp_span.end());
  };

  std::vector<uint8_t> hio_weights, hio_scales, hio_zps;
  std::vector<uint8_t> hoi_weights, hoi_scales, hoi_zps;
  run_case(/*oi_transposed=*/true, hio_data, scales_hio, &hio_weights,
           &hio_scales, &hio_zps);
  run_case(/*oi_transposed=*/false, hoi_data, scales_hoi, &hoi_weights,
           &hoi_scales, &hoi_zps);

  EXPECT_FALSE(hio_weights.empty());
  EXPECT_EQ(hio_weights, hoi_weights);
  EXPECT_EQ(hio_scales, hoi_scales);
  EXPECT_EQ(hio_zps, hoi_zps);
}

TEST(SharedMemoryManagerHostTest,
     BlockwiseV2BatchedWeightsSharedAcrossSignatures) {
  int8_t packed_int2[6] = {0x4E, 0x4E, 0x4E, 0x4E, 0x4E, 0x4E};
  float scales[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float zp[1] = {-0.5f};

  IntArrayGuard w_dims{2, 3, 4};
  IntArrayGuard s_dims{2, 3, 1};
  IntArrayGuard zp_dims{1, 1, 1};
  IntArrayGuard bshape{1, 1, 4};
  TfLiteBlockwiseQuantizationV2 quant = {
      .scale = 1, .zero_point = 2, .block_shape = bshape.array};
  TfLiteTensor tensors[3] = {
      MakeConstTensor(kTfLiteInt2, w_dims.array, packed_int2,
                      sizeof(packed_int2)),
      MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales)),
      MakeConstTensor(kTfLiteFloat32, zp_dims.array, zp, sizeof(zp)),
  };
  tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &quant};

  TfLiteContext context = {};
  context.tensors = tensors;
  context.tensors_size = 3;

  ValueIdToSharedTensorMap weights_map;
  ValueIdToSharedTensorMap params_map;
  int create_count = 0;
  auto create_fn = [&](TensorDescriptor& td, size_t,
                       ::litert::ml_drift::ReleaseDataCallback,
                       std::unique_ptr<GpuSpatialTensor>& out) {
    ++create_count;
    out = std::make_unique<FakeGpuSpatialTensor>(td);
    return absl::OkStatus();
  };

  ::litert::ml_drift::SharedTfliteTensor shared_info;
  shared_info.tflite_tensor_id = 0;
  shared_info.global_id = 77;
  shared_info.oi_transposed = false;

  // Signature 1: creates weights, scale, and zero_point.
  BatchedFcGraph g1 =
      BuildBatchedFcGraph(OHWI(3, 2, 1, 4), OperationType::kFullyConnectedInt2);
  SharedMemoryManager mgr1(
      MakeDefaultGpuInfo(), MakeCreateInfo(),
      std::make_unique<GraphFloat32Adapter>(g1.graph), create_fn, &context,
      weights_map, params_map, /*has_prepacked_tflite_tensors=*/false,
      /*serialization_cache=*/nullptr, /*madvise_original_tensors=*/false);
  absl::flat_hash_map<ValueId, SharedMemoryManager::GlobalId> ext1;
  ASSERT_OK(
      mgr1.RegisterExternalConstantTensors(g1.weights->id, shared_info, ext1));
  EXPECT_EQ(create_count, 3);
  ASSERT_EQ(ext1.size(), 3);

  // Signature 2: reuses the cached weights, scale, and zero_point without
  // creating new GPU tensors.
  BatchedFcGraph g2 =
      BuildBatchedFcGraph(OHWI(3, 2, 1, 4), OperationType::kFullyConnectedInt2);
  SharedMemoryManager mgr2(
      MakeDefaultGpuInfo(), MakeCreateInfo(),
      std::make_unique<GraphFloat32Adapter>(g2.graph), create_fn, &context,
      weights_map, params_map, /*has_prepacked_tflite_tensors=*/false,
      /*serialization_cache=*/nullptr, /*madvise_original_tensors=*/false);
  absl::flat_hash_map<ValueId, SharedMemoryManager::GlobalId> ext2;
  ASSERT_OK(
      mgr2.RegisterExternalConstantTensors(g2.weights->id, shared_info, ext2));
  EXPECT_EQ(create_count, 3);
  ASSERT_EQ(ext2.size(), 3);
  EXPECT_EQ(ext2.at(g2.weights->id).value, 77);
  EXPECT_TRUE(ext2.at(g2.weights->id).IsSourceId());
  ASSERT_EQ(g2.graph.FindInputs(g2.fc_node->id).size(), 4);
}

TEST(SharedMemoryManagerHostTest, BlockwiseV2ErrorGuards) {
  int8_t weights_data[24] = {};
  float scales[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  IntArrayGuard w_dims{2, 4, 3};
  IntArrayGuard s_dims{2, 1, 3};
  IntArrayGuard bshape{1, 4, 1};
  TfLiteBlockwiseQuantizationV2 quant = {.scale = 1,
                                         .zero_point = kTfLiteOptionalTensor,
                                         .block_shape = bshape.array};

  auto create_fn = [](TensorDescriptor& td, size_t,
                      ::litert::ml_drift::ReleaseDataCallback,
                      std::unique_ptr<GpuSpatialTensor>& out) {
    out = std::make_unique<FakeGpuSpatialTensor>(td);
    return absl::OkStatus();
  };

  // 1. dequant_forced = true on kTfLiteBlockwiseQuantizationV2 ->
  // Unimplemented.
  {
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, w_dims.array, weights_data, 24),
        MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales)),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &quant};
    TfLiteContext context = {};
    context.tensors = tensors;
    context.tensors_size = 2;

    BatchedFcGraph g = BuildBatchedFcGraph(OHWI(3, 2, 1, 4),
                                           OperationType::kFullyConnectedInt8);
    ValueIdToSharedTensorMap weights_map, params_map;
    SharedMemoryManager mgr(MakeDefaultGpuInfo(), MakeCreateInfo(),
                            std::make_unique<GraphFloat32Adapter>(g.graph),
                            create_fn, &context, weights_map, params_map, false,
                            nullptr, false);
    ::litert::ml_drift::SharedTfliteTensor shared_info;
    shared_info.tflite_tensor_id = 0;
    shared_info.global_id = 1;
    shared_info.dequant_forced = true;
    absl::flat_hash_map<ValueId, SharedMemoryManager::GlobalId> ext;
    EXPECT_THAT(
        mgr.RegisterExternalConstantTensors(g.weights->id, shared_info, ext),
        StatusIs(absl::StatusCode::kUnimplemented));
  }

  // 2. Null host data without external buffer -> FailedPrecondition.
  {
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, w_dims.array, nullptr, 24),
        MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales)),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &quant};
    TfLiteContext context = {};
    context.tensors = tensors;
    context.tensors_size = 2;

    BatchedFcGraph g = BuildBatchedFcGraph(OHWI(3, 2, 1, 4),
                                           OperationType::kFullyConnectedInt8);
    ValueIdToSharedTensorMap weights_map, params_map;
    SharedMemoryManager mgr(MakeDefaultGpuInfo(), MakeCreateInfo(),
                            std::make_unique<GraphFloat32Adapter>(g.graph),
                            create_fn, &context, weights_map, params_map, false,
                            nullptr, false);
    ::litert::ml_drift::SharedTfliteTensor shared_info;
    shared_info.tflite_tensor_id = 0;
    shared_info.global_id = 2;
    absl::flat_hash_map<ValueId, SharedMemoryManager::GlobalId> ext;
    EXPECT_THAT(
        mgr.RegisterExternalConstantTensors(g.weights->id, shared_info, ext),
        StatusIs(absl::StatusCode::kFailedPrecondition));
  }

  // 3. Externalized batched weights -> Unimplemented.
  {
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, w_dims.array, nullptr, 24),
        MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales)),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &quant};
    TfLiteContext context = {};
    context.tensors = tensors;
    context.tensors_size = 2;

    BatchedFcGraph g = BuildBatchedFcGraph(OHWI(3, 2, 1, 4),
                                           OperationType::kFullyConnectedInt8);
    ValueIdToSharedTensorMap weights_map, params_map;
    SharedMemoryManager mgr(
        MakeDefaultGpuInfo(), MakeCreateInfo(),
        std::make_unique<GraphFloat32Adapter>(g.graph), create_fn, &context,
        weights_map, params_map, false, nullptr, false,
        /*experimental_int4_unpacking=*/false,
        /*experimental_int2_unpacking=*/false,
        /*create_tensor_from_device_buffer_func=*/nullptr,
        /*maybe_bind_tensor_data_func=*/nullptr,
        /*packing_lookup_func=*/nullptr,
        /*maybe_get_external_buffer_id_func=*/
        [](int) -> absl::StatusOr<uint32_t> { return 99U; });
    ::litert::ml_drift::SharedTfliteTensor shared_info;
    shared_info.tflite_tensor_id = 0;
    shared_info.global_id = 3;
    absl::flat_hash_map<ValueId, SharedMemoryManager::GlobalId> ext;
    EXPECT_THAT(
        mgr.RegisterExternalConstantTensors(g.weights->id, shared_info, ext),
        StatusIs(absl::StatusCode::kUnimplemented));
  }

  // 4. Undersized weights buffer -> InvalidArgument.
  {
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, w_dims.array, weights_data, 12),
        MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales)),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &quant};
    TfLiteContext context = {};
    context.tensors = tensors;
    context.tensors_size = 2;

    BatchedFcGraph g = BuildBatchedFcGraph(OHWI(3, 2, 1, 4),
                                           OperationType::kFullyConnectedInt8);
    ValueIdToSharedTensorMap weights_map, params_map;
    SharedMemoryManager mgr(MakeDefaultGpuInfo(), MakeCreateInfo(),
                            std::make_unique<GraphFloat32Adapter>(g.graph),
                            create_fn, &context, weights_map, params_map, false,
                            nullptr, false);
    ::litert::ml_drift::SharedTfliteTensor shared_info;
    shared_info.tflite_tensor_id = 0;
    shared_info.global_id = 4;
    absl::flat_hash_map<ValueId, SharedMemoryManager::GlobalId> ext;
    EXPECT_THAT(
        mgr.RegisterExternalConstantTensors(g.weights->id, shared_info, ext),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 5. Non-positive weights shape dimension -> InvalidArgument.
  {
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, w_dims.array, weights_data, 24),
        MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales)),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &quant};
    TfLiteContext context = {};
    context.tensors = tensors;
    context.tensors_size = 2;

    BatchedFcGraph g = BuildBatchedFcGraph(OHWI(3, 0, 1, 4),
                                           OperationType::kFullyConnectedInt8);
    ValueIdToSharedTensorMap weights_map, params_map;
    SharedMemoryManager mgr(MakeDefaultGpuInfo(), MakeCreateInfo(),
                            std::make_unique<GraphFloat32Adapter>(g.graph),
                            create_fn, &context, weights_map, params_map, false,
                            nullptr, false);
    ::litert::ml_drift::SharedTfliteTensor shared_info;
    shared_info.tflite_tensor_id = 0;
    shared_info.global_id = 5;
    absl::flat_hash_map<ValueId, SharedMemoryManager::GlobalId> ext;
    EXPECT_THAT(
        mgr.RegisterExternalConstantTensors(g.weights->id, shared_info, ext),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 6. Overflowing weights shape element count -> InvalidArgument.
  {
    TfLiteTensor tensors[2] = {
        MakeConstTensor(kTfLiteInt8, w_dims.array, weights_data, 24),
        MakeConstTensor(kTfLiteFloat32, s_dims.array, scales, sizeof(scales)),
    };
    tensors[0].quantization = {kTfLiteBlockwiseQuantizationV2, &quant};
    TfLiteContext context = {};
    context.tensors = tensors;
    context.tensors_size = 2;

    BatchedFcGraph g =
        BuildBatchedFcGraph(OHWI(1 << 20, 1 << 20, 1 << 20, 1 << 20),
                            OperationType::kFullyConnectedInt8);
    ValueIdToSharedTensorMap weights_map, params_map;
    SharedMemoryManager mgr(MakeDefaultGpuInfo(), MakeCreateInfo(),
                            std::make_unique<GraphFloat32Adapter>(g.graph),
                            create_fn, &context, weights_map, params_map, false,
                            nullptr, false);
    ::litert::ml_drift::SharedTfliteTensor shared_info;
    shared_info.tflite_tensor_id = 0;
    shared_info.global_id = 6;
    absl::flat_hash_map<ValueId, SharedMemoryManager::GlobalId> ext;
    EXPECT_THAT(
        mgr.RegisterExternalConstantTensors(g.weights->id, shared_info, ext),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
}  // namespace ml_drift
