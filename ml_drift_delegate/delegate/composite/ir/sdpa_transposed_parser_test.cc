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

#include "ml_drift_delegate/delegate/composite/ir/sdpa_transposed_parser.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"
#include "ml_drift_delegate/tflite/convert/convert_testing_utils.h"
#include "ml_drift_delegate/tflite/convert/stub_delegate.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/interpreter.h"

namespace litert::ml_drift::ir {
namespace {

using ::testing::Each;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::SizeIs;

TfLiteStablehloCompositeParams* CreateSdpaTransposedParams(
    std::optional<float> softcap = std::nullopt,
    std::optional<bool> is_causal = std::nullopt,
    std::optional<bool> from_cache_update = std::nullopt) {
  size_t total_size = sizeof(TfLiteStablehloCompositeParams);
  std::vector<uint8_t> buffer;

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    if (softcap.has_value()) {
      fbb.Float("softcap", *softcap);
    }
    if (is_causal.has_value()) {
      fbb.Bool("is_causal", *is_causal);
    }
    if (from_cache_update.has_value()) {
      fbb.Bool("from_cache_update", *from_cache_update);
    }
  });
  fbb.Finish();
  buffer = fbb.GetBuffer();
  total_size += buffer.size();

  void* block = calloc(1, total_size);
  TfLiteStablehloCompositeParams* params =
      reinterpret_cast<TfLiteStablehloCompositeParams*>(block);
  params->name = "odml.sdpa_transposed";

  uint8_t* attr_data = reinterpret_cast<uint8_t*>(params + 1);
  params->attributes = attr_data;
  params->attributes_size = buffer.size();
  memcpy(attr_data, buffer.data(), buffer.size());

  return params;
}

class ConvertSdpaTransposedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    gpu_info_.vendor = ::ml_drift::GpuVendor::kApple;
    gpu_info_.gpu_api = ::ml_drift::GpuApi::kMetal;
    const IrModelBuilderOptions options = {.gpu_info = &gpu_info_};
    CustomIrOpMap custom_parsers;
    custom_parsers["odml.sdpa_transposed"] = GetSdpaTransposedParser();
    delegate_ = CreateStubDelegate(options, std::move(custom_parsers));
    ASSERT_TRUE(delegate_);
  }

  void TearDown() override { DeleteStubDelegate(delegate_); }

  ::ml_drift::GpuInfo gpu_info_;
  TfLiteDelegate* delegate_;
};

TEST_F(ConvertSdpaTransposedTest, BasicFp32) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});    // q
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // k
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // v
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});   // result

  TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams();
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sdpa_transposed"));
  EXPECT_THAT(op->inputs, SizeIs(4));
  EXPECT_THAT(op->outputs, SizeIs(1));

  const auto* attr =
      std::any_cast<::litert::ml_drift::SdpaTransposedAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->bmm1_weights.desc.type, ::ml_drift::DataType::kFloat32);
  EXPECT_EQ(attr->bmm2_weights.desc.type, ::ml_drift::DataType::kFloat32);
  EXPECT_FALSE(attr->from_cache_update);
}

TEST_F(ConvertSdpaTransposedTest, BasicFp16) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat16, {1, 1, 4, 64});    // q
  builder.AddInput(kTfLiteFloat16, {1, 1, 128, 64});  // k
  builder.AddInput(kTfLiteFloat16, {1, 1, 128, 64});  // v
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param tensor
  builder.AddOutput(kTfLiteFloat16, {1, 1, 4, 64});   // result

  TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams();
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sdpa_transposed"));

  const auto* attr =
      std::any_cast<::litert::ml_drift::SdpaTransposedAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->bmm1_weights.desc.type, ::ml_drift::DataType::kFloat16);
  EXPECT_EQ(attr->bmm2_weights.desc.type, ::ml_drift::DataType::kFloat16);
}

TEST_F(ConvertSdpaTransposedTest, ThreeInputs) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});    // q
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // k
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // v
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});   // result

  TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams();
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sdpa_transposed"));
  EXPECT_THAT(op->inputs, SizeIs(3));
  EXPECT_THAT(op->outputs, SizeIs(1));
}

TEST_F(ConvertSdpaTransposedTest, FiveInputsWithMaskAndParams) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});    // q
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // k
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // v
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 128});   // mask
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});   // result

  TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams();
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sdpa_transposed"));
  EXPECT_THAT(op->inputs, SizeIs(5));
  EXPECT_THAT(op->outputs, SizeIs(1));
}

TEST_F(ConvertSdpaTransposedTest, WithSoftcap) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});    // q
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // k
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // v
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});   // result

  TfLiteStablehloCompositeParams* params =
      CreateSdpaTransposedParams(/*softcap=*/30.0f);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sdpa_transposed"));

  const auto* attr =
      std::any_cast<::litert::ml_drift::SdpaTransposedAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_TRUE(attr->softcap.has_value());
  EXPECT_EQ(*attr->softcap, 30.0f);
}

TEST_F(ConvertSdpaTransposedTest, BoolMaskKeptWhenIsCausalFalse) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});    // q (prefill)
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // k
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // v
  builder.AddInput(kTfLiteBool, {1, 1, 4, 128});      // bool mask
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});   // result

  TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams(
      /*softcap=*/std::nullopt, /*is_causal=*/false,
      /*from_cache_update=*/true);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sdpa_transposed"));
  // Because is_causal=false was explicitly specified, the BOOL mask input must
  // NOT be pruned even on Apple.
  EXPECT_THAT(op->inputs, SizeIs(5));

  const auto* attr =
      std::any_cast<::litert::ml_drift::SdpaTransposedAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_FALSE(attr->is_causal);
}

// The fixture's Apple Metal `GpuInfo` selects the MMA Flash prefill kernel.
TEST_F(ConvertSdpaTransposedTest, BoolMaskPrunedWhenIsCausalTrueOnApple) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 1, 4, 64});    // q (prefill)
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // k
  builder.AddInput(kTfLiteFloat32, {1, 1, 128, 64});  // v
  builder.AddInput(kTfLiteBool, {1, 1, 4, 128});      // bool mask
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});       // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 4, 64});   // result

  TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams(
      /*softcap=*/std::nullopt, /*is_causal=*/true,
      /*from_cache_update=*/true);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sdpa_transposed"));
  // On Apple, the global BOOL causal mask is pruned leaving 4 inputs (q, k, v,
  // params) and attr.is_causal = true.
  EXPECT_THAT(op->inputs, SizeIs(4));

  const auto* attr =
      std::any_cast<::litert::ml_drift::SdpaTransposedAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_TRUE(attr->is_causal);
}

// The MMA prefill kernel needs query heads that are a multiple of the K/V
// heads and a cache of whole 4-key blocks, so other prefill shapes keep the
// BOOL causal mask on Apple.
TEST_F(ConvertSdpaTransposedTest,
       BoolMaskKeptForPrefillUnsupportedShapesOnApple) {
  struct Shapes {
    int q_heads;
    int kv_heads;
    int cache_size;
  };
  const Shapes cases[] = {
      {/*q_heads=*/12, /*kv_heads=*/8, /*cache_size=*/128},
      {/*q_heads=*/8, /*kv_heads=*/2, /*cache_size=*/130},
  };
  for (const Shapes& s : cases) {
    SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
    builder.AddInput(kTfLiteFloat32, {1, s.q_heads, 4, 64});  // q (prefill)
    builder.AddInput(kTfLiteFloat32, {1, s.kv_heads, s.cache_size, 64});  // k
    builder.AddInput(kTfLiteFloat32, {1, s.kv_heads, 64, s.cache_size});  // v
    builder.AddInput(kTfLiteBool, {1, 1, 4, s.cache_size});    // bool mask
    builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});              // param tensor
    builder.AddOutput(kTfLiteFloat32, {1, s.q_heads, 4, 64});  // result

    TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams(
        /*softcap=*/std::nullopt, /*is_causal=*/true,
        /*from_cache_update=*/true);
    builder.SetParameters(params);

    auto interpreter = builder.Build();
    ASSERT_NE(interpreter, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

    const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
    ASSERT_TRUE(ir_model);
    ASSERT_THAT(ir_model->ops(), SizeIs(1));
    EXPECT_THAT(ir_model->ops()[0]->inputs, SizeIs(5))
        << "q_heads=" << s.q_heads << ", kv_heads=" << s.kv_heads
        << ", cache_size=" << s.cache_size;
  }
}

// Expects the single `sdpa_transposed` IR op built for a decode (`q_w = 1`)
// BOOL-masked causal SDPA with `head_dim` to have `expected_inputs` inputs.
void ExpectDecodeSdpaInputs(TfLiteDelegate* delegate, int head_dim,
                            int expected_inputs) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 8, 1, head_dim});       // q (decode)
  builder.AddInput(kTfLiteFloat32, {1, 2, 128, head_dim});     // k
  builder.AddInput(kTfLiteFloat32, {1, 2, head_dim, 128});     // v
  builder.AddInput(kTfLiteBool, {1, 1, 1, 128});               // bool mask
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});                // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 8 * head_dim});  // result

  TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams(
      /*softcap=*/std::nullopt, /*is_causal=*/true,
      /*from_cache_update=*/true);
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sdpa_transposed"));
  EXPECT_THAT(op->inputs, SizeIs(expected_inputs)) << "head_dim=" << head_dim;
}

// Flash-Decode supports every head_dim whose head_dim / 4 channel slices
// divide 256, so the BOOL causal mask is pruned (q, k, v, params remain).
TEST_F(ConvertSdpaTransposedTest, BoolMaskPrunedForDecodeHeadDimsOnApple) {
  for (int head_dim : {4, 64, 128, 256, 512, 1024}) {
    ExpectDecodeSdpaInputs(delegate_, head_dim, /*expected_inputs=*/4);
  }
}

// Other head_dims may not take Flash-Decode, so the mask is kept.
TEST_F(ConvertSdpaTransposedTest, BoolMaskKeptForDecodeUnsupportedHeadDims) {
  for (int head_dim : {96, 384, 2048}) {
    ExpectDecodeSdpaInputs(delegate_, head_dim, /*expected_inputs=*/5);
  }
}

// Non-Metal backends without the fused SDPA kernels (OpenGL, Vulkan) keep the
// BOOL causal mask.
TEST_F(ConvertSdpaTransposedTest, BoolMaskKeptOnBackendsWithoutFusedSdpa) {
  for (::ml_drift::GpuApi api :
       {::ml_drift::GpuApi::kOpenGl, ::ml_drift::GpuApi::kVulkan}) {
    gpu_info_.gpu_api = api;
    ExpectDecodeSdpaInputs(delegate_, /*head_dim=*/128, /*expected_inputs=*/5);
  }
}

// OpenCL and WebGPU run decode on the work-group Flash-Decode kernel, so the
// BOOL causal mask is pruned for the supported head dims. On WebGPU, the whole
// GQA group (4 query heads per K/V head here) must fit in one work group, which
// needs more than the default 16 KB of workgroup storage.
TEST_F(ConvertSdpaTransposedTest, BoolMaskPrunedForDecodeOnOpenClAndWebGpu) {
  for (::ml_drift::GpuApi api :
       {::ml_drift::GpuApi::kOpenCl, ::ml_drift::GpuApi::kWebGpu}) {
    gpu_info_.vendor = ::ml_drift::GpuVendor::kQualcomm;
    gpu_info_.gpu_api = api;
    gpu_info_.webgpu_info.max_compute_workgroup_storage_size = 32 * 1024;
    for (int head_dim : {64, 128, 256}) {
      ExpectDecodeSdpaInputs(delegate_, head_dim, /*expected_inputs=*/4);
    }
  }
}

// With the default 16 KB of workgroup storage, only part of the GQA group fits
// in one work group, so WebGPU decode keeps the decomposed graph and its mask.
TEST_F(ConvertSdpaTransposedTest,
       BoolMaskKeptForWebGpuDecodeSplittingGqaGroup) {
  gpu_info_.vendor = ::ml_drift::GpuVendor::kQualcomm;
  gpu_info_.gpu_api = ::ml_drift::GpuApi::kWebGpu;
  gpu_info_.webgpu_info.max_compute_workgroup_storage_size = 16 * 1024;
  for (int head_dim : {64, 256}) {
    ExpectDecodeSdpaInputs(delegate_, head_dim, /*expected_inputs=*/5);
  }
}

// On OpenCL and WebGPU, prefill uses the work-group Flash kernel (and prunes
// the BOOL causal mask) only when the `[1, heads, tokens, cache_size]` F16
// score tensor of the decomposed graph would exceed
// `kMaxDecomposedSdpaScoreBytes`: 32 heads x 1024 tokens x 16384 keys x 2 B =
// 1 GiB here, while a 128-entry cache keeps the decomposed graph and its mask.
TEST_F(ConvertSdpaTransposedTest,
       BoolMaskPrunedForLargeScorePrefillOnOpenClAndWebGpu) {
  for (::ml_drift::GpuApi api :
       {::ml_drift::GpuApi::kOpenCl, ::ml_drift::GpuApi::kWebGpu}) {
    gpu_info_.vendor = ::ml_drift::GpuVendor::kQualcomm;
    gpu_info_.gpu_api = api;
    for (int cache_size : {128, 16384}) {
      SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
      builder.AddInput(kTfLiteFloat32, {1, 32, 1024, 64});       // q (prefill)
      builder.AddInput(kTfLiteFloat32, {1, 8, cache_size, 64});  // k
      builder.AddInput(kTfLiteFloat32, {1, 8, 64, cache_size});  // v
      builder.AddInput(kTfLiteBool, {1, 1, 1024, cache_size});   // bool mask
      builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});              // param tensor
      builder.AddOutput(kTfLiteFloat32, {1, 32, 1024, 64});      // result

      TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams(
          /*softcap=*/std::nullopt, /*is_causal=*/true,
          /*from_cache_update=*/true);
      builder.SetParameters(params);

      auto interpreter = builder.Build();
      ASSERT_NE(interpreter, nullptr);
      ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

      const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
      ASSERT_TRUE(ir_model);
      ASSERT_THAT(ir_model->ops(), SizeIs(1));
      const auto& op = ir_model->ops()[0];
      EXPECT_THAT(op->inputs, SizeIs(cache_size == 16384 ? 4 : 5))
          << "cache_size=" << cache_size;
      const auto* attr =
          std::any_cast<::litert::ml_drift::SdpaTransposedAttributes>(
              &op->attr);
      ASSERT_NE(attr, nullptr);
      EXPECT_TRUE(attr->is_causal);
      EXPECT_TRUE(attr->is_prefill);
    }
  }
}

// Sets the per-tensor quantization scale of an affine quantized `tensor`.
void SetQuantizationScale(TfLiteTensor* tensor, float scale) {
  ASSERT_EQ(tensor->quantization.type, kTfLiteAffineQuantization);
  auto* quantization =
      static_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  ASSERT_EQ(quantization->scale->size, 1);
  quantization->scale->data[0] = scale;
}

// INT8 symmetric per-tensor K/V caches are handed to the kernel as kUint8 byte
// tensors, read by both matmuls as int8 external weights dequantized with one
// scale per output channel.
TEST_F(ConvertSdpaTransposedTest, Int8KvCache) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 8, 1, 64});     // q (decode)
  builder.AddInput(kTfLiteInt8, {1, 2, 128, 64});      // k
  builder.AddInput(kTfLiteInt8, {1, 2, 64, 128});      // v
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});        // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 512});   // result

  TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams();
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);
  SetQuantizationScale(interpreter->tensor(1), 0.25f);
  SetQuantizationScale(interpreter->tensor(2), 0.125f);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate_), kTfLiteOk);

  const ::ml_drift::ir::IrModel* ir_model = GetIrModel(delegate_);
  ASSERT_TRUE(ir_model);

  ASSERT_THAT(ir_model->ops(), SizeIs(1));
  const auto& op = ir_model->ops()[0];
  EXPECT_THAT(op->name, Eq("sdpa_transposed"));
  ASSERT_THAT(op->inputs, SizeIs(4));
  EXPECT_THAT(op->outputs, SizeIs(1));

  // The int8 caches are retagged as raw kUint8 byte tensors.
  for (int input : {1, 2}) {
    const ::ml_drift::ir::IrTensor* cache = ir_model->tensor(op->inputs[input]);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->desc.GetDataType(), ::ml_drift::DataType::kUint8);
    EXPECT_FALSE(cache->quant_params.has_value());
  }

  const auto* attr =
      std::any_cast<::litert::ml_drift::SdpaTransposedAttributes>(&op->attr);
  ASSERT_NE(attr, nullptr);
  EXPECT_TRUE(attr->from_cache_update);
  EXPECT_FALSE(attr->is_prefill);
  EXPECT_EQ(attr->bmm1_weights.desc.type, ::ml_drift::DataType::kUint8);
  EXPECT_EQ(attr->bmm1_weights.desc.layout,
            ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4);
  EXPECT_EQ(attr->bmm2_weights.desc.type, ::ml_drift::DataType::kUint8);
  // One scale per cache slot for Q*K^T and per head_dim element for P*V.
  ASSERT_TRUE(attr->scale_k.has_value());
  EXPECT_EQ(attr->scale_k->shape.v, 128);
  EXPECT_THAT(attr->scale_k->data, Each(FloatEq(0.25f)));
  ASSERT_TRUE(attr->scale_v.has_value());
  EXPECT_EQ(attr->scale_v->shape.v, 64);
  EXPECT_THAT(attr->scale_v->data, Each(FloatEq(0.125f)));
}

// Mixed int8 K / float V caches are not supported and must be left to the CPU
// decomposition.
TEST_F(ConvertSdpaTransposedTest, Int8KeyWithFloatValueIsNotSupported) {
  SingleOpInterpreterBuilder builder(kTfLiteBuiltinStablehloComposite);
  builder.AddInput(kTfLiteFloat32, {1, 8, 1, 64});     // q (decode)
  builder.AddInput(kTfLiteInt8, {1, 2, 128, 64});      // k
  builder.AddInput(kTfLiteFloat32, {1, 2, 64, 128});   // v
  builder.AddInput(kTfLiteInt32, {1, 1, 1, 7});        // param tensor
  builder.AddOutput(kTfLiteFloat32, {1, 1, 1, 512});   // result

  TfLiteStablehloCompositeParams* params = CreateSdpaTransposedParams();
  builder.SetParameters(params);

  auto interpreter = builder.Build();
  ASSERT_NE(interpreter, nullptr);

  const auto* pair = interpreter->node_and_registration(0);
  const CustomIrOpParser parser = GetSdpaTransposedParser();
  const absl::Status status = parser.is_supported(
      interpreter->primary_subgraph().context(), &pair->first, &pair->second);
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable) << status;
}

}  // namespace
}  // namespace litert::ml_drift::ir
