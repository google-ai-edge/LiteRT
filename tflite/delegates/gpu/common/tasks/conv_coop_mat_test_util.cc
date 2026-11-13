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

#include "tflite/delegates/gpu/common/tasks/conv_coop_mat_test_util.h"

#include <cmath>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "tflite/delegates/gpu/common/data_type.h"
#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/common/precision.h"
#include "tflite/delegates/gpu/common/shape.h"
#include "tflite/delegates/gpu/common/status.h"
#include "tflite/delegates/gpu/common/task/gpu_operation.h"
#include "tflite/delegates/gpu/common/task/tensor_desc.h"
#include "tflite/delegates/gpu/common/task/testing_util.h"
#include "tflite/delegates/gpu/common/tasks/conv_coop_mat.h"
#include "tflite/delegates/gpu/common/tensor.h"
#include "tflite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {

struct ConvTestCase {
  HW kernel;
  HW stride;
  HW dilation;
  HW padding;
  int output_channels;
  int input_channels = 64;
};

// The reference uses scalar OHWI convolution, independently of the GPU's tiled
// weight packing and matrix multiplication, to check actual operation results.
TensorFloat32 ReferenceConvolution(const TensorFloat32& src,
                                  const Convolution2DAttributes& attr,
                                  const BHWC& dst_shape) {
  TensorFloat32 dst;
  dst.shape = dst_shape;
  dst.data.resize(dst_shape.DimensionsProduct());
  for (int y = 0; y < dst_shape.h; ++y) {
    for (int x = 0; x < dst_shape.w; ++x) {
      for (int o = 0; o < dst_shape.c; ++o) {
        float sum = attr.bias.data[o];
        for (int ky = 0; ky < attr.weights.shape.h; ++ky) {
          const int sy = y * attr.strides.h - attr.padding.prepended.h +
                         ky * attr.dilations.h;
          for (int kx = 0; kx < attr.weights.shape.w; ++kx) {
            const int sx = x * attr.strides.w - attr.padding.prepended.w +
                           kx * attr.dilations.w;
            if (sy < 0 || sy >= src.shape.h || sx < 0 || sx >= src.shape.w) {
              continue;
            }
            for (int i = 0; i < src.shape.c; ++i) {
              const int wi = ((o * attr.weights.shape.h + ky) *
                                  attr.weights.shape.w +
                              kx) * src.shape.c + i;
              sum += src.data[(sy * src.shape.w + sx) * src.shape.c + i] *
                     attr.weights.data[wi];
            }
          }
        }
        dst.data[(y * dst_shape.w + x) * dst_shape.c + o] = sum;
      }
    }
  }
  return dst;
}

absl::Status RunConfiguration(TestExecutionEnvironment* env,
                              const OperationDef& definition,
                              const ConvTestCase& test_case) {
  const BHWC src_shape(1, 17, 25, test_case.input_channels);
  Convolution2DAttributes attr;
  attr.strides = test_case.stride;
  attr.dilations = test_case.dilation;
  attr.padding.prepended = test_case.padding;
  attr.padding.appended = test_case.padding;
  attr.weights.shape = OHWI(test_case.output_channels, test_case.kernel.h,
                            test_case.kernel.w, src_shape.c);
  const BHWC dst_shape(
      1,
      (src_shape.h + 2 * test_case.padding.h -
       (test_case.kernel.h - 1) * test_case.dilation.h - 1) /
              test_case.stride.h + 1,
      (src_shape.w + 2 * test_case.padding.w -
       (test_case.kernel.w - 1) * test_case.dilation.w - 1) /
              test_case.stride.w + 1,
      test_case.output_channels);
  if (!IsConvCoopMatSupported(env->GetGpuInfo(), definition, attr, dst_shape)) {
    return absl::OkStatus();
  }

  TensorFloat32 src;
  src.shape = src_shape;
  src.data.resize(src.shape.DimensionsProduct());
  for (int i = 0; i < static_cast<int>(src.data.size()); ++i) {
    // Nonuniform, signed values represented exactly in both float and half.
    src.data[i] = static_cast<float>((i * 7 + i / 11) % 9 - 4) / 16.0f;
  }
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct());
  for (int i = 0; i < static_cast<int>(attr.weights.data.size()); ++i) {
    attr.weights.data[i] =
        static_cast<float>((i * 3 + i / 13) % 9 - 4) / 32.0f;
  }
  attr.bias.shape = Linear(test_case.output_channels);
  attr.bias.data.resize(test_case.output_channels);
  for (int i = 0; i < test_case.output_channels; ++i) {
    attr.bias.data[i] = static_cast<float>(i % 5 - 2) / 16.0f;
  }

  const auto expected = ReferenceConvolution(src, attr, dst_shape);
  TensorFloat32 actual;
  auto operation = CreateConvCoopMat(definition, attr);
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      src, std::make_unique<ConvCoopMat>(std::move(operation)), dst_shape,
      &actual));
  for (const auto value : actual.data) {
    if (!std::isfinite(value)) {
      return absl::InternalError("Convolution produced a nonfinite output");
    }
  }
  // The cooperative matrix path accumulates in the storage type: with F16,
  // every one of the up to kh*kw*cin_groups*16 mulAdd steps may round, so
  // the F16 eps is wider than the exact-arithmetic value in conv_generic.
  const float eps = definition.precision == CalculationsPrecision::F32
                        ? 1.0e-4f : 2.0e-2f;
  return PointWiseNear(expected.data, actual.data, eps);
}

absl::Status RunTest(TestExecutionEnvironment* env,
                     const ConvTestCase& test_case) {
  for (const auto precision : env->GetSupportedPrecisions()) {
    const auto data_type = DeduceDataTypeFromPrecision(precision);
    for (const auto storage : env->GetSupportedStorages(data_type)) {
      // conv_generic covers HWC only. Both layouts run here because batch
      // reference addressing (SetBatchRef) is part of the contract under
      // test and only BHWC exercises the batch coordinate path.
      for (const auto layout : {Layout::HWC, Layout::BHWC}) {
        OperationDef definition;
        definition.precision = precision;
        definition.src_tensors.push_back({data_type, storage, layout});
        definition.dst_tensors = definition.src_tensors;
        const auto status = RunConfiguration(env, definition, test_case);
        if (!status.ok()) {
          return absl::Status(
              status.code(),
              absl::StrCat("precision=", static_cast<int>(precision),
                           ", storage=", static_cast<int>(storage),
                           ", layout=", static_cast<int>(layout), ": ",
                           status.message()));
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ConvCoopMat1x1Test(TestExecutionEnvironment* env) {
  return RunTest(env, {HW(1, 1), HW(1, 1), HW(1, 1), HW(0, 0), 64});
}

absl::Status ConvCoopMatSmallChannelCountsTest(TestExecutionEnvironment* env) {
  // Exercise one/two accumulator slots, partial vectors, and an idle subgroup.
  // The factory is tested directly even when the selector prefers another conv.
  for (const int output_channels : {1, 16, 17, 32, 33, 48}) {
    RETURN_IF_ERROR(RunTest(
        env, {HW(1, 1), HW(1, 1), HW(1, 1), HW(0, 0), output_channels, 16}));
  }
  return absl::OkStatus();
}

absl::Status ConvCoopMatKnownWeightsTest(TestExecutionEnvironment* env) {
  for (const auto precision : env->GetSupportedPrecisions()) {
    const auto type = DeduceDataTypeFromPrecision(precision);
    for (const auto storage : env->GetSupportedStorages(type)) {
      for (const auto layout : {Layout::HWC, Layout::BHWC}) {
        OperationDef definition;
        definition.precision = precision;
        definition.src_tensors.push_back({type, storage, layout});
        definition.dst_tensors = definition.src_tensors;
        for (const int channels : {17, 65, 129}) {
          Convolution2DAttributes attr;
          attr.strides = HW(1, 1);
          attr.dilations = HW(1, 1);
          attr.padding.prepended = HW(0, 0);
          attr.padding.appended = HW(0, 0);
          attr.weights.shape = OHWI(channels, 1, 1, 16);
          const BHWC dst_shape(1, 3, 3, channels);
          if (!IsConvCoopMatSupported(env->GetGpuInfo(), definition, attr,
                                     dst_shape)) {
            continue;
          }
          TensorFloat32 src;
          src.shape = BHWC(1, 3, 3, 16);
          src.data.assign(src.shape.DimensionsProduct(), 1.0f);
          attr.weights.data.resize(channels * 16);
          for (int o = 0; o < channels; ++o) {
            for (int i = 0; i < 16; ++i) {
              attr.weights.data[o * 16 + i] = (o + 1) / 32.0f;
            }
          }
          attr.bias.shape = Linear(channels);
          attr.bias.data.assign(channels, 0.0f);
          TensorFloat32 expected;
          expected.shape = dst_shape;
          expected.data.resize(dst_shape.DimensionsProduct());
          for (int i = 0; i < static_cast<int>(expected.data.size()); ++i) {
            // Sixteen ones times the constant weight: exactly (o+1)/2.
            expected.data[i] = (i % channels + 1) / 2.0f;
          }
          TensorFloat32 actual;
          auto operation = CreateConvCoopMat(definition, attr);
          RETURN_IF_ERROR(env->ExecuteGPUOperation(
              src, std::make_unique<ConvCoopMat>(std::move(operation)),
              dst_shape, &actual));
          for (const auto value : actual.data) {
            if (!std::isfinite(value)) {
              return absl::InternalError("Convolution produced nonfinite output");
            }
          }
          RETURN_IF_ERROR(PointWiseNear(expected.data, actual.data, 1.0e-3f));
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConvCoopMat1x1OutputChannelsTailTest(TestExecutionEnvironment* env) {
  return RunTest(env, {HW(1, 1), HW(1, 1), HW(1, 1), HW(0, 0), 65});
}

absl::Status ConvCoopMat3x3PaddingTest(TestExecutionEnvironment* env) {
  return RunTest(env, {HW(3, 3), HW(1, 1), HW(1, 1), HW(1, 1), 65});
}

absl::Status ConvCoopMat3x3OutputChannelsTailTest(TestExecutionEnvironment* env) {
  return RunTest(env, {HW(3, 3), HW(1, 1), HW(1, 1), HW(1, 1), 112});
}

absl::Status ConvCoopMatStrideDilationTest(TestExecutionEnvironment* env) {
  return RunTest(env, {HW(3, 2), HW(2, 1), HW(1, 2), HW(1, 1), 129});
}

}  // namespace gpu
}  // namespace tflite
