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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_COOP_MAT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_COOP_MAT_H_

#include <vector>

#include "tflite/delegates/gpu/common/gpu_info.h"
#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/common/shape.h"
#include "tflite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {

// Direct OpenCL convolution using 16x16 cooperative matrices. Each 8x8
// workgroup uses two 32-thread waves to process an 8x2 output pixel tile.
class ConvCoopMat : public GPUOperation {
 public:
  ConvCoopMat() = default;
  ConvCoopMat(ConvCoopMat&&) = default;
  ConvCoopMat& operator=(ConvCoopMat&&) = default;
  ConvCoopMat(const ConvCoopMat&) = delete;
  ConvCoopMat& operator=(const ConvCoopMat&) = delete;

  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }

 private:
  ConvCoopMat(const OperationDef& definition,
              const Convolution2DAttributes& attr);
  friend ConvCoopMat CreateConvCoopMat(const OperationDef& definition,
                                     const Convolution2DAttributes& attr);

  int out_g_batch_ = 1;
};

ConvCoopMat CreateConvCoopMat(const OperationDef& definition,
                            const Convolution2DAttributes& attr);

bool IsConvCoopMatSupported(const GpuInfo& gpu_info,
                           const OperationDef& definition,
                           const Convolution2DAttributes& attr,
                           const BHWC& dst_shape);

bool IsGoodTaskSizeForConvCoopMat(const Convolution2DAttributes& attr,
                               const BHWC& dst_shape);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_COOP_MAT_H_
