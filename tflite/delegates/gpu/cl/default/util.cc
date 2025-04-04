/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <utility>
#include <vector>

#include "tflite/delegates/gpu/cl/opencl_wrapper.h"

namespace tflite {
namespace gpu {
namespace cl {

absl::Status CreateQcomConvolutionFilter(cl_context context, int kernel_x,
                                         int kernel_y, cl_mem* filter,
                                         const void* data) {
  return absl::UnavailableError("CreateQcomConvolutionFilter not available.");
}

std::vector<std::string> GetUnsupportedExtensions() {
  return {"cl_qcom_accelerated_image_ops", "cl_qcom_recordable_queues"};
}

std::vector<std::pair<std::string, std::string>> GetClSpecificDefines() {
  return {};
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
