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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_UTIL_H_

#include <string>
#include <vector>

#include "tflite/delegates/gpu/common/gpu_info.h"
#include "tflite/delegates/gpu/common/precision.h"
#include "tflite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tflite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

std::string MemoryTypeToCLType(MemoryType type);

std::string MemoryTypeToMetalType(MemoryType type);

// Returns float4 mask for last plane(batch of 4 channels)
// assumes that plane size is 4;
// for example we have 7 channels, in our data structures we align it to 8
// but 8s-channel will be empty, then last plane (batch of 4 channels) will
// have this mask (1, 1, 1, 0).
float4 GetMaskForLastPlane(int channels);

// task_size as amount of FLT4 processed elements.
int GetRecommendedBlockSizeForConv(const GpuInfo& gpu_info,
                                   CalculationsPrecision precision,
                                   int task_size);

int3 GetWorkGroupsCount(const int3& grid_size, const int3& work_group_size);

std::string GetTypeDeclaration(const GpuInfo& gpu_info, DataType data_type,
                               int vec_size);

std::string GetZeroValue(const GpuInfo& gpu_info, DataType data_type,
                         int vec_size);

std::string GetOneValue(const GpuInfo& gpu_info, DataType data_type,
                        int vec_size);

// Returns expression that can be substituted for converted value
// Intended to be used with absl::Substitute
// Example usage:
//   auto conversion_function = GetTypeConversion(gpu_info, UINT8, FLOAT32, 4);
//   auto code = absl::Substitute(conversion_function, "value_name");
std::string GetTypeConversion(const GpuInfo& gpu_info, DataType src_type,
                              DataType dst_type, int vec_size);

std::string GetNextWord(const std::string& code, size_t first_position);

size_t FindEnclosingBracket(const std::string& text, size_t first_pos,
                            char bracket);

absl::Status ParseArgsInsideBrackets(const std::string& text,
                                     size_t open_bracket_pos,
                                     size_t* close_bracket_pos,
                                     std::vector<std::string>* args);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_UTIL_H_
