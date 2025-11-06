/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "litert/samples/async_segmentation/timing_utils.h"

#include <iostream>

void PrintTiming(const ProfilingTimestamps& profiling_timestamps) {
  std::cout << "LoadImage took "
            << profiling_timestamps.load_image_end_time -
                   profiling_timestamps.load_image_start_time
            << std::endl;
  std::cout << "PreprocessInputForSegmentation took "
            << profiling_timestamps.pre_process_end_time -
                   profiling_timestamps.pre_process_start_time
            << std::endl;
  std::cout << "Inference took "
            << profiling_timestamps.inference_end_time -
                   profiling_timestamps.inference_start_time
            << std::endl;
  std::cout << "PostprocessSegmentation took "
            << profiling_timestamps.post_process_end_time -
                   profiling_timestamps.post_process_start_time
            << std::endl;
  std::cout << "SaveImage took "
            << profiling_timestamps.save_image_end_time -
                   profiling_timestamps.save_image_start_time
            << std::endl;
  std::cout << "E2E (pre + inference + post) took "
            << profiling_timestamps.e2e_end_time -
                   profiling_timestamps.e2e_start_time
            << std::endl;
}
