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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_WEIGHT_HOLDER_CALIBRATION_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_WEIGHT_HOLDER_CALIBRATION_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"

namespace litert::nvidia {

// Where TensorRT placed the weight holders of a plan, and the rest of the
// engine's weight data.
struct TensorRtWeightHolderCalibration {
  struct Run {
    uint64_t offset = 0;
    std::vector<uint8_t> data;
  };
  // nvinfer1::IWeightsManager::getSize() of an engine of the plan.
  uint64_t weight_data_size = 0;
  // Per holder, in the order given: the offset of its first byte in the
  // weight data. The segment of the holder starts at the next multiple of
  // the granule.
  std::vector<uint64_t> holder_offsets;
  // The weight data outside the segments, where it is not zero: the constants
  // the plan keeps, in whole pages. In increasing offset order.
  std::vector<Run> private_runs;
};

// Whether the TensorRT SDK this was built with has weight placeholders and a
// weights manager (TensorRT-RTX 1.7).
bool TensorRtWeightHoldersSupported();

// TensorRT lays out the weights of an engine in an order of its own, and an
// engine's weight data is only available once all its weights are loaded.
// This deserializes `plan` on the current device, refits its holders with
// zeros that start with a marker, and reads the weight data back to find the
// holders and to keep the rest. Engines deserialized from the plan later take
// the rest from here and their segments from memory the application maps, so
// they need no refit.
//
// The device holds the weight data of the engine while this runs.
Expected<TensorRtWeightHolderCalibration> CalibrateTensorRtWeightHolders(
    const void* plan, size_t plan_size,
    absl::Span<const TensorRtWeightHolderBuildData> holders, uint64_t granule);

}  // namespace litert::nvidia

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_WEIGHT_HOLDER_CALIBRATION_H_
