// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SHARED_CONST_TENSOR_MAP_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SHARED_CONST_TENSOR_MAP_H_

#include <optional>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift {

// SharedTfliteTensor stores the information about the shared constant tensor.
struct SharedTfliteTensor {
  // The index of the TfLiteTensor which will be shared.
  int tflite_tensor_id;
  // The global id of the tensor in the tflite model.
  int global_id;
  // The flag indicating whether the dequantization will be forced before
  // sharing.
  bool dequant_forced = false;
  // The preferred layout of the tensor. This is used to make sure we use Linear
  // layout for shared bias tensors.
  std::optional<::ml_drift::Layout> layout;

  bool operator==(const SharedTfliteTensor& other) const {
    return this->tflite_tensor_id == other.tflite_tensor_id &&
           this->global_id == other.global_id &&
           this->dequant_forced == other.dequant_forced;
  }
};

// SharedConstTensorsMap stores the information about the GraphFloat32 runtime
// Values, which are actually the shared constant tensors, which need to be
// passed to the inference by the user. In this map, the key is the Value->id
// of the GraphFloat32 value and the value is a pair of global tensor id in the
// tflite model and the pointer to the tflite tensor itself.
using SharedConstTensorsMap =
    absl::flat_hash_map<::ml_drift::ValueId, SharedTfliteTensor>;

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_SHARED_CONST_TENSOR_MAP_H_
