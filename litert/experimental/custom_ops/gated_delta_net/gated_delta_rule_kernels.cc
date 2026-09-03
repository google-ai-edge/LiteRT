// Copyright 2026 The LiteRT Torch Authors.
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
// ==============================================================================

#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.h"
#include "pybind11/pybind11.h"  // from @pybind11

extern "C" void TfLiteRegisterer(tflite::MutableOpResolver* resolver) {
  litert_torch::gdn_kernels::TfLiteRegisterer(resolver);
}

namespace litert_torch {
namespace gdn_kernels {

PYBIND11_MODULE(gated_delta_rule_kernels, m) {
  m.def("tflite_registerer", [](uintptr_t resolver) {
    TfLiteRegisterer(reinterpret_cast<tflite::MutableOpResolver*>(resolver));
  });
}

}  // namespace gdn_kernels
}  // namespace litert_torch
