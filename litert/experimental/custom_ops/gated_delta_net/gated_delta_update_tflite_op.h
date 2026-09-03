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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_TFLITE_OP_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_TFLITE_OP_H_

#include "tflite/c/common.h"
#include "tflite/mutable_op_resolver.h"

namespace litert_torch {
namespace gdn_kernels {

TfLiteRegistration* GetTrilInvRegistration();
TfLiteRegistration* GetGatedDeltaUpdateRegistration();

void TfLiteRegisterer(tflite::MutableOpResolver* resolver);

}  // namespace gdn_kernels
}  // namespace litert_torch

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_TFLITE_OP_H_
