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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SLOT_NETWORK_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SLOT_NETWORK_H_

#include <cstdint>
#include <string>
#include <vector>

#include "litert/cc/litert_expected.h"
#include "litert/vendors/nvidia/decode_slots.h"
#include "NvInfer.h"

namespace litert::nvidia {

// The batched decode engine of `slots` sequence slots, built from the network
// of the single-sequence decode step: every layer of the step is repeated
// once per slot, over that slot's own inputs and outputs (its caches, its
// recurrent states), except
//  - the constants (the weights and their holders), which every slot shares;
//  - the CUDA GEMV plugins of the blockwise INT4 projections (the dense
//    weights and the LM head), which become one plugin over the rows of all
//    the slots (slot rows: each row's bits are the ones it gets alone, the
//    weights are read once).
// A slot's inputs and outputs are named SlotTensorName(name, slot). The
// network built from the result computes, for each slot, what the
// single-sequence engine computes for that slot's bindings, bit for bit as
// long as TensorRT-RTX picks the same kernels for a repeated layer.
// `merged` counts the projections read once for all the slots.
Expected<nvinfer1::INetworkDefinition*> BuildSlotDecodeNetwork(
    nvinfer1::IBuilder& builder, nvinfer1::INetworkDefinition& step,
    int32_t slots, int32_t* merged);

// LITERT_NVIDIA_DISPATCH_SLOTS: the most sequence slots a batched decode step
// takes (0 or unset: no batched decode, the default). The compiler builds a
// batched engine of each count from 2 to it beside every decode engine.
int32_t DecodeSlotsRequested();

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SLOT_NETWORK_H_
