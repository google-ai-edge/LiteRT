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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_LOWERING_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_LOWERING_H_

#include <string>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_model.h"

namespace litert::nvidia {

// compiler/tensorrt_graph_builder.h, which depends on this library.
struct TensorRtBuildResult;
class TensorRtWeightStoreBuilder;

// Experimental: the compiler's side of the decode megakernel, off unless
// LITERT_NVIDIA_MEGAKERNEL is set (README.md). With the switch set, the
// decode signature of a Gemma 4 mixture-of-experts export is selected whole
// as one partition and compiled to one plugin layer that runs the step as a
// program (megakernel_plugin.h); every other signature and model takes the
// compiler's usual path. With it unset, none of this runs.

// Whether LITERT_NVIDIA_MEGAKERNEL asks for the megakernel (set, not empty
// and not 0).
bool MegakernelRequested();

// Whether `subgraph`, named `name`, is a decode signature the megakernel
// takes whole: named "decode", with mixture-of-experts ops. Its structure is
// validated when it is compiled (megakernel_lowering.inc), where a body the
// program does not run fails the compilation and says why.
bool IsMegakernelBody(const litert::compiler::Subgraph& subgraph,
                      const std::string& name);

// Remembers that the partition holding the whole of `subgraph` is a
// megakernel body, by the names of the subgraph's outputs; and whether
// `partition` is one that was remembered.
void RememberMegakernelBody(const litert::compiler::Subgraph& subgraph);
bool IsMegakernelPartition(const litert::compiler::Subgraph& partition);

// BuildTensorRtEngine for a megakernel body's partition
// (compiler/tensorrt_graph_builder.cc defines it, since the lowering is part
// of its builder: megakernel_lowering.inc).
Expected<TensorRtBuildResult> BuildMegakernelEngine(
    const litert::compiler::Subgraph& subgraph,
    absl::Span<const std::string> read_only_value_cache_inputs,
    TensorRtWeightStoreBuilder* weight_store,
    absl::Span<const std::string> shared_cache_inputs);

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_LOWERING_H_
