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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/experimental/megakernel/program.h"

// The decode step of program.h's model in one launch of a persistent kernel
// (the megakernel), and the same step as today's per-op launches (the
// reference). Both read the same program and write the same outputs. The
// megakernel accumulates in FP32 in the same order every run, but not in the
// per-op kernels' order (README.md, Numerics): its attention and its qkv
// norms and RoPE are copies of the per-op kernels' device code
// (decode_attention.cuh, qkv_norm_rope.cuh), its projections and experts sum
// in the order its weight stream delivers them, and the steps the engine's
// graph runs as TensorRT layers (the embedding cast, the cache writes, the
// post-attention norm and residual, the final norm and the soft cap) follow
// megakernel_steps.h, rounded to BF16 where the graph holds a BF16 tensor.

// Builds the program of `model` for a grid of `ctas` CTAs (0: as many as
// the current device keeps resident). Checks the shapes the kernels take. A
// model without pointers builds the shapes alone (program.h).
extern "C" cudaError_t LiteRtNvidiaMegakernelBuildProgram(
    const LiteRtNvidiaMegakernelModel* model, int32_t ctas,
    LiteRtNvidiaMegakernelProgram* program);

// The persistent state of a program's launches (the instructions' counters
// and the step epoch): zeroed once before the first launch, kept between
// steps, and never shared by two launches in flight.
extern "C" size_t LiteRtNvidiaMegakernelStateBytes(void);

// The trace of one launch when tracing: bytes for `program`. For CTA c and
// instruction i, four %globaltimer stamps (ns) at trace[(c *
// instruction_count + i) * 4 + j]: j = 0 the CTA reached the instruction,
// 1 its waits were satisfied, 2 its prologue (the small math every CTA
// repeats and the staging of the instruction's input) was done, 3 its items
// were done; then the SM of each CTA at trace[ctas * instruction_count * 4 +
// c].
extern "C" size_t LiteRtNvidiaMegakernelTraceBytes(
    const LiteRtNvidiaMegakernelProgram* program);

// One decode step. `program` is the host copy, `device_program` the same
// bytes in device memory; `workspace` holds program->workspace_bytes. With
// a non-null `trace` (LiteRtNvidiaMegakernelTraceBytes of device memory)
// the CTAs record when each instruction started, finished waiting and
// ended. One cooperative launch: the device must keep program->ctas CTAs
// resident.
extern "C" cudaError_t LiteRtNvidiaMegakernelLaunch(
    const LiteRtNvidiaMegakernelProgram* program,
    const LiteRtNvidiaMegakernelProgram* device_program,
    const LiteRtNvidiaMegakernelStep* step, void* workspace, void* state,
    uint64_t* trace, cudaStream_t stream);

// What the reference launches need besides the stream: the MoE layer
// block's side stream for the dense feed-forward and the events that fork
// and join it (LiteRtNvidiaMoeBlockParams).
struct LiteRtNvidiaMegakernelReferenceStreams {
  cudaStream_t side_stream;
  cudaEvent_t fork;
  cudaEvent_t join;
};

// The step as today's per-op launches, as the megakernel plugin layer runs
// it (megakernel_plugin.h): the projections
// (LiteRtNvidiaLaunchBf16Int4BlockGemvRows[Normed]),
// LiteRtNvidiaLaunchQkvNormRope, LiteRtNvidiaLaunchDecodeAttention,
// LiteRtNvidiaLaunchMoeLayerBlock, and megakernel_steps.h for the
// steps the engine's graph runs. Uses the same workspace layout as the
// megakernel (in its own workspace: the two may run side by side), so their
// buffers compare.
extern "C" cudaError_t LiteRtNvidiaMegakernelLaunchReference(
    const LiteRtNvidiaMegakernelProgram* program,
    const LiteRtNvidiaMegakernelStep* step, void* workspace,
    const LiteRtNvidiaMegakernelReferenceStreams* streams, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_H_
