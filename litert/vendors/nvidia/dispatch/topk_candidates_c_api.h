// Copyright 2026 The ODML Authors. Licensed under the Apache License,
// Version 2.0.
//
// Top-k candidate selection on CUDA logits through the NVIDIA dispatch
// library: the k largest logits of every row, as LiteRT-LM's CPU sampler
// (TopKTokenIds in runtime/components/sampling_cpu_util.cc) selects them,
// so that the sampler's policy (softmax, temperature, top-p, the draw, the
// scores) can run on k values on the host instead of on a vocabulary copied
// from the device. Nothing is drawn here.

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TOPK_CANDIDATES_C_API_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TOPK_CANDIDATES_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque state. A selector is bound to its first CUDA device and is not
// thread-safe.
typedef void* LiteRtDispatchNvidiaTopKCandidates;

// The largest k a selector takes (1024).
int32_t LiteRtDispatchNvidiaTopKCandidatesMaxK(void);

LiteRtStatus LiteRtDispatchNvidiaTopKCandidatesCreate(
    LiteRtDispatchNvidiaTopKCandidates* selector);

void LiteRtDispatchNvidiaTopKCandidatesDestroy(
    LiteRtDispatchNvidiaTopKCandidates selector);

// Row flags of LiteRtDispatchNvidiaTopKCandidatesSelect: the row holds a
// NaN, and its candidates are unspecified.
#define LITERT_DISPATCH_NVIDIA_TOPK_CANDIDATES_NAN 1

// Selects the k largest logits of every row of a dense [1, rows, vocab_size]
// float32 or float16 NVIDIA CUDA buffer, 1 <= k <= min(vocab_size, MaxK).
// ids and values point to rows * k writable host slots and get, for row r at
// [r * k, r * k + k), the candidates in the order logit descending then
// index ascending, ties at the k-th value resolved by lowest index, each
// value the logit as a float (a float16 widened exactly). row_flags points to
// rows writable host slots and gets 0, or the flag above. The call waits for
// the buffer's event, uploads what the host wrote to the buffer through its
// lock since the device last had it (a constraint's mask), and completes
// before it returns. Unsupported buffers (host memory, other element types,
// padded layouts) return kLiteRtStatusErrorUnsupported, so a caller can keep
// its host sampler; a selection that fails an internal check returns
// kLiteRtStatusErrorRuntimeFailure. No logits are modified.
LiteRtStatus LiteRtDispatchNvidiaTopKCandidatesSelect(
    LiteRtDispatchNvidiaTopKCandidates selector, LiteRtTensorBuffer logits,
    size_t rows, size_t vocab_size, int32_t k, int32_t* ids, float* values,
    int32_t* row_flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TOPK_CANDIDATES_C_API_H_
