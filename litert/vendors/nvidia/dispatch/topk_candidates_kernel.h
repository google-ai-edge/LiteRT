// Copyright 2026 The ODML Authors. Licensed under the Apache License,
// Version 2.0.
//
// The k largest logits of every row, on the device: the candidates that
// LiteRT-LM's CPU sampler selects first (TopKTokenIds in
// runtime/components/sampling_cpu_util.cc), in its order, so that its
// sampling policy can run on k values instead of a vocabulary.

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TOPK_CANDIDATES_KERNEL_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TOPK_CANDIDATES_KERNEL_H_

#include <driver_types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The largest k the kernels take (the candidates are sorted in shared memory).
int LiteRtNvidiaTopKCandidatesMaxK(void);

// Row flags: the row holds a NaN (its candidates are then unspecified), or
// the selection failed an internal check.
enum {
  kLiteRtNvidiaTopKCandidatesNaN = 1,
  kLiteRtNvidiaTopKCandidatesInconsistent = 2,
};

// For every row of `rows` x `count` logits: the k largest (1 <= k <= count),
// in the order logit descending then index ascending, as ids[row * k + i]
// and values[row * k + i] (the logit as a float; a float16 widened exactly).
// The k-th value's ties are resolved by lowest index; a negative zero equals
// zero. flags[row] gets the flags above, else 0. All pointers are device
// memory; one block of 1024 threads per row runs on `stream`.
cudaError_t LiteRtNvidiaLaunchF32TopKCandidateRows(
    const float* logits, size_t rows, size_t count, int k, int32_t* ids,
    float* values, int32_t* flags, cudaStream_t stream);

cudaError_t LiteRtNvidiaLaunchF16TopKCandidateRows(
    const uint16_t* logits, size_t rows, size_t count, int k, int32_t* ids,
    float* values, int32_t* flags, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TOPK_CANDIDATES_KERNEL_H_
