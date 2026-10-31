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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_STEP_SAMPLER_C_API_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_STEP_SAMPLER_C_API_H_

#include <stdint.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// The session's sampler tells the decode step that feeds itself (bytecode.h,
// TensorRtStepFeed) how the host samples, so that a step run ahead samples
// its logits on the device the way the host will: the k largest logits in
// the host's order (logit descending, index ascending), the host's softmax
// with the temperature, its top-p prefix and its draw. k of 1 (or no call)
// leaves the steps ahead on the argmax. Process-wide, for the one session a
// process decodes with.
LiteRtStatus LiteRtDispatchNvidiaStepSamplerSet(int32_t k, float p,
                                                float temperature);

// The uniform values in [0, 1) the host's generator will produce for its
// next samples, in order: the first for the logits of the step the host
// invokes next, and so on. The host pushes them after each sample it takes,
// from a copy of its generator; up to 8 are kept. A step ahead whose draw is
// not among them samples the argmax.
LiteRtStatus LiteRtDispatchNvidiaStepSamplerPushUniforms(const double* uniforms,
                                                         int32_t count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_STEP_SAMPLER_C_API_H_
