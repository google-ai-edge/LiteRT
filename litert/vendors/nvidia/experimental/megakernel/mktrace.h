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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MKTRACE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MKTRACE_H_

#include <cstdint>
#include <string>

#include "litert/vendors/nvidia/experimental/megakernel/program.h"

namespace litert::nvidia::megakernel {

// The trace of one megakernel launch (LiteRtNvidiaMegakernelTraceBytes,
// copied to the host) as Chrome trace JSON, for Perfetto or chrome://tracing:
// a track per CTA, named with its SM, and on it per instruction a "wait"
// slice (reached until its waits were satisfied), a "prologue" slice (the
// small math every CTA repeats, the staging of the input) and a slice of the
// instruction's own items. Times are microseconds from the first stamp.
bool WriteChromeTrace(const LiteRtNvidiaMegakernelProgram& program,
                      const uint64_t* trace, const std::string& path);

// A text summary of the same trace: the step's span, the share of the CTAs'
// time spent waiting, in prologues and in items, and per opcode its count,
// its mean wait, prologue and item time per CTA and its mean span (from the
// first CTA starting its items to the last finishing them).
std::string SummarizeTrace(const LiteRtNvidiaMegakernelProgram& program,
                           const uint64_t* trace);

// The opcode's name ("qkv", "attention", ...).
const char* OpcodeName(int32_t opcode);

}  // namespace litert::nvidia::megakernel

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MKTRACE_H_
