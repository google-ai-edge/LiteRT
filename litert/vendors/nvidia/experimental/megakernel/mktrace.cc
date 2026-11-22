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

#include "litert/vendors/nvidia/experimental/megakernel/mktrace.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "litert/vendors/nvidia/experimental/megakernel/program.h"

namespace litert::nvidia::megakernel {
namespace {

// The stamps of CTA c at instruction i: reached, waited, prologue done, done.
const uint64_t* Stamps(const LiteRtNvidiaMegakernelProgram& p,
                       const uint64_t* trace, int c, int i) {
  return trace + (static_cast<size_t>(c) * p.instruction_count + i) * 4;
}

uint64_t FirstStamp(const LiteRtNvidiaMegakernelProgram& p,
                    const uint64_t* trace) {
  uint64_t first = std::numeric_limits<uint64_t>::max();
  for (int c = 0; c < p.ctas; ++c) {
    first = std::min(first, Stamps(p, trace, c, 0)[0]);
  }
  return first;
}

std::string Format(const char* format, double a, double b = 0, double c = 0,
                   double d = 0, double e = 0) {
  char buffer[256];
  std::snprintf(buffer, sizeof(buffer), format, a, b, c, d, e);
  return buffer;
}

}  // namespace

const char* OpcodeName(int32_t opcode) {
  switch (opcode) {
    case kLiteRtNvidiaMegakernelOpQkv:
      return "qkv";
    case kLiteRtNvidiaMegakernelOpNormRope:
      return "norm_rope";
    case kLiteRtNvidiaMegakernelOpAttention:
      return "attention";
    case kLiteRtNvidiaMegakernelOpAttentionCombine:
      return "combine";
    case kLiteRtNvidiaMegakernelOpOutput:
      return "o_proj";
    case kLiteRtNvidiaMegakernelOpRouter:
      return "router";
    case kLiteRtNvidiaMegakernelOpDenseGateUp:
      return "dense_gate_up";
    case kLiteRtNvidiaMegakernelOpExpertHidden:
      return "expert_hidden";
    case kLiteRtNvidiaMegakernelOpDenseDown:
      return "dense_down";
    case kLiteRtNvidiaMegakernelOpExpertDown:
      return "expert_down";
    case kLiteRtNvidiaMegakernelOpHead:
      return "head";
    default:
      return "unknown";
  }
}

bool WriteChromeTrace(const LiteRtNvidiaMegakernelProgram& p,
                      const uint64_t* trace, const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) return false;
  const uint64_t first = FirstStamp(p, trace);
  const uint64_t* sms =
      trace + static_cast<size_t>(p.ctas) * p.instruction_count * 4;
  auto us = [&](uint64_t t) { return static_cast<double>(t - first) / 1000.0; };
  std::fprintf(file, "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n");
  bool comma = false;
  for (int c = 0; c < p.ctas; ++c) {
    std::fprintf(
        file,
        "%s{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":%d,"
        "\"args\":{\"name\":\"CTA %d (SM %d)\"}}",
        comma ? ",\n" : "", c, c, static_cast<int>(sms[c]));
    comma = true;
    std::fprintf(file,
                 ",\n{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":0,"
                 "\"tid\":%d,\"args\":{\"sort_index\":%d}}",
                 c, static_cast<int>(sms[c]) * 4 + c / std::max(1, p.ctas / 2));
    for (int i = 0; i < p.instruction_count; ++i) {
      const uint64_t* s = Stamps(p, trace, c, i);
      const auto& ins = p.instruction[i];
      const char* name = OpcodeName(ins.opcode);
      const struct {
        const char* kind;
        uint64_t begin, end;
      } slices[3] = {{"wait", s[0], s[1]},
                     {"prologue", s[1], s[2]},
                     {nullptr, s[2], s[3]}};
      for (const auto& slice : slices) {
        if (slice.end <= slice.begin) continue;
        std::fprintf(file,
                     ",\n{\"name\":\"%s%s%s L%d\",\"cat\":\"%s\",\"ph\":\"X\","
                     "\"pid\":0,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,"
                     "\"args\":{\"instruction\":%d,\"items\":%d}}",
                     slice.kind != nullptr ? slice.kind : "",
                     slice.kind != nullptr ? " " : "", name, ins.layer,
                     slice.kind != nullptr ? slice.kind : name, c,
                     us(slice.begin), us(slice.end) - us(slice.begin), i,
                     ins.items);
      }
    }
  }
  std::fprintf(file, "\n]}\n");
  return std::fclose(file) == 0;
}

std::string SummarizeTrace(const LiteRtNvidiaMegakernelProgram& p,
                           const uint64_t* trace) {
  const uint64_t first = FirstStamp(p, trace);
  uint64_t last = 0;
  double wait = 0, prologue = 0, work = 0;
  struct Totals {
    int count = 0;
    double wait = 0, prologue = 0, work = 0, span = 0;
  };
  std::map<int32_t, Totals> by_opcode;
  for (int i = 0; i < p.instruction_count; ++i) {
    uint64_t begin = std::numeric_limits<uint64_t>::max();
    uint64_t end = 0;
    Totals& t = by_opcode[p.instruction[i].opcode];
    ++t.count;
    for (int c = 0; c < p.ctas; ++c) {
      const uint64_t* s = Stamps(p, trace, c, i);
      t.wait += static_cast<double>(s[1] - s[0]) / p.ctas;
      t.prologue += static_cast<double>(s[2] - s[1]) / p.ctas;
      t.work += static_cast<double>(s[3] - s[2]) / p.ctas;
      wait += static_cast<double>(s[1] - s[0]);
      prologue += static_cast<double>(s[2] - s[1]);
      work += static_cast<double>(s[3] - s[2]);
      begin = std::min(begin, s[2]);
      end = std::max(end, s[3]);
      last = std::max(last, s[3]);
    }
    t.span += static_cast<double>(end - begin);
  }
  const double total = wait + prologue + work;
  std::string out = Format(
      "step %.1f us on %.0f CTAs: waiting %.1f%%, prologues %.1f%%, items "
      "%.1f%% of the CTAs' time\n",
      static_cast<double>(last - first) / 1000.0, p.ctas, 100.0 * wait / total,
      100.0 * prologue / total, 100.0 * work / total);
  out +=
      "opcode          count  wait/CTA  prologue/CTA  items/CTA  span   (us, "
      "summed over the instructions)\n";
  for (const auto& [opcode, t] : by_opcode) {
    char line[256];
    std::snprintf(line, sizeof(line), "%-14s %6d %9.1f %13.1f %10.1f %7.1f\n",
                  OpcodeName(opcode), t.count, t.wait / 1000.0,
                  t.prologue / 1000.0, t.work / 1000.0, t.span / 1000.0);
    out += line;
  }
  return out;
}

}  // namespace litert::nvidia::megakernel
