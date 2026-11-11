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

#include "litert/compiler/plugin/region_op_semantics.h"

#include <gtest/gtest.h>
#include "litert/c/litert_op_code.h"

namespace litert::internal {
namespace {

TEST(RegionOpSemanticsTest, ControlFlowOpsInvokeWholeSubgraphs) {
  for (const LiteRtOpCode code :
       {kLiteRtOpCodeTflWhile, kLiteRtOpCodeTflIf, kLiteRtOpCodeTflCallOnce,
        kLiteRtOpCodeShloWhile, kLiteRtOpCodeShloCase}) {
    EXPECT_EQ(RegionInvocationOf(code), RegionInvocation::kWholeSubgraph)
        << code;
  }
}

TEST(RegionOpSemanticsTest, ComputationRegionOpsAreAtomic) {
  for (const LiteRtOpCode code :
       {kLiteRtOpCodeShloReduce, kLiteRtOpCodeShloWindow, kLiteRtOpCodeShloSort,
        kLiteRtOpCodeShloScatter, kLiteRtOpCodeShloCustomCall}) {
    EXPECT_EQ(RegionInvocationOf(code), RegionInvocation::kAtomic) << code;
  }
}

TEST(RegionOpSemanticsTest, UnknownOpsDefaultToAtomic) {
  EXPECT_EQ(RegionInvocationOf(kLiteRtOpCodeTflMul), RegionInvocation::kAtomic);
  EXPECT_EQ(
      RegionInvocationOf(static_cast<LiteRtOpCode>(kLiteRtOpCodeMaxValue + 1)),
      RegionInvocation::kAtomic);
}

}  // namespace
}  // namespace litert::internal
