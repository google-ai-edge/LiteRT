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

#include "litert/ats/register_single_ops.h"

#include <cstddef>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register_core_ops.h"
#include "litert/ats/register_no_op.h"
#include "litert/ats/register_one_hot.h"
#include "litert/ats/register_pad.h"
#include "litert/ats/register_pooling.h"
#include "litert/ats/register_reshape.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterSingleOpsImpl(const AtsConf& options, size_t& test_id,
                           typename Fixture::Capture& cap) {
  RegisterCoreOps(options, test_id, cap);
  RegisterNoOp(options, test_id, /*iters=*/10, cap);
  RegisterPooling(options, test_id, /*iters=*/10, cap);
  RegisterOneHot(options, test_id, /*iters=*/10, cap);
  RegisterReshape(options, test_id, /*iters=*/10, cap);
  RegisterPad(options, test_id, /*iters=*/10, cap);
}

}  // namespace

void RegisterSingleOps(const AtsConf& options, size_t& test_id,
                       AtsInferenceTest::Capture& cap) {
  RegisterSingleOpsImpl<AtsInferenceTest>(options, test_id, cap);
}

void RegisterSingleOps(const AtsConf& options, size_t& test_id,
                       AtsCompileTest::Capture& cap) {
  RegisterSingleOpsImpl<AtsCompileTest>(options, test_id, cap);
}

}  // namespace litert::testing
