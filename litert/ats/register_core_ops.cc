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

#include "litert/ats/register_core_ops.h"

#include <cstddef>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register_batch_matmul.h"
#include "litert/ats/register_binary_broadcast.h"
#include "litert/ats/register_binary_no_bcast.h"
#include "litert/ats/register_concatenation.h"
#include "litert/ats/register_conv_2d.h"
#include "litert/ats/register_depthwise_conv_2d.h"
#include "litert/ats/register_fully_connected.h"
#include "litert/ats/register_mean.h"
#include "litert/ats/register_reduction.h"
#include "litert/ats/register_select_v2.h"
#include "litert/ats/register_softmax.h"
#include "litert/ats/register_transpose.h"
#include "litert/ats/register_unary.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterCoreOpsImpl(const AtsConf& options, size_t& test_id,
                         typename Fixture::Capture& cap) {
  RegisterBinaryNoBroadcast(options, test_id, /*iters=*/10, cap);
  RegisterBinaryBroadcast(options, test_id, /*iters=*/10, cap);
  RegisterUnary(options, test_id, /*iters=*/10, cap);
  RegisterConv2d(options, test_id, /*iters=*/10, cap);
  RegisterDepthwiseConv2d(options, test_id, /*iters=*/10, cap);
  RegisterReduction(options, test_id, /*iters=*/10, cap);
  RegisterMean(options, test_id, /*iters=*/10, cap);
  RegisterTranspose(options, test_id, /*iters=*/10, cap);
  RegisterBatchMatmul(options, test_id, /*iters=*/10, cap);
  RegisterFullyConnected(options, test_id, /*iters=*/10, cap);
  RegisterConcatenation(options, test_id, /*iters=*/10, cap);
  RegisterSoftmax(options, test_id, /*iters=*/10, cap);
  RegisterSelectV2(options, test_id, /*iters=*/10, cap);
}

}  // namespace

void RegisterCoreOps(const AtsConf& options, size_t& test_id,
                     AtsInferenceTest::Capture& cap) {
  RegisterCoreOpsImpl<AtsInferenceTest>(options, test_id, cap);
}

void RegisterCoreOps(const AtsConf& options, size_t& test_id,
                     AtsCompileTest::Capture& cap) {
  RegisterCoreOpsImpl<AtsCompileTest>(options, test_id, cap);
}

}  // namespace litert::testing
