// Copyright (c) 2025 MediaTek Inc.
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

#include "litert/vendors/mediatek/compiler/legalizations/rms_norm_op_legalization.h"

#include <optional>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/load_test_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {
namespace {

TEST(RmsNormOpLegalizationTest, LegalizeFloat32RmsNorm) {
  LrtMediatekOptions* options;
  LrtCreateMediatekOptions(&options);
  auto adapter = NeuronAdapterApi::Create(std::nullopt, options);
  ASSERT_TRUE(adapter.HasValue());

  auto model_wrap = testing::LoadTestFileModel("rms_norm_composite.tflite");
  auto subgraph = model_wrap.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());

  auto compiler_ctx = LrtGetCompilerContext();
  litert::compiler::Subgraph compiler_subgraph(compiler_ctx, subgraph->Get());
  auto ops = compiler_subgraph.Ops();
  ASSERT_FALSE(ops.empty());

  auto neuron_model = (*adapter)->CreateModel();
  ASSERT_TRUE(neuron_model.HasValue());

  OperandMap operand_map(**adapter, neuron_model->get());
  auto status = LegalizeRmsNormOp(**adapter, neuron_model->get(), operand_map, ops[0]);
  EXPECT_TRUE(status.HasValue());

  LrtDestroyMediatekOptions(options);
}

}  // namespace
}  // namespace litert::mediatek
