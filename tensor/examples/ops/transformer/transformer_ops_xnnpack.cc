/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/examples/ops/transformer/transformer_ops_common.h"
#include "tensor/examples/ops/transformer/transformer_ops_graph.h"
#include "tensor/internal/graph.h"

namespace litert::tensor::graph {

absl::Status OpMixin<FillAttentionMaskOperation, XnnpackMixinTag>::ToXnnpack(
    const graph::Operation& op, XnnpackBuildContext& ctx) const {
  return LowerFillAttentionMaskOp(op, ctx);
}

absl::Status OpMixin<FillRopeCosSinOperation, XnnpackMixinTag>::ToXnnpack(
    const graph::Operation& op, XnnpackBuildContext& ctx) const {
  return LowerFillRopeCosSinOp(op, ctx);
}

absl::Status OpMixin<RmsNormOperation, XnnpackMixinTag>::ToXnnpack(
    const graph::Operation& op, XnnpackBuildContext& ctx) const {
  return LowerRmsNormLikeOp<XnnpackMixinTag, RmsNormOperation>(op, ctx,
                                                               "RmsNorm");
}

absl::Status OpMixin<QkNormOperation, XnnpackMixinTag>::ToXnnpack(
    const graph::Operation& op, XnnpackBuildContext& ctx) const {
  return LowerRmsNormLikeOp<XnnpackMixinTag, QkNormOperation>(op, ctx,
                                                              "QkNorm");
}

}  // namespace litert::tensor::graph
