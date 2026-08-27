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

// This header explicitely declares some OpMixins specializations. When
// included, nothing declared/defined in this files will ever be explicitely
// used. Without including it, XNNPack graph creation will not work for these
// ops.

// IWYU pragma: always_keep

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_XNNPACK_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_XNNPACK_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/examples/ops/transformer/transformer_ops_graph.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/mixin.h"

namespace litert::tensor::graph {

template <>
class OpMixin<FillAttentionMaskOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FillRopeCosSinOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RmsNormOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

}  // namespace litert::tensor::graph

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_XNNPACK_H_
