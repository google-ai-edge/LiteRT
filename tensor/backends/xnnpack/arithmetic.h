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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_ARITHMETIC_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_ARITHMETIC_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/mixin.h"

namespace litert::tensor {

// Tag to identify the XNNPACK mixin.
struct XnnpackMixinTag {};

class ExternalBuffer;

namespace graph {

// XNNPACK mixin for the Add operation.
template <>
class OpMixin<AddOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MulOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SubOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DivOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MaximumOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MinimumOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<PowOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<AbsOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SquareOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RsqrtOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SqrtOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LogOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CeilOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FloorOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SignOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RoundOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<NegOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TanhOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LogisticOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CosOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CastOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DequantizeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ReluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<Relu6Operation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LeakyReluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<EluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<HardSwishOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<PReluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<L2NormalizationOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SinOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<GeluOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SoftmaxOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<AveragePool2DOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MaxPool2DOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<Conv2DOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthwiseConv2DOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FullyConnectedOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<BatchMatMulOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MeanOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SliceOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ConcatenationOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ReshapeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SqueezeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpandDimsOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TileOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeBilinearOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeNearestNeighborOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConvOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConv2DOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<GatherOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SpaceToDepthOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthToSpaceOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SplitOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RopeOperation, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const graph::Operation& op,
                         XnnpackBuildContext& ctx) const override;
};
}  // namespace graph

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_ARITHMETIC_H_
