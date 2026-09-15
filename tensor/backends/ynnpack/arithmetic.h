/* Copyright 2026 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_ARITHMETIC_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_ARITHMETIC_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/backends/ynnpack/conversion.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/mixin.h"

namespace litert::tensor {

// Tag to identify the YNNPACK mixin.
struct YnnpackMixinTag {};

class ExternalBuffer;

namespace graph {

// YNNPACK mixin for the Add operation.
template <>
class OpMixin<AddOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MulOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SubOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DivOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MaximumOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MinimumOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<PowOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<AbsOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SquareOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RsqrtOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SqrtOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LogOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CeilOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FloorOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SignOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RoundOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<NegOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TanhOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LogisticOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CosOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CastOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DequantizeOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ReluOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<Relu6Operation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LeakyReluOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<EluOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<HardSwishOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<PReluOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<L2NormalizationOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SinOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<GeluOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SoftmaxOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<AveragePool2DOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MaxPool2DOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<Conv2DOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthwiseConv2DOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FullyConnectedOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<BatchMatMulOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MeanOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SliceOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ConcatenationOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ReshapeOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SqueezeOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpandDimsOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TileOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeBilinearOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeNearestNeighborOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConvOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConv2DOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<GatherOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SpaceToDepthOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthToSpaceOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SplitOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RopeOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};
}  // namespace graph

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_ARITHMETIC_H_
