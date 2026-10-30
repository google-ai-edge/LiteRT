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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ARITHMETIC_ML_DRIFT_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ARITHMETIC_ML_DRIFT_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/mixin.h"
#include "tensor/internal/type_id.h"

namespace litert::tensor {
struct MlDriftMixinTag {};

class MlDriftOperation : public graph::BackendExtension {
 public:
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<MlDriftOperation>();
  }
  virtual absl::Status ToMlDrift(const graph::Operation& op,
                                 MlDriftBuildContext& ctx) const = 0;
};

namespace graph {

template <>
class OpMixin<AddOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SubOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<DivOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<AbsOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ReluOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<Relu6Operation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<LeakyReluOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<EluOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<HardSwishOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<PReluOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<L2NormalizationOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<MulOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<RsqrtOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SqrtOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<LogOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SquareOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<NegOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<CeilOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<FloorOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<FloorDivOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<FloorModOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SignOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<RoundOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<EqualOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<NotEqualOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<GreaterEqualOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<QuantizeOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<DequantizeOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ReshapeOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SqueezeOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpandDimsOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<Conv2DOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthwiseConv2DOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<AveragePool2DOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<MaxPool2DOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<FullyConnectedOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SoftmaxOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<BatchMatMulOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<GeluOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SinOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<CosOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<LogisticOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<TanhOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<CastOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<GatherOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<OneHotOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<CumsumOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ReverseOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthToSpaceOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SumOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SelectOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SelectV2Operation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SliceOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<EmbeddingLookupOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ConcatenationOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<PackOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<UnpackOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<SplitOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<DynamicUpdateSliceOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ReduceMaxOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ArgMaxOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<MeanOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<LessOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<LogicalAndOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<LogicalNotOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<MaximumOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<PowOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ProbeOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeBilinearOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeNearestNeighborOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConvOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConv2DOperation, MlDriftMixinTag>
    : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<LstmOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

template <>
class OpMixin<TileOperation, MlDriftMixinTag> : public MlDriftOperation {
 public:
  absl::Status ToMlDrift(const graph::Operation& op,
                         MlDriftBuildContext& ctx) const override;
};

}  // namespace graph

}  // namespace litert::tensor
#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ARITHMETIC_ML_DRIFT_H_
