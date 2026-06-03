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
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_ARITHMETIC_TFLITE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_ARITHMETIC_TFLITE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/mixin.h"
#include "tensor/internal/type_id.h"
#include "tflite/schema/mutable/schema_generated.h"
#include "tflite/types/half.h"

namespace litert::tensor {

struct TfLiteMixinTag {};

template <>
struct ApiType<tflite::half> : internal::StorageImpl<Type::kFP16, fp16_t> {};

namespace graph {

struct TfLiteOpBuildInfo {
  ::tflite::BuiltinOperator builtin_code;
  std::optional<::tflite::BuiltinOptionsUnion> builtin_options = std::nullopt;

  // Present only when builtin_code is BuiltinOperator_CUSTOM.
  const std::string* custom_code = nullptr;
  const std::vector<uint8_t>* custom_options = nullptr;

  template <typename OpCodeT>
  explicit TfLiteOpBuildInfo(OpCodeT code)
      : builtin_code(static_cast<tflite::BuiltinOperator>(code)) {}

  template <typename OpCodeT, typename OpOptionsT>
  explicit TfLiteOpBuildInfo(OpCodeT code, OpOptionsT&& options)
      : builtin_code(static_cast<tflite::BuiltinOperator>(code)),
        builtin_options(::tflite::BuiltinOptionsUnion()) {
    builtin_options->Set(std::forward<OpOptionsT>(options));
  }
};

// Base class for operations that defines conversion to TfLite flatbuffer.
class TfLiteOperation : public graph::BackendExtension {
 public:
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<TfLiteOperation>();
  }
  virtual absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const = 0;
};

template <>
class OpMixin<AddOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<MulOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<AbsOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ReluOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<Relu6Operation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ReluN1To1Operation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ZerosLikeOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<Relu0To1Operation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<LeakyReluOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<EluOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<HardSwishOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<PReluOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<L2NormalizationOperation, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SubOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<DivOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SquareOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<RsqrtOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<PowOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<NegOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SqrtOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ExpOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<LogOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<CeilOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<FloorOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<FloorDivOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<FloorModOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SignOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<RoundOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<TransposeOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<LogisticOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<EmbeddingLookupOperation, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<DynamicUpdateSliceOperation, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<TileOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<GeluOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<TanhOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<CastOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SelectOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SelectV2Operation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SliceOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<LessOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<GreaterOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<GreaterEqualOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<NotEqualOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};
template <>
class OpMixin<EqualOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<MinimumOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<MaximumOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<LogicalAndOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<LogicalOrOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<LogicalNotOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<BitwiseXorOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<RightShiftOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<CosOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SinOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SoftmaxOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<LogSoftmaxOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ReshapeOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SqueezeOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ExpandDimsOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<BatchMatMulOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<FullyConnectedOperation, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ConcatenationOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<PackOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<UnpackOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SplitOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<CustomOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<AveragePool2DOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<MaxPool2DOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<Conv2DOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<DepthwiseConv2DOperation, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<PadOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<PadV2Operation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SumOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<TopKOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<QuantizeOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<DequantizeOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<CumsumOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ReverseOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<SpaceToDepthOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<DepthToSpaceOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<GatherOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<GatherNdOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<OneHotOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ReduceMaxOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<MeanOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ProbeOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ResizeBilinearOperation, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ResizeNearestNeighborOperation, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<NonMaxSuppressionV5Operation, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<TransposeConvOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<TransposeConv2DOperation, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<LstmOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

template <>
class OpMixin<ArgMaxOperation, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override;
};

}  // namespace graph
}  // namespace litert::tensor
#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_ARITHMETIC_TFLITE_H_
