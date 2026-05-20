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
#include "tensor/internal/mixin.h"
#include "tflite/schema/mutable/schema_generated.h"

namespace litert::tensor {

struct TfLiteMixinTag {};

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
class TfLiteOperation {
 public:
  virtual ~TfLiteOperation() = default;
  virtual absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const = 0;
};

template <>
class OpMixin<AddOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<MulOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<AbsOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ReluOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<Relu6OperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<LeakyReluOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<EluOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<HardSwishOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<PReluOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<L2NormalizationOperationTag, TfLiteMixinTag> :
    public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SubOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<DivOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SquareOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<RsqrtOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<PowOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<NegOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SqrtOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ExpOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<LogOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<CeilOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<FloorOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<FloorDivOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<FloorModOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SignOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<RoundOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<TransposeOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<LogisticOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<EmbeddingLookupOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<DynamicUpdateSliceOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<TileOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<GeluOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<TanhOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<CastOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SelectOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SelectV2OperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SliceOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<LessOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<GreaterOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<GreaterEqualOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<NotEqualOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};
template <>
class OpMixin<EqualOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<MinimumOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<MaximumOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<LogicalAndOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<LogicalOrOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<LogicalNotOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<BitwiseXorOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<RightShiftOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<CosOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SinOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SoftmaxOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<LogSoftmaxOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ReshapeOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SqueezeOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ExpandDimsOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<BatchMatMulOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<FullyConnectedOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ConcatenationOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<PackOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<UnpackOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SplitOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<CustomOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<AveragePool2DOperationTag, TfLiteMixinTag> :
    public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<MaxPool2DOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<Conv2DOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<DepthwiseConv2DOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<PadOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<PadV2OperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SumOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<TopKOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<QuantizeOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<DequantizeOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<CumsumOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ReverseOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<SpaceToDepthOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<DepthToSpaceOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<GatherOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<GatherNdOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<OneHotOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ReduceMaxOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<MeanOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ProbeOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ResizeBilinearOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ResizeNearestNeighborOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<NonMaxSuppressionV5OperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<TransposeConvOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<TransposeConv2DOperationTag, TfLiteMixinTag>
    : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<LstmOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

template <>
class OpMixin<ArgMaxOperationTag, TfLiteMixinTag> : public TfLiteOperation {
 public:
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;
};

}  // namespace graph
}  // namespace litert::tensor
#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TFLITE_ARITHMETIC_TFLITE_H_
