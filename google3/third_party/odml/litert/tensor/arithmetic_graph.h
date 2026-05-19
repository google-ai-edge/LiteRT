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
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_GRAPH_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/internal/graph.h"
#include "third_party/odml/litert/tensor/internal/mixin.h"

namespace litert::tensor {

// Possible fused activation functions.
enum FusedActivation {
  kActNone = 0,
  kActRelu,
  kActReluN1To1,  // min(max(-1, x), 1)
  kActRelu6,      // min(max(0, x), 6)
  kActTanh,
  kActSignBit,
  kActSigmoid,
};

// Possible padding types.
enum Padding {
  kPaddingSame = 0,
  kPaddingValid,
};

}  // namespace litert::tensor

namespace litert::tensor::graph {

struct BinaryOperationData {
  litert::tensor::FusedActivation activation = litert::tensor::kActNone;
};

template <class... Mixins>
struct AddOperation : BinaryOperationData,
                      virtual Operation,
                      virtual OpMixin<struct AddOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Add"; }
};

template <class... Mixins>
struct MulOperation : BinaryOperationData,
                      virtual Operation,
                      virtual OpMixin<struct MulOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Mul"; }
};

template <class... Mixins>
struct AbsOperation : virtual Operation,
                      virtual OpMixin<struct AbsOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Abs"; }
};

template <class... Mixins>
struct ReluOperation : virtual Operation,
                       virtual OpMixin<struct ReluOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Relu"; }
};

template <class... Mixins>
struct Relu6Operation : virtual Operation,
                        virtual OpMixin<struct Relu6OperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Relu6"; }
};

struct LeakyReluOperationData {
  float alpha;
};

template <class... Mixins>
struct LeakyReluOperation
    : LeakyReluOperationData,
      virtual Operation,
      virtual OpMixin<struct LeakyReluOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "LeakyRelu"; }
};

template <class... Mixins>
struct EluOperation : virtual Operation,
                      virtual OpMixin<struct EluOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Elu"; }
};

struct HardSwishOperationTag {};

template <class... Mixins>
struct HardSwishOperation
    : virtual Operation,
      virtual OpMixin<struct HardSwishOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "HardSwish"; }
};

struct PReluOperationTag {};

template <class... Mixins>
struct PReluOperation
    : virtual Operation,
      virtual OpMixin<struct PReluOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "PRelu"; }
};

struct L2NormalizationOperationTag {};

template <class... Mixins>
struct L2NormalizationOperation
    : virtual Operation,
      virtual OpMixin<struct L2NormalizationOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "L2Normalization"; }
};

template <class... Mixins>
struct SubOperation : BinaryOperationData,
                      virtual Operation,
                      virtual OpMixin<struct SubOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Sub"; }
};

template <class... Mixins>
struct DivOperation : BinaryOperationData,
                      virtual Operation,
                      virtual OpMixin<struct DivOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Div"; }
};

template <class... Mixins>
struct SquareOperation : virtual Operation,
                         virtual OpMixin<struct SquareOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Square"; }
};

template <class... Mixins>
struct RsqrtOperation : virtual Operation,
                        virtual OpMixin<struct RsqrtOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Rsqrt"; }
};

template <class... Mixins>
struct PowOperation : virtual Operation,
                      virtual OpMixin<struct PowOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Pow"; }
};

template <class... Mixins>
struct NegOperation : virtual Operation,
                      virtual OpMixin<struct NegOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Neg"; }
};

template <class... Mixins>
struct PadOperation : virtual Operation,
                      virtual OpMixin<struct PadOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Pad"; }
};

template <class... Mixins>
struct PadV2Operation : virtual Operation,
                        virtual OpMixin<struct PadV2OperationTag, Mixins>... {
  absl::string_view GetName() const override { return "PadV2"; }
};

template <class... Mixins>
struct SqrtOperation : virtual Operation,
                       virtual OpMixin<struct SqrtOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Sqrt"; }
};

template <class... Mixins>
struct ExpOperation : virtual Operation,
                      virtual OpMixin<struct ExpOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Exp"; }
};

struct LogOperationTag {};

template <class... Mixins>
struct LogOperation : virtual Operation,
                      virtual OpMixin<struct LogOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Log"; }
};

struct CeilOperationTag {};

template <class... Mixins>
struct CeilOperation : virtual Operation,
                       virtual OpMixin<struct CeilOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Ceil"; }
};

struct FloorOperationTag {};

template <class... Mixins>
struct FloorOperation : virtual Operation,
                        virtual OpMixin<struct FloorOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Floor"; }
};

template <class... Mixins>
struct FloorDivOperation
    : virtual Operation,
      virtual OpMixin<struct FloorDivOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "FloorDiv"; }
};

template <class... Mixins>
struct FloorModOperation
    : virtual Operation,
      virtual OpMixin<struct FloorModOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "FloorMod"; }
};

struct SignOperationTag {};

template <class... Mixins>
struct SignOperation : virtual Operation,
                       virtual OpMixin<struct SignOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Sign"; }
};

struct RoundOperationTag {};

template <class... Mixins>
struct RoundOperation : virtual Operation,
                        virtual OpMixin<struct RoundOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Round"; }
};

struct SoftmaxOperationData {
  float beta;
};

template <class... Mixins>
struct SoftmaxOperation
    : SoftmaxOperationData,
      virtual Operation,
      virtual OpMixin<struct SoftmaxOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Softmax"; }
};

struct LogSoftmaxOperationTag {};

template <class... Mixins>
struct LogSoftmaxOperation
    : virtual Operation,
      virtual OpMixin<struct LogSoftmaxOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "LogSoftmax"; }
};

struct SumOperationData {
  bool keep_dims;
};

template <class... Mixins>
struct SumOperation : SumOperationData,
                      virtual Operation,
                      virtual OpMixin<struct SumOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Sum"; }
};

struct ReduceMaxOperationData {
  bool keep_dims;
};

template <class... Mixins>
struct ReduceMaxOperation
    : ReduceMaxOperationData,
      virtual Operation,
      virtual OpMixin<struct ReduceMaxOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "ReduceMax"; }
};

struct MeanOperationData {
  bool keep_dims;
};

template <class... Mixins>
struct MeanOperation : MeanOperationData,
                       virtual Operation,
                       virtual OpMixin<struct MeanOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Mean"; }
};

struct BatchMatMulOperationData {
  bool adj_x;
  bool adj_y;
};

template <class... Mixins>
struct BatchMatMulOperation
    : BatchMatMulOperationData,
      virtual Operation,
      virtual OpMixin<struct BatchMatMulOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "BatchMatMul"; }
};

struct FullyConnectedOperationData {
  litert::tensor::FusedActivation activation;
  bool keep_num_dims;
};

template <class... Mixins>
struct FullyConnectedOperation
    : FullyConnectedOperationData,
      virtual Operation,
      virtual OpMixin<struct FullyConnectedOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "FullyConnected"; }
};

struct ConcatenationOperationData {
  int axis;
  litert::tensor::FusedActivation activation;
};

template <class... Mixins>
struct ConcatenationOperation
    : ConcatenationOperationData,
      virtual Operation,
      virtual OpMixin<struct ConcatenationOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Concatenation"; }
};

struct PackOperationData {
  int axis;
};

template <class... Mixins>
struct PackOperation : PackOperationData,
                       virtual Operation,
                       virtual OpMixin<struct PackOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Pack"; }
};

struct UnpackOperationData {
  int num;
  int axis;
};

template <class... Mixins>
struct UnpackOperation : UnpackOperationData,
                         virtual Operation,
                         virtual OpMixin<struct UnpackOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Unpack"; }
};

struct SpaceToDepthOperationData {
  int block_size;
};

template <class... Mixins>
struct SpaceToDepthOperation
    : SpaceToDepthOperationData,
      virtual Operation,
      virtual OpMixin<struct SpaceToDepthOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "SpaceToDepth"; }
};

struct DepthToSpaceOperationData {
  int block_size;
};

template <class... Mixins>
struct DepthToSpaceOperation
    : DepthToSpaceOperationData,
      virtual Operation,
      virtual OpMixin<struct DepthToSpaceOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "DepthToSpace"; }
};

struct SplitOperationData {
  int num_splits;
};

template <class... Mixins>
struct SplitOperation : SplitOperationData,
                        virtual Operation,
                        virtual OpMixin<struct SplitOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Split"; }
};

struct AveragePool2DOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
  int filter_height;
  int filter_width;
  litert::tensor::FusedActivation activation;
};

template <class... Mixins>
struct AveragePool2DOperation
    : AveragePool2DOperationData,
      virtual Operation,
      virtual OpMixin<struct AveragePool2DOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "AveragePool2D"; }
};

struct MaxPool2DOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
  int filter_height;
  int filter_width;
  litert::tensor::FusedActivation activation;
};

template <class... Mixins>
struct MaxPool2DOperation
    : MaxPool2DOperationData,
      virtual Operation,
      virtual OpMixin<struct MaxPool2DOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "MaxPool2D"; }
};

struct Conv2DOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
  int dilation_h_factor;
  int dilation_w_factor;
  litert::tensor::FusedActivation activation;
};

template <class... Mixins>
struct Conv2DOperation : Conv2DOperationData,
                         virtual Operation,
                         virtual OpMixin<struct Conv2DOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Conv2DOperation"; }
};

struct DepthwiseConv2DOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
  int dilation_h_factor;
  int dilation_w_factor;
  int depth_multiplier;
  litert::tensor::FusedActivation activation;
};

template <class... Mixins>
struct DepthwiseConv2DOperation
    : DepthwiseConv2DOperationData,
      virtual Operation,
      virtual OpMixin<struct DepthwiseConv2DOperationTag, Mixins>... {
  absl::string_view GetName() const override {
    return "DepthwiseConv2DOperation";
  }
};

template <class... Mixins>
struct TransposeOperation
    : virtual Operation,
      virtual OpMixin<struct TransposeOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Transpose"; }
};

template <class... Mixins>
struct TileOperation : virtual Operation,
                       virtual OpMixin<struct TileOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Tile"; }
};

struct LstmOperationTag {};

template <class... Mixins>
struct LstmOperation : virtual Operation,
                       virtual OpMixin<struct LstmOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Lstm"; }
};

struct GeluOperationData {
  bool approximate;
};

template <class... Mixins>
struct GeluOperation : GeluOperationData,
                       virtual Operation,
                       virtual OpMixin<struct GeluOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Gelu"; }
};

template <class... Mixins>
struct TanhOperation : virtual Operation,
                       virtual OpMixin<struct TanhOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Tanh"; }
};

struct CastOperationData {
  Type to;
};

template <class... Mixins>
struct CastOperation : CastOperationData,
                       virtual Operation,
                       virtual OpMixin<struct CastOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Cast"; }
};

template <class... Mixins>
struct SelectOperation : virtual Operation,
                         virtual OpMixin<struct SelectOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Select"; }
};

template <class... Mixins>
struct SelectV2Operation
    : virtual Operation,
      virtual OpMixin<struct SelectV2OperationTag, Mixins>... {
  absl::string_view GetName() const override { return "SelectV2"; }
};

template <class... Mixins>
struct SliceOperation : virtual Operation,
                        virtual OpMixin<struct SliceOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Slice"; }
};

template <class... Mixins>
struct LessOperation : virtual Operation,
                       virtual OpMixin<struct LessOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Less"; }
};

template <class... Mixins>
struct GreaterOperation
    : virtual Operation,
      virtual OpMixin<struct GreaterOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Greater"; }
};

template <class... Mixins>
struct GreaterEqualOperation
    : virtual Operation,
      virtual OpMixin<struct GreaterEqualOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "GreaterEqual"; }
};

template <class... Mixins>
struct EqualOperation : virtual Operation,
                        virtual OpMixin<struct EqualOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Equal"; }
};

template <class... Mixins>
struct NotEqualOperation
    : virtual Operation,
      virtual OpMixin<struct NotEqualOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "NotEqual"; }
};

template <class... Mixins>
struct MinimumOperation
    : virtual Operation,
      virtual OpMixin<struct MinimumOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Minimum"; }
};

template <class... Mixins>
struct MaximumOperation
    : virtual Operation,
      virtual OpMixin<struct MaximumOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Maximum"; }
};

template <class... Mixins>
struct LogicalAndOperation
    : virtual Operation,
      virtual OpMixin<struct LogicalAndOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "LogicalAnd"; }
};

template <class... Mixins>
struct LogicalOrOperation
    : virtual Operation,
      virtual OpMixin<struct LogicalOrOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "LogicalOr"; }
};

template <class... Mixins>
struct LogicalNotOperation
    : virtual Operation,
      virtual OpMixin<struct LogicalNotOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "LogicalNot"; }
};

template <class... Mixins>
struct BitwiseXorOperation
    : virtual Operation,
      virtual OpMixin<struct BitwiseXorOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "BitwiseXor"; }
};

template <class... Mixins>
struct RightShiftOperation
    : virtual Operation,
      virtual OpMixin<struct RightShiftOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "RightShift"; }
};

template <class... Mixins>
struct CosOperation : virtual Operation,
                      virtual OpMixin<struct CosOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Cos"; }
};

template <class... Mixins>
struct SinOperation : virtual Operation,
                      virtual OpMixin<struct SinOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Sin"; }
};

struct ReshapeOperationData {
  std::vector<int> new_shape;
};

template <class... Mixins>
struct ReshapeOperation
    : ReshapeOperationData,
      virtual Operation,
      virtual OpMixin<struct ReshapeOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Reshape"; }
};

struct SqueezeOperationData {
  std::vector<int> squeeze_dims;
};

template <class... Mixins>
struct SqueezeOperation
    : SqueezeOperationData,
      virtual Operation,
      virtual OpMixin<struct SqueezeOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Squeeze"; }
};

struct ExpandDimsOperationData {
  int axis;
};

template <class... Mixins>
struct ExpandDimsOperation
    : ExpandDimsOperationData,
      virtual Operation,
      virtual OpMixin<struct ExpandDimsOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "ExpandDims"; }
};

template <class... Mixins>
struct LogisticOperation
    : virtual Operation,
      virtual OpMixin<struct LogisticOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Logistic"; }
};

template <class... Mixins>
struct EmbeddingLookupOperation
    : virtual Operation,
      virtual OpMixin<struct EmbeddingLookupOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "EmbeddingLookup"; }
};

template <class... Mixins>
struct DynamicUpdateSliceOperation
    : virtual Operation,
      virtual OpMixin<struct DynamicUpdateSliceOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "DynamicUpdateSlice"; }
};

struct CustomOperationData {
  std::string custom_code;
  std::vector<uint8_t> custom_options;
};

template <class... Mixins>
struct CustomOperation : CustomOperationData,
                         virtual Operation,
                         virtual OpMixin<struct CustomOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Custom"; }
};

template <class... Mixins>
struct TopKOperation : virtual Operation,
                       virtual OpMixin<struct TopKOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "TopK"; }
};

struct ArgMaxOperationData {
  Type output_type = Type::kI64;
};

struct ArgMaxOperationTag {};

template <class... Mixins>
struct ArgMaxOperation : ArgMaxOperationData,
                         virtual Operation,
                         virtual OpMixin<struct ArgMaxOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "ArgMax"; }
};

template <class... Mixins>
struct QuantizeOperation
    : virtual Operation,
      virtual OpMixin<struct QuantizeOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Quantize"; }
};

struct CumsumOperationData {
  bool exclusive;
  bool reverse;
};

template <class... Mixins>
struct CumsumOperation : CumsumOperationData,
                         virtual Operation,
                         virtual OpMixin<struct CumsumOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Cumsum"; }
};

struct ReverseOperationTag {};

template <class... Mixins>
struct ReverseOperation
    : virtual Operation,
      virtual OpMixin<struct ReverseOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Reverse"; }
};

template <class... Mixins>
struct DequantizeOperation
    : virtual Operation,
      virtual OpMixin<struct DequantizeOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Dequantize"; }
};

struct GatherOperationData {
  int axis;
  int batch_dims;
};

template <class... Mixins>
struct GatherOperation : GatherOperationData,
                         virtual Operation,
                         virtual OpMixin<struct GatherOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Gather"; }
};

struct GatherNdOperationTag {};

template <class... Mixins>
struct GatherNdOperation
    : virtual Operation,
      virtual OpMixin<struct GatherNdOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "GatherNd"; }
};

struct OneHotOperationData {
  int axis;
};

template <class... Mixins>
struct OneHotOperation : OneHotOperationData,
                         virtual Operation,
                         virtual OpMixin<struct OneHotOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "OneHot"; }
};

template <class... Mixins>
struct ProbeOperation : virtual Operation,
                        virtual OpMixin<struct ProbeOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "Probe"; }
};

struct ResizeBilinearOperationData {
  bool align_corners;
  bool half_pixel_centers;
};

template <class... Mixins>
struct ResizeBilinearOperation
    : ResizeBilinearOperationData,
      virtual Operation,
      virtual OpMixin<struct ResizeBilinearOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "ResizeBilinear"; }
};

struct ResizeNearestNeighborOperationData {
  bool align_corners;
  bool half_pixel_centers;
};

template <class... Mixins>
struct ResizeNearestNeighborOperation
    : ResizeNearestNeighborOperationData,
      virtual Operation,
      virtual OpMixin<struct ResizeNearestNeighborOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "ResizeNearestNeighbor"; }
};

template <class... Mixins>
struct NonMaxSuppressionV5Operation
    : virtual Operation,
      virtual OpMixin<struct NonMaxSuppressionV5OperationTag, Mixins>... {
  absl::string_view GetName() const override { return "NonMaxSuppressionV5"; }
};

struct TransposeConvOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
};

template <class... Mixins>
struct TransposeConvOperation
    : TransposeConvOperationData,
      virtual Operation,
      virtual OpMixin<struct TransposeConvOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "TransposeConv"; }
};

struct TransposeConv2DOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
};

template <class... Mixins>
struct TransposeConv2DOperation
    : TransposeConv2DOperationData,
      virtual Operation,
      virtual OpMixin<struct TransposeConv2DOperationTag, Mixins>... {
  absl::string_view GetName() const override { return "TransposeConv2D"; }
};

}  // namespace litert::tensor::graph

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_GRAPH_H_
