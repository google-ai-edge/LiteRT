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
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"

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

struct AddOperation : BinaryOperationData, Operation {
  absl::string_view GetName() const override { return "Add"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct MulOperation : BinaryOperationData, Operation {
  absl::string_view GetName() const override { return "Mul"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct AbsOperation : Operation {
  absl::string_view GetName() const override { return "Abs"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ReluOperation : Operation {
  absl::string_view GetName() const override { return "Relu"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct Relu6Operation : Operation {
  absl::string_view GetName() const override { return "Relu6"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ReluN1To1Operation : Operation {
  absl::string_view GetName() const override { return "ReluN1To1"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ZerosLikeOperation : Operation {
  absl::string_view GetName() const override { return "ZerosLike"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct Relu0To1Operation : Operation {
  absl::string_view GetName() const override { return "Relu0To1"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct LeakyReluOperationData {
  float alpha;
};

struct LeakyReluOperation : LeakyReluOperationData, Operation {
  absl::string_view GetName() const override { return "LeakyRelu"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct EluOperation : Operation {
  absl::string_view GetName() const override { return "Elu"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct HardSwishOperationTag {};

struct HardSwishOperation : Operation {
  absl::string_view GetName() const override { return "HardSwish"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct PReluOperationTag {};

struct PReluOperation : Operation {
  absl::string_view GetName() const override { return "PRelu"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct L2NormalizationOperationTag {};

struct L2NormalizationOperation : Operation {
  absl::string_view GetName() const override { return "L2Normalization"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SubOperation : BinaryOperationData, Operation {
  absl::string_view GetName() const override { return "Sub"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct DivOperation : BinaryOperationData, Operation {
  absl::string_view GetName() const override { return "Div"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SquareOperation : Operation {
  absl::string_view GetName() const override { return "Square"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct RsqrtOperation : Operation {
  absl::string_view GetName() const override { return "Rsqrt"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct PowOperation : Operation {
  absl::string_view GetName() const override { return "Pow"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct NegOperation : Operation {
  absl::string_view GetName() const override { return "Neg"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct PadOperation : Operation {
  absl::string_view GetName() const override { return "Pad"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct PadV2Operation : Operation {
  absl::string_view GetName() const override { return "PadV2"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SqrtOperation : Operation {
  absl::string_view GetName() const override { return "Sqrt"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ExpOperation : Operation {
  absl::string_view GetName() const override { return "Exp"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct LogOperationTag {};

struct LogOperation : Operation {
  absl::string_view GetName() const override { return "Log"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct CeilOperationTag {};

struct CeilOperation : Operation {
  absl::string_view GetName() const override { return "Ceil"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct FloorOperationTag {};

struct FloorOperation : Operation {
  absl::string_view GetName() const override { return "Floor"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct FloorDivOperation : Operation {
  absl::string_view GetName() const override { return "FloorDiv"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct FloorModOperation : Operation {
  absl::string_view GetName() const override { return "FloorMod"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SignOperationTag {};

struct SignOperation : Operation {
  absl::string_view GetName() const override { return "Sign"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct RoundOperationTag {};

struct RoundOperation : Operation {
  absl::string_view GetName() const override { return "Round"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SoftmaxOperationData {
  float beta;
};

struct SoftmaxOperation : SoftmaxOperationData, Operation {
  absl::string_view GetName() const override { return "Softmax"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct LogSoftmaxOperationTag {};

struct LogSoftmaxOperation : Operation {
  absl::string_view GetName() const override { return "LogSoftmax"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SumOperationData {
  bool keep_dims;
};

struct SumOperation : SumOperationData, Operation {
  absl::string_view GetName() const override { return "Sum"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ReduceMaxOperationData {
  bool keep_dims;
};

struct ReduceMaxOperation : ReduceMaxOperationData, Operation {
  absl::string_view GetName() const override { return "ReduceMax"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct MeanOperationData {
  bool keep_dims;
};

struct MeanOperation : MeanOperationData, Operation {
  absl::string_view GetName() const override { return "Mean"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct BatchMatMulOperationData {
  bool adj_x;
  bool adj_y;
};

struct BatchMatMulOperation : BatchMatMulOperationData, Operation {
  absl::string_view GetName() const override { return "BatchMatMul"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct FullyConnectedOperationData {
  litert::tensor::FusedActivation activation;
  bool keep_num_dims;
};

struct FullyConnectedOperation : FullyConnectedOperationData, Operation {
  absl::string_view GetName() const override { return "FullyConnected"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ConcatenationOperationData {
  int axis;
  litert::tensor::FusedActivation activation;
};

struct ConcatenationOperation : ConcatenationOperationData, Operation {
  absl::string_view GetName() const override { return "Concatenation"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct PackOperationData {
  int axis;
};

struct PackOperation : PackOperationData, Operation {
  absl::string_view GetName() const override { return "Pack"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct UnpackOperationData {
  int num;
  int axis;
};

struct UnpackOperation : UnpackOperationData, Operation {
  absl::string_view GetName() const override { return "Unpack"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SpaceToDepthOperationData {
  int block_size;
};

struct SpaceToDepthOperation : SpaceToDepthOperationData, Operation {
  absl::string_view GetName() const override { return "SpaceToDepth"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct DepthToSpaceOperationData {
  int block_size;
};

struct DepthToSpaceOperation : DepthToSpaceOperationData, Operation {
  absl::string_view GetName() const override { return "DepthToSpace"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SplitOperationData {
  int num_splits;
};

struct SplitOperation : SplitOperationData, Operation {
  absl::string_view GetName() const override { return "Split"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct AveragePool2DOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
  int filter_height;
  int filter_width;
  litert::tensor::FusedActivation activation;
};

struct AveragePool2DOperation : AveragePool2DOperationData, Operation {
  absl::string_view GetName() const override { return "AveragePool2D"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct MaxPool2DOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
  int filter_height;
  int filter_width;
  litert::tensor::FusedActivation activation;
};

struct MaxPool2DOperation : MaxPool2DOperationData, Operation {
  absl::string_view GetName() const override { return "MaxPool2D"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct Conv2DOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
  int dilation_h_factor;
  int dilation_w_factor;
  litert::tensor::FusedActivation activation;
};

struct Conv2DOperation : Conv2DOperationData, Operation {
  absl::string_view GetName() const override { return "Conv2DOperation"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
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

struct DepthwiseConv2DOperation : DepthwiseConv2DOperationData, Operation {
  absl::string_view GetName() const override {
    return "DepthwiseConv2DOperation";
  }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct TransposeOperation : Operation {
  absl::string_view GetName() const override { return "Transpose"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct TileOperation : Operation {
  absl::string_view GetName() const override { return "Tile"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct LstmOperationTag {};

struct LstmOperation : Operation {
  absl::string_view GetName() const override { return "Lstm"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct GeluOperationData {
  bool approximate;
};

struct GeluOperation : GeluOperationData, Operation {
  absl::string_view GetName() const override { return "Gelu"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct TanhOperation : Operation {
  absl::string_view GetName() const override { return "Tanh"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct CastOperationData {
  Type to;
};

struct CastOperation : CastOperationData, Operation {
  absl::string_view GetName() const override { return "Cast"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SelectOperation : Operation {
  absl::string_view GetName() const override { return "Select"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SelectV2Operation : Operation {
  absl::string_view GetName() const override { return "SelectV2"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SliceOperation : Operation {
  absl::string_view GetName() const override { return "Slice"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct LessOperation : Operation {
  absl::string_view GetName() const override { return "Less"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct GreaterOperation : Operation {
  absl::string_view GetName() const override { return "Greater"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct GreaterEqualOperation : Operation {
  absl::string_view GetName() const override { return "GreaterEqual"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct EqualOperation : Operation {
  absl::string_view GetName() const override { return "Equal"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct NotEqualOperation : Operation {
  absl::string_view GetName() const override { return "NotEqual"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct MinimumOperation : Operation {
  absl::string_view GetName() const override { return "Minimum"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct MaximumOperation : Operation {
  absl::string_view GetName() const override { return "Maximum"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct LogicalAndOperation : Operation {
  absl::string_view GetName() const override { return "LogicalAnd"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct LogicalOrOperation : Operation {
  absl::string_view GetName() const override { return "LogicalOr"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct LogicalNotOperation : Operation {
  absl::string_view GetName() const override { return "LogicalNot"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct BitwiseXorOperation : Operation {
  absl::string_view GetName() const override { return "BitwiseXor"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct RightShiftOperation : Operation {
  absl::string_view GetName() const override { return "RightShift"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct CosOperation : Operation {
  absl::string_view GetName() const override { return "Cos"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SinOperation : Operation {
  absl::string_view GetName() const override { return "Sin"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ReshapeOperationData {
  std::vector<int> new_shape;
};

struct ReshapeOperation : ReshapeOperationData, Operation {
  absl::string_view GetName() const override { return "Reshape"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SqueezeOperationData {
  std::vector<int> squeeze_dims;
};

struct SqueezeOperation : SqueezeOperationData, Operation {
  absl::string_view GetName() const override { return "Squeeze"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ExpandDimsOperationData {
  int axis;
};

struct ExpandDimsOperation : ExpandDimsOperationData, Operation {
  absl::string_view GetName() const override { return "ExpandDims"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct LogisticOperation : Operation {
  absl::string_view GetName() const override { return "Logistic"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct EmbeddingLookupOperation : Operation {
  absl::string_view GetName() const override { return "EmbeddingLookup"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct DynamicUpdateSliceOperation : Operation {
  absl::string_view GetName() const override { return "DynamicUpdateSlice"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct CustomOperationData {
  std::string custom_code;
  std::vector<uint8_t> custom_options;
};

struct CustomOperation : CustomOperationData, Operation {
  absl::string_view GetName() const override { return "Custom"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct TopKOperation : Operation {
  absl::string_view GetName() const override { return "TopK"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ArgMaxOperationData {
  Type output_type = Type::kI64;
};

struct ArgMaxOperationTag {};

struct ArgMaxOperation : ArgMaxOperationData, Operation {
  absl::string_view GetName() const override { return "ArgMax"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct QuantizeOperation : Operation {
  absl::string_view GetName() const override { return "Quantize"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct CumsumOperationData {
  bool exclusive;
  bool reverse;
};

struct CumsumOperation : CumsumOperationData, Operation {
  absl::string_view GetName() const override { return "Cumsum"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ReverseOperationTag {};

struct ReverseOperation : Operation {
  absl::string_view GetName() const override { return "Reverse"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct DequantizeOperation : Operation {
  absl::string_view GetName() const override { return "Dequantize"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct GatherOperationData {
  int axis;
  int batch_dims;
};

struct GatherOperation : GatherOperationData, Operation {
  absl::string_view GetName() const override { return "Gather"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct GatherNdOperationTag {};

struct GatherNdOperation : Operation {
  absl::string_view GetName() const override { return "GatherNd"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct OneHotOperationData {
  int axis;
};

struct OneHotOperation : OneHotOperationData, Operation {
  absl::string_view GetName() const override { return "OneHot"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ProbeOperation : Operation {
  absl::string_view GetName() const override { return "Probe"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ResizeBilinearOperationData {
  bool align_corners;
  bool half_pixel_centers;
};

struct ResizeBilinearOperation : ResizeBilinearOperationData, Operation {
  absl::string_view GetName() const override { return "ResizeBilinear"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ResizeNearestNeighborOperationData {
  bool align_corners;
  bool half_pixel_centers;
};

struct ResizeNearestNeighborOperation : ResizeNearestNeighborOperationData,
                                        Operation {
  absl::string_view GetName() const override { return "ResizeNearestNeighbor"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct NonMaxSuppressionV5Operation : Operation {
  absl::string_view GetName() const override { return "NonMaxSuppressionV5"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct TransposeConvOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
};

struct TransposeConvOperation : TransposeConvOperationData, Operation {
  absl::string_view GetName() const override { return "TransposeConv"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct TransposeConv2DOperationData {
  litert::tensor::Padding padding;
  int stride_h;
  int stride_w;
};

struct TransposeConv2DOperation : TransposeConv2DOperationData, Operation {
  absl::string_view GetName() const override { return "TransposeConv2D"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

}  // namespace litert::tensor::graph

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_GRAPH_H_
