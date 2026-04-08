// Copyright 2024 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_C_TYPES_PRINTING_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_C_TYPES_PRINTING_H_

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_logging.h"

/// @file
/// @brief Provides `AbslStringify` specializations for types in the LiteRT C
/// API.
///
/// This allows LiteRT C types to be seamlessly used with Abseil's string
/// formatting and logging utilities.
/// @todo Migrate the code in `tools/dump.h` to leverage the Abseil stringify
/// framework.

namespace litert {

// String representations of LiteRtElementType.
inline constexpr absl::string_view kElementTypeStrI32 = "i32";
inline constexpr absl::string_view kElementTypeStrI64 = "i64";
inline constexpr absl::string_view kElementTypeStrF32 = "f32";
inline constexpr absl::string_view kElementTypeStrF16 = "f16";
inline constexpr absl::string_view kElementTypeStrF64 = "f64";
inline constexpr absl::string_view kElementTypeStrI16 = "i16";
inline constexpr absl::string_view kElementTypeStrI8 = "i8";
inline constexpr absl::string_view kElementTypeStrU8 = "u8";
inline constexpr absl::string_view kElementTypeStrU16 = "u16";
inline constexpr absl::string_view kElementTypeStrU32 = "u32";
inline constexpr absl::string_view kElementTypeStrU64 = "u64";
inline constexpr absl::string_view kElementTypeStrI4 = "i4";
inline constexpr absl::string_view kElementTypeStrI2 = "i2";
inline constexpr absl::string_view kElementTypeStrI1 = "i1";

// String representations of LiteRtOpCode.
inline constexpr absl::string_view kOpCodeStrShloAbs = "shlo.abs";
inline constexpr absl::string_view kOpCodeStrShloAdd = "shlo.add";
inline constexpr absl::string_view kOpCodeStrShloAnd = "shlo.and";
inline constexpr absl::string_view kOpCodeStrShloBroadcastInDim =
    "shlo.broadcast_in_dim";
inline constexpr absl::string_view kOpCodeStrShloClamp = "shlo.clamp";
inline constexpr absl::string_view kOpCodeStrShloCompare = "shlo.compare";
inline constexpr absl::string_view kOpCodeStrShloComposite = "shlo.composite";
inline constexpr absl::string_view kOpCodeStrShloConcatenate =
    "shlo.concatenate";
inline constexpr absl::string_view kOpCodeStrShloConvert = "shlo.convert";
inline constexpr absl::string_view kOpCodeStrShloConvolution =
    "shlo.convolution";
inline constexpr absl::string_view kOpCodeStrShloCosine = "shlo.cosine";
inline constexpr absl::string_view kOpCodeStrShloCustomCall =
    "shlo.custom_call";
inline constexpr absl::string_view kOpCodeStrShloDivide = "shlo.divide";
inline constexpr absl::string_view kOpCodeStrShloDynamicSlice =
    "shlo.dynamic_slice";
inline constexpr absl::string_view kOpCodeStrShloDynamicUpdateSlice =
    "shlo.dynamic_update_slice";
inline constexpr absl::string_view kOpCodeStrShloExponential =
    "shlo.exponential";
inline constexpr absl::string_view kOpCodeStrShloFloor = "shlo.floor";
inline constexpr absl::string_view kOpCodeStrShloGather = "shlo.gather";
inline constexpr absl::string_view kOpCodeStrShloGeneral = "shlo.general";
inline constexpr absl::string_view kOpCodeStrShloIota = "shlo.iota";
inline constexpr absl::string_view kOpCodeStrShloLog = "shlo.log";
inline constexpr absl::string_view kOpCodeStrShloLogistic = "shlo.logistic";
inline constexpr absl::string_view kOpCodeStrShloMaximum = "shlo.maximum";
inline constexpr absl::string_view kOpCodeStrShloMinimum = "shlo.minimum";
inline constexpr absl::string_view kOpCodeStrShloMultiply = "shlo.multiply";
inline constexpr absl::string_view kOpCodeStrShloNegate = "shlo.negate";
inline constexpr absl::string_view kOpCodeStrShloOr = "shlo.or";
inline constexpr absl::string_view kOpCodeStrShloPad = "shlo.pad";
inline constexpr absl::string_view kOpCodeStrShloPower = "shlo.power";
inline constexpr absl::string_view kOpCodeStrShloReduce = "shlo.reduce";
inline constexpr absl::string_view kOpCodeStrShloRemainder = "shlo.remainder";
inline constexpr absl::string_view kOpCodeStrShloReshape = "shlo.reshape";
inline constexpr absl::string_view kOpCodeStrShloRngBitGenerator =
    "shlo.rng_bit_generator";
inline constexpr absl::string_view kOpCodeStrShloRsqrt = "shlo.rsqrt";
inline constexpr absl::string_view kOpCodeStrShloScatter = "shlo.scatter";
inline constexpr absl::string_view kOpCodeStrShloSelect = "shlo.select";
inline constexpr absl::string_view kOpCodeStrShloSlice = "shlo.slice";
inline constexpr absl::string_view kOpCodeStrShloSort = "shlo.sort";
inline constexpr absl::string_view kOpCodeStrShloSubtract = "shlo.subtract";
inline constexpr absl::string_view kOpCodeStrShloTanh = "shlo.tanh";
inline constexpr absl::string_view kOpCodeStrShloTranspose = "shlo.transpose";
inline constexpr absl::string_view kOpCodeStrShloWhile = "shlo.while";
inline constexpr absl::string_view kOpCodeStrShloWindow = "shlo.window";
inline constexpr absl::string_view kOpCodeStrTflAbs = "tfl.abs";
inline constexpr absl::string_view kOpCodeStrTflAdd = "tfl.add";
inline constexpr absl::string_view kOpCodeStrTflAddN = "tfl.add_n";
inline constexpr absl::string_view kOpCodeStrTflArgMax = "tfl.arg_max";
inline constexpr absl::string_view kOpCodeStrTflArgMin = "tfl.arg_min";
inline constexpr absl::string_view kOpCodeStrTflAssignVariable =
    "tfl.assign_variable";
inline constexpr absl::string_view kOpCodeStrTflAtan2 = "tfl.atan2";
inline constexpr absl::string_view kOpCodeStrTflAveragePool2d =
    "tfl.average_pool_2d";
inline constexpr absl::string_view kOpCodeStrTflBatchMatmul =
    "tfl.batch_matmul";
inline constexpr absl::string_view kOpCodeStrTflBatchToSpaceNd =
    "tfl.batch_to_space_nd";
inline constexpr absl::string_view kOpCodeStrTflBidirectionalSequenceLstm =
    "tfl.bidirectional_sequence_lstm";
inline constexpr absl::string_view kOpCodeStrTflBidirectionalSequenceRnn =
    "tfl.bidirectional_sequence_rnn";
inline constexpr absl::string_view kOpCodeStrTflBitcast = "tfl.bitcast";
inline constexpr absl::string_view kOpCodeStrTflBitwiseXor = "tfl.bitwise_xor";
inline constexpr absl::string_view kOpCodeStrTflBroadcastArgs =
    "tfl.broadcast_args";
inline constexpr absl::string_view kOpCodeStrTflBroadcastTo =
    "tfl.broadcast_to";
inline constexpr absl::string_view kOpCodeStrTflBucketize = "tfl.bucketize";
inline constexpr absl::string_view kOpCodeStrTflCall = "tfl.call";
inline constexpr absl::string_view kOpCodeStrTflCallOnce = "tfl.call_once";
inline constexpr absl::string_view kOpCodeStrTflCast = "tfl.cast";
inline constexpr absl::string_view kOpCodeStrTflCeil = "tfl.ceil";
inline constexpr absl::string_view kOpCodeStrTflComplexAbs = "tfl.complex_abs";
inline constexpr absl::string_view kOpCodeStrTflConcatEmbeddings =
    "tfl.concat_embeddings";
inline constexpr absl::string_view kOpCodeStrTflConcatenation =
    "tfl.concatenation";
inline constexpr absl::string_view kOpCodeStrTflConv2d = "tfl.conv_2d";
inline constexpr absl::string_view kOpCodeStrTflConv3d = "tfl.conv_3d";
inline constexpr absl::string_view kOpCodeStrTflConv3dTranspose =
    "tfl.conv_3d_transpose";
inline constexpr absl::string_view kOpCodeStrTflCos = "tfl.cos";
inline constexpr absl::string_view kOpCodeStrTflCumsum = "tfl.cumsum";
inline constexpr absl::string_view kOpCodeStrTflCustom = "tfl.custom_op";
inline constexpr absl::string_view kOpCodeStrTflDelegate = "tfl.delegate";
inline constexpr absl::string_view kOpCodeStrTflDensify = "tfl.densify";
inline constexpr absl::string_view kOpCodeStrTflDepthToSpace =
    "tfl.depth_to_space";
inline constexpr absl::string_view kOpCodeStrTflDepthwiseConv2d =
    "tfl.depthwise_conv_2d";
inline constexpr absl::string_view kOpCodeStrTflDequantize = "tfl.dequantize";
inline constexpr absl::string_view kOpCodeStrTflDilate = "tfl.dilate";
inline constexpr absl::string_view kOpCodeStrTflDiv = "tfl.div";
inline constexpr absl::string_view kOpCodeStrTflDynamicUpdateSlice =
    "tfl.dynamic_update_slice";
inline constexpr absl::string_view kOpCodeStrTflElu = "tfl.elu";
inline constexpr absl::string_view kOpCodeStrTflEmbeddingLookup =
    "tfl.embedding_lookup";
inline constexpr absl::string_view kOpCodeStrTflEmbeddingLookupSparse =
    "tfl.embedding_lookup_sparse";
inline constexpr absl::string_view kOpCodeStrTflEqual = "tfl.equal";
inline constexpr absl::string_view kOpCodeStrTflExp = "tfl.exp";
inline constexpr absl::string_view kOpCodeStrTflExpandDims = "tfl.expand_dims";
inline constexpr absl::string_view kOpCodeStrTflFakeQuant = "tfl.fake_quant";
inline constexpr absl::string_view kOpCodeStrTflFill = "tfl.fill";
inline constexpr absl::string_view kOpCodeStrTflFloor = "tfl.floor";
inline constexpr absl::string_view kOpCodeStrTflFloorDiv = "tfl.floor_div";
inline constexpr absl::string_view kOpCodeStrTflFloorMod = "tfl.floor_mod";
inline constexpr absl::string_view kOpCodeStrTflFullyConnected =
    "tfl.fully_connected";
inline constexpr absl::string_view kOpCodeStrTflGather = "tfl.gather";
inline constexpr absl::string_view kOpCodeStrTflGatherNd = "tfl.gather_nd";
inline constexpr absl::string_view kOpCodeStrTflGelu = "tfl.gelu";
inline constexpr absl::string_view kOpCodeStrTflGreater = "tfl.greater";
inline constexpr absl::string_view kOpCodeStrTflGreaterEqual =
    "tfl.greater_equal";
inline constexpr absl::string_view kOpCodeStrTflHardSwish = "tfl.hard_swish";
inline constexpr absl::string_view kOpCodeStrTflHashtable = "tfl.hashtable";
inline constexpr absl::string_view kOpCodeStrTflHashtableFind =
    "tfl.hashtable_find";
inline constexpr absl::string_view kOpCodeStrTflHashtableImport =
    "tfl.hashtable_import";
inline constexpr absl::string_view kOpCodeStrTflHashtableLookup =
    "tfl.hashtable_lookup";
inline constexpr absl::string_view kOpCodeStrTflHashtableSize =
    "tfl.hashtable_size";
inline constexpr absl::string_view kOpCodeStrTflIf = "tfl.if";
inline constexpr absl::string_view kOpCodeStrTflImag = "tfl.imag";
inline constexpr absl::string_view kOpCodeStrTflL2Normalization =
    "tfl.l2_normalization";
inline constexpr absl::string_view kOpCodeStrTflL2Pool2d = "tfl.l2_pool_2d";
inline constexpr absl::string_view kOpCodeStrTflLeakyRelu = "tfl.leaky_relu";
inline constexpr absl::string_view kOpCodeStrTflLess = "tfl.less";
inline constexpr absl::string_view kOpCodeStrTflLessEqual = "tfl.less_equal";
inline constexpr absl::string_view kOpCodeStrTflLocalResponseNormalization =
    "tfl.local_response_normalization";
inline constexpr absl::string_view kOpCodeStrTflLog = "tfl.log";
inline constexpr absl::string_view kOpCodeStrTflLogSoftmax = "tfl.log_softmax";
inline constexpr absl::string_view kOpCodeStrTflLogicalAnd = "tfl.logical_and";
inline constexpr absl::string_view kOpCodeStrTflLogicalNot = "tfl.logical_not";
inline constexpr absl::string_view kOpCodeStrTflLogicalOr = "tfl.logical_or";
inline constexpr absl::string_view kOpCodeStrTflLogistic = "tfl.logistic";
inline constexpr absl::string_view kOpCodeStrTflLshProjection =
    "tfl.lsh_projection";
inline constexpr absl::string_view kOpCodeStrTflLstm = "tfl.lstm";
inline constexpr absl::string_view kOpCodeStrTflMatrixDiag = "tfl.matrix_diag";
inline constexpr absl::string_view kOpCodeStrTflMatrixSetDiag =
    "tfl.matrix_set_diag";
inline constexpr absl::string_view kOpCodeStrTflMaxPool2d = "tfl.max_pool_2d";
inline constexpr absl::string_view kOpCodeStrTflMaximum = "tfl.maximum";
inline constexpr absl::string_view kOpCodeStrTflMean = "tfl.mean";
inline constexpr absl::string_view kOpCodeStrTflMinimum = "tfl.minimum";
inline constexpr absl::string_view kOpCodeStrTflMirrorPad = "tfl.mirror_pad";
inline constexpr absl::string_view kOpCodeStrTflMul = "tfl.mul";
inline constexpr absl::string_view kOpCodeStrTflMultinomial = "tfl.multinomial";
inline constexpr absl::string_view kOpCodeStrTflNeg = "tfl.neg";
inline constexpr absl::string_view kOpCodeStrTflNonMaxSuppressionV4 =
    "tfl.non_max_suppression_v4";
inline constexpr absl::string_view kOpCodeStrTflNonMaxSuppressionV5 =
    "tfl.non_max_suppression_v5";
inline constexpr absl::string_view kOpCodeStrTflNotEqual = "tfl.not_equal";
inline constexpr absl::string_view kOpCodeStrTflOneHot = "tfl.one_hot";
inline constexpr absl::string_view kOpCodeStrTflPack = "tfl.pack";
inline constexpr absl::string_view kOpCodeStrTflPad = "tfl.pad";
inline constexpr absl::string_view kOpCodeStrTflPadv2 = "tfl.pad_v2";
inline constexpr absl::string_view
    kOpCodeStrTflPlaceholderForGreaterOpCodeTfls =
        "tfl.placeholder_for_greater_op_codes";
inline constexpr absl::string_view kOpCodeStrTflPow = "tfl.pow";
inline constexpr absl::string_view kOpCodeStrTflPrelu = "tfl.prelu";
inline constexpr absl::string_view kOpCodeStrTflQuantize = "tfl.quantize";
inline constexpr absl::string_view kOpCodeStrTflRandomStandardNormal =
    "tfl.random_standard_normal";
inline constexpr absl::string_view kOpCodeStrTflRandomUniform =
    "tfl.random_uniform";
inline constexpr absl::string_view kOpCodeStrTflRange = "tfl.range";
inline constexpr absl::string_view kOpCodeStrTflRank = "tfl.rank";
inline constexpr absl::string_view kOpCodeStrTflReadVariable =
    "tfl.read_variable";
inline constexpr absl::string_view kOpCodeStrTflReal = "tfl.real";
inline constexpr absl::string_view kOpCodeStrTflReduceAll = "tfl.reduce_all";
inline constexpr absl::string_view kOpCodeStrTflReduceAny = "tfl.reduce_any";
inline constexpr absl::string_view kOpCodeStrTflReduceMax = "tfl.reduce_max";
inline constexpr absl::string_view kOpCodeStrTflReduceMin = "tfl.reduce_min";
inline constexpr absl::string_view kOpCodeStrTflReduceProd = "tfl.reduce_prod";
inline constexpr absl::string_view kOpCodeStrTflReduceWindow =
    "tfl.reduce_window";
inline constexpr absl::string_view kOpCodeStrTflRelu = "tfl.relu";
inline constexpr absl::string_view kOpCodeStrTflRelu0To1 = "tfl.relu_0_to_1";
inline constexpr absl::string_view kOpCodeStrTflRelu6 = "tfl.relu6";
inline constexpr absl::string_view kOpCodeStrTflReluN1To1 = "tfl.relu_n1_to_1";
inline constexpr absl::string_view kOpCodeStrTflReshape = "tfl.reshape";
inline constexpr absl::string_view kOpCodeStrTflResizeBilinear =
    "tfl.resize_bilinear";
inline constexpr absl::string_view kOpCodeStrTflResizeNearestNeighbor =
    "tfl.resize_nearest_neighbor";
inline constexpr absl::string_view kOpCodeStrTflReverseSequence =
    "tfl.reverse_sequence";
inline constexpr absl::string_view kOpCodeStrTflReverseV2 = "tfl.reverse_v2";
inline constexpr absl::string_view kOpCodeStrTflRfft2d = "tfl.rfft_2d";
inline constexpr absl::string_view kOpCodeStrTflRightShift = "tfl.right_shift";
inline constexpr absl::string_view kOpCodeStrTflRnn = "tfl.rnn";
inline constexpr absl::string_view kOpCodeStrTflRound = "tfl.round";
inline constexpr absl::string_view kOpCodeStrTflRsqrt = "tfl.rsqrt";
inline constexpr absl::string_view kOpCodeStrTflScatterNd = "tfl.scatter_nd";
inline constexpr absl::string_view kOpCodeStrTflSegmentSum = "tfl.segment_sum";
inline constexpr absl::string_view kOpCodeStrTflSelect = "tfl.select";
inline constexpr absl::string_view kOpCodeStrTflSelectV2 = "tfl.select_v2";
inline constexpr absl::string_view kOpCodeStrTflShape = "tfl.shape";
inline constexpr absl::string_view kOpCodeStrTflSign = "tfl.sign";
inline constexpr absl::string_view kOpCodeStrTflSin = "tfl.sin";
inline constexpr absl::string_view kOpCodeStrTflSkipGram = "tfl.skip_gram";
inline constexpr absl::string_view kOpCodeStrTflSlice = "tfl.slice";
inline constexpr absl::string_view kOpCodeStrTflSoftmax = "tfl.softmax";
inline constexpr absl::string_view kOpCodeStrTflSpaceToBatchNd =
    "tfl.space_to_batch_nd";
inline constexpr absl::string_view kOpCodeStrTflSpaceToDepth =
    "tfl.space_to_depth";
inline constexpr absl::string_view kOpCodeStrTflSparseToDense =
    "tfl.sparse_to_dense";
inline constexpr absl::string_view kOpCodeStrTflSplit = "tfl.split";
inline constexpr absl::string_view kOpCodeStrTflSplitV = "tfl.split_v";
inline constexpr absl::string_view kOpCodeStrTflSqrt = "tfl.sqrt";
inline constexpr absl::string_view kOpCodeStrTflSquare = "tfl.square";
inline constexpr absl::string_view kOpCodeStrTflSquaredDifference =
    "tfl.squared_difference";
inline constexpr absl::string_view kOpCodeStrTflSqueeze = "tfl.squeeze";
inline constexpr absl::string_view kOpCodeStrTflStridedSlice =
    "tfl.strided_slice";
inline constexpr absl::string_view kOpCodeStrTflSub = "tfl.sub";
inline constexpr absl::string_view kOpCodeStrTflSum = "tfl.sum";
inline constexpr absl::string_view kOpCodeStrTflSvdf = "tfl.svdf";
inline constexpr absl::string_view kOpCodeStrTflTanh = "tfl.tanh";
inline constexpr absl::string_view kOpCodeStrTflTile = "tfl.tile";
inline constexpr absl::string_view kOpCodeStrTflTopkV2 = "tfl.topk_v2";
inline constexpr absl::string_view kOpCodeStrTflTranspose = "tfl.transpose";
inline constexpr absl::string_view kOpCodeStrTflTransposeConv =
    "tfl.transpose_conv";
inline constexpr absl::string_view kOpCodeStrTflUnidirectionalSequenceLstm =
    "tfl.unidirectional_sequence_lstm";
inline constexpr absl::string_view kOpCodeStrTflUnidirectionalSequenceRnn =
    "tfl.unidirectional_sequence_rnn";
inline constexpr absl::string_view kOpCodeStrTflUnique = "tfl.unique";
inline constexpr absl::string_view kOpCodeStrTflUnpack = "tfl.unpack";
inline constexpr absl::string_view kOpCodeStrTflUnsortedSegmentMax =
    "tfl.unsorted_segment_max";
inline constexpr absl::string_view kOpCodeStrTflUnsortedSegmentMin =
    "tfl.unsorted_segment_min";
inline constexpr absl::string_view kOpCodeStrTflUnsortedSegmentProd =
    "tfl.unsorted_segment_prod";
inline constexpr absl::string_view kOpCodeStrTflUnsortedSegmentSum =
    "tfl.unsorted_segment_sum";
inline constexpr absl::string_view kOpCodeStrTflVarHandle = "tfl.var_handle";
inline constexpr absl::string_view kOpCodeStrTflWhere = "tfl.where";
inline constexpr absl::string_view kOpCodeStrTflWhile = "tfl.while";
inline constexpr absl::string_view kOpCodeStrTflZerosLike = "tfl.zeros_like";

}  // namespace litert

/// @brief `AbslStringify` specialization for `LiteRtElementType`.
template <class Sink>
void AbslStringify(Sink& sink, const LiteRtElementType& type) {
  absl::string_view dtype_str;
  switch (type) {
    case kLiteRtElementTypeInt32:
      dtype_str = ::litert::kElementTypeStrI32;
      break;
    case kLiteRtElementTypeInt64:
      dtype_str = ::litert::kElementTypeStrI64;
      break;
    case kLiteRtElementTypeFloat32:
      dtype_str = ::litert::kElementTypeStrF32;
      break;
    case kLiteRtElementTypeFloat16:
      dtype_str = ::litert::kElementTypeStrF16;
      break;
    case kLiteRtElementTypeFloat64:
      dtype_str = ::litert::kElementTypeStrF64;
      break;
    case kLiteRtElementTypeInt16:
      dtype_str = ::litert::kElementTypeStrI16;
      break;
    case kLiteRtElementTypeInt8:
      dtype_str = ::litert::kElementTypeStrI8;
      break;
    case kLiteRtElementTypeUInt8:
      dtype_str = ::litert::kElementTypeStrU8;
      break;
    case kLiteRtElementTypeUInt16:
      dtype_str = ::litert::kElementTypeStrU16;
      break;
    case kLiteRtElementTypeUInt32:
      dtype_str = ::litert::kElementTypeStrU32;
      break;
    case kLiteRtElementTypeUInt64:
      dtype_str = ::litert::kElementTypeStrU64;
      break;
    case kLiteRtElementTypeInt4:
      dtype_str = ::litert::kElementTypeStrI4;
      break;
    case kLiteRtElementTypeInt2:
      dtype_str = ::litert::kElementTypeStrI2;
      break;
    case kLiteRtElementTypeBool:
      dtype_str = ::litert::kElementTypeStrI1;
      break;
    default:
      dtype_str = ::litert::kNoPrinterTag;
      break;
  }

  absl::Format(&sink, "%s", dtype_str);
}

/// @brief `AbslStringify` specialization for `LiteRtLayout`.
template <class Sink>
void AbslStringify(Sink& sink, const LiteRtLayout& layout) {
  absl::Format(
      &sink, "<%s>",
      absl::StrJoin(absl::MakeConstSpan(layout.dimensions, layout.rank), "x"));
}

/// @brief `AbslStringify` specialization for `LiteRtRankedTensorType`.
template <class Sink>
void AbslStringify(Sink& sink, const LiteRtRankedTensorType& type) {
  const auto& layout = type.layout;
  absl::Format(&sink, "%ud_%v%v", layout.rank, type.element_type, layout);
}

namespace litert {
// Helper function to get the string representation of LiteRtOpCode.
inline absl::string_view GetOpCodeStringView(const LiteRtOpCode& code) {
  switch (code) {
    case kLiteRtOpCodeShloAbs:
      return kOpCodeStrShloAbs;
    case kLiteRtOpCodeShloAdd:
      return kOpCodeStrShloAdd;
    case kLiteRtOpCodeShloAnd:
      return kOpCodeStrShloAnd;
    case kLiteRtOpCodeShloBroadcastInDim:
      return kOpCodeStrShloBroadcastInDim;
    case kLiteRtOpCodeShloClamp:
      return kOpCodeStrShloClamp;
    case kLiteRtOpCodeShloCompare:
      return kOpCodeStrShloCompare;
    case kLiteRtOpCodeShloComposite:
      return kOpCodeStrShloComposite;
    case kLiteRtOpCodeShloConcatenate:
      return kOpCodeStrShloConcatenate;
    case kLiteRtOpCodeShloConvert:
      return kOpCodeStrShloConvert;
    case kLiteRtOpCodeShloConvolution:
      return kOpCodeStrShloConvolution;
    case kLiteRtOpCodeShloCosine:
      return kOpCodeStrShloCosine;
    case kLiteRtOpCodeShloCustomCall:
      return kOpCodeStrShloCustomCall;
    case kLiteRtOpCodeShloDivide:
      return kOpCodeStrShloDivide;
    case kLiteRtOpCodeShloDynamicSlice:
      return kOpCodeStrShloDynamicSlice;
    case kLiteRtOpCodeShloDynamicUpdateSlice:
      return kOpCodeStrShloDynamicUpdateSlice;
    case kLiteRtOpCodeShloExponential:
      return kOpCodeStrShloExponential;
    case kLiteRtOpCodeShloFloor:
      return kOpCodeStrShloFloor;
    case kLiteRtOpCodeShloGather:
      return kOpCodeStrShloGather;
    case kLiteRtOpCodeShloGeneral:
      return kOpCodeStrShloGeneral;
    case kLiteRtOpCodeShloIota:
      return kOpCodeStrShloIota;
    case kLiteRtOpCodeShloLog:
      return kOpCodeStrShloLog;
    case kLiteRtOpCodeShloLogistic:
      return kOpCodeStrShloLogistic;
    case kLiteRtOpCodeShloMaximum:
      return kOpCodeStrShloMaximum;
    case kLiteRtOpCodeShloMinimum:
      return kOpCodeStrShloMinimum;
    case kLiteRtOpCodeShloMultiply:
      return kOpCodeStrShloMultiply;
    case kLiteRtOpCodeShloNegate:
      return kOpCodeStrShloNegate;
    case kLiteRtOpCodeShloOr:
      return kOpCodeStrShloOr;
    case kLiteRtOpCodeShloPad:
      return kOpCodeStrShloPad;
    case kLiteRtOpCodeShloPower:
      return kOpCodeStrShloPower;
    case kLiteRtOpCodeShloReduce:
      return kOpCodeStrShloReduce;
    case kLiteRtOpCodeShloRemainder:
      return kOpCodeStrShloRemainder;
    case kLiteRtOpCodeShloReshape:
      return kOpCodeStrShloReshape;
    case kLiteRtOpCodeShloRngBitGenerator:
      return kOpCodeStrShloRngBitGenerator;
    case kLiteRtOpCodeShloRsqrt:
      return kOpCodeStrShloRsqrt;
    case kLiteRtOpCodeShloScatter:
      return kOpCodeStrShloScatter;
    case kLiteRtOpCodeShloSelect:
      return kOpCodeStrShloSelect;
    case kLiteRtOpCodeShloSlice:
      return kOpCodeStrShloSlice;
    case kLiteRtOpCodeShloSort:
      return kOpCodeStrShloSort;
    case kLiteRtOpCodeShloSubtract:
      return kOpCodeStrShloSubtract;
    case kLiteRtOpCodeShloTanh:
      return kOpCodeStrShloTanh;
    case kLiteRtOpCodeShloTranspose:
      return kOpCodeStrShloTranspose;
    case kLiteRtOpCodeShloWhile:
      return kOpCodeStrShloWhile;
    case kLiteRtOpCodeShloWindow:
      return kOpCodeStrShloWindow;
    case kLiteRtOpCodeTflAbs:
      return kOpCodeStrTflAbs;
    case kLiteRtOpCodeTflAdd:
      return kOpCodeStrTflAdd;
    case kLiteRtOpCodeTflAddN:
      return kOpCodeStrTflAddN;
    case kLiteRtOpCodeTflArgMax:
      return kOpCodeStrTflArgMax;
    case kLiteRtOpCodeTflArgMin:
      return kOpCodeStrTflArgMin;
    case kLiteRtOpCodeTflAssignVariable:
      return kOpCodeStrTflAssignVariable;
    case kLiteRtOpCodeTflAtan2:
      return kOpCodeStrTflAtan2;
    case kLiteRtOpCodeTflAveragePool2d:
      return kOpCodeStrTflAveragePool2d;
    case kLiteRtOpCodeTflBatchMatmul:
      return kOpCodeStrTflBatchMatmul;
    case kLiteRtOpCodeTflBatchToSpaceNd:
      return kOpCodeStrTflBatchToSpaceNd;
    case kLiteRtOpCodeTflBidirectionalSequenceLstm:
      return kOpCodeStrTflBidirectionalSequenceLstm;
    case kLiteRtOpCodeTflBidirectionalSequenceRnn:
      return kOpCodeStrTflBidirectionalSequenceRnn;
    case kLiteRtOpCodeTflBitcast:
      return kOpCodeStrTflBitcast;
    case kLiteRtOpCodeTflBitwiseXor:
      return kOpCodeStrTflBitwiseXor;
    case kLiteRtOpCodeTflBroadcastArgs:
      return kOpCodeStrTflBroadcastArgs;
    case kLiteRtOpCodeTflBroadcastTo:
      return kOpCodeStrTflBroadcastTo;
    case kLiteRtOpCodeTflBucketize:
      return kOpCodeStrTflBucketize;
    case kLiteRtOpCodeTflCall:
      return kOpCodeStrTflCall;
    case kLiteRtOpCodeTflCallOnce:
      return kOpCodeStrTflCallOnce;
    case kLiteRtOpCodeTflCast:
      return kOpCodeStrTflCast;
    case kLiteRtOpCodeTflCeil:
      return kOpCodeStrTflCeil;
    case kLiteRtOpCodeTflComplexAbs:
      return kOpCodeStrTflComplexAbs;
    case kLiteRtOpCodeTflConcatEmbeddings:
      return kOpCodeStrTflConcatEmbeddings;
    case kLiteRtOpCodeTflConcatenation:
      return kOpCodeStrTflConcatenation;
    case kLiteRtOpCodeTflConv2d:
      return kOpCodeStrTflConv2d;
    case kLiteRtOpCodeTflConv3d:
      return kOpCodeStrTflConv3d;
    case kLiteRtOpCodeTflConv3dTranspose:
      return kOpCodeStrTflConv3dTranspose;
    case kLiteRtOpCodeTflCos:
      return kOpCodeStrTflCos;
    case kLiteRtOpCodeTflCumsum:
      return kOpCodeStrTflCumsum;
    case kLiteRtOpCodeTflCustom:
      return kOpCodeStrTflCustom;
    case kLiteRtOpCodeTflDelegate:
      return kOpCodeStrTflDelegate;
    case kLiteRtOpCodeTflDensify:
      return kOpCodeStrTflDensify;
    case kLiteRtOpCodeTflDepthToSpace:
      return kOpCodeStrTflDepthToSpace;
    case kLiteRtOpCodeTflDepthwiseConv2d:
      return kOpCodeStrTflDepthwiseConv2d;
    case kLiteRtOpCodeTflDequantize:
      return kOpCodeStrTflDequantize;
    case kLiteRtOpCodeTflDilate:
      return kOpCodeStrTflDilate;
    case kLiteRtOpCodeTflDiv:
      return kOpCodeStrTflDiv;
    case kLiteRtOpCodeTflDynamicUpdateSlice:
      return kOpCodeStrTflDynamicUpdateSlice;
    case kLiteRtOpCodeTflElu:
      return kOpCodeStrTflElu;
    case kLiteRtOpCodeTflEmbeddingLookup:
      return kOpCodeStrTflEmbeddingLookup;
    case kLiteRtOpCodeTflEmbeddingLookupSparse:
      return kOpCodeStrTflEmbeddingLookupSparse;
    case kLiteRtOpCodeTflEqual:
      return kOpCodeStrTflEqual;
    case kLiteRtOpCodeTflExp:
      return kOpCodeStrTflExp;
    case kLiteRtOpCodeTflExpandDims:
      return kOpCodeStrTflExpandDims;
    case kLiteRtOpCodeTflFakeQuant:
      return kOpCodeStrTflFakeQuant;
    case kLiteRtOpCodeTflFill:
      return kOpCodeStrTflFill;
    case kLiteRtOpCodeTflFloor:
      return kOpCodeStrTflFloor;
    case kLiteRtOpCodeTflFloorDiv:
      return kOpCodeStrTflFloorDiv;
    case kLiteRtOpCodeTflFloorMod:
      return kOpCodeStrTflFloorMod;
    case kLiteRtOpCodeTflFullyConnected:
      return kOpCodeStrTflFullyConnected;
    case kLiteRtOpCodeTflGather:
      return kOpCodeStrTflGather;
    case kLiteRtOpCodeTflGatherNd:
      return kOpCodeStrTflGatherNd;
    case kLiteRtOpCodeTflGelu:
      return kOpCodeStrTflGelu;
    case kLiteRtOpCodeTflGreater:
      return kOpCodeStrTflGreater;
    case kLiteRtOpCodeTflGreaterEqual:
      return kOpCodeStrTflGreaterEqual;
    case kLiteRtOpCodeTflHardSwish:
      return kOpCodeStrTflHardSwish;
    case kLiteRtOpCodeTflHashtable:
      return kOpCodeStrTflHashtable;
    case kLiteRtOpCodeTflHashtableFind:
      return kOpCodeStrTflHashtableFind;
    case kLiteRtOpCodeTflHashtableImport:
      return kOpCodeStrTflHashtableImport;
    case kLiteRtOpCodeTflHashtableLookup:
      return kOpCodeStrTflHashtableLookup;
    case kLiteRtOpCodeTflHashtableSize:
      return kOpCodeStrTflHashtableSize;
    case kLiteRtOpCodeTflIf:
      return kOpCodeStrTflIf;
    case kLiteRtOpCodeTflImag:
      return kOpCodeStrTflImag;
    case kLiteRtOpCodeTflL2Normalization:
      return kOpCodeStrTflL2Normalization;
    case kLiteRtOpCodeTflL2Pool2d:
      return kOpCodeStrTflL2Pool2d;
    case kLiteRtOpCodeTflLeakyRelu:
      return kOpCodeStrTflLeakyRelu;
    case kLiteRtOpCodeTflLess:
      return kOpCodeStrTflLess;
    case kLiteRtOpCodeTflLessEqual:
      return kOpCodeStrTflLessEqual;
    case kLiteRtOpCodeTflLocalResponseNormalization:
      return kOpCodeStrTflLocalResponseNormalization;
    case kLiteRtOpCodeTflLog:
      return kOpCodeStrTflLog;
    case kLiteRtOpCodeTflLogSoftmax:
      return kOpCodeStrTflLogSoftmax;
    case kLiteRtOpCodeTflLogicalAnd:
      return kOpCodeStrTflLogicalAnd;
    case kLiteRtOpCodeTflLogicalNot:
      return kOpCodeStrTflLogicalNot;
    case kLiteRtOpCodeTflLogicalOr:
      return kOpCodeStrTflLogicalOr;
    case kLiteRtOpCodeTflLogistic:
      return kOpCodeStrTflLogistic;
    case kLiteRtOpCodeTflLshProjection:
      return kOpCodeStrTflLshProjection;
    case kLiteRtOpCodeTflLstm:
      return kOpCodeStrTflLstm;
    case kLiteRtOpCodeTflMatrixDiag:
      return kOpCodeStrTflMatrixDiag;
    case kLiteRtOpCodeTflMatrixSetDiag:
      return kOpCodeStrTflMatrixSetDiag;
    case kLiteRtOpCodeTflMaxPool2d:
      return kOpCodeStrTflMaxPool2d;
    case kLiteRtOpCodeTflMaximum:
      return kOpCodeStrTflMaximum;
    case kLiteRtOpCodeTflMean:
      return kOpCodeStrTflMean;
    case kLiteRtOpCodeTflMinimum:
      return kOpCodeStrTflMinimum;
    case kLiteRtOpCodeTflMirrorPad:
      return kOpCodeStrTflMirrorPad;
    case kLiteRtOpCodeTflMul:
      return kOpCodeStrTflMul;
    case kLiteRtOpCodeTflMultinomial:
      return kOpCodeStrTflMultinomial;
    case kLiteRtOpCodeTflNeg:
      return kOpCodeStrTflNeg;
    case kLiteRtOpCodeTflNonMaxSuppressionV4:
      return kOpCodeStrTflNonMaxSuppressionV4;
    case kLiteRtOpCodeTflNonMaxSuppressionV5:
      return kOpCodeStrTflNonMaxSuppressionV5;
    case kLiteRtOpCodeTflNotEqual:
      return kOpCodeStrTflNotEqual;
    case kLiteRtOpCodeTflOneHot:
      return kOpCodeStrTflOneHot;
    case kLiteRtOpCodeTflPack:
      return kOpCodeStrTflPack;
    case kLiteRtOpCodeTflPad:
      return kOpCodeStrTflPad;
    case kLiteRtOpCodeTflPadv2:
      return kOpCodeStrTflPadv2;
    case kLiteRtOpCodeTflPlaceholderForGreaterOpCodeTfls:
      return kOpCodeStrTflPlaceholderForGreaterOpCodeTfls;
    case kLiteRtOpCodeTflPow:
      return kOpCodeStrTflPow;
    case kLiteRtOpCodeTflPrelu:
      return kOpCodeStrTflPrelu;
    case kLiteRtOpCodeTflQuantize:
      return kOpCodeStrTflQuantize;
    case kLiteRtOpCodeTflRandomStandardNormal:
      return kOpCodeStrTflRandomStandardNormal;
    case kLiteRtOpCodeTflRandomUniform:
      return kOpCodeStrTflRandomUniform;
    case kLiteRtOpCodeTflRange:
      return kOpCodeStrTflRange;
    case kLiteRtOpCodeTflRank:
      return kOpCodeStrTflRank;
    case kLiteRtOpCodeTflReadVariable:
      return kOpCodeStrTflReadVariable;
    case kLiteRtOpCodeTflReal:
      return kOpCodeStrTflReal;
    case kLiteRtOpCodeTflReduceAll:
      return kOpCodeStrTflReduceAll;
    case kLiteRtOpCodeTflReduceAny:
      return kOpCodeStrTflReduceAny;
    case kLiteRtOpCodeTflReduceMax:
      return kOpCodeStrTflReduceMax;
    case kLiteRtOpCodeTflReduceMin:
      return kOpCodeStrTflReduceMin;
    case kLiteRtOpCodeTflReduceProd:
      return kOpCodeStrTflReduceProd;
    case kLiteRtOpCodeTflReduceWindow:
      return kOpCodeStrTflReduceWindow;
    case kLiteRtOpCodeTflRelu:
      return kOpCodeStrTflRelu;
    case kLiteRtOpCodeTflRelu0To1:
      return kOpCodeStrTflRelu0To1;
    case kLiteRtOpCodeTflRelu6:
      return kOpCodeStrTflRelu6;
    case kLiteRtOpCodeTflReluN1To1:
      return kOpCodeStrTflReluN1To1;
    case kLiteRtOpCodeTflReshape:
      return kOpCodeStrTflReshape;
    case kLiteRtOpCodeTflResizeBilinear:
      return kOpCodeStrTflResizeBilinear;
    case kLiteRtOpCodeTflResizeNearestNeighbor:
      return kOpCodeStrTflResizeNearestNeighbor;
    case kLiteRtOpCodeTflReverseSequence:
      return kOpCodeStrTflReverseSequence;
    case kLiteRtOpCodeTflReverseV2:
      return kOpCodeStrTflReverseV2;
    case kLiteRtOpCodeTflRfft2d:
      return kOpCodeStrTflRfft2d;
    case kLiteRtOpCodeTflRightShift:
      return kOpCodeStrTflRightShift;
    case kLiteRtOpCodeTflRnn:
      return kOpCodeStrTflRnn;
    case kLiteRtOpCodeTflRound:
      return kOpCodeStrTflRound;
    case kLiteRtOpCodeTflRsqrt:
      return kOpCodeStrTflRsqrt;
    case kLiteRtOpCodeTflScatterNd:
      return kOpCodeStrTflScatterNd;
    case kLiteRtOpCodeTflSegmentSum:
      return kOpCodeStrTflSegmentSum;
    case kLiteRtOpCodeTflSelect:
      return kOpCodeStrTflSelect;
    case kLiteRtOpCodeTflSelectV2:
      return kOpCodeStrTflSelectV2;
    case kLiteRtOpCodeTflShape:
      return kOpCodeStrTflShape;
    case kLiteRtOpCodeTflSign:
      return kOpCodeStrTflSign;
    case kLiteRtOpCodeTflSin:
      return kOpCodeStrTflSin;
    case kLiteRtOpCodeTflSkipGram:
      return kOpCodeStrTflSkipGram;
    case kLiteRtOpCodeTflSlice:
      return kOpCodeStrTflSlice;
    case kLiteRtOpCodeTflSoftmax:
      return kOpCodeStrTflSoftmax;
    case kLiteRtOpCodeTflSpaceToBatchNd:
      return kOpCodeStrTflSpaceToBatchNd;
    case kLiteRtOpCodeTflSpaceToDepth:
      return kOpCodeStrTflSpaceToDepth;
    case kLiteRtOpCodeTflSparseToDense:
      return kOpCodeStrTflSparseToDense;
    case kLiteRtOpCodeTflSplit:
      return kOpCodeStrTflSplit;
    case kLiteRtOpCodeTflSplitV:
      return kOpCodeStrTflSplitV;
    case kLiteRtOpCodeTflSqrt:
      return kOpCodeStrTflSqrt;
    case kLiteRtOpCodeTflSquare:
      return kOpCodeStrTflSquare;
    case kLiteRtOpCodeTflSquaredDifference:
      return kOpCodeStrTflSquaredDifference;
    case kLiteRtOpCodeTflSqueeze:
      return kOpCodeStrTflSqueeze;
    case kLiteRtOpCodeTflStridedSlice:
      return kOpCodeStrTflStridedSlice;
    case kLiteRtOpCodeTflSub:
      return kOpCodeStrTflSub;
    case kLiteRtOpCodeTflSum:
      return kOpCodeStrTflSum;
    case kLiteRtOpCodeTflSvdf:
      return kOpCodeStrTflSvdf;
    case kLiteRtOpCodeTflTanh:
      return kOpCodeStrTflTanh;
    case kLiteRtOpCodeTflTile:
      return kOpCodeStrTflTile;
    case kLiteRtOpCodeTflTopkV2:
      return kOpCodeStrTflTopkV2;
    case kLiteRtOpCodeTflTranspose:
      return kOpCodeStrTflTranspose;
    case kLiteRtOpCodeTflTransposeConv:
      return kOpCodeStrTflTransposeConv;
    case kLiteRtOpCodeTflUnidirectionalSequenceLstm:
      return kOpCodeStrTflUnidirectionalSequenceLstm;
    case kLiteRtOpCodeTflUnidirectionalSequenceRnn:
      return kOpCodeStrTflUnidirectionalSequenceRnn;
    case kLiteRtOpCodeTflUnique:
      return kOpCodeStrTflUnique;
    case kLiteRtOpCodeTflUnpack:
      return kOpCodeStrTflUnpack;
    case kLiteRtOpCodeTflUnsortedSegmentMax:
      return kOpCodeStrTflUnsortedSegmentMax;
    case kLiteRtOpCodeTflUnsortedSegmentMin:
      return kOpCodeStrTflUnsortedSegmentMin;
    case kLiteRtOpCodeTflUnsortedSegmentProd:
      return kOpCodeStrTflUnsortedSegmentProd;
    case kLiteRtOpCodeTflUnsortedSegmentSum:
      return kOpCodeStrTflUnsortedSegmentSum;
    case kLiteRtOpCodeTflVarHandle:
      return kOpCodeStrTflVarHandle;
    case kLiteRtOpCodeTflWhere:
      return kOpCodeStrTflWhere;
    case kLiteRtOpCodeTflWhile:
      return kOpCodeStrTflWhile;
    case kLiteRtOpCodeTflZerosLike:
      return kOpCodeStrTflZerosLike;
    default:
      return kNoPrinterTag;
  }
}
}  // namespace litert

/// @brief `AbslStringify` specialization for `LiteRtOpCode`.
template <class Sink>
void AbslStringify(Sink& sink, const LiteRtOpCode& code) {
  absl::Format(&sink, "%s", ::litert::GetOpCodeStringView(code));
}

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_C_TYPES_PRINTING_H_
