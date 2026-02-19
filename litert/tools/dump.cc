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

#include "litert/tools/dump.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"

#if !defined(LITERT_WINDOWS_OS)
#include <dlfcn.h>
#endif  // !defined(LITERT_WINDOWS_OS)

#ifndef __ANDROID__
#if __has_include(<link.h>)
#include <link.h>
#endif
#endif

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/model/model.h"

namespace litert::internal {

namespace {

static constexpr int kMaxDisplayCount = 16;

void DumpNode(const LiteRtTensorT& tensor, std::ostream& out) {
  switch (tensor.Type().first) {
    case kLiteRtRankedTensorType:
      Dump(tensor.Type().second.ranked_tensor_type, out);
      break;
    case kLiteRtUnrankedTensorType:
      Dump(tensor.Type().second.unranked_tensor_type.element_type, out);
      break;
    default:
      out << "UKNOWN_TENSOR_TYPE" << tensor.Type().first;
  }
  Dump(tensor.Qparams(), out);
}

void DumpNode(const LiteRtOpT& op, std::ostream& out) {
  Dump(op.OpCode(), out);
}

void DumpSignature(const std::vector<LiteRtTensor>& ins,
                   const std::vector<LiteRtTensor>& outs, std::ostream& out) {
  out << "(";
  for (auto it = ins.begin(); it < ins.end(); ++it) {
    DumpNode(**it, out);
    if (it != ins.end() - 1) {
      out << ", ";
    }
  }
  out << ")";

  out << " -> ";
  const bool paren_outs = outs.size() != 1;
  if (paren_outs) {
    out << "(";
  }
  for (auto it = outs.begin(); it < outs.end(); ++it) {
    DumpNode(**it, out);
    if (it != outs.end() - 1) {
      out << ", ";
    }
  }
  if (paren_outs) {
    out << ")";
  }
}

}  // namespace

void Dump(LiteRtOpCode code, std::ostream& out) {
  switch (code) {
#define DUMP_OP(op_enum, op_name) \
  case kLiteRtOpCode##op_enum:    \
    out << op_name;               \
    break;

    DUMP_OP(TflAdd, "TFL_ADD");
    DUMP_OP(TflAveragePool2d, "TFL_AVERAGE_POOL_2D");
    DUMP_OP(TflConcatenation, "TFL_CONCATENATION");
    DUMP_OP(TflConv2d, "TFL_CONV_2D");
    DUMP_OP(TflDepthwiseConv2d, "TFL_DEPTHWISE_CONV_2D");
    DUMP_OP(TflDepthToSpace, "TFL_DEPTH_TO_SPACE");
    DUMP_OP(TflDequantize, "TFL_DEQUANTIZE");
    DUMP_OP(TflEmbeddingLookup, "TFL_EMBEDDING_LOOKUP");
    DUMP_OP(TflFloor, "TFL_FLOOR");
    DUMP_OP(TflFullyConnected, "TFL_FULLY_CONNECTED");
    DUMP_OP(TflHashtableLookup, "TFL_HASHTABLE_LOOKUP");
    DUMP_OP(TflL2Normalization, "TFL_L2_NORMALIZATION");
    DUMP_OP(TflL2Pool2d, "TFL_L2_POOL_2D");
    DUMP_OP(TflLocalResponseNormalization, "TFL_LOCAL_RESPONSE_NORMALIZATION");
    DUMP_OP(TflLogistic, "TFL_LOGISTIC");
    DUMP_OP(TflLshProjection, "TFL_LSH_PROJECTION");
    DUMP_OP(TflLstm, "TFL_LSTM");
    DUMP_OP(TflMaxPool2d, "TFL_MAX_POOL_2D");
    DUMP_OP(TflMul, "TFL_MUL");
    DUMP_OP(TflRelu, "TFL_RELU");
    DUMP_OP(TflReluN1To1, "TFL_RELU_N1_TO_1");
    DUMP_OP(TflRelu6, "TFL_RELU6");
    DUMP_OP(TflReshape, "TFL_RESHAPE");
    DUMP_OP(TflResizeBilinear, "TFL_RESIZE_BILINEAR");
    DUMP_OP(TflRnn, "TFL_RNN");
    DUMP_OP(TflSoftmax, "TFL_SOFTMAX");
    DUMP_OP(TflSpaceToDepth, "TFL_SPACE_TO_DEPTH");
    DUMP_OP(TflSvdf, "TFL_SVDF");
    DUMP_OP(TflTanh, "TFL_TANH");
    DUMP_OP(TflConcatEmbeddings, "TFL_CONCAT_EMBEDDINGS");
    DUMP_OP(TflSkipGram, "TFL_SKIP_GRAM");
    DUMP_OP(TflCall, "TFL_CALL");
    DUMP_OP(TflCustom, "TFL_CUSTOM_OP");
    DUMP_OP(TflEmbeddingLookupSparse, "TFL_EMBEDDING_LOOKUP_SPARSE");
    DUMP_OP(TflPad, "TFL_PAD");
    DUMP_OP(TflUnidirectionalSequenceRnn, "TFL_UNIDIRECTIONAL_SEQUENCE_RNN");
    DUMP_OP(TflGather, "TFL_GATHER");
    DUMP_OP(TflBatchToSpaceNd, "TFL_BATCH_TO_SPACE_ND");
    DUMP_OP(TflSpaceToBatchNd, "TFL_SPACE_TO_BATCH_ND");
    DUMP_OP(TflTranspose, "TFL_TRANSPOSE");
    DUMP_OP(TflMean, "TFL_MEAN");
    DUMP_OP(TflSub, "TFL_SUB");
    DUMP_OP(TflDiv, "TFL_DIV");
    DUMP_OP(TflSqueeze, "TFL_SQUEEZE");
    DUMP_OP(TflUnidirectionalSequenceLstm, "TFL_UNIDIRECTIONAL_SEQUENCE_LSTM");
    DUMP_OP(TflStridedSlice, "TFL_STRIDED_SLICE");
    DUMP_OP(TflBidirectionalSequenceRnn, "TFL_BIDIRECTIONAL_SEQUENCE_RNN");
    DUMP_OP(TflExp, "TFL_EXP");
    DUMP_OP(TflTopkV2, "TFL_TOPK_V2");
    DUMP_OP(TflSplit, "TFL_SPLIT");
    DUMP_OP(TflLogSoftmax, "TFL_LOG_SOFTMAX");
    DUMP_OP(TflDelegate, "TFL_DELEGATE");
    DUMP_OP(TflBidirectionalSequenceLstm, "TFL_BIDIRECTIONAL_SEQUENCE_LSTM");
    DUMP_OP(TflCast, "TFL_CAST");
    DUMP_OP(TflPrelu, "TFL_PRELU");
    DUMP_OP(TflMaximum, "TFL_MAXIMUM");
    DUMP_OP(TflArgMax, "TFL_ARG_MAX");
    DUMP_OP(TflMinimum, "TFL_MINIMUM");
    DUMP_OP(TflLess, "TFL_LESS");
    DUMP_OP(TflNeg, "TFL_NEG");
    DUMP_OP(TflPadv2, "TFL_PAD_V2");
    DUMP_OP(TflGreater, "TFL_GREATER");
    DUMP_OP(TflGreaterEqual, "TFL_GREATER_EQUAL");
    DUMP_OP(TflLessEqual, "TFL_LESS_EQUAL");
    DUMP_OP(TflSelect, "TFL_SELECT");
    DUMP_OP(TflSlice, "TFL_SLICE");
    DUMP_OP(TflSin, "TFL_SIN");
    DUMP_OP(TflTransposeConv, "TFL_TRANSPOSE_CONV");
    DUMP_OP(TflSparseToDense, "TFL_SPARSE_TO_DENSE");
    DUMP_OP(TflTile, "TFL_TILE");
    DUMP_OP(TflExpandDims, "TFL_EXPAND_DIMS");
    DUMP_OP(TflEqual, "TFL_EQUAL");
    DUMP_OP(TflNotEqual, "TFL_NOT_EQUAL");
    DUMP_OP(TflLog, "TFL_LOG");
    DUMP_OP(TflSum, "TFL_SUM");
    DUMP_OP(TflSqrt, "TFL_SQRT");
    DUMP_OP(TflRsqrt, "TFL_RSQRT");
    DUMP_OP(TflShape, "TFL_SHAPE");
    DUMP_OP(TflPow, "TFL_POW");
    DUMP_OP(TflArgMin, "TFL_ARG_MIN");
    DUMP_OP(TflFakeQuant, "TFL_FAKE_QUANT");
    DUMP_OP(TflReduceProd, "TFL_REDUCE_PROD");
    DUMP_OP(TflReduceMax, "TFL_REDUCE_MAX");
    DUMP_OP(TflPack, "TFL_PACK");
    DUMP_OP(TflLogicalOr, "TFL_LOGICAL_OR");
    DUMP_OP(TflOneHot, "TFL_ONE_HOT");
    DUMP_OP(TflLogicalAnd, "TFL_LOGICAL_AND");
    DUMP_OP(TflLogicalNot, "TFL_LOGICAL_NOT");
    DUMP_OP(TflUnpack, "TFL_UNPACK");
    DUMP_OP(TflReduceMin, "TFL_REDUCE_MIN");
    DUMP_OP(TflFloorDiv, "TFL_FLOOR_DIV");
    DUMP_OP(TflReduceAny, "TFL_REDUCE_ANY");
    DUMP_OP(TflSquare, "TFL_SQUARE");
    DUMP_OP(TflZerosLike, "TFL_ZEROS_LIKE");
    DUMP_OP(TflFill, "TFL_FILL");
    DUMP_OP(TflFloorMod, "TFL_FLOOR_MOD");
    DUMP_OP(TflRange, "TFL_RANGE");
    DUMP_OP(TflResizeNearestNeighbor, "TFL_RESIZE_NEAREST_NEIGHBOR");
    DUMP_OP(TflLeakyRelu, "TFL_LEAKY_RELU");
    DUMP_OP(TflSquaredDifference, "TFL_SQUARED_DIFFERENCE");
    DUMP_OP(TflMirrorPad, "TFL_MIRROR_PAD");
    DUMP_OP(TflAbs, "TFL_ABS");
    DUMP_OP(TflSplitV, "TFL_SPLIT_V");
    DUMP_OP(TflUnique, "TFL_UNIQUE");
    DUMP_OP(TflCeil, "TFL_CEIL");
    DUMP_OP(TflReverseV2, "TFL_REVERSE_V2");
    DUMP_OP(TflAddN, "TFL_ADD_N");
    DUMP_OP(TflGatherNd, "TFL_GATHER_ND");
    DUMP_OP(TflCos, "TFL_COS");
    DUMP_OP(TflWhere, "TFL_WHERE");
    DUMP_OP(TflRank, "TFL_RANK");
    DUMP_OP(TflElu, "TFL_ELU");
    DUMP_OP(TflReverseSequence, "TFL_REVERSE_SEQUENCE");
    DUMP_OP(TflMatrixDiag, "TFL_MATRIX_DIAG");
    DUMP_OP(TflQuantize, "TFL_QUANTIZE");
    DUMP_OP(TflMatrixSetDiag, "TFL_MATRIX_SET_DIAG");
    DUMP_OP(TflRound, "TFL_ROUND");
    DUMP_OP(TflHardSwish, "TFL_HARD_SWISH");
    DUMP_OP(TflIf, "TFL_IF");
    DUMP_OP(TflWhile, "TFL_WHILE");
    DUMP_OP(TflNonMaxSuppressionV4, "TFL_NON_MAX_SUPPRESSION_V4");
    DUMP_OP(TflNonMaxSuppressionV5, "TFL_NON_MAX_SUPPRESSION_V5");
    DUMP_OP(TflScatterNd, "TFL_SCATTER_ND");
    DUMP_OP(TflSelectV2, "TFL_SELECT_V2");
    DUMP_OP(TflDensify, "TFL_DENSIFY");
    DUMP_OP(TflSegmentSum, "TFL_SEGMENT_SUM");
    DUMP_OP(TflBatchMatmul, "TFL_BATCH_MATMUL");
    DUMP_OP(TflPlaceholderForGreaterOpCodeTfls,
            "TFL_PLACEHOLDER_FOR_GREATER_OP_CODES");
    DUMP_OP(TflCumsum, "TFL_CUMSUM");
    DUMP_OP(TflCallOnce, "TFL_CALL_ONCE");
    DUMP_OP(TflBroadcastTo, "TFL_BROADCAST_TO");
    DUMP_OP(TflRfft2d, "TFL_RFFT_2D");
    DUMP_OP(TflConv3d, "TFL_CONV_3D");
    DUMP_OP(TflImag, "TFL_IMAG");
    DUMP_OP(TflReal, "TFL_REAL");
    DUMP_OP(TflComplexAbs, "TFL_COMPLEX_ABS");
    DUMP_OP(TflHashtable, "TFL_HASHTABLE");
    DUMP_OP(TflHashtableFind, "TFL_HASHTABLE_FIND");
    DUMP_OP(TflHashtableImport, "TFL_HASHTABLE_IMPORT");
    DUMP_OP(TflHashtableSize, "TFL_HASHTABLE_SIZE");
    DUMP_OP(TflReduceAll, "TFL_REDUCE_ALL");
    DUMP_OP(TflConv3dTranspose, "TFL_CONV_3D_TRANSPOSE");
    DUMP_OP(TflVarHandle, "TFL_VAR_HANDLE");
    DUMP_OP(TflReadVariable, "TFL_READ_VARIABLE");
    DUMP_OP(TflAssignVariable, "TFL_ASSIGN_VARIABLE");
    DUMP_OP(TflBroadcastArgs, "TFL_BROADCAST_ARGS");
    DUMP_OP(TflRandomStandardNormal, "TFL_RANDOM_STANDARD_NORMAL");
    DUMP_OP(TflBucketize, "TFL_BUCKETIZE");
    DUMP_OP(TflRandomUniform, "TFL_RANDOM_UNIFORM");
    DUMP_OP(TflMultinomial, "TFL_MULTINOMIAL");
    DUMP_OP(TflGelu, "TFL_GELU");
    DUMP_OP(TflDynamicUpdateSlice, "TFL_DYNAMIC_UPDATE_SLICE");
    DUMP_OP(TflRelu0To1, "TFL_RELU_0_TO_1");
    DUMP_OP(TflUnsortedSegmentProd, "TFL_UNSORTED_SEGMENT_PROD");
    DUMP_OP(TflUnsortedSegmentMax, "TFL_UNSORTED_SEGMENT_MAX");
    DUMP_OP(TflUnsortedSegmentSum, "TFL_UNSORTED_SEGMENT_SUM");
    DUMP_OP(TflAtan2, "TFL_ATAN2");
    DUMP_OP(TflUnsortedSegmentMin, "TFL_UNSORTED_SEGMENT_MIN");
    DUMP_OP(TflSign, "TFL_SIGN");
    DUMP_OP(TflBitcast, "TFL_BITCAST");
    DUMP_OP(TflBitwiseXor, "TFL_BITWISE_XOR");
    DUMP_OP(TflRightShift, "TFL_RIGHT_SHIFT");
    DUMP_OP(ShloLogistic, "SHLO_LOGISTIC");
    DUMP_OP(ShloAdd, "SHLO_ADD");
    DUMP_OP(ShloDivide, "SHLO_DIVIDE");
    DUMP_OP(ShloMultiply, "SHLO_MULTIPLY");
    DUMP_OP(ShloMaximum, "SHLO_MAXIMUM");
    DUMP_OP(ShloReshape, "SHLO_RESHAPE");
    DUMP_OP(ShloClamp, "SHLO_CLAMP");
    DUMP_OP(ShloConcatenate, "SHLO_CONCATENATE");
    DUMP_OP(ShloBroadcastInDim, "SHLO_BROADCAST_IN_DIM");
    DUMP_OP(ShloConvolution, "SHLO_CONVOLUTION");
    DUMP_OP(ShloSlice, "SHLO_SLICE");
    DUMP_OP(ShloCustomCall, "SHLO_CUSTOM_CALL");
    DUMP_OP(ShloReduce, "SHLO_REDUCE");
    DUMP_OP(ShloAbs, "SHLO_ABS");
    DUMP_OP(ShloAnd, "SHLO_AND");
    DUMP_OP(ShloCosine, "SHLO_COSINE");
    DUMP_OP(ShloExponential, "SHLO_EXPONENTIAL");
    DUMP_OP(ShloFloor, "SHLO_FLOOR");
    DUMP_OP(ShloLog, "SHLO_LOG");
    DUMP_OP(ShloMinimum, "SHLO_MINIMUM");
    DUMP_OP(ShloNegate, "SHLO_NEGATE");
    DUMP_OP(ShloOr, "SHLO_OR");
    DUMP_OP(ShloPower, "SHLO_POWER");
    DUMP_OP(ShloRemainder, "SHLO_REMAINDER");
    DUMP_OP(ShloRsqrt, "SHLO_RSQRT");
    DUMP_OP(ShloSelect, "SHLO_SELECT");
    DUMP_OP(ShloSubtract, "SHLO_SUBTRACT");
    DUMP_OP(ShloTanh, "SHLO_TANH");
    DUMP_OP(ShloScatter, "SHLO_SCATTER");
    DUMP_OP(ShloCompare, "SHLO_COMPARE");
    DUMP_OP(ShloConvert, "SHLO_CONVERT");
    DUMP_OP(ShloDynamicSlice, "SHLO_DYNAMIC_SLICE");
    DUMP_OP(ShloDynamicUpdateSlice, "SHLO_DYNAMIC_UPDATE_SLICE");
    DUMP_OP(ShloPad, "SHLO_PAD");
    DUMP_OP(ShloIota, "SHLO_IOTA");
    DUMP_OP(ShloGeneral, "SHLO_GENERAL");
    DUMP_OP(ShloWindow, "SHLO_WINDOW");
    DUMP_OP(ShloSort, "SHLO_SORT");
    DUMP_OP(ShloWhile, "SHLO_WHILE");
    DUMP_OP(ShloGather, "SHLO_GATHER");
    DUMP_OP(ShloTranspose, "SHLO_TRANSPOSE");
    DUMP_OP(TflDilate, "TFL_DILATE");
    DUMP_OP(ShloRngBitGenerator, "SHLO_RNG_BIT_GENERATOR");
    DUMP_OP(TflReduceWindow, "TFL_REDUCE_WINDOW");
    DUMP_OP(ShloComposite, "SHLO_COMPOSITE");

#undef DUMP_OP
    default:
      out << "UNKNOWN_OP_CODE: " << code;
      break;
  }
};

// Dump details about the given LiteRtElementType to the given stream.
void Dump(LiteRtElementType type, std::ostream& out) {
  switch (type) {
    case kLiteRtElementTypeFloat32:
      out << "f32";
      break;
    case kLiteRtElementTypeInt32:
      out << "i32";
      break;
    case kLiteRtElementTypeFloat64:
      out << "f64";
      break;
    case kLiteRtElementTypeInt64:
      out << "i64";
      break;
    case kLiteRtElementTypeFloat16:
      out << "f16";
      break;
    case kLiteRtElementTypeInt16:
      out << "i16";
      break;
    case kLiteRtElementTypeInt8:
      out << "i8";
      break;
    case kLiteRtElementTypeUInt8:
      out << "ui8";
      break;
    case kLiteRtElementTypeBool:
      out << "i1";
      break;
    default:
      out << "UKNNOWN_ELEMENT_TYPE: " << type;
  }
}

void Dump(const LiteRtRankedTensorType& type, std::ostream& out) {
  out << "<";
  for (int i = 0; i < type.layout.rank; ++i) {
    out << type.layout.dimensions[i] << "x";
  }
  Dump(type.element_type, out);
  out << ">";
}

void Dump(const LiteRtTensorT& tensor, std::ostream& out) {
  out << "LiteRtTensor : ";
  DumpNode(tensor, out);
  out << " [ ";
  if (tensor.DefiningOp() == nullptr) {
    out << "*";
  } else {
    DumpNode(*tensor.DefiningOp(), out);
  }
  out << " ] ";

  out << "(";
  for (auto it = tensor.Users().begin(); it < tensor.Users().end(); ++it) {
    DumpNode(**it, out);
    if (it != tensor.Users().end() - 1) {
      out << ", ";
    }
  }
  out << ")";
  out << "\n";
}

void Dump(const LiteRtOpT& op, std::ostream& out) {
  out << "LiteRtOp : [ ";
  DumpNode(op, out);
  if (auto custom_code = op.CustomCode(); custom_code.HasValue()) {
    out << " : " << std::string(*custom_code);
  }
  out << " ] ";
  DumpSignature(op.Inputs(), op.Outputs(), out);
  out << "\n";
}

void Dump(const LiteRtSubgraphT& subgraph, std::ostream& out) {
  constexpr absl::string_view kSubgraphTpl =
      "LiteRtSubgraph : [ #ops=%d #tensors=%d ] ";
  out << absl::StreamFormat(kSubgraphTpl, subgraph.Ops().size(),
                            subgraph.Tensors().size());
  DumpSignature(subgraph.Inputs(), subgraph.Outputs(), out);
  out << "\n";
}

void Dump(const CompilerPlugin& plugin, std::ostream& out) {
  constexpr absl::string_view kPluginDumpTpl =
      "SocManufacturer: %s\nSocModels: { ";
  out << absl::StreamFormat(kPluginDumpTpl, plugin.SocManufacturer());

  for (auto it = plugin.SocModels().begin(); it < plugin.SocModels().end();
       ++it) {
    out << *it;
    if (it != plugin.SocModels().end() - 1) {
      out << ",";
    }
    out << " ";
  }

  out << "}\n";
}

void Dump(const LiteRtModelT& model, std::ostream& out) {
  out << absl::StreamFormat("LiteRtModel : [ #subgraphs=%d ]\n",
                            model.Subgraphs().size());
}

void DumpOptions(const LiteRtOpT& op, std::ostream& out) {
  auto& opts = litert::internal::GetTflOptions(op);
  if (opts.value == nullptr) {
    out << "null options\n";
    return;
  }
  switch (op.OpCode()) {
    case kLiteRtOpCodeTflAdd:
      out << "fused_activation_function: "
          << opts.AsAddOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflMul:
      out << "fused_activation_function: "
          << opts.AsMulOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflBatchMatmul:
      out << "adj_x: " << opts.AsBatchMatMulOptions()->adj_x << "\n";
      out << "adj_y: " << opts.AsBatchMatMulOptions()->adj_y << "\n";
      out << "asymmetric_quantize_input: "
          << opts.AsBatchMatMulOptions()->asymmetric_quantize_inputs << "\n";
      break;
    case kLiteRtOpCodeTflConcatenation:
      out << "axis: " << opts.AsConcatenationOptions()->axis << "\n";
      out << "fused_activation_function: "
          << opts.AsConcatenationOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflDiv:
      out << "fused_activation_function: "
          << opts.AsDivOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflFullyConnected:
      out << "weights_format: "
          << opts.AsFullyConnectedOptions()->weights_format << "\n";
      out << "keep_num_dims: " << opts.AsFullyConnectedOptions()->keep_num_dims
          << "\n";
      out << "quantized_bias_type: "
          << opts.AsFullyConnectedOptions()->quantized_bias_type << "\n";
      out << "asymmetric_quantize_input: "
          << opts.AsFullyConnectedOptions()->asymmetric_quantize_inputs << "\n";
      out << "fused_activation_function: "
          << opts.AsFullyConnectedOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflSoftmax:
      out << "beta: " << opts.AsSoftmaxOptions()->beta << "\n";
      break;
    case kLiteRtOpCodeTflStridedSlice:
      out << "begin_mask: " << opts.AsStridedSliceOptions()->begin_mask << "\n";
      out << "end_mask: " << opts.AsStridedSliceOptions()->end_mask << "\n";
      out << "ellipsis_mask: " << opts.AsStridedSliceOptions()->ellipsis_mask
          << "\n";
      out << "new_axis_mask: " << opts.AsStridedSliceOptions()->new_axis_mask
          << "\n";
      out << "shrink_axis_mask: "
          << opts.AsStridedSliceOptions()->shrink_axis_mask << "\n";
      out << "offset: " << opts.AsStridedSliceOptions()->offset << "\n";
      break;
    case kLiteRtOpCodeTflSub:
      out << "fused_activation_function: "
          << opts.AsSubOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflReshape:
      out << "new_shape: ";
      if (opts.AsReshapeOptions() != nullptr) {
        const int32_t* new_shape = opts.AsReshapeOptions()->new_shape.data();
        int32_t new_shape_size = opts.AsReshapeOptions()->new_shape.size();
        for (int i = 0; i < new_shape_size; ++i) {
          out << new_shape[i] << " ";
        }
      }
      break;
    case kLiteRtOpCodeTflSum:
      out << "keepdims: " << opts.AsReducerOptions()->keep_dims << "\n";
      break;
    case kLiteRtOpCodeTflPack:
      out << "axis: " << opts.AsPackOptions()->axis << "\n";
      break;
    default:
      out << "No options for op code: " << op.OpCode();
      break;
  }
}

void Dump(Quantization quantization, std::ostream& out) {
  int max_display_count;
  switch (quantization.first) {
    case kLiteRtQuantizationNone:
      return;
    case kLiteRtQuantizationPerTensor:
      out << absl::StreamFormat(" <q PerTensor [ .z = %ld, .s = %f ]>",
                                quantization.second.per_tensor.zero_point,
                                quantization.second.per_tensor.scale);
      return;
    case kLiteRtQuantizationPerChannel:
      max_display_count =
          kMaxDisplayCount < quantization.second.per_channel.num_channels
              ? kMaxDisplayCount
              : quantization.second.per_channel.num_channels;
      out << absl::StreamFormat(" <q PerChannel [ .z = [ ");
      for (int i = 0; i < max_display_count; ++i) {
        out << absl::StreamFormat(
            "%ld, ", quantization.second.per_channel.zero_points[i]);
      }
      out << "...], .s = [ ";
      for (int i = 0; i < max_display_count; ++i) {
        out << absl::StreamFormat("%f, ",
                                  quantization.second.per_channel.scales[i]);
      }
      out << "...], ";
      out << absl::StreamFormat(
          ".d = %d>", quantization.second.per_channel.quantized_dimension);
      return;
    default:
      out << " <q UNKNOWN>";
      return;
  }
}

}  // namespace litert::internal
