// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/intel_openvino/compiler/decoder.h"

#include <map>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_op_options.h"
#include "litert/vendors/intel_openvino/utils.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace openvino {

// This has been picked from the openvino build:
// build/src/frontends/tensorflow_lite/src/schema_generated.h
constexpr std::array<std::pair<LiteRtOpCode, const char*>, 159> kLitertOvMap{
    {{kLiteRtOpCodeTflAdd, "ADD"},
     {kLiteRtOpCodeTflAveragePool2d, "AVERAGE_POOL_2D"},
     {kLiteRtOpCodeTflConcatenation, "CONCATENATION"},
     {kLiteRtOpCodeTflConv2d, "CONV_2D"},
     {kLiteRtOpCodeTflDepthwiseConv2d, "DEPTHWISE_CONV_2D"},
     {kLiteRtOpCodeTflDepthToSpace, "DEPTH_TO_SPACE"},
     {kLiteRtOpCodeTflDequantize, "DEQUANTIZE"},
     {kLiteRtOpCodeTflEmbeddingLookup, "EMBEDDING_LOOKUP"},
     {kLiteRtOpCodeTflFloor, "FLOOR"},
     {kLiteRtOpCodeTflFullyConnected, "FULLY_CONNECTED"},
     {kLiteRtOpCodeTflHashtableLookup, "HASHTABLE_LOOKUP"},
     {kLiteRtOpCodeTflL2Normalization, "L2_NORMALIZATION"},
     {kLiteRtOpCodeTflL2Pool2d, "L2_POOL_2D"},
     {kLiteRtOpCodeTflLocalResponseNormalization,
      "LOCAL_RESPONSE_NORMALIZATION"},
     {kLiteRtOpCodeTflLogistic, "LOGISTIC"},
     {kLiteRtOpCodeTflLshProjection, "LSH_PROJECTION"},
     {kLiteRtOpCodeTflLstm, "LSTM"},
     {kLiteRtOpCodeTflMaxPool2d, "MAX_POOL_2D"},
     {kLiteRtOpCodeTflMul, "MUL"},
     {kLiteRtOpCodeTflRelu, "RELU"},
     {kLiteRtOpCodeTflReluN1To1, "RELU_N1_TO_1"},
     {kLiteRtOpCodeTflRelu6, "RELU6"},
     {kLiteRtOpCodeTflReshape, "RESHAPE"},
     {kLiteRtOpCodeTflResizeBilinear, "RESIZE_BILINEAR"},
     {kLiteRtOpCodeTflRnn, "RNN"},
     {kLiteRtOpCodeTflSoftmax, "SOFTMAX"},
     {kLiteRtOpCodeTflSpaceToDepth, "SPACE_TO_DEPTH"},
     {kLiteRtOpCodeTflSvdf, "SVDF"},
     {kLiteRtOpCodeTflTanh, "TANH"},
     {kLiteRtOpCodeTflConcatEmbeddings, "CONCAT_EMBEDDINGS"},
     {kLiteRtOpCodeTflSkipGram, "SKIP_GRAM"},
     {kLiteRtOpCodeTflCall, "CALL"},
     {kLiteRtOpCodeTflCustom, "CUSTOM"},
     {kLiteRtOpCodeTflEmbeddingLookupSparse, "EMBEDDING_LOOKUP_SPARSE"},
     {kLiteRtOpCodeTflPad, "PAD"},
     {kLiteRtOpCodeTflUnidirectionalSequenceRnn, "UNIDIRECTIONAL_SEQUENCE_RNN"},
     {kLiteRtOpCodeTflGather, "GATHER"},
     {kLiteRtOpCodeTflBatchToSpaceNd, "BATCH_TO_SPACE_ND"},
     {kLiteRtOpCodeTflSpaceToBatchNd, "SPACE_TO_BATCH_ND"},
     {kLiteRtOpCodeTflTranspose, "TRANSPOSE"},
     {kLiteRtOpCodeTflMean, "MEAN"},
     {kLiteRtOpCodeTflSub, "SUB"},
     {kLiteRtOpCodeTflDiv, "DIV"},
     {kLiteRtOpCodeTflSqueeze, "SQUEEZE"},
     {kLiteRtOpCodeTflUnidirectionalSequenceLstm,
      "UNIDIRECTIONAL_SEQUENCE_LSTM"},
     {kLiteRtOpCodeTflStridedSlice, "STRIDED_SLICE"},
     {kLiteRtOpCodeTflBidirectionalSequenceRnn, "BIDIRECTIONAL_SEQUENCE_RNN"},
     {kLiteRtOpCodeTflExp, "EXP"},
     {kLiteRtOpCodeTflTopkV2, "TOPK_V2"},
     {kLiteRtOpCodeTflSplit, "SPLIT"},
     {kLiteRtOpCodeTflLogSoftmax, "LOG_SOFTMAX"},
     {kLiteRtOpCodeTflDelegate, "DELEGATE"},
     {kLiteRtOpCodeTflBidirectionalSequenceLstm, "BIDIRECTIONAL_SEQUENCE_LSTM"},
     {kLiteRtOpCodeTflCast, "CAST"},
     {kLiteRtOpCodeTflPrelu, "PRELU"},
     {kLiteRtOpCodeTflMaximum, "MAXIMUM"},
     {kLiteRtOpCodeTflArgMax, "ARG_MAX"},
     {kLiteRtOpCodeTflMinimum, "MINIMUM"},
     {kLiteRtOpCodeTflLess, "LESS"},
     {kLiteRtOpCodeTflNeg, "NEG"},
     {kLiteRtOpCodeTflPadv2, "PADV2"},
     {kLiteRtOpCodeTflGreater, "GREATER"},
     {kLiteRtOpCodeTflGreaterEqual, "GREATER_EQUAL"},
     {kLiteRtOpCodeTflLessEqual, "LESS_EQUAL"},
     {kLiteRtOpCodeTflSelect, "SELECT"},
     {kLiteRtOpCodeTflSlice, "SLICE"},
     {kLiteRtOpCodeTflSin, "SIN"},
     {kLiteRtOpCodeTflTransposeConv, "TRANSPOSE_CONV"},
     {kLiteRtOpCodeTflSparseToDense, "SPARSE_TO_DENSE"},
     {kLiteRtOpCodeTflTile, "TILE"},
     {kLiteRtOpCodeTflExpandDims, "EXPAND_DIMS"},
     {kLiteRtOpCodeTflEqual, "EQUAL"},
     {kLiteRtOpCodeTflNotEqual, "NOT_EQUAL"},
     {kLiteRtOpCodeTflLog, "LOG"},
     {kLiteRtOpCodeTflSum, "SUM"},
     {kLiteRtOpCodeTflSqrt, "SQRT"},
     {kLiteRtOpCodeTflRsqrt, "RSQRT"},
     {kLiteRtOpCodeTflShape, "SHAPE"},
     {kLiteRtOpCodeTflPow, "POW"},
     {kLiteRtOpCodeTflArgMin, "ARG_MIN"},
     {kLiteRtOpCodeTflFakeQuant, "FAKE_QUANT"},
     {kLiteRtOpCodeTflReduceProd, "REDUCE_PROD"},
     {kLiteRtOpCodeTflReduceMax, "REDUCE_MAX"},
     {kLiteRtOpCodeTflPack, "PACK"},
     {kLiteRtOpCodeTflLogicalOr, "LOGICAL_OR"},
     {kLiteRtOpCodeTflOneHot, "ONE_HOT"},
     {kLiteRtOpCodeTflLogicalAnd, "LOGICAL_AND"},
     {kLiteRtOpCodeTflLogicalNot, "LOGICAL_NOT"},
     {kLiteRtOpCodeTflUnpack, "UNPACK"},
     {kLiteRtOpCodeTflReduceMin, "REDUCE_MIN"},
     {kLiteRtOpCodeTflFloorDiv, "FLOOR_DIV"},
     {kLiteRtOpCodeTflReduceAny, "REDUCE_ANY"},
     {kLiteRtOpCodeTflSquare, "SQUARE"},
     {kLiteRtOpCodeTflZerosLike, "ZEROS_LIKE"},
     {kLiteRtOpCodeTflFill, "FILL"},
     {kLiteRtOpCodeTflFloorMod, "FLOOR_MOD"},
     {kLiteRtOpCodeTflRange, "RANGE"},
     {kLiteRtOpCodeTflResizeNearestNeighbor, "RESIZE_NEAREST_NEIGHBOR"},
     {kLiteRtOpCodeTflLeakyRelu, "LEAKY_RELU"},
     {kLiteRtOpCodeTflSquaredDifference, "SQUARED_DIFFERENCE"},
     {kLiteRtOpCodeTflMirrorPad, "MIRROR_PAD"},
     {kLiteRtOpCodeTflAbs, "ABS"},
     {kLiteRtOpCodeTflSplitV, "SPLIT_V"},
     {kLiteRtOpCodeTflUnique, "UNIQUE"},
     {kLiteRtOpCodeTflCeil, "CEIL"},
     {kLiteRtOpCodeTflReverseV2, "REVERSE_V2"},
     {kLiteRtOpCodeTflAddN, "ADD_N"},
     {kLiteRtOpCodeTflGatherNd, "GATHER_ND"},
     {kLiteRtOpCodeTflCos, "COS"},
     {kLiteRtOpCodeTflWhere, "WHERE"},
     {kLiteRtOpCodeTflRank, "RANK"},
     {kLiteRtOpCodeTflElu, "ELU"},
     {kLiteRtOpCodeTflReverseSequence, "REVERSE_SEQUENCE"},
     {kLiteRtOpCodeTflMatrixDiag, "MATRIX_DIAG"},
     {kLiteRtOpCodeTflQuantize, "QUANTIZE"},
     {kLiteRtOpCodeTflMatrixSetDiag, "MATRIX_SET_DIAG"},
     {kLiteRtOpCodeTflRound, "ROUND"},
     {kLiteRtOpCodeTflHardSwish, "HARD_SWISH"},
     {kLiteRtOpCodeTflIf, "IF"},
     {kLiteRtOpCodeTflWhile, "WHILE"},
     {kLiteRtOpCodeTflNonMaxSuppressionV4, "NON_MAX_SUPPRESSION_V4"},
     {kLiteRtOpCodeTflNonMaxSuppressionV5, "NON_MAX_SUPPRESSION_V5"},
     {kLiteRtOpCodeTflScatterNd, "SCATTER_ND"},
     {kLiteRtOpCodeTflSelectV2, "SELECT_V2"},
     {kLiteRtOpCodeTflDensify, "DENSIFY"},
     {kLiteRtOpCodeTflSegmentSum, "SEGMENT_SUM"},
     {kLiteRtOpCodeTflBatchMatmul, "BATCH_MATMUL"},
     {kLiteRtOpCodeTflPlaceholderForGreaterOpCodeTfls,
      "PLACEHOLDER_FOR_GREATER_OP_CODES"},
     {kLiteRtOpCodeTflCumsum, "CUMSUM"},
     {kLiteRtOpCodeTflCallOnce, "CALL_ONCE"},
     {kLiteRtOpCodeTflBroadcastTo, "BROADCAST_TO"},
     {kLiteRtOpCodeTflRfft2d, "RFFT2D"},
     {kLiteRtOpCodeTflConv3d, "CONV_3D"},
     {kLiteRtOpCodeTflImag, "IMAG"},
     {kLiteRtOpCodeTflReal, "REAL"},
     {kLiteRtOpCodeTflComplexAbs, "COMPLEX_ABS"},
     {kLiteRtOpCodeTflHashtable, "HASHTABLE"},
     {kLiteRtOpCodeTflHashtableFind, "HASHTABLE_FIND"},
     {kLiteRtOpCodeTflHashtableImport, "HASHTABLE_IMPORT"},
     {kLiteRtOpCodeTflHashtableSize, "HASHTABLE_SIZE"},
     {kLiteRtOpCodeTflReduceAll, "REDUCE_ALL"},
     {kLiteRtOpCodeTflConv3dTranspose, "CONV_3D_TRANSPOSE"},
     {kLiteRtOpCodeTflVarHandle, "VAR_HANDLE"},
     {kLiteRtOpCodeTflReadVariable, "READ_VARIABLE"},
     {kLiteRtOpCodeTflAssignVariable, "ASSIGN_VARIABLE"},
     {kLiteRtOpCodeTflBroadcastArgs, "BROADCAST_ARGS"},
     {kLiteRtOpCodeTflRandomStandardNormal, "RANDOM_STANDARD_NORMAL"},
     {kLiteRtOpCodeTflBucketize, "BUCKETIZE"},
     {kLiteRtOpCodeTflRandomUniform, "RANDOM_UNIFORM"},
     {kLiteRtOpCodeTflMultinomial, "MULTINOMIAL"},
     {kLiteRtOpCodeTflGelu, "GELU"},
     {kLiteRtOpCodeTflDynamicUpdateSlice, "DYNAMIC_UPDATE_SLICE"},
     {kLiteRtOpCodeTflRelu0To1, "RELU_0_TO_1"},
     {kLiteRtOpCodeTflUnsortedSegmentProd, "UNSORTED_SEGMENT_PROD"},
     {kLiteRtOpCodeTflUnsortedSegmentMax, "UNSORTED_SEGMENT_MAX"},
     {kLiteRtOpCodeTflUnsortedSegmentSum, "UNSORTED_SEGMENT_SUM"},
     {kLiteRtOpCodeTflAtan2, "ATAN2"},
     {kLiteRtOpCodeTflUnsortedSegmentMin, "UNSORTED_SEGMENT_MIN"},
     {kLiteRtOpCodeTflSign, "SIGN"}}};

constexpr const char* GetOvOpType(const LiteRtOpCode op_code) {
  for (const auto& entry : kLitertOvMap) {
    if (entry.first == op_code) return entry.second;
  }
  LITERT_LOG(LITERT_WARNING, "op_code(%d) not supported", op_code);
  return "";
}

DecoderOperation::DecoderOperation(
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>
        input_tensor_info,
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>
        output_tensor_info,
    const litert::Op& litert_op, size_t node_index)
    : input_tensor_info_(input_tensor_info),
      output_tensor_info_(output_tensor_info),
      litert_op_(litert_op.Get()),
      litert_op_code_(litert_op.Code()) {
  op_type_ = GetOvOpType(litert_op_code_);
  op_name_ = op_type_ + "_id_" + std::to_string(node_index);
  LITERT_LOG(LITERT_VERBOSE, "op_type(%s) op_name(%s)", op_type_.c_str(),
             op_name_.c_str());
}

#define ERROR_LOG_STR(attr, op_name)    \
  litert::Unexpected(                   \
      kLiteRtStatusErrorRuntimeFailure, \
      "Failed to get " + std::string(attr) + " for " + std::string(op_name))

ov::Any DecoderOperation::get_attribute(const std::string& name) const {
  LITERT_LOG(LITERT_VERBOSE, "get_attr %s for %s", name.c_str(),
             op_name_.c_str());
  auto res = fetch_attribute(name);
  if (::litert::ErrorStatusBuilder::IsError(res)) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().c_str());
    return nullptr;
  }
  return res.Value();
}

litert::Expected<ov::Any> DecoderOperation::fetch_attribute(
    const std::string& name) const {
  switch (litert_op_code_) {
    case LiteRtOpCode::kLiteRtOpCodeTflConv2d:
      if (name == "strides") {
        int32_t stride_w;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetConv2dStrideWOption(litert_op_, &stride_w),
            ERROR_LOG_STR("stride_w", op_name_.c_str()));
        int32_t stride_h;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetConv2dStrideHOption(litert_op_, &stride_h),
            ERROR_LOG_STR("stride_h", op_name_.c_str()));
        return ov::Any(std::vector<int64_t>{1, stride_h, stride_w, 1});
      } else if (name == "padding") {
        uint32_t padding;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetConv2dPaddingOption(litert_op_, &padding),
            ERROR_LOG_STR("padding", op_name_.c_str()));
        return ov::Any(std::string(
            tflite::EnumNamePadding(static_cast<tflite::Padding>(padding))));
      } else if (name == "dilations") {
        int32_t dilation_w_factor;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetConv2dDilationWOption(litert_op_, &dilation_w_factor),
            ERROR_LOG_STR("dilation_w_factor", op_name_.c_str()));
        int32_t dilation_h_factor;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetConv2dDilationHOption(litert_op_, &dilation_h_factor),
            ERROR_LOG_STR("dilation_h_factor", op_name_.c_str()));
        return ov::Any(
            std::vector<int64_t>{1, dilation_h_factor, dilation_w_factor, 1});
      } else if (name == "activation") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetConv2dFusedActivationOption(litert_op_, &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      } else if (name == "data_format") {
        return ov::Any("NHWC");
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflDepthwiseConv2d:
      if (name == "strides") {
        int32_t stride_w;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetDepthwiseConv2dStrideWOption(litert_op_, &stride_w),
            ERROR_LOG_STR("stride_w", op_name_.c_str()));
        int32_t stride_h;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetDepthwiseConv2dStrideHOption(litert_op_, &stride_h),
            ERROR_LOG_STR("stride_h", op_name_.c_str()));
        return ov::Any(std::vector<int64_t>{1, stride_h, stride_w, 1});
      } else if (name == "padding") {
        uint32_t padding;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetDepthwiseConv2dPaddingOption(litert_op_, &padding),
            ERROR_LOG_STR("padding", op_name_.c_str()));
        return ov::Any(std::string(
            tflite::EnumNamePadding(static_cast<tflite::Padding>(padding))));
      } else if (name == "dilations") {
        int32_t dilation_w_factor;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetDepthwiseConv2dDilationWOption(litert_op_,
                                                    &dilation_w_factor),
            ERROR_LOG_STR("dilation_w_factor", op_name_.c_str()));
        int32_t dilation_h_factor;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetDepthwiseConv2dDilationHOptions(litert_op_,
                                                     &dilation_h_factor),
            ERROR_LOG_STR("dilation_h_factor", op_name_.c_str()));
        return ov::Any(
            std::vector<int64_t>{1, dilation_h_factor, dilation_w_factor, 1});
      } else if (name == "activation") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetDepthwiseConv2dFusedActivationOption(litert_op_,
                                                          &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      } else if (name == "group") {
        // This information(depth_multiplier) is marked as redundant in litert.
        // TODO: Need to check what is the correct value to be returned.
        return ov::Any(0);
      } else if (name == "data_format") {
        return ov::Any("NHWC");
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflSplit:
      if (name == "num_split") {
        int32_t num_split;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetSplitNumSplitsOption(litert_op_, &num_split),
            ERROR_LOG_STR("num_split", op_name_.c_str()));
        return ov::Any(static_cast<int64_t>(num_split));
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflFullyConnected:
      if (name == "weights_format") {
        uint32_t weights_format;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetFullyConnectedWeightsFormatOption(litert_op_,
                                                       &weights_format),
            ERROR_LOG_STR("weights_format", op_name_.c_str()));
        return ov::Any(static_cast<int8_t>(weights_format));
      } else if (name == "keep_num_dims") {
        bool keep_num_dims;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetFullyConnectedKeepNumDimsOption(litert_op_,
                                                     &keep_num_dims),
            ERROR_LOG_STR("keep_num_dims", op_name_.c_str()));
        return ov::Any(keep_num_dims);
      } else if (name == "fused_activation_function") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetFullyConnectedFusedActivationOption(litert_op_,
                                                         &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflAdd:
      if (name == "fused_activation_function") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetAddFusedActivationOption(litert_op_, &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflReshape:
      if (name == "new_shape") {
        const int32_t* reshape_new_shape;
        int32_t new_shape_size;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetReshapeNewShapeOption(litert_op_, &reshape_new_shape,
                                           &new_shape_size),
            ERROR_LOG_STR("new_shape", op_name_.c_str()));
        std::vector<int64_t> new_shape(new_shape_size);
        for (int i = 0; i < new_shape_size; ++i) {
          new_shape[i] = reshape_new_shape[i];
        }
        return ov::Any(new_shape);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflMean:
      if (name == "keep_dims") {
        bool keep_dims;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetMeanKeepDimsOption(litert_op_, &keep_dims),
            ERROR_LOG_STR("keep_dims", op_name_.c_str()));
        return ov::Any(keep_dims);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflResizeBilinear:
      if (name == "align_corners") {
        bool align_corners;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetResizeBilinearAlignCornersOption(litert_op_,
                                                      &align_corners),
            ERROR_LOG_STR("align_corners", op_name_.c_str()));
        return ov::Any(align_corners);
      } else if (name == "half_pixel_centers") {
        bool half_pixel_centers;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetResizeBilinearHalfPixelCenterOption(litert_op_,
                                                         &half_pixel_centers),
            ERROR_LOG_STR("half_pixel_centers", op_name_.c_str()));
        return ov::Any(half_pixel_centers);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflResizeNearestNeighbor:
      if (name == "align_corners") {
        bool align_corners;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetResizeNearestNeighborAlignCornersOption(litert_op_,
                                                             &align_corners),
            ERROR_LOG_STR("align_corners", op_name_.c_str()));
        return ov::Any(align_corners);
      } else if (name == "half_pixel_centers") {
        bool half_pixel_centers;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetResizeNearestNeighborHalfPixelCenterOption(
                litert_op_, &half_pixel_centers),
            ERROR_LOG_STR("half_pixel_centers", op_name_.c_str()));
        return ov::Any(half_pixel_centers);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflConcatenation:
      if (name == "axis") {
        int32_t axis;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetConcatenationAxisOption(litert_op_, &axis),
            ERROR_LOG_STR("axis", op_name_.c_str()));
        return ov::Any(axis);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflMaxPool2d:
      if (name == "strides") {
        int32_t stride_w;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetMaxPool2dStrideWOption(litert_op_, &stride_w),
            ERROR_LOG_STR("stride_w", op_name_.c_str()));
        int32_t stride_h;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetMaxPool2dStrideHOption(litert_op_, &stride_h),
            ERROR_LOG_STR("stride_h", op_name_.c_str()));
        return ov::Any(std::vector<int64_t>{1, stride_h, stride_w, 1});
      } else if (name == "padding") {
        uint32_t padding;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetMaxPool2dPaddingOption(litert_op_, &padding),
            ERROR_LOG_STR("padding", op_name_.c_str()));
        return ov::Any(std::string(
            tflite::EnumNamePadding(static_cast<tflite::Padding>(padding))));
      } else if (name == "ksize") {
        int32_t filter_width;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetMaxPool2dFilterWidthOption(litert_op_, &filter_width),
            ERROR_LOG_STR("filter_width", op_name_.c_str()));
        int32_t filter_height;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetMaxPool2dFilterHeightOption(litert_op_, &filter_height),
            ERROR_LOG_STR("filter_height", op_name_.c_str()));
        return ov::Any(std::vector<int64_t>{1, filter_height, filter_width, 1});
      } else if (name == "activation") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetMaxPool2dFusedActivationOption(litert_op_,
                                                    &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      } else if (name == "data_format") {
        return ov::Any("NHWC");
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflAveragePool2d:
      if (name == "strides") {
        int32_t stride_w;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetAveragePool2dStrideWOption(litert_op_, &stride_w),
            ERROR_LOG_STR("stride_w", op_name_.c_str()));
        int32_t stride_h;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetAveragePool2dStrideHOption(litert_op_, &stride_h),
            ERROR_LOG_STR("stride_h", op_name_.c_str()));
        return ov::Any(std::vector<int64_t>{1, stride_h, stride_w, 1});
      } else if (name == "padding") {
        uint32_t padding;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetAveragePool2dPaddingOption(litert_op_, &padding),
            ERROR_LOG_STR("padding", op_name_.c_str()));
        return ov::Any(std::string(
            tflite::EnumNamePadding(static_cast<tflite::Padding>(padding))));
      } else if (name == "ksize") {
        int32_t filter_width;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetAveragePool2dFilterWidthOption(litert_op_, &filter_width),
            ERROR_LOG_STR("filter_width", op_name_.c_str()));
        int32_t filter_height;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetAveragePool2dFilterHeightOption(litert_op_,
                                                     &filter_height),
            ERROR_LOG_STR("filter_height", op_name_.c_str()));
        return ov::Any(std::vector<int64_t>{1, filter_height, filter_width, 1});
      } else if (name == "activation") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetAveragePool2dFusedActivationOption(litert_op_,
                                                        &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      } else if (name == "data_format") {
        return ov::Any("NHWC");
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflMul:
      if (name == "fused_activation_function") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetMulFusedActivationOption(litert_op_, &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflTransposeConv:
      if (name == "strides") {
        int32_t stride_w;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetTransposeConvStrideWOption(litert_op_, &stride_w),
            ERROR_LOG_STR("stride_w", op_name_.c_str()));
        int32_t stride_h;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetTransposeConvStrideHOption(litert_op_, &stride_h),
            ERROR_LOG_STR("stride_h", op_name_.c_str()));
        return ov::Any(std::vector<int64_t>{1, stride_h, stride_w, 1});
      } else if (name == "padding") {
        uint32_t padding;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetTransposeConvPaddingOption(litert_op_, &padding),
            ERROR_LOG_STR("padding", op_name_.c_str()));
        return ov::Any(std::string(
            tflite::EnumNamePadding(static_cast<tflite::Padding>(padding))));
      } else if (name == "dilations") {
        // TODO: This information is not available in litert. Returning value
        // similar to OV tflite decoder.
        return ov::Any(std::vector<int64_t>{1, 1, 1, 1});
      } else if (name == "activation") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetTransposeConvFusedActivationOption(litert_op_,
                                                        &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      } else if (name == "data_format") {
        return ov::Any("NHWC");
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflSoftmax:
      if (name == "beta") {
        float beta;
        LITERT_RETURN_IF_ERROR(LiteRtGetSoftmaxBetaOption(litert_op_, &beta),
                               ERROR_LOG_STR("beta", op_name_.c_str()));
        return ov::Any(beta);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflMirrorPad:
      if (name == "mode") {
        // TODO: Currently litert_options doesn't provide an option for this.
        // Hence hardcoding to "REFLECT" mode.
        return ov::Any(std::string("REFLECT"));
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflStridedSlice:
      if (name == "begin_mask") {
        int32_t begin_mask;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetStridedSliceBeginMaskOption(litert_op_, &begin_mask),
            ERROR_LOG_STR("begin_mask", op_name_.c_str()));
        return ov::Any(begin_mask);
      } else if (name == "end_mask") {
        int32_t end_mask;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetStridedSliceEndMaskOption(litert_op_, &end_mask),
            ERROR_LOG_STR("end_mask", op_name_.c_str()));
        return ov::Any(end_mask);
      } else if (name == "new_axis_mask") {
        int32_t new_axis_mask;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetStridedSliceNewAxisMaskOption(litert_op_, &new_axis_mask),
            ERROR_LOG_STR("new_axis_mask", op_name_.c_str()));
        return ov::Any(new_axis_mask);
      } else if (name == "ellipsis_mask") {
        int32_t ellipsis_mask;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetStridedSliceEllipsisMaskOption(litert_op_, &ellipsis_mask),
            ERROR_LOG_STR("ellipsis_mask", op_name_.c_str()));
        return ov::Any(ellipsis_mask);
      } else if (name == "shrink_axis_mask") {
        int32_t shrink_axis_mask;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetStridedSliceShrinkAxisMaskOption(litert_op_,
                                                      &shrink_axis_mask),
            ERROR_LOG_STR("shrink_axis_mask", op_name_.c_str()));
        return ov::Any(shrink_axis_mask);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflDepthToSpace:
      if (name == "block_size") {
        int32_t block_size;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetDepthToSpaceBlockSizeOption(litert_op_, &block_size),
            ERROR_LOG_STR("block_size", op_name_.c_str()));
        return ov::Any(block_size);
      } else if (name == "data_format") {
        return ov::Any("NHWC");
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflGather:
      if (name == "axis") {
        int32_t axis;
        LITERT_RETURN_IF_ERROR(LiteRtGetGatherAxisOption(litert_op_, &axis),
                               ERROR_LOG_STR("axis", op_name_.c_str()));
        return ov::Any(axis);
      } else if (name == "batch_dims") {
        int32_t batch_dims;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetGatherBatchDimsOption(litert_op_, &batch_dims),
            ERROR_LOG_STR("batch_dims", op_name_.c_str()));
        return ov::Any(batch_dims);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflBatchMatmul:
      if (name == "adj_x") {
        bool adj_x;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetBatchMatmulAdjXOption(litert_op_, &adj_x),
            ERROR_LOG_STR("adj_x", op_name_.c_str()));
        return ov::Any(adj_x);
      } else if (name == "adj_y") {
        bool adj_y;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetBatchMatmulAdjYOption(litert_op_, &adj_y),
            ERROR_LOG_STR("adj_y", op_name_.c_str()));
        return ov::Any(adj_y);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflLeakyRelu:
      if (name == "alpha") {
        float alpha;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetLeakyReluAlphaOption(litert_op_, &alpha),
            ERROR_LOG_STR("alpha", op_name_.c_str()));
        return ov::Any(alpha);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflPack:
      if (name == "axis") {
        int32_t axis;
        LITERT_RETURN_IF_ERROR(LiteRtGetPackAxisOption(litert_op_, &axis),
                               ERROR_LOG_STR("axis", op_name_.c_str()));
        return ov::Any(axis);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflCast:
      if (name == "DstT") {
        return ov::Any(output_tensor_info_[0].m_element_type);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflDiv:
      if (name == "fused_activation_function") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetDivFusedActivationOption(litert_op_, &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflCumsum:
      if (name == "exclusive") {
        bool exclusive;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetCumsumExclusiveOption(litert_op_, &exclusive),
            ERROR_LOG_STR("exclusive", op_name_.c_str()));
        return ov::Any(exclusive);
      } else if (name == "reverse") {
        bool reverse;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetCumsumReverseOption(litert_op_, &reverse),
            ERROR_LOG_STR("reverse", op_name_.c_str()));
        return ov::Any(reverse);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflSub:
      if (name == "fused_activation_function") {
        uint32_t fused_activation;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetSubFusedActivationOption(litert_op_, &fused_activation),
            ERROR_LOG_STR("fused_activation", op_name_.c_str()));
        return ov::Any(tflite::EnumNameActivationFunctionType(
            static_cast<tflite::ActivationFunctionType>(fused_activation)));
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflGelu:
      if (name == "approximate") {
        bool approximate;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetGeluApproximateOption(litert_op_, &approximate),
            ERROR_LOG_STR("approximate", op_name_.c_str()));
        return ov::Any(approximate);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflGatherNd:
      if (name == "batch_dims") {
        // No information available in litert_options.
        return ov::Any(0);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflSum:
      if (name == "keep_dims") {
        bool keep_dims;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetSumKeepDimsOption(litert_op_, &keep_dims),
            ERROR_LOG_STR("keep_dims", op_name_.c_str()));
        return ov::Any(keep_dims);
      }
      break;
    case LiteRtOpCode::kLiteRtOpCodeTflReduceMax:
      if (name == "keep_dims") {
        bool keep_dims;
        LITERT_RETURN_IF_ERROR(
            LiteRtGetReduceMaxKeepDimsOption(litert_op_, &keep_dims),
            ERROR_LOG_STR("keep_dims", op_name_.c_str()));
        return ov::Any(keep_dims);
      }
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Unsupported op type %s", op_type_.c_str());
      return ov::Any(nullptr);
  }
  LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
  return ov::Any(nullptr);
}

}  // namespace openvino
}  // namespace litert
