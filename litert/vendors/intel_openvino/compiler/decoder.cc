// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>

#include "decoder.h"
#include "litert/c/litert_op_options.h"
#include "litert/tools/dump.h"

namespace litert {
namespace openvino {

// This has been picked from the openvino build:
// build/src/frontends/tensorflow_lite/src/schema_generated.h
constexpr std::array<std::pair<LiteRtOpCode, const char*>, 159> kLitertOvMap{{
        {kLiteRtOpCodeTflAdd, "ADD"},
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
        {kLiteRtOpCodeTflLocalResponseNormalization, "LOCAL_RESPONSE_NORMALIZATION"},
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
        {kLiteRtOpCodeTflUnidirectionalSequenceLstm, "UNIDIRECTIONAL_SEQUENCE_LSTM"},
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
        {kLiteRtOpCodeTflPlaceholderForGreaterOpCodeTfls, "PLACEHOLDER_FOR_GREATER_OP_CODES"},
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
        {kLiteRtOpCodeTflSign, "SIGN"}
}};

constexpr const char* GetOvOpType(const LiteRtOpCode op_code) {
    for (const auto& entry : kLitertOvMap) {
        if (entry.first == op_code)
		return entry.second;
    }
    return "";
}

DecoderOperation::DecoderOperation(
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> input_tensor_info,
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> output_tensor_info,
    const litert::Op& litert_op, size_t node_index)
    : input_tensor_info_(input_tensor_info),
      output_tensor_info_(output_tensor_info),
      litert_op_(litert_op.Get()),
      litert_op_code_(litert_op.Code()) {
    op_type_ = GetOvOpType(litert_op_code_);
    op_name_ = op_type_ + "_id_" + std::to_string(node_index);
    LITERT_LOG(LITERT_VERBOSE, "op_type(%s) op_name(%s)", op_type_.c_str(), op_name_.c_str());
}

#define DECODER_CHECK_STATUS(status, attr)                                                       \
    if (status != kLiteRtStatusOk) {                                                             \
        LITERT_LOG(LITERT_ERROR, "Failed(%d) to get %s for %s", status, attr, op_name_.c_str()); \
        return nullptr;                                                                          \
    }

ov::Any DecoderOperation::get_attribute(const std::string& name) const {
    LITERT_LOG(LITERT_VERBOSE, "get_attr %s for %s", name.c_str(), op_name_.c_str());
    switch (litert_op_code_) {
        case LiteRtOpCode::kLiteRtOpCodeTflConv2d:
            if (name == "strides") {
                int32_t stride_w;
                LiteRtStatus status = LiteRtGetConv2dStrideWOption(litert_op_, &stride_w);
                DECODER_CHECK_STATUS(status, "stride_w");
                int32_t stride_h;
                status = LiteRtGetConv2dStrideHOption(litert_op_, &stride_h);
                DECODER_CHECK_STATUS(status, "stride_h");
                return std::vector<int64_t>{1, stride_h, stride_w, 1};
            } else if (name == "padding") {
                uint32_t padding;
                LiteRtStatus status = LiteRtGetConv2dPaddingOption(litert_op_, &padding);
                DECODER_CHECK_STATUS(status, "padding");
                return std::string(tflite::EnumNamePadding(static_cast<tflite::Padding>(padding)));
            } else if (name == "dilations") {
                int32_t dilation_w_factor;
                LiteRtStatus status =
                    LiteRtGetConv2dDilationWOption(litert_op_, &dilation_w_factor);
                DECODER_CHECK_STATUS(status, "dilation_w_factor");
                int32_t dilation_h_factor;
                status = LiteRtGetConv2dDilationHOption(litert_op_, &dilation_h_factor);
                DECODER_CHECK_STATUS(status, "dilation_h_factor");
                return std::vector<int64_t>{1, dilation_h_factor, dilation_w_factor, 1};
            } else if (name == "activation") {
                uint32_t fused_activation;
                LiteRtStatus status =
                    LiteRtGetConv2dFusedActivationOption(litert_op_, &fused_activation);
                DECODER_CHECK_STATUS(status, "fused_activation");
                return tflite::EnumNameActivationFunctionType(
                    static_cast<tflite::ActivationFunctionType>(fused_activation));
            } else if (name == "data_format") {
                return "NHWC";
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflDepthwiseConv2d:
            if (name == "strides") {
                int32_t stride_w;
                LiteRtStatus status = LiteRtGetDepthwiseConv2dStrideWOption(litert_op_, &stride_w);
                DECODER_CHECK_STATUS(status, "stride_w");
                int32_t stride_h;
                status = LiteRtGetDepthwiseConv2dStrideHOption(litert_op_, &stride_h);
                DECODER_CHECK_STATUS(status, "stride_h");
                return std::vector<int64_t>{1, stride_h, stride_w, 1};
            } else if (name == "padding") {
                uint32_t padding;
                LiteRtStatus status = LiteRtGetDepthwiseConv2dPaddingOption(litert_op_, &padding);
                DECODER_CHECK_STATUS(status, "padding");
                return std::string(tflite::EnumNamePadding(static_cast<tflite::Padding>(padding)));
            } else if (name == "dilations") {
                int32_t dilation_w_factor;
                LiteRtStatus status =
                    LiteRtGetDepthwiseConv2dDilationWOption(litert_op_, &dilation_w_factor);
                DECODER_CHECK_STATUS(status, "dilation_w_factor");
                int32_t dilation_h_factor;
                status = LiteRtGetDepthwiseConv2dDilationHOptions(litert_op_, &dilation_h_factor);
                DECODER_CHECK_STATUS(status, "dilation_h_factor");
                return std::vector<int64_t>{1, dilation_h_factor, dilation_w_factor, 1};
            } else if (name == "activation") {
                uint32_t fused_activation;
                LiteRtStatus status =
                    LiteRtGetDepthwiseConv2dFusedActivationOption(litert_op_, &fused_activation);
                DECODER_CHECK_STATUS(status, "fused_activation");
                return tflite::EnumNameActivationFunctionType(
                    static_cast<tflite::ActivationFunctionType>(fused_activation));
            } else if (name == "group") {
                // This information(depth_multiplier) is marked as redundant in litert.
                // TODO: Need to check what is the correct value to be returned.
                return 0;
            } else if (name == "data_format") {
                return "NHWC";
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflSplit:
            if (name == "num_split") {
                int32_t num_split;
                LiteRtStatus status = LiteRtGetSplitNumSplitsOption(litert_op_, &num_split);
                DECODER_CHECK_STATUS(status, "num_split");
                return static_cast<int64_t>(num_split);
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflFullyConnected:
            if (name == "weights_format") {
                uint32_t weights_format;
                LiteRtStatus status =
                    LiteRtGetFullyConnectedWeightsFormatOption(litert_op_, &weights_format);
                DECODER_CHECK_STATUS(status, "weights_format");
                return static_cast<int8_t>(weights_format);
            } else if (name == "keep_num_dims") {
                bool keep_num_dims;
                LiteRtStatus status =
                    LiteRtGetFullyConnectedKeepNumDimsOption(litert_op_, &keep_num_dims);
                DECODER_CHECK_STATUS(status, "keep_num_dims");
                return keep_num_dims;
            } else if (name == "fused_activation_function") {
                uint32_t fused_activation;
                LiteRtStatus status =
                    LiteRtGetFullyConnectedFusedActivationOption(litert_op_, &fused_activation);
                DECODER_CHECK_STATUS(status, "fused_activation");
                return tflite::EnumNameActivationFunctionType(
                    static_cast<tflite::ActivationFunctionType>(fused_activation));
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflAdd:
            if (name == "fused_activation_function") {
                uint32_t fused_activation;
                LiteRtStatus status =
                    LiteRtGetAddFusedActivationOption(litert_op_, &fused_activation);
                DECODER_CHECK_STATUS(status, "fused_activation");
                return tflite::EnumNameActivationFunctionType(
                    static_cast<tflite::ActivationFunctionType>(fused_activation));
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflReshape:
            if (name == "new_shape") {
                const int32_t* reshape_new_shape;
                int32_t new_shape_size;
                LiteRtStatus status =
                    LiteRtGetReshapeNewShapeOption(litert_op_, &reshape_new_shape, &new_shape_size);
                if (status == kLiteRtStatusErrorInvalidArgument) {
                    LITERT_LOG(LITERT_INFO, "New shape unavailable for %s", name.c_str());
                    return {};
                }
                std::vector<int64_t> new_shape(new_shape_size);
                for (int i = 0; i < new_shape_size; ++i) {
                    new_shape[i] = reshape_new_shape[i];
                }
                return new_shape;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflMean:
            if (name == "keep_dims") {
                bool keep_dims;
                LiteRtStatus status = LiteRtGetMeanKeepDimsOption(litert_op_, &keep_dims);
                DECODER_CHECK_STATUS(status, "keep_dims");
                return keep_dims;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflResizeBilinear:
            if (name == "align_corners") {
                bool align_corners;
                LiteRtStatus status =
                    LiteRtGetResizeBilinearAlignCornersOption(litert_op_, &align_corners);
                DECODER_CHECK_STATUS(status, "align_corners");
                return align_corners;
            } else if (name == "half_pixel_centers") {
                bool half_pixel_centers;
                LiteRtStatus status =
                    LiteRtGetResizeBilinearHalfPixelCenterOption(litert_op_, &half_pixel_centers);
                DECODER_CHECK_STATUS(status, "half_pixel_centers");
                return half_pixel_centers;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflResizeNearestNeighbor:
            if (name == "align_corners") {
                bool align_corners;
                LiteRtStatus status =
                    LiteRtGetResizeNearestNeighborAlignCornersOption(litert_op_, &align_corners);
                DECODER_CHECK_STATUS(status, "align_corners");
                return align_corners;
            } else if (name == "half_pixel_centers") {
                bool half_pixel_centers;
                LiteRtStatus status = LiteRtGetResizeNearestNeighborHalfPixelCenterOption(
                    litert_op_, &half_pixel_centers);
                DECODER_CHECK_STATUS(status, "half_pixel_centers");
                return half_pixel_centers;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflConcatenation:
            if (name == "axis") {
                int32_t axis;
                LiteRtStatus status = LiteRtGetConcatenationAxisOption(litert_op_, &axis);
                DECODER_CHECK_STATUS(status, "axis");
                return axis;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflMaxPool2d:
            if (name == "strides") {
                int32_t stride_w;
                LiteRtStatus status = LiteRtGetMaxPool2dStrideWOption(litert_op_, &stride_w);
                DECODER_CHECK_STATUS(status, "stride_w");
                int32_t stride_h;
                status = LiteRtGetMaxPool2dStrideHOption(litert_op_, &stride_h);
                DECODER_CHECK_STATUS(status, "stride_h");
                return std::vector<int64_t>{1, stride_h, stride_w, 1};
            } else if (name == "padding") {
                uint32_t padding;
                LiteRtStatus status = LiteRtGetMaxPool2dPaddingOption(litert_op_, &padding);
                DECODER_CHECK_STATUS(status, "padding");
                return std::string(tflite::EnumNamePadding(static_cast<tflite::Padding>(padding)));
            } else if (name == "ksize") {
                int32_t filter_width;
                LiteRtStatus status =
                    LiteRtGetMaxPool2dFilterWidthOption(litert_op_, &filter_width);
                DECODER_CHECK_STATUS(status, "filter_width");
                int32_t filter_height;
                status = LiteRtGetMaxPool2dFilterHeightOption(litert_op_, &filter_height);
                DECODER_CHECK_STATUS(status, "filter_height");
                return std::vector<int64_t>{1, filter_height, filter_width, 1};
            } else if (name == "activation") {
                uint32_t fused_activation;
                LiteRtStatus status =
                    LiteRtGetMaxPool2dFusedActivationOption(litert_op_, &fused_activation);
                DECODER_CHECK_STATUS(status, "fused_activation");
                return tflite::EnumNameActivationFunctionType(
                    static_cast<tflite::ActivationFunctionType>(fused_activation));
            } else if (name == "data_format") {
                return "NHWC";
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflAveragePool2d:
            if (name == "strides") {
                int32_t stride_w;
                LiteRtStatus status = LiteRtGetAveragePool2dStrideWOption(litert_op_, &stride_w);
                DECODER_CHECK_STATUS(status, "stride_w");
                int32_t stride_h;
                status = LiteRtGetAveragePool2dStrideHOption(litert_op_, &stride_h);
                DECODER_CHECK_STATUS(status, "stride_h");
                return std::vector<int64_t>{1, stride_h, stride_w, 1};
            } else if (name == "padding") {
                uint32_t padding;
                LiteRtStatus status = LiteRtGetAveragePool2dPaddingOption(litert_op_, &padding);
                DECODER_CHECK_STATUS(status, "padding");
                return std::string(tflite::EnumNamePadding(static_cast<tflite::Padding>(padding)));
            } else if (name == "ksize") {
                int32_t filter_width;
                LiteRtStatus status =
                    LiteRtGetAveragePool2dFilterWidthOption(litert_op_, &filter_width);
                DECODER_CHECK_STATUS(status, "filter_width");
                int32_t filter_height;
                status = LiteRtGetAveragePool2dFilterHeightOption(litert_op_, &filter_height);
                DECODER_CHECK_STATUS(status, "filter_height");
                return std::vector<int64_t>{1, filter_height, filter_width, 1};
            } else if (name == "activation") {
                uint32_t fused_activation;
                LiteRtStatus status =
                    LiteRtGetAveragePool2dFusedActivationOption(litert_op_, &fused_activation);
                DECODER_CHECK_STATUS(status, "fused_activation");
                return tflite::EnumNameActivationFunctionType(
                    static_cast<tflite::ActivationFunctionType>(fused_activation));
            } else if (name == "data_format") {
                return "NHWC";
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflMul:
            if (name == "fused_activation_function") {
                uint32_t fused_activation;
                LiteRtStatus status =
                    LiteRtGetMulFusedActivationOption(litert_op_, &fused_activation);
                DECODER_CHECK_STATUS(status, "fused_activation");
                return tflite::EnumNameActivationFunctionType(
                    static_cast<tflite::ActivationFunctionType>(fused_activation));
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflTransposeConv:
            if (name == "strides") {
                int32_t stride_w;
                LiteRtStatus status = LiteRtGetTransposeConvStrideWOption(litert_op_, &stride_w);
                DECODER_CHECK_STATUS(status, "stride_w");
                int32_t stride_h;
                status = LiteRtGetTransposeConvStrideHOption(litert_op_, &stride_h);
                DECODER_CHECK_STATUS(status, "stride_h");
                return std::vector<int64_t>{1, stride_h, stride_w, 1};
            } else if (name == "padding") {
                uint32_t padding;
                LiteRtStatus status = LiteRtGetTransposeConvPaddingOption(litert_op_, &padding);
                DECODER_CHECK_STATUS(status, "padding");
                return std::string(tflite::EnumNamePadding(static_cast<tflite::Padding>(padding)));
            } else if (name == "dilations") {
                // TODO: This information is not available in litert. Returning value similar to OV
                // tflite decoder.
                return std::vector<int64_t>{1, 1, 1, 1};
            } else if (name == "activation") {
                uint32_t fused_activation;
                LiteRtStatus status =
                    LiteRtGetTransposeConvFusedActivationOption(litert_op_, &fused_activation);
                DECODER_CHECK_STATUS(status, "fused_activation");
                return tflite::EnumNameActivationFunctionType(
                    static_cast<tflite::ActivationFunctionType>(fused_activation));
            } else if (name == "data_format") {
                return "NHWC";
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflSoftmax:
            if (name == "beta") {
                float beta;
                LiteRtStatus status = LiteRtGetSoftmaxBetaOption(litert_op_, &beta);
                DECODER_CHECK_STATUS(status, "beta");
                return beta;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflMirrorPad:
            if (name == "mode") {
                // TODO: Currently litert_options doesn't provide an option for this. Hence
                // hardcoding to "REFLECT" mode.
                return std::string("REFLECT");
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflStridedSlice:
            if (name == "begin_mask") {
                int32_t begin_mask;
                LiteRtStatus status = LiteRtGetStridedSliceBeginMaskOption(litert_op_, &begin_mask);
                DECODER_CHECK_STATUS(status, "begin_mask");
                return begin_mask;
            } else if (name == "end_mask") {
                int32_t end_mask;
                LiteRtStatus status = LiteRtGetStridedSliceEndMaskOption(litert_op_, &end_mask);
                DECODER_CHECK_STATUS(status, "end_mask");
                return end_mask;
            } else if (name == "new_axis_mask") {
                int32_t new_axis_mask;
                LiteRtStatus status =
                    LiteRtGetStridedSliceNewAxisMaskOption(litert_op_, &new_axis_mask);
                DECODER_CHECK_STATUS(status, "new_axis_mask");
                return new_axis_mask;
            } else if (name == "ellipsis_mask") {
                int32_t ellipsis_mask;
                LiteRtStatus status =
                    LiteRtGetStridedSliceEllipsisMaskOption(litert_op_, &ellipsis_mask);
                DECODER_CHECK_STATUS(status, "ellipsis_mask");
                return ellipsis_mask;
            } else if (name == "shrink_axis_mask") {
                int32_t shrink_axis_mask;
                LiteRtStatus status =
                    LiteRtGetStridedSliceShrinkAxisMaskOption(litert_op_, &shrink_axis_mask);
                DECODER_CHECK_STATUS(status, "shrink_axis_mask");
                return shrink_axis_mask;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflDepthToSpace:
            if (name == "block_size") {
                int32_t block_size;
                LiteRtStatus status = LiteRtGetDepthToSpaceBlockSizeOption(litert_op_, &block_size);
                DECODER_CHECK_STATUS(status, "block_size");
                return block_size;
            } else if (name == "data_format") {
                return "NHWC";
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflGather:
            if (name == "axis") {
                int32_t axis;
                LiteRtStatus status = LiteRtGetGatherAxisOption(litert_op_, &axis);
                DECODER_CHECK_STATUS(status, "axis");
                return axis;
            } else if (name == "batch_dims") {
                int32_t batch_dims;
                LiteRtStatus status = LiteRtGetGatherBatchDimsOption(litert_op_, &batch_dims);
                DECODER_CHECK_STATUS(status, "batch_dims");
                return batch_dims;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflBatchMatmul:
            if (name == "adj_x") {
                bool adj_x;
                LiteRtStatus status = LiteRtGetBatchMatmulAdjXOption(litert_op_, &adj_x);
                DECODER_CHECK_STATUS(status, "adj_x");
                return adj_x;
            } else if (name == "adj_y") {
                bool adj_y;
                LiteRtStatus status = LiteRtGetBatchMatmulAdjYOption(litert_op_, &adj_y);
                DECODER_CHECK_STATUS(status, "adj_y");
                return adj_y;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflLeakyRelu:
            if (name == "alpha") {
                float alpha;
                LiteRtStatus status = LiteRtGetLeakyReluAlphaOption(litert_op_, &alpha);
                DECODER_CHECK_STATUS(status, "alpha");
                return alpha;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        case LiteRtOpCode::kLiteRtOpCodeTflPack:
            if (name == "axis") {
                int32_t axis;
                LiteRtStatus status = LiteRtGetPackAxisOption(litert_op_, &axis);
                DECODER_CHECK_STATUS(status, "axis");
                return axis;
            } else {
                LITERT_LOG(LITERT_ERROR, "Unsupported attribute %s", name.c_str());
                return nullptr;
            }
        default:
            LITERT_LOG(LITERT_ERROR, "Unsupported op type %s", op_type_.c_str());
            return nullptr;
    }
}

}  // namespace openvino
}  // namespace litert
